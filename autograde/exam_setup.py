"""考前准备：录入试卷题目、导入/生成参考答案和评分细则、识别答题纸版面。"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import docparse as D
from . import objective as O
from . import prompts as P
from .imaging import IMAGE_EXTS, PDF_EXTS, imwrite, load_pages, to_data_url
from .layout import analyze_page, header_sheet, render_regions
from .llm import DeepSeek, LLMError, image_part, text_part
from .store import Store, model_supports_images

Progress = Callable[[float, str], None]
KINDS = ["解答题", "填空题", "选择题", "学生信息", "其他"]
EXAM_KINDS = {"exam": "考试", "quiz": "周测", "homework": "平时作业"}


def is_scored(exam: dict) -> bool:
    """平时作业不打分：只判对错（每题按 1 分记，0.5 是半对）。考试、周测照常打分。"""
    return (exam.get("kind") or "exam") != "homework"


def scoring_questions(exam: dict) -> List[dict]:
    """批改用的题目：打分的考试原样返回；平时作业每题按 1 分（对 1、半对 0.5、错 0），小问不再分分值。"""
    qs = exam.get("questions") or []
    if is_scored(exam):
        return qs
    return [dict(q, score=1, unscored=True, parts=[dict(p, score=None) for p in q.get("parts") or []]) for q in qs]


def _noop(p: float, m: str) -> None:
    pass


# ================================================================ 规范化

_ROMAN = {"Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "Ⅴ": "5", "Ⅵ": "6", "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5"}


def norm_number(s) -> str:
    s = str(s if s is not None else "").strip()
    s = re.sub(r"[第题号\s\.．、:：()（）\[\]【】]", "", s)
    return s


def norm_label(s, k: int = 0) -> str:
    t = str(s or "").strip()
    t = re.sub(r"[第问小题\s：:．.]", "", t)
    t = t.strip("()（）[]【】")
    t = _ROMAN.get(t, t)
    if not t:
        t = str(k + 1)
    return f"({t})"


def norm_type(t) -> str:
    t = str(t or "")
    if any(k in t for k in ("解答", "证明", "计算", "作图", "应用", "综合")):
        return "解答题"
    if "填空" in t:
        return "填空题"
    if "选" in t:
        return "选择题"
    return "其他" if t else ""


def _num(v) -> Optional[float]:
    try:
        if v is None or v == "":
            return None
        f = float(v)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        m = re.search(r"\d+(\.\d+)?", str(v))
        return _num(m.group(0)) if m else None


def new_question(number: str, qtype: str = "解答题") -> dict:
    return {"number": number, "type": qtype, "score": None, "stem": "", "figure": "", "parts": [],
            "answer": "", "rubric": "", "notes": "", "answer_source": "none", "confirmed": False,
            "paper_crops": [], "answer_crops": [], "checks": [], "score_convention": "", "issues": ""}


def norm_question(raw: dict) -> dict:
    q = new_question(norm_number(raw.get("number")), norm_type(raw.get("type")) or "其他")
    q["score"] = _num(raw.get("score"))
    q["stem"] = str(raw.get("stem") or "").strip()
    q["figure"] = str(raw.get("figure") or "").strip()
    parts = []
    for k, p in enumerate(raw.get("parts") or []):
        if isinstance(p, dict):
            parts.append({"label": norm_label(p.get("label"), k), "text": str(p.get("text") or "").strip(),
                          "score": _num(p.get("score"))})
    q["parts"] = parts
    fold_parts_into_stem(q)
    return q


def fold_parts_into_stem(q: dict) -> None:
    """小问文字如果单独放在 parts[].text 里，并回题干（题干必须是题目全文）。"""
    extra = []
    for p in q.get("parts") or []:
        t = (p.get("text") or "").strip()
        if t and t[:12] not in (q.get("stem") or ""):
            extra.append(f"{p['label']} {t}")
        p["text"] = ""
    if extra:
        q["stem"] = ((q.get("stem") or "").rstrip() + "\n" + "\n".join(extra)).strip()


def _numkey(n: str):
    m = re.match(r"(\d+)", n or "")
    return (int(m.group(1)) if m else 10 ** 6, n)


def merge_questions(items: List[dict]) -> List[dict]:
    """同一题号出现多次（跨页）时合并。"""
    out: Dict[str, dict] = {}
    for q in items:
        n = q["number"]
        if not n:
            continue
        if n not in out:
            out[n] = q
            continue
        o = out[n]
        if q["stem"] and q["stem"] not in o["stem"]:
            o["stem"] = (o["stem"] + "\n" + q["stem"]).strip()
        if q["figure"] and q["figure"] not in o["figure"]:
            o["figure"] = (o["figure"] + "；" + q["figure"]).strip("；")
        labels = {p["label"] for p in o["parts"]}
        o["parts"] += [p for p in q["parts"] if p["label"] not in labels]
        o["score"] = o["score"] if o["score"] is not None else q["score"]
        if o["type"] in ("", "其他") and q["type"]:
            o["type"] = q["type"]
        o["paper_crops"] = list(o.get("paper_crops") or []) + [c for c in q.get("paper_crops") or [] if c not in (o.get("paper_crops") or [])]
    return sorted(out.values(), key=lambda q: _numkey(q["number"]))


# ================================================================ 读文件

def _nonblank(img: np.ndarray) -> bool:
    g = cv2.cvtColor(cv2.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return float((g < 160).mean()) > 0.002


def doc_images(path: Path, dpi: int = 170) -> List[np.ndarray]:
    """图片或 PDF 的每一页（去掉空白页）。"""
    if path.suffix.lower() in PDF_EXTS or path.suffix.lower() in IMAGE_EXTS:
        return [img for img in load_pages(path, dpi=dpi) if _nonblank(img)]
    return []


def load_text_files(paths: Sequence[Path]) -> str:
    out = []
    for p in paths:
        if p.suffix.lower() in (".txt", ".md", ".markdown"):
            out.append(p.read_text(encoding="utf-8", errors="ignore"))
    return "\n\n".join(out)


def _open_pdf(path: Path) -> Optional[D.PdfText]:
    if path.suffix.lower() not in PDF_EXTS:
        return None
    try:
        return D.PdfText(path)
    except Exception:
        return None


def _save_crops(d: Path, prefix: str, number: str, imgs: List[np.ndarray]) -> List[str]:
    rels = []
    for k, img in enumerate(imgs):
        rel = f"source/{prefix}_q{number}_{k}.jpg"
        imwrite(d / rel, img, quality=88)
        rels.append(rel)
    return rels


def _load_crops(d: Path, rels: Sequence[str]) -> List[np.ndarray]:
    out = []
    for r in rels or []:
        img = cv2.imread(str(d / r))
        if img is not None:
            out.append(img)
    return out


def _add_check(q: dict, where: str, msg: str) -> None:
    q.setdefault("checks", [])
    item = {"where": where, "msg": msg}
    if item not in q["checks"]:
        q["checks"].append(item)


def _compare_checks(q: dict, where: str, source: str, produced: str) -> None:
    c = D.compare_text(source, produced)
    if not c.get("checked"):
        return
    if c["extra_ratio"] > 0.12 and c["extra"]:
        _add_check(q, where, "录入的文字里有原文中找不到的内容：" + "；".join(f"“{x[:30]}”" for x in c["extra"][:3]))
    if c["missing_ratio"] > 0.12 and c["missing"]:
        _add_check(q, where, "可能漏录了原文的内容：" + "；".join(f"“{x[:30]}”" for x in c["missing"][:3]))


# ================================================================ 录题

def _pool(st: dict) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=max(1, int(st.get("max_workers", 4))))


def _paper_from_pdf(pdf: D.PdfText) -> Tuple[List[dict], List[str], Optional[tuple]]:
    """数字版试卷：按题号位置切题、读分值（不调用模型）。返回 (题目, 大题标题, 答案部分起点)。"""
    head = D.find_answer_heading(pdf)
    stop = None
    if head is not None and len(D.find_anchors(pdf, end=head.pos)) >= 2:
        stop = head.pos           # 试卷后面附了答案：题目只取答案标题之前的部分
    anchors = D.find_anchors(pdf, end=stop)
    if len(anchors) < 2:
        return [], [], None
    segs = D.segments(pdf, anchors, stop=stop)
    sections = D.find_sections(pdf)
    sec_scores: Dict[str, float] = {}
    sec_titles = []
    for r, title, _t in sections:
        if stop is not None and r.pos >= stop:
            continue
        sec_titles.append(title)
        sec_scores.update(D.parse_section_scores(title, [a.number for a in anchors if a.section == title]))
    qs = []
    for a in anchors:
        sg = segs.get(a.number)
        q = new_question(a.number, a.qtype or "")
        q["section"] = a.section
        head_txt = re.sub(r"\s+", "", "".join(r.text for r in (sg.rows[:3] if sg else [])))
        sc, pscore = D.parse_question_header(head_txt)
        q["score"] = _num(sc) if sc is not None else _num(sec_scores.get(a.number))
        labels = D.part_labels(pdf, sg) if sg else []
        for lab in pscore:
            if lab not in labels:
                labels.append(lab)
        q["parts"] = [{"label": lab, "text": "", "score": _num(pscore.get(lab))} for lab in labels]
        q["paper_text"] = D.strip_header_text(sg.text) if sg else ""
        q["_crops"] = [pdf.crop(pc, dpi=200) for pc in (sg.pieces if sg else [])]
        qs.append(q)
    # 大题总分减去已知分值，补出唯一缺的那道题（例如最后一题没写“本题满分”）
    for title in sec_titles:
        tot = D.section_total(title)
        mine = [q for q in qs if q.get("section") == title]
        unknown = [q for q in mine if q["score"] is None]
        if tot and len(unknown) == 1 and len(mine) > 1:
            rest = tot - sum(float(q["score"]) for q in mine if q["score"] is not None)
            if rest > 0:
                unknown[0]["score"] = _num(rest)
                unknown[0]["score_from"] = "大题总分推算"
    return qs, sec_titles, stop


def _transcribe_paper_q(llm: DeepSeek, st: dict, q: dict) -> None:
    crops = q.get("_crops") or []
    content = [image_part(to_data_url(c, 2000)) for c in crops]
    content.append(text_part(P.paper_q_user(q["number"], len(crops), q.get("paper_text", ""))))
    obj, _ = llm.chat_json([{"role": "system", "content": P.PAPER_SYSTEM}, {"role": "user", "content": content}],
                           st["vision_model"], thinking=False, max_tokens=8000)
    obj = obj if isinstance(obj, dict) else {}
    q["stem"] = str(obj.get("stem") or "").strip()
    q["figure"] = str(obj.get("figure") or "").strip()
    if not q.get("type"):
        q["type"] = norm_type(obj.get("type")) or "其他"
    if q.get("score") is None:
        q["score"] = _num(obj.get("score"))
    model_parts = [{"label": norm_label(x.get("label"), k), "score": _num(x.get("score"))}
                   for k, x in enumerate(obj.get("parts") or []) if isinstance(x, dict)]
    if not q["parts"]:
        q["parts"] = [{"label": x["label"], "text": "", "score": x["score"]} for x in model_parts]
    else:
        byl = {x["label"]: x for x in model_parts}
        for p in q["parts"]:
            if p.get("score") is None and byl.get(p["label"], {}).get("score") is not None:
                p["score"] = byl[p["label"]]["score"]
    if q.get("paper_text"):
        _compare_checks(q, "题目", q["paper_text"], q["stem"])
    for p in q["parts"]:
        if p["label"] not in q["stem"].replace("（", "(").replace("）", ")"):
            _add_check(q, "题目", f"题干里没有找到小问 {p['label']}，可能漏录")
            break


def parse_paper(llm: Optional[DeepSeek], st: dict, d: Path, paths: Sequence[Path],
                progress: Progress = _noop) -> Tuple[List[dict], List[str], List[tuple], List[str]]:
    """返回 (题目, 大题标题, 附带答案的 PDF [(路径, 起点)], 日志)。"""
    questions: List[dict] = []
    notes: List[str] = []
    combined: List[tuple] = []
    log: List[str] = []
    fallback_imgs: List[np.ndarray] = []
    text = load_text_files(paths)
    for path in paths:
        pdf = _open_pdf(path)
        if pdf is not None and pdf.usable():
            qs, secs, stop = _paper_from_pdf(pdf)
            if qs:
                questions += qs
                notes += secs
                if stop is not None:
                    combined.append((path, stop))
                log.append(f"{path.name}：按题号位置切出 {len(qs)} 道题")
                pdf.close()
                continue
        if pdf is not None:
            pdf.close()
        fallback_imgs += doc_images(path)
    # 数字版：逐题录入（只录解答题和题型不明的题）
    todo = [q for q in questions if q.get("_crops") and q.get("type") in ("解答题", "填空题", "")]
    if llm is not None and todo:
        done = 0
        with _pool(st) as ex:
            futs = {ex.submit(_transcribe_paper_q, llm, st, q): q for q in todo}
            for f in as_completed(futs):
                q = futs[f]
                done += 1
                progress(0.3 * done / len(todo), f"录入第 {q['number']} 题（{done}/{len(todo)}）")
                try:
                    f.result()
                except LLMError as e:
                    _add_check(q, "题目", f"题目录入失败：{e}")
    # 扫描件/照片/文字：整份一次录入
    if llm is not None and fallback_imgs:
        progress(0.15, f"录入试卷（{len(fallback_imgs)} 页图片）…")
        content = [image_part(to_data_url(img, 2000)) for img in fallback_imgs]
        content.append(text_part(P.paper_doc_user(len(fallback_imgs))))
        obj, _ = llm.chat_json([{"role": "system", "content": P.PAPER_SYSTEM}, {"role": "user", "content": content}],
                               st["vision_model"], thinking=False, max_tokens=24000)
        got = [norm_question(x) for x in (obj or {}).get("questions") or [] if isinstance(x, dict)]
        questions += got
        notes += [str(x) for x in ((obj or {}).get("section_notes") or []) if x] if isinstance((obj or {}).get("section_notes"), list) else []
        log.append(f"图片试卷：模型录入 {len(got)} 道题（扫描件没有文字层，无法和原文逐字核对，请对照原图检查）")
    if llm is not None and text.strip():
        obj, _ = llm.chat_json([{"role": "system", "content": P.PAPER_SYSTEM}, {"role": "user", "content": P.paper_text_user(text)}],
                               st["vision_model"], thinking=False, max_tokens=24000)
        got = [norm_question(x) for x in (obj or {}).get("questions") or [] if isinstance(x, dict)]
        questions += got
        log.append(f"文字试卷：整理出 {len(got)} 道题")
    if not questions:
        log.append("没有从试卷里录到题目：文件可能是空白页、加密 PDF 或模型没有返回内容，请检查文件，或者在下面手动添加题目")
    for q in questions:
        crops = q.pop("_crops", None)
        q.pop("paper_text", None)
        if crops:
            q["paper_crops"] = _save_crops(d, "paper", q["number"], crops)
    return merge_questions(questions), notes, combined, log


# ================================================================ 参考答案

def _transcribe_answer_q(llm: DeepSeek, st: dict, number: str, crops: List[np.ndarray], text: str, stem: str) -> str:
    content = [image_part(to_data_url(c, 2000)) for c in crops]
    content.append(text_part(P.answer_q_user(number, len(crops), text, stem)))
    obj, _ = llm.chat_json([{"role": "system", "content": P.ANSWERS_SYSTEM}, {"role": "user", "content": content}],
                           st["vision_model"], thinking=False, max_tokens=12000)
    return str((obj or {}).get("answer") or "").strip() if isinstance(obj, dict) else ""


def parse_answers(llm: Optional[DeepSeek], st: dict, d: Path, paths: Sequence[Path], questions: List[dict],
                  combined: Sequence[tuple] = (), progress: Progress = _noop) -> List[str]:
    """把参考答案合并进 questions（就地修改），返回日志。

    数字版 PDF：按题号位置把每道题的答案单独裁出来，逐题交给模型录入，并用文字层核对；
    分值标注（……X分）直接从文字层读出，推算各小问分值。
    扫描件/照片：整份答案一次交给模型（不再逐页，避免空白页、跨页造成的错乱）。"""
    log: List[str] = []
    sol = {q["number"]: q for q in questions if q.get("type") == "解答题"}
    objq = {q["number"]: q for q in questions if q.get("type") in ("填空题", "选择题")}
    obj_found: Dict[str, dict] = {}
    found: Dict[str, dict] = {}
    fallback_imgs: List[np.ndarray] = []
    sources = [(p, None, False) for p in paths] + [(p, pos, True) for p, pos in combined]
    for path, start, is_combined in sources:
        pdf = _open_pdf(path)
        ok = False
        if pdf is not None and pdf.usable():
            if start is None:   # 答案文件前面还附了试卷时，从“参考答案”标题开始
                head = D.find_answer_heading(pdf)
                if head is not None and len(D.find_anchors(pdf, end=head.pos)) >= 2:
                    start = head.pos
            if objq:
                obj_found.update({n: v for n, v in _objective_from_pdf(llm, st, d, pdf, start, objq).items()
                                  if n not in obj_found})
            anchors = D.find_anchors(pdf, start=start)
            segs = D.segments(pdf, anchors)
            for a in anchors:
                sg = segs.get(a.number)
                if a.number in sol and a.number not in found and sg and sg.pieces:
                    found[a.number] = {"crops": [pdf.crop(pc, dpi=200) for pc in sg.pieces], "text": sg.text,
                                       "markers": D.answer_markers(pdf, sg), "file": path.name}
            ok = any(a.number in sol for a in anchors)
        if pdf is not None:
            pdf.close()
        if not ok and not is_combined:
            fallback_imgs += doc_images(path)
    if llm is None:
        return log
    todo = [n for n in found]
    done = 0
    with _pool(st) as ex:
        futs = {ex.submit(_transcribe_answer_q, llm, st, n, found[n]["crops"], found[n]["text"],
                          P.question_text(sol[n])): n for n in todo}
        for f in as_completed(futs):
            n = futs[f]
            done += 1
            progress(done / max(1, len(todo)), f"录入第 {n} 题的参考答案（{done}/{len(todo)}）")
            try:
                found[n]["answer"] = f.result()
            except LLMError as e:
                found[n]["answer"] = ""
                found[n]["error"] = str(e)
    missing = [n for n in sol if n not in found]
    obj_missing = [n for n in sorted(objq, key=_numkey) if n not in obj_found]
    if fallback_imgs and (missing or obj_missing):
        progress(0.9, f"录入参考答案（{len(fallback_imgs)} 页图片）…")
        content = [image_part(to_data_url(img, 2000)) for img in fallback_imgs]
        content.append(text_part(P.answers_doc_user(len(fallback_imgs), missing, obj_missing)))
        obj, _ = llm.chat_json([{"role": "system", "content": P.ANSWERS_SYSTEM}, {"role": "user", "content": content}],
                               st["vision_model"], thinking=False, max_tokens=32000)
        for it in (obj or {}).get("items") or []:
            n = norm_number((it or {}).get("number")) if isinstance(it, dict) else ""
            if n in sol and n not in found and str(it.get("answer") or "").strip():
                found[n] = {"answer": str(it["answer"]).strip(), "crops": [], "text": "", "markers": None, "file": "图片"}
        objd = (obj or {}).get("objective") if isinstance(obj, dict) else None
        if isinstance(objd, dict):
            for k, v in objd.items():
                n = norm_number(k)
                if n in objq and n not in obj_found and str(v or "").strip():
                    obj_found[n] = {"answer": _clean_obj_answer(objq[n]["type"], v), "crops": []}
    text = load_text_files(paths)
    missing = [n for n in sol if n not in found]
    if text.strip() and missing:
        obj, _ = llm.chat_json([{"role": "system", "content": P.ANSWERS_SYSTEM},
                                {"role": "user", "content": P.answers_text_user(text, missing)}],
                               st["vision_model"], thinking=False, max_tokens=32000)
        for it in (obj or {}).get("items") or []:
            n = norm_number((it or {}).get("number")) if isinstance(it, dict) else ""
            if n in sol and n not in found and str(it.get("answer") or "").strip():
                found[n] = {"answer": str(it["answer"]).strip(), "crops": [], "text": "", "markers": None, "file": "文字"}

    for n, f in found.items():
        q = sol[n]
        q["checks"] = [c for c in q.get("checks", []) if c.get("where") != "答案"]
        if f.get("error"):
            _add_check(q, "答案", f"参考答案录入失败：{f['error']}")
            continue
        ans = f.get("answer") or ""
        if not ans:
            _add_check(q, "答案", "模型没有录出这道题的答案，请对照原图手动填写")
            continue
        q["answer"] = ans
        q["answer_source"] = "key"
        q["confirmed"] = False
        q["answer_crops"] = _save_crops(d, "answer", n, f["crops"]) if f["crops"] else []
        if f["text"]:
            _compare_checks(q, "答案", f["text"], ans)
        dup = D.restart_labels(ans)
        if dup:
            _add_check(q, "答案", f"答案里小问 {dup[0]} 出现了两次，可能混进了别的题的内容")
        _apply_marker_scores(q, f["markers"] if f["markers"] is not None else D.markers_from_text(ans))
    for n, f in obj_found.items():
        q = objq[n]
        if f["answer"]:
            q["answer"] = f["answer"]
            q["answer_source"] = "key"
            q["answer_crops"] = f.get("crops") or []
    if obj_found:
        log.append(f"选择题、填空题答案：录入 {len(obj_found)} 道")
    lost_obj = [n for n in sorted(objq, key=_numkey) if n not in obj_found]
    if lost_obj and objq:
        log.append(f"参考答案里没有找到第 {'、'.join(lost_obj)} 题（选择/填空）的答案")
    if found:
        log.append(f"参考答案：录入第 {'、'.join(sorted(found, key=_numkey))} 题")
    lost = [n for n in sol if n not in found]
    if lost:
        log.append(f"参考答案里没有找到第 {'、'.join(sorted(lost, key=_numkey))} 题")
    return log


def _clean_obj_answer(kind: str, v) -> str:
    t = str(v or "").strip().strip("$").strip()
    if kind == "选择题":
        bare = re.sub(r"[\s、，,;；和及与/]+", "", t)
        if re.fullmatch(r"[A-Ha-h]{1,8}", bare):             # "B"、"BD"、"b、d"
            letters = bare.upper()
        else:                                                # "选 B"、"B. 3"、"\text{B}"；不能把 \frac 当成 FAC
            t = re.sub(r"\\[A-Za-z]+", " ", t)
            letters = "".join(re.findall(r"(?<![A-Za-z])([A-H]{1,8})(?![A-Za-z])", t))
        t = "".join(sorted(set(letters)))
    return t


def _objective_from_pdf(llm: Optional[DeepSeek], st: dict, d: Path, pdf: D.PdfText, start,
                        objq: Dict[str, dict]) -> Dict[str, dict]:
    """数字版答案里的选择题、填空题答案：选择题直接从文字读字母；填空题（公式是乱码）整段截图交给模型。"""
    types = {n: q["type"] for n, q in objq.items()}
    out: Dict[str, dict] = {}
    for kind, (pieces, rows) in D.objective_spans(pdf, types, start).items():
        nums = sorted([n for n, t in types.items() if t == kind], key=_numkey)
        crops = [pdf.crop(pc, dpi=220) for pc in pieces]
        rels = _save_crops(d, "answer", "choice" if kind == "选择题" else "fill", crops)
        got = D.parse_choice_answers(rows, nums) if kind == "选择题" else {}
        miss = [n for n in nums if n not in got]
        if miss and llm is not None and crops:
            content = [image_part(to_data_url(c, 2000)) for c in crops]
            content.append(text_part(P.objective_answers_user(kind, miss, len(crops), "\n".join(r.text for r in rows))))
            try:
                obj, _ = llm.chat_json([{"role": "system", "content": P.ANSWERS_SYSTEM}, {"role": "user", "content": content}],
                                       st["vision_model"], thinking=False, max_tokens=6000)
                for k, v in ((obj or {}).get("answers") or {}).items():
                    n = norm_number(k)
                    if n in miss and str(v or "").strip():
                        got[n] = v
            except LLMError:
                pass
        for n, v in got.items():
            out[n] = {"answer": _clean_obj_answer(kind, v), "crops": rels}
    return out


def _apply_marker_scores(q: dict, markers) -> None:
    """用答案里的“……X分”推出各小问分值，并和试卷上写的分值互相核对。"""
    if not markers:
        q["score_convention"] = ""
        return
    stated = {p["label"]: float(p["score"]) for p in q.get("parts") or [] if p.get("score") is not None}
    total = float(q["score"]) if q.get("score") is not None else None
    derived, conv = D.scores_from_markers(list(markers), stated, total)
    q["score_convention"] = conv
    derived = {k: v for k, v in derived.items() if k != "(全题)"} or derived
    if not q.get("parts") and set(derived) != {"(全题)"}:
        q["parts"] = [{"label": k, "text": "", "score": _num(v)} for k, v in derived.items()]
    else:
        for p in q.get("parts") or []:
            v = derived.get(p["label"])
            if v is None:
                continue
            if p.get("score") is None:
                p["score"] = _num(v)
            elif abs(float(p["score"]) - v) > 1e-6:
                _add_check(q, "答案", f"答案里 {p['label']} 的分值标注合计 {v:g} 分，试卷写的是 {p['score']} 分，请核对")
    if total is None and derived:
        q["score"] = _num(sum(derived.values()))
    elif total is not None and derived and set(derived) != {"(全题)"} and abs(sum(derived.values()) - total) > 1e-6:
        _add_check(q, "答案", f"答案里各小问分值合计 {sum(derived.values()):g} 分，与满分 {total:g} 分不一致")


def generate_answer(llm: DeepSeek, st: dict, q: dict, paper_imgs: List[np.ndarray]) -> None:
    """让模型解题并写出评分细则（就地修改 q，标记为 AI 生成、待确认）。paper_imgs 是这道题的原题截图。"""
    model = st["answer_model"]
    content = []
    if model_supports_images(model):
        for img in paper_imgs[:4]:
            content.append(image_part(to_data_url(img, 2000)))
    content.append(text_part(P.gen_user(q)))
    obj, _ = llm.chat_json([{"role": "system", "content": P.GEN_SYSTEM}, {"role": "user", "content": content}],
                           model, thinking=True, effort=st.get("answer_effort", "max"), max_tokens=48000)
    q["answer"] = str(obj.get("answer") or "").strip()
    q["rubric"] = str(obj.get("rubric") or "").strip()
    q["issues"] = str(obj.get("issues") or "").strip()
    _apply_part_scores(q, obj.get("parts") or [])
    fix_part_scores(q)
    q["answer_source"] = "ai"
    q["answer_crops"] = []
    q["score_convention"] = ""
    q["confirmed"] = False


def _same_obj_answer(kind: str, a: str, b: str) -> bool:
    if not a or not b:
        return False
    if kind == "选择题":
        return sorted(set(a)) == sorted(set(b))
    return O.exact_match(a, b)


def solve_objective(llm: DeepSeek, st: dict, q: dict, paper_imgs: List[np.ndarray], fresh: bool = False) -> None:
    """没有答案的选择题、填空题：让模型独立解两遍。两遍一致就用；不一致取把握大的那个，并标出来请老师确认。
    就地修改 q（answer_source = "ai"，待确认）。fresh=True（老师点“AI 重做”）时不用上次的缓存结果。"""
    model = st["answer_model"]
    vision = model_supports_images(model)
    tries = []
    for k in range(2):
        content = [image_part(to_data_url(c, 1600)) for c in paper_imgs[:3]] if vision else []
        content.append(text_part(P.obj_solve_user(q, len(content), k)))
        obj, _ = llm.chat_json([{"role": "system", "content": P.OBJ_SOLVE_SYSTEM}, {"role": "user", "content": content}],
                               model, thinking=True, effort="high", max_tokens=24000, use_cache=not fresh)
        obj = obj if isinstance(obj, dict) else {}
        conf = str(obj.get("confidence") or "").lower()
        tries.append({"answer": _clean_obj_answer(q["type"], obj.get("answer")), "brief": str(obj.get("brief") or "")[:300],
                      "confidence": conf if conf in ("high", "medium", "low") else "medium"})
    a, b = tries
    rank = {"high": 2, "medium": 1, "low": 0}
    pick = a if (a["answer"] and rank[a["confidence"]] >= rank[b["confidence"]]) or not b["answer"] else b
    q["answer"] = pick["answer"]
    q["answer_source"] = "ai"
    q["confirmed"] = False
    q["ai_solve"] = {"answers": [a["answer"], b["answer"]], "brief": pick["brief"],
                     "agree": _same_obj_answer(q["type"], a["answer"], b["answer"]),
                     "confidence": pick["confidence"]}
    q["checks"] = [c for c in q.get("checks") or [] if c.get("where") != "答案"]
    if not pick["answer"]:
        _add_check(q, "答案", "AI 没有做出答案，请老师填写")
    elif not q["ai_solve"]["agree"]:
        _add_check(q, "答案", f"AI 两次独立解答不一样（{a['answer'] or '空'} / {b['answer'] or '空'}），请老师确认")
    elif pick["confidence"] == "low":
        _add_check(q, "答案", "AI 对这个答案把握不大，请老师确认")


def generate_objective_answers(llm: DeepSeek, st: dict, d: Path, questions: List[dict], redo: bool = False,
                               progress: Progress = _noop) -> List[str]:
    """给没有答案的选择题、填空题做答案（redo=True 时 AI 做过、老师还没确认的也重做）。返回日志。"""
    todo = [q for q in questions if q.get("type") in ("选择题", "填空题")
            and (not (q.get("answer") or "").strip() or (redo and q.get("answer_source") == "ai" and not q.get("confirmed")))
            and (q.get("paper_crops") or (q.get("stem") or "").strip())]
    log: List[str] = []
    if not todo:
        return log

    def run(q):
        try:
            solve_objective(llm, st, q, _load_crops(d, q.get("paper_crops")), fresh=redo and bool((q.get("answer") or "").strip()))
            return None
        except LLMError as e:
            return f"第 {q['number']} 题 AI 解题失败：{e}"
    with _pool(st) as ex:
        for k, err in enumerate(ex.map(run, todo)):
            progress((k + 1) / len(todo), f"AI 做选择、填空题 {k + 1}/{len(todo)}")
            if err:
                log.append(err)
    bad = [q["number"] for q in todo if any(c.get("where") == "答案" for c in q.get("checks") or [])]
    log.append(f"AI 做了 {len(todo)} 道选择、填空题的答案" + (f"，其中第 {'、'.join(bad)} 题请老师确认" if bad else "，两遍解答一致"))
    return log


def _apply_part_scores(q: dict, parts) -> None:
    if not parts:
        return
    norm = [{"label": norm_label(p.get("label"), k), "score": _num(p.get("score"))}
            for k, p in enumerate(parts) if isinstance(p, dict)]
    if not q["parts"]:
        q["parts"] = [{"label": p["label"], "text": "", "score": p["score"]} for p in norm]
        return
    byl = {p["label"]: p for p in norm}
    for k, p in enumerate(q["parts"]):
        src = byl.get(p["label"]) or (norm[k] if k < len(norm) else None)
        if src and p.get("score") is None and src["score"] is not None:
            p["score"] = src["score"]


def fix_part_scores(q: dict) -> None:
    """小问分值缺失时补齐，使各小问之和等于满分。"""
    parts = q.get("parts") or []
    total = q.get("score")
    if not parts or total is None:
        return
    known = sum(p["score"] for p in parts if p.get("score") is not None)
    unknown = [p for p in parts if p.get("score") is None]
    if unknown:
        rest = max(0, total - known)
        each = rest / len(unknown)
        for k, p in enumerate(unknown):
            p["score"] = int(each) if k < len(unknown) - 1 else rest - int(each) * (len(unknown) - 1)


# ================================================================ 版面

def load_sheet_sources(paths: Sequence[Path], dpi: int) -> List[Tuple[np.ndarray, Optional[list], int]]:
    """答题纸每页：(图像, PDF 文字碎片或 None, dpi)。数字版 PDF 的文字可直接用来标题号。"""
    out = []
    for p in paths:
        pdf = _open_pdf(p)
        imgs = load_pages(p, dpi=dpi)
        for i, img in enumerate(imgs):
            frags = pdf.frags[i] if pdf is not None and i < len(pdf.frags) and pdf.frags[i] else None
            out.append((img, frags, dpi))
        if pdf is not None:
            pdf.close()
    return out


def build_template(store: Store, exam_id: str, sources: List[Tuple[np.ndarray, Optional[list], int]], questions: List[dict],
                   llm: Optional[DeepSeek], st: dict, source: str, progress: Progress = _noop) -> dict:
    d = store.exam_dir(exam_id)
    pages = []
    layouts = []
    text_labeled = []
    for p, (img, frags, dpi) in enumerate(sources):
        progress(0.1 + 0.3 * p / max(1, len(sources)), f"分析版面：第 {p + 1} 页")
        lay = analyze_page(img)
        layouts.append(lay)
        imwrite(d / "template" / f"page_{p}.png", lay.image)
        imwrite(d / "template" / f"page_{p}_lines.png", lay.line_mask)
        H, W = lay.image.shape[:2]
        regions = [{"id": f"R{k + 1}", "rect": [int(v) for v in b], "kind": "", "questions": [],
                    "full_score": None, "header": ""} for k, b in enumerate(lay.boxes)]
        n_text = 0
        numbers = []
        if frags:
            k = dpi / 72.0
            to_t = lambda x, y, k=k, lay=lay: lay.map_point(x * k, y * k)
            n_text = D.label_boxes_from_text(frags, regions, to_t)
            numbers = [[q, round(a, 1), round(b, 1), round(c, 1), round(e, 1)]
                       for q, a, b, c, e in O.number_marks(frags, to_t, (0, 0, W, H))]
        text_labeled.append(n_text)
        pages.append({"page": p, "image": f"template/page_{p}.png", "lines": f"template/page_{p}_lines.png",
                      "size": [W, H], "print_color": lay.print_color, "skew": round(lay.skew_deg, 3),
                      "markers": [m.to_list() for m in lay.markers], "regions": regions, "notes": list(lay.notes),
                      "unboxed": "", "numbers": numbers,
                      "labeled_by": "pdf_text" if n_text and n_text == len(regions) else ""})
    tmpl = {"pages": pages, "source": source, "labeled_by": "none", "version": 2}
    need_ai = [pg for pg in pages if pg["regions"] and pg["labeled_by"] != "pdf_text"]
    if not need_ai and pages:
        tmpl["labeled_by"] = "pdf_text"
    elif llm is not None and llm.api_key:
        try:
            for k, pg in enumerate(need_ai):
                progress(0.45 + 0.45 * k / max(1, len(need_ai)), f"识别题号：第 {pg['page'] + 1} 页")
                label_page(llm, st, pg, layouts[pg["page"]].image, len(pages), questions)
                pg["labeled_by"] = "ai"
            tmpl["labeled_by"] = "ai" if not any(text_labeled) else "pdf_text+ai"
        except LLMError as e:
            for pg in pages:
                pg["notes"].append(f"AI 标注失败：{e}")
            heuristic_label(tmpl, questions)
            tmpl["labeled_by"] = "heuristic"
    else:
        heuristic_label(tmpl, questions)
        tmpl["labeled_by"] = "heuristic"
    for pg, lay in zip(pages, layouts):
        attach_objective(pg, lay.image, source)
    render_previews(store, exam_id, tmpl)
    progress(1.0, "版面识别完成")
    return tmpl


def attach_objective(pg: dict, image: np.ndarray, source: str, only: Optional[set] = None) -> None:
    """填空题区域找出每个空（横线），选择题区域找出每道题的选项框；空白模板上记下每个选项框本身的暗度作基准。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    nums = [tuple(x) for x in pg.get("numbers") or []]
    for r in pg["regions"]:
        if only is not None and r["id"] not in only:
            continue
        r.pop("slots", None)
        r.pop("groups", None)
        if r.get("kind") not in ("填空题", "选择题"):
            continue
        x0, y0, x1, y1 = r["rect"]
        mine = [t for t in nums if x0 <= (t[1] + t[3]) / 2 <= x1 and y0 <= (t[2] + t[4]) / 2 <= y1] or None
        qs = r.get("questions") or []
        if r["kind"] == "填空题":
            r["slots"] = O.detect_fill_slots(image, r["rect"], mine, qs)
            found = [x["q"] for x in r["slots"] if x["q"]]
        else:
            gs = O.detect_choice_groups(image, r["rect"], mine, qs)
            if source == "blank":
                paper = O.paper_level(gray, r["rect"])
                for g in gs:
                    for o, b in zip(g["options"], O.option_darkness(gray, g, paper)):
                        o["base"] = b
            r["groups"] = gs
            found = [x["q"] for x in gs if x["q"]]
        if found and (not qs or set(found) - set(qs)):
            r["questions"] = sorted(set(qs) | set(found), key=_numkey)


def label_page(llm: DeepSeek, st: dict, page: dict, image: np.ndarray, n_pages: int, questions: List[dict]) -> None:
    regions = page["regions"]
    som = render_regions(image, regions, with_info=False)
    content = [image_part(to_data_url(som, 2400))]
    for k in range(0, len(regions), 6):  # 每张放大图最多 6 个区域，保证字看得清
        content.append(image_part(to_data_url(header_sheet(image, regions[k:k + 6]), 2400)))
    content.append(text_part(P.layout_user(page["page"] + 1, n_pages, [r["id"] for r in regions], questions)))
    obj, _ = llm.chat_json([{"role": "system", "content": P.LAYOUT_SYSTEM}, {"role": "user", "content": content}],
                           st["vision_model"], thinking=True, effort="high", max_tokens=16000)
    byid = {r["id"]: r for r in regions}
    for it in (obj or {}).get("regions") or []:
        r = byid.get(str(it.get("id", "")).strip().upper())
        if r is None or r.get("kind"):   # 已经从 PDF 文字读出来的不覆盖
            continue
        kind = str(it.get("kind") or "")
        r["kind"] = kind if kind in KINDS else (norm_type(kind) or "其他")
        r["questions"] = [norm_number(x) for x in (it.get("questions") or []) if norm_number(x)]
        r["full_score"] = _num(it.get("full_score"))
        r["header"] = str(it.get("header") or "")[:60]
    page["unboxed"] = str((obj or {}).get("unboxed") or "")


def heuristic_label(tmpl: dict, questions: List[dict]) -> None:
    """没有 AI 时的兜底：大框按阅读顺序依次对应解答题题号。"""
    sol = [q["number"] for q in questions if q.get("type") == "解答题"]
    sol.sort(key=_numkey)
    big = []
    taken = {n for pg in tmpl["pages"] for r in pg["regions"] if r.get("kind") == "解答题" for n in r.get("questions") or []}
    sol = [n for n in sol if n not in taken]
    for pg in tmpl["pages"]:
        W, H = pg["size"]
        for r in pg["regions"]:
            if r.get("kind"):
                continue
            x0, y0, x1, y1 = r["rect"]
            if (y1 - y0) >= 0.12 * H and (x1 - x0) * (y1 - y0) >= 0.04 * W * H:
                big.append(r)
            elif not r.get("kind"):
                r["kind"] = "其他"
    # 解答题一般排在最后：大框比解答题多时，前面多出来的（通常是填空题框）不算
    if sol and len(big) > len(sol):
        for r in big[:len(big) - len(sol)]:
            r["kind"] = r.get("kind") or "其他"
        big = big[len(big) - len(sol):]
    for k, r in enumerate(big):
        r["kind"] = "解答题"
        if k < len(sol):
            r["questions"] = [sol[k]]


def render_previews(store: Store, exam_id: str, tmpl: dict) -> None:
    d = store.exam_dir(exam_id)
    for pg in tmpl["pages"]:
        img = cv2.imread(str(d / pg["image"]))
        if img is None:
            continue
        prev = render_regions(img, pg["regions"], with_info=True, markers=pg["markers"])
        imwrite(d / "template" / f"page_{pg['page']}_preview.jpg", prev)
        pg["preview"] = f"template/page_{pg['page']}_preview.jpg"


def objective_places(tmpl: Optional[dict]) -> Dict[str, dict]:
    """答题纸上每道填空/选择题的位置：{题号: {"kind", "page", "region", "slot"|"group"}}。
    区域没能拆出每道题时，记为整块区域（"whole": True）。"""
    out: Dict[str, dict] = {}
    if not tmpl:
        return out
    for pg in tmpl["pages"]:
        for r in pg["regions"]:
            if r.get("kind") == "填空题":
                for sl in r.get("slots") or []:
                    if sl.get("q") and sl["q"] not in out:
                        out[sl["q"]] = {"kind": "填空题", "page": pg["page"], "region": r["id"], "slot": sl}
            elif r.get("kind") == "选择题":
                for g in r.get("groups") or []:
                    if g.get("q") and g["q"] not in out:
                        out[g["q"]] = {"kind": "选择题", "page": pg["page"], "region": r["id"], "group": g}
            else:
                continue
            for n in r.get("questions") or []:
                if n not in out:
                    out[n] = {"kind": r["kind"], "page": pg["page"], "region": r["id"], "whole": True}
    return out


def _objective_warnings(exam: dict, qs: List[dict]) -> List[str]:
    warns = []
    tmpl = exam.get("template")
    places = objective_places(tmpl)
    for kind in ("填空题", "选择题"):
        mine = [q for q in qs if q.get("type") == kind]
        if not mine:
            continue
        no_ans = [q["number"] for q in mine if not (q.get("answer") or "").strip()]
        no_score = [q["number"] for q in mine if q.get("score") in (None, "")] if is_scored(exam) else []
        ai = [q["number"] for q in mine if q.get("answer_source") == "ai" and not q.get("confirmed") and (q.get("answer") or "").strip()]
        if no_ans:
            warns.append(f"{kind}第 {'、'.join(no_ans)} 题还没有参考答案，这些题不会批改"
                         + "（可以在表格上方点“让 AI 做”）")
        if ai:
            warns.append(f"{kind}第 {'、'.join(ai)} 题的答案是 AI 做的，请核对（表格里标了“AI”，核对过点“确认”）")
        for q in mine:
            for c in q.get("checks") or []:
                if c.get("where") == "答案" and not q.get("confirmed"):
                    warns.append(f"第 {q['number']} 题：{c.get('msg', '')}")
        if no_score:
            warns.append(f"{kind}第 {'、'.join(no_score)} 题没有分值")
        if tmpl:
            lost = [q["number"] for q in mine if q["number"] not in places]
            whole = [q["number"] for q in mine if places.get(q["number"], {}).get("whole")]
            if lost:
                warns.append(f"{kind}第 {'、'.join(lost)} 题在答题纸上没有找到作答位置，请在“答题纸版面”里检查")
            if whole:
                warns.append(f"{kind}第 {'、'.join(whole)} 题没能定位到每道题的作答位置，会整块交给模型识别，请在“答题纸版面”里检查")
        for title in exam.get("section_notes") or []:
            if D.section_type(title) == kind:
                tot = D.section_total(title)
                got = sum(float(q["score"]) for q in mine if q.get("score") not in (None, ""))
                if tot and not no_score and abs(got - tot) > 1e-6:
                    warns.append(f"{kind}各题分值合计 {got:g} 分，与大题说明的 {tot:g} 分不一致，请核对")
    return warns


def regions_for(tmpl: Optional[dict], number: str) -> List[Tuple[int, dict]]:
    out = []
    if not tmpl:
        return out
    for pg in tmpl["pages"]:
        for r in pg["regions"]:
            if number in (r.get("questions") or []) and r.get("kind") in ("解答题", "", None, "其他"):
                out.append((pg["page"], r))
    return out


def reconcile(exam: dict) -> List[str]:
    """核对试卷题目和答题纸区域，补齐分值，返回提示。"""
    warns: List[str] = []
    qs = exam["questions"]
    byn = {q["number"]: q for q in qs}
    tmpl = exam.get("template")
    if tmpl:
        for pg in tmpl["pages"]:
            for r in pg["regions"]:
                if r.get("kind") != "解答题":
                    continue
                for n in r.get("questions") or []:
                    if n not in byn:
                        q = new_question(n, "解答题")
                        q["score"] = r.get("full_score") if len(r.get("questions")) == 1 else None
                        qs.append(q)
                        byn[n] = q
                        warns.append(f"答题纸上有第 {n} 题的作答区，但试卷里没录到这道题，请补充题目和答案")
                    q = byn[n]
                    if q.get("type") in ("", "其他"):
                        q["type"] = "解答题"
                    fs = r.get("full_score")
                    if fs and len(r.get("questions")) == 1:
                        if q.get("score") is None:
                            q["score"] = fs
                        elif float(q["score"]) != float(fs):
                            warns.append(f"第 {n} 题：试卷写 {q['score']} 分，答题纸印 {fs} 分，请核对")
    qs.sort(key=lambda q: _numkey(q["number"]))
    if tmpl:
        unnumbered = [f"第{pg['page'] + 1}页 {r['id']}" for pg in tmpl["pages"] for r in pg["regions"]
                      if r.get("kind") == "解答题" and not r.get("questions")]
        if unnumbered:
            warns.append(f"版面上有 {len(unnumbered)} 个解答题框还没填题号（{'、'.join(unnumbered[:6])}），请在“答题纸版面”里点开填写")
    gen = (tmpl or {}).get("generated")
    if gen and gen.get("sig"):
        from .sheetgen import questions_sig
        if questions_sig(qs, bool((gen.get("plan") or {}).get("scored", True))) != gen["sig"]:
            warns.append("答题纸是按改动之前的题目生成的（题号、题型、分值或选项数变了），请到“答题纸版面”里重新生成；"
                         "已经印好的旧答题纸和新版面对不上")
        elif bool((gen.get("plan") or {}).get("scored", True)) != is_scored(exam):
            warns.append("答题纸上" + ("没有印分值，现在要打分" if is_scored(exam) else "印了分值，现在是平时作业（不打分）")
                         + "；不影响批改，想换掉的话到“答题纸版面”里重新生成")
    if not qs:
        warns.append("还没有题目：上传试卷让 AI 录题，或者在“题目与评分细则”里按题号快速添加")
    warns += _objective_warnings(exam, qs)
    scored = is_scored(exam)
    blind = []          # 平时作业：既没有题目也没有答案的解答题
    for q in qs:
        fold_parts_into_stem(q)
        if q.get("type") != "解答题":
            continue
        fix_part_scores(q)
        for c in q.get("checks") or []:
            warns.append(f"第 {q['number']} 题{c.get('where', '')}：{c.get('msg', '')}")
        dup = D.restart_labels(q.get("answer") or "") if not q.get("confirmed") else []
        if dup and not any("出现了两次" in c.get("msg", "") for c in q.get("checks") or []):
            warns.append(f"第 {q['number']} 题答案：小问 {dup[0]} 出现了两次，可能混进了别的题的内容，请核对或重新录入答案")
        if tmpl and not regions_for(tmpl, q["number"]):
            warns.append(f"第 {q['number']} 题在答题纸上没有对应的作答区域，请在“版面”里标注")
        if q.get("score") is None and scored:
            warns.append(f"第 {q['number']} 题没有满分值")
        parts = q.get("parts") or []
        if scored and parts and q.get("score") is not None and all(p.get("score") is not None for p in parts):
            s = sum(float(p["score"]) for p in parts)
            if abs(s - float(q["score"])) > 1e-6:
                warns.append(f"第 {q['number']} 题小问分值之和（{s:g}）不等于满分（{q['score']}）")
        if not q.get("answer"):
            if scored:
                warns.append(f"第 {q['number']} 题还没有参考答案")
            elif not (q.get("stem") or "").strip() and not q.get("paper_crops"):
                blind.append(q["number"])           # 作业：有题目的，批改时 AI 自己先做再判，不用提醒
        elif q.get("answer_source") == "ai" and not q.get("confirmed"):
            warns.append(f"第 {q['number']} 题的参考答案{'和细则' if scored else ''}是 AI 生成的，请核对后点“确认”")
        if q.get("stem") == "" and q.get("paper_crops"):
            warns.append(f"第 {q['number']} 题题干还没录入")
    if blind:
        warns.append(f"解答题第 {'、'.join(blind)} 题既没有题目也没有参考答案，AI 只能看学生写的过程判断对错，可能不准；"
                     "可以上传试卷（拍照也行），或者填上题目、参考答案")
    exam["questions"] = qs
    return warns


# ================================================================ 总流程

def run_setup(store: Store, exam_id: str, llm: Optional[DeepSeek], st: dict, progress: Progress = _noop,
              generate_missing: bool = True) -> dict:
    exam = store.load_exam(exam_id)
    d = store.exam_dir(exam_id)
    files = exam["files"]
    log: List[str] = []
    (d / "source").mkdir(exist_ok=True)

    paper_paths = [d / f for f in files.get("paper", [])]
    questions = exam.get("questions") or []
    combined: List[tuple] = []
    if paper_paths and llm is not None:
        progress(0.02, "录入试卷题目…")
        qs, notes, combined, plog = parse_paper(llm, st, d, paper_paths, lambda f, m: progress(0.02 + 0.3 * f, m))
        log += plog
        old = {q["number"]: q for q in questions}
        for q in qs:   # 没有重新上传答案时，保留老师已经填过的答案
            o = old.get(q["number"])
            if o and o.get("answer"):
                q.update({k: o[k] for k in ("answer", "rubric", "notes", "answer_source", "confirmed",
                                            "answer_crops", "score_convention") if k in o})
                q["checks"] = [c for c in o.get("checks", []) if c.get("where") == "答案"] + q.get("checks", [])
        questions = qs
        exam["section_notes"] = notes
        n_sol = len([q for q in qs if q["type"] == "解答题"])
        log.append(f"试卷共 {len(qs)} 道题，其中解答题 {n_sol} 道")

    ans_paths = [d / f for f in files.get("answers", [])]
    if (ans_paths or combined) and llm is not None:
        progress(0.35, "录入参考答案…")
        log += parse_answers(llm, st, d, ans_paths, questions, combined, lambda f, m: progress(0.35 + 0.17 * f, m))

    sheet_paths = [d / f for f in files.get("sheet", [])]
    old_tmpl = exam.get("template") or {}
    keep_layout = old_tmpl.get("labeled_by") == "teacher" or bool(old_tmpl.get("generated"))
    if keep_layout and sheet_paths:
        log.append("答题纸是按题目生成的，或者版面老师修改过，这次没有重新识别（要重新识别请到“答题纸版面”里操作）")
    if sheet_paths and not keep_layout:
        progress(0.53, "识别答题纸版面…")
        sources = load_sheet_sources(sheet_paths, int(st.get("scan_dpi", 250)))
        exam["template"] = build_template(store, exam_id, sources, questions, llm, st, "blank",
                                          lambda f, m: progress(0.53 + 0.17 * f, m))
    exam["questions"] = questions
    reconcile(exam)

    if generate_missing and llm is not None:
        log += generate_objective_answers(llm, st, d, exam["questions"],
                                          progress=lambda f, m: progress(0.62 + 0.1 * f, m))
        todo = [q for q in exam["questions"] if q["type"] == "解答题" and not q.get("answer") and q.get("stem")]
        if todo:
            progress(0.72, f"生成 {len(todo)} 道解答题的参考答案和评分细则（较慢，请稍候）…")

            def gen(q):
                try:
                    generate_answer(llm, st, q, _load_crops(d, q.get("paper_crops")))
                    return None
                except LLMError as e:
                    return f"第 {q['number']} 题参考答案生成失败：{e}"
            with _pool(st) as ex:
                for k, err in enumerate(ex.map(gen, todo)):
                    progress(0.72 + 0.27 * (k + 1) / len(todo), f"参考答案 {k + 1}/{len(todo)}")
                    if err:
                        log.append(err)
            log.append(f"AI 生成了 {len(todo)} 道题的参考答案和评分细则，请核对")

    def apply(e: dict):
        e["questions"] = exam["questions"]
        if exam.get("template") is not None:
            e["template"] = exam["template"]
        e["section_notes"] = exam.get("section_notes", [])
        e["warnings"] = reconcile(e)
        e["setup_log"] = log
    exam = store.update_exam(exam_id, apply)
    progress(1.0, "准备完成")
    return exam


def run_layout(store: Store, exam_id: str, img_paths: List[Path], llm: Optional[DeepSeek], st: dict,
               source: str, progress: Progress = _noop) -> dict:
    """单独（重新）识别版面：空白答题纸，或一份学生答题纸。"""
    exam = store.load_exam(exam_id)
    sources = load_sheet_sources(img_paths, int(st.get("scan_dpi", 250)))
    n = int(exam.get("pages_per_sheet") or 1)
    if source == "scan":
        sources = [(img, None, dpi) for img, _f, dpi in sources[:n]]
    tmpl = build_template(store, exam_id, sources, exam.get("questions") or [], llm, st, source, progress)

    def apply(e: dict):
        e["template"] = tmpl
        e["warnings"] = reconcile(e)
    return store.update_exam(exam_id, apply)


def generate_sheet(store: Store, exam_id: str, paper: str = "A4", id_digits: int = 4, space: str = "normal",
                   title: str = "", notice: bool = True, heights: Optional[dict] = None) -> dict:
    """按题目生成答题纸（PDF）并直接把它设为这场考试的版面。heights：老师拖动定下的解答题方框高度 {题号: mm}。
    平时作业（不打分）不印分值，方框高度按小问数给。"""
    from . import sheetgen
    exam = store.load_exam(exam_id)
    plan = sheetgen.plan_sheet(exam.get("questions") or [], title or exam.get("title") or "", paper, id_digits, space, notice,
                               heights=heights, scored=is_scored(exam))
    tmpl = sheetgen.build_template(store.exam_dir(exam_id), exam_id, plan)
    render_previews(store, exam_id, tmpl)

    def apply(e: dict):
        e["template"] = tmpl
        e["pages_per_sheet"] = len(plan["pages"])
        e["warnings"] = reconcile(e)
    return store.update_exam(exam_id, apply)


_TYPE_WORDS = [("多选", "选择题", True), ("多项选择", "选择题", True), ("不定项", "选择题", True),
               ("单选", "选择题", False), ("单项选择", "选择题", False), ("选择", "选择题", False),
               ("填空", "填空题", False), ("解答", "解答题", False), ("大题", "解答题", False), ("计算", "解答题", False),
               ("证明", "解答题", False), ("应用", "解答题", False), ("简答", "解答题", False)]


def parse_structure(text: str) -> Tuple[List[dict], List[str]]:
    """没有试卷文件时，按“题号 题型 分值”快速录入，每行一段，例如：
        1-8 单选 每题5分
        9-11 多选 6分
        12-14 填空 5分
        15 解答 13分 2问
        16-17 解答 15分
    返回 (题目列表, 看不懂的行)。"""
    out: List[dict] = []
    bad: List[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        s = line.translate(str.maketrans("０１２３４５６７８９～—－，、；：", "0123456789~--,,;:"))
        kind = multi = None
        for word, t, m in _TYPE_WORDS:
            if word in s:
                kind, multi = t, m
                break
        rest = re.sub(r"第|题|号", " ", s)
        nums: List[str] = []
        m = re.search(r"(\d+)\s*(?:-|~|至|到)\s*(\d+)", rest)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a <= b <= a + 60:
                nums = [str(i) for i in range(a, b + 1)]
            rest_after = rest[:m.start()] + " " + rest[m.end():]
        else:
            lead = re.match(r"\s*((?:\d+\s*,\s*)*\d+)(?!\s*(?:分|问|个|项|空))", rest)
            if lead:
                nums = [x.strip() for x in lead.group(1).split(",") if x.strip()]
                rest_after = rest[lead.end():]
            else:
                rest_after = rest
        if not kind or not nums:
            bad.append(line)
            continue
        score = None
        total = re.search(r"共\s*(\d+(?:\.\d+)?)\s*分", rest_after)
        each = re.search(r"(\d+(?:\.\d+)?)\s*分", rest_after)
        if total and not re.search(r"每\s*(?:小)?题", rest_after):
            score = round(float(total.group(1)) / len(nums), 2)
        elif each:
            score = float(each.group(1))
        parts_n = re.search(r"(\d+)\s*(?:个)?\s*(?:小问|问)", rest_after)
        opts = re.search(r"(\d+)\s*(?:个)?\s*选项", rest_after) or re.search(r"\b(A-?[E-H])\b", rest_after)
        for n in nums:
            q = new_question(n, kind)
            q["score"] = int(score) if score is not None and float(score).is_integer() else score
            if kind == "选择题":
                q["multi"] = bool(multi)
                if opts:
                    g = opts.group(1)
                    q["options"] = int(g) if g.isdigit() else ord(g[-1]) - ord("A") + 1
            if kind == "解答题" and parts_n:
                q["parts"] = [{"label": f"({k + 1})", "text": "", "score": None} for k in range(min(8, int(parts_n.group(1))))]
            out.append(q)
    return out, bad


def merge_structure(existing: List[dict], parsed: List[dict]) -> List[dict]:
    """快速录入的题目并进现有题目：同题号的只改题型、分值、选项数、小问数（答案和题干保留）。"""
    by = {q["number"]: q for q in existing}
    for q in parsed:
        o = by.get(q["number"])
        if o is None:
            by[q["number"]] = q
            continue
        o["type"] = q["type"]
        if q.get("score") is not None:
            o["score"] = q["score"]
        for k in ("multi", "options"):
            if k in q:
                o[k] = q[k]
        if q.get("parts") and not o.get("parts"):
            o["parts"] = q["parts"]
    return sorted(by.values(), key=lambda q: _numkey(q["number"]))


def save_template_edit(store: Store, exam_id: str, pages: List[dict]) -> dict:
    """老师在网页上修改了区域（增删改）。填空/选择区域如果位置或题型变了，重新找横线和选项框。"""
    d = store.exam_dir(exam_id)
    changed: Dict[int, set] = {}

    def apply(e: dict):
        tmpl = e.get("template")
        if not tmpl:
            raise ValueError("还没有版面")
        byp = {pg["page"]: pg for pg in tmpl["pages"]}
        for new in pages:
            pg = byp.get(int(new.get("page", -1)))
            if pg is None:
                continue
            W, H = pg["size"]
            old_by_id = {r["id"]: r for r in pg["regions"]}
            regions = []
            for k, r in enumerate(new.get("regions") or []):
                x0, y0, x1, y1 = [float(v) for v in r["rect"]]
                x0, x1 = sorted((max(0, min(W, x0)), max(0, min(W, x1))))
                y0, y1 = sorted((max(0, min(H, y0)), max(0, min(H, y1))))
                if x1 - x0 < 5 or y1 - y0 < 5:
                    continue
                kind = r.get("kind") if r.get("kind") in KINDS else "其他"
                rid = str(r.get("id") or f"R{k + 1}")[:12]
                nr = {"id": rid, "rect": [int(x0), int(y0), int(x1), int(y1)],
                      "kind": kind, "questions": [norm_number(x) for x in (r.get("questions") or []) if norm_number(x)],
                      "full_score": _num(r.get("full_score")), "header": str(r.get("header") or "")[:60]}
                o = old_by_id.get(rid)
                same = o is not None and o.get("rect") == nr["rect"] and o.get("kind") == kind and not r.get("redetect")
                if same and o.get("id_grid"):
                    nr["id_grid"] = o["id_grid"]          # 生成的答题纸上涂学号的格子
                if same and kind in ("填空题", "选择题") and (o.get("slots") or o.get("groups")) and \
                        all(x.get("src") == "generated" for x in (o.get("slots") or o.get("groups"))):
                    for key in ("slots", "groups"):        # 生成的格子位置是准的：只接受改题号，位置和基准暗度用原来的
                        if o.get(key) and isinstance(r.get(key), list) and len(r[key]) == len(o[key]):
                            r[key] = [{**ox, "q": x.get("q")} for x, ox in zip(r[key], o[key])]
                if kind == "填空题" and same and isinstance(r.get("slots"), list):
                    nr["slots"] = [{**sl, "q": norm_number(sl.get("q")), "src": "teacher" if norm_number(sl.get("q")) != norm_number(osl.get("q")) else osl.get("src")}
                                   for sl, osl in zip(r["slots"], o.get("slots") or r["slots"])]
                elif kind == "选择题" and same and isinstance(r.get("groups"), list):
                    nr["groups"] = [{**g, "q": norm_number(g.get("q")), "src": "teacher" if norm_number(g.get("q")) != norm_number(og.get("q")) else og.get("src")}
                                    for g, og in zip(r["groups"], o.get("groups") or r["groups"])]
                elif kind in ("填空题", "选择题"):
                    changed.setdefault(pg["page"], set()).add(rid)
                if kind in ("填空题", "选择题") and (nr.get("slots") or nr.get("groups")):
                    found = [x["q"] for x in (nr.get("slots") or nr.get("groups")) if x.get("q")]
                    nr["questions"] = sorted(set(found), key=_numkey) if found else nr["questions"]
                regions.append(nr)
            pg["regions"] = regions
        tmpl["labeled_by"] = "teacher"
        e["warnings"] = reconcile(e)
    exam = store.update_exam(exam_id, apply)
    if changed:
        tmpl = exam["template"]
        for pg in tmpl["pages"]:
            if pg["page"] in changed:
                img = cv2.imread(str(d / pg["image"]))
                if img is not None:
                    attach_objective(pg, img, tmpl.get("source", "blank"), only=changed[pg["page"]])
        exam = store.update_exam(exam_id, lambda e: (e.__setitem__("template", tmpl), e.__setitem__("warnings", reconcile(e))))
    render_previews(store, exam_id, exam["template"])
    store.update_exam(exam_id, lambda e: e.__setitem__("template", exam["template"]))
    return store.load_exam(exam_id)
