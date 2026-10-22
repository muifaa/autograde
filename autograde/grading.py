"""单道解答题：手写转写 → 按评分细则给分 → 校验 → （可选）多次独立评分取均值。"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import prompts as P
from .exam_setup import _num, norm_label
from .imaging import to_data_url
from .llm import DeepSeek, image_part, text_part
from .store import model_supports_images

_LEG_RE = re.compile(r"\[\[\s*清晰度\s*[:：]\s*(高|中|低)\s*\]\]")


def split_legibility(text: str) -> Tuple[str, str]:
    m = _LEG_RE.search(text or "")
    leg = m.group(1) if m else ""
    return _LEG_RE.sub("", text or "").strip(), leg


def transcribe(llm: DeepSeek, st: dict, q: dict, crops: List[np.ndarray]) -> dict:
    think = bool(st.get("transcribe_thinking"))
    content = [image_part(to_data_url(c, int(st.get("crop_max_side", 2000)))) for c in crops]
    content.append(text_part(P.transcribe_user(q, len(crops))))
    res = llm.chat([{"role": "system", "content": P.TRANSCRIBE_SYSTEM}, {"role": "user", "content": content}],
                   st["vision_model"], thinking=think, effort="low" if think else None,
                   temperature=0.0, max_tokens=8000)
    text, leg = split_legibility(res["content"])
    return {"text": text, "legibility": leg, "cached": res.get("cached", False)}


def round_step(x: float, step: float) -> float:
    step = float(step or 1)
    v = math.floor(x / step + 0.5 - 1e-9) * step
    return int(v) if float(v).is_integer() else round(v, 2)


def part_defs(q: dict) -> List[dict]:
    parts = [p for p in (q.get("parts") or []) if p.get("score") is not None]
    if parts and abs(sum(float(p["score"]) for p in parts) - float(q["score"])) < 1e-6:
        return [{"label": p["label"], "score": float(p["score"])} for p in parts]
    return [{"label": "(全题)", "score": float(q["score"])}]


def validate_grade(obj: dict, q: dict, step: float = 1.0) -> dict:
    """把模型输出整理成可靠的结构：分数取整、不超满分、总分=小问之和。"""
    defs = part_defs(q)
    raw = [p for p in (obj.get("parts") or []) if isinstance(p, dict)] if isinstance(obj, dict) else []
    bylabel = {norm_label(p.get("label"), k): p for k, p in enumerate(raw)}
    problems: List[str] = []
    parts = []
    for k, d in enumerate(defs):
        mp = bylabel.get(d["label"])
        if mp is None and len(raw) == len(defs):
            mp = raw[k]
        if mp is None and len(defs) == 1 and raw:
            # 只有一个小问但模型拆开写了：求和
            mp = {"score": sum(_num(p.get("score")) or 0 for p in raw), "points": sum([p.get("points") or [] for p in raw], []),
                  "comment": "；".join(str(p.get("comment") or "") for p in raw)}
        mx = d["score"]
        if mp is None:
            problems.append(f"模型没有给出{d['label']}的分数")
            score, points, comment = 0, [], ""
        else:
            s = _num(mp.get("score"))
            if s is None:
                problems.append(f"{d['label']}的分数无法识别")
                s = 0
            if s > mx + 1e-6 or s < 0:
                problems.append(f"{d['label']}给分 {s} 超出范围，已截断")
            score = round_step(min(max(float(s), 0.0), mx), step)
            points = []
            for pt in mp.get("points") or []:
                if isinstance(pt, dict):
                    points.append({"item": str(pt.get("item") or "")[:200], "max": _num(pt.get("max")),
                                   "got": _num(pt.get("got")), "reason": str(pt.get("reason") or "")[:300]})
            comment = str(mp.get("comment") or "")[:500]
        parts.append({"label": d["label"], "max": mx, "score": score, "points": points, "comment": comment})
    total = round_step(sum(p["score"] for p in parts), step)
    mt = _num(obj.get("total")) if isinstance(obj, dict) else None
    if mt is not None and abs(float(mt) - total) > 1e-6:
        problems.append(f"模型写的总分 {mt} 与小问之和 {total} 不一致，按小问之和")
    conf = str(obj.get("confidence") or "").lower() if isinstance(obj, dict) else ""
    conf = conf if conf in ("high", "medium", "low") else "medium"
    needs = bool(obj.get("needs_review")) if isinstance(obj, dict) else True
    notes = [str(obj.get("review_notes") or "")] if isinstance(obj, dict) else []
    if conf == "low":
        needs = True
    if problems:
        needs = True
    return {"score": total, "max": float(q["score"]), "parts": parts,
            "comment": str(obj.get("comment") or "")[:500] if isinstance(obj, dict) else "",
            "deductions": [str(x)[:200] for x in (obj.get("deductions") or [])] if isinstance(obj, dict) else [],
            "confidence": conf, "needs_review": needs,
            "review_notes": "；".join(x for x in notes + problems if x)}


def grade_once(llm: DeepSeek, st: dict, q: dict, transcription: str, legibility: str,
               crops: List[np.ndarray], sample: int = 0, use_cache: bool = True,
               figure_imgs: Optional[List[np.ndarray]] = None) -> dict:
    model = st["grading_model"]
    vision = model_supports_images(model)
    with_img = bool(st.get("grading_attach_image", True)) and vision and bool(crops)
    step = float(st.get("score_step", 1))
    student = P.grade_student_block(q, transcription, legibility, with_img, step)
    if sample:
        student += f"\n（独立评阅 #{sample + 1}）"
    content = [text_part(P.grade_question_block(q))]
    if figure_imgs and vision:   # 有配图的题、题干没录入的题：附上试卷原题截图
        cap = "【试卷原题截图（看配图用）】" if (q.get("stem") or "").strip() else "【试卷原题截图（题目文字没有录入，以图为准）】"
        content += [text_part(cap)] + [image_part(to_data_url(c, 1600)) for c in figure_imgs[:2]]
    content.append(text_part(student))
    if with_img:
        content += [image_part(to_data_url(c, int(st.get("crop_max_side", 2000)))) for c in crops]
    obj, res = llm.chat_json([{"role": "system", "content": P.GRADE_SYSTEM}, {"role": "user", "content": content}],
                             model, thinking=True, effort=st.get("grading_effort", "high"), max_tokens=32000,
                             use_cache=use_cache)
    out = validate_grade(obj, q, step)
    out["model"] = res.get("model") or model
    return out


VERDICT_SCORE = {"对": 1.0, "半对": 0.5, "错": 0.0}


def _verdict(v) -> Optional[str]:
    v = str(v or "").strip()
    if v in VERDICT_SCORE:
        return v
    if v in ("正确", "全对", "√", "✓"):
        return "对"
    if v in ("部分正确", "部分对", "半", "½"):
        return "半对"
    if v in ("错误", "不对", "×", "✗", "未作答"):
        return "错"
    return None


def grade_homework(llm: DeepSeek, st: dict, q: dict, transcription: str, legibility: str, crops: List[np.ndarray],
                   use_cache: bool = True, figure_imgs: Optional[List[np.ndarray]] = None) -> dict:
    """平时作业：不打分，只判“对/半对/错”，指出错在哪。结果按 1 分制存（对 1、半对 0.5、错 0），和打分的题同样结构。"""
    model = st["grading_model"]
    vision = model_supports_images(model)
    with_img = bool(st.get("grading_attach_image", True)) and vision and bool(crops)
    content = [text_part(P.hw_question_block(q))]
    if figure_imgs and vision:
        content += [text_part("【试卷原题截图】")] + [image_part(to_data_url(c, 1600)) for c in figure_imgs[:2]]
    content.append(text_part(P.hw_student_block(q, transcription, legibility, with_img)))
    if with_img:
        content += [image_part(to_data_url(c, int(st.get("crop_max_side", 2000)))) for c in crops]
    obj, res = llm.chat_json([{"role": "system", "content": P.HW_SYSTEM}, {"role": "user", "content": content}],
                             model, thinking=True, effort=st.get("grading_effort", "high"), max_tokens=16000,
                             use_cache=use_cache)
    obj = obj if isinstance(obj, dict) else {}
    problems = []
    verdict = _verdict(obj.get("verdict"))
    parts = []
    for k, p in enumerate(obj.get("parts") or []):
        if not isinstance(p, dict):
            continue
        pv = _verdict(p.get("verdict"))
        parts.append({"label": norm_label(p.get("label"), k) if p.get("label") not in (None, "", "(全题)") else "(全题)",
                      "verdict": pv or "", "score": VERDICT_SCORE.get(pv), "max": 1.0, "points": [],
                      "comment": str(p.get("comment") or "")[:300]})
    if verdict is None:                      # 没给总判断：按小问推
        vs = [p["verdict"] for p in parts if p["verdict"]]
        verdict = "对" if vs and all(v == "对" for v in vs) else ("错" if vs and all(v == "错" for v in vs) else ("半对" if vs else None))
        if verdict is None:
            problems.append("模型没有给出对错判断")
            verdict = "错"
    conf = str(obj.get("confidence") or "").lower()
    conf = conf if conf in ("high", "medium", "low") else "medium"
    needs = bool(obj.get("needs_review")) or conf == "low" or bool(problems)
    if legibility == "低":
        needs = True
        problems.append("字迹识别清晰度低")
    err = str(obj.get("error") or "").strip()[:300]
    return {"score": VERDICT_SCORE[verdict], "max": 1.0, "verdict": verdict, "parts": parts if len(parts) > 1 else [],
            "error": err, "comment": str(obj.get("comment") or "")[:300], "deductions": [err] if err else [],
            "confidence": conf, "needs_review": needs, "mode": "homework",
            "review_notes": "；".join(x for x in [str(obj.get("review_notes") or "")] + problems if x),
            "samples": [VERDICT_SCORE[verdict]], "model": res.get("model") or model}


def grade(llm: DeepSeek, st: dict, q: dict, transcription: str, legibility: str, crops: List[np.ndarray],
          use_cache: bool = True, figure_imgs: Optional[List[np.ndarray]] = None) -> dict:
    """按设置做 1 次或多次独立评分。多次时各小问取平均再按给分单位取整，分差过大标记复核。"""
    n = max(1, int(st.get("samples", 1)))
    step = float(st.get("score_step", 1))
    samples = [grade_once(llm, st, q, transcription, legibility, crops, k, use_cache, figure_imgs) for k in range(n)]
    if legibility == "低":
        for s in samples:
            s["needs_review"] = True
            s["review_notes"] = "；".join(x for x in (s["review_notes"], "字迹识别清晰度低") if x)
    if n == 1:
        r = dict(samples[0])
        r["samples"] = [samples[0]["score"]]
        return r
    totals = [s["score"] for s in samples]
    mean_total = sum(totals) / n
    ref = min(samples, key=lambda s: abs(s["score"] - mean_total))
    parts = []
    for k, p in enumerate(ref["parts"]):
        avg = sum(s["parts"][k]["score"] for s in samples) / n
        parts.append(dict(p, score=round_step(avg, step)))
    r = dict(ref)
    r["parts"] = parts
    r["score"] = round_step(sum(p["score"] for p in parts), step)
    r["samples"] = totals
    gap = max(totals) - min(totals)
    notes = [s["review_notes"] for s in samples if s["review_notes"]]
    if gap > float(st.get("review_gap", 2)):
        r["needs_review"] = True
        notes.append(f"{n} 次独立评分相差 {gap:g} 分（{'、'.join(f'{t:g}' for t in totals)}）")
    r["needs_review"] = r["needs_review"] or any(s["needs_review"] for s in samples)
    r["review_notes"] = "；".join(dict.fromkeys(notes))
    return r


# ================================================================ 客观题

def _answers_obj(obj) -> Tuple[Dict[str, str], List[str]]:
    from .exam_setup import norm_number
    ans, unclear = {}, []
    if isinstance(obj, dict):
        for k, v in (obj.get("answers") or {}).items():
            ans[norm_number(k)] = str(v if v is not None else "").strip().strip("$").strip()
        unclear = [norm_number(x) for x in (obj.get("unclear") or [])]
    return ans, unclear


def transcribe_fills(llm: DeepSeek, st: dict, items: List[Tuple[str, np.ndarray]]) -> Tuple[Dict[str, str], List[str]]:
    """填空题：把每个空的截图拼成几张图，一次转写。items: [(题号, 截图)]。"""
    from .objective import contact_sheets
    if not items:
        return {}, []
    sheets = contact_sheets(items)
    content = [image_part(to_data_url(im, 2000)) for im in sheets]
    content.append(text_part(P.fill_transcribe_user([n for n, _ in items], len(sheets))))
    obj, _ = llm.chat_json([{"role": "system", "content": P.FILL_SYSTEM}, {"role": "user", "content": content}],
                           st["vision_model"], thinking=False, temperature=0.0, max_tokens=4000)
    return _answers_obj(obj)


def read_fill_region(llm: DeepSeek, st: dict, img: np.ndarray, numbers: List[str]) -> Tuple[Dict[str, str], List[str]]:
    content = [image_part(to_data_url(img, 2400)), text_part(P.fill_region_user(numbers))]
    obj, _ = llm.chat_json([{"role": "system", "content": P.FILL_SYSTEM}, {"role": "user", "content": content}],
                           st["vision_model"], thinking=False, temperature=0.0, max_tokens=4000)
    return _answers_obj(obj)


def read_choice_region(llm: DeepSeek, st: dict, img: np.ndarray, numbers: List[str]) -> Tuple[Dict[str, str], List[str]]:
    content = [image_part(to_data_url(img, 2400)), text_part(P.choice_region_user(numbers))]
    obj, _ = llm.chat_json([{"role": "system", "content": P.CHOICE_SYSTEM}, {"role": "user", "content": content}],
                           st["vision_model"], thinking=False, temperature=0.0, max_tokens=2000)
    ans, unclear = _answers_obj(obj)
    return {k: re.sub(r"[^A-H]", "", v.upper()) for k, v in ans.items()}, unclear


def judge_fills(llm: DeepSeek, st: dict, items: List[dict]) -> Dict[str, dict]:
    """填空题判对错：items = [{"number", "question", "reference", "student", "notes"}]，一次判完。"""
    from .exam_setup import norm_number
    if not items:
        return {}
    obj, _ = llm.chat_json([{"role": "system", "content": P.JUDGE_SYSTEM},
                            {"role": "user", "content": P.fill_judge_user(items, bool(st.get("fill_strict")))}],
                           st["grading_model"], thinking=True, effort=st.get("judge_effort", "high"), max_tokens=16000)
    out = {}
    for k, v in ((obj or {}).get("results") or {}).items() if isinstance(obj, dict) else []:
        if isinstance(v, dict):
            out[norm_number(k)] = {"correct": bool(v.get("correct")), "uncertain": bool(v.get("uncertain")),
                                   "reason": str(v.get("reason") or "")[:200]}
    return out
