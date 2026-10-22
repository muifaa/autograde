"""批改流程：扫描件 → 对齐、按份分组 → 裁出解答题 → 手写转写 + 按细则给分 → 回写分数、导出成绩。"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from . import objective as O
from . import prompts as P
from . import roster as RS
from . import sheetgen
from .align import ScanPrep, align_page, crop_region, flatten, line_precision, warp_page
from .exam_setup import _load_crops, _num, is_scored, objective_places, regions_for, scoring_questions
from .grading import grade, grade_homework, judge_fills, read_choice_region, read_fill_region, transcribe, transcribe_fills
from .imaging import count_pages, imwrite, limit_long, load_page, resize_long, strip_red, to_data_url
from .imaging import iter_pages as imaging_iter_pages
from .llm import DeepSeek, LLMError, image_part, text_part
from .report import annotate_sheet, export_csv, graded_numbers, sides_text
from .store import Store

Progress = Callable[[float, str], None]


def _noop(f: float, m: str) -> None:
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def iter_pages(paths: List[Path], dpi: int) -> Iterator[Tuple[Path, int, Optional[np.ndarray], str]]:
    """所有扫描件逐页产出 (文件, 页序号, 图像或 None, 读不出来的原因)。一次只有一页在内存里。"""
    for p in paths:
        for i, img, err in imaging_iter_pages(p, dpi=dpi):
            yield p, i, img, err


def decode_qr(img: np.ndarray) -> List[str]:
    try:
        small = limit_long(img, 3000)
        ok, infos, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(small)
        return [s for s in (infos or []) if s] if ok else []
    except Exception:
        return []


def rubric_hash(q: dict) -> str:
    keys = ("stem", "figure", "parts", "answer", "rubric", "notes", "score")
    return hashlib.sha1(json.dumps({k: q.get(k) for k in keys}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:10]


# ---------------------------------------------------------------- 第一步：对齐和裁切

def template_hash(tmpl: Optional[dict]) -> str:
    if not tmpl:
        return ""
    key = [(pg["page"], pg["size"], pg.get("markers"),
            [(r["id"], r["rect"], r.get("kind"), r.get("questions")) for r in pg["regions"]]) for pg in tmpl["pages"]]
    return hashlib.sha1(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:10]


def _load_refs(d: Path, tmpl: dict) -> dict:
    return {pg["page"]: (cv2.imread(str(d / pg["image"]), cv2.IMREAD_GRAYSCALE),
                         cv2.imread(str(d / pg["lines"]), cv2.IMREAD_GRAYSCALE)) for pg in tmpl["pages"]}


SURE_QUALITY = 0.85   # 定位点对齐后模板线条压中这么多，就不用再跑慢的图像特征匹配了


def _best_alignment(img: np.ndarray, tmpl: dict, refs: dict, only_page: Optional[int] = None):
    """和答题纸的每一页比，取对得最好的一页。先只用定位点（快）；都不够好时再对定位点不行的页做 SIFT 兜底。"""
    pages = [pg for pg in tmpl["pages"] if only_page is None or pg["page"] == only_page]
    prep = ScanPrep(img)
    first = {}
    for pg in pages:
        ref, lines = refs[pg["page"]]
        try:
            first[pg["page"]] = align_page(img, pg, ref, lines, prep=prep, features=False)
        except Exception:
            first[pg["page"]] = None
    results = dict(first)
    top = max((a for a in first.values() if a is not None), key=lambda a: a.quality, default=None)
    if top is None or top.quality < SURE_QUALITY or top.matched < 4:
        for pg in pages:
            a = first[pg["page"]]
            if a is not None and a.quality >= 0.6 and a.matched >= 4:
                continue                      # 这一页定位点已经对上了，SIFT 也不会更好
            ref, lines = refs[pg["page"]]
            try:
                results[pg["page"]] = align_page(img, pg, ref, lines, prep=prep, features=True)
            except Exception:
                pass
    best, best_pg = None, None
    for pg in pages:
        al = results.get(pg["page"])
        if al is not None and (best is None or al.quality > best.quality):
            best, best_pg = al, pg
    if best is not None and best.quality >= SURE_QUALITY:
        return best, best_pg
    return _flatten_choice(img, prep, first, results, pages, refs, best, best_pg)


def _f1(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def _flatten_choice(img, prep, first: dict, results: dict, pages: list, refs: dict, best, best_pg):
    """慢路径（手机拍的照片、纸面不平、透视明显）：把有希望的一两页“拉平”再比。
    挑页不只看模板线条压中了多少，还看照片上的线条有多少能被这一页解释——正反面的框有不少重合，
    拉平之后对错了页也能压中一大半线条。拉平的起点用定位点的对齐，不行再试图像特征的对齐
    （特征点在纸面弯曲时可能整体偏出一格，从那里拉平会对到相邻的横线上）。"""
    order = sorted([pg for pg in pages if results.get(pg["page"]) is not None and results[pg["page"]].quality >= 0.12],
                   key=lambda pg: -results[pg["page"]].quality)[:2]
    cands = []
    for pg in order:
        for al in (first.get(pg["page"]), results[pg["page"]]):
            if al is not None and al.quality >= 0.08 and not any(al is c for c, _ in cands):
                cands.append((al, pg))
    scored = []
    for al, pg in cands:
        if scored and max(s[0] for s in scored) >= 0.9:
            break                                      # 已经对得很好了：不用再试别的页、别的起点
        lines = refs[pg["page"]][1]
        drop_red = pg.get("print_color", "black") == "black"
        g, f = prep.gray(drop_red)                     # 检测尺度的灰度图，H 换算过去
        al.precision = line_precision(g, np.diag([f, f, 1.0]) @ al.H, lines)
        scored.append((_f1(al.quality, al.precision), al, pg))
        try:
            fl = flatten(img, al, pg, lines, prep=prep)
        except Exception:                              # 拉平失败就用原来的对齐
            traceback.print_exc()
            fl = None
        if fl is not None:
            fl.precision = line_precision(fl.gray, fl.H, lines)
            if fl.precision >= 0.6:                    # 照片上多出来的线解释不了：多半是对错了页
                scored.append((_f1(fl.quality, fl.precision), fl, pg))
    if not scored:
        return best, best_pg
    _, al, pg = max(scored, key=lambda s: s[0])
    return al, pg


def _process_page(img, best, pg, tmpl, nums, sdir: Path, sid: str, st: dict, page_rec: dict,
                  qbyn: Optional[dict] = None, tmpl_gray: Optional[np.ndarray] = None):
    """对齐成功的一页：存预览图、学生信息图，按模板裁出解答题；选择题识别涂卡，填空题逐空截图。
    返回 ({解答题题号: [裁切图路径]}, {客观题题号: 记录})。"""
    p = pg["page"]
    size = pg["size"]
    img = best.source(img)                      # 手机照片：用拉平后的整页图（H 也是对它的）
    warped = warp_page(img, best.H, size, 1.0)
    imwrite(sdir / f"page_{p}.jpg", warped, quality=85)
    page_rec.update(display=f"work/{sid}/page_{p}.jpg", display_scale=1.0,
                    align={"method": best.method, "quality": round(best.quality, 3), "matched": best.matched,
                           "total": best.total, "notes": best.notes})
    if p == 0:
        imwrite(sdir / "student_page.jpg", limit_long(warp_page(img, best.H, size, 1.4), 2800), quality=85)
    Lt = max(size)
    drop_red = pg.get("print_color", "black") == "black"
    out: Dict[str, List[str]] = {}
    for n in nums:
        for rp, r in regions_for(tmpl, n):
            if rp != p:
                continue
            crop = crop_region(img, best.H, r["rect"], pad=0.004 * Lt, max_side=int(st.get("crop_max_side", 2000)),
                               drop_red=drop_red)
            fn = f"q{n}_p{p}_{r['id']}.jpg"
            imwrite(sdir / fn, crop, quality=90)
            out.setdefault(n, []).append(f"work/{sid}/{fn}")
    objs: Dict[str, dict] = {}
    id_regions = [r for r in pg["regions"] if r.get("id_grid")]
    if qbyn is not None or id_regions:
        wgray = cv2.cvtColor(strip_red(warped) if drop_red else warped, cv2.COLOR_BGR2GRAY)
        if qbyn is not None:
            objs = _objective_on_page(img, best, pg, tmpl, [n for n in nums if n in qbyn], qbyn, wgray, tmpl_gray,
                                      sdir, sid, st, drop_red)
        for r in id_regions:                      # 涂的学号（生成的答题纸才有）
            got = sheetgen.read_student_id(wgray, r)
            if got is not None:
                page_rec["id_read"] = got
                break
    return out, objs


def _objective_on_page(img, best, pg, tmpl, nums, qbyn, wgray, tmpl_gray, sdir: Path, sid: str, st: dict,
                       drop_red: bool) -> Dict[str, dict]:
    """一页上的选择题、填空题：选择题按涂黑程度识别（不调用模型），填空题逐空截图并判断有没有写。"""
    p = pg["page"]
    Lt = max(pg["size"])
    places = objective_places(tmpl)
    regions = {r["id"]: r for r in pg["regions"]}
    region_crops: Dict[str, str] = {}
    papers: Dict[str, float] = {}
    out: Dict[str, dict] = {}
    for n in nums:
        pl = places.get(n)
        q = qbyn[n]
        if not pl or pl["page"] != p or q.get("type") not in ("填空题", "选择题"):
            continue
        rec = {"number": n, "type": pl["kind"], "max": q.get("score"), "crops": [], "status": "pending"}
        if pl.get("whole"):
            rid = pl["region"]
            if rid not in region_crops:
                crop = crop_region(img, best.H, regions[rid]["rect"], pad=0.004 * Lt, max_side=2400, drop_red=drop_red)
                imwrite(sdir / f"obj_{rid}_p{p}.jpg", crop, quality=90)
                region_crops[rid] = f"work/{sid}/obj_{rid}_p{p}.jpg"
            rec.update(crops=[region_crops[rid]], whole=True)
        elif pl["kind"] == "填空题":
            sl = pl["slot"]
            x0, y0, x1, y1 = sl["rect"]
            # 往上多裁一点：竖着写的分数（分子在横线上方很高）常常顶出格子，又不碰到上一行的横线
            crop = crop_region(img, best.H, [x0, y0 - 0.22 * (y1 - y0), x1, y1], pad=3, max_side=1600, drop_red=drop_red)
            imwrite(sdir / f"fill_{n}.jpg", crop, quality=90)
            ink = O.ink_added(wgray, tmpl_gray, sl["rect"]) if tmpl_gray is not None else 1.0
            rec.update(crops=[f"work/{sid}/fill_{n}.jpg"], ink=round(ink, 4))
        else:
            g = pl["group"]
            rid = pl["region"]
            if rid not in papers:
                papers[rid] = O.paper_level(wgray, regions[rid]["rect"])
            base = [o.get("base") for o in g["options"]]
            base = base if tmpl.get("source") == "blank" and all(b is not None for b in base) else None
            res = O.read_choice(wgray, g, papers[rid], base)
            crop = crop_region(img, best.H, g["rect"], pad=6, max_side=1400, drop_red=drop_red)
            imwrite(sdir / f"choice_{n}.jpg", crop, quality=90)
            rec.update(crops=[f"work/{sid}/choice_{n}.jpg"], omr=res, student=res["picked"])
            score_choice_rec(rec, q, st)
        out[n] = rec
    return out


def score_choice_rec(rec: dict, q: dict, st: dict) -> None:
    """选择题按识别（或老师更正）的选项给分。"""
    full = float(q.get("score") or 0)
    mp = "proportional" if q.get("unscored") else str(st.get("multi_partial", "proportional"))   # 作业：没选全就是“半对”
    sc, why = O.score_choice(rec.get("student") or "", q.get("answer") or "", full, mp)
    if q.get("unscored") and 0 < sc < full - 1e-9:
        sc = full / 2
    flags = [] if rec.get("student_edited") else list((rec.get("omr") or {}).get("flags") or [])
    if why == "多涂" and not rec.get("student_edited"):
        flags.append("涂了不止一个选项")
    rec["ai"] = {"score": _num(round(sc, 2)), "max": q.get("score"), "correct": full > 0 and sc >= full - 1e-9,
                 "reason": why, "method": "teacher_fix" if rec.get("student_edited") else "omr",
                 "needs_review": bool(flags), "review_notes": "；".join(flags)}
    rec["status"] = "done"


def _failed_page(img, best, p: int, sdir: Path, sid: str, page_rec: dict) -> None:
    imwrite(sdir / f"page_{p}.jpg", resize_long(img, 2000)[0], quality=85)
    page_rec.update(display=f"work/{sid}/page_{p}.jpg", display_scale=None,
                    align={"method": best.method if best else "none",
                           "quality": round(best.quality, 3) if best else 0.0, "notes": ["对齐失败"]})


def _page_qr(img: np.ndarray, best, best_pg: Optional[dict]) -> Optional[dict]:
    """生成的答题纸每页右上角的二维码：对齐了就按模板位置裁下来读（快）；没对齐或没读出来再整页找。"""
    if best is not None and best_pg and best_pg.get("qr"):
        got = sheetgen.read_page_qr(best.source(img), best.H, best_pg["qr"]["rect"])
        if got:
            return got
    try:   # 没对齐：整页找。手机照片上的二维码小、还斜着，先用原图大小试（不超过 4500 像素）
        for side in (4500, 2400):
            got = sheetgen.parse_qr(sheetgen.decode_qr_image(limit_long(img, side)))
            if got or max(img.shape[:2]) <= side:
                return got
    except Exception:
        return None
    return None


def _exam_title(store: Store, exam_id: str) -> str:
    try:
        return f"《{store.load_exam(exam_id)['title']}》"
    except Exception:
        return "别的考试"


BLANK_DARK = 0.0015   # 对不上答题纸、深色像素又不到 0.15% 的页，当作空白页跳过


def looks_blank(img: np.ndarray) -> bool:
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    paper = float(np.percentile(g, 90))
    return float((g < paper - 70).mean()) < BLANK_DARK


def _missing_pages(sheet: dict, n_pages: int) -> List[int]:
    have = {int(pr.get("template_page", -1)) for pr in sheet.get("pages") or []}
    return [p for p in range(n_pages) if p not in have]


def _last_seq(d: Path, exam: dict) -> int:
    """已用过的最大答卷编号（删掉、合并掉的答卷编号也不会再用）。"""
    used = [int(exam.get("sheet_seq") or 0)]
    for name in list(exam.get("sheets") or []) + [p.name for p in (d / "work").glob("S*")]:
        m = re.fullmatch(r"S(\d+)", name)
        if m:
            used.append(int(m.group(1)))
    return max(used)


def _new_sheet(seq: int, thash: str) -> dict:
    return {"id": f"S{seq:03d}", "created_at": _now(), "pages": [], "files": [], "questions": {},
            "student": {}, "align_ok": True, "annotated": [], "template_hash": thash, "missing_pages": []}


def prepare_sheets(store: Store, exam_id: str, files: List[str], st: dict, progress: Progress = _noop) -> List[str]:
    """逐页读扫描件 → 认出是答题纸的第几页 → 按顺序分成一份份答卷 → 裁切。

    一页一页地处理（几百页的 PDF 也只占一页的内存）；空白页跳过；某页出错只影响这一页。
    这次上传的情况（几页、分成几份、哪份缺页、哪页对不上）记在 exam["scan_report"] 里给老师看。"""
    exam = store.load_exam(exam_id)
    tmpl = exam.get("template")
    if not tmpl or not tmpl.get("pages"):
        raise ValueError("请先识别答题纸版面")
    nums = graded_numbers(exam)
    if not nums:
        raise ValueError("没有可批改的题：请检查题目的分值、参考答案和版面上的题号标注")
    qbyn = {q["number"]: q for q in scoring_questions(exam)}
    d = store.exam_dir(exam_id)
    refs = _load_refs(d, tmpl)
    thash = template_hash(tmpl)
    n_pages = len(tmpl["pages"])
    _check_template_files(d, tmpl)
    paths = [d / f for f in files]
    counts = [count_pages(p) for p in paths]
    total = sum(counts) or 1
    multi = len(paths) > 1

    def where(path: Path, idx: int) -> str:
        return (f"《{path.name}》" if multi else "扫描件") + f"第 {idx + 1} 页"

    seq = _last_seq(d, exam)
    new: List[dict] = []
    notes: List[str] = []
    skipped = 0
    old_version = 0
    foreign: Dict[str, List[str]] = {}            # 别的考试的答题纸：{考试 id: [哪几页]}
    gen = tmpl.get("generated")
    cur, last = None, -1
    t0 = time.time()
    for k, (path, idx, img, err) in enumerate(iter_pages(paths, int(st.get("scan_dpi", 250)))):
        eta = ""
        if k >= 5:
            left = (time.time() - t0) / k * (total - k)
            eta = f"，还要约 {max(1, round(left / 60))} 分钟" if left >= 60 else ""
        progress(0.15 * k / total, f"对齐答题纸 {k + 1}/{total}" + (f"，已分出 {len(new)} 份" if new else "") + eta)
        if img is None:
            notes.append(f"{where(path, idx)}读不出来，已跳过（{err[:80]}）")
            skipped += 1
            continue
        try:
            best, best_pg = _best_alignment(img, tmpl, refs)
        except Exception:
            traceback.print_exc()
            best, best_pg = None, None
        ok = best is not None and best.ok
        qr = None
        if gen:                                   # 生成的答题纸：每页二维码写着是哪次考试、第几页
            qr = _page_qr(img, best if ok else None, best_pg)
            if qr and qr["exam"] != exam_id:
                foreign.setdefault(qr["exam"], []).append(where(path, idx))
                skipped += 1
                continue
            if qr and 0 <= qr["page"] < n_pages and (not ok or best_pg["page"] != qr["page"]):
                b2, pg2 = _best_alignment(img, tmpl, refs, only_page=qr["page"])
                if b2 is not None and b2.ok:
                    best, best_pg, ok = b2, pg2, True
            if qr and qr.get("ver") != gen.get("ver"):
                old_version += 1
        if not ok and qr is None and looks_blank(img):
            notes.append(f"{where(path, idx)}是空白页，已跳过")
            skipped += 1
            continue
        p = best_pg["page"] if ok else (qr["page"] if qr and 0 <= qr["page"] < n_pages else (last + 1) % n_pages)
        other_student = bool(cur is not None and qr and qr.get("sid") and cur["student"].get("qr_sid")
                             and RS.norm_sid(qr["sid"]) != RS.norm_sid(cur["student"]["qr_sid"]))
        if cur is None or p <= last or other_student:
            seq += 1
            cur = _new_sheet(seq, thash)
            cur["first_page"] = where(path, idx)
            new.append(cur)
        last = p
        if qr and qr.get("sid"):
            cur["student"]["qr_sid"] = qr["sid"]
        sid = cur["id"]
        sdir = d / "work" / sid
        rel = str(path.relative_to(d))
        page_rec = {"file": rel, "index": idx, "template_page": p, "align": None}
        if ok:
            try:
                crops, objs = _process_page(img, best, best_pg, tmpl, nums, sdir, sid, st, page_rec,
                                            qbyn, refs[best_pg["page"]][0])
                for n, lst in crops.items():
                    cur["questions"].setdefault(n, {"number": n, "type": "解答题", "max": qbyn[n].get("score"),
                                                    "crops": [], "status": "pending"})["crops"].extend(lst)
                for n, rec in objs.items():
                    cur["questions"].setdefault(n, rec)
                if page_rec.get("id_read"):
                    cur["student"]["omr"] = page_rec.pop("id_read")
                if p == 0 and not gen:                # 别的答题纸上贴的二维码/条形码（生成的答题纸二维码只标考试和页码）
                    codes = decode_qr(img)
                    if codes:
                        cur["student"]["qr"] = codes
            except Exception as e:            # 个别页处理出错：这一页交给老师人工批改，其余照常
                traceback.print_exc()
                ok = False
                notes.append(f"{where(path, idx)}（{sid} 第 {p + 1} 页）处理出错：{type(e).__name__}，请人工批改这一页")
        else:
            notes.append(f"{where(path, idx)}没能和答题纸对上，按顺序归到 {sid} 的第 {p + 1} 页，请人工批改"
                         "（手机拍的照片：整张纸、四个角的黑方块都要拍进去）")
        if not ok:
            _failed_page(img, best, p, sdir, sid, page_rec)
            cur["align_ok"] = False
        cur["pages"].append(page_rec)
        cur["files"].append({"file": rel, "index": idx})
        del img
    for other, wh in foreign.items():
        title = _exam_title(store, other)
        notes.insert(0, (f"{wh[0]}是{title}的答题纸" if len(wh) == 1 else
                         f"有 {len(wh)} 页是{title}的答题纸（{'、'.join(wh[:6])}{'……' if len(wh) > 6 else ''}）")
                     + "，不是这次的，已跳过" + ("，请到那场考试里上传" if title != "别的考试" else ""))
    if old_version:
        notes.insert(0, f"有 {old_version} 页是重新生成答题纸之前打印的旧版本，版面可能对不上，请检查这些答卷的裁切图")
    idx_roster = RS.exam_index(store.root, exam)
    for s in new:                                 # 涂的学号 / 每人一份的二维码 → 对名单补姓名
        if s["student"].get("omr") or s["student"].get("qr_sid"):
            RS.resolve(s["student"], idx_roster)
    partial = 0
    holes = {}                                    # 以前上传的、缺页的答卷：补扫的页可以并进去
    for sid0 in exam.get("sheets") or []:
        try:
            mp = store.load_sheet(exam_id, sid0).get("missing_pages") or []
        except KeyError:
            continue
        if mp:
            holes[sid0] = set(mp)
    for s in new:
        s["missing_pages"] = _missing_pages(s, n_pages)
        if s["missing_pages"]:
            partial += 1
            have = {int(pr["template_page"]) for pr in s["pages"]}
            fits = [sid0 for sid0, mp in holes.items() if have <= mp]
            notes.append(f"{s['id']}（从{s['first_page']}开始）缺{sides_text(s['missing_pages'], n_pages)}，"
                         f"那一面上的题没有批改" + (f"。如果这是补扫的，打开 {'、'.join(fits[:5])} 把它合并进去" if fits else ""))
        for n in nums:
            if n not in s["questions"]:
                s["questions"][n] = {"number": n, "type": qbyn[n].get("type"), "max": qbyn[n].get("score"), "crops": [],
                                     "status": "align_failed" if not s["align_ok"] else "missing"}
        store.save_sheet(exam_id, s)
    if n_pages > 1 and partial >= max(3, 0.3 * len(new)):
        notes.insert(0, f"有 {partial} 份答卷只找到了一部分页面。如果扫描时是先扫完所有正面、再扫所有反面，"
                        f"请改用双面扫描（让每份答卷的正反面挨在一起）后重新上传。")
    report = {"at": _now(), "files": list(files), "pages": sum(counts), "sheets": len(new), "skipped": skipped,
              "partial": partial, "seconds": round(time.time() - t0), "notes": notes, "sheet_ids": [s["id"] for s in new]}

    def apply(e):
        e["sheets"] = list(e.get("sheets") or []) + [s["id"] for s in new]
        e["files"]["scans"] = list(e["files"].get("scans") or []) + [f for f in files if f not in e["files"].get("scans", [])]
        e["scan_report"] = report
        e["sheet_seq"] = max(int(e.get("sheet_seq") or 0), seq)
    store.update_exam(exam_id, apply)
    return [s["id"] for s in new]


def recrop_sheet(store: Store, exam_id: str, sid: str, st: dict) -> bool:
    """版面改过之后：从原始扫描件重新对齐、裁切这份答卷。老师改过的分数保留，转写会重新做。"""
    exam = store.load_exam(exam_id)
    tmpl = exam["template"]
    nums = graded_numbers(exam)
    qbyn = {q["number"]: q for q in scoring_questions(exam)}
    d = store.exam_dir(exam_id)
    refs = _load_refs(d, tmpl)
    sheet = store.load_sheet(exam_id, sid)
    sdir = d / "work" / sid
    crops: Dict[str, List[str]] = {}
    objrecs: Dict[str, dict] = {}
    align_ok = True
    pages = []
    id_read = None
    for prec in sheet.get("pages") or []:
        img = load_page(d / prec["file"], int(prec.get("index", 0)), dpi=int(st.get("scan_dpi", 250)))
        want = prec.get("template_page") if tmpl.get("generated") else None   # 生成的答题纸：页码二维码已经认过
        best, best_pg = _best_alignment(img, tmpl, refs, only_page=want if want is not None and want < len(tmpl["pages"]) else None)
        if want is not None and (best is None or not best.ok):
            best, best_pg = _best_alignment(img, tmpl, refs)
        rec = {"file": prec["file"], "index": prec.get("index", 0), "align": None}
        if best is not None and best.ok:
            rec["template_page"] = best_pg["page"]
            sub, objs = _process_page(img, best, best_pg, tmpl, nums, sdir, sid, st, rec, qbyn, refs[best_pg["page"]][0])
            id_read = rec.pop("id_read", None) or id_read
            for n, lst in sub.items():
                crops.setdefault(n, []).extend(lst)
            for n, r in objs.items():
                objrecs.setdefault(n, r)
        else:
            rec["template_page"] = prec.get("template_page", 0)
            _failed_page(img, best, rec["template_page"], sdir, sid, rec)
            align_ok = False
        pages.append(rec)
        del img
    thash = template_hash(tmpl)
    idx_roster = RS.exam_index(store.root, exam) if id_read else None

    def apply(s):
        s["pages"] = pages
        s["align_ok"] = align_ok
        s["template_hash"] = thash
        s["missing_pages"] = _missing_pages(s, len(tmpl["pages"]))
        if id_read is not None:
            s.setdefault("student", {})["omr"] = id_read
            RS.resolve(s["student"], idx_roster)
        old = s.get("questions") or {}
        newq = {}
        for n in nums:
            o = old.get(n) or {}
            if n in objrecs:
                q = objrecs[n]
                if o.get("student_edited"):          # 老师更正过的识别结果保留
                    q.update(student=o.get("student"), student_edited=True)
                    if q["type"] == "选择题":
                        score_choice_rec(q, qbyn[n], st)
                    else:
                        q["status"] = "pending"
            else:
                q = {"number": n, "type": qbyn[n].get("type"), "max": qbyn[n].get("score"), "crops": crops.get(n, []),
                     "status": "pending" if crops.get(n) else ("align_failed" if not align_ok else "missing")}
                if o.get("crops") == q["crops"] and o.get("transcription") is not None:
                    q.update({k: o[k] for k in ("transcription", "legibility", "transcription_edited") if k in o})
            if o.get("teacher"):
                q["teacher"] = o["teacher"]
            newq[n] = q
        s["questions"] = newq
    store.update_sheet(exam_id, sid, apply)
    return True


# ---------------------------------------------------------------- 第二步：调用模型

def identify_student(llm: DeepSeek, st: dict, img: np.ndarray) -> dict:
    content = [image_part(to_data_url(img, 2800)), text_part(P.STUDENT_USER)]
    obj, _ = llm.chat_json([{"role": "system", "content": P.STUDENT_SYSTEM}, {"role": "user", "content": content}],
                           st["vision_model"], thinking=False, max_tokens=600)
    obj = obj if isinstance(obj, dict) else {}
    return {"name": str(obj.get("name") or "").strip()[:30], "student_id": str(obj.get("student_id") or "").strip()[:40],
            "class": str(obj.get("class") or "").strip()[:30], "source": "ai"}


def run_llm(store: Store, exam_id: str, llm: DeepSeek, st: dict, sheet_ids: List[str],
            numbers: Optional[List[str]] = None, retranscribe: bool = False, fresh: bool = False,
            progress: Progress = _noop, p0: float = 0.15, p1: float = 0.97) -> dict:
    exam = store.load_exam(exam_id)
    d = store.exam_dir(exam_id)
    qbyn = {q["number"]: q for q in scoring_questions(exam)}
    nums = [n for n in graded_numbers(exam) if numbers is None or n in numbers]
    obj_nums = [n for n in nums if qbyn[n].get("type") in ("填空题", "选择题")]
    sub_nums = [n for n in nums if n not in obj_nums]
    idx_roster = RS.exam_index(store.root, exam)
    scored = is_scored(exam)
    tasks: List[Tuple[str, str, Optional[str]]] = []
    for sid in sheet_ids:
        s = store.load_sheet(exam_id, sid)
        if (st.get("identify_student") and not (s.get("student") or {}).get("name")
                and (d / "work" / sid / "student_page.jpg").exists() and numbers is None):
            tasks.append(("student", sid, None))
        for n in sub_nums:
            qr = (s.get("questions") or {}).get(n)
            if qr and qr.get("crops"):
                tasks.append(("q", sid, n))
        if any(((s.get("questions") or {}).get(n) or {}).get("crops") for n in obj_nums):
            tasks.append(("obj", sid, None))
    stats = {"done": 0, "errors": 0, "total": len(tasks)}

    def do(task):
        kind, sid, n = task
        if kind == "obj":
            grade_objective(store, exam_id, llm, st, sid, obj_nums, retranscribe)
            return
        if kind == "student":
            img = cv2.imread(str(d / "work" / sid / "student_page.jpg"))
            info = identify_student(llm, st, img)

            def save_ai(s):
                s["student"]["ai"] = info
                RS.resolve(s["student"], idx_roster)     # 有名单：按 AI 认出的学号/姓名对上名单
            store.update_sheet(exam_id, sid, save_ai)
            return
        q = qbyn[n]
        s = store.load_sheet(exam_id, sid)
        qr = s["questions"][n]
        crops = [cv2.imread(str(d / c)) for c in qr["crops"]]
        crops = [c for c in crops if c is not None]
        text, leg = qr.get("transcription"), qr.get("legibility", "")
        if retranscribe or text is None:
            store.update_sheet(exam_id, sid, lambda s: s["questions"][n].update(status="transcribing"))
            tr = transcribe(llm, st, q, crops)
            text, leg = tr["text"], tr["legibility"]

            def save_tr(s):
                s["questions"][n].update(transcription=text, legibility=leg, transcription_edited=False)
            store.update_sheet(exam_id, sid, save_tr)
        store.update_sheet(exam_id, sid, lambda s: s["questions"][n].update(status="grading"))
        # 有配图的题、题干没录入的题：附上试卷原题截图
        fig = _load_crops(d, q.get("paper_crops")) if (q.get("figure") or not (q.get("stem") or "").strip()) else None
        judge = grade if scored else grade_homework            # 平时作业：只判对错
        g = judge(llm, st, q, text, leg, crops, use_cache=not fresh, figure_imgs=fig)

        def save_g(s):
            s["questions"][n].update(ai=g, status="done", error="", graded_at=_now(), rubric_hash=rubric_hash(q),
                                     max=q.get("score"))
        store.update_sheet(exam_id, sid, save_g)

    def safe(task):
        try:
            do(task)
            return None
        except (LLMError, Exception) as e:  # 单题失败不影响其他题
            kind, sid, n = task
            msg = str(e) if isinstance(e, LLMError) else f"{type(e).__name__}: {e}"
            if kind == "q":
                store.update_sheet(exam_id, sid, lambda s: s["questions"][n].update(status="error", error=msg[:500]))
            return msg

    workers = min(64, max(1, int(st.get("max_workers", 16))))
    last_err = ""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(safe, t) for t in tasks]
        for k, f in enumerate(as_completed(futs)):
            err = f.result()
            if err:
                stats["errors"] += 1
                last_err = err
            else:
                stats["done"] += 1
            left = (time.time() - t0) / (k + 1) * (len(tasks) - k - 1)
            eta = f"，还要约 {round(left / 60)} 分钟" if k + 1 >= min(20, len(tasks)) and left >= 90 else ""
            progress(p0 + (p1 - p0) * (k + 1) / max(1, len(tasks)),
                     f"批改中 {k + 1}/{len(tasks)}" + eta
                     + (f"（{stats['errors']} 个失败：{last_err[:60]}）" if stats["errors"] else ""))
    stats["last_error"] = last_err
    return stats


BLANK_INK = 0.0006   # 填空横线附近新增墨迹低于这个比例，认为没写（不调用模型）


def grade_objective(store: Store, exam_id: str, llm: DeepSeek, st: dict, sid: str, numbers: List[str],
                    retranscribe: bool = False) -> None:
    """一份答卷的选择题、填空题：选择题按涂卡结果给分；填空题拼图一次识别、先比对写法、不一致的再让模型判等价。"""
    exam = store.load_exam(exam_id)
    d = store.exam_dir(exam_id)
    qbyn = {q["number"]: q for q in scoring_questions(exam)}
    sheet = store.load_sheet(exam_id, sid)
    recs = {n: dict(sheet["questions"][n]) for n in numbers
            if n in (sheet.get("questions") or {}) and sheet["questions"][n].get("crops")}
    if not recs:
        return

    def img(path):
        return cv2.imread(str(d / path))

    # 选择题：整块区域（没能定位选项框）交给模型；其余按涂卡结果重新给分（参考答案可能改过）
    whole: Dict[str, List[str]] = {}
    for n, r in recs.items():
        if r["type"] == "选择题" and r.get("whole") and not r.get("student_edited") and (retranscribe or r.get("student") is None):
            whole.setdefault(r["crops"][0], []).append(n)
    for crop, ns in whole.items():
        ans, unclear = read_choice_region(llm, st, img(crop), ns)
        for n in ns:
            recs[n]["student"] = ans.get(n, "")
            recs[n]["omr"] = {"flags": ["整块识别（模型看图），请核对"] + (["模型拿不准"] if n in unclear else [])}
    for n, r in recs.items():
        if r["type"] == "选择题" and r.get("student") is not None:
            score_choice_rec(r, qbyn[n], st)

    # 填空题：识别
    items, whole_fill = [], {}
    for n, r in recs.items():
        if r["type"] != "填空题" or r.get("student_edited") or (r.get("student") is not None and not retranscribe):
            continue
        if r.get("whole"):
            whole_fill.setdefault(r["crops"][0], []).append(n)
        elif r.get("ink") is not None and r["ink"] < BLANK_INK:
            r.update(student="", unclear=False)
        else:
            im = img(r["crops"][0])
            if im is not None:
                items.append((n, im))
    if items:
        ans, unclear = transcribe_fills(llm, st, items)
        for n, _ in items:
            recs[n].update(student=ans.get(n, ""), unclear=n in unclear or "[?" in ans.get(n, ""))
    for crop, ns in whole_fill.items():
        ans, unclear = read_fill_region(llm, st, img(crop), ns)
        for n in ns:
            recs[n].update(student=ans.get(n, ""), unclear=True if n in unclear else ("[?" in ans.get(n, "")))

    # 填空题：判对错
    judge = []
    for n, r in recs.items():
        if r["type"] != "填空题" or r.get("student") is None:
            continue
        q = qbyn[n]
        ref = (q.get("answer") or "").strip()
        full = float(q.get("score") or 0)
        stu = (r.get("student") or "").strip()
        key = [ref, stu, full]
        if r.get("status") == "done" and r.get("judged") == key and r.get("ai"):
            continue                      # 答案、识别结果、分值都没变：不重判，结果保持稳定
        r["judged"] = key
        if not stu:
            r["ai"] = {"score": 0, "max": q.get("score"), "correct": False, "reason": "未作答", "method": "blank",
                       "needs_review": False, "review_notes": ""}
            r["status"] = "done"
        elif O.exact_match(stu, ref):
            r["ai"] = {"score": _num(full), "max": q.get("score"), "correct": True, "reason": "与参考答案一致", "method": "exact",
                       "needs_review": bool(r.get("unclear")), "review_notes": "识别不确定" if r.get("unclear") else ""}
            r["status"] = "done"
        else:
            judge.append({"number": n, "question": q.get("stem") or "", "reference": ref, "student": stu,
                          "notes": q.get("notes") or ""})
    if judge:
        res = judge_fills(llm, st, judge)
        for it in judge:
            n = it["number"]
            q = qbyn[n]
            full = float(q.get("score") or 0)
            j = res.get(n)
            r = recs[n]
            if j is None:
                r["ai"] = {"score": 0, "max": q.get("score"), "correct": False, "reason": "模型没有给出判断",
                           "method": "judge", "needs_review": True, "review_notes": "模型没有给出判断，请老师判"}
            else:
                notes = [x for x in (("识别不确定" if r.get("unclear") else ""), ("模型拿不准" if j["uncertain"] else "")) if x]
                r["ai"] = {"score": _num(full if j["correct"] else 0), "max": q.get("score"), "correct": j["correct"],
                           "reason": j["reason"], "method": "judge", "needs_review": bool(notes), "review_notes": "；".join(notes)}
            r["status"] = "done"

    def save(s):
        for n, r in recs.items():
            cur = s["questions"].get(n) or {}
            r["teacher"] = cur.get("teacher") or r.get("teacher") or {}
            r["graded_at"] = _now()
            s["questions"][n] = r
    store.update_sheet(exam_id, sid, save)


def _check_template_files(d: Path, tmpl: dict) -> None:
    for pg in tmpl["pages"]:
        if not (d / pg["image"]).exists():
            raise ValueError("版面图片丢失，请重新识别答题纸版面")


def set_student_answer(store: Store, exam_id: str, sheet_id: str, number: str, answer: str) -> dict:
    """老师更正识别结果：选择题直接重新给分；填空题之后由模型重新判（或老师直接判对错）。"""
    exam = store.load_exam(exam_id)
    q = next((x for x in scoring_questions(exam) if x["number"] == number), None)
    if q is None:
        raise KeyError(number)

    def apply(s):
        r = s["questions"][number]
        if r.get("type") == "选择题":
            r["student"] = re.sub(r"[^A-H]", "", (answer or "").upper())
            r["student_edited"] = True
            score_choice_rec(r, q, store.settings())
        else:
            r["student"] = (answer or "").strip()
            r["student_edited"] = True
            if O.exact_match(r["student"], q.get("answer") or "") or not r["student"]:
                ok = bool(r["student"])
                r["ai"] = {"score": _num(float(q.get("score") or 0) if ok else 0), "max": q.get("score"), "correct": ok,
                           "reason": "与参考答案一致" if ok else "未作答", "method": "exact" if ok else "blank",
                           "needs_review": False, "review_notes": ""}
                r["status"] = "done"
            else:
                r["status"] = "pending"
    s = store.update_sheet(exam_id, sheet_id, apply)
    ann = annotate_sheet(store, store.load_exam(exam_id), s)
    store.update_sheet(exam_id, sheet_id, lambda x: x.__setitem__("annotated", ann))
    export_csv(store, exam_id)
    return store.load_sheet(exam_id, sheet_id)


# ---------------------------------------------------------------- 第三步：回写、导出

def finalize(store: Store, exam_id: str, sheet_ids: Optional[List[str]] = None) -> None:
    exam = store.load_exam(exam_id)
    for sid in sheet_ids or exam.get("sheets") or []:
        try:
            s = store.load_sheet(exam_id, sid)
        except KeyError:
            continue
        ann = annotate_sheet(store, exam, s)
        store.update_sheet(exam_id, sid, lambda x: x.__setitem__("annotated", ann))
    export_csv(store, exam_id)


def grade_scans(store: Store, exam_id: str, llm: DeepSeek, st: dict, files: List[str],
                progress: Progress = _noop) -> dict:
    sids = prepare_sheets(store, exam_id, files, st, progress)
    stats = run_llm(store, exam_id, llm, st, sids, progress=progress)
    progress(0.98, "回写分数、导出成绩…")
    finalize(store, exam_id, sids)
    stats["sheets"] = len(sids)
    return stats


def regrade(store: Store, exam_id: str, llm: DeepSeek, st: dict, sheet_ids: Optional[List[str]] = None,
            numbers: Optional[List[str]] = None, retranscribe: bool = False, fresh: bool = False,
            progress: Progress = _noop) -> dict:
    exam = store.load_exam(exam_id)
    sids = sheet_ids or list(exam.get("sheets") or [])
    thash = template_hash(exam.get("template"))
    stale = [sid for sid in sids if store.load_sheet(exam_id, sid).get("template_hash") != thash]
    for k, sid in enumerate(stale):   # 版面改过：按新版面重新裁切
        progress(0.1 * k / len(stale), f"版面有改动，重新裁切 {k + 1}/{len(stale)}")
        recrop_sheet(store, exam_id, sid, st)
    stats = run_llm(store, exam_id, llm, st, sids, numbers, retranscribe, fresh, progress, 0.1 if stale else 0.0, 0.97)
    stats["recropped"] = len(stale)
    finalize(store, exam_id, sids)
    return stats


def set_teacher_score(store: Store, exam_id: str, sheet_id: str, number: str, score=None, parts=None,
                      comment: str = "", accept: bool = False, clear: bool = False) -> dict:
    """老师改分 / 确认 AI 分数 / 撤销改分。"""
    def apply(s):
        qr = s["questions"][number]
        ai = qr.get("ai") or {}
        mx = float(qr.get("max") or ai.get("max") or 0)
        if clear:
            qr["teacher"] = {}
            return
        t = {"comment": comment or "", "at": _now()}
        if accept:
            if ai.get("score") is None:
                raise ValueError("还没有 AI 分数")
            t.update(score=ai["score"], accepted=True)
        elif parts is not None:
            vals = []
            ai_parts = ai.get("parts") or []
            for k, v in enumerate(parts):
                pm = float(ai_parts[k]["max"]) if k < len(ai_parts) else mx
                v = float(v)
                if v < 0 or v > pm + 1e-6:
                    raise ValueError(f"第 {k + 1} 小问分数超出范围（0~{pm:g}）")
                vals.append(v)
            t.update(score=sum(vals), parts=vals)
        else:
            v = float(score)
            if v < 0 or v > mx + 1e-6:
                raise ValueError(f"分数超出范围（0~{mx:g}）")
            t.update(score=v)
        qr["teacher"] = t
    s = store.update_sheet(exam_id, sheet_id, apply)
    exam = store.load_exam(exam_id)
    ann = annotate_sheet(store, exam, s)
    store.update_sheet(exam_id, sheet_id, lambda x: x.__setitem__("annotated", ann))
    export_csv(store, exam_id)
    return store.load_sheet(exam_id, sheet_id)


def _resolve_notes(store: Store, exam_id: str, done: Dict[str, Tuple[str, bool]]) -> None:
    """合并、删除答卷之后，把“最近一次上传”里相关的提醒标成已处理（前面加 ✓，后面注明怎么处理的）。
    done: {答卷号: (说明, 是否只看以答卷号开头的提醒)}；同时重算还缺页的份数。"""
    exam = store.load_exam(exam_id)
    rep = exam.get("scan_report")
    if not rep:
        return
    alive = set(exam.get("sheets") or [])
    partial = 0
    for sid in rep.get("sheet_ids") or []:
        if sid in alive:
            try:
                partial += bool(store.load_sheet(exam_id, sid).get("missing_pages"))
            except KeyError:
                pass

    def apply(e):
        r = e.get("scan_report") or {}
        notes = []
        for n in r.get("notes") or []:
            for sid, (why, lead) in done.items():
                hit = n.startswith(sid + "（") if lead else re.search(rf"(?<![A-Za-z0-9]){sid}(?![0-9])", n) is not None
                if hit and not n.startswith("✓"):
                    n = f"✓ {n}（{why}）"
            notes.append(n)
        r["notes"] = notes
        if r.get("sheet_ids"):
            r["partial"] = partial
    store.update_exam(exam_id, apply)


def delete_sheet(store: Store, exam_id: str, sheet_id: str) -> None:
    """删掉一份答卷（扫错了、重复扫了）。原始扫描件不删。"""
    store.load_sheet(exam_id, sheet_id)        # 不存在就抛 KeyError
    store.update_exam(exam_id, lambda e: e.__setitem__("sheets", [s for s in e.get("sheets") or [] if s != sheet_id]))
    shutil.rmtree(store.exam_dir(exam_id) / "work" / sheet_id, ignore_errors=True)
    _resolve_notes(store, exam_id, {sheet_id: ("已删除", False)})
    export_csv(store, exam_id)


def merge_sheets(store: Store, exam_id: str, target: str, source: str) -> Tuple[dict, List[str]]:
    """把 source（比如补扫上传的缺页）并进 target。两份不能有同一页。
    补上的页上已经批好的题，识别和评分结果一起带过来；返回 (合并后的答卷, 还要批的题号)——
    一道题的作答区跨了这两份的页面（图拼起来要重批），或者补扫的那份还没批完。"""
    if target == source:
        raise ValueError("不能和自己合并")
    exam = store.load_exam(exam_id)
    n_pages = len((exam.get("template") or {}).get("pages") or [])
    d = store.exam_dir(exam_id)
    t, s = store.load_sheet(exam_id, target), store.load_sheet(exam_id, source)
    both = {int(p["template_page"]) for p in t["pages"]} & {int(p["template_page"]) for p in s["pages"]}
    if both:
        raise ValueError(f"{target} 和 {source} 都有{sides_text(sorted(both), n_pages)}，不能合并")
    tdir, sdir = d / "work" / target, d / "work" / source
    tdir.mkdir(parents=True, exist_ok=True)
    moved: Dict[str, str] = {}

    def move(rel):
        if not isinstance(rel, str) or not rel.startswith(f"work/{source}/"):
            return rel
        if rel not in moved:
            name = rel.split("/", 2)[2]
            dst = tdir / name
            if dst.exists():
                dst = tdir / f"{source}_{name}"
            if (sdir / name).exists():
                shutil.move(str(sdir / name), str(dst))
            moved[rel] = f"work/{target}/{dst.name}"
        return moved[rel]

    for pr in s["pages"]:
        pr["display"] = move(pr.get("display"))
    if (sdir / "student_page.jpg").exists() and not (tdir / "student_page.jpg").exists():
        shutil.move(str(sdir / "student_page.jpg"), str(tdir / "student_page.jpg"))

    def apply(x):
        x["pages"] = sorted(x["pages"] + s["pages"], key=lambda p: int(p["template_page"]))
        x["files"] = list(x.get("files") or []) + list(s.get("files") or [])
        x["align_ok"] = bool(x.get("align_ok", True) and s.get("align_ok", True))
        if x.get("template_hash") != s.get("template_hash"):
            x["template_hash"] = ""                    # 有一份是按旧版面裁的：下次重新给分时整份重裁
        stu = x.setdefault("student", {})
        for k, v in (s.get("student") or {}).items():
            if v and not stu.get(k):
                stu[k] = v
        for n, sq in (s.get("questions") or {}).items():
            sq = dict(sq, crops=[move(c) for c in sq.get("crops") or []])
            tq = (x.get("questions") or {}).get(n)
            if not sq["crops"]:
                continue
            if not tq or not tq.get("crops"):
                if (tq or {}).get("teacher") and not sq.get("teacher"):
                    sq["teacher"] = tq["teacher"]      # 老师之前给过分的，仍以老师为准（可以撤销）
                x["questions"][n] = sq                 # 这道题整个在补上的页上：识别和评分结果一起带过来
            else:                                      # 一道题跨了两份的页面：图按页序拼起来，重新批
                def page_of(c):
                    m = re.search(r"_p(\d+)_", c)
                    return int(m.group(1)) if m else 0
                x["questions"][n] = dict({k: v for k, v in tq.items() if k not in ("transcription", "ai", "legibility")},
                                         crops=sorted(tq["crops"] + sq["crops"], key=page_of), status="pending")
        x["missing_pages"] = _missing_pages(x, n_pages)
        x["merged_from"] = list(x.get("merged_from") or []) + [source]
    merged = store.update_sheet(exam_id, target, apply)
    regrade_nums = [n for n, q in merged["questions"].items() if q.get("crops") and q.get("status") in ("pending", "error")]
    store.update_exam(exam_id, lambda e: e.__setitem__("sheets", [i for i in e.get("sheets") or [] if i != source]))
    shutil.rmtree(sdir, ignore_errors=True)
    left = merged.get("missing_pages") or []
    _resolve_notes(store, exam_id, {
        source: (f"已并进 {target}", True),
        target: (f"已把 {source} 并进来" + (f"，还缺{sides_text(left, n_pages)}" if left else "，已补齐"), True)})
    ann = annotate_sheet(store, store.load_exam(exam_id), merged)
    merged = store.update_sheet(exam_id, target, lambda x: x.__setitem__("annotated", ann))
    export_csv(store, exam_id)
    return merged, regrade_nums


def resolve_students(store: Store, exam_id: str) -> int:
    """名单改了、或者换了对照的班级：把每份答卷的学生信息重新对一遍名单（不调用模型；老师手动改过的不动）。"""
    exam = store.load_exam(exam_id)
    idx = RS.exam_index(store.root, exam)
    n = 0
    for sid in exam.get("sheets") or []:
        def apply(s):
            stu = s.setdefault("student", {})
            if stu.get("source") != "teacher" and (stu.get("omr") or stu.get("qr_sid") or stu.get("ai")
                                                   or stu.get("name") or stu.get("student_id")):
                RS.resolve(stu, idx)
        try:
            store.update_sheet(exam_id, sid, apply)
            n += 1
        except KeyError:
            continue
    export_csv(store, exam_id)
    return n


def set_transcription(store: Store, exam_id: str, sheet_id: str, number: str, text: str) -> dict:
    def apply(s):
        s["questions"][number].update(transcription=text, transcription_edited=True)
    return store.update_sheet(exam_id, sheet_id, apply)
