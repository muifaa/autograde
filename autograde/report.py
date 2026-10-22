"""成绩汇总、CSV 导出、在答题纸上回写分数。"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from . import roster as RS
from .exam_setup import _numkey, is_scored, objective_places, regions_for, scoring_questions
from .imaging import draw_texts, imwrite
from .store import Store


def final_score(qr: dict) -> Optional[float]:
    t = (qr.get("teacher") or {}).get("score")
    if t is not None:
        return t
    ai = qr.get("ai") or {}
    return ai.get("score")


def question_state(qr: dict) -> str:
    if (qr.get("teacher") or {}).get("score") is not None:
        return "teacher"
    if qr.get("status") == "error":
        return "error"
    if qr.get("status") != "done":
        return qr.get("status") or "pending"
    return "review" if (qr.get("ai") or {}).get("needs_review") else "ok"


def sheet_summary(sheet: dict, numbers: List[str]) -> dict:
    row = {"id": sheet["id"], "student": sheet.get("student") or {}, "files": sheet.get("files") or [],
           "align_ok": sheet.get("align_ok", True), "missing_pages": sheet.get("missing_pages") or [],
           "present_pages": sorted({int(p.get("template_page", 0)) for p in sheet.get("pages") or []}),
           "first_page": sheet.get("first_page") or "",
           "scores": {}, "states": {}, "total": 0.0, "max_total": 0.0}
    review = []
    for n in numbers:
        qr = (sheet.get("questions") or {}).get(n)
        if not qr:
            row["states"][n] = "missing"
            continue
        s = final_score(qr)
        row["scores"][n] = s
        st = question_state(qr)
        row["states"][n] = st
        if st in ("review", "error"):
            review.append(n)
        if s is not None:
            row["total"] += float(s)
        mx = (qr.get("ai") or {}).get("max") or qr.get("max") or 0
        row["max_total"] += float(mx)
    row["needs_review"] = review
    return row


SECTION_ORDER = ["选择题", "填空题", "解答题"]


def graded_numbers(exam: dict) -> List[str]:
    """要批改的题：解答题要有作答区域；选择/填空题要有参考答案和作答位置；都要有分值。"""
    tmpl = exam.get("template")
    places = objective_places(tmpl)
    nums = []
    for q in scoring_questions(exam):
        t, n = q.get("type"), q["number"]
        if not q.get("score"):
            continue
        if t == "解答题" and regions_for(tmpl, n):
            nums.append(n)
        elif t in ("填空题", "选择题") and (q.get("answer") or "").strip() and n in places:
            nums.append(n)
    return sorted(nums, key=_numkey)


def results(store: Store, exam_id: str) -> dict:
    exam = store.load_exam(exam_id)
    nums = graded_numbers(exam)
    qbyn = {q["number"]: q for q in scoring_questions(exam)}
    types = {n: qbyn[n].get("type") for n in nums}
    rows = [sheet_summary(s, nums) for s in store.list_sheets(exam_id)]
    n_sides = len((exam.get("template") or {}).get("pages") or []) or 1
    for r in rows:
        r["n_sides"] = n_sides
        r["missing_text"] = sides_text(r["missing_pages"], n_sides) if r["missing_pages"] else ""
        r["present_text"] = sides_text(r["present_pages"], n_sides)
    sections = [t for t in SECTION_ORDER if t in types.values()]
    for r in rows:
        r["sections"] = {t: sum(float(r["scores"].get(n) or 0) for n in nums if types[n] == t) for t in sections}
    stats = {}
    for n in nums:
        vals = [r["scores"].get(n) for r in rows if r["scores"].get(n) is not None]
        q = qbyn.get(n, {})
        full = float(q.get("score") or 0)
        stats[n] = {"count": len(vals), "avg": round(sum(vals) / len(vals), 2) if vals else None,
                    "max": q.get("score"), "review": sum(1 for r in rows if r["states"].get(n) in ("review", "error")),
                    "right": sum(1 for v in vals if full and v >= full - 1e-9),
                    "half": sum(1 for v in vals if full and 0 < v < full - 1e-9),
                    "wrong": sum(1 for v in vals if v <= 0)}
    sec_max = {t: sum(float(qbyn[n].get("score") or 0) for n in nums if types[n] == t) for t in sections}
    absent, roster_info = _students(store, exam, rows)
    return {"numbers": nums, "types": types, "sections": sections, "section_max": sec_max, "rows": rows, "stats": stats,
            "absent": absent, "roster": roster_info, "scored": is_scored(exam), "kind": exam.get("kind") or "exam"}


VERDICT_TEXT = {1.0: "对", 0.5: "半对", 0.0: "错"}


def _verdict_text(v) -> str:
    if v is None:
        return ""
    v = float(v)
    return "对" if v >= 1 - 1e-9 else ("错" if v <= 0 else "半对")


def _students(store: Store, exam: dict, rows: List[dict]):
    """每行补上拼音（搜索用）和学号问题（没涂清、不在名单、重号）；再找出名单里还没有答卷的人。"""
    by_sid: dict = {}
    for r in rows:
        stu = r["student"]
        r["py"], r["pyi"] = RS.pinyin_keys(stu.get("name", ""))
        k = RS.norm_sid(stu.get("student_id"))
        if k:
            by_sid.setdefault(k, []).append(r["id"])
    for r in rows:
        issues = list(r["student"].get("issues") or [])
        k = RS.norm_sid(r["student"].get("student_id"))
        if k and len(by_sid.get(k, [])) > 1:
            others = [x for x in by_sid[k] if x != r["id"]]
            issues.append(f"学号 {r['student']['student_id']} 和 {'、'.join(others[:3])}" + (f" 等 {len(others)} 份" if len(others) > 3 else "")
                          + " 重复")
        r["id_issues"] = issues
    bound = exam.get("roster") or []
    idx = RS.exam_index(store.root, exam)
    info = {"bound": bound, "classes": idx["classes"] if bound else [], "n": idx["n"], "auto": not bound}
    if not idx["n"]:
        return [], info
    if bound:
        scope = set(bound)
    else:                                   # 没指定：看答卷上认出来的学生都在哪几个班
        scope = {e["class_id"] for k in by_sid for e in idx["by_sid"].get(k, [])}
        info["classes"] = sorted({e["class"] for k in by_sid for e in idx["by_sid"].get(k, [])})
    if not scope:
        return [], info
    names_seen = {r["student"].get("name") for r in rows if r["student"].get("name")}
    absent = []
    for k, ents in idx["by_sid"].items():
        for e in ents:
            if e["class_id"] in scope and k not in by_sid and e["name"] not in names_seen:
                absent.append({"sid": e["sid"], "name": e["name"], "class": e["class"]})
    absent.sort(key=lambda e: (e["class"], RS._sid_key(e["sid"])))
    return absent, info


def side_name(p: int, n: int) -> str:
    """扫描的第 p 面（从 0 起）怎么称呼：两面的答题纸说正面/反面（A3 两栏一面印两页，说“第几页”会和印的页码混）。"""
    if n <= 2:
        return "正面" if p == 0 else "反面"
    return f"第 {p // 2 + 1} 张{'正面' if p % 2 == 0 else '反面'}"


def sides_text(pages, n: int) -> str:
    return "、".join(side_name(int(p), n) for p in pages)


def _remark(r: dict) -> str:
    out = []
    if r.get("id_issues"):
        out.append("学号待核对：" + "；".join(r["id_issues"][:2]))
    if r.get("missing_pages"):
        out.append(f"缺{sides_text(r['missing_pages'], r.get('n_sides') or 2)}（这一面上的题没有批改）")
    if not r.get("align_ok", True):
        out.append("有页面没能和答题纸对齐，请人工批改")
    return "；".join(out)


def export_csv(store: Store, exam_id: str) -> Path:
    res = results(store, exam_id)
    nums = res["numbers"]
    buf = io.StringIO()
    w = csv.writer(buf)
    secs = res["sections"]
    scored = res.get("scored", True)
    if scored:
        w.writerow(["序号", "姓名", "学号", "班级", "扫描文件"] + [f"第{n}题" for n in nums]
                   + [f"{t}合计" for t in secs] + ["总分", "待复核题号", "备注"])
    else:                                       # 平时作业：每题写对/半对/错，最后是做对几题（半对算半题）
        w.writerow(["序号", "姓名", "学号", "班级", "扫描文件"] + [f"第{n}题" for n in nums]
                   + [f"{t}做对" for t in secs] + [f"做对题数（共{len(nums)}题）", "待复核题号", "备注"])
    for r in res["rows"]:
        stu = r["student"]
        files = "；".join(sorted({f.get("file", "") for f in r["files"]}))
        cells = ([("" if r["scores"].get(n) is None else f"{r['scores'][n]:g}") for n in nums] if scored
                 else [_verdict_text(r["scores"].get(n)) for n in nums])
        w.writerow([r["id"], stu.get("name", ""), stu.get("student_id", ""), stu.get("class", ""), files]
                   + cells + [f"{r['sections'][t]:g}" for t in secs]
                   + [f"{r['total']:g}", "、".join(r["needs_review"]), _remark(r)])
    for a in res.get("absent") or []:          # 名单里没有答卷的人（没交、没扫到）
        w.writerow(["", a["name"], a["sid"], a["class"], ""] + [""] * len(nums) + [""] * len(secs) + ["", "", "没有答卷"])
    path = store.exam_dir(exam_id) / "scores.csv"
    path.write_text("﻿" + buf.getvalue(), encoding="utf-8")   # 带 BOM，Excel 直接打开不乱码
    return path


def annotate_sheet(store: Store, exam: dict, sheet: dict) -> List[str]:
    """在对齐后的答题纸上用红字写分数（类似人工阅卷的效果）。"""
    d = store.exam_dir(exam["id"])
    tmpl = exam.get("template") or {}
    scored = is_scored(exam)
    out = []
    for pg in sheet.get("pages") or []:
        src = d / pg["display"] if pg.get("display") else None
        if not src or not src.exists():
            continue
        img = cv2.imread(str(src))
        if img is None:
            continue
        L = max(img.shape[:2])
        sc = float(pg.get("display_scale") or 1.0)
        texts = []
        page_total, page_max = 0.0, 0.0
        for n, qr in (sheet.get("questions") or {}).items():
            s = final_score(qr)
            ai = qr.get("ai") or {}
            if s is None:
                continue
            mx = ai.get("max") or 0
            for (p, r) in regions_for(tmpl, n):
                if p != pg["template_page"]:
                    continue
                x0, y0, x1, y1 = [v * sc for v in r["rect"]]
                teacher = (qr.get("teacher") or {}).get("score") is not None
                review = (not teacher) and ai.get("needs_review")
                if not scored:                     # 平时作业：画 √ / 半对 / ×，错的写一句错在哪
                    size = max(26, int(L / 45))
                    kind = "tick" if float(s) >= 1 - 1e-9 else ("cross" if float(s) <= 0 else "half")
                    color = (0, 110, 255) if review else (20, 20, 220)
                    _draw_mark(img, kind, x1 - 1.3 * size, y0 + 1.1 * size, 1.6 * size, color)
                    why = "" if kind == "tick" or teacher else (ai.get("error") or ai.get("comment") or "")
                    if why or review:
                        texts.append({"xy": (x1 - 2.4 * size, y0 + 0.6 * size), "text": (why[:36] + ("…" if len(why) > 36 else ""))
                                      + ("  待复核" if review else ""), "size": max(18, int(L / 75)), "color": color,
                                      "stroke": 3, "anchor": "ra"})
                    break
                line = f"第{n}题 {s:g}分（满分{mx:g}）"
                texts.append({"xy": (x1 - 10, y0 + 8), "text": line + ("  待复核" if review else ""),
                              "size": max(20, int(L / 60)), "color": (0, 110, 255) if review else (20, 20, 220),
                              "stroke": 3, "anchor": "ra"})
                parts = ai.get("parts") or []
                if len(parts) > 1 and not teacher:
                    detail = "  ".join(f"{p['label']}{p['score']:g}/{p['max']:g}" for p in parts)
                    texts.append({"xy": (x1 - 10, y0 + 12 + max(20, int(L / 60)) * 1.4), "text": detail,
                                  "size": max(16, int(L / 80)), "color": (20, 20, 220), "stroke": 2, "anchor": "ra"})
                break
            page_total += float(s)
            page_max += float(mx)
        otexts, shapes = _objective_marks(tmpl, sheet, pg["template_page"], sc, L, scored)
        texts += otexts
        for kind, cx, cy, sz, color in shapes:
            _draw_mark(img, kind, cx, cy, sz, color)
        if pg["template_page"] == 0:
            stu = sheet.get("student") or {}
            name = stu.get("name") or sheet["id"]
            qs = sheet.get("questions") or {}
            secs = {}
            for q in qs.values():
                v = final_score(q)
                if v is not None:
                    t = q.get("type") or "解答题"
                    secs[t] = secs.get(t, 0.0) + float(v)
            tot = sum(secs.values())
            detail = "，".join(f"{t[:2]} {secs[t]:g}" for t in SECTION_ORDER if t in secs)
            if scored:
                head = f"{name}  总分 {tot:g}" + (f"（{detail}）" if len(secs) > 1 else "")
            else:
                n_all = sum(1 for q in qs.values() if final_score(q) is not None)
                head = f"{name}  做对 {tot:g}/{n_all}"
            texts.append({"xy": (int(0.06 * L), int(0.012 * L)), "text": head,
                          "size": max(24, int(L / 45)), "color": (20, 20, 220), "stroke": 3})
        img = draw_texts(img, texts)
        name = f"annotated_{pg['template_page']}.jpg"
        imwrite(d / "work" / sheet["id"] / name, img, quality=88)
        out.append(f"work/{sheet['id']}/{name}")
    return out


def _draw_mark(img, kind: str, cx: float, cy: float, sz: float, color) -> None:
    """像红笔一样画 √ 和 ×（不用字体里的根号）。"""
    t = max(2, int(sz / 9))
    if kind == "tick":
        pts = np.array([[cx - 0.45 * sz, cy], [cx - 0.12 * sz, cy + 0.35 * sz], [cx + 0.5 * sz, cy - 0.45 * sz]], np.int32)
        cv2.polylines(img, [pts], False, color, t, cv2.LINE_AA)
    elif kind == "cross":
        h = 0.35 * sz
        cv2.line(img, (int(cx - h), int(cy - h)), (int(cx + h), int(cy + h)), color, t, cv2.LINE_AA)
        cv2.line(img, (int(cx - h), int(cy + h)), (int(cx + h), int(cy - h)), color, t, cv2.LINE_AA)
    elif kind == "half":
        _draw_mark(img, "tick", cx, cy, sz, color)
        cv2.line(img, (int(cx), int(cy - 0.1 * sz)), (int(cx + 0.35 * sz), int(cy + 0.3 * sz)), color, t, cv2.LINE_AA)


def _objective_marks(tmpl: dict, sheet: dict, page: int, sc: float, L: int, scored: bool = True):
    """填空题、选择题：每题打 √ / ×（部分得分写分数），每块区域右上角上方写小计。返回 (文字, 图形)。"""
    places = objective_places(tmpl)
    qs = sheet.get("questions") or {}
    texts, shapes = [], []
    sub: dict = {}
    size = max(22, int(L / 55))
    for n, pl in places.items():
        if pl["page"] != page or n not in qs:
            continue
        qr = qs[n]
        v = final_score(qr)
        if v is None:
            continue
        mx = float(qr.get("max") or (qr.get("ai") or {}).get("max") or 0)
        rid = pl["region"]
        sub.setdefault(rid, [0.0, 0.0])
        sub[rid][0] += float(v)
        sub[rid][1] += mx
        if pl.get("whole"):
            continue
        teacher = (qr.get("teacher") or {}).get("score") is not None
        review = not teacher and (qr.get("ai") or {}).get("needs_review")
        color = (0, 110, 255) if review else (20, 20, 220)
        right = mx > 0 and float(v) >= mx - 1e-9
        kind = "tick" if right else ("cross" if float(v) <= 0 else "half")
        if pl.get("slot"):
            x0, y0, x1, y1 = [t * sc for t in pl["slot"]["rect"]]
            cx, cy = x1 - 0.55 * size, (y0 + y1) / 2 + 0.05 * size
        else:
            x0, y0, x1, y1 = [t * sc for t in pl["group"]["rect"]]
            cx, cy = x1 + 0.9 * size, (y0 + y1) / 2
        shapes.append((kind, cx, cy, 1.2 * size, color))
        extra = (f"{float(v):g}" if kind == "half" and scored else "") + ("?" if review else "")
        if extra:
            texts.append({"xy": (cx + 0.7 * size, cy), "text": extra, "size": int(0.8 * size), "color": color,
                          "stroke": 2, "anchor": "lm"})
    for pg in tmpl.get("pages") or []:
        if pg["page"] != page:
            continue
        for r in pg["regions"]:
            if r["id"] in sub:
                x0, y0, x1, y1 = [t * sc for t in r["rect"]]
                got, mx = sub[r["id"]]
                texts.append({"xy": (x1 - 4, y0 - 4), "text": f"{r.get('kind', '')[:2]} {'' if scored else '对 '}{got:g}/{mx:g}", "size": size,
                              "color": (20, 20, 220), "stroke": 3, "anchor": "rb"})
    return texts, shapes
