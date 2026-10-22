"""客观题：在答题纸模板上找填空横线和选择题选项框；在学生答卷上识别涂卡、判断填空是否留空；评分规则。

模板坐标就是版面参考图（长边 2000 像素）的像素坐标。
"""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .imaging import draw_texts

LETTERS = "ABCDEFGH"


# ================================================================ 工具

def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _binarize(gray: np.ndarray) -> np.ndarray:
    blk = max(15, int(min(gray.shape[:2]) / 6) | 1)
    a = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blk, 18)
    _, o = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.bitwise_and(a, o) if o.mean() < 0.5 * 255 else a


def _inner(rect, inset: int, shape) -> Tuple[int, int, int, int]:
    H, W = shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in rect]
    return max(0, x0 + inset), max(0, y0 + inset), min(W, x1 - inset), min(H, y1 - inset)


def _runs(mask_1d: np.ndarray, min_gap: int = 0) -> List[Tuple[int, int]]:
    """一维 0/1 序列里连续为 1 的段；中间空开不超过 min_gap 个 0 的合并成一段。返回 [起, 止)。"""
    idx = np.flatnonzero(mask_1d)
    if idx.size == 0:
        return []
    runs = []
    s = p = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - p - 1 > min_gap:
            runs.append((s, p + 1))
            s = i
        p = i
    runs.append((s, p + 1))
    return runs


def _two_means_threshold(vals: Sequence[float]) -> Optional[float]:
    """一维两类聚类的分界（用于区分“字内间隙”和“选项之间的间隙”）。"""
    v = np.sort(np.asarray(vals, float))
    if v.size < 4 or v[-1] - v[0] < 2:
        return None
    best, thr = None, None
    for k in range(1, v.size):
        a, b = v[:k], v[k:]
        cost = a.var() * a.size + b.var() * b.size
        if best is None or cost < best:
            best, thr = cost, (a[-1] + b[0]) / 2.0
    lo, hi = v[v <= thr], v[v > thr]
    if lo.size == 0 or hi.size == 0 or hi.mean() < 1.8 * max(1.0, lo.mean()):
        return None
    return float(thr)


# ================================================================ 模板：填空横线

def detect_fill_slots(img: np.ndarray, rect, numbers: Optional[List[Tuple[str, float, float, float, float]]] = None,
                      questions: Sequence[str] = ()) -> List[dict]:
    """在填空题区域里找横线，配上题号。numbers: 模板坐标的题号文字 [(题号, x0, y0, x1, y1)]（来自 PDF 文字层）。
    没有文字层时按“先行后列”的顺序套用 questions。返回 [{"q", "rect", "line", "src"}]。"""
    gray = _gray(img)
    x0, y0, x1, y1 = _inner(rect, 3, gray.shape)
    if x1 - x0 < 30 or y1 - y0 < 15:
        return []
    sub = gray[y0:y1, x0:x1]
    bw = _binarize(sub)
    W = x1 - x0
    min_len = max(30, int(0.10 * W))
    m = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    lines = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if h > 8 or w > 0.9 * W:
            continue
        cy = y + h / 2.0
        if cy < 4 or cy > (y1 - y0) - 4:
            continue
        lines.append([x0 + x, y0 + cy, x0 + x + w])
    if not lines:
        return []
    lines.sort(key=lambda L: (L[1], L[0]))
    ys = sorted(L[1] for L in lines)
    rows = []
    for y in ys:
        if not rows or y - rows[-1] > 6:
            rows.append(y)
    pitch = float(np.median(np.diff(rows))) if len(rows) >= 2 else 0.45 * (y1 - y0)
    slots = []
    used = set()
    if numbers:
        for q, nx0, ny0, nx1, ny1 in sorted(numbers, key=lambda t: (t[2], t[1])):
            nh = max(4.0, ny1 - ny0)
            best, bd = None, None
            for k, (lx0, ly, lx1) in enumerate(lines):
                if k in used or lx0 < nx1 - 6 or lx0 - nx1 > 0.3 * W:
                    continue
                if abs(ly - ny1) > max(10.0, 1.6 * nh):
                    continue
                d = lx0 - nx1
                if bd is None or d < bd:
                    best, bd = k, d
            if best is None:
                continue
            used.add(best)
            lx0, ly, lx1 = lines[best]
            slots.append({"q": q, "line": [round(lx0), round(ly), round(lx1)], "src": "pdf_text",
                          "rect": [round(nx1 + 2), round(max(y0, ly - 0.92 * pitch)), round(lx1 + 3),
                                   round(min(y1, ly + 0.22 * pitch))]})
    else:
        # 先按行、再按列排好横线，依次对应 questions
        ordered = []
        for ry in rows:
            ordered += sorted([L for L in lines if abs(L[1] - ry) <= 6], key=lambda L: L[0])
        qs = list(questions)
        for k, (lx0, ly, lx1) in enumerate(ordered):
            q = qs[k] if k < len(qs) else ""
            left = lx0 - 0.06 * W
            prev = [L for L in ordered if abs(L[1] - ly) <= 6 and L[2] < lx0]
            if prev:
                left = max(left, max(L[2] for L in prev) + 0.05 * W)
            slots.append({"q": q, "line": [round(lx0), round(ly), round(lx1)], "src": "order",
                          "rect": [round(max(x0, left)), round(max(y0, ly - 0.92 * pitch)), round(lx1 + 3),
                                   round(min(y1, ly + 0.22 * pitch))]})
    return slots


# ================================================================ 模板：选择题选项框

def detect_choice_groups(img: np.ndarray, rect, numbers: Optional[List[Tuple[str, float, float, float, float]]] = None,
                         questions: Sequence[str] = ()) -> List[dict]:
    """在选择题区域里找每道题的选项框（[A] [B] [C] [D]、圆圈、方框都可以，涂黑的也行）。
    做法：按行切开 → 每行按列投影得到墨迹段 → 用间隙大小的两类聚类把墨迹段并成“字块” →
    每道题 = 题号字块 + 后面的选项字块。返回 [{"q", "rect", "options": [{"label", "rect"}], "src"}]。"""
    gray = _gray(img)
    x0, y0, x1, y1 = _inner(rect, 4, gray.shape)
    if x1 - x0 < 30 or y1 - y0 < 10:
        return []
    sub = gray[y0:y1, x0:x1]
    bw = _binarize(sub)
    # 去掉框线之类的长线
    long_h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, (x1 - x0) // 8), 1)))
    long_v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, (y1 - y0) // 2))))
    bw = cv2.bitwise_and(bw, cv2.bitwise_not(cv2.bitwise_or(long_h, long_v)))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) if bw.mean() > 0.05 * 255 else bw
    prof = (bw > 0).sum(axis=1)
    bands = [(a, b) for a, b in _runs(prof > 0, min_gap=2) if b - a >= 5]
    if not bands:
        return []
    row_runs = [_runs((bw[a:b] > 0).sum(axis=0) > 0, 0) for a, b in bands]
    gaps = sorted({r2[0] - r1[1] for runs in row_runs for r1, r2 in zip(runs, runs[1:])})
    med_h = float(np.median([b - a for a, b in bands]))

    def merge(runs, thr):
        out = []
        for r in runs:
            if out and r[0] - out[-1][1] <= thr:
                out[-1][1] = r[1]
            else:
                out.append([r[0], r[1]])
        return out

    def option_runs(blocks, W):
        """一行里连续出现、宽度相近的字块（= 一道题的几个选项），返回 [(起, 止)] 下标区间。"""
        ok = [abs((b[1] - b[0]) - W) <= 0.4 * W for b in blocks]
        res, k = [], 0
        while k < len(blocks):
            if ok[k]:
                j = k
                while j + 1 < len(blocks) and ok[j + 1] and blocks[j + 1][0] - blocks[j][1] <= 1.2 * W:
                    j += 1
                # 选项之间的间距是均匀的；题号和第一个选项之间的缝明显更大，在那里断开
                seg = list(range(k, j + 1))
                gs = [blocks[t + 1][0] - blocks[t][1] for t in seg[:-1]]
                if gs:
                    typical = float(np.median(gs)) if len(gs) >= 3 else min(gs)
                    cut = [t for t, g in zip(seg[:-1], gs) if g > 1.8 * max(1.0, typical) and g - typical > 2]
                    bounds = [k] + [t + 1 for t in cut] + [j + 1]
                    for u, v in zip(bounds, bounds[1:]):
                        if v - u >= 2:
                            res.append((u, v))
                k = j + 1
            else:
                k += 1
        return res

    # 试不同的“合并间隙”，选出能让各行都得到整齐选项的那个（字内的小缝要合并，选项之间的缝不能合并）
    best = None
    for thr in [0.5] + [g + 0.5 for g in gaps if g < 4 * med_h]:
        rows_blocks = [merge(runs, thr) for runs in row_runs]
        widths = [b[1] - b[0] for rb in rows_blocks for b in rb if b[1] - b[0] >= 0.3 * med_h]
        if not widths:
            continue
        p75 = float(np.percentile(widths, 75))
        W = float(np.median([w for w in widths if w >= 0.5 * p75]))
        found = [option_runs(rb, W) for rb in rows_blocks]
        lens = [e - st for f in found for st, e in f]
        if not lens:
            continue
        common = max(set(lens), key=lens.count)
        score = sum(L for L in lens if L == common) - 0.5 * sum(1 for L in lens if L != common)
        if best is None or score > best[0] + 1e-9:
            best = (score, thr, W, rows_blocks, found)
    if best is None:
        return []
    _, thr, W, rows_blocks, found = best
    groups_all = []
    for (a, b), blocks, runs_idx in zip(bands, rows_blocks, found):
        prev_end = 0
        for st, e in runs_idx:
            opts = blocks[st:e]
            numb = blocks[prev_end:st]
            prev_end = e
            gx0 = numb[0][0] if numb else opts[0][0]
            q = None
            if numbers:
                cands = [t for t in numbers if t[2] < y0 + b and t[4] > y0 + a and t[3] <= x0 + opts[0][0] + 2]
                if cands:
                    t = max(cands, key=lambda t: t[3])
                    if (x0 + opts[0][0]) - t[3] < 6 * W:
                        q = t[0]
                        gx0 = min(gx0, t[1] - x0)
            groups_all.append({"q": q, "src": "pdf_text" if q else "order",
                               "rect": [x0 + gx0, y0 + a, x0 + opts[-1][1], y0 + b],
                               "options": [{"label": LETTERS[k], "rect": [x0 + o[0] - 1, y0 + a - 1, x0 + o[1] + 1, y0 + b + 1]}
                                           for k, o in enumerate(opts[:len(LETTERS)])]})
    # 没有文字层：按“先行后列”套题号
    if not numbers:
        qs = list(questions)
        for k, g in enumerate(sorted(groups_all, key=lambda g: (g["rect"][1], g["rect"][0]))):
            g["q"] = qs[k] if k < len(qs) else ""
    # 统一选项个数（取众数），个数不对的标出来
    if groups_all:
        cnt = np.bincount([len(g["options"]) for g in groups_all]).argmax()
        for g in groups_all:
            if len(g["options"]) != cnt:
                g["odd"] = True
    return groups_all


# ================================================================ 学生答卷：涂卡

def _darkness(gray: np.ndarray, rect, paper: float) -> float:
    x0, y0, x1, y1 = _inner(rect, 1, gray.shape)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    return float(np.clip(1.0 - patch.mean() / max(1.0, paper), 0.0, 1.0))


def paper_level(gray: np.ndarray, rect) -> float:
    x0, y0, x1, y1 = _inner(rect, 0, gray.shape)
    patch = gray[y0:y1, x0:x1]
    return float(np.percentile(patch, 90)) if patch.size else 255.0


def option_darkness(gray: np.ndarray, group: dict, paper: float) -> List[float]:
    return [round(_darkness(gray, o["rect"], paper), 3) for o in group["options"]]


MARK = 0.28      # 比没涂的选项暗这么多，算涂了（真实铅笔涂卡一般在 0.4 以上）
WEAK = 0.10      # 介于两者之间：涂得浅或擦过，交给老师看


def read_choice(scan_gray: np.ndarray, group: dict, scan_paper: float,
                base: Optional[List[float]] = None) -> dict:
    """scan_gray：已对齐到模板坐标的学生答卷（灰度）。base：空白模板上同一选项的暗度（模板来自学生答卷时为 None）。
    返回 {"picked": "B", "d": [...], "delta": [...], "flags": [...]}，flags 非空表示需要老师看一眼。"""
    d = option_darkness(scan_gray, group, scan_paper)
    if base and len(base) == len(d):
        diff = [a - b for a, b in zip(d, base)]
        off = min(diff)                       # 扫描比模板整体偏暗/偏亮：用最浅的那个选项抵消
        delta = [round(x - off, 3) for x in diff]
    else:
        lo = min(d)
        delta = [round(x - lo, 3) for x in d]
    labels = [o["label"] for o in group["options"]]
    strong = [l for l, x in zip(labels, delta) if x >= MARK]
    weak = [l for l, x in zip(labels, delta) if WEAK <= x < MARK]
    flags = []
    if strong:
        picked = "".join(strong)
        if weak:
            flags.append(f"选项 {''.join(weak)} 有较浅的涂擦痕迹")
    elif len(weak) == 1:
        picked = weak[0]
        flags.append(f"选项 {picked} 涂得较浅，请确认")
    else:
        picked = ""
        if weak:
            flags.append(f"选项 {''.join(weak)} 都有较浅的痕迹，没有明确涂的选项")
    if base and min(a - b for a, b in zip(d, base)) > 0.25:
        flags.append("所有选项都偏暗，可能全涂了或扫描太暗")
    return {"picked": picked, "d": d, "delta": delta, "flags": flags}


def score_choice(picked: str, ref: str, full: float, multi_partial: str = "proportional") -> Tuple[float, str]:
    """单选：对得满分。多选（参考答案多个字母）：全对满分；没有选错但没选全按比例；有选错 0 分。"""
    ref = "".join(sorted(set((ref or "").upper())))
    got = "".join(sorted(set((picked or "").upper())))
    if not ref:
        return 0.0, "没有参考答案"
    if not got:
        return 0.0, "未作答"
    if len(ref) == 1:
        if got == ref:
            return float(full), "正确"
        return 0.0, "多涂" if len(got) > 1 else "错误"
    if set(got) - set(ref):
        return 0.0, "有错选"
    if got == ref:
        return float(full), "正确"
    if multi_partial == "none":
        return 0.0, "没选全"
    if multi_partial.replace(".", "", 1).isdigit():
        return min(float(multi_partial), float(full)), "没选全"
    return float(full) * len(got) / len(ref), "没选全"


# ================================================================ 学生答卷：填空是否留空

def ink_added(scan_gray: np.ndarray, tmpl_gray: np.ndarray, rect) -> float:
    """学生在这个区域里比空白模板多写了多少墨迹（占面积比例）。两张图都已在模板坐标系。"""
    x0, y0, x1, y1 = _inner(rect, 1, scan_gray.shape)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    s = scan_gray[y0:y1, x0:x1]
    t = tmpl_gray[y0:y1, x0:x1]
    ps, pt = np.percentile(s, 90), np.percentile(t, 90)
    sd = s < 0.6 * ps
    td = cv2.dilate((t < 0.75 * pt).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return float((sd & ~td).mean())


# ================================================================ 填空：答案比对

_FW = str.maketrans("（）［］｛｝，；：＝＋－－×÷．、｜＜＞　", "()[]{},;:=+--*/.,|<> ")


def norm_answer(s: str) -> str:
    """用于“完全一致”快速判对的规范化（只处理写法差异，不做数学推导）。"""
    t = (s or "").strip()
    t = t.translate(_FW)
    t = t.replace("$", "")
    t = re.sub(r"\\(left|right|displaystyle|textstyle|,|;|!|quad|qquad)", "", t)
    t = re.sub(r"\\[dt]frac", r"\\frac", t)
    t = re.sub(r"\\mathrm\{([^{}]*)\}|\\text\{([^{}]*)\}|\\mathbf\{([^{}]*)\}|\\boldsymbol\{([^{}]*)\}",
               lambda m: next(g for g in m.groups() if g is not None), t)
    t = t.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "π").replace("\\sqrt", "√")
    t = t.replace("\\leqslant", "≤").replace("\\leq", "≤").replace("\\le", "≤")
    t = t.replace("\\geqslant", "≥").replace("\\geq", "≥").replace("\\ge", "≥").replace("\\neq", "≠")
    t = t.replace("\\infty", "∞").replace("\\cup", "∪").replace("\\cap", "∩").replace("\\in", "∈")
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"\{(\w)\}", r"\1", t)          # {5} -> 5
    t = t.rstrip("。.")
    return t


def exact_match(student: str, ref: str) -> bool:
    a, b = norm_answer(student), norm_answer(ref)
    if not a or not b:
        return False
    if a == b:
        return True
    # 参考答案里写了几种都算对的形式，如 “2 或 -1” 不在这里处理，只认“（或写成 …）”
    alts = [norm_answer(x) for x in re.split(r"（?或写?成|\(or|；|;", ref) if x.strip()]
    return a in alts


# ================================================================ 填空：拼图（一次识别多个空）

def contact_sheets(items: List[Tuple[str, np.ndarray]], per_sheet: int = 6, row_h: int = 130,
                   max_w: int = 1500) -> List[np.ndarray]:
    """把每个空的截图按行拼起来，左边写上程序标的题号（不依赖模型去认印刷的题号）。"""
    sheets = []
    for k in range(0, len(items), per_sheet):
        rows = []
        for q, img in items[k:k + per_sheet]:
            h, w = img.shape[:2]
            s = row_h / float(max(1, h))
            im = cv2.resize(img, (max(1, min(max_w, int(w * s))), row_h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            tag = np.full((row_h, 120, 3), 235, np.uint8)
            tag = draw_texts(tag, [{"xy": (60, row_h // 2), "text": f"第{q}题", "size": 30, "color": (30, 30, 30), "anchor": "mm"}])
            row = np.hstack([tag, np.full((row_h, 6, 3), 90, np.uint8), im])
            rows.append(row)
        W = max(r.shape[1] for r in rows)
        out = []
        for r in rows:
            r = cv2.copyMakeBorder(r, 0, 0, 0, W - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            out += [r, np.full((5, W, 3), 90, np.uint8)]
        sheets.append(np.vstack(out[:-1]))
    return sheets


def number_marks(frags, to_tmpl, rect) -> List[Tuple[str, float, float, float, float]]:
    """PDF 文字层里落在区域内的题号（模板坐标）。“13”这种单独的数字直接用；
    “1  [A] [B]…”这种连在一起的，按字符比例估计题号的宽度。"""
    out = []
    x0r, y0r, x1r, y1r = rect
    for f in frags or []:
        m = re.match(r"^\s*(\d{1,2})\s*[\.．、]?", f.text)
        if not m:
            continue
        rest = f.text[m.end():].strip()
        if rest and not re.match(r"^[\[［【(（]?\s*[A-H]", rest):
            continue
        ax, ay = to_tmpl(f.x0, f.y0)
        bx, by = to_tmpl(f.x1, f.y1)
        fx0, fy0, fx1, fy1 = min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)
        n = len(f.text.rstrip())
        if rest:
            fx1 = fx0 + (fx1 - fx0) * max(1, len(m.group(0).rstrip())) / max(1, n)
        cx, cy = (fx0 + fx1) / 2, (fy0 + fy1) / 2
        if x0r <= cx <= x1r and y0r <= cy <= y1r:
            out.append((m.group(1), fx0, fy0, fx1, fy1))
    return out
