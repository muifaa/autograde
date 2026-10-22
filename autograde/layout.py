"""版面分析：在答题纸上找出所有封闭的题框（作答区域），与具体版式无关。

思路：提取长横线/长竖线 → 修补断线 → 找被线条围起来的大块空白区域。
解答题框上方的打分栏、填空题后面的小方框等小格子会因为面积太小被过滤掉。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .imaging import red_ink_ratio, resize_long, strip_red, draw_texts
from .markers import Marker, detect_markers, select_page_markers

WORK_LONG_SIDE = 2000  # 模板坐标系：参考图长边 2000 像素


@dataclass
class PageLayout:
    image: np.ndarray                  # 参考图（已纠斜、已去红笔），模板坐标系就是它的像素坐标
    markers: List[Marker]
    boxes: List[Tuple[int, int, int, int]]
    line_mask: np.ndarray              # 用于检查对齐质量
    print_color: str = "black"
    skew_deg: float = 0.0
    notes: List[str] = field(default_factory=list)
    to_tmpl: Optional[np.ndarray] = None   # 2x3：输入图像素坐标 -> 模板坐标

    def map_point(self, x: float, y: float) -> Tuple[float, float]:
        M = self.to_tmpl if self.to_tmpl is not None else np.array([[1.0, 0, 0], [0, 1.0, 0]])
        return (float(M[0, 0] * x + M[0, 1] * y + M[0, 2]), float(M[1, 0] * x + M[1, 1] * y + M[1, 2]))


def _binarize(gray: np.ndarray) -> np.ndarray:
    L = max(gray.shape)
    blk = int(L / 60) | 1
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blk, 15)


def _segments(bw: np.ndarray, horizontal: bool) -> List[Tuple[float, float, float, float]]:
    """返回线段 (位置, 起点, 终点, 粗细)。横线位置是 y，竖线位置是 x。"""
    H, W = bw.shape
    L = max(H, W)
    gap = max(3, int(round(0.0035 * L)))
    minlen = max(15, int(round(0.025 * L)))
    if horizontal:
        m = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (gap, 1)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (minlen, 1)))
    else:
        m = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, gap)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, minlen)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
    segs = []
    max_thick = max(3, 0.012 * L)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if horizontal:
            if h > max_thick or w < minlen:
                continue
            segs.append((float(cen[i][1]), float(x), float(x + w - 1), float(area) / w))
        else:
            if w > max_thick or h < minlen:
                continue
            segs.append((float(cen[i][0]), float(y), float(y + h - 1), float(area) / h))
    return _merge_collinear(segs, L)


def _merge_collinear(segs, L):
    tol_pos = max(2.0, 0.003 * L)
    tol_gap = 0.015 * L
    segs = sorted(segs)
    changed = True
    while changed:
        changed = False
        out: List[list] = []
        for p, a, b, t in segs:
            for o in out:
                if abs(o[0] - p) <= tol_pos and max(0.0, a - o[2], o[1] - b) <= tol_gap:
                    wa, wb = o[2] - o[1], b - a
                    o[0] = (o[0] * wa + p * wb) / max(1.0, wa + wb)
                    o[1], o[2], o[3] = min(o[1], a), max(o[2], b), max(o[3], t)
                    changed = True
                    break
            else:
                out.append([p, a, b, t])
        segs = [tuple(o) for o in out]
    return segs


def _snap(hs, vs, L):
    """线段端点离垂直方向的线很近时（扫描后边角处常有断口），把它延长过去，让框闭合。"""
    d, tol = 0.025 * L, 0.004 * L
    vs2 = []
    for x, a, b, t in vs:
        cross = [y for (y, x0, x1, _) in hs if x0 - tol <= x <= x1 + tol]
        up = [y for y in cross if a - d <= y < a + tol]
        if up:
            a = min(a, min(up, key=lambda y: abs(y - a)))
        dn = [y for y in cross if b - tol < y <= b + d]
        if dn:
            b = max(b, min(dn, key=lambda y: abs(y - b)))
        vs2.append((x, a, b, t))
    hs2 = []
    for y, a, b, t in hs:
        cross = [x for (x, y0, y1, _) in vs2 if y0 - tol <= y <= y1 + tol]
        lf = [x for x in cross if a - d <= x < a + tol]
        if lf:
            a = min(a, min(lf, key=lambda x: abs(x - a)))
        rt = [x for x in cross if b - tol < x <= b + d]
        if rt:
            b = max(b, min(rt, key=lambda x: abs(x - b)))
        hs2.append((y, a, b, t))
    return hs2, vs2


def estimate_skew(gray: np.ndarray) -> float:
    """用长直线（题框边线）估计倾斜角（度，逆时针为正，可直接传给 rotate 纠正）。"""
    bw = _binarize(gray)
    L = max(bw.shape)
    lines = cv2.HoughLinesP(bw, 1, np.pi / 1800, threshold=int(0.05 * L),
                            minLineLength=int(0.12 * L), maxLineGap=max(2, int(0.003 * L)))
    if lines is None:
        return 0.0
    angles, weights = [], []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):  # OpenCV 4: Nx1x4, OpenCV 5: Nx4
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = (dx * dx + dy * dy) ** 0.5
        ang = np.degrees(np.arctan2(dy, dx))
        # 横线：偏离 0°；竖线：偏离 ±90°。统一换算成“图像需要转多少度”
        if abs(ang) < 5 or abs(abs(ang) - 180) < 5:
            dev = ang if abs(ang) < 5 else (ang - 180 if ang > 0 else ang + 180)
        elif abs(abs(ang) - 90) < 5:
            dev = (ang - 90) if ang > 0 else (ang + 90)
        else:
            continue
        angles.append(dev)
        weights.append(length)
    if not angles:
        return 0.0
    order = np.argsort(angles)
    a, wts = np.array(angles)[order], np.array(weights, float)[order]
    c = np.cumsum(wts)
    # 图像坐标 y 向下，线段角度为正表示顺时针倾斜，需要逆时针转回来
    return float(a[np.searchsorted(c, c[-1] / 2.0)])


def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), deg, 1.0)
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=border)


def find_boxes(gray: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """返回 (封闭区域列表 [x0,y0,x1,y1], 修补后的线条图)。"""
    H, W = gray.shape
    L = max(H, W)
    bw = _binarize(gray)
    hs = _segments(bw, True)
    vs = _segments(bw, False)
    hs, vs = _snap(hs, vs, L)
    clean = np.zeros_like(bw)
    ext = int(round(0.004 * L))
    th = max(2, int(round(0.0015 * L)))
    for y, a, b, t in hs:
        cv2.line(clean, (int(a) - ext, int(round(y))), (int(b) + ext, int(round(y))), 255, th)
    for x, a, b, t in vs:
        cv2.line(clean, (int(round(x)), int(a) - ext), (int(round(x)), int(b) + ext), 255, th)

    cnts, hier = cv2.findContours(clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    if hier is not None:
        hier = hier[0]
        page_area = float(H * W)
        for i, c in enumerate(cnts):
            if hier[i][3] == -1:  # 只要“洞”
                continue
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area < 0.008 * page_area or area > 0.6 * page_area:
                continue
            if min(w, h) < 0.025 * min(H, W) or area / float(w * h) < 0.85:
                continue
            boxes.append((x, y, x + w, y + h))
    # 去掉“套着别的框”的大框（比如整页外框）
    def contains(o, i, m=3):
        return o != i and o[0] - m <= i[0] and o[1] - m <= i[1] and o[2] + m >= i[2] and o[3] + m >= i[3]
    boxes = [b for b in boxes if not any(contains(b, o) for o in boxes)]
    boxes = _reading_order(boxes, W, H)
    return boxes, clean


def _reading_order(boxes, W, H):
    landscape = W > H * 1.15
    def key(b):
        cx = (b[0] + b[2]) / 2.0
        col = int(cx > W / 2.0) if landscape else 0
        return (col, b[1], b[0])
    return sorted(boxes, key=key)


def analyze_page(bgr: np.ndarray, print_color: Optional[str] = None) -> PageLayout:
    """分析一页答题纸（空白答题纸或学生扫描件均可）。"""
    notes = []
    img, f = resize_long(bgr, WORK_LONG_SIDE)
    T = np.array([[f, 0.0, 0.0], [0.0, f, 0.0]])
    if print_color is None:
        print_color = "red" if red_ink_ratio(img) > 0.35 else "black"
    if print_color == "black":
        img = strip_red(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    skew = estimate_skew(gray)
    if 0.05 < abs(skew) < 5:
        H0, W0 = img.shape[:2]
        R = cv2.getRotationMatrix2D((W0 / 2.0, H0 / 2.0), skew, 1.0)
        T = R @ np.vstack([T, [0, 0, 1.0]])
        img = rotate(img, skew)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        notes.append(f"已纠正倾斜 {skew:.2f}°")
    boxes, clean = find_boxes(gray)
    markers = select_page_markers(detect_markers(gray), boxes)
    if len(markers) < 4:
        notes.append("定位点少于 4 个，扫描件将改用图像特征对齐")
    if not boxes:
        notes.append("没有找到封闭的题框，需要手动框选作答区域")
    return PageLayout(image=img, markers=markers, boxes=boxes, line_mask=clean,
                      print_color=print_color, skew_deg=skew, notes=notes, to_tmpl=T)


# ---------------------------------------------------------------- 标注图

PALETTE = [(230, 90, 30), (40, 160, 40), (200, 60, 180), (0, 140, 230), (30, 30, 200),
           (150, 110, 0), (120, 60, 200), (0, 170, 170), (90, 90, 90), (200, 120, 60)]


def render_regions(image: np.ndarray, regions: List[dict], with_info: bool = False,
                   markers: Optional[list] = None) -> np.ndarray:
    """把区域画到参考图上。with_info=False 时只写编号（给模型看），True 时写题号题型（给老师看）。"""
    out = image.copy()
    overlay = out.copy()
    L = max(out.shape[:2])
    th = max(2, int(L / 500))
    texts = []
    for k, r in enumerate(regions):
        col = PALETTE[k % len(PALETTE)]
        x0, y0, x1, y1 = [int(v) for v in r["rect"]]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), col, -1)
        cv2.rectangle(out, (x0, y0), (x1, y1), col, th)
        label = r["id"]
        if with_info:
            kind = r.get("kind") or "未标注"
            qs = r.get("questions") or []
            if qs:
                label += f" 第{','.join(qs)}题" if len(qs) <= 3 else f" 第{qs[0]}-{qs[-1]}题"
            label += f" {kind}"
            if r.get("full_score"):
                label += f" {r['full_score']}分"
        if with_info:   # 给老师看：标签放左上角
            texts.append({"xy": (x0 + th + 4, y0 + th + 4), "text": label, "size": max(18, int(L / 70)),
                          "color": (255, 255, 255), "bg": col})
        else:           # 给模型看：标签放右下角，避免挡住左上角印刷的题号
            texts.append({"xy": (x1 - th - 6, y1 - th - 6), "text": label, "size": max(22, int(L / 55)),
                          "color": (255, 255, 255), "bg": col, "anchor": "rd"})
    out = cv2.addWeighted(overlay, 0.12, out, 0.88, 0)
    if markers:
        for m in markers:
            x, y, s = m[:3]
            cv2.rectangle(out, (int(x - s), int(y - s)), (int(x + s), int(y + s)), (0, 0, 255), max(1, th // 2))
    return draw_texts(out, texts)


def header_sheet(image: np.ndarray, regions: List[dict], target_text_px: int = 26) -> np.ndarray:
    """把每个区域顶部（含上方一点点）放大拼成一张图，方便模型看清题号和分值。"""
    H, W = image.shape[:2]
    L = max(H, W)
    scale = max(1.0, target_text_px / (0.0068 * L))  # 印刷小四号字约为长边的 0.68%
    rows = []
    for r in regions:
        x0, y0, x1, y1 = [int(v) for v in r["rect"]]
        top = max(0, y0 - int(0.035 * L))
        bot = min(y1, y0 + int(0.06 * L))
        crop = image[top:bot, max(0, x0 - 4):min(W, x1 + 4)]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if crop.shape[1] > 1500:
            crop = crop[:, :1500]
        tag = np.full((crop.shape[0], 90, 3), 255, np.uint8)
        tag = draw_texts(tag, [{"xy": (8, 8), "text": r["id"], "size": 30, "color": (0, 0, 200)}])
        row = np.hstack([tag, crop])
        rows.append(cv2.copyMakeBorder(row, 6, 6, 0, 0, cv2.BORDER_CONSTANT, value=(180, 180, 180)))
    if not rows:
        return np.full((50, 200, 3), 255, np.uint8)
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows]
    return np.vstack(rows)
