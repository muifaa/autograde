"""定位点检测：答题卡上常见的“回”字形定位块（也兼容实心方块）。"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

import cv2
import numpy as np


@dataclass
class Marker:
    x: float
    y: float
    size: float
    nest: int          # 2 = 回字形（外框-白环-内芯），1 = 空心方框，0 = 实心方块
    isolation: float   # 周围一圈的黑色像素比例，二维码里的“回”字通常很高

    def to_list(self) -> list:
        return [round(self.x, 2), round(self.y, 2), round(self.size, 2)]

    def as_dict(self) -> dict:
        return asdict(self)


def _depths(hier: np.ndarray) -> np.ndarray:
    n = len(hier)
    depth = np.full(n, -1, dtype=int)
    for i in range(n):
        d, j = 0, hier[i][3]
        while j != -1:
            d += 1
            j = hier[j][3]
        depth[i] = d
    return depth


def _candidates(bw: np.ndarray, min_px: float, max_px: float) -> List[Marker]:
    cnts, hier = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    depth = _depths(hier)
    found = []
    Hh, Ww = bw.shape[:2]
    for i, c in enumerate(cnts):
        if depth[i] % 2:          # 奇数层是“洞”，不是黑块的外轮廓
            continue
        x, y, w, h = cv2.boundingRect(c)
        if x <= 1 or y <= 1 or x + w >= Ww - 1 or y + h >= Hh - 1:
            continue              # 被扫描边缘切掉一部分的，位置不准，不要
        ar = w / float(h)
        if 1.75 < ar < 2.35 or 1.75 < 1 / ar < 2.35:
            # 两个紧挨着的定位点（常用来标方向）在模糊的扫描件上会粘成一个长方块：拆成两个
            if cv2.contourArea(c) / float(w * h) > 0.75:
                q = min(w, h)
                if min_px <= q <= max_px:
                    if w > h:
                        found += [Marker(x + q / 2.0, y + h / 2.0, q, 0, 0.0), Marker(x + w - q / 2.0, y + h / 2.0, q, 0, 0.0)]
                    else:
                        found += [Marker(x + w / 2.0, y + q / 2.0, q, 0, 0.0), Marker(x + w / 2.0, y + h - q / 2.0, q, 0, 0.0)]
            continue
        s = (w + h) / 2.0
        if s < min_px or s > max_px or not (0.75 < ar < 1.33):
            continue
        area = cv2.contourArea(c)
        if area / float(w * h) < 0.80:
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        nest = 0
        child = hier[i][2]
        while child != -1:  # 找居中的洞
            cxr, cyr, cw, ch = cv2.boundingRect(cnts[child])
            if (abs(cxr + cw / 2.0 - cx) < 0.2 * s and abs(cyr + ch / 2.0 - cy) < 0.2 * s
                    and 0.25 < cw / float(w) < 0.9):
                nest = 1
                g = hier[child][2]
                while g != -1:  # 洞里再找居中的芯
                    gx, gy, gw, gh = cv2.boundingRect(cnts[g])
                    if (abs(gx + gw / 2.0 - cx) < 0.2 * s and abs(gy + gh / 2.0 - cy) < 0.2 * s
                            and 0.12 < gw / float(w) < 0.7):
                        nest = 2
                        break
                    g = hier[g][0]
                break
            child = hier[child][0]
        if nest == 0:
            # 实心块：要求内部足够黑
            inner = bw[y + h // 4: y + 3 * h // 4 + 1, x + w // 4: x + 3 * w // 4 + 1]
            if inner.size == 0 or inner.mean() < 0.85 * 255:
                continue
        found.append(Marker(cx, cy, s, nest, 0.0))
    return found


def _isolation(bw: np.ndarray, m: Marker) -> float:
    H, W = bw.shape
    s = m.size
    x0, x1 = int(max(0, m.x - 2 * s)), int(min(W, m.x + 2 * s))
    y0, y1 = int(max(0, m.y - 2 * s)), int(min(H, m.y + 2 * s))
    win = bw[y0:y1, x0:x1] > 0
    if win.size == 0:
        return 1.0
    keep = np.ones_like(win)
    ix0, ix1 = int(m.x - 0.75 * s) - x0, int(m.x + 0.75 * s) - x0
    iy0, iy1 = int(m.y - 0.75 * s) - y0, int(m.y + 0.75 * s) - y0
    keep[max(0, iy0):max(0, iy1), max(0, ix0):max(0, ix1)] = False
    ring = win[keep]
    return float(ring.mean()) if ring.size else 1.0


def detect_markers(gray: np.ndarray, min_frac: float = 0.004, max_frac: float = 0.03) -> List[Marker]:
    """检测所有像定位点的方块（会包含一些二维码里的“回”字，交给后续匹配去筛）。"""
    H, W = gray.shape[:2]
    L = float(max(H, W))
    min_px, max_px = max(6.0, min_frac * L), max_frac * L
    thr, bw1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = float(np.percentile(gray, 0.5))
    # 偏暗的阈值：模糊的扫描件里“回”字的白环会发灰，Otsu 容易把它也算成黑的
    _, bw_dark = cv2.threshold(gray, (thr + ink) / 2.0, 255, cv2.THRESH_BINARY_INV)
    blk = int(L / 40) | 1
    bw_ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blk, 20)
    # 去掉长直线（题框边线）：一是计算“周围是否空白”，二是把贴着边线的定位点分离出来
    ll = max(15, int(0.03 * L))
    lines = cv2.morphologyEx(bw1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (ll, 1))) | \
        cv2.morphologyEx(bw1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, ll)))
    not_lines = cv2.bitwise_not(cv2.dilate(lines, np.ones((3, 3), np.uint8)))
    bw_iso = cv2.bitwise_and(bw_dark, not_lines)
    bw_nl = cv2.bitwise_and(bw1, not_lines)
    merged: List[Marker] = []
    for bw in (bw_dark, bw1, bw_nl, bw_ad):
        for m in _candidates(bw, min_px, max_px):
            dup = next((o for o in merged if abs(o.x - m.x) < 0.5 * o.size and abs(o.y - m.y) < 0.5 * o.size), None)
            if dup is None:
                m.isolation = _isolation(bw_iso, m)
                merged.append(m)
            elif m.nest > dup.nest:
                dup.nest = m.nest
    return merged


def select_page_markers(markers: List[Marker], boxes=None) -> List[Marker]:
    """建模板时用：只保留像“页面定位点”的那些（不在题框内、周围空白、大小一致、结构一致）。"""
    if boxes:
        def inside(m):
            return any(b[0] - 2 <= m.x <= b[2] + 2 and b[1] - 2 <= m.y <= b[3] + 2 for b in boxes)
        markers = [m for m in markers if not inside(m)]
    if not markers:
        return []
    best_nest = max(m.nest for m in markers)
    min_nest = 1 if best_nest >= 1 else 0
    ms = [m for m in markers if m.nest >= min_nest and m.isolation < 0.12]
    if len(ms) < 3:
        ms = [m for m in markers if m.isolation < 0.12]
    if not ms:
        return []
    med = float(np.median([m.size for m in ms]))
    return [m for m in ms if 0.7 * med < m.size < 1.35 * med]
