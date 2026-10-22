"""把学生扫描件对齐到模板坐标系，并按模板裁切作答区域。

优先用定位点（快、准、能判断倒放），定位点不够时退回到图像特征匹配（SIFT），
两种方法都会用模板上的线条做一次“对齐质量”检查。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .imaging import resize_long, strip_red
from .markers import Marker, detect_markers

DETECT_LONG_SIDE = 2400


@dataclass
class Alignment:
    H: np.ndarray            # 模板坐标 -> 扫描件原图坐标（有 image 时是 -> image 的坐标）
    method: str              # "markers" / "features"，拉平过的再加 "+flatten"
    quality: float           # 0~1，模板线条在扫描件上被“压中”的比例
    matched: int = 0         # 匹配上的定位点或特征点数
    total: int = 0
    rms: float = 0.0         # 定位点重投影误差（模板像素）
    notes: List[str] = field(default_factory=list)
    anchors: List[tuple] = field(default_factory=list, repr=False)          # 对上了的定位点（模板坐标）
    image: Optional[np.ndarray] = field(default=None, repr=False, compare=False)  # 手机照片拉平后的整页图
    gray: Optional[np.ndarray] = field(default=None, repr=False, compare=False)   # 同上，去红笔后的灰度图
    precision: Optional[float] = None   # 照片上的长线条有多少落在模板线条上（挑页用，只在慢路径里算）

    @property
    def ok(self) -> bool:
        return self.quality >= 0.45

    def source(self, scan: np.ndarray) -> np.ndarray:
        """按 H 裁切时该用的图：拉平过的用拉平后的图，否则就是原图。"""
        return self.image if self.image is not None else scan


# ---------------------------------------------------------------- 定位点匹配

def _mutual_nn(P: np.ndarray, D: np.ndarray, tol: float):
    """P[k] 与 D[l] 互为最近邻且距离 < tol 的配对。"""
    d = np.linalg.norm(P[:, None, :] - D[None, :, :], axis=2)
    pairs = []
    for k in range(len(P)):
        l = int(np.argmin(d[k]))
        if d[k, l] < tol and int(np.argmin(d[:, l])) == k:
            pairs.append((k, l, float(d[k, l])))
    return pairs


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    with np.errstate(divide="ignore", invalid="ignore"):   # 退化的候选变换：结果是 inf/nan，后面会被筛掉
        return p[:, :2] / p[:, 2:3]


def _fit_points(src: np.ndarray, dst: np.ndarray, msize_scan: float) -> Optional[np.ndarray]:
    """按点数选模型：≥6 个用 RANSAC 去掉错配再拟合透视变换；5 个直接拟合透视；3–4 个只拟合仿射
    （4 个点拟合透视是恰好定解，配错一个也看不出来，仿射还有余量能平均掉误差）。"""
    n = len(src)
    if n >= 6:
        Hr, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.5 * msize_scan)
        if Hr is not None and int(mask.sum()) >= 5:
            keep = mask.ravel().astype(bool)
            Hl, _ = cv2.findHomography(src[keep], dst[keep], 0)
            return Hl if Hl is not None else Hr
    if n >= 5:
        Hn, _ = cv2.findHomography(src, dst, 0)
        return Hn
    if n == 4:
        A, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
    elif n == 3:
        A = cv2.getAffineTransform(src[:3], dst[:3])
    else:
        return None
    return None if A is None else np.vstack([A, [0, 0, 1]])


# 逐步收紧的配对距离（定位点边长的倍数）。第一轮放宽：手机斜着拍的照片有透视，
# 按“旋转+缩放”猜出来的位置离远处的定位点会差出一个多边长；每轮用更多的点拟合更准的变换，再收紧。
REFINE_TOLS = (2.2, 1.6, 1.2, 0.9, 0.7, 0.6)


def _refine(T, D, Hm, msize_scan, iters=None):
    pairs = []
    for tol in REFINE_TOLS:
        pairs = _mutual_nn(_project(Hm, T), D, msize_scan * tol)
        if len(pairs) < 3:
            return None, []
        src = np.float32([T[k] for k, _, _ in pairs])
        dst = np.float32([D[l] for _, l, _ in pairs])
        Hn = _fit_points(src, dst, msize_scan)
        if Hn is None:
            return None, []
        Hm = Hn
    # 最后一轮：RANSAC 去掉个别错配，再用内点做最小二乘
    pairs = _mutual_nn(_project(Hm, T), D, msize_scan * 0.6)
    if len(pairs) >= 6:
        src = np.float32([T[k] for k, _, _ in pairs])
        dst = np.float32([D[l] for _, l, _ in pairs])
        Hr, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.3 * msize_scan)
        if Hr is not None and int(mask.sum()) >= 5:
            keep = mask.ravel().astype(bool)
            Hl, _ = cv2.findHomography(src[keep], dst[keep], 0)
            Hm = Hl if Hl is not None else Hr
            pairs = [p for p in _mutual_nn(_project(Hm, T), D, msize_scan * 0.6) if p[2] < 0.3 * msize_scan]
    return Hm, pairs


def match_markers(tmpl: Sequence[Sequence[float]], found: Sequence[Marker]):
    """tmpl: [[x,y,size],...]（模板坐标）；found: 扫描件上检测到的候选（检测坐标）。

    返回若干候选解 [(H, pairs, rms)]，按匹配数从多到少排列。H 为 模板->检测坐标。
    """
    # 候选太多时只取最像定位点的 40 个
    found = sorted(found, key=lambda m: (-m.nest, m.isolation))[:40]
    T = np.array([[m[0], m[1]] for m in tmpl], float)
    Ts = np.array([m[2] for m in tmpl], float)
    D = np.array([[m.x, m.y] for m in found], float)
    Ds = np.array([m.size for m in found], float)
    n, mcount = len(T), len(D)
    if n < 3 or mcount < 3:
        return []
    diag = np.hypot(*(T.max(0) - T.min(0)))
    tp = np.array([(a, b) for a in range(n) for b in range(a + 1, n)
                   if np.hypot(*(T[b] - T[a])) > 0.3 * diag], int)
    if len(tp) == 0:
        return []
    TA, TB = T[tp[:, 0]], T[tp[:, 1]]
    dT = TB - TA
    LT = np.hypot(dT[:, 0], dT[:, 1])
    angT = np.arctan2(dT[:, 1], dT[:, 0])
    TsAB = Ts[tp[:, 0]] + Ts[tp[:, 1]]
    tsize = float(np.median(Ts))
    hyps = []
    for i in range(mcount):
        for j in range(mcount):
            if i == j:
                continue
            dD = D[j] - D[i]
            ld = np.hypot(*dD)
            s = ld / LT
            ok = (s / ((Ds[i] + Ds[j]) / TsAB) > 0.65) & (s / ((Ds[i] + Ds[j]) / TsAB) < 1.5)
            if not ok.any():
                continue
            idx = np.nonzero(ok)[0]
            ang = np.arctan2(dD[1], dD[0]) - angT[idx]
            ss = s[idx]
            c, sn = np.cos(ang) * ss, np.sin(ang) * ss
            # P[k] = R_k @ T + t_k,  t_k = D[i] - R_k @ TA_k
            ta = TA[idx]
            tx = D[i][0] - (c * ta[:, 0] - sn * ta[:, 1])
            ty = D[i][1] - (sn * ta[:, 0] + c * ta[:, 1])
            Px = c[:, None] * T[None, :, 0] - sn[:, None] * T[None, :, 1] + tx[:, None]
            Py = sn[:, None] * T[None, :, 0] + c[:, None] * T[None, :, 1] + ty[:, None]
            dist = np.sqrt((Px[:, :, None] - D[None, None, :, 0]) ** 2 +
                           (Py[:, :, None] - D[None, None, :, 1]) ** 2).min(axis=2)
            tol = (1.5 * tsize * ss)[:, None]
            inl = dist < tol
            cnt = inl.sum(axis=1)
            for k in np.nonzero(cnt >= 3)[0]:
                err = float(dist[k][inl[k]].mean()) / (tsize * ss[k])
                R = np.array([[c[k], -sn[k]], [sn[k], c[k]]])
                hyps.append((int(cnt[k]), -err, R, np.array([tx[k], ty[k]]), float(ss[k])))
    if not hyps:
        return []
    hyps.sort(key=lambda h: (h[0], h[1]), reverse=True)
    results, seen = [], []
    for cnt, nerr, R, t, s in hyps[:60]:
        ang = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        key = (round(ang / 3), round(t[0] / (5 * tsize * s)), round(t[1] / (5 * tsize * s)))
        if key in seen:
            continue
        seen.append(key)
        H0 = np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1.0]])
        Hm, pairs = _refine(T, D, H0, tsize * s)
        if Hm is None or len(pairs) < 3:
            continue
        P = _project(Hm, T)
        rms_scan = float(np.sqrt(np.mean([np.sum((P[k] - D[l]) ** 2) for k, l, _ in pairs])))
        results.append((Hm, pairs, rms_scan / s))
        if len(results) >= 6:
            break
    results.sort(key=lambda r: (len(r[1]), -r[2]), reverse=True)
    return results


# ---------------------------------------------------------------- 特征匹配（无定位点时）

def match_features(ref_gray: np.ndarray, scan_gray: np.ndarray):
    """SIFT + RANSAC。ref_gray 是模板参考图，scan_gray 是检测尺度的扫描件。返回 (H, inliers)。"""
    ref_s, f1 = resize_long(ref_gray, 1600)
    scn_s, f2 = resize_long(scan_gray, 1600)
    sift = cv2.SIFT_create(nfeatures=5000)
    k1, d1 = sift.detectAndCompute(ref_s, None)
    k2, d2 = sift.detectAndCompute(scn_s, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None, 0
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 15:
        return None, len(good)
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    Hs, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.999)
    if Hs is None:
        return None, 0
    S1 = np.diag([f1, f1, 1.0])
    S2inv = np.diag([1 / f2, 1 / f2, 1.0])
    return S2inv @ Hs @ S1, int(mask.sum())


# ---------------------------------------------------------------- 质量检查

def _warp_to_template(scan_gray: np.ndarray, H_t2s: np.ndarray, size, scale: float = 0.5) -> np.ndarray:
    W, Ht = size
    S = np.diag([1 / scale, 1 / scale, 1.0])
    M = H_t2s @ S
    return cv2.warpPerspective(scan_gray, M, (int(W * scale), int(Ht * scale)),
                               flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP, borderValue=255)


def alignment_quality(scan_gray: np.ndarray, H_t2s: np.ndarray, line_mask: np.ndarray) -> float:
    """模板上的线条，有多少落在扫描件的黑色像素上（允许 1 像素误差）。"""
    h, w = line_mask.shape
    scale = 0.5
    warped = _warp_to_template(scan_gray, H_t2s, (w, h), scale)
    L = max(warped.shape)
    bw = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, int(L / 60) | 1, 10)
    bw = cv2.dilate(bw, np.ones((3, 3), np.uint8))
    lm = cv2.resize(line_mask, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_AREA) > 127
    lm = cv2.erode(lm.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    n = int(lm.sum())
    if n == 0:
        return 0.0
    return float((bw[lm] > 0).sum()) / n


def _long_lines(bw: np.ndarray, k: int):
    """二值图里的长横线、长竖线（框线、填空横线；手写的字基本都会被去掉）。"""
    h = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    v = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    return h, v


def _ink(gray: np.ndarray) -> np.ndarray:
    """细的深色笔画、线条（二值）。用黑帽变换：照片边上露出来的大片深色桌面、阴影不算，光线不匀也不怕。"""
    L = max(gray.shape[:2])
    k = max(9, int(L / 150)) | 1                       # 长边 2000 时 13 像素：比线宽粗得多，比桌面小得多
    bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    return cv2.threshold(bh, 25, 255, cv2.THRESH_BINARY)[1]


def line_precision(scan_gray: np.ndarray, H_t2s: np.ndarray, line_mask: np.ndarray) -> float:
    """反过来查：扫描件上的长直线（框线、横线）有多少落在模板线条附近。
    答题纸正反面的框有不少是重合的，照片拉平之后对错了页，模板线条也能“压中”一大半；
    但照片上多出来的那些线（另一面才有的横线、框）这一页解释不了，这个比例就低。"""
    Hh, W = line_mask.shape[:2]
    g = _warp_to_template(scan_gray, H_t2s, (W, Hh), 1.0)
    L = max(W, Hh)
    h, v = _long_lines(_ink(g), max(15, int(0.03 * L)))
    lines = (h | v) > 0
    m = int(0.03 * L)                          # 纸边、定位点附近不算
    lines[:m] = False
    lines[-m:] = False
    lines[:, :m] = False
    lines[:, -m:] = False
    n = int(lines.sum())
    if n == 0:
        return 0.0
    near = cv2.dilate((line_mask > 127).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    return float((lines & near).sum()) / n


def _content_score(scan_gray, H_t2s, ref_gray) -> float:
    h, w = ref_gray.shape
    warped = _warp_to_template(scan_gray, H_t2s, (w, h), 0.25)
    ref = cv2.resize(ref_gray, (warped.shape[1], warped.shape[0]), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(warped.astype(np.float32), (5, 5), 0)
    b = cv2.GaussianBlur(ref.astype(np.float32), (5, 5), 0)
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1.0
    return float((a * b).sum()) / den


# ---------------------------------------------------------------- 总入口

class ScanPrep:
    """同一张扫描件要和答题纸的每一页都比一比：缩小、去红笔、找定位点、定位点配对只做一次。"""

    def __init__(self, scan_bgr: np.ndarray):
        self.scan = scan_bgr
        self._gray: dict = {}
        self._full: dict = {}
        self._found: dict = {}
        self._sols: dict = {}

    def full_gray(self, drop_red: bool) -> np.ndarray:
        """原图大小、去掉红笔的灰度图（拉平照片时用，算一次）。"""
        if drop_red not in self._full:
            self._full[drop_red] = cv2.cvtColor(strip_red(self.scan) if drop_red else self.scan, cv2.COLOR_BGR2GRAY)
        return self._full[drop_red]

    def gray(self, drop_red: bool):
        if drop_red not in self._gray:
            small, f = resize_long(self.scan, min(DETECT_LONG_SIDE, max(self.scan.shape[:2])))
            if drop_red:
                small = strip_red(small)
            self._gray[drop_red] = (cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), f)
        return self._gray[drop_red]

    def marker_solutions(self, drop_red: bool, tm):
        key = (drop_red, tuple(tuple(round(float(v), 2) for v in m[:3]) for m in tm))
        if key not in self._sols:
            if drop_red not in self._found:
                self._found[drop_red] = detect_markers(self.gray(drop_red)[0])
            self._sols[key] = match_markers(tm, self._found[drop_red])
        return self._sols[key]


def align_page(scan_bgr: np.ndarray, page_tmpl: dict, ref_gray: np.ndarray, line_mask: np.ndarray,
               prep: Optional[ScanPrep] = None, features: bool = True) -> Optional[Alignment]:
    """page_tmpl: 模板里的一页（含 markers / size / print_color）。
    prep: 同一张扫描件的公共预处理（和多页模板比较时传同一个）；features=False 时不做 SIFT 兜底。"""
    drop_red = page_tmpl.get("print_color", "black") == "black"
    prep = prep if prep is not None and prep.scan is scan_bgr else ScanPrep(scan_bgr)
    gray, f = prep.gray(drop_red)
    to_full = np.diag([1 / f, 1 / f, 1.0])  # 检测坐标 -> 原图坐标
    candidates: List[Alignment] = []

    tm = page_tmpl.get("markers") or []
    if len(tm) >= 4:
        sols = prep.marker_solutions(drop_red, tm)
        for k, (Hm, pairs, rms) in enumerate(sols[:4]):
            Hf = to_full @ Hm
            q = alignment_quality(gray, Hm, line_mask)
            al = Alignment(Hf, "markers", q, matched=len(pairs), total=len(tm), rms=rms,
                           anchors=[(float(tm[i][0]), float(tm[i][1])) for i, _, _ in pairs])
            candidates.append(al)
        # 定位点对称导致的方向歧义：匹配数相同的候选用整体内容相关性来裁决
        if len(candidates) >= 2 and candidates[0].matched == candidates[1].matched:
            scores = [_content_score(gray, np.linalg.inv(to_full) @ c.H, ref_gray) for c in candidates
                      if c.matched == candidates[0].matched]
            best = int(np.argmax(scores))
            candidates.insert(0, candidates.pop(best))
        if candidates:
            top = candidates[0]
            if top.matched < max(4, int(0.5 * len(tm))):
                top.notes.append(f"只匹配上 {top.matched}/{len(tm)} 个定位点")

    use_features = features and (not candidates or candidates[0].quality < 0.6 or candidates[0].matched < 4)
    if use_features:
        Hs, inl = match_features(ref_gray, gray)
        if Hs is not None:
            q = alignment_quality(gray, Hs, line_mask)
            candidates.append(Alignment(to_full @ Hs, "features", q, matched=inl))
    if not candidates:
        return None
    best = candidates[0] if not use_features else max(candidates, key=lambda c: c.quality)
    return best


# ---------------------------------------------------------------- 手机照片：把弯曲的纸面拉平

# 每一轮：(块边长, 步长, 搜索范围, 位移场多项式的阶数)，单位是长边 2000 的模板像素（A3 约 4.8 像素/毫米）
FLATTEN_PASSES = ((300, 150, 80, 3), (200, 100, 30, 4), (160, 80, 14, 4))
FLATTEN_MAX = 0.07       # 位移超过长边的 7%（A3 约 3 厘米）多半是对错了页，不要


def _best_shift(t: np.ndarray, prof: np.ndarray, R: int):
    """模板的投影 t（长 n）在照片的投影 prof（长 n+2R）上滑动，取归一化相关最大的位移。
    返回 (位移, 峰值, 峰值比离它 6 像素以外的次峰高多少)。"""
    n = len(t)
    win = np.lib.stride_tricks.sliding_window_view(prof, n)          # (2R+1, n)
    tc = t - t.mean()
    wc = win - win.mean(axis=1, keepdims=True)
    den = np.sqrt(float((tc * tc).sum()) * (wc * wc).sum(axis=1))
    sc = np.where(den > 1e-9, (wc @ tc) / np.maximum(den, 1e-9), -1.0)
    k = int(np.argmax(sc))
    far = np.abs(np.arange(len(sc)) - k) >= 6
    second = float(sc[far].max()) if far.any() else -1.0
    sub = 0.0
    if 0 < k < len(sc) - 1:                       # 抛物线插值到亚像素
        a, b, c = float(sc[k - 1]), float(sc[k]), float(sc[k + 1])
        dd = a - 2 * b + c
        if dd < 0:
            sub = float(np.clip(0.5 * (a - c) / dd, -0.5, 0.5))
    return k - R + sub, float(sc[k]), float(sc[k]) - second


def _measure_shifts(warped_gray: np.ndarray, line_mask: np.ndarray, tile: int, step: int, R: int, k_line: int) -> dict:
    """逐块比较：模板上的横线在照片里上下偏了多少（dy），竖线左右偏了多少（dx）。
    横线只能定上下、竖线只能定左右，所以分开量。返回 {"dx"/"dy": [[x, y, 位移, 权重], ...]}（模板坐标）。"""
    Hh, W = line_mask.shape[:2]
    ih, iv = _long_lines(_ink(warped_gray), k_line)
    th, tv = _long_lines((line_mask > 127).astype(np.uint8) * 255, k_line)

    def blur(a):
        return cv2.GaussianBlur(a.astype(np.float32), (0, 0), 1.5)
    ih, iv, th, tv = blur(ih), blur(iv), blur(th), blur(tv)
    out = {"dx": [], "dy": []}
    for y0 in range(0, max(1, Hh - tile // 2), step):
        for x0 in range(0, max(1, W - tile // 2), step):
            y1, x1 = min(Hh, y0 + tile), min(W, x0 + tile)
            t = th[y0:y1, x0:x1].sum(1)
            if t.sum() > 255 * 1.5 * (x1 - x0):          # 块里至少有半条横线（线宽约 3 像素）
                prof = np.zeros(y1 - y0 + 2 * R, np.float32)
                a, b = max(0, y0 - R), min(Hh, y1 + R)
                prof[a - (y0 - R): b - (y0 - R)] = ih[a:b, x0:x1].sum(1)
                s, pk, gap = _best_shift(t, prof, R)
                if pk > 0.5 and gap > 0.08:
                    out["dy"].append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, s, pk * min(1.0, 4 * gap)))
            t = tv[y0:y1, x0:x1].sum(0)
            if t.sum() > 255 * 1.5 * (y1 - y0):
                prof = np.zeros(x1 - x0 + 2 * R, np.float32)
                a, b = max(0, x0 - R), min(W, x1 + R)
                prof[a - (x0 - R): b - (x0 - R)] = iv[y0:y1, a:b].sum(0)
                s, pk, gap = _best_shift(t, prof, R)
                if pk > 0.5 and gap > 0.08:
                    out["dx"].append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, s, pk * min(1.0, 4 * gap)))
    return {k: np.array(v, float).reshape(-1, 4) for k, v in out.items()}


def _poly(u, v, deg: int) -> np.ndarray:
    return np.stack([u ** i * v ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


def _fit_field(s: np.ndarray, size, deg: int) -> Optional[dict]:
    """位移样本 -> 平滑的多项式位移场。稳健拟合（去掉离群的块）；样本少时自动降阶，太少返回 None。"""
    W, Hh = size
    while deg > 1 and len(s) < 2 * (deg + 1) * (deg + 2) // 2:
        deg -= 1
    if len(s) < 8:
        return None
    x, y, d, w = s.T
    M = _poly(x / W - 0.5, y / Hh - 0.5, deg)
    keep = np.ones(len(d), bool)
    coef = None
    for _ in range(4):
        if keep.sum() < M.shape[1] + 3:
            break
        sw = np.sqrt(w[keep])
        coef = np.linalg.lstsq(M[keep] * sw[:, None], d[keep] * sw, rcond=None)[0]
        res = d - M @ coef
        sig = 1.4826 * float(np.median(np.abs(res[keep]))) + 0.5
        keep = np.abs(res) < 3 * sig
    if coef is None or not keep.any():
        return None
    # 多项式在没有线条的纸边外推会乱飞：限制在实测位移的范围里
    return {"coef": coef, "deg": deg, "lo": float(d[keep].min()) - 3.0, "hi": float(d[keep].max()) + 3.0}


def _eval_field(f: Optional[dict], x, y, size):
    if f is None:
        return np.zeros(np.shape(x))
    W, Hh = size
    return np.clip(_poly(np.asarray(x) / W - 0.5, np.asarray(y) / Hh - 0.5, f["deg"]) @ f["coef"], f["lo"], f["hi"])


def _flatten_maps(H_t2s: np.ndarray, size, k: float, fx, fy, grid: int = 8):
    """输出图的像素 q（模板坐标 × k）-> 模板坐标 p = q/k -> 加上位移场 -> H -> 原图坐标。
    先在每 grid 像素一个点的粗网格上算，再双线性放大（位移场和透视都很平滑，误差远小于 1 像素）。"""
    W, Hh = size
    ow, oh = int(round(W * k)), int(round(Hh * k))
    nx, ny = ow // grid + 2, oh // grid + 2
    # cv2.resize 放大 grid 倍时，粗网格第 j 格落在输出像素 (j+0.5)*grid-0.5 上
    gx = ((np.arange(nx) + 0.5) * grid - 0.5) / k
    gy = ((np.arange(ny) + 0.5) * grid - 0.5) / k
    X, Y = np.meshgrid(gx, gy)
    px = X + _eval_field(fx, X, Y, size)
    py = Y + _eval_field(fy, X, Y, size)
    P = np.stack([px.ravel(), py.ravel(), np.ones(px.size)], 1) @ H_t2s.T
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = (P[:, 0] / P[:, 2]).reshape(X.shape).astype(np.float32)
        sy = (P[:, 1] / P[:, 2]).reshape(X.shape).astype(np.float32)
    mx = cv2.resize(sx, (nx * grid, ny * grid), interpolation=cv2.INTER_LINEAR)[:oh, :ow]
    my = cv2.resize(sy, (nx * grid, ny * grid), interpolation=cv2.INTER_LINEAR)[:oh, :ow]
    return mx, my


def flatten(scan_bgr: np.ndarray, al: Alignment, page_tmpl: dict, line_mask: np.ndarray,
            prep: Optional[ScanPrep] = None) -> Optional[Alignment]:
    """手机拍的答题纸：纸不平（中间鼓、边上翘）再加透视，一个透视变换对不准，越往纸边偏得越多。
    以已有的对齐（定位点或特征点）为起点，逐块量出模板上的横线、竖线在照片里偏了多少，拟合成平滑的位移场，
    把照片“拉平”成模板坐标下的整页图。返回新的 Alignment：image 是拉平后的图，H 是 模板 -> 这张图 的缩放。"""
    W, Hh = [int(v) for v in page_tmpl["size"]]
    L = max(W, Hh)
    f = L / 2000.0
    drop_red = page_tmpl.get("print_color", "black") == "black"
    prep = prep if prep is not None and prep.scan is scan_bgr else ScanPrep(scan_bgr)
    gray = prep.full_gray(drop_red)
    lm = line_mask if line_mask.shape[:2] == (Hh, W) else cv2.resize(line_mask, (W, Hh), interpolation=cv2.INTER_NEAREST)
    H = al.H
    anchors = np.array(al.anchors, float).reshape(-1, 2)
    fields = {"dx": None, "dy": None}
    k_line = max(15, int(0.02 * L))
    for tile, step, R, deg in FLATTEN_PASSES:
        tile, step, R = int(tile * f), max(1, int(step * f)), max(4, int(R * f))
        if fields["dx"] is None and fields["dy"] is None:
            warped = _warp_to_template(gray, H, (W, Hh), 1.0)
        else:
            mx, my = _flatten_maps(H, (W, Hh), 1.0, fields["dx"], fields["dy"])
            warped = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        shifts = _measure_shifts(warped, lm, tile, step, R, k_line)
        for key in ("dx", "dy"):
            s = shifts[key]
            if len(s) < 8:
                continue
            x, y, d, w = s.T
            d = d + _eval_field(fields[key], x, y, (W, Hh))       # 累加：这一轮量的是在上一轮基础上还差多少
            if len(anchors):                                       # 对上了的定位点处本来就准，钉住纸边
                x, y = np.r_[x, anchors[:, 0]], np.r_[y, anchors[:, 1]]
                d, w = np.r_[d, np.zeros(len(anchors))], np.r_[w, np.full(len(anchors), 0.5)]
            got = _fit_field(np.stack([x, y, d, w], 1), (W, Hh), deg)
            if got is not None:
                fields[key] = got
    if fields["dx"] is None and fields["dy"] is None:
        return None
    gx, gy = np.meshgrid(np.linspace(0, W, 21), np.linspace(0, Hh, 15))
    mag = float(np.hypot(_eval_field(fields["dx"], gx, gy, (W, Hh)), _eval_field(fields["dy"], gx, gy, (W, Hh))).max())
    if mag > FLATTEN_MAX * L:
        return None
    k = float(np.clip(local_scale(H, W / 2.0, Hh / 2.0), 1.0, 2.5))   # 拉平后的图保持照片原来的清晰度
    mx, my = _flatten_maps(H, (W, Hh), k, fields["dx"], fields["dy"])
    flat = cv2.remap(scan_bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    fg = cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    Hk = np.diag([k, k, 1.0])
    q = alignment_quality(fg, Hk, line_mask)
    return Alignment(Hk, al.method + "+flatten", q, matched=al.matched, total=al.total, rms=al.rms,
                     notes=list(al.notes) + ["照片：已按印刷的线条把纸面拉平"],
                     anchors=list(al.anchors), image=flat, gray=fg)


# ---------------------------------------------------------------- 裁切

def local_scale(H: np.ndarray, x: float, y: float) -> float:
    """模板坐标 (x,y) 处，1 个模板像素对应扫描件上多少像素。"""
    p = np.array([x, y, 1.0])
    w = H[2] @ p
    J = (H[:2, :2] * w - np.outer(H[:2] @ p, H[2, :2])) / (w * w)
    return float(np.sqrt(abs(np.linalg.det(J))))


def crop_region(scan_bgr: np.ndarray, H_t2s: np.ndarray, rect, pad: float = 0.0,
                max_side: int = 2000, drop_red: bool = True) -> np.ndarray:
    """按模板坐标的矩形，从扫描原图上以原始分辨率裁出（同时完成透视校正）。"""
    x0, y0, x1, y1 = [float(v) for v in rect]
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    w_t, h_t = x1 - x0, y1 - y0
    s = local_scale(H_t2s, (x0 + x1) / 2, (y0 + y1) / 2)
    s = min(s, max_side / max(w_t, h_t))
    s = max(s, 0.3)
    out_w, out_h = max(1, int(round(w_t * s))), max(1, int(round(h_t * s)))
    S = np.array([[1 / s, 0, x0], [0, 1 / s, y0], [0, 0, 1.0]])
    M = H_t2s @ S
    out = cv2.warpPerspective(scan_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return strip_red(out) if drop_red else out


def warp_page(scan_bgr: np.ndarray, H_t2s: np.ndarray, size, scale: float = 1.0) -> np.ndarray:
    """把整页扫描件变换到模板坐标系（用于预览）。"""
    W, Ht = size
    S = np.diag([1 / scale, 1 / scale, 1.0])
    return cv2.warpPerspective(scan_bgr, H_t2s @ S, (int(W * scale), int(Ht * scale)),
                               flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP, borderValue=(255, 255, 255))
