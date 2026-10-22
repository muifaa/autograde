"""把一页答题纸图像做成“手机拍的照片”：斜着拍（透视）、纸面中间鼓起（弯曲）、光线不匀、纸发灰、
周围露出桌面、轻微模糊和噪点，再像 Mac 那样转成“1 像素 = 1 点”的 PDF（页面 1.4 米宽）。"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np


def phone_photo(page: np.ndarray, rng: np.random.Generator, out_size=(4032, 3024), bend: float = 0.012,
                tilt: float = 0.035, rotate: float = 2.0, coverage: float = 0.86) -> np.ndarray:
    """page：答题纸图像（BGR，横版或竖版都行）。bend：纸面鼓起的高度（占纸长边的比例）；
    tilt：透视的程度（远处一边缩短的比例）；coverage：纸占照片宽（或高）的比例。"""
    W, H = out_size
    ph, pw = page.shape[:2]
    if (pw > ph) != (W > H):
        W, H = H, W
    # 纸在照片里的四个角：先居中缩放，再加透视（上边窄、下边宽）和一点旋转
    s = coverage * min(W / pw, H / ph)
    cw, chh = pw * s, ph * s
    cx, cy = W / 2 + rng.uniform(-0.02, 0.02) * W, H / 2 + rng.uniform(-0.02, 0.02) * H
    t = tilt * cw
    dst = np.float32([[cx - cw / 2 + t, cy - chh / 2], [cx + cw / 2 - t, cy - chh / 2],
                      [cx + cw / 2, cy + chh / 2], [cx - cw / 2, cy + chh / 2]])
    R = cv2.getRotationMatrix2D((cx, cy), rotate, 1.0)
    dst = (np.hstack([dst, np.ones((4, 1), np.float32)]) @ R.T).astype(np.float32)
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    Hm = cv2.getPerspectiveTransform(src, dst)
    Hinv = np.linalg.inv(Hm)
    # 每个照片像素 -> 纸上坐标：先去掉透视，再去掉弯曲（纸横向中间鼓起，竖线跟着弯）
    X, Y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    P = np.stack([X.ravel(), Y.ravel(), np.ones(X.size, np.float32)], 1) @ Hinv.T.astype(np.float32)
    u = (P[:, 0] / P[:, 2]).reshape(H, W)
    v = (P[:, 1] / P[:, 2]).reshape(H, W)
    amp = bend * max(pw, ph)
    v = v - amp * np.sin(np.pi * np.clip(u / pw, 0, 1)) * (1 - 0.6 * (v / ph))     # 上边鼓得多、下边少
    u = u - 0.4 * amp * np.sin(np.pi * np.clip(v / ph, 0, 1)) * (u / pw - 0.5) * 2
    paper = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) if page.ndim == 3 else page
    paper = (paper.astype(np.float32) * (215 / 255.0)).astype(np.uint8)           # 照片里的“白纸”是灰的
    paper = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)
    inside = (u >= 0) & (u < pw - 1) & (v >= 0) & (v < ph - 1)
    shot = cv2.remap(paper, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    desk = np.zeros((H, W, 3), np.float32)
    desk[:] = (95, 105, 120)
    desk += rng.normal(0, 6, (H, W, 1)).astype(np.float32)
    out = np.where(inside[:, :, None], shot.astype(np.float32), desk)
    # 光线不匀：一边亮一边暗，再加一点暗角
    gx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
    gy = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
    light = 1.0 - 0.13 * (gx * rng.uniform(0.5, 1.0) + gy * rng.uniform(-0.5, 0.5)) - 0.10 * (gx ** 2 + gy ** 2)
    out = out * light[:, :, None]
    out = cv2.GaussianBlur(out, (0, 0), 0.9)
    out += rng.normal(0, 3, out.shape[:2])[:, :, None].astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(buf, 1)


def photos_to_pdf(path: Path, imgs: Sequence[np.ndarray]) -> None:
    """像 Mac 的“预览”导出那样：页面尺寸 = 照片像素数（单位当成点），4032×3024 的照片是 1.4 米宽的一页。"""
    import pymupdf
    doc = pymupdf.open()
    for im in imgs:
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        h, w = im.shape[:2]
        page = doc.new_page(width=w, height=h)
        page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(str(path))


def load_all(paths: List[Path]) -> List[np.ndarray]:
    return [cv2.imread(str(p)) for p in paths]
