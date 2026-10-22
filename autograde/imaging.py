"""读取图片 / PDF，以及一些通用的图像处理工具。"""
from __future__ import annotations

import base64
import glob
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageSequence

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
try:                                   # iPhone 拍的 HEIC 照片（装了 pillow-heif 才能读）
    import pillow_heif
    pillow_heif.register_heif_opener()
    IMAGE_EXTS |= {".heic", ".heif"}
except Exception:                      # pragma: no cover - 没装就不支持 HEIC
    pass
PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = IMAGE_EXTS | PDF_EXTS

Image.MAX_IMAGE_PIXELS = 400_000_000  # 600dpi 的 A3 扫描件也能打开
MAX_SIDE = 7000                       # 读进来的一页最长边（A3 400dpi 约 6600 像素）；再大只是更慢、更占内存
HUGE_PAGE_PT = 1700                   # PDF 页面长边超过这么多点（60 厘米）：多半是手机照片转的 PDF


# ---------------------------------------------------------------- 读取

def load_pages(path: str | os.PathLike, dpi: int = 250) -> List[np.ndarray]:
    """把一个文件读成若干页 BGR 图像（PDF 每页一张，多页 TIFF 每帧一张）。"""
    p = Path(path)
    if p.suffix.lower() in PDF_EXTS:
        return _load_pdf(p, dpi)
    return _load_image_file(p)


def load_page(path: str | os.PathLike, index: int, dpi: int = 250) -> np.ndarray:
    """只读取文件里的第 index 页（PDF 不会把整份都渲染一遍）。"""
    p = Path(path)
    if p.suffix.lower() in PDF_EXTS:
        with _pymupdf().open(str(p)) as doc:
            return _render(doc[index], dpi)
    with Image.open(p) as im:
        im.seek(index)
        fr = im.copy()
        try:
            fr = ImageOps.exif_transpose(fr)
        except Exception:
            pass
        return limit_long(_pil_to_bgr(fr), MAX_SIDE)


def _pymupdf():
    try:
        import pymupdf  # PyMuPDF >= 1.24
    except ImportError:  # pragma: no cover - 旧版本
        import fitz as pymupdf  # type: ignore
    return pymupdf


def _native_zoom(page) -> Optional[float]:
    """整页就是一张图（扫描件、照片转的 PDF）时，按这张图原本的分辨率渲染要放大几倍。"""
    try:
        infos = page.get_image_info()
    except Exception:
        return None
    area = float(page.rect.width * page.rect.height) or 1.0
    for info in infos:
        x0, y0, x1, y1 = info.get("bbox") or (0, 0, 0, 0)
        if (x1 - x0) * (y1 - y0) >= 0.8 * area and info.get("width") and x1 > x0 and y1 > y0:
            return max(info["width"] / (x1 - x0), info["height"] / (y1 - y0))
    return None


def render_zoom(page, dpi: int) -> float:
    """PDF 一页按多大倍数渲染。手机照片在 Mac 上转成的 PDF，页面尺寸是按“1 像素 = 1 点”算的
    （4032×3024 的照片变成 1.4 米宽的一页），按 250dpi 渲染会放大成 1.4 亿像素、占好几 GB 内存；
    这种页按照片本来的分辨率渲染。所有页的长边都不超过 MAX_SIDE。"""
    zoom = dpi / 72.0
    long_pt = float(max(page.rect.width, page.rect.height)) or 1.0
    if long_pt > HUGE_PAGE_PT:
        native = _native_zoom(page)
        zoom = min(zoom, native if native else 1.0)
    return min(zoom, MAX_SIDE / long_pt)


def _render(page, dpi: int) -> np.ndarray:
    z = render_zoom(page, dpi)
    pix = page.get_pixmap(matrix=_pymupdf().Matrix(z, z), alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR))


def _load_pdf(p: Path, dpi: int) -> List[np.ndarray]:
    with _pymupdf().open(str(p)) as doc:
        return [_render(page, dpi) for page in doc]


def count_pages(path: str | os.PathLike) -> int:
    p = Path(path)
    if p.suffix.lower() in PDF_EXTS:
        with _pymupdf().open(str(p)) as doc:
            return doc.page_count
    try:
        with Image.open(p) as im:
            return getattr(im, "n_frames", 1)
    except Exception:
        return 1


def iter_pages(path: str | os.PathLike, dpi: int = 250) -> Iterator[Tuple[int, Optional[np.ndarray], str]]:
    """逐页读取：一次只把一页放进内存（扫描仪导出的几百页、几百 MB 的 PDF 也不会占满内存）。

    产出 (页序号, BGR 图像, "")；某一页读不出来时产出 (页序号, None, 错误信息)，不影响后面的页。"""
    p = Path(path)
    if p.suffix.lower() in PDF_EXTS:
        pymupdf = _pymupdf()
        with pymupdf.open(str(p)) as doc:
            for i in range(doc.page_count):
                try:
                    img = _render(doc[i], dpi)
                except Exception as e:  # 个别页损坏
                    yield i, None, f"{type(e).__name__}: {e}"
                    continue
                try:
                    pymupdf.TOOLS.store_shrink(100)   # 释放 MuPDF 缓存的已解码图片
                except Exception:
                    pass
                yield i, img, ""
        return
    try:
        with Image.open(p) as im:
            for i, frame in enumerate(ImageSequence.Iterator(im)):
                try:
                    fr = frame.copy()
                    try:
                        fr = ImageOps.exif_transpose(fr)
                    except Exception:
                        pass
                    img = limit_long(_pil_to_bgr(fr), MAX_SIDE)
                except Exception as e:
                    yield i, None, f"{type(e).__name__}: {e}"
                    continue
                yield i, img, ""
    except Exception as e:
        yield 0, None, f"{type(e).__name__}: {e}"


def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg
    img = img.convert("RGB")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def _load_image_file(p: Path) -> List[np.ndarray]:
    pages = []
    with Image.open(p) as im:
        for frame in ImageSequence.Iterator(im):
            fr = frame.copy()
            try:
                fr = ImageOps.exif_transpose(fr)  # 手机拍照的方向信息
            except Exception:
                pass
            pages.append(limit_long(_pil_to_bgr(fr), MAX_SIDE))
    return pages


def imwrite(path: str | os.PathLike, img: np.ndarray, quality: int = 90) -> None:
    """支持中文路径的写图。"""
    ext = Path(path).suffix.lower() or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"无法写入图片 {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def imread(path: str | os.PathLike) -> np.ndarray:
    """支持中文路径的读图（只取第一页）。"""
    return load_pages(path)[0]


# ---------------------------------------------------------------- 预处理

def red_mask(bgr: np.ndarray, thresh: int = 45) -> np.ndarray:
    """红色笔迹（老师批改痕迹）的掩码：鲜红色 + 偏浅的粉红色笔画边缘。"""
    b, g, r = [c.astype(np.int16) for c in cv2.split(bgr)]
    strong = (r - np.maximum(g, b)) > thresh
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s_, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    pink = ((h <= 12) | (h >= 150)) & (s_ > 45) & (v > 110) & (r > g)
    return strong | pink


def _background(bgr: np.ndarray) -> np.ndarray:
    """估计纸张底色（缩小后做中值滤波再放大，速度快）。"""
    h, w = bgr.shape[:2]
    f = 4 if min(h, w) > 400 else 1
    small = cv2.resize(bgr, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 7)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def strip_red(bgr: np.ndarray, thresh: int = 45) -> np.ndarray:
    """去掉红色笔迹（用纸张底色填补）。黑色印刷的答题纸默认这样处理，红色印刷的答题纸不要用。"""
    core = red_mask(bgr, thresh)
    if not core.any():
        return bgr
    ring = (cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~core
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    replace = core | (ring & (gray > 150))   # 边缘只抹浅色像素，保住旁边的黑色字迹
    out = bgr.copy()
    out[replace] = _background(bgr)[replace]
    return out


def red_ink_ratio(bgr: np.ndarray) -> float:
    """深色像素里红色的比例，用来判断答题纸是不是红色印刷。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < 200
    n = int(dark.sum())
    if n == 0:
        return 0.0
    return float((red_mask(bgr) & dark).sum()) / n


def to_gray(bgr: np.ndarray, drop_red: bool = False) -> np.ndarray:
    if drop_red:
        bgr = strip_red(bgr)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr


def resize_long(img: np.ndarray, long_side: int) -> Tuple[np.ndarray, float]:
    """等比缩放到长边为 long_side，返回 (图, 缩放系数)。"""
    h, w = img.shape[:2]
    s = long_side / float(max(h, w))
    if abs(s - 1.0) < 1e-3:
        return img, 1.0
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=interp), s


def limit_long(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    return resize_long(img, max_side)[0]


def to_data_url(img: np.ndarray, max_side: int = 2048, quality: int = 88) -> str:
    """编码成 base64 data URL，给视觉模型用。"""
    img = limit_long(img, max_side)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("图片编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------- 字体与绘制

_FONT_CANDIDATES = [
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
_font_path: Optional[str] = None
_font_searched = False


def find_cjk_font() -> Optional[str]:
    global _font_path, _font_searched
    if _font_searched:
        return _font_path
    _font_searched = True
    env = os.environ.get("AUTOGRADE_FONT")
    cands = ([env] if env else []) + _FONT_CANDIDATES
    for c in cands:
        if c and os.path.exists(c):
            _font_path = c
            return c
    patterns = ["/System/Library/Fonts/**/*PingFang*", "/System/Library/Fonts/**/*Heiti*",
                "/Library/Fonts/**/*", "/usr/share/fonts/**/*CJK*", "/usr/share/fonts/**/*wqy*",
                os.path.expanduser("~/Library/Fonts/*"), "C:/Windows/Fonts/*"]
    keys = ("pingfang", "heiti", "hiragino sans gb", "songti", "cjk", "wqy", "msyh", "simhei", "simsun", "noto sans sc")
    for pat in patterns:
        for f in glob.glob(pat, recursive=True):
            if f.lower().endswith((".ttf", ".ttc", ".otf")) and any(k in os.path.basename(f).lower() for k in keys):
                _font_path = f
                return f
    return None


def get_font(size: int) -> ImageFont.ImageFont:
    path = find_cjk_font()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:  # pragma: no cover
        return ImageFont.load_default()


def draw_texts(bgr: np.ndarray, items: List[dict]) -> np.ndarray:
    """在图上写字（支持中文）。items: {xy, text, size, color(BGR), bg(BGR|None), anchor}"""
    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    for it in items:
        font = get_font(int(it.get("size", 24)))
        color = tuple(int(c) for c in it.get("color", (0, 0, 255))[::-1])
        anchor = it.get("anchor", "la")
        x, y = it["xy"]
        if it.get("bg") is not None:
            box = d.textbbox((x, y), it["text"], font=font, anchor=anchor)
            pad = max(2, int(it.get("size", 24) * 0.18))
            d.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad),
                        fill=tuple(int(c) for c in it["bg"][::-1]))
        sw = int(it.get("stroke", 0))
        if sw:
            d.text((x, y), it["text"], font=font, fill=color, anchor=anchor, stroke_width=sw,
                   stroke_fill=tuple(int(c) for c in it.get("stroke_color", (255, 255, 255))[::-1]))
        else:
            d.text((x, y), it["text"], font=font, fill=color, anchor=anchor)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
