"""数字版 PDF（Word/WPS 导出的试卷、答案、答题纸）的文字层解析。

不依赖模型，做三件事：
1. 找题号位置，把试卷/答案按题切成区域（处理跨页、空白页、页眉页脚、“考生注意”里的 1. 2. 3.）。
2. 从文字里读分值：“（本题满分15分…第(1)小题满分6分…）”、大题说明、答案里的“……6 分”。
3. 用文字层的汉字核对模型录入的内容（多出来的、漏掉的）。

公式在文字层里通常是乱码（MathType/公式编辑器），所以公式内容仍由模型看图录入；
这里只用文字层里可靠的部分：汉字、题号、小问编号、分值。
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import pymupdf
except ImportError:  # pragma: no cover - 旧版 PyMuPDF
    import fitz as pymupdf  # type: ignore

# ---------------------------------------------------------------- 正则

ANCHOR_RE = re.compile(r"^\s*(?:第\s*)?(\d{1,2})\s*(?:题)?\s*[\.．、:：](?![\d])")
SECTION_RE = re.compile(r"^\s*([一二三四五六七八九十]{1,3})\s*[、\.．]\s*(.{0,40})")
LABEL_RE = re.compile(r"^\s*(?:(?:第\s*)?\d{1,2}\s*(?:题)?\s*[\.．、:：]\s*)?[（(]\s*(\d{1,2}|[ⅠⅡⅢⅣⅤⅥ]|[ivx]{1,3}|[一二三四五六])\s*[)）]")
MARKER_RE = re.compile(r"(?:[\.…·．。‥⋯]{2,}|[（(])\s*[^\d（()）]{0,6}?(\d{1,2}(?:\.\d)?)\s*分\s*[)）]?\s*$")
ANSWER_HEAD_RE = re.compile(r"^.{0,16}(参考答案|答案|评分标准|评分细则|解析)(及评分标准|与评分标准|和评分标准|与解析|及解析)?[:：]?$")
CJK_RE = re.compile(r"[一-鿿]")

_LABEL_MAP = {"Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "Ⅴ": "5", "Ⅵ": "6", "i": "1", "ii": "2", "iii": "3",
              "iv": "4", "v": "5", "一": "1", "二": "2", "三": "3", "四": "4", "五": "5", "六": "6"}
_TYPE_WORDS = [("解答", "解答题"), ("证明", "解答题"), ("计算", "解答题"), ("应用", "解答题"), ("作图", "解答题"),
               ("填空", "填空题"), ("多选", "选择题"), ("单选", "选择题"), ("选择", "选择题"), ("判断", "其他")]


def section_type(text: str) -> str:
    for w, t in _TYPE_WORDS:
        if w in text:
            return t
    return ""


def norm_part_label(raw: str) -> str:
    raw = raw.strip()
    return f"({_LABEL_MAP.get(raw, raw)})"


# ---------------------------------------------------------------- 数据结构

@dataclass
class Frag:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str


@dataclass
class Row:
    page: int
    col: int
    x0: float
    y0: float
    x1: float
    y1: float
    frags: List[Frag] = field(default_factory=list)

    @property
    def text(self) -> str:
        """按横坐标拼接；碎片之间有明显空隙时补一个空格（避免“1.”和后面的“5”粘成“1.5”）。"""
        out, last = "", None
        for f in sorted(self.frags, key=lambda f: f.x0):
            if last is not None and f.x0 - last > 1.5 and not out.endswith(" "):
                out += " "
            out += f.text
            last = f.x1 if last is None else max(last, f.x1)
        return out.strip()

    @property
    def first_x(self) -> float:
        return min(f.x0 for f in self.frags)

    @property
    def pos(self) -> Tuple[int, int, float]:
        return (self.page, self.col, self.y0)


@dataclass
class Unit:
    """一页里的一栏（竖版页面只有一栏，横版 A3 两栏）。"""
    page: int
    col: int
    x0: float
    x1: float
    rows: List[Row]
    content: Optional[Tuple[float, float, float, float]]  # 去掉页眉页脚后的内容范围

    @property
    def blank(self) -> bool:
        return self.content is None


@dataclass
class Anchor:
    number: str
    row: Row
    section: str = ""        # 所在大题的标题
    qtype: str = ""          # 由大题标题推断的题型


@dataclass
class Piece:
    page: int
    rect: Tuple[float, float, float, float]   # PDF 坐标（点）


@dataclass
class Segment:
    anchor: Anchor
    pieces: List[Piece]
    rows: List[Row]

    @property
    def text(self) -> str:
        return "\n".join(r.text for r in self.rows)


# ---------------------------------------------------------------- 读取 PDF

class PdfText:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self.doc = pymupdf.open(self.path)
        self.sizes = [(pg.rect.width, pg.rect.height) for pg in self.doc]
        self.frags: List[List[Frag]] = []
        self.drawings: List[List[Tuple[float, float, float, float]]] = []
        for i, pg in enumerate(self.doc):
            fr = []
            d = pg.get_text("dict")
            for b in d.get("blocks", []):
                if b.get("type") != 0:
                    continue
                for ln in b.get("lines", []):
                    dx, dy = ln.get("dir", (1, 0))
                    if abs(dy) > 0.2:      # 竖排文字（侧边的“姓名”“班级”等）不参与
                        continue
                    t = "".join(s.get("text", "") for s in ln.get("spans", []))
                    if not t.strip():
                        continue
                    x0, y0, x1, y1 = ln["bbox"]
                    fr.append(Frag(i, x0, y0, x1, y1, t))
            self.frags.append(fr)
            rects = []
            try:
                for dr in pg.get_drawings():
                    r = dr.get("rect")
                    if r is not None and r.width * r.height < 0.8 * pg.rect.width * pg.rect.height:
                        rects.append((r.x0, r.y0, r.x1, r.y1))
            except Exception:
                pass
            self.drawings.append(rects)
        self.units = self._build_units()
        self.header_footer: set = set()
        self._mark_header_footer()

    def close(self):
        self.doc.close()

    # 质量判断：是不是有可用的文字层
    def cjk_count(self) -> int:
        return sum(len(CJK_RE.findall(f.text)) for fr in self.frags for f in fr)

    def nonblank_pages(self) -> List[int]:
        return sorted({u.page for u in self.units if not u.blank})

    def usable(self) -> bool:
        pages = self.nonblank_pages()
        return bool(pages) and self.cjk_count() >= 30 * len(pages)

    # 栏与行
    def _columns(self, page: int) -> List[Tuple[float, float]]:
        w, h = self.sizes[page]
        if w > 1.2 * h:   # 横版（A3/8K）按两栏处理
            return [(0.0, w / 2.0), (w / 2.0, w)]
        return [(0.0, w)]

    def _build_units(self) -> List[Unit]:
        units = []
        for p, frs in enumerate(self.frags):
            cols = self._columns(p)
            for c, (cx0, cx1) in enumerate(cols):
                mine = [f for f in frs if cx0 <= (f.x0 + f.x1) / 2.0 < cx1]
                rows = _cluster_rows(mine, p, c)
                units.append(Unit(p, c, cx0, cx1, rows, None))
        return units

    def _mark_header_footer(self):
        """页眉页脚：页面最上/最下一条里，带“第X页/共X页”、只有页码，或者在多页重复出现（数字不同）的行。"""
        seen: Counter = Counter()
        cand = []
        for u in self.units:
            w, h = self.sizes[u.page]
            for r in u.rows:
                if r.y1 < 0.12 * h or r.y0 > 0.85 * h:
                    key = re.sub(r"\d+", "#", re.sub(r"\s+", "", r.text))
                    seen[key] += 1
                    cand.append((u, r, key, r.y1 < 0.07 * h or r.y0 > 0.91 * h))
        for u, r, key, edge in cand:
            t = re.sub(r"\s+", "", r.text)
            pageno = re.search(r"第#?\d*页|共\d+页|^[-—–]?\d{1,3}[-—–]?$|^\d+/\d+$", t)
            if pageno or (seen[key] >= 2 and len(self.sizes) >= 2 and len(t) <= 40) or (edge and "页" in t):
                self.header_footer.add(id(r))
        for u in self.units:
            u.rows = [r for r in u.rows if id(r) not in self.header_footer]
            xs = [(r.x0, r.y0, r.x1, r.y1) for r in u.rows]
            for d in self.drawings[u.page]:
                cx = (d[0] + d[2]) / 2.0
                if u.x0 <= cx < u.x1 and (d[2] - d[0]) * (d[3] - d[1]) > 4:
                    w, h = self.sizes[u.page]
                    if 0.05 * h < d[1] and d[3] < 0.93 * h:
                        xs.append(d)
            if u.rows:
                u.content = (min(a[0] for a in xs), min(a[1] for a in xs), max(a[2] for a in xs), max(a[3] for a in xs))
            else:
                u.content = None

    # 渲染与裁切
    def render(self, page: int, dpi: int = 200) -> np.ndarray:
        pix = self.doc[page].get_pixmap(dpi=dpi, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return np.ascontiguousarray(cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR) if pix.n >= 3
                                    else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR))

    def crop(self, piece: Piece, dpi: int = 200) -> np.ndarray:
        x0, y0, x1, y1 = piece.rect
        clip = pymupdf.Rect(x0, y0, x1, y1) & self.doc[piece.page].rect
        pix = self.doc[piece.page].get_pixmap(dpi=dpi, alpha=False, clip=clip)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return np.ascontiguousarray(cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR) if pix.n >= 3
                                    else cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR))

    def all_rows(self) -> List[Row]:
        return [r for u in self.units for r in u.rows]

    def unit_of(self, row: Row) -> Unit:
        return next(u for u in self.units if u.page == row.page and u.col == row.col)


def _cluster_rows(frags: List[Frag], page: int, col: int) -> List[Row]:
    """把碎片按垂直位置合成“视觉行”。公式里的上下标、分子分母高度不同，按重叠比例归行。"""
    rows: List[Row] = []
    for f in sorted(frags, key=lambda f: ((f.y0 + f.y1) / 2.0, f.x0)):
        h = max(1.0, f.y1 - f.y0)
        best, best_ov = None, 0.0
        for r in rows[-4:]:
            ov = min(f.y1, r.y1) - max(f.y0, r.y0)
            rh = max(1.0, r.y1 - r.y0)
            frac = ov / min(h, rh)
            if frac > best_ov:
                best, best_ov = r, frac
        if best is not None and best_ov >= 0.55:
            best.frags.append(f)
            best.x0, best.y0 = min(best.x0, f.x0), min(best.y0, f.y0)
            best.x1, best.y1 = max(best.x1, f.x1), max(best.y1, f.y1)
        else:
            rows.append(Row(page, col, f.x0, f.y0, f.x1, f.y1, [f]))
    return sorted(rows, key=lambda r: (r.y0, r.x0))


# ---------------------------------------------------------------- 题号定位

def _reading_key(pos):
    return pos


def find_sections(pdf: PdfText) -> List[Tuple[Row, str, str]]:
    out = []
    for u in pdf.units:
        if u.blank:
            continue
        left = u.content[0]
        for r in u.rows:
            m = SECTION_RE.match(r.text)
            if m and r.first_x <= left + 0.06 * (u.content[2] - left):
                out.append((r, r.text, section_type(m.group(2))))
    return sorted(out, key=lambda t: t[0].pos)


def find_answer_heading(pdf: PdfText) -> Optional[Row]:
    for u in pdf.units:
        for r in u.rows:
            t = re.sub(r"\s+", "", r.text)
            if 2 <= len(t) <= 24 and ANSWER_HEAD_RE.match(t):
                return r
    return None


def find_anchors(pdf: PdfText, start=None, end=None) -> List[Anchor]:
    """题号行：行首是“18．/18./18、/第18题”，且在这一栏的左边缘。取最长的递增链（排除“考生注意 1. 2. 3.”之类）。"""
    sections = [s for s in find_sections(pdf) if (start is None or s[0].pos >= start) and (end is None or s[0].pos < end)]
    first_sec = sections[0][0].pos if sections else None
    cands: List[Tuple[int, Row]] = []
    for u in pdf.units:
        if u.blank:
            continue
        left, right = u.content[0], u.content[2]
        tol = 0.05 * max(1.0, right - left)
        for r in u.rows:
            if start is not None and r.pos < start:
                continue
            if end is not None and r.pos >= end:
                continue
            if first_sec is not None and r.pos < first_sec:
                continue
            m = ANCHOR_RE.match(r.text)
            if m and r.first_x <= left + tol:
                n = int(m.group(1))
                if 1 <= n <= 60:
                    cands.append((n, r))
    cands.sort(key=lambda c: c[1].pos)
    # 动态规划：链长最长 → 跳号最少 → 前驱离得最近
    best: List[Tuple[int, int, int]] = []
    prev: List[int] = []
    for i, (n, r) in enumerate(cands):
        b, pj = (1, 0, 0), -1
        for j in range(i):
            nj = cands[j][0]
            if nj >= n:
                continue
            L, G, _ = best[j]
            score = (L + 1, G - (n - nj - 1), j)
            if score > b:
                b, pj = score, j
        best.append(b)
        prev.append(pj)
    if not cands:
        return []
    i = max(range(len(cands)), key=lambda k: best[k][:2] + (k,))
    chain = []
    while i != -1:
        chain.append(cands[i])
        i = prev[i]
    chain.reverse()
    anchors = []
    for n, r in chain:
        sec = [s for s in sections if s[0].pos <= r.pos]
        st, stype = (sec[-1][1], sec[-1][2]) if sec else ("", "")
        anchors.append(Anchor(str(n), r, st, stype))
    return anchors


# ---------------------------------------------------------------- 按题切区域

def span(pdf: PdfText, start, end, pad: float = 4.0) -> Tuple[List[Piece], List[Row]]:
    """从阅读位置 start 到 end（不含）的内容：每栏/每页一块截图区域 + 其中的文字行。"""
    pieces, rows = [], []
    for u in pdf.units:
        if u.blank:
            continue
        ukey = (u.page, u.col)
        if ukey < (start[0], start[1]):
            continue
        if end is not None and ukey > (end[0], end[1]):
            break
        cx0, cy0, cx1, cy1 = u.content
        top = start[2] - pad if ukey == (start[0], start[1]) else cy0 - pad
        bottom = (end[2] - pad) if (end is not None and ukey == (end[0], end[1])) else cy1 + pad
        mine = [r for r in u.rows if top - 1 <= (r.y0 + r.y1) / 2.0 <= bottom]
        if not mine or bottom - top < 6:
            continue
        # 去掉题目下面留给学生作答的大片空白：裁到最后一行文字/图形为止
        used = max(r.y1 for r in mine)
        for dr in pdf.drawings[u.page]:
            dcx, dcy = (dr[0] + dr[2]) / 2.0, (dr[1] + dr[3]) / 2.0
            if u.x0 <= dcx < u.x1 and top <= dcy <= bottom:
                used = max(used, dr[3])
        bottom = min(bottom, used + 2 * pad)
        pieces.append(Piece(u.page, (max(0.0, cx0 - pad), top, cx1 + pad, bottom)))
        rows += sorted(mine, key=lambda r: (r.y0, r.x0))
    return pieces, rows


def segments(pdf: PdfText, anchors: List[Anchor], stop=None, pad: float = 4.0) -> Dict[str, Segment]:
    """每道题从它的题号行开始，到下一个题号行/大题标题/答案标题为止（可以跨栏、跨页）。"""
    bounds = sorted({a.row.pos for a in anchors} | {s[0].pos for s in find_sections(pdf)}
                    | ({stop} if stop is not None else set()))
    out: Dict[str, Segment] = {}
    for a in anchors:
        start = a.row.pos
        nxt = [b for b in bounds if b > start]
        pieces, rows = span(pdf, start, nxt[0] if nxt else None, pad)
        out.setdefault(a.number, Segment(a, pieces, rows))
    return out


def objective_spans(pdf: PdfText, types: Dict[str, str], start=None) -> Dict[str, Tuple[List[Piece], List[Row]]]:
    """答案里“填空题”“选择题”两段（答案常常一行写好几题，没法逐题切，就整段取出）。
    types: {题号: 题型}（来自试卷）。优先用大题标题定位，没有标题就用题号范围。"""
    out = {}
    secs = [s_ for s_ in find_sections(pdf) if start is None or s_[0].pos >= start]
    anchors = find_anchors(pdf, start=start)
    for kind in ("选择题", "填空题"):
        nums = {n for n, t in types.items() if t == kind}
        if not nums:
            continue
        sec = next((s_ for s_ in secs if s_[2] == kind), None)
        if sec is not None:
            st = sec[0].pos
        else:
            first = [a for a in anchors if a.number in nums]
            if not first:
                continue
            st = first[0].row.pos
        later = [s_[0].pos for s_ in secs if s_[0].pos > st]
        later += [a.row.pos for a in anchors if a.row.pos > st and a.number not in nums]
        end = min(later) if later else None
        pieces, rows = span(pdf, st, end)
        if pieces:
            out[kind] = (pieces, rows)
    return out


CHOICE_PAIR_RE = re.compile(r"(\d{1,2})\s*[\.．、:：]?\s*([A-H](?:\s*[,，、]?\s*[A-H]){0,4})(?![A-Za-z])")


def parse_choice_answers(rows: List[Row], numbers: Sequence[str]) -> Dict[str, str]:
    """选择题答案：“13.B 14.C …”，或者表格式（一行题号、下一行字母）。"""
    want = set(numbers)
    out: Dict[str, str] = {}
    texts = [r.text for r in rows]
    for t in texts:
        for n, letters in CHOICE_PAIR_RE.findall(t):
            if n in want and n not in out:
                out[n] = re.sub(r"[^A-H]", "", letters)
    if len(out) < len(want):
        for a, b in zip(texts, texts[1:]):
            ns = re.findall(r"\d{1,2}", a)
            ls = re.findall(r"(?<![A-Za-z])[A-H]{1,4}(?![A-Za-z])", b)
            if len(ns) >= 2 and len(ns) == len(ls):
                for n, l in zip(ns, ls):
                    if n in want and n not in out:
                        out[n] = l
    return out


# ---------------------------------------------------------------- 分值

def parse_section_scores(title: str, numbers: Sequence[str]) -> Dict[str, float]:
    """“其中1－6题每题4分，7－12题每题5分” / “每小题5分” → {题号: 分值}"""
    out: Dict[str, float] = {}
    t = title.replace(" ", "")
    for a, b, s in re.findall(r"(\d+)[－\-–—~～至到](\d+)(?:小)?题(?:每(?:小)?题)?(\d+(?:\.\d+)?)分", t):
        for n in range(int(a), int(b) + 1):
            out[str(n)] = float(s)
    if not out:
        m = re.search(r"每(?:小)?题(?:满分)?(\d+(?:\.\d+)?)分", t)
        if m:
            for n in numbers:
                out[n] = float(m.group(1))
    return out


def section_total(title: str) -> Optional[float]:
    m = re.search(r"(?:满分|共)\s*(\d+(?:\.\d+)?)\s*分", title.replace(" ", ""))
    return float(m.group(1)) if m else None


def parse_question_header(text: str) -> Tuple[Optional[float], Dict[str, float]]:
    """“（本题满分15分，本题共有2个小题，第(1)小题满分6分，第(2)小题满分9分）” / “17.（12分）”"""
    t = re.sub(r"\s+", "", text)
    score = None
    m = re.search(r"本(?:小)?题满分(\d+(?:\.\d+)?)分", t) or re.search(r"^\d{1,2}[\.．、][（(](\d+(?:\.\d+)?)分[)）]", t)
    if m:
        score = float(m.group(1))
    parts = {}
    for lab, s in re.findall(r"第[（(]?(\d|[ⅠⅡⅢⅣ一二三四])[)）]?(?:小)?(?:题|问)满分(\d+(?:\.\d+)?)分", t):
        parts[norm_part_label(lab)] = float(s)
    return score, parts


def strip_header_text(text: str) -> str:
    """去掉“（本题满分15分，…第(1)小题满分6分…）”这类分值说明（里面可以再套一层括号）。"""
    return re.sub(r"[（(]\s*本\s*(?:小)?\s*题\s*满\s*分(?:[^（）()]|[（(][^（）()]*[)）])*[)）]", "", text)


def row_label(row: Row, unit: Unit) -> Optional[str]:
    if unit.blank:
        return None
    left, right = unit.content[0], unit.content[2]
    if row.first_x > left + 0.12 * max(1.0, right - left):
        return None
    m = LABEL_RE.match(row.text)
    return norm_part_label(m.group(1)) if m else None


def part_labels(pdf: PdfText, seg: Segment) -> List[str]:
    labels = []
    for r in seg.rows:
        lab = row_label(r, pdf.unit_of(r))
        if lab and lab not in labels:
            labels.append(lab)
    return labels


def row_marker(row: Row) -> Optional[float]:
    """一行里的“……X分”。先在每个碎片里找（碎片内顺序可靠），找不到再看整行
    （公式里的单个数字有时排在分值后面，所以整行末尾允许跟一两个非汉字字符）。"""
    for f in sorted(row.frags, key=lambda f: -f.x1):
        m = MARKER_RE.search(f.text.rstrip())
        if m:
            return float(m.group(1))
    m = re.search(MARKER_RE.pattern.replace(r"\s*$", r"[^\u4e00-\u9fff]{0,3}$"), row.text)
    return float(m.group(1)) if m else None


def answer_markers(pdf: PdfText, seg: Segment) -> List[Tuple[str, float]]:
    """答案里的“……X分”，按小问归类：[(小问, X), ...]"""
    out = []
    cur = "(全题)"
    for r in seg.rows:
        lab = row_label(r, pdf.unit_of(r))
        if lab:
            cur = lab
        v = row_marker(r)
        if v is not None:
            out.append((cur, v))
    return out


def markers_from_text(text: str) -> List[Tuple[str, float]]:
    """同上，但作用在模型录入的答案文字上（扫描件没有文字层时用）。"""
    out = []
    cur = "(全题)"
    for line in (text or "").splitlines():
        m = LABEL_RE.match(line)
        if m:
            cur = norm_part_label(m.group(1))
        for v in re.findall(r"(?:[\.…·．。‥⋯]{2,}|[（(])\s*(\d{1,2}(?:\.\d)?)\s*分", line):
            out.append((cur, float(v)))
    return out


def scores_from_markers(markers: List[Tuple[str, float]], stated: Dict[str, float],
                        total: Optional[float]) -> Tuple[Dict[str, float], str]:
    """由分值标注推出各小问分值，并判断标注方式：
    cumulative（每小问内累计：2,4,6 → 6分）/ cumulative_all（全题累计）/ step（每步分值相加）。"""
    if not markers:
        return {}, ""
    by: Dict[str, List[float]] = {}
    order = []
    for lab, v in markers:
        if lab not in by:
            by[lab] = []
            order.append(lab)
        by[lab].append(v)
    nondec = all(all(b >= a for a, b in zip(vs, vs[1:])) for vs in by.values())
    cum = {lab: vs[-1] for lab, vs in by.items()}
    step = {lab: sum(vs) for lab, vs in by.items()}
    flat = [v for _, v in markers]
    cum_all = {}
    if len(order) > 1 and all(b >= a for a, b in zip(flat, flat[1:])):
        last = 0.0
        for lab in order:
            cum_all[lab] = by[lab][-1] - last
            last = by[lab][-1]

    def fits(d):
        if stated and all(abs(d.get(k, -1) - v) < 1e-6 for k, v in stated.items() if k in d):
            if any(k in d for k in stated):
                return True
        if total is not None and abs(sum(d.values()) - total) < 1e-6 and not stated:
            return True
        return False
    for name, d in (("cumulative", cum), ("cumulative_all", cum_all), ("step", step)):
        if d and fits(d):
            return d, name
    if nondec:
        return cum, "cumulative"
    return step, "step"


# ---------------------------------------------------------------- 核对

def _cjk_seq(text: str) -> str:
    return "".join(CJK_RE.findall(text or ""))


def _bigrams(seq: str) -> Counter:
    return Counter(seq[i:i + 2] for i in range(len(seq) - 1))


def _runs(seq: str, bad: set, min_len: int = 4) -> List[str]:
    """把“不在对方里的二元组”连成片段，方便给老师看。"""
    out, cur = [], ""
    for i in range(len(seq) - 1):
        bg = seq[i:i + 2]
        if bg in bad:
            cur = cur + bg[1] if cur else bg
        else:
            if len(cur) >= min_len:
                out.append(cur)
            cur = ""
    if len(cur) >= min_len:
        out.append(cur)
    return out


def compare_text(source: str, produced: str) -> dict:
    """用汉字二元组比较“原文（文字层）”和“模型录入”。公式是乱码，不参与。"""
    s, p = _cjk_seq(source), _cjk_seq(produced)
    bs, bp = _bigrams(s), _bigrams(p)
    if len(bs) < 6:
        return {"checked": False}
    extra = set(bp) - set(bs)
    missing = set(bs) - set(bp)
    extra_ratio = sum(bp[b] for b in extra) / max(1, sum(bp.values()))
    missing_ratio = sum(bs[b] for b in missing) / max(1, sum(bs.values()))
    return {"checked": True, "extra_ratio": round(extra_ratio, 3), "missing_ratio": round(missing_ratio, 3),
            "extra": _runs(p, extra)[:5], "missing": _runs(s, missing)[:5]}


RESTART_RE = re.compile(r"^\s*(?:解|证明|证|答|分析)?\s*[:：]?\s*[（(]\s*(\d{1,2}|[ⅠⅡⅢⅣ])\s*[)）]")


def restart_labels(text: str) -> List[str]:
    """答案里小问编号又从头出现（如第二次出现“解：（1）”），通常说明混进了另一道题。"""
    seen, dup = [], []
    for line in (text or "").splitlines():
        m = LABEL_RE.match(line) or RESTART_RE.match(line)
        if m:
            lab = norm_part_label(m.group(1))
            if lab in seen and lab not in dup:
                dup.append(lab)
            seen.append(lab)
    return dup


# ---------------------------------------------------------------- 答题纸：用文字层标题号

SHEET_Q_RE = re.compile(r"^\s*(?:第\s*)?(\d{1,2})\s*(?:题)?\s*[\.．、]?\s*[（(]\s*(\d+(?:\.\d+)?)\s*分\s*[)）]")
CONT_RE = re.compile(r"(?:接|续)\s*第?\s*(\d{1,2})\s*题|第\s*(\d{1,2})\s*题\s*[（(]?\s*续")


def label_boxes_from_text(frags: List[Frag], boxes: List[dict], to_tmpl) -> int:
    """frags: 一页答题纸的文字碎片（PDF 坐标）；boxes: 模板坐标的区域；to_tmpl(x, y) 把 PDF 坐标转到模板坐标。
    就地填写 kind/questions/full_score/header，返回标注了几个框。"""
    pts = []
    for f in frags:
        x0, y0 = to_tmpl(f.x0, f.y0)
        x1, y1 = to_tmpl(f.x1, f.y1)
        pts.append((f.text.strip(), min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))
    n = 0
    for b in boxes:
        bx0, by0, bx1, by1 = b["rect"]
        bh = by1 - by0
        inside = sorted([p for p in pts if bx0 - 2 <= (p[1] + p[3]) / 2 <= bx1 + 2 and by0 - 2 <= (p[2] + p[4]) / 2 <= by1 + 2],
                        key=lambda p: (round(p[2] / 8), p[1]))
        above = [p for p in pts if p[3] >= bx0 and p[1] <= bx1 and 0 <= by0 - p[4] <= 60]
        head = [p for p in inside if p[2] <= by0 + max(40, 0.15 * bh)]
        texts = [p[0] for p in inside]
        q = next((m for p in head for m in [SHEET_Q_RE.match(p[0])] if m), None)
        cont = next((m.group(1) or m.group(2) for t in texts[:4] for m in [CONT_RE.search(t)] if m), None)
        joined = " ".join(texts)
        above_txt = " ".join(p[0] for p in sorted(above, key=lambda p: p[2]))
        nums = [t for t in texts if re.fullmatch(r"\d{1,2}", t)]
        sec = section_label(above_txt)
        sec_kind = section_type(sec)
        if q:
            b.update(kind="解答题", questions=[q.group(1)], full_score=float(q.group(2)), header=q.group(0)[:60])
        elif cont:
            b.update(kind="解答题", questions=[cont], full_score=None, header=f"接第{cont}题")
        elif sec_kind in ("选择题", "填空题") and nums:
            b.update(kind=sec_kind, questions=nums, full_score=None, header=sec)
        elif re.search(r"[\[［【]\s*A\s*[\]］】]", joined):
            b.update(kind="选择题", questions=nums, full_score=None, header=section_label(above_txt) or "选择题")
        elif any(k in joined for k in ("姓名", "班级", "考号", "学号", "准考证")):
            b.update(kind="学生信息", questions=[], full_score=None, header="")
        elif "填空" in above_txt or len(nums) >= 2:
            b.update(kind="填空题", questions=nums, full_score=None, header=section_label(above_txt) or "填空题")
        else:
            continue
        n += 1
    return n


def section_label(text: str) -> str:
    m = SECTION_RE.search(text or "")
    return m.group(0)[:30] if m else ""
