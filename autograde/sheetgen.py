"""按题目生成答题纸：打印或复印后学生作答，扫描回来直接批改，不用再识别版面。

一页的构成（单位 mm）：
- 四角、左右边中间（A3 还有上下边中间）印“回”字定位点，左下角多一个，用来判断方向；
- 右上角二维码：哪次考试、第几页、答题纸版本（“每人一份”时还有学号）。通用版每份都一样，可以直接复印；
- 第 1 页：姓名、班级、注意事项，右边是涂学号的格子（每一位 0–9 各一格）；
- 然后按试卷的题序排：选择题（涂卡）、填空题（横线）、解答题（方框，按分值给空间，放不下就接到下一栏/下一页）。

生成时就知道每个格子、每条横线、每个方框的确切位置，直接写成版面模板（plan → template）。
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import objective as O
from .imaging import imwrite
from .layout import WORK_LONG_SIDE

MM = 72.0 / 25.4                       # 1 mm = 2.835 pt
PAPERS = {"A4": (210.0, 297.0, 1), "A3": (420.0, 297.0, 2)}
PAPER_LABEL = {"A4": "A4 竖版", "A3": "A3 横版（两栏）"}
SPACES = {"compact": 0.8, "normal": 1.0, "loose": 1.25}
SPACE_LABEL = {"compact": "紧凑", "normal": "标准", "loose": "宽松"}

MARKER = 5.0          # “回”字定位点边长
EDGE = 8.0            # 定位点中心到纸边
MARGIN = 13.0         # 内容到纸边
GUTTER = 12.0         # 两栏之间
QR_SIZE = 21.0
TOP = 38.0            # 正文从这里开始（上面是标题和二维码）
BOTTOM_PAD = 14.0     # 正文到下边
SEC_TITLE = 7.0       # 大题标题行
GAP = 4.0             # 块与块之间

OPT_W, OPT_H, OPT_PITCH = 5.4, 3.4, 6.6     # 选择题选项格
ROW_PITCH = 5.8                             # 选择题行距
NUM_W = 7.5                                 # 题号宽
BLOCK_GAP = 6.0                             # 两组选择题之间
ID_W, ID_H, ID_COL, ID_ROW = 4.6, 3.0, 6.4, 4.3   # 学号格子
FILL_ROW = 12.5                             # 填空行距（竖着写的分数要高一点的地方）

INK = (0, 0, 0)
GRAY = (0.3, 0.3, 0.3)
QR_PREFIX = "AG1"
CN_NUM = "一二三四五六七八九十"


# ================================================================ 题目 → 排版计划

def option_count(q: dict) -> int:
    """选择题有几个选项：老师设过就用设的；否则看题干里的 A．B．C．D．E．；默认 4 个。"""
    n = q.get("options")
    try:
        if n and 2 <= int(n) <= 8:
            return int(n)
    except (TypeError, ValueError):
        pass
    found = set(re.findall(r"(?:^|[\s　（(，,；;。])([A-H])\s*[\.．、:：]", q.get("stem") or ""))
    if {"A", "B", "C", "D"} <= found:
        return max(4, ord(max(found)) - ord("A") + 1)
    return 4


def is_multi(q: dict) -> bool:
    return bool(q.get("multi")) or len(re.sub(r"[^A-H]", "", (q.get("answer") or "").upper())) > 1


def _key(n: str):
    m = re.match(r"(\d+)", str(n))
    return (int(m.group(1)) if m else 10 ** 6, str(n))


def sheet_questions(questions: Sequence[dict]) -> List[dict]:
    """要印在答题纸上的题（按题号排）。“其他”题型有分值的按解答题给方框。"""
    out = []
    for q in sorted(questions, key=lambda q: _key(q.get("number", ""))):
        t = q.get("type")
        if t in ("选择题", "填空题", "解答题"):
            out.append(q)
        elif q.get("score"):
            out.append(dict(q, type="解答题"))
    return out


def questions_sig(questions: Sequence[dict], scored: bool = True) -> str:
    """影响版面的题目信息（题号、题型、分值、选项数、小问数）的指纹：变了就该重新生成答题纸。
    不打分的作业答题纸不印分值、方框高度也不看分值，分值不算在内。"""
    key = [(q["number"], q["type"], q.get("score") if scored else None, option_count(q) if q["type"] == "选择题" else 0,
            len(q.get("parts") or [])) for q in sheet_questions(questions)]
    return hashlib.sha1(json.dumps(key, ensure_ascii=False).encode()).hexdigest()[:10]


def _columns(paper: str) -> List[Tuple[float, float]]:
    W, H, ncol = PAPERS[paper]
    x0, x1 = MARGIN, W - MARGIN
    cw = (x1 - x0 - GUTTER * (ncol - 1)) / ncol
    return [(x0 + k * (cw + GUTTER), x0 + k * (cw + GUTTER) + cw) for k in range(ncol)]


def _markers(paper: str) -> List[List[float]]:
    W, H, ncol = PAPERS[paper]
    pts = [(EDGE, EDGE), (W - EDGE, EDGE), (EDGE, H - EDGE), (W - EDGE, H - EDGE),
           (EDGE + MARKER + 2.0, H - EDGE),                 # 左下角多一个：分正反方向
           (EDGE, H / 2), (W - EDGE, H / 2)]
    if ncol == 2:
        pts += [(W / 2, EDGE), (W / 2, H - EDGE)]
    return [[round(x, 2), round(y, 2), MARKER] for x, y in pts]


def answer_height(q: dict, space: float, col_h: float, scored: bool = True) -> float:
    """解答题方框的默认高度（mm）：打分的按分值（12 分约 11 厘米）；不打分的作业按小问数。"""
    try:
        s = float(q.get("score") or 0) if scored else 0.0
    except (TypeError, ValueError):
        s = 0
    nparts = len(q.get("parts") or [])
    if s > 0:
        h = 26 + 7.2 * s + 4.0 * max(0, nparts - 2)
    elif not scored:
        h = 60.0 + 20.0 * max(1, nparts)
    else:
        h = 100.0 + 4.0 * max(0, nparts - 2)
    return round(max(45.0, min(h * space, col_h)), 1)


MIN_BOX = 25.0    # 老师拖动时，方框最矮 2.5 厘米


def _score_text(qs: List[dict]) -> str:
    sc = [q.get("score") for q in qs]
    if sc and all(s not in (None, "") for s in sc):
        vals = [float(s) for s in sc]
        if len(set(vals)) == 1:
            return f"共 {len(qs)} 题，每题 {vals[0]:g} 分"
        return f"共 {len(qs)} 题，共 {sum(vals):g} 分"
    return f"共 {len(qs)} 题"


class _Flow:
    """按栏往下排；一栏放不下就换下一栏、下一页。"""

    def __init__(self, paper: str):
        self.paper = paper
        self.W, self.H, self.ncol = PAPERS[paper]
        self.cols = _columns(paper)
        self.bottom = self.H - BOTTOM_PAD
        self.pages: List[dict] = []
        self.c = 0
        self.y = TOP
        self.new_page()

    @property
    def page(self) -> dict:
        return self.pages[-1]

    @property
    def x0(self) -> float:
        return self.cols[self.c][0]

    @property
    def x1(self) -> float:
        return self.cols[self.c][1]

    def new_page(self) -> None:
        self.pages.append({"markers": _markers(self.paper),
                           "qr": [round(self.W - MARGIN - QR_SIZE, 2), MARGIN - 1, round(self.W - MARGIN, 2),
                                  MARGIN - 1 + QR_SIZE],
                           "info": None, "sections": [], "boxes": []})
        self.c = 0
        self.y = TOP

    def next_col(self) -> None:
        if self.c + 1 < self.ncol:
            self.c += 1
            self.y = TOP
        else:
            self.new_page()

    def room(self) -> float:
        return self.bottom - self.y

    def need(self, h: float) -> None:
        if self.room() < h and self.y > TOP + 0.1:
            self.next_col()


def _info_block(fl: _Flow, digits: int, notice: bool) -> None:
    x0, x1 = fl.x0, fl.x1
    y0 = fl.y
    grid = None
    h = 34.0
    if digits:
        gh = 6.0 + 6.5 + 10 * ID_ROW + 2.5
        h = max(h, gh + 4.0)
        gx1 = x1 - 3.0
        gx0 = gx1 - digits * ID_COL
        top = y0 + 2.0
        cols = []
        for k in range(digits):
            cx = gx0 + (k + 0.5) * ID_COL
            write = [round(cx - 2.8, 2), round(top + 6.0, 2), round(cx + 2.8, 2), round(top + 12.0, 2)]
            bubbles = []
            for dgt in range(10):
                by = top + 13.5 + dgt * ID_ROW
                bubbles.append({"label": str(dgt), "rect": [round(cx - ID_W / 2, 2), round(by, 2),
                                                            round(cx + ID_W / 2, 2), round(by + ID_H, 2)]})
            cols.append({"pos": k, "write": write, "options": bubbles,
                         "rect": [round(cx - ID_COL / 2, 2), round(top + 13.0, 2), round(cx + ID_COL / 2, 2),
                                  round(top + 13.5 + 9 * ID_ROW + ID_H + 0.5, 2)]})
        grid = {"digits": digits, "rect": [round(gx0 - 1.5, 2), round(top, 2), round(gx1 + 1.5, 2), round(y0 + h - 2.0, 2)],
                "label_xy": [round((gx0 + gx1) / 2, 2), round(top + 4.3, 2)], "cols": cols}
    left1 = (grid["rect"][0] - 4.0) if grid else (x1 - 3.0)
    lines = []
    if notice:
        lines = [("答题前在规定位置写好姓名、班级" + ("，并用 2B 铅笔把学号涂黑（每一列涂一个数字）。" if digits else "。")),
                 "选择题用 2B 铅笔把所选选项的方框涂满涂黑；修改时用橡皮擦干净。",
                 "填空题、解答题用黑色签字笔写在对应的横线上或方框内，写在框外的不批改。",
                 "保持答题纸整洁，不要折叠、弄破，不要在二维码和四角的黑块上写画。"]
    fl.page["info"] = {"rect": [round(x0, 2), round(y0, 2), round(x1, 2), round(y0 + h, 2)],
                       "name": [round(x0 + 3.0, 2), round(y0 + 9.0, 2), round(x0 + 3.0 + min(52.0, (left1 - x0) * 0.55), 2)],
                       "klass": [round(x0 + 3.0 + min(58.0, (left1 - x0) * 0.6), 2), round(y0 + 9.0, 2), round(left1, 2)],
                       "notice": lines, "notice_box": [round(x0 + 3.0, 2), round(y0 + 15.0, 2), round(left1, 2), round(y0 + h - 2.5, 2)],
                       "grid": grid}
    fl.y = y0 + h + GAP


def _choice_section(fl: _Flow, qs: List[dict], title: str) -> None:
    nopt = max(option_count(q) for q in qs)
    block_w = NUM_W + nopt * OPT_PITCH
    blocks = [qs[i:i + 5] for i in range(0, len(qs), 5)]
    first = True
    while blocks:
        cw = fl.x1 - fl.x0 - 6.0
        per_band = max(1, int((cw + BLOCK_GAP) // (block_w + BLOCK_GAP)))
        need = SEC_TITLE + 3.0 + 5 * ROW_PITCH + 2.0
        fl.need(need)
        sec = {"kind": "选择题", "title": title if first else title.split("（")[0] + "（续）",
               "title_xy": [round(fl.x0, 2), round(fl.y + 5.0, 2)], "groups": [], "questions": []}
        box_top = fl.y + SEC_TITLE
        y = box_top + 2.5
        while blocks:
            band = blocks[:per_band]
            rows = max(len(b) for b in band)
            band_h = rows * ROW_PITCH + 1.0
            if y + band_h + 2.0 > fl.bottom and sec["groups"]:
                break
            for k, blk in enumerate(band):
                bx = fl.x0 + 3.0 + k * (block_w + BLOCK_GAP)
                for r, q in enumerate(blk):
                    ry = y + r * ROW_PITCH
                    n = option_count(q)
                    opts = []
                    for j in range(n):
                        ox = bx + NUM_W + j * OPT_PITCH
                        opts.append({"label": "ABCDEFGH"[j], "rect": [round(ox, 2), round(ry, 2), round(ox + OPT_W, 2),
                                                                     round(ry + OPT_H, 2)]})
                    sec["groups"].append({"q": q["number"], "multi": is_multi(q),
                                          "num_xy": [round(bx + NUM_W - 1.6, 2), round(ry + OPT_H - 0.45, 2)],
                                          "rect": [round(bx, 2), round(ry - 0.6, 2), round(bx + NUM_W + n * OPT_PITCH - 1.2, 2),
                                                   round(ry + OPT_H + 0.6, 2)], "options": opts})
                    sec["questions"].append(q["number"])
            blocks = blocks[per_band:]
            y += band_h + 1.5
        sec["rect"] = [round(fl.x0, 2), round(box_top, 2), round(fl.x1, 2), round(y + 0.5, 2)]
        fl.page["sections"].append(sec)
        fl.y = y + 0.5 + GAP
        first = False
        if blocks:
            fl.next_col()


def _fill_section(fl: _Flow, qs: List[dict], title: str) -> None:
    rest = list(qs)
    first = True
    while rest:
        fl.need(SEC_TITLE + 2.0 + FILL_ROW + 2.0)
        sec = {"kind": "填空题", "title": title if first else title.split("（")[0] + "（续）",
               "title_xy": [round(fl.x0, 2), round(fl.y + 5.0, 2)], "slots": [], "questions": []}
        box_top = fl.y + SEC_TITLE
        y = box_top + 1.5
        cw = fl.x1 - fl.x0 - 6.0
        per_row = 2
        sw = (cw - 6.0 * (per_row - 1)) / per_row
        while rest:
            if y + FILL_ROW + 1.5 > fl.bottom and sec["slots"]:
                break
            row = rest[:per_row]
            for k, q in enumerate(row):
                sx = fl.x0 + 3.0 + k * (sw + 6.0)
                ly = y + FILL_ROW - 2.2
                lx0, lx1 = sx + 8.0, sx + sw
                sec["slots"].append({"q": q["number"], "num_xy": [round(sx, 2), round(ly - 0.8, 2)],
                                     "line": [round(lx0, 2), round(ly, 2), round(lx1, 2)],
                                     "rect": [round(lx0 - 1.0, 2), round(ly - FILL_ROW + 2.7, 2), round(lx1 + 1.0, 2), round(ly + 1.4, 2)]})
                sec["questions"].append(q["number"])
            rest = rest[per_row:]
            y += FILL_ROW
        sec["rect"] = [round(fl.x0, 2), round(box_top, 2), round(fl.x1, 2), round(y + 1.5, 2)]
        fl.page["sections"].append(sec)
        fl.y = y + 1.5 + GAP
        first = False
        if rest:
            fl.next_col()


SHRINK = 0.75     # 放不下时，方框可以缩到原来的 75%，免得为差一点空间就换栏


def _answer_section(fl: _Flow, qs: List[dict], title: str, space: float, heights: Optional[dict] = None,
                    scored: bool = True) -> None:
    col_h = fl.bottom - TOP
    heights = heights or {}

    def height(q):
        h = heights.get(q["number"])
        if h:                                   # 老师拖过的：按老师定的高度，不再自动缩
            h = max(MIN_BOX, float(h))
            return (col_h if col_h < h < col_h + 25.0 else round(h, 1)), True
        return answer_height(q, space, col_h, scored), False
    first, _ = height(qs[0])
    # 标题要和第一道题的方框放在一起：剩下的地方够放下它（可以缩一点），或者够先放一截（≥50mm，其余接到下一栏）
    fl.need(SEC_TITLE + (50.0 if first >= 75.0 else max(45.0, first * SHRINK)))
    fl.page["sections"].append({"kind": "解答题", "title": title, "title_xy": [round(fl.x0, 2), round(fl.y + 5.0, 2)],
                                "rect": None, "questions": [q["number"] for q in qs]})
    fl.y += SEC_TITLE
    for q in qs:
        h, exact = height(q)
        sc = q.get("score") if scored else None
        label = f"{q['number']}.（{float(sc):g} 分）" if sc not in (None, "") else f"{q['number']}."
        cont = False
        while h > 0.5:
            room = fl.room()
            at_top = fl.y <= TOP + 0.1
            if h <= room:
                take = h
            elif not exact and not cont and room >= max(45.0, SHRINK * h):
                take = h = room                  # 差一点放不下：方框稍微缩一点
            elif room >= 50.0 and (h - room >= 25.0 or at_top):
                take = room                      # 这一栏剩下的先用上，其余接到下一栏
                if h - room < MIN_BOX:           # 只剩不到 2.5 厘米：不值得再开一栏，方框就到这一栏底为止
                    h = room
            else:
                fl.next_col()
                continue
            fl.page["boxes"].append({"q": q["number"], "score": sc, "cont": cont,
                                     "label": f"{q['number']}.（续）" if cont else label,
                                     "rect": [round(fl.x0, 2), round(fl.y, 2), round(fl.x1, 2), round(fl.y + take, 2)]})
            fl.y += take + GAP
            h -= take
            cont = True
            if h > 0.5:
                fl.next_col()


def plan_sheet(questions: Sequence[dict], title: str, paper: str = "A4", id_digits: int = 4,
               space: str = "normal", notice: bool = True, heights: Optional[dict] = None, scored: bool = True) -> dict:
    """排版：返回每页每个元素的位置（mm）。纯数据，存进版面模板，打印和读卡都按它来。
    heights：老师拖动定下的解答题方框高度 {题号: mm}；scored=False（平时作业不打分）时不印分值。"""
    if paper not in PAPERS:
        raise ValueError(f"不支持的纸张：{paper}")
    qs = sheet_questions(questions)
    if not qs:
        raise ValueError("还没有题目：先在第 1 步录入题目（上传试卷，或者按题号快速添加）")
    id_digits = max(0, min(12, int(id_digits or 0)))
    sp = SPACES.get(space, 1.0)
    fl = _Flow(paper)
    _info_block(fl, id_digits, notice)
    runs: List[List[dict]] = []
    for q in qs:
        if runs and runs[-1][0]["type"] == q["type"]:
            runs[-1].append(q)
        else:
            runs.append([q])
    for k, run in enumerate(runs):
        kind = run[0]["type"]
        head = f"{CN_NUM[k] if k < len(CN_NUM) else k + 1}、{kind}"
        info = _score_text(run) if scored else f"共 {len(run)} 题"
        if kind == "选择题":
            multi = [q["number"] for q in run if is_multi(q)]
            extra = f"；第 {'、'.join(multi)} 题为多选题" if multi and len(multi) < len(run) else ("；多选题" if multi else "")
            _choice_section(fl, run, f"{head}（{info}{extra}）")
        elif kind == "填空题":
            _fill_section(fl, run, f"{head}（{info}）")
        else:
            _answer_section(fl, run, f"{head}（{info}。解答应写出文字说明、证明过程或演算步骤）", sp, heights, scored)
    W, H, ncol = PAPERS[paper]
    heights = {k: round(float(v), 1) for k, v in (heights or {}).items() if v}
    plan = {"paper": paper, "size": [W, H], "cols": ncol, "title": (title or "").strip() or "答题纸",
            "id_digits": id_digits, "space": space, "notice": bool(notice), "pages": fl.pages,
            "questions": [q["number"] for q in qs], "sig": questions_sig(questions, scored), "heights": heights,
            "scored": bool(scored)}
    plan["ver"] = hashlib.sha1(json.dumps({k: plan[k] for k in ("paper", "id_digits", "pages")}, sort_keys=True)
                               .encode()).hexdigest()[:6]
    return plan


# ================================================================ 画 PDF

def qr_text(exam_id: str, page: int, pages: int, ver: str, sid: str = "") -> str:
    return "|".join([QR_PREFIX, exam_id, f"{page + 1}/{pages}", ver] + ([sid] if sid else []))


def parse_qr(text: str) -> Optional[dict]:
    parts = (text or "").strip().split("|")
    if len(parts) < 4 or parts[0] != QR_PREFIX:
        return None
    m = re.fullmatch(r"(\d+)/(\d+)", parts[2])
    if not m:
        return None
    return {"exam": parts[1], "page": int(m.group(1)) - 1, "pages": int(m.group(2)), "ver": parts[3],
            "sid": parts[4] if len(parts) > 4 else ""}


def _qr_modules(text: str) -> np.ndarray:
    p = cv2.QRCodeEncoder.Params()
    p.correction_level = cv2.QRCODE_ENCODER_CORRECT_LEVEL_Q
    img = cv2.QRCodeEncoder.create(p).encode(text)
    return img < 128                     # True = 黑模块（含 2 格静区）


def _font():
    import pymupdf
    return pymupdf.Font("china-s")


def _text(page, font, x, y, text, size, color=INK, bold=False, align="left"):
    w = font.text_length(text, fontsize=size) / MM
    if align == "center":
        x -= w / 2
    elif align == "right":
        x -= w
    page.insert_text((x * MM, y * MM), text, fontsize=size, fontname="cjk", color=color,
                     render_mode=2 if bold else 0, border_width=0.035 if bold else 1)
    return w


def _fit(font, text, size, max_w, min_size=8.0):
    while size > min_size and font.text_length(text, fontsize=size) / MM > max_w:
        size -= 0.5
    while len(text) > 4 and font.text_length(text, fontsize=size) / MM > max_w:
        text = text[:-2] + "…"
    return text, size


def _wrap(font, text, size, max_w) -> List[str]:
    out, cur = [], ""
    for ch in text:
        if font.text_length(cur + ch, fontsize=size) / MM > max_w and cur:
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _draw_page(page, font, plan: dict, p: int, qr: str, student: Optional[dict]) -> None:
    import pymupdf
    R = lambda r: pymupdf.Rect(r[0] * MM, r[1] * MM, r[2] * MM, r[3] * MM)
    W, H = plan["size"]
    pg = plan["pages"][p]
    n = len(plan["pages"])
    sh = page.new_shape()
    # 定位点：外框黑、白环、黑芯
    for x, y, s in pg["markers"]:
        sh.draw_rect(R([x - s / 2, y - s / 2, x + s / 2, y + s / 2]))
        sh.finish(color=None, fill=INK, width=0)
        sh.draw_rect(R([x - s * 0.32, y - s * 0.32, x + s * 0.32, y + s * 0.32]))
        sh.finish(color=None, fill=(1, 1, 1), width=0)
        sh.draw_rect(R([x - s * 0.18, y - s * 0.18, x + s * 0.18, y + s * 0.18]))
        sh.finish(color=None, fill=INK, width=0)
    # 二维码（矢量画，打印清楚）
    mods = _qr_modules(qr)
    q0, q1 = pg["qr"][:2], pg["qr"][2:]
    cell = (q1[0] - q0[0]) / mods.shape[1]
    for i, j in zip(*np.nonzero(mods)):
        sh.draw_rect(R([q0[0] + j * cell, q0[1] + i * cell, q0[0] + (j + 1) * cell + 0.02, q0[1] + (i + 1) * cell + 0.02]))
    sh.finish(color=None, fill=INK, width=0)
    sh.commit()
    # 标题
    title, size = _fit(font, f"{plan['title']}　答题纸", 15.0, pg["qr"][0] - MARGIN - 6)
    _text(page, font, MARGIN, MARGIN + 8.0, title, size, bold=True)
    if p == 0:
        _text(page, font, MARGIN, MARGIN + 16.5, "选择题和学号用 2B 铅笔填涂；其余用黑色签字笔，写在各题对应的位置。", 8.5, GRAY)
    else:
        sub = "姓名 ______________　班级 __________"
        if student:
            sub = f"姓名 {student.get('name', '')}　班级 {student.get('class', '')}"
        _text(page, font, MARGIN, MARGIN + 16.5, sub, 9.0, GRAY)
    cols = _columns(plan["paper"])            # 页码按栏编：A3 正面左栏第 1 页、右栏第 2 页，反面第 3、4 页
    for c, (cx0, cx1) in enumerate(cols):
        _text(page, font, (cx0 + cx1) / 2, H - 8.6, f"第 {p * len(cols) + c + 1} 页（共 {n * len(cols)} 页）", 8.0, GRAY,
              align="center")
    sh = page.new_shape()
    info = pg.get("info")
    if info:
        sh.draw_rect(R(info["rect"]))
        sh.finish(color=INK, width=0.8)
        nx0, ny, nx1 = info["name"]
        kx0, _, kx1 = info["klass"]
        _text(page, font, nx0, ny, "姓名", 10.5)
        _text(page, font, kx0, ny, "班级", 10.5)
        sh.draw_line(((nx0 + 10) * MM, (ny + 1.2) * MM), (nx1 * MM, (ny + 1.2) * MM))
        sh.draw_line(((kx0 + 10) * MM, (ny + 1.2) * MM), (kx1 * MM, (ny + 1.2) * MM))
        sh.finish(color=INK, width=0.6)
        if student:
            _text(page, font, nx0 + 12, ny - 0.3, student.get("name", ""), 12.0)
            _text(page, font, kx0 + 12, ny - 0.3, student.get("class", ""), 11.0)
        nb = info["notice_box"]
        if info["notice"]:
            y = nb[1] + 3.2
            _text(page, font, nb[0], y, "注意事项", 8.0, bold=True)
            y += 4.2
            for k, line in enumerate(info["notice"]):
                for j, piece in enumerate(_wrap(font, f"{k + 1}. {line}", 7.3, nb[2] - nb[0] - 4)):
                    if y > nb[3] - 5.5:
                        break
                    _text(page, font, nb[0] + (3.0 if j else 0.0), y, piece, 7.3, GRAY)
                    y += 3.6
        # 填涂样例
        ex = [nb[0], nb[3] - 2.6]
        _text(page, font, ex[0], ex[1], "正确填涂", 7.3, GRAY)
        sx = ex[0] + 13.0
        sh.draw_rect(R([sx, ex[1] - 2.7, sx + OPT_W, ex[1] - 2.7 + OPT_H]))
        sh.finish(color=None, fill=INK, width=0)
        _text(page, font, sx + 8.0, ex[1], "错误填涂  ✓  ×  ○  —", 7.3, GRAY)
        grid = info.get("grid")
        if grid:
            sh.draw_rect(R(grid["rect"]))
            sh.finish(color=INK, width=0.6)
            lx, ly = grid["label_xy"]
            _text(page, font, lx, ly, "学　号", 9.0, bold=True, align="center")
            sid = (student or {}).get("sid_padded") or ""
            for col in grid["cols"]:
                sh.draw_rect(R(col["write"]))
                sh.finish(color=INK, width=0.5)
                digit = sid[col["pos"]] if col["pos"] < len(sid) else ""
                if digit:
                    w = col["write"]
                    _text(page, font, (w[0] + w[2]) / 2, w[3] - 1.4, digit, 10.0, align="center")
                for o in col["options"]:
                    sh.draw_rect(R(o["rect"]))
                    sh.finish(color=GRAY, width=0.45)
                    x0, y0, x1, y1 = o["rect"]
                    if digit == o["label"]:
                        sh.draw_rect(R([x0 + 0.25, y0 + 0.25, x1 - 0.25, y1 - 0.25]))
                        sh.finish(color=None, fill=INK, width=0)
                    else:
                        _text(page, font, (x0 + x1) / 2, y1 - 0.75, o["label"], 6.3, GRAY, align="center")
    for sec in pg["sections"]:
        tx, ty = sec["title_xy"]
        t, size = _fit(font, sec["title"], 10.0, (_columns(plan["paper"])[0][1] - _columns(plan["paper"])[0][0]) - 2)
        _text(page, font, tx, ty, t, size, bold=True)
        if sec["kind"] == "选择题":
            sh.draw_rect(R(sec["rect"]))
            sh.finish(color=INK, width=0.8)
            for g in sec["groups"]:
                nx, ny = g["num_xy"]
                _text(page, font, nx, ny, g["q"], 9.0, align="right")
                for o in g["options"]:
                    sh.draw_rect(R(o["rect"]))
                    sh.finish(color=GRAY, width=0.45)
                    x0, y0, x1, y1 = o["rect"]
                    _text(page, font, (x0 + x1) / 2, y1 - 0.8, o["label"], 7.0, GRAY, align="center")
        elif sec["kind"] == "填空题":
            sh.draw_rect(R(sec["rect"]))
            sh.finish(color=INK, width=0.8)
            for sl in sec["slots"]:
                nx, ny = sl["num_xy"]
                _text(page, font, nx, ny, f"{sl['q']}.", 10.0)
                x0, y, x1 = sl["line"]
                sh.draw_line((x0 * MM, y * MM), (x1 * MM, y * MM))
            sh.finish(color=INK, width=0.6)
    for b in pg["boxes"]:
        sh.draw_rect(R(b["rect"]))
        sh.finish(color=INK, width=0.8)
        _text(page, font, b["rect"][0] + 2.5, b["rect"][1] + 5.5, b["label"], 10.0, bold=not b["cont"])
    sh.commit()


def render_pdf(plan: dict, exam_id: str, students: Optional[List[dict]] = None) -> bytes:
    """students 为空：通用版（每份都一样，可以复印）；否则每人一份（姓名、班级印好，学号涂好，二维码里有学号）。"""
    import pymupdf
    font = _font()
    W, H = plan["size"]
    n = len(plan["pages"])
    doc = pymupdf.open()
    for stu in (students or [None]):
        if stu is not None:
            sid = re.sub(r"\D", "", str(stu.get("sid") or ""))
            stu = dict(stu, sid_padded=sid.zfill(plan["id_digits"]) if 0 < len(sid) <= plan["id_digits"] else "")
        for p in range(n):
            page = doc.new_page(width=W * MM, height=H * MM)
            page.insert_font(fontname="cjk", fontbuffer=font.buffer)
            _draw_page(page, font, plan, p, qr_text(exam_id, p, n, plan["ver"], (stu or {}).get("sid", "")), stu)
    try:
        doc.subset_fonts()
    except Exception:  # 老版本 PyMuPDF 需要 fontTools 才能子集化：不子集化也能用，只是文件大一些
        pass
    return doc.tobytes(garbage=3, deflate=True)


# ================================================================ 排版计划 → 版面模板

def build_template(d, exam_id: str, plan: dict) -> dict:
    """把排版计划变成和“识别出来的版面”同样结构的模板：参考图、线条图、定位点、区域（含选项格、横线、学号格）。"""
    import pymupdf
    pdf = render_pdf(plan, exam_id)
    (d / "generated").mkdir(parents=True, exist_ok=True)
    (d / "generated" / f"answer_sheet_{plan['ver']}.pdf").write_bytes(pdf)
    W, H = plan["size"]
    long_mm = max(W, H)
    f = WORK_LONG_SIDE / long_mm                      # 模板像素 / mm
    zoom = WORK_LONG_SIDE / (long_mm * MM)
    pages = []
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        for p, page in enumerate(doc):
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            img = np.ascontiguousarray(cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            def T(x, y):
                return round(x * f, 1), round(y * f, 1)

            def rect(r):
                return [*T(r[0], r[1]), *T(r[2], r[3])]
            pp = plan["pages"][p]
            # 位置都是自己排的，不用再识别：定位点按计划位置，线条图（对齐质量检查用）直接按方框和横线画
            markers = [[*T(x, y), round(s * f, 2)] for x, y, s in pp["markers"]]
            imwrite(d / "template" / f"page_{p}.png", img)
            imwrite(d / "template" / f"page_{p}_lines.png", _line_mask(pp, img.shape[:2], f))
            regions = []

            def add(kind, r, questions=(), **extra):
                regions.append(dict({"id": f"R{len(regions) + 1}", "rect": [int(round(v)) for v in rect(r)], "kind": kind,
                                     "questions": list(questions), "full_score": None, "header": ""}, **extra))
                return regions[-1]
            info = pp.get("info")
            if info:
                reg = add("学生信息", info["rect"], header="姓名、班级、学号")
                grid = info.get("grid")
                if grid:
                    cols = []
                    for col in grid["cols"]:
                        g = {"pos": col["pos"], "rect": rect(col["rect"]),
                             "options": [{"label": o["label"], "rect": rect(o["rect"])} for o in col["options"]]}
                        cols.append(g)
                    gr = rect(grid["rect"])
                    paper = O.paper_level(gray, [int(v) for v in gr])
                    for g in cols:
                        for o, b in zip(g["options"], O.option_darkness(gray, g, paper)):
                            o["base"] = b
                    reg["id_grid"] = {"digits": grid["digits"], "rect": gr, "cols": cols}
            for sec in pp["sections"]:
                if sec["kind"] == "选择题":
                    groups = [{"q": g["q"], "src": "generated", "rect": rect(g["rect"]),
                               "options": [{"label": o["label"], "rect": rect(o["rect"])} for o in g["options"]]}
                              for g in sec["groups"]]
                    reg = add("选择题", sec["rect"], sec["questions"], groups=groups, header=sec["title"][:60])
                    paper = O.paper_level(gray, reg["rect"])
                    for g in groups:
                        for o, b in zip(g["options"], O.option_darkness(gray, g, paper)):
                            o["base"] = b
                elif sec["kind"] == "填空题":
                    slots = []
                    for sl in sec["slots"]:
                        x0, y, x1 = sl["line"]
                        a, b = T(x0, y)
                        c, _ = T(x1, y)
                        slots.append({"q": sl["q"], "src": "generated", "rect": rect(sl["rect"]),
                                      "line": [round(a), round(b), round(c)]})
                    add("填空题", sec["rect"], sec["questions"], slots=slots, header=sec["title"][:60])
            for bx in pp["boxes"]:
                add("解答题", bx["rect"], [bx["q"]], full_score=bx.get("score"), header=bx["label"])
            Wt, Ht = img.shape[1], img.shape[0]
            pages.append({"page": p, "image": f"template/page_{p}.png", "lines": f"template/page_{p}_lines.png",
                          "size": [Wt, Ht], "print_color": "black", "skew": 0.0, "markers": markers,
                          "regions": regions, "notes": [], "unboxed": "", "numbers": [], "labeled_by": "generated",
                          "qr": {"rect": rect(pp["qr"])}})
    return {"pages": pages, "source": "blank", "labeled_by": "generated", "version": 2,
            "generated": {"plan": plan, "ver": plan["ver"], "sig": plan["sig"], "paper": plan["paper"],
                          "id_digits": plan["id_digits"], "space": plan["space"], "title": plan["title"],
                          "pdf": f"generated/answer_sheet_{plan['ver']}.pdf", "at": time.strftime("%Y-%m-%d %H:%M:%S")}}


def _line_mask(pp: dict, shape, f: float) -> np.ndarray:
    """模板上的长线条（框线、横线），和识别版面时 find_boxes 画出来的线条图同样口径（线宽约 3 像素）。"""
    mask = np.zeros(shape, np.uint8)
    th = max(2, int(round(0.0015 * max(shape))))

    def box(r):
        cv2.rectangle(mask, (int(round(r[0] * f)), int(round(r[1] * f))), (int(round(r[2] * f)), int(round(r[3] * f))), 255, th)
    info = pp.get("info")
    if info:
        box(info["rect"])
        if info.get("grid"):
            box(info["grid"]["rect"])
    for sec in pp["sections"]:
        if sec.get("rect"):
            box(sec["rect"])
        for sl in sec.get("slots") or []:
            x0, y, x1 = sl["line"]
            cv2.line(mask, (int(round(x0 * f)), int(round(y * f))), (int(round(x1 * f)), int(round(y * f))), 255, th)
    for b in pp["boxes"]:
        box(b["rect"])
    return mask


# ================================================================ 读卡：二维码、学号

def _qr_detectors():
    dets = [cv2.QRCodeDetector()]
    if hasattr(cv2, "QRCodeDetectorAruco"):       # OpenCV 4.8+：找码更稳，手机照片上又小又斜的码也能读
        try:
            dets.append(cv2.QRCodeDetectorAruco())
        except cv2.error:
            pass
    return dets


def decode_qr_image(img: np.ndarray) -> str:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tries = [g]
    if max(g.shape) < 500:
        tries.append(cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tries.append(bw)
    for det in _qr_detectors():
        for t in tries:
            try:
                txt, pts, _ = det.detectAndDecode(t)
            except cv2.error:
                continue
            if txt:
                return txt
    return ""


def read_page_qr(scan: np.ndarray, H_t2s: np.ndarray, rect) -> Optional[dict]:
    """按模板上二维码的位置，从扫描原图上裁下来解码。"""
    from .align import crop_region
    w = rect[2] - rect[0]
    try:
        crop = crop_region(scan, H_t2s, rect, pad=0.12 * w, max_side=600, drop_red=False)
    except cv2.error:
        return None
    return parse_qr(decode_qr_image(crop))


def read_student_id(wgray: np.ndarray, region: dict) -> Optional[dict]:
    """涂的学号：每一位一列 0–9，按涂黑程度读（和选择题涂卡一样，不调用模型）。
    返回 {"id": "2507" 或带 ? 的, "complete": bool, "flags": [...]}。"""
    grid = region.get("id_grid")
    if not grid:
        return None
    paper = O.paper_level(wgray, [int(v) for v in grid["rect"]])
    digits, flags = [], []
    for k, col in enumerate(grid["cols"]):
        base = [o.get("base") for o in col["options"]]
        res = O.read_choice(wgray, col, paper, base if all(b is not None for b in base) else None)
        p = res["picked"]
        if len(p) == 1:
            digits.append(p)
        else:
            digits.append("?")
            flags.append(f"第 {k + 1} 位没涂" if not p else f"第 {k + 1} 位涂了 {'、'.join(p)} 几个数字")
        flags += [f"第 {k + 1} 位" + x.replace("选项", "数字") for x in res["flags"] if len(p) == 1 or not p]
    sid = "".join(digits)
    if set(sid) == {"?"}:
        return {"id": "", "complete": False, "flags": ["学号没涂"]}
    return {"id": sid, "complete": "?" not in sid, "flags": flags}
