"""生成一份和样张完全不同版式的答题卡（A4 竖版、正反两面、实心方块定位点），用来测试“换一套答题纸也能自动识别”。"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
W, H = A4

# 每个解答题框：(页, 题号, 分值, x, y_top, w, h)，单位 mm，y 从上往下
SOLUTION_BOXES = [
    (0, "17", 10, 12, 150, 186, 62),
    (0, "18", 12, 12, 216, 186, 66),
    (1, "19", 12, 12, 14, 186, 88),
    (1, "20", 12, 12, 106, 186, 84),
    (1, "21", 14, 12, 194, 186, 90),
]


def _box(c, x, y_top, w, h, lw=0.9):
    c.setLineWidth(lw)
    c.rect(x * mm, H - (y_top + h) * mm, w * mm, h * mm)


def _marker(c, x, y_top, s=5):
    c.setFillColorRGB(0, 0, 0)
    c.rect(x * mm, H - (y_top + s) * mm, s * mm, s * mm, fill=1, stroke=0)


def make(path: Path) -> list:
    c = canvas.Canvas(str(path), pagesize=A4)
    for page in (0, 1):
        # 四角实心定位块 + 一个不对称的辅助块（第1页在右侧中部，第2页在左侧中部）
        for (x, y) in [(6, 5), (199, 5), (6, 287), (199, 287)]:
            _marker(c, x, y)
        _marker(c, 199 if page == 0 else 6, 140)
        c.setFont("STSong-Light", 9)
        c.drawCentredString(W / 2, 6 * mm, f"第{page + 1}页（共2页）")
        if page == 0:
            c.setFont("STSong-Light", 16)
            c.drawCentredString(W / 2, H - 22 * mm, "高一数学期中考试 答题卡")
            c.setFont("STSong-Light", 11)
            c.drawString(16 * mm, H - 34 * mm, "姓名：____________  班级：__________  考号：________________")
            # 选择题
            c.setFont("STSong-Light", 11)
            c.drawString(12 * mm, H - 44 * mm, "一、选择题（每小题5分，共40分）")
            _box(c, 12, 47, 186, 30)
            c.setFont("STSong-Light", 10)
            for k in range(8):
                col, row = k % 4, k // 4
                c.drawString((18 + col * 46) * mm, H - (57 + row * 12) * mm, f"{k + 1}  [A] [B] [C] [D]")
            # 填空题
            c.setFont("STSong-Light", 11)
            c.drawString(12 * mm, H - 84 * mm, "二、填空题（每小题5分，共20分）")
            _box(c, 12, 87, 186, 52)
            for k in range(4):
                col, row = k % 2, k // 2
                y = H - (106 + row * 22) * mm
                c.drawString((18 + col * 93) * mm, y + 1 * mm, f"{k + 9}.")
                c.line((25 + col * 93) * mm, y, (100 + col * 93) * mm, y)
            c.drawString(12 * mm, H - 146 * mm, "三、解答题（共60分）")
        for (pg, num, sc, x, yt, w, h) in SOLUTION_BOXES:
            if pg != page:
                continue
            _box(c, x, yt, w, h)
            c.setFont("STSong-Light", 11)
            c.drawString((x + 3) * mm, H - (yt + 6) * mm, f"{num}.（{sc}分）")
        c.showPage()
    c.save()
    return SOLUTION_BOXES


if __name__ == "__main__":
    import sys
    make(Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_sheet.pdf"))
