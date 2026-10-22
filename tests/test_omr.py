"""涂卡识别：用真实扫描的答题纸（铅笔涂卡）检查选项框定位和识别。

python tests/test_omr.py 扫描样张.png  期望的答案（如 DCBB）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
from autograde import objective as O
from autograde.imaging import load_pages
from autograde.layout import analyze_page


def main(path: str, expect: str):
    lay = analyze_page(load_pages(path)[0])
    hits = []
    for rect in lay.boxes:
        groups = O.detect_choice_groups(lay.image, rect, None, [])
        if len(groups) >= 2 and all(len(g["options"]) == 4 for g in groups):
            hits.append((rect, groups))
    assert hits, "没有找到选择题区域"
    rect, groups = hits[0]
    g = cv2.cvtColor(lay.image, cv2.COLOR_BGR2GRAY)
    paper = O.paper_level(g, rect)
    got = "".join(O.read_choice(g, gr, paper, None)["picked"] or "-" for gr in groups)
    print("选择题区域", rect, "识别结果", got)
    assert got == expect, (got, expect)
    # 填空横线（扫描件没有文字层，按先行后列排）
    for rect2 in lay.boxes:
        slots = O.detect_fill_slots(lay.image, rect2, None, [str(i) for i in range(1, 11)])
        if len(slots) >= 6:
            print("填空区域", rect2, "横线", len(slots))
            assert len(slots) == 10
            break
    else:
        raise AssertionError("没有找到填空横线")
    print("OMR OK")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
