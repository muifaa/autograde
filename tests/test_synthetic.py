"""换一套完全不同的答题卡：自动识别版面（不用 AI，走兜底规则）→ 正反两面的学生扫描件 → 对齐、分组、裁切。"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from autograde import exam_setup, pipeline
from autograde.exam_setup import new_question
from autograde.imaging import imwrite, load_pages
from autograde.store import Store
import synthetic_sheet


def scribble(img, rect_px, text, seed):
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = rect_px
    y = y0 + int(0.25 * (y1 - y0))
    for k in range(4):
        cv2.putText(img, f"{text} line {k + 1}: x = {rng.integers(2, 99)}", (x0 + 40, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    1.3, (40, 30, 20), 2, cv2.LINE_AA)
        y += int(0.17 * (y1 - y0))


def transform(img, kind):
    h, w = img.shape[:2]
    if kind == "slight":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -0.8, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255)), np.vstack([M, [0, 0, 1]])
    if kind == "rot180":
        M = np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1.0]])
        return cv2.rotate(img, cv2.ROTATE_180), M
    if kind == "small_jpeg":
        M = np.diag([0.8, 0.8, 1.0])
        out = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 45])
        return cv2.imdecode(buf, 1), M
    if kind == "persp":
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[18, 6], [w - 4, 22], [w - 25, h - 3], [8, h - 20]])
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w, h), borderValue=(255, 255, 255)), M
    raise ValueError(kind)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ag_syn_"))
    pdf = tmp / "sheet.pdf"
    boxes = synthetic_sheet.make(pdf)
    store = Store(str(tmp / "data"))
    st = store.settings()
    exam = store.create_exam("合成答题卡测试", pages_per_sheet=2)
    eid = exam["id"]
    qs = []
    for (_, num, sc, *_r) in boxes:
        q = new_question(num, "解答题")
        q["score"] = sc
        q["answer"] = "（略）"
        qs.append(q)
    store.update_exam(eid, lambda e: e.__setitem__("questions", qs))

    # 先按“扫描件”处理（没有文字层，没有 AI）：走兜底规则
    blank = [(img, None, 150) for img in load_pages(pdf, dpi=150)]
    tmpl0 = exam_setup.build_template(store, eid, blank, qs, None, st, "blank")
    print("no-text template labeled_by:", tmpl0["labeled_by"])
    got0 = {r["questions"][0]: pg["page"] for pg in tmpl0["pages"] for r in pg["regions"] if r["questions"]}
    assert sorted(got0) == ["17", "18", "19", "20", "21"], got0
    # 数字版 PDF：直接用文字层标题号
    tmpl = exam_setup.build_template(store, eid, exam_setup.load_sheet_sources([pdf], 150), qs, None, st, "blank")
    store.update_exam(eid, lambda e: e.__setitem__("template", tmpl))
    print("template labeled_by:", tmpl["labeled_by"])
    assert tmpl["labeled_by"] == "pdf_text", tmpl["labeled_by"]
    scores = {r["questions"][0]: r["full_score"] for pg in tmpl["pages"] for r in pg["regions"] if r["kind"] == "解答题"}
    assert scores == {"17": 10, "18": 12, "19": 12, "20": 12, "21": 14}, scores
    for pg in tmpl["pages"]:
        print(f" page {pg['page']}: markers={len(pg['markers'])} notes={pg['notes']}")
        for r in pg["regions"]:
            print("   ", r["id"], r["kind"], r["questions"], r["rect"])
    labeled = {r["questions"][0]: (pg["page"], r) for pg in tmpl["pages"] for r in pg["regions"]
               if r["questions"] and r["kind"] == "解答题"}
    assert sorted(labeled) == ["17", "18", "19", "20", "21"], labeled
    # 选择题一行并排 4 道（2 行共 8 道），填空题 2×2：都要拆到每道题
    from autograde.exam_setup import objective_places
    places = objective_places(tmpl)
    assert sorted((n for n, p in places.items() if p.get("group")), key=int) == [str(i) for i in range(1, 9)], places.keys()
    assert all(len(places[str(i)]["group"]["options"]) == 4 for i in range(1, 9))
    assert sorted((n for n, p in places.items() if p.get("slot")), key=int) == ["9", "10", "11", "12"]
    print("objective places OK: choice 1-8 (4 per row), fill 9-12")
    for (pg, num, *_r) in boxes:
        assert labeled[num][0] == pg, f"question {num} on wrong page"

    # 学生扫描件：200dpi，正面 + 背面；学生甲背面倒放
    pages200 = load_pages(pdf, dpi=200)
    f = 200 / 25.4
    scans = []
    for sidx, kinds in enumerate([("slight", "rot180"), ("small_jpeg", "persp")]):
        for p in (0, 1):
            img = pages200[p].copy()
            for (pg, num, sc, x, yt, w, h) in boxes:
                if pg == p:
                    scribble(img, (int(x * f), int(yt * f), int((x + w) * f), int((yt + h) * f)), f"S{sidx + 1} Q{num}", sidx * 10 + p)
            out, M = transform(img, kinds[p])
            name = f"scans/stu{sidx + 1}_p{p + 1}.jpg"
            imwrite(store.exam_dir(eid) / name, out, quality=90)
            scans.append((name, p, M))
    files = [s[0] for s in scans]
    sids = pipeline.prepare_sheets(store, eid, files, st)
    print("sheets:", sids)
    assert len(sids) == 2, sids
    # 检查每页对齐误差（模板坐标 -> 扫描件坐标，与真实变换比较）
    exam = store.load_exam(eid)
    d = store.exam_dir(eid)
    for sid in sids:
        s = store.load_sheet(eid, sid)
        for prec in s["pages"]:
            print(f"  {sid} {prec['file']} -> 模板第{prec['template_page'] + 1}页 align={prec['align']}")
        for n, qr in s["questions"].items():
            assert qr["crops"], f"{sid} Q{n} 没有裁切图"
    # 拼一张裁切结果总览图，人工看一眼
    tiles = []
    for sid in sids:
        s = store.load_sheet(eid, sid)
        for n in ["17", "18", "19", "20", "21"]:
            c = cv2.imread(str(d / s["questions"][n]["crops"][0]))
            c = cv2.resize(c, (700, int(700 * c.shape[0] / c.shape[1])))
            tiles.append(cv2.copyMakeBorder(c, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(0, 0, 255)))
    width = max(t.shape[1] for t in tiles)
    grid = np.vstack([cv2.copyMakeBorder(t, 0, 0, 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles])
    out = Path("/home/claude/work/scratch/synthetic_crops.jpg")
    imwrite(out, cv2.resize(grid, None, fx=0.5, fy=0.5))
    shutil.copy(d / "template" / "page_1_preview.jpg", "/home/claude/work/scratch/synthetic_preview_p2.jpg")
    print("OK", tmp)


if __name__ == "__main__":
    main()
