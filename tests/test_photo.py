"""手机拍的答题纸（不用扫描仪）：斜着拍、纸面弯曲、光线不匀、周围露出桌面，Mac 转成 PDF（页面 1.4 米宽）。

核对：
- PDF 页面按照片本来的分辨率读，不会放大成上亿像素；
- 对齐时把弯曲的纸面拉平，每页都能对上（不再是“没能和答题纸对上，请人工批改”）；
- 学号、选择题涂卡读对；填空、解答题都裁出来；正反面认对（两页的框有不少是重合的）；
- 手机照片上又小又斜的二维码也能读出来；
- 一叠里混着正常扫描件和照片也行；照片的版面快速路径之外的扫描件不受影响。

python tests/test_photo.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_sheetgen as T            # 渲染、模拟作答、模拟扫描
from photo_sim import phone_photo, photos_to_pdf
from autograde import exam_setup, pipeline, sheetgen
from autograde.imaging import count_pages, iter_pages
from autograde.store import Store

OUT = Path(tempfile.gettempdir()) / "autograde_photo"
STRUCT = "1-10 填空 5分\n11-14 单选 5分\n15-17 解答 10分"
KEY = {"11": "C", "12": "D", "13": "C", "14": "B"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ag_photo_"))
    store = Store(str(tmp / "data"))
    st = store.save_settings({"api_key": "", "scan_dpi": 250})
    rng = np.random.default_rng(7)
    qs, bad = exam_setup.parse_structure(STRUCT)
    for q in qs:
        q["answer"] = KEY.get(q["number"], "1" if q["type"] == "填空题" else "略")
    ex = store.create_exam("高二下数学小练习3", 1, "quiz")
    eid = ex["id"]
    store.update_exam(eid, lambda e: e.__setitem__("questions", qs))
    # 和用户那张一样的版面：A3 两面，15 题跨两栏，16、17 各占一栏
    ex = exam_setup.generate_sheet(store, eid, "A3", 6, "normal", heights={"15": 306, "16": 244, "17": 244})
    plan = ex["template"]["generated"]["plan"]
    assert len(plan["pages"]) == 2
    blank = T.render(sheetgen.render_pdf(plan, eid), 200)       # 手机照片里 A3 纸大约 200dpi

    picks = [{"11": "C", "12": "D", "13": "C", "14": "B"}, {"11": "A", "12": "D", "13": "", "14": "B"}]
    sids = ["250931", "250917"]
    photos = []
    for k, (sid, ch) in enumerate(zip(sids, picks)):
        pages = T.fill_sheet(blank, plan, rng, sid, ch, dpi=200)
        for p, img in enumerate(pages):
            photos.append(phone_photo(img, rng, bend=0.010 + 0.004 * k, tilt=0.03 + 0.01 * p, rotate=(-1) ** p * 1.5))
    cv2.imwrite(str(OUT / "photo_p1.jpg"), cv2.resize(photos[0], None, fx=0.3, fy=0.3))
    d = store.exam_dir(eid)
    photos_to_pdf(d / "scans" / "IMG_2284.pdf", photos)

    # 读 PDF：按照片原本的分辨率，不放大
    shapes = [img.shape for _, img, _ in iter_pages(d / "scans" / "IMG_2284.pdf", 250)]
    assert count_pages(d / "scans" / "IMG_2284.pdf") == 4 and all(s[:2] == (3024, 4032) for s in shapes), shapes

    t = time.time()
    got = pipeline.prepare_sheets(store, eid, ["scans/IMG_2284.pdf"], st, lambda f, m: None)
    secs = time.time() - t
    rep = store.load_exam(eid)["scan_report"]
    print(f"prepare {secs:.1f}s sheets {got} notes {rep['notes']}")
    assert got == ["S001", "S002"] and not rep["notes"], (got, rep["notes"])
    for sid, want, ch in zip(got, sids, picks):
        s = store.load_sheet(eid, sid)
        al = [(p["template_page"], p["align"]["method"], p["align"]["quality"]) for p in s["pages"]]
        print(sid, "align", al, "id", s["student"].get("omr"))
        assert s["align_ok"] and [p for p, _, _ in al] == [0, 1], al
        assert all("flatten" in m and q >= 0.8 for _, m, q in al), al
        assert s["student"]["omr"]["id"] == want, s["student"]["omr"]
        read = {n: s["questions"][n]["student"] for n in ch}
        assert read == ch, (read, ch)
        assert all(s["questions"][str(n)]["crops"] for n in range(1, 18)), sid
        assert len(s["questions"]["15"]["crops"]) == 2
    shutil.copy(d / "work" / "S001" / "page_0.jpg", OUT / "flattened_p1.jpg")
    shutil.copy(d / "work" / "S001" / "q15_p0_R5.jpg", OUT / "q15_part2.jpg")

    # 二维码：对不上版面时整页找（手机照片上的码又小又斜）
    qr = pipeline._page_qr(photos[1], None, None)
    assert qr and qr["exam"] == eid and qr["page"] == 1, qr

    # 一叠里混着扫描件和照片：扫描件走快速路径（不拉平）
    scan = [T.scan(img, rng) for img in T.fill_sheet(blank, plan, rng, "250903", {"11": "B"}, dpi=200)]
    T.to_pdf(d / "scans" / "mixed.pdf", scan + photos[2:4], dpi=200)
    got2 = pipeline.prepare_sheets(store, eid, ["scans/mixed.pdf"], st, lambda f, m: None)
    s3, s4 = (store.load_sheet(eid, x) for x in got2)
    m3 = [p["align"]["method"] for p in s3["pages"]]
    m4 = [p["align"]["method"] for p in s4["pages"]]
    print("mixed:", got2, m3, m4)
    assert len(got2) == 2 and all("flatten" not in m for m in m3) and all("flatten" in m for m in m4), (m3, m4)
    assert s3["student"]["omr"]["id"] == "250903" and s4["student"]["omr"]["id"] == "250917"
    # iPhone 的 HEIC 照片（装了 pillow-heif 才支持）
    from autograde.imaging import SUPPORTED_EXTS
    if ".heic" in SUPPORTED_EXTS:
        from PIL import Image
        Image.fromarray(cv2.cvtColor(photos[0], cv2.COLOR_BGR2RGB)).save(d / "scans" / "IMG_0001.heic", quality=90)
        got3 = pipeline.prepare_sheets(store, eid, ["scans/IMG_0001.heic"], st, lambda f, m: None)
        s5 = store.load_sheet(eid, got3[0])
        assert s5["align_ok"] and s5["student"]["omr"]["id"] == sids[0], (s5["pages"], s5["student"])
        print("HEIC OK:", got3, s5["pages"][0]["align"]["method"])
    else:
        print("（没装 pillow-heif，跳过 HEIC）")
    print("\nPHOTO OK", tmp, "images in", OUT)


if __name__ == "__main__":
    main()
