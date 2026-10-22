"""端到端测试（不经过网页）：模拟 DeepSeek → 建考试 → 识别版面 → 批改多份扫描件。

python tests/e2e_core.py /path/to/sample.png
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from autograde.store import Store
from autograde.llm import DeepSeek, Usage
from autograde import exam_setup, pipeline, report
from autograde.imaging import imwrite, load_pages

PORT = 8765


def start_fake():
    import uvicorn
    import fake_deepseek
    cfg = uvicorn.Config(fake_deepseek.app, host="127.0.0.1", port=PORT, log_level="warning")
    srv = uvicorn.Server(cfg)
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(50):
        if srv.started:
            return srv
        time.sleep(0.1)
    raise RuntimeError("fake server not started")


def perturb(img, kind):
    h, w = img.shape[:2]
    if kind == "rot180":
        return cv2.rotate(cv2.resize(img, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_CUBIC), cv2.ROTATE_180)
    if kind == "skew":
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 1.3, 1.5)
        M[:, 2] += [(w * 1.5 - w) / 2 + 20, (h * 1.5 - h) / 2 + 20]
        return cv2.warpAffine(img, M, (int(w * 1.5) + 40, int(h * 1.5) + 40), borderValue=(255, 255, 255))
    if kind == "jpeg":
        out = cv2.resize(img, None, fx=1.6, fy=1.62, interpolation=cv2.INTER_LINEAR)
        ok, buf = cv2.imencode(".jpg", cv2.GaussianBlur(out, (3, 3), 0), [cv2.IMWRITE_JPEG_QUALITY, 40])
        return cv2.imdecode(buf, 1)
    raise ValueError(kind)


def make_pdf(path: Path, images):
    import pymupdf
    doc = pymupdf.open()
    for img in images:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        h, w = img.shape[:2]
        page = doc.new_page(width=w * 72 / 150, height=h * 72 / 150)
        page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(str(path))


def main(sample_path: str):
    srv = start_fake()
    tmp = Path(tempfile.mkdtemp(prefix="ag_e2e_"))
    store = Store(str(tmp / "data"))
    st = store.save_settings({"api_key": "sk-fake", "base_url": f"http://127.0.0.1:{PORT}", "samples": 2,
                              "max_workers": 4, "scan_dpi": 150})
    sample = load_pages(sample_path)[0]

    exam = store.create_exam("高二数学小练2（测试）", 1)
    eid = exam["id"]
    d = store.exam_dir(eid)
    # 试卷：随便一张图片做成 PDF（模拟服务器只认提示词）
    paper = np.full((1600, 1130, 3), 255, np.uint8)
    cv2.putText(paper, "PAPER", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 8)
    make_pdf(d / "paper" / "paper.pdf", [paper])
    shutil.copy(sample_path, d / "sheet" / "sheet.png")
    store.update_exam(eid, lambda e: e["files"].update(paper=["paper/paper.pdf"], sheet=["sheet/sheet.png"]))

    def prog(f, m):
        print(f"  [{f * 100:5.1f}%] {m}")

    llm = DeepSeek(st["api_key"], st["base_url"], cache_dir=str(d / "cache"), usage=Usage())
    t = time.time()
    exam = exam_setup.run_setup(store, eid, llm, st, prog)
    print("setup %.1fs" % (time.time() - t))
    sol = [q for q in exam["questions"] if q["type"] == "解答题"]
    for q in sol:
        print("  Q", q["number"], q["score"], [(p["label"], p["score"]) for p in q["parts"]], q["answer_source"],
              "stem:", q["stem"][:40])
    for pg in exam["template"]["pages"]:
        for r in pg["regions"]:
            print("  region", r["id"], r["kind"], r["questions"][:4], r["full_score"])
    print("  warnings:", exam["warnings"])
    assert len(sol) == 3, "should parse 3 解答题"
    assert "\\perp" in sol[0]["stem"], "LaTeX repair failed: " + sol[0]["stem"]

    # 扫描件：原图 + 倒放 + 倾斜 + 模糊 JPEG，其中两份和一张空白页放进一个 PDF（空白页应跳过）；
    # 再加一张别的纸（有字但不是这张答题纸，应对齐失败、单独成一份让老师人工批）
    scans = d / "scans"
    imwrite(scans / "001.png", sample)
    imwrite(scans / "002_rot180.jpg", perturb(sample, "rot180"))
    blank = np.clip(np.full((1300, 1800, 3), 246, np.float32) + np.random.default_rng(1).normal(0, 3, (1300, 1800, 1)),
                    0, 255).astype(np.uint8)
    make_pdf(scans / "003_batch.pdf", [perturb(sample, "skew"), blank, perturb(sample, "jpeg")])
    junk = np.full((1300, 1800, 3), 250, np.uint8)
    for k in range(14):
        cv2.putText(junk, f"Notice {k}: the library will be closed on Friday afternoon", (60, 90 + 85 * k),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (30, 30, 30), 3)
    imwrite(scans / "004_other.png", junk)
    files = ["scans/001.png", "scans/002_rot180.jpg", "scans/003_batch.pdf", "scans/004_other.png"]
    t = time.time()
    stats = pipeline.grade_scans(store, eid, llm, st, files, prog)
    print("grading %.1fs" % (time.time() - t), stats)
    res = report.results(store, eid)
    for r in res["rows"]:
        print("  ", r["id"], r["student"].get("name"), r["scores"], r["states"], "total", r["total"], "/", r["max_total"],
              "align_ok", r["align_ok"])
    print("  stats", res["stats"])
    print("  usage", llm.usage.snapshot())
    s1 = store.load_sheet(eid, "S001")
    q17 = s1["questions"]["17"]["ai"]
    print("  S001 Q17:", q17["score"], q17["parts"][1]["score"], q17["needs_review"], q17["review_notes"])
    assert q17["parts"][1]["score"] <= q17["parts"][1]["max"], "score clamp failed"
    assert len(res["rows"]) == 5, f"expected 5 sheets, got {len(res['rows'])}"
    assert res["rows"][4]["align_ok"] is False
    rep = store.load_exam(eid)["scan_report"]
    print("  scan report:", rep)
    assert rep["pages"] == 6 and rep["sheets"] == 5 and rep["skipped"] == 1, rep
    assert any("003_batch.pdf》第 2 页是空白页" in n for n in rep["notes"]), rep["notes"]
    assert any("004_other.png》第 1 页没能和答题纸对上" in n for n in rep["notes"]), rep["notes"]
    csv = (d / "scores.csv").read_text(encoding="utf-8-sig")
    print(csv)
    assert "请人工批改" in csv.splitlines()[5]
    # 老师改分
    s = pipeline.set_teacher_score(store, eid, "S002", "16", score=10, comment="(1)缺依据")
    print("  teacher override S002 Q16 ->", report.final_score(s["questions"]["16"]))
    print("annotated:", s["annotated"])

    # 改转写后只重评这一题
    pipeline.set_transcription(store, eid, "S001", "15", "(1) 证：……（老师改过的转写）")
    pipeline.regrade(store, eid, llm, st, ["S001"], ["15"])
    s = store.load_sheet(eid, "S001")
    assert s["questions"]["15"]["transcription_edited"] and "老师改过" in s["questions"]["15"]["transcription"]
    print("  edited transcription kept & regraded ->", s["questions"]["15"]["ai"]["score"])

    # 老师按小问改分
    s = pipeline.set_teacher_score(store, eid, "S003", "15", parts=[5, 4])
    assert report.final_score(s["questions"]["15"]) == 9
    try:
        pipeline.set_teacher_score(store, eid, "S003", "15", parts=[9, 4])
        raise AssertionError("should reject out-of-range part score")
    except ValueError as e:
        print("  rejected:", e)

    # 批改之后改版面：第17题的作答跨两个框（R4 也写了第17题）→ 重新批改会按新版面重新裁切
    exam = store.load_exam(eid)
    pages = exam["template"]["pages"]
    for r in pages[0]["regions"]:
        if r["id"] == "R4":
            r["questions"] = ["16", "17"]
    exam_setup.save_template_edit(store, eid, [{"page": 0, "regions": pages[0]["regions"]}])
    stats = pipeline.regrade(store, eid, llm, st)
    s2 = store.load_sheet(eid, "S002")
    print("  after layout edit:", stats, "Q17 crops:", s2["questions"]["17"]["crops"])
    assert stats["recropped"] == 5 and len(s2["questions"]["17"]["crops"]) == 2
    assert report.final_score(s2["questions"]["16"]) == 10, "teacher override lost after recrop"
    print("DATA DIR", tmp)
    srv.should_exit = True


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/work/sample.png")
