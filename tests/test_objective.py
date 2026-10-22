"""选择题、填空题批改的端到端测试（用真实的试卷/答案/答题纸 PDF，学生答卷是模拟的扫描件）。

python tests/test_objective.py 试卷.pdf 答案.pdf 答题纸.pdf
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

from autograde import exam_setup, pipeline, report
from autograde.exam_setup import objective_places
from autograde.imaging import load_pages
from autograde.layout import analyze_page
from autograde.llm import DeepSeek, Usage
from autograde.store import Store

PORT = 8773
DPI = 200


def start_fake():
    import uvicorn
    import fake_deepseek
    srv = uvicorn.Server(uvicorn.Config(fake_deepseek.app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    return srv, fake_deepseek


def make_student(sheet_pdf: str, tmpl: dict, out_pdf: Path):
    """在空白答题纸上“写”填空、涂选择题、写解答题，再轻微旋转缩放、JPEG 压缩，存成两页 PDF。"""
    pages = load_pages(sheet_pdf, dpi=DPI)
    places = objective_places(tmpl)
    outs = []
    for p, img in enumerate(pages):
        lay = analyze_page(img)
        inv = np.linalg.inv(np.vstack([lay.to_tmpl, [0, 0, 1]]))

        def T(x, y):
            v = inv @ [x, y, 1]
            return int(v[0]), int(v[1])
        stu = img.copy()
        for n, pl in places.items():
            if pl["page"] != p:
                continue
            if pl.get("slot") and n != "12":                  # 第 12 题留空
                x0, y0, x1, y1 = pl["slot"]["rect"]
                cv2.putText(stu, f"ans{n}", T(x0 + 40, y1 - 14), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.3, (30, 30, 30), 3)
            if pl.get("group"):
                marks = {"13": [("B", 60)], "14": [("A", 70)], "15": [("C", 178)], "16": [("B", 60), ("D", 65)]}[n]
                for letter, level in marks:
                    o = next(o for o in pl["group"]["options"] if o["label"] == letter)
                    a, b = T(o["rect"][0] + 1, o["rect"][1] + 1)
                    c, d = T(o["rect"][2] - 1, o["rect"][3] - 1)
                    stu[b:d, a:c] = np.minimum(stu[b:d, a:c], level)
        for pg in tmpl["pages"]:
            if pg["page"] != p:
                continue
            for r in pg["regions"]:
                if r["kind"] == "解答题":
                    x0, y0, x1, y1 = r["rect"]
                    for k in range(3):
                        cv2.putText(stu, f"solution line {k + 1}", T(x0 + 60, y0 + 90 + 60 * k),
                                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (30, 30, 30), 2)
        M = cv2.getRotationMatrix2D((stu.shape[1] / 2, stu.shape[0] / 2), 0.7 if p == 0 else -0.5, 0.97)
        scan = cv2.warpAffine(stu, M, (stu.shape[1], stu.shape[0]), borderValue=(255, 255, 255))
        ok, buf = cv2.imencode(".jpg", scan, [cv2.IMWRITE_JPEG_QUALITY, 70])
        outs.append(cv2.imdecode(buf, 1))
    import pymupdf
    doc = pymupdf.open()
    for im in outs:
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        h, w = im.shape[:2]
        page = doc.new_page(width=w * 72 / DPI, height=h * 72 / DPI)
        page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(str(out_pdf))


def main(paper: str, answers: str, sheet: str):
    srv, fake = start_fake()
    tmp = Path(tempfile.mkdtemp(prefix="ag_obj_"))
    store = Store(str(tmp / "data"))
    st = store.save_settings({"api_key": "sk-fake", "base_url": f"http://127.0.0.1:{PORT}", "max_workers": 6,
                              "scan_dpi": DPI, "samples": 1})
    exam = store.create_exam("客观题测试", 2)
    eid = exam["id"]
    d = store.exam_dir(eid)
    for sub, src in (("paper", paper), ("answers", answers), ("sheet", sheet)):
        shutil.copy(src, d / sub / f"{sub}.pdf")
    store.update_exam(eid, lambda e: e["files"].update(paper=["paper/paper.pdf"], answers=["answers/answers.pdf"],
                                                       sheet=["sheet/sheet.pdf"]))
    llm = DeepSeek(st["api_key"], st["base_url"], cache_dir=str(d / "cache"), usage=Usage())
    exam = exam_setup.run_setup(store, eid, llm, st, lambda f, m: None, generate_missing=False)
    print("graded numbers:", report.graded_numbers(exam))

    make_student(sheet, exam["template"], d / "scans" / "stu1.pdf")
    t = time.time()
    stats = pipeline.grade_scans(store, eid, llm, st, ["scans/stu1.pdf"], lambda f, m: None)
    print(f"grading {time.time() - t:.1f}s", stats, "calls", fake.STATE["by_task"])
    s = store.load_sheet(eid, "S001")
    qs = s["questions"]
    for n in [str(i) for i in range(1, 17)]:
        r = qs[n]
        ai = r.get("ai") or {}
        print(f"  {n:>2} {r['type']} 学生={r.get('student')!r:28} 得分={ai.get('score')}/{ai.get('max')} {ai.get('reason', '')} "
              f"{'【复核】' + ai.get('review_notes', '') if ai.get('needs_review') else ''} ink={r.get('ink')}")
    res = report.results(store, eid)
    row = res["rows"][0]
    print("sections:", row["sections"], "total:", row["total"], "section max:", res["section_max"])
    csv = (d / "scores.csv").read_text(encoding="utf-8-sig").splitlines()
    print(csv[0])
    print(csv[1])

    # ---- 断言
    asked = [n for call in fake.STATE.get("fill_asked", []) for n in call]
    assert "12" not in asked and sorted(asked, key=int) == [str(i) for i in range(1, 12)], asked
    assert fake.STATE["by_task"].get("fill_read") == 1, "填空题应一次识别整张卷子的所有空"
    assert fake.STATE["by_task"].get("fill_judge") == 1
    sc = {n: qs[n]["ai"]["score"] for n in qs if qs[n].get("type") in ("填空题", "选择题")}
    exp_fill = {"1": 4, "2": 4, "3": 0, "4": 4, "5": 0, "6": 4, "7": 5, "8": 5, "9": 5, "10": 5, "11": 5, "12": 0}
    exp_choice = {"13": 4, "14": 0, "15": 5, "16": 0}
    assert {n: sc[n] for n in exp_fill} == exp_fill, {n: sc[n] for n in exp_fill}
    assert {n: sc[n] for n in exp_choice} == exp_choice, {n: sc[n] for n in exp_choice}
    assert [qs[n]["student"] for n in ("13", "14", "15", "16")] == ["B", "A", "C", "BD"]
    assert qs["15"]["ai"]["needs_review"] and qs["16"]["ai"]["needs_review"] and not qs["13"]["ai"]["needs_review"]
    assert qs["12"]["ai"]["reason"] == "未作答" and qs["1"]["ai"]["method"] == "exact" and qs["9"]["ai"]["method"] == "judge"
    assert row["sections"]["填空题"] == 41 and row["sections"]["选择题"] == 9, row["sections"]
    assert res["section_max"] == {"选择题": 18.0, "填空题": 54.0, "解答题": 78.0}, res["section_max"]
    assert "选择题合计" in csv[0] and "填空题合计" in csv[0] and "总分" in csv[0]

    # ---- 老师更正：第 15 题确认是 C（不再复核）；第 14 题其实涂的是 B（改后变对）；第 3 题老师判对
    s = pipeline.set_student_answer(store, eid, "S001", "15", "C")
    assert not s["questions"]["15"]["ai"]["needs_review"] and s["questions"]["15"]["ai"]["score"] == 5
    s = pipeline.set_student_answer(store, eid, "S001", "14", "C")
    assert s["questions"]["14"]["ai"]["score"] == 4, s["questions"]["14"]["ai"]
    s = pipeline.set_teacher_score(store, eid, "S001", "3", score=4)
    row = report.results(store, eid)["rows"][0]
    assert row["sections"]["填空题"] == 45 and row["sections"]["选择题"] == 13, row["sections"]
    # ---- 改了参考答案后重新给分：不重新识别，只重新判
    def change(e):
        for q in e["questions"]:
            if q["number"] == "13":
                q["answer"] = "A"
    store.update_exam(eid, change)
    before = dict(fake.STATE["by_task"])
    pipeline.regrade(store, eid, llm, st, ["S001"], ["13", "5"])
    s = store.load_sheet(eid, "S001")
    assert s["questions"]["13"]["ai"]["score"] == 0, "改了参考答案后应重新给分"
    assert fake.STATE["by_task"].get("fill_read") == before.get("fill_read"), "重新给分不应重新识别填空"
    shutil.copy(d / s["annotated"][0], "/home/claude/work/scratch/obj_annotated.jpg")
    print("\nOBJECTIVE OK", tmp)
    srv.should_exit = True


if __name__ == "__main__":
    main(*sys.argv[1:4])
