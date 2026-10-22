"""合并 / 删除答卷（不需要模型、不需要图片内容）：补扫的缺页并进原来那份；跨页的题按页序拼图并重批；
老师给过的分保留；同一页不能合并；删掉的编号不再使用；“最近一次上传”的提醒标成已处理。"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autograde import pipeline
from autograde.imaging import imwrite
from autograde.store import Store


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ag_merge_"))
    store = Store(str(tmp / "data"))
    exam = store.create_exam("合并测试", 2)
    eid = exam["id"]
    d = store.exam_dir(eid)
    region = lambda rid, q, y: {"id": rid, "kind": "解答题", "questions": [q], "rect": [100, y, 900, y + 400], "full_score": None}
    tmpl = {"source": "blank", "pages": [
        {"page": 0, "size": [1000, 1400], "regions": [region("R1", "21", 900)], "markers": []},
        {"page": 1, "size": [1000, 1400], "regions": [region("R2", "20", 100), region("R3", "21", 700)], "markers": []}]}

    def setup(e):
        e["template"] = tmpl
        e["questions"] = [{"number": "20", "type": "解答题", "score": 16}, {"number": "21", "type": "解答题", "score": 18}]
        e["sheets"] = ["S001", "S002", "S003"]
        e["sheet_seq"] = 3
        e["scan_report"] = {"pages": 2, "sheets": 1, "partial": 1, "sheet_ids": ["S002"], "skipped": 0,
                            "notes": ["S002（从扫描件第 1 页开始）缺第 1 页，这几页上的题没有批改。如果这是补扫的缺页，打开 S001 把它合并进去"]}
    store.update_exam(eid, setup)
    blank = np.full((140, 100, 3), 255, np.uint8)

    def files(sid, names):
        for n in names:
            imwrite(d / "work" / sid / n, blank)
        return [f"work/{sid}/{n}" for n in names]

    # S001：只有正面（第 21 题的前半段，老师已经给第 20 题打了 0 分，因为当时缺页）
    f1 = files("S001", ["page_0.jpg", "q21_p0_R1.jpg", "student_page.jpg"])
    store.save_sheet(eid, {"id": "S001", "pages": [{"file": "scans/a.pdf", "index": 0, "template_page": 0, "display": f1[0]}],
                           "files": [{"file": "scans/a.pdf", "index": 0}], "align_ok": True, "template_hash": "h",
                           "student": {"name": "张三"}, "missing_pages": [1], "annotated": [],
                           "questions": {"20": {"number": "20", "type": "解答题", "max": 16, "crops": [], "status": "missing",
                                                "teacher": {"score": 0}},
                                         "21": {"number": "21", "type": "解答题", "max": 18, "crops": [f1[1]], "status": "done",
                                                "transcription": "前半段", "ai": {"score": 5, "max": 18}}}})
    # S002：补扫的反面（第 20 题整题、第 21 题后半段，都已经批过）
    f2 = files("S002", ["page_1.jpg", "q20_p1_R2.jpg", "q21_p1_R3.jpg"])
    store.save_sheet(eid, {"id": "S002", "pages": [{"file": "scans/b.pdf", "index": 0, "template_page": 1, "display": f2[0]}],
                           "files": [{"file": "scans/b.pdf", "index": 0}], "align_ok": True, "template_hash": "h",
                           "student": {"name": "", "qr": ["x"]}, "missing_pages": [0], "annotated": [],
                           "questions": {"20": {"number": "20", "type": "解答题", "max": 16, "crops": [f2[1]], "status": "done",
                                                "transcription": "整题", "ai": {"score": 12, "max": 16}},
                                         "21": {"number": "21", "type": "解答题", "max": 18, "crops": [f2[2]], "status": "done",
                                                "transcription": "后半段", "ai": {"score": 7, "max": 18}}}})
    # S003：完整的另一份
    f3 = files("S003", ["page_0.jpg", "page_1.jpg"])
    store.save_sheet(eid, {"id": "S003", "pages": [{"file": "scans/a.pdf", "index": 1, "template_page": 0, "display": f3[0]},
                                                   {"file": "scans/a.pdf", "index": 2, "template_page": 1, "display": f3[1]}],
                           "files": [], "align_ok": True, "template_hash": "h", "student": {}, "missing_pages": [],
                           "annotated": [], "questions": {}})

    # 同一页不能合并
    try:
        pipeline.merge_sheets(store, eid, "S001", "S003")
        raise AssertionError("S001 和 S003 都有正面，应该拒绝")
    except ValueError as e:
        assert "都有正面" in str(e), e

    merged, todo = pipeline.merge_sheets(store, eid, "S001", "S002")
    q = merged["questions"]
    assert [p["template_page"] for p in merged["pages"]] == [0, 1] and merged["missing_pages"] == []
    assert merged["pages"][1]["display"] == "work/S001/page_1.jpg" and (d / "work/S001/page_1.jpg").exists()
    assert q["20"]["crops"] == ["work/S001/q20_p1_R2.jpg"] and q["20"]["ai"]["score"] == 12, q["20"]
    assert q["20"]["teacher"] == {"score": 0}, "老师给过的分要保留"
    assert q["21"]["crops"] == ["work/S001/q21_p0_R1.jpg", "work/S001/q21_p1_R3.jpg"], q["21"]["crops"]
    assert q["21"]["status"] == "pending" and "ai" not in q["21"] and "transcription" not in q["21"]
    assert todo == ["21"], todo
    assert merged["student"]["name"] == "张三" and merged["student"]["qr"] == ["x"]
    assert merged["merged_from"] == ["S002"] and len(merged["annotated"]) == 2
    assert all((d / a).exists() for a in merged["annotated"])
    assert not (d / "work" / "S002").exists()
    e = store.load_exam(eid)
    assert e["sheets"] == ["S001", "S003"], e["sheets"]
    assert e["scan_report"]["notes"][0].startswith("✓ S002") and e["scan_report"]["notes"][0].endswith("（已并进 S001）")
    assert e["scan_report"]["partial"] == 0
    csv = (d / "scores.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(csv) == 3 and "缺" not in csv[1], csv

    # 删除；之后新答卷的编号从 S004 开始（不重用 S002、S003）
    pipeline.delete_sheet(store, eid, "S003")
    e = store.load_exam(eid)
    assert e["sheets"] == ["S001"] and not (d / "work" / "S003").exists()
    assert pipeline._last_seq(d, e) == 3
    try:
        pipeline.delete_sheet(store, eid, "S003")
        raise AssertionError("删不存在的答卷应该报 KeyError")
    except KeyError:
        pass
    print("MERGE OK", tmp)


if __name__ == "__main__":
    main()
