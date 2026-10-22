"""用真实的试卷/答案/答题纸 PDF 走一遍“建考试”（模型用 tests/fake_deepseek.py 模拟）。

python tests/test_real_pdfs.py 试卷.pdf 答案.pdf 答题纸.pdf
检查：题号切分、题干包含小问、分值（试卷说明、答案里的……X分、答题纸印的分值三方一致）、
答案只含本题内容（模拟服务器会给第 18 题故意多写一段别的题，必须被标出来）、答题纸用文字层标题号。
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from autograde import exam_setup
from autograde.llm import DeepSeek, Usage
from autograde.store import Store

PORT = 8771


def start_fake():
    import uvicorn
    import fake_deepseek
    srv = uvicorn.Server(uvicorn.Config(fake_deepseek.app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    return srv, fake_deepseek


def main(paper: str, answers: str, sheet: str):
    srv, fake = start_fake()
    tmp = Path(tempfile.mkdtemp(prefix="ag_real_"))
    store = Store(str(tmp / "data"))
    st = store.save_settings({"api_key": "sk-fake", "base_url": f"http://127.0.0.1:{PORT}", "max_workers": 6})
    exam = store.create_exam("高一数学（真实 PDF 测试）", 2)
    eid = exam["id"]
    d = store.exam_dir(eid)
    for sub, src in (("paper", paper), ("answers", answers), ("sheet", sheet)):
        shutil.copy(src, d / sub / f"{sub}.pdf")
    store.update_exam(eid, lambda e: e["files"].update(paper=["paper/paper.pdf"], answers=["answers/answers.pdf"],
                                                       sheet=["sheet/sheet.pdf"]))
    llm = DeepSeek(st["api_key"], st["base_url"], cache_dir=str(d / "cache"), usage=Usage())
    t = time.time()
    exam = exam_setup.run_setup(store, eid, llm, st, lambda f, m: None)
    print(f"setup {time.time() - t:.1f}s, model calls: {fake.STATE['by_task']}")
    print("log:", exam.get("setup_log"))
    sol = [q for q in exam["questions"] if q["type"] == "解答题"]
    for q in sol:
        print(f"\n== 第{q['number']}题 {q['score']}分 {[(p['label'], p['score']) for p in q['parts']]} 标注方式={q.get('score_convention')} "
              f"截图={len(q.get('paper_crops', []))}/{len(q.get('answer_crops', []))} 答案来源={q['answer_source']}")
        print("   题干:", q["stem"][:150].replace("\n", " ⏎ "))
        print("   答案:", q["answer"][:100].replace("\n", " ⏎ "), "…", q["answer"][-60:].replace("\n", " ⏎ "))
        for c in q.get("checks") or []:
            print("   ⚠", c["where"], c["msg"][:120])
    others = [q for q in exam["questions"] if q["type"] != "解答题"]
    print("\n其他题:", [(q["number"], q["type"], q["score"]) for q in others])
    print("\n版面:", exam["template"]["labeled_by"])
    for pg in exam["template"]["pages"]:
        print("  第", pg["page"] + 1, "页", [(r["id"], r["kind"], r["questions"][:3], r["full_score"]) for r in pg["regions"]])
    print("\n提示:")
    for w in exam["warnings"]:
        print("  -", w)

    # ---- 断言
    by = {q["number"]: q for q in sol}
    assert sorted(by, key=int) == ["17", "18", "19", "20", "21"], list(by)
    assert [by[n]["score"] for n in ["17", "18", "19", "20", "21"]] == [14, 15, 15, 16, 18]
    expect = {"17": [6, 8], "18": [6, 9], "19": [7, 8], "20": [7, 9], "21": [4, 6, 8]}
    for n, ps in expect.items():
        assert [p["score"] for p in by[n]["parts"]] == ps, (n, by[n]["parts"])
        assert by[n]["score_convention"] == "cumulative", (n, by[n]["score_convention"])
        assert "（1）" in by[n]["stem"] or "(1)" in by[n]["stem"], f"第{n}题题干没有小问"
        assert by[n]["paper_crops"] and by[n]["answer_crops"], n
    assert len(by["18"]["answer_crops"]) == 2 and len(by["20"]["answer_crops"]) == 2, "跨页答案应有两张截图"
    assert "9" in by["18"]["answer"] and "坐标" in by["18"]["answer"], "第18题答案缺了跨页的最后一步"
    msgs18 = " ".join(c["msg"] for c in by["18"]["checks"])
    assert "找不到" in msgs18 and "出现了两次" in msgs18, by["18"]["checks"]
    for n in ["17", "19", "20", "21"]:
        assert not [c for c in by[n]["checks"] if c["where"] == "答案"], (n, by[n]["checks"])
    assert by["20"]["figure"], "第20题应有图形说明"
    assert exam["template"]["labeled_by"] == "pdf_text"
    labels = {n: r for pg in exam["template"]["pages"] for r in pg["regions"] for n in r["questions"] if r["kind"] == "解答题"}
    assert sorted(labels, key=int) == ["17", "18", "19", "20", "21"]
    assert "llm" not in fake.STATE["by_task"] and "layout" not in fake.STATE["by_task"], "答题纸不该再调用模型"
    # ---- 选择题、填空题
    objs = {q["number"]: q for q in exam["questions"] if q["type"] in ("填空题", "选择题")}
    assert [objs[str(n)]["score"] for n in range(1, 17)] == [4] * 6 + [5] * 6 + [4, 4, 5, 5], "分值应为 1–6 题 4 分、7–12 题 5 分、13–14 题 4 分、15–16 题 5 分"
    assert [objs[str(n)]["answer"] for n in range(13, 17)] == ["B", "C", "C", "D"]
    assert objs["11"]["answer"] == "[2,4]" and objs["1"]["answer"] == "\\sqrt{5}", (objs["11"]["answer"], objs["1"]["answer"])
    assert all(objs[str(n)]["stem"] for n in range(1, 13)), "填空题题干应录入（判分时参考单位/形式）"
    assert fake.STATE["by_task"].get("answer_obj") == 1, "选择题答案应直接从文字读出，只有填空题调用一次模型"
    from autograde.exam_setup import objective_places
    places = objective_places(exam["template"])
    assert all(places[str(n)].get("slot") for n in range(1, 13)), "1–12 题应各有一条横线"
    assert all(len(places[str(n)]["group"]["options"]) == 4 for n in range(13, 17)), "13–16 题应各有 4 个选项框"
    assert all("base" in o for n in range(13, 17) for o in places[str(n)]["group"]["options"])
    assert not [w for w in exam["warnings"] if "填空题" in w or "选择题" in w], exam["warnings"]
    print("\nALL CHECKS PASSED", tmp)
    srv.should_exit = True


if __name__ == "__main__":
    main(*sys.argv[1:4])
