"""平时作业（不打分）和“AI 做选择、填空题答案”的端到端测试（模型用本地模拟接口，走网页用的 HTTP 接口）：

建作业（只写题目结构）→ 让 AI 做选择、填空题（每题两遍，不一致的标出来）→ 老师确认/改答案 → AI 重做 →
生成答题纸（不印分值）→ 模拟学生作答、扫描、上传 → 解答题判 对/半对/错、写错在哪 → 成绩表写对错 →
老师改判 → 改成周测（打分）自动重新批改 → 再改回作业（命中缓存，不重复花钱）。
另外核对 A3 答题纸按栏编页码（正面 1、2，反面 3、4）。

python tests/test_homework.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_sheetgen as T   # 复用：渲染、模拟作答、模拟扫描

FAKE, APP = 8778, 8020
OUT = Path(tempfile.gettempdir()) / "autograde_homework"
STRUCT = "1-4 单选\n5-6 填空\n7 解答 2问\n8 解答"
SOLVE = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "\\sqrt{5}", "6": "\\frac{\\pi}{2}"}
ROSTER = "学号\t姓名\n2501\t张三\n2502\t李四\n2503\t王五\n"


def start(app, port):
    import uvicorn
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    return srv


def wait_job(base, job, timeout=300):
    t = time.time()
    while time.time() - t < timeout:
        j = requests.get(f"{base}/api/jobs/{job['id']}").json()
        if j["status"] != "running":
            assert j["status"] == "done", j
            return j
        time.sleep(0.2)
    raise TimeoutError(job)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ag_hw_"))
    data = tmp / "data"
    data.mkdir()
    (data / "settings.json").write_text(json.dumps({"api_key": "sk-fake", "base_url": f"http://127.0.0.1:{FAKE}",
                                                    "samples": 1, "max_workers": 6, "scan_dpi": 220}), encoding="utf-8")
    os.environ["AUTOGRADE_DATA"] = str(data)
    import fake_deepseek as fake
    from autograde import server
    s1 = start(fake.app, FAKE)
    s2 = start(server.app, APP)
    base = f"http://127.0.0.1:{APP}"
    api = lambda m, u, **kw: _ok(requests.request(m, base + u, **kw))   # noqa: E731

    # ---- 建作业：只写题目结构，不写分值
    r = api("POST", "/api/exams", data={"title": "5.2 导数的运算 作业", "kind": "homework", "structure": STRUCT,
                                         "generate_answers": "true"})
    ex = r["exam"]
    eid = ex["id"]
    assert r["job"] is None and ex["kind"] == "homework", r
    assert [(q["number"], q["type"], q["score"]) for q in ex["questions"]] == \
        [(str(i), "选择题", None) for i in range(1, 5)] + [("5", "填空题", None), ("6", "填空题", None),
                                                          ("7", "解答题", None), ("8", "解答题", None)], ex["questions"]
    assert not [w for w in ex["warnings"] if "满分" in w or "分值" in w], ex["warnings"]
    assert api("GET", "/api/exams")[0]["kind"] == "homework"

    # ---- 没有题目内容：AI 没法做，什么也不改
    j = wait_job(base, api("POST", f"/api/exams/{eid}/objective/solve", json={})["job"])
    ex = api("GET", f"/api/exams/{eid}")
    assert not any(q["answer"] for q in ex["questions"]) and not fake.STATE["by_task"].get("obj_solve"), j

    # ---- 录入题干后让 AI 做：第 3 题两遍做得不一样
    for q in ex["questions"]:
        if q["type"] != "解答题":
            q["stem"] = f"第{q['number']}题 已知 f(x)=x^2，……" + (" A. 1 B. 2 C. 3 D. 4" if q["type"] == "选择题" else "")
    ex = api("PUT", f"/api/exams/{eid}/questions", json={"questions": ex["questions"]})
    fake.STATE["solve_key"] = SOLVE
    fake.STATE["solve_disagree"] = {"3"}
    wait_job(base, api("POST", f"/api/exams/{eid}/objective/solve", json={})["job"])
    ex = api("GET", f"/api/exams/{eid}")
    qs = {q["number"]: q for q in ex["questions"]}
    assert {n: qs[n]["answer"] for n in SOLVE} == SOLVE, {n: qs[n]["answer"] for n in SOLVE}
    assert all(qs[n]["answer_source"] == "ai" and not qs[n]["confirmed"] for n in SOLVE)
    assert qs["3"]["ai_solve"]["answers"] == ["C", "D"] and not qs["3"]["ai_solve"]["agree"], qs["3"]["ai_solve"]
    assert all(qs[n]["ai_solve"]["agree"] for n in SOLVE if n != "3")
    assert [c["msg"] for c in qs["3"]["checks"]] == ["AI 两次独立解答不一样（C / D）"
                                                    "，请老师确认"], qs["3"]["checks"]
    assert not qs["7"]["answer"], "解答题没有参考答案也可以（作业批改时 AI 自己做）"
    assert fake.STATE["by_task"]["obj_solve"] == 12, fake.STATE["by_task"]
    assert any("AI 做的" in w for w in ex["warnings"]), ex["warnings"]
    print("AI solve OK:", {n: qs[n]["answer"] for n in SOLVE}, ex["warnings"])

    # ---- 老师：确认第 3 题（用 C），第 4 题改成多选 BD；AI 重做时这两题不动，其余没确认的重做（不用缓存）
    qs["3"]["confirmed"] = True
    qs["4"]["answer"] = "BD"
    ex = api("PUT", f"/api/exams/{eid}/questions", json={"questions": list(qs.values())})
    qs = {q["number"]: q for q in ex["questions"]}
    assert qs["3"]["confirmed"] and not qs["3"]["checks"] and qs["3"]["answer_source"] == "ai"
    assert qs["4"]["answer_source"] == "teacher" and "ai_solve" not in qs["4"], qs["4"]
    fake.STATE["solve_key"] = dict(SOLVE, **{"1": "D"})        # 重做时模型给了不同的答案
    wait_job(base, api("POST", f"/api/exams/{eid}/objective/solve", json={"redo": True})["job"])
    ex = api("GET", f"/api/exams/{eid}")
    qs = {q["number"]: q for q in ex["questions"]}
    assert fake.STATE["by_task"]["obj_solve"] == 12 + 8, fake.STATE["by_task"]     # 1、2、5、6 题重做（各两遍）；3、4 不动
    assert (qs["1"]["answer"], qs["3"]["answer"], qs["4"]["answer"]) == ("D", "C", "BD")
    qs["1"]["answer"] = "A"                                      # 老师改回来
    for n in ("2", "5", "6"):
        qs[n]["confirmed"] = True
    ex = api("PUT", f"/api/exams/{eid}/questions", json={"questions": list(qs.values())})
    assert not [w for w in ex["warnings"] if "AI 做的" in w], ex["warnings"]

    # ---- 生成答题纸：不印分值
    ex = api("POST", f"/api/exams/{eid}/answer-sheet", json={"paper": "A4", "id_digits": 4})
    plan = ex["template"]["generated"]["plan"]
    assert plan["scored"] is False
    pdf = requests.get(f"{base}/api/exams/{eid}/answer-sheet.pdf").content
    import pymupdf
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        text = "".join(p.get_text() for p in doc)
    assert "分" not in text and "共 4 题" in text and "共 2 题" in text, text[:800]
    assert ex["graded_numbers"] == [str(i) for i in range(1, 9)], ex["graded_numbers"]

    # ---- 学生作答、扫描、上传
    api("POST", "/api/rosters/import", data={"name": "高二3班", "text": ROSTER, "mode": "replace"})
    rng = np.random.default_rng(3)
    blank = T.render(pdf)
    picks = {"2501": {"1": "A", "2": "B", "3": "C", "4": "BD"},          # 张三：选择题全对
             "2502": {"1": "B", "2": "B", "4": "B"},                     # 李四：第 1 题错，第 3 题没涂，第 4 题（多选）没选全
             "2503": {"1": "A", "2": "A", "3": "C", "4": "C"}}           # 王五：第 2、4 题错
    names = {"2501": "张三", "2502": "李四", "2503": "王五"}
    scans = [T.scan(img, rng) for sid, ch in picks.items() for img in T.fill_sheet(blank, plan, rng, sid, ch, name=names[sid])]
    T.to_pdf(tmp / "hw.pdf", scans)
    fake.STATE["hw_verdict"] = {"7": "半对", "8": "错"}
    by0 = dict(fake.STATE["by_task"])
    with open(tmp / "hw.pdf", "rb") as fh:
        job = api("POST", f"/api/exams/{eid}/scans", files={"files": ("hw.pdf", fh, "application/pdf")})["job"]
    wait_job(base, job)
    calls = {k: v - by0.get(k, 0) for k, v in fake.STATE["by_task"].items() if v != by0.get(k, 0)}
    print("grading calls:", calls)
    assert calls.get("homework") == 6 and not calls.get("grade"), calls

    res = api("GET", f"/api/exams/{eid}/results")
    assert res["scored"] is False and res["kind"] == "homework"
    rows = {r["student"]["name"]: r for r in res["rows"]}
    assert sorted(rows) == ["张三", "李四", "王五"], list(rows)
    exp = {"张三": {"1": 1, "2": 1, "3": 1, "4": 1}, "李四": {"1": 0, "2": 1, "3": 0, "4": 0.5}, "王五": {"1": 1, "2": 0, "3": 1, "4": 0}}
    for nm, want in exp.items():
        r = rows[nm]
        got = {n: r["scores"][n] for n in want}
        assert got == want, (nm, got)
        # 填空：模拟的识别结果第 5 题是 4/5（错），第 6 题和答案一样（对）；解答题按模拟的判断：7 半对、8 错
        assert (r["scores"]["5"], r["scores"]["6"], r["scores"]["7"], r["scores"]["8"]) == (0, 1, 0.5, 0), (nm, r["scores"])
        assert r["max_total"] == 8, r["max_total"]
    assert rows["张三"]["total"] == 5.5 and rows["王五"]["total"] == 3.5, (rows["张三"]["total"], rows["王五"]["total"])
    assert res["stats"]["7"]["half"] == 3 and res["stats"]["8"]["wrong"] == 3 and res["stats"]["1"]["right"] == 2, res["stats"]
    zs = rows["张三"]["id"]
    sh = api("GET", f"/api/exams/{eid}/sheets/{zs}")
    a7 = sh["questions"]["7"]["ai"]
    assert (a7["verdict"], a7["mode"], a7["max"], sh["questions"]["7"]["max"]) == ("半对", "homework", 1.0, 1), a7
    assert [(p["label"], p["verdict"]) for p in a7["parts"]] == [("(1)", "对"), ("(2)", "错")], a7["parts"]
    assert "计算错误" in a7["error"] and a7["comment"], a7
    assert sh["annotated"], "批改后的答题纸"
    shutil.copy(data / "exams" / eid / sh["annotated"][0], OUT / "annotated_hw.jpg")

    # ---- 成绩表：每题写 对/半对/错，最后是做对几题
    csv = requests.get(f"{base}/api/exams/{eid}/export.csv").content.decode("utf-8-sig").splitlines()
    head = csv[0].split(",")
    assert head[5:13] == [f"第{i}题" for i in range(1, 9)] and "做对题数（共8题）" in head, head
    line = next(l for l in csv if "张三" in l).split(",")
    assert line[5:13] == ["对", "对", "对", "对", "错", "对", "半对", "错"], line
    assert line[head.index("做对题数（共8题）")] == "5.5", line
    print(csv[0]); print(",".join(line))

    # ---- 老师改判：第 8 题判对；分数超出 1 的拒绝
    s = api("PUT", f"/api/exams/{eid}/sheets/{zs}/q/8/teacher", json={"score": 1})
    assert s["questions"]["8"]["teacher"]["score"] == 1
    r = requests.put(f"{base}/api/exams/{eid}/sheets/{zs}/q/8/teacher", json={"score": 12})
    assert r.status_code == 400, r.text
    res = api("GET", f"/api/exams/{eid}/results")
    assert next(r for r in res["rows"] if r["id"] == zs)["total"] == 6.5
    csv = requests.get(f"{base}/api/exams/{eid}/export.csv").content.decode("utf-8-sig")
    assert ",对,半对,对,4,1,1.5,6.5," in csv, csv

    # ---- 改成周测（打分）：填上分值，改类型 → 自动按分数重新批（老师的改判清掉）
    ex = api("GET", f"/api/exams/{eid}")
    score = {"1": 5, "2": 5, "3": 5, "4": 5, "5": 5, "6": 5, "7": 12, "8": 10}
    for q in ex["questions"]:
        q["score"] = score[q["number"]]
        if q["number"] == "7":
            q["parts"] = [{"label": "(1)", "score": 5, "text": ""}, {"label": "(2)", "score": 7, "text": ""}]
    api("PUT", f"/api/exams/{eid}/questions", json={"questions": ex["questions"]})
    by0 = dict(fake.STATE["by_task"])
    ex = api("PUT", f"/api/exams/{eid}/kind", json={"kind": "quiz"})
    assert ex["kind"] == "quiz" and ex["job"] and ex["job"]["status"] == "running", ex.get("job")
    wait_job(base, ex["job"])
    calls = {k: v - by0.get(k, 0) for k, v in fake.STATE["by_task"].items() if v != by0.get(k, 0)}
    assert calls.get("grade") == 6 and not calls.get("homework"), calls
    res = api("GET", f"/api/exams/{eid}/results")
    assert res["scored"] is True
    r = next(r for r in res["rows"] if r["id"] == zs)
    assert (r["scores"]["1"], r["scores"]["5"], r["scores"]["6"], r["scores"]["7"]) == (5, 0, 5, 11), r["scores"]
    assert r["max_total"] == 52 and r["total"] == 5 * 4 + 0 + 5 + 11 + r["scores"]["8"], r
    sh = api("GET", f"/api/exams/{eid}/sheets/{zs}")
    assert not sh["questions"]["8"].get("teacher"), "改类型后老师的改判清掉"
    ex = api("GET", f"/api/exams/{eid}")
    assert any("没有印分值" in w for w in ex["warnings"]), ex["warnings"]
    csv = requests.get(f"{base}/api/exams/{eid}/export.csv").content.decode("utf-8-sig").splitlines()
    assert csv[0].split(",")[-3] == "总分", csv[0]

    # ---- 再改回作业：判对错的结果命中缓存，不再调用模型
    by0 = dict(fake.STATE["by_task"])
    ex = api("PUT", f"/api/exams/{eid}/kind", json={"kind": "homework"})
    wait_job(base, ex["job"])
    calls = {k: v - by0.get(k, 0) for k, v in fake.STATE["by_task"].items() if v != by0.get(k, 0)}
    assert not calls.get("homework") and not calls.get("grade"), calls
    res = api("GET", f"/api/exams/{eid}/results")
    r = next(r for r in res["rows"] if r["id"] == zs)
    assert res["scored"] is False and r["scores"]["7"] == 0.5 and r["total"] == 5.5, r["scores"]
    assert not any("没有印分值" in w or "印了分值" in w for w in api("GET", f"/api/exams/{eid}")["warnings"])
    # 考试 ↔ 周测：都打分，不用重新批
    ex = api("PUT", f"/api/exams/{eid}/kind", json={"kind": "exam"})
    wait_job(base, ex["job"])
    ex2 = api("PUT", f"/api/exams/{eid}/kind", json={"kind": "quiz"})
    assert ex2["kind"] == "quiz" and ex2["job"]["id"] == ex["job"]["id"] and ex2["job"]["status"] == "done", "打分 → 打分不需要重新批"
    r = requests.put(f"{base}/api/exams/{eid}/kind", json={"kind": "test"})
    assert r.status_code == 400

    # ---- A3：正面左右两栏是第 1、2 页，反面是第 3、4 页
    e3 = api("POST", "/api/exams", data={"title": "期中", "kind": "exam",
                                          "structure": "1-8 单选 5分\n9-12 填空 5分\n13 解答 13分\n14-17 解答 15分\n18 解答 17分"})["exam"]
    e3 = api("POST", f"/api/exams/{e3['id']}/answer-sheet", json={"paper": "A3", "id_digits": 8})
    n3 = len(e3["template"]["generated"]["plan"]["pages"])
    with pymupdf.open(stream=requests.get(f"{base}/api/exams/{e3['id']}/answer-sheet.pdf").content, filetype="pdf") as doc:
        assert doc.page_count == n3 >= 2, (doc.page_count, n3)
        for p, page in enumerate(doc):
            W, H = page.rect.width, page.rect.height
            for c in (0, 1):
                label = f"第 {2 * p + c + 1} 页（共 {2 * n3} 页）"
                hits = page.search_for(label)
                assert len(hits) == 1 and hits[0].y0 > H - 40, (p, label, hits)
                assert (hits[0].x1 < W / 2) if c == 0 else (hits[0].x0 > W / 2), (p, c, hits[0], W)
            assert not page.search_for(f"第 {2 * p + 3} 页（共"), p
    print("A3 page numbers OK:", n3, "sides,", 2 * n3, "pages")

    print("\nHOMEWORK OK", tmp, "annotated:", OUT / "annotated_hw.jpg")
    s1.should_exit = s2.should_exit = True


def _ok(r):
    assert r.status_code == 200, (r.status_code, r.text[:500])
    return r.json()


if __name__ == "__main__":
    main()
