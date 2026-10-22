"""生成答题纸的端到端测试（模型用本地模拟接口）：
快速录入题目 → 生成答题纸（A4，涂 4 位学号）→ 导入名单 → 模拟学生作答、扫描（歪斜、倒放、JPEG）→ 批改。

核对：
- 版面模板直接来自生成器：每道选择/填空题都有确切位置，解答题有方框；
- 涂的学号读对，对上名单补出姓名；没涂清楚的，按模型认出的手写姓名对上名单并提示确认；
- “每人一份”的答题纸（姓名印好、学号预先涂好、二维码带学号）；
- 混进别的作业的答题纸被挑出来；涂错成别人的学号 → 重号提示；名单里没交的人列出来；
- 选择题涂卡结果；A3 两栏、8 位学号的版本也能读。

python tests/test_sheetgen.py
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

from autograde import exam_setup, pipeline, report, roster, sheetgen
from autograde.exam_setup import objective_places, regions_for
from autograde.imaging import draw_texts
from autograde.llm import DeepSeek, Usage
from autograde.store import Store

PORT = 8774
DPI = 200
OUT = Path(tempfile.gettempdir()) / "autograde_sheetgen"

STRUCT = """1-6 单选 每题5分
7-8 多选 6分
9-12 填空 5分
13 解答 12分 2问
14 解答 14分 3问
15 解答 16分"""
ROSTER = "学号\t姓名\t性别\n2501\t张三\t男\n2502\t李四\t女\n2503\t王五\t男\n2504\t赵六\t女\n2505\t钱七\t男\n"
KEY = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "A", "6": "B", "7": "BD", "8": "AC",
       "9": "\\sqrt{5}", "10": "5-\\mathrm{i}", "11": "[2,4]", "12": "-\\frac{16}{15}"}


def start_fake():
    import uvicorn
    import fake_deepseek
    srv = uvicorn.Server(uvicorn.Config(fake_deepseek.app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    return srv, fake_deepseek


def render(pdf: bytes, dpi: int = DPI):
    import pymupdf
    out = []
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            out.append(cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR))
    return out


def px(v, dpi=DPI):
    return int(round(v * dpi / 25.4))


def pencil(img, rect, rng, dpi=DPI, cover=0.9):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    a, b = px(x0 + w * (1 - cover) / 2, dpi), px(y0 + h * (1 - cover) / 2, dpi)
    c, d = px(x1 - w * (1 - cover) / 2, dpi), px(y1 - h * (1 - cover) / 2, dpi)
    patch = img[b:d, a:c].astype(np.int16)
    tone = rng.integers(55, 85) + rng.normal(0, 8, patch.shape[:2])[:, :, None]
    img[b:d, a:c] = np.minimum(patch, tone).clip(0, 255).astype(np.uint8)


def fill_sheet(pages, plan, rng, sid="", choices=None, fills=True, name="", extra_digit=None, dpi=DPI):
    """在渲染好的空白答题纸上“作答”：涂学号、涂选择题、写填空、写解答题、写姓名。"""
    out = [p.copy() for p in pages]
    for p, pp in enumerate(plan["pages"]):
        img = out[p]
        info = pp.get("info")
        if info and info.get("grid") and sid:
            for col in info["grid"]["cols"]:
                d = sid[col["pos"]]
                pencil(img, col["options"][int(d)]["rect"], rng, dpi)
                w = col["write"]
                cv2.putText(img, d, (px(w[0] + 1.2, dpi), px(w[3] - 1.2, dpi)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
            if extra_digit is not None:                   # 最后一位多涂了一个
                col = info["grid"]["cols"][-1]
                pencil(img, col["options"][extra_digit]["rect"], rng, dpi)
        if info and name:
            x0, y, _ = info["name"]
            out[p] = img = draw_texts(img, [{"xy": (px(x0 + 12, dpi), px(y - 6.5, dpi)), "text": name, "size": px(5.5, dpi),
                                             "color": (35, 35, 35)}])
        for sec in pp["sections"]:
            if sec["kind"] == "选择题" and choices:
                for g in sec["groups"]:
                    for letter in choices.get(g["q"], ""):
                        o = next(o for o in g["options"] if o["label"] == letter)
                        pencil(img, o["rect"], rng, dpi)
            if sec["kind"] == "填空题" and fills:
                for sl in sec["slots"]:
                    x0, y, x1 = sl["line"]
                    cv2.putText(img, f"ans {sl['q']}", (px(x0 + 6, dpi), px(y - 1.5, dpi)), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                1.1, (35, 30, 30), 2)
        for b in pp["boxes"]:
            x0, y0, x1, y1 = b["rect"]
            for k in range(3):
                yy = y0 + 14 + 9 * k
                if yy < y1 - 4:
                    cv2.putText(img, f"step {k + 1}: x = {rng.integers(2, 99)}", (px(x0 + 8, dpi), px(yy, dpi)),
                                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, (40, 30, 25), 2)
    return out


def scan(img, rng, angle=None, upside=False, scale=None):
    h, w = img.shape[:2]
    ang = float(rng.uniform(-1.0, 1.0)) if angle is None else angle
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, float(rng.uniform(0.97, 1.0)) if scale is None else scale)
    M[:, 2] += rng.uniform(-15, 15, size=2)
    out = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    out = np.clip(out.astype(np.float32) * 0.96 + rng.normal(0, 2.5, out.shape[:2])[:, :, None], 0, 255).astype(np.uint8)
    if upside:
        out = cv2.rotate(out, cv2.ROTATE_180)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return cv2.imdecode(buf, 1)


def to_pdf(path: Path, imgs, dpi=DPI):
    import pymupdf
    doc = pymupdf.open()
    for im in imgs:
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        h, w = im.shape[:2]
        page = doc.new_page(width=w * 72 / dpi, height=h * 72 / dpi)
        page.insert_image(page.rect, stream=buf.tobytes())
    doc.save(str(path))


def main():
    srv, fake = start_fake()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ag_gen_"))
    store = Store(str(tmp / "data"))
    st = store.save_settings({"api_key": "sk-fake", "base_url": f"http://127.0.0.1:{PORT}", "max_workers": 6,
                              "scan_dpi": 220, "samples": 1})
    rng = np.random.default_rng(11)

    # ---- 题目：快速录入 + 填参考答案
    qs, bad = exam_setup.parse_structure(STRUCT)
    assert not bad and len(qs) == 15, (bad, len(qs))
    assert [q["type"] for q in qs[:9]] == ["选择题"] * 8 + ["填空题"] and qs[6]["multi"] and not qs[0]["multi"]
    assert qs[12]["score"] == 12 and len(qs[13]["parts"]) == 3
    for q in qs:
        q["answer"] = KEY.get(q["number"], "（略）" if q["type"] == "解答题" else "")
    exam = store.create_exam("第3周作业", 1)
    eid = exam["id"]
    store.update_exam(eid, lambda e: e.__setitem__("questions", qs))

    # ---- 名单
    rows, bad = roster.parse_text(ROSTER)
    assert len(rows) == 5 and not bad and rows[0] == {"sid": "2501", "name": "张三", "class": ""}, (rows, bad)
    cls = roster.upsert(store.root, "高一5班", [{"sid": r["sid"], "name": r["name"]} for r in rows])

    # ---- 生成答题纸（A4、4 位学号）
    t = time.time()
    exam = exam_setup.generate_sheet(store, eid, "A4", 4, "normal")
    print(f"generate {time.time() - t:.2f}s")
    tmpl = exam["template"]
    plan = tmpl["generated"]["plan"]
    n_pages = len(plan["pages"])
    assert exam["pages_per_sheet"] == n_pages == len(tmpl["pages"]), (exam["pages_per_sheet"], n_pages)
    places = objective_places(tmpl)
    assert all(places[str(i)].get("group") for i in range(1, 9)), places
    assert all(places[str(i)].get("slot") for i in range(9, 13))
    assert all(len(places[str(i)]["group"]["options"]) == 4 for i in range(1, 9))
    assert all(o.get("base") is not None for i in range(1, 9) for o in places[str(i)]["group"]["options"])
    assert all(regions_for(tmpl, str(n)) for n in (13, 14, 15))
    assert all(len(pg["markers"]) == 7 for pg in tmpl["pages"]), [len(pg["markers"]) for pg in tmpl["pages"]]
    grid = next(r for r in tmpl["pages"][0]["regions"] if r.get("id_grid"))["id_grid"]
    assert grid["digits"] == 4 and len(grid["cols"][0]["options"]) == 10
    assert not [w for w in exam["warnings"] if "答题纸" in w and "重新生成" in w], exam["warnings"]
    print("pages", n_pages, "markers", [len(pg["markers"]) for pg in tmpl["pages"]],
          "regions", [[(r["kind"], r["questions"][:3]) for r in pg["regions"]] for pg in tmpl["pages"]])
    blank = render(sheetgen.render_pdf(plan, eid))
    personal_pdf = sheetgen.render_pdf(plan, eid, [{"sid": "2503", "name": "王五", "class": "高一5班"}])
    personal = render(personal_pdf)
    cv2.imwrite(str(OUT / "personal_p1.jpg"), cv2.resize(personal[0], None, fx=0.5, fy=0.5))

    # ---- 另一次作业的答题纸（应该被挑出来）
    other = store.create_exam("第2周作业", 1)
    oq, _ = exam_setup.parse_structure("1-4 单选 5分\n5 解答 10分")
    store.update_exam(other["id"], lambda e: e.__setitem__("questions", oq))
    other = exam_setup.generate_sheet(store, other["id"], "A4", 4)
    foreign = render(sheetgen.render_pdf(other["template"]["generated"]["plan"], other["id"]))

    # ---- 学生作答 + 扫描
    ch = {
        "张三": {"1": "A", "2": "B", "3": "C", "4": "D", "5": "A", "6": "B", "7": "BD", "8": "AC"},
        "李四": {"1": "A", "2": "C", "3": "C", "4": "D", "5": "B", "6": "B", "7": "B", "8": "AC"},
        "王五": {"1": "B", "2": "B", "3": "C", "4": "", "5": "A", "6": "B", "7": "BD", "8": "C"},
        "钱七": {"1": "A", "2": "B", "3": "D", "4": "D", "5": "A", "6": "A", "7": "BCD", "8": "AC"},
    }
    sheets = [
        fill_sheet(blank, plan, rng, "2501", ch["张三"], name="张三"),
        fill_sheet(blank, plan, rng, "2502", ch["李四"], name="李四", extra_digit=5),     # 最后一位涂了 2 和 5
        fill_sheet(personal, plan, rng, "", ch["王五"]),                                   # 每人一份：学号已经印好涂好
        foreign,                                                                          # 别的作业
        fill_sheet(blank, plan, rng, "2501", ch["钱七"], name="钱七"),                    # 涂成了张三的学号
    ]
    scans = []
    for k, pages in enumerate(sheets):
        for p, img in enumerate(pages):
            scans.append(scan(img, rng, upside=(k == 2 and p == len(pages) - 1)))
    d = store.exam_dir(eid)
    to_pdf(d / "scans" / "batch.pdf", scans)
    cv2.imwrite(str(OUT / "scan_p1.jpg"), cv2.resize(scans[0], None, fx=0.5, fy=0.5))

    # ---- 批改
    fake.STATE["student_reply"] = {"name": "李四", "student_id": "", "class": ""}   # 模型认出的手写姓名
    llm = DeepSeek(st["api_key"], st["base_url"], cache_dir=str(d / "cache"), usage=Usage())
    t = time.time()
    stats = pipeline.grade_scans(store, eid, llm, st, ["scans/batch.pdf"], lambda f, m: None)
    print(f"grading {time.time() - t:.1f}s", stats, "calls", fake.STATE["by_task"])
    exam = store.load_exam(eid)
    rep = exam["scan_report"]
    print("scan report:", rep["notes"])
    res = report.results(store, eid)
    rows = {r["id"]: r for r in res["rows"]}
    for r in res["rows"]:
        s = r["student"]
        print(" ", r["id"], s.get("name"), s.get("student_id"), s.get("source"), r["id_issues"], "omr", (s.get("omr") or {}).get("id"),
              "qr", s.get("qr_sid", ""))
    print("absent:", res["absent"], "roster:", res["roster"])

    assert len(rows) == 4, list(rows)
    assert any("《第2周作业》的答题纸，不是这次的" in n for n in rep["notes"]), rep["notes"]
    s1, s2, s3, s4 = (rows[f"S00{i}"]["student"] for i in range(1, 5))
    assert (s1["name"], s1["student_id"], s1["source"], s1["class"]) == ("张三", "2501", "roster", "高一5班"), s1
    assert s2["omr"]["id"] == "250?" and not s2["omr"]["complete"], s2["omr"]
    assert (s2["name"], s2["student_id"], s2["source"]) == ("李四", "2502", "roster_name"), s2
    assert any("按手写姓名" in x for x in rows["S002"]["id_issues"]), rows["S002"]["id_issues"]
    assert (s3["name"], s3["student_id"], s3["qr_sid"]) == ("王五", "2503", "2503"), s3
    assert s3["omr"]["id"] == "2503" and s3["omr"]["complete"]
    assert s4["student_id"] == "2501" and any("重复" in x for x in rows["S004"]["id_issues"]), rows["S004"]
    assert any("重复" in x for x in rows["S001"]["id_issues"])
    assert sorted(a["name"] for a in res["absent"]) == ["赵六", "钱七"], res["absent"]
    assert fake.STATE["by_task"].get("student") == 1, "只有学号没涂清的那份才需要模型认姓名"
    want = {"S001": ch["张三"], "S002": ch["李四"], "S003": ch["王五"], "S004": ch["钱七"]}
    for sid, picks in want.items():
        sh = store.load_sheet(eid, sid)
        got = {n: sh["questions"][n].get("student") for n in picks}
        assert got == picks, (sid, got, picks)
        assert all(sh["questions"][str(n)]["crops"] for n in range(9, 16)), sid
        assert sh["missing_pages"] == [] and sh["align_ok"], sid
    sc = store.load_sheet(eid, "S001")["questions"]
    assert sc["7"]["ai"]["score"] == 6 and sc["1"]["ai"]["score"] == 5
    sc = store.load_sheet(eid, "S002")["questions"]
    assert sc["7"]["ai"]["score"] == 3 and sc["7"]["ai"]["reason"] == "没选全", sc["7"]["ai"]
    csv = (d / "scores.csv").read_text(encoding="utf-8-sig").splitlines()
    assert any("没有答卷" in l and "赵六" in l for l in csv), csv[-3:]

    # ---- 老师改了 S004 的学号（其实是钱七）→ 重号和缺交都消失
    pipeline_set = store.update_sheet(eid, "S004", lambda s: s["student"].update(name="钱七", student_id="2505", source="teacher", issues=[]))
    res = report.results(store, eid)
    rows = {r["id"]: r for r in res["rows"]}
    assert not rows["S001"]["id_issues"] and [a["name"] for a in res["absent"]] == ["赵六"], (rows["S001"]["id_issues"], res["absent"])
    assert pipeline_set["student"]["source"] == "teacher"

    # ---- 名单改名：重新对照后姓名跟着变（老师改过的不动）
    roster.upsert(store.root, "高一5班", [{"sid": "2501", "name": "张三丰"}, {"sid": "2502", "name": "李四"},
                                          {"sid": "2503", "name": "王五"}, {"sid": "2504", "name": "赵六"}], cid=cls["id"])
    pipeline.resolve_students(store, eid)
    assert store.load_sheet(eid, "S001")["student"]["name"] == "张三丰"
    assert store.load_sheet(eid, "S004")["student"]["name"] == "钱七"

    # ---- 题目改了 → 提醒重新生成
    store.update_exam(eid, lambda e: (e["questions"].append(exam_setup.new_question("16", "解答题")),
                                      e.__setitem__("warnings", exam_setup.reconcile(e))))
    assert any("重新生成" in w for w in store.load_exam(eid)["warnings"])

    # ---- A3 两栏、8 位学号：读学号
    e3 = store.create_exam("期中考试", 1)
    q3, _ = exam_setup.parse_structure("1-12 填空 5分\n13-16 单选 5分\n17 解答 14分\n18 解答 15分\n19 解答 15分\n20 解答 16分\n21 解答 18分")
    store.update_exam(e3["id"], lambda e: e.__setitem__("questions", q3))
    e3 = exam_setup.generate_sheet(store, e3["id"], "A3", 8, "normal")
    plan3 = e3["template"]["generated"]["plan"]
    assert plan3["paper"] == "A3" and len(plan3["pages"]) <= 3, len(plan3["pages"])
    assert all(len(pg["markers"]) == 9 for pg in e3["template"]["pages"])
    pages3 = fill_sheet(render(sheetgen.render_pdf(plan3, e3["id"]), 200), plan3, rng, "25090315",
                        {str(i): "ABCD"[i % 4] for i in range(13, 17)})
    d3 = store.exam_dir(e3["id"])
    to_pdf(d3 / "scans" / "a3.pdf", [scan(p, rng, angle=0.6) for p in pages3])
    store.update_exam(e3["id"], lambda e: [q.__setitem__("answer", "A" if q["type"] == "选择题" else ("1" if q["type"] == "填空题" else "略"))
                                          for q in e["questions"]])
    sids = pipeline.prepare_sheets(store, e3["id"], ["scans/a3.pdf"], st)
    s = store.load_sheet(e3["id"], sids[0])
    assert s["student"]["omr"]["id"] == "25090315", s["student"]["omr"]
    assert {n: s["questions"][n]["student"] for n in ("13", "14", "15", "16")} == {str(i): "ABCD"[i % 4] for i in range(13, 17)}
    print("A3 OK:", len(plan3["pages"]), "pages; id", s["student"]["omr"]["id"])
    shutil.copy(d / "template" / "page_0_preview.jpg", OUT / "template_preview.jpg")
    print("\nSHEETGEN OK", tmp, "images in", OUT)
    srv.should_exit = True


if __name__ == "__main__":
    main()
