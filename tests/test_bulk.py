"""大批量扫描件压测：用真实答题纸模拟 75 位学生的答卷（正反两面），扫描仪式的 300dpi 彩色 JPEG，
合成一个 149 页的 PDF（第 38 份缺反面、第 71 份缺正面、中间夹了一张空白页），
通过 HTTP 上传给真实服务（模型用本地模拟接口），记录服务进程的内存峰值和各阶段用时，并核对分组和涂卡结果。

python tests/test_bulk.py [试卷.pdf 答案.pdf 答题纸.pdf]
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from autograde import exam_setup
from autograde.exam_setup import objective_places
from autograde.imaging import load_pages
from autograde.layout import analyze_page
from autograde.llm import DeepSeek, Usage
from autograde.store import Store

INPUTS = Path("/home/claude/work/inputs")
DEFAULT = [INPUTS / "53750920-1791416779685_____1/in.pdf", INPUTS / "56910cad-1791416779684____________2/in.pdf",
           INPUTS / "b003f8c9-1791416779684______________/in.pdf"]
CACHE = Path(os.environ.get("AG_BULK_CACHE") or Path(tempfile.gettempdir()) / "autograde_bulk")   # 生成的模拟扫描件缓存在这里
FAKE_PORT, APP_PORT = 8781, 8019
N_STUDENTS = 75
SCAN_DPI = 300
DROP = {(37, 1), (70, 0)}        # (学生序号, 页)：第 38 份缺反面，第 71 份缺正面
BLANK_AFTER = 12                 # 第 13 份答卷后面夹了一张空白页
UPSIDE = {5, 50}                 # 这两份整份倒着放进扫描仪
LETTERS = "ABCD"


def expected_choice(k: int, j: int) -> str:
    return LETTERS[(k + j) % 4]


def wait_http(url, t=30):
    for _ in range(int(t * 10)):
        try:
            requests.get(url, timeout=1)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("not up: " + url)


def make_scan_pdf(sheet_pdf: Path, tmpl: dict, out: Path, only=None) -> dict:
    """生成模拟扫描件。返回 {pdf 页序号: (学生序号, 页)}。only=[(学生序号, 页), ...] 时只生成这几页（模拟补扫）。"""
    import pymupdf
    rng = np.random.default_rng(7)
    base = load_pages(sheet_pdf, dpi=SCAN_DPI)
    places = objective_places(tmpl)
    choice_nums = sorted((n for n, p in places.items() if p.get("group")), key=int)
    to_img = []
    for img in base:
        lay = analyze_page(img)
        to_img.append(np.linalg.inv(np.vstack([lay.to_tmpl, [0, 0, 1]])))
    noise = [rng.normal(0, 3.0, size=img.shape[:2]).astype(np.float32) for img in base]
    doc = pymupdf.open()
    index = {}

    def add(im):
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        h, w = im.shape[:2]
        page = doc.new_page(width=w * 72 / SCAN_DPI, height=h * 72 / SCAN_DPI)
        page.insert_image(page.rect, stream=buf.tobytes())

    for k in range(N_STUDENTS):
        for p, img in enumerate(base):
            if (only is None and (k, p) in DROP) or (only is not None and (k, p) not in only):
                continue
            Hi = to_img[p]

            def T(x, y):
                v = Hi @ [x, y, 1]
                return int(v[0] / v[2]), int(v[1] / v[2])
            stu = img.copy()
            for n, pl in places.items():
                if pl["page"] != p:
                    continue
                if pl.get("slot") and (k + int(n)) % 7:                       # 少数空着不写
                    x0, y0, x1, y1 = pl["slot"]["rect"]
                    cv2.putText(stu, f"ans{n}", T(x0 + 40, y1 - 14), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3.0, (35, 30, 30), 6)
                if pl.get("group"):
                    letter = expected_choice(k, choice_nums.index(n))
                    o = next(o for o in pl["group"]["options"] if o["label"] == letter)
                    a, b = T(o["rect"][0] + 1, o["rect"][1] + 1)
                    c, d = T(o["rect"][2] - 1, o["rect"][3] - 1)
                    stu[b:d, a:c] = np.minimum(stu[b:d, a:c], int(rng.integers(45, 90)))
            for pg in tmpl["pages"]:
                if pg["page"] != p:
                    continue
                for r in pg["regions"]:
                    if r["kind"] == "解答题":
                        x0, y0, x1, y1 = r["rect"]
                        for line in range(int(rng.integers(0, 6))):
                            cv2.putText(stu, f"S{k + 1} step {line + 1}: x = {rng.integers(2, 99)}",
                                        T(x0 + 60, y0 + 110 + 70 * line), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.6, (40, 30, 25), 5)
            h, w = stu.shape[:2]
            ang = 2.5 if k == 20 else float(rng.uniform(-1.2, 1.2))
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, float(rng.uniform(0.97, 1.0)))
            M[:, 2] += rng.uniform(-25, 25, size=2)
            scan = cv2.warpAffine(stu, M, (w, h), borderValue=(255, 255, 255))
            scan = np.clip(scan.astype(np.float32) * 0.955 + np.roll(noise[p], int(rng.integers(0, 500)), axis=1)[:, :, None],
                           0, 255).astype(np.uint8)
            if k in UPSIDE:
                scan = cv2.rotate(scan, cv2.ROTATE_180)
            index[doc.page_count] = (k, p)
            add(scan)
        if k == BLANK_AFTER and only is None:
            blank =np.clip(np.full(base[0].shape, 243, np.float32) + noise[0][:, :, None], 0, 255).astype(np.uint8)
            index[doc.page_count] = (None, None)
            add(blank)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out))
    return index


def rss(pid):
    vals = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("VmRSS", "VmHWM")):
                k, v = line.split(":")
                vals[k] = int(v.split()[0]) / 1024
    except Exception:
        pass
    return vals


def main(paper: Path, answers: Path, sheet: Path):
    tmp = Path(tempfile.mkdtemp(prefix="ag_bulk_"))
    data = tmp / "data"
    data.mkdir()
    settings = {"api_key": "sk-fake-bulk", "base_url": f"http://127.0.0.1:{FAKE_PORT}", "max_workers": 8, "samples": 2}
    (data / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
    env = dict(os.environ, AUTOGRADE_DATA=str(data), PYTHONPATH=str(ROOT))
    fake = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_deepseek.py"), str(FAKE_PORT)], env=env)
    app = None
    try:
        wait_http(f"http://127.0.0.1:{FAKE_PORT}/stats")
        # ---- 建考试（进程内）：试卷、答案、答题纸
        store = Store(str(data))
        st = store.settings()
        exam = store.create_exam("大批量压测", 2)
        eid = exam["id"]
        d = store.exam_dir(eid)
        for sub, src in (("paper", paper), ("answers", answers), ("sheet", sheet)):
            shutil.copy(src, d / sub / f"{sub}.pdf")
        store.update_exam(eid, lambda e: e["files"].update(paper=["paper/paper.pdf"], answers=["answers/answers.pdf"],
                                                           sheet=["sheet/sheet.pdf"]))
        llm = DeepSeek(st["api_key"], st["base_url"], cache_dir=str(d / "cache"), usage=Usage())
        exam = exam_setup.run_setup(store, eid, llm, st, lambda f, m: None, generate_missing=False)
        tmpl = exam["template"]
        choice_nums = sorted((n for n, p in objective_places(tmpl).items() if p.get("group")), key=int)

        # ---- 模拟扫描件（缓存）
        import hashlib
        key = hashlib.sha1(Path(sheet).read_bytes()).hexdigest()[:8]     # 换了答题纸就重新生成
        pdf = CACHE / f"scan_149_{key}.pdf"
        idx_file = CACHE / f"scan_149_{key}.json"
        if not (pdf.exists() and idx_file.exists()):
            t = time.time()
            index = make_scan_pdf(sheet, tmpl, pdf)
            idx_file.write_text(json.dumps({str(k): v for k, v in index.items()}))
            print(f"generated {len(index)} pages in {time.time() - t:.0f}s")
        index = {int(k): v for k, v in json.loads(idx_file.read_text()).items()}
        size_mb = pdf.stat().st_size / 2 ** 20
        print(f"scan pdf: {len(index)} pages, {size_mb:.0f} MB")
        assert len(index) == 149

        # ---- 真服务：HTTP 上传
        app = subprocess.Popen([os.environ.get("AG_SERVER_PY", sys.executable), str(ROOT / "run.py"), "--port", str(APP_PORT),
                                "--no-browser"], env=env)        # AG_SERVER_PY：用别的 Python（比如 3.9）跑服务
        base = f"http://127.0.0.1:{APP_PORT}"
        wait_http(base + "/")
        peak = {"VmRSS": 0.0}
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                v = rss(app.pid).get("VmRSS", 0)
                peak["VmRSS"] = max(peak["VmRSS"], v)
                time.sleep(0.2)
        threading.Thread(target=sample, daemon=True).start()
        idle = rss(app.pid).get("VmRSS", 0)
        t0 = time.time()
        out = subprocess.run(["curl", "-s", "-F", f"files=@{pdf};type=application/pdf",
                              f"{base}/api/exams/{eid}/scans"], capture_output=True, text=True)
        t_up = time.time() - t0
        job = json.loads(out.stdout)["job"]
        print(f"upload {size_mb:.0f} MB: {t_up:.1f}s, server RSS after upload {rss(app.pid).get('VmRSS', 0):.0f} MB (idle {idle:.0f} MB)")
        phases, last_phase = {}, None
        while True:
            j = requests.get(f"{base}/api/jobs/{job['id']}").json()
            ph = j["message"].split(" ")[0].rstrip("…")
            if ph != last_phase:
                phases[ph] = time.time() - t0
                last_phase = ph
                print(f"  [{time.time() - t0:6.1f}s] {j['message']}  RSS {rss(app.pid).get('VmRSS', 0):.0f} MB")
            if j["status"] != "running":
                break
            time.sleep(0.5)
        total = time.time() - t0
        stop.set()
        hwm = rss(app.pid).get("VmHWM", 0)
        print(f"job {j['status']} in {total:.0f}s; server peak RSS {max(peak['VmRSS'], hwm):.0f} MB; result {j.get('result')}")
        assert j["status"] == "done", j

        # ---- 核对
        res = requests.get(f"{base}/api/exams/{eid}/results").json()
        rows = {r["id"]: r for r in res["rows"]}
        assert len(rows) == N_STUDENTS, len(rows)
        ex = requests.get(f"{base}/api/exams/{eid}").json()
        rep = ex.get("scan_report") or {}
        print("scan report:", json.dumps(rep, ensure_ascii=False, indent=1))
        bad = []
        for k in range(N_STUDENTS):
            sid = f"S{k + 1:03d}"
            s = requests.get(f"{base}/api/exams/{eid}/sheets/{sid}").json()
            pages = sorted(pr["index"] for pr in s["pages"])
            want = sorted(i for i, v in index.items() if v[0] == k)
            if pages != want:
                bad.append(f"{sid}: pages {pages} != {want}")
            miss = [p for p in (0, 1) if (k, p) in DROP]
            if (s.get("missing_pages") or []) != miss:
                bad.append(f"{sid}: missing_pages {s.get('missing_pages')} != {miss}")
            if 0 not in miss:
                got = [s["questions"][n].get("student") for n in choice_nums]
                exp = [expected_choice(k, j) for j in range(len(choice_nums))]
                if got != exp:
                    bad.append(f"{sid}: choices {got} != {exp}")
            if not s.get("align_ok", True):
                bad.append(f"{sid}: align failed")
        assert not bad, "\n".join(bad)
        notes = "\n".join(rep.get("notes") or [])
        assert rep.get("pages") == 149 and rep.get("sheets") == N_STUDENTS, rep
        assert "S038" in notes and "S071" in notes and "空白" in notes, notes
        assert rows["S038"]["missing_pages"] == [1] and rows["S038"]["states"]["20"] == "missing"
        csv = requests.get(f"{base}/api/exams/{eid}/export.csv").content.decode("utf-8-sig").splitlines()
        assert len(csv) == N_STUDENTS + 1 and "缺反面" in csv[38], csv[38]
        z = requests.get(f"{base}/api/exams/{eid}/export.zip")
        assert z.status_code == 200 and len(z.content) > 1_000_000
        print(f"export.zip {len(z.content) / 2 ** 20:.0f} MB; final RSS {rss(app.pid).get('VmRSS', 0):.0f} MB, "
              f"peak {rss(app.pid).get('VmHWM', 0):.0f} MB")
        time.sleep(0.5)
        assert not list(d.glob("export_*.zip")), "导出用的临时 zip 应该发完就删"

        # 网页上看一眼：复核页顶部的“最近一次上传”情况、缺页的 S038
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            page = b.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            page.goto(f"{base}/#/exam/{eid}/review/S038")
            page.wait_for_selector("text=这份答卷缺反面")
            page.wait_for_function("[...document.images].every(i => i.complete)")
            time.sleep(0.3)
            page.screenshot(path=str(CACHE / "review_after_upload.png"))
            b.close()

        # ---- 补扫：找到第 38 份缺的那一面，单独扫成一页上传 → 新的一份（只有第 2 页）→ 合并进 S038
        rescan = tmp / "rescan.pdf"
        make_scan_pdf(sheet, tmpl, rescan, only=[(37, 1)])
        out = subprocess.run(["curl", "-s", "-F", f"files=@{rescan};type=application/pdf", f"{base}/api/exams/{eid}/scans"],
                             capture_output=True, text=True)
        job = json.loads(out.stdout)["job"]
        while requests.get(f"{base}/api/jobs/{job['id']}").json()["status"] == "running":
            time.sleep(0.5)
        ex = requests.get(f"{base}/api/exams/{eid}").json()
        assert ex["sheets"][-1] == "S076" and ex["scan_report"]["sheets"] == 1, ex["scan_report"]
        assert any("S076" in n and "缺正面" in n for n in ex["scan_report"]["notes"]), ex["scan_report"]["notes"]
        assert any("S076" in n and "打开 S038 把它合并进去" in n for n in ex["scan_report"]["notes"]), ex["scan_report"]["notes"]
        rows = {r["id"]: r for r in requests.get(f"{base}/api/exams/{eid}/results").json()["rows"]}
        assert rows["S076"]["present_pages"] == [1] and rows["S038"]["missing_pages"] == [1]
        bad_merge = requests.post(f"{base}/api/exams/{eid}/sheets/S038/merge", json={"source": "S039"})
        assert bad_merge.status_code == 400 and "都有正面" in bad_merge.json()["detail"], bad_merge.text

        # 在网页上操作：复核页顶部的上传情况、S038 的缺页提示 → 选 S076 合并；再删掉 S075
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            page = b.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("dialog", lambda dlg: dlg.accept())
            page.goto(f"{base}/#/exam/{eid}/review/S038")
            page.wait_for_selector("text=这份答卷缺反面")
            page.wait_for_selector("text=最近一次上传")
            page.wait_for_selector("text=如果这是补扫的")           # 提醒默认展开
            time.sleep(0.4)
            page.screenshot(path=str(CACHE / "review_missing.png"))
            opts = page.eval_on_selector_all(".notice select option", "els => els.map(e => e.value)")
            assert opts and opts[0] == "S076", opts          # 新上传的（补扫）排在最前面
            page.select_option(".notice select", "S076")
            page.click(".notice button:text-is('合并')")
            page.wait_for_function("!document.body.innerText.includes('这份答卷缺反面')", timeout=30000)
            page.wait_for_selector(".roster a.on >> text=S038")
            page.wait_for_selector("text=提醒都已处理")             # 补扫那条提醒标成已处理
            page.goto(f"{base}/#/exam/{eid}/review/S075")
            page.wait_for_selector("h2:text-is('S075')")
            page.click("button:text-is('删除这份')")
            page.wait_for_function("(() => { const r = document.querySelector('.roster');"
                                   " return !!r && r.innerText.includes('S074') && !r.innerText.includes('S075'); })()",
                                   timeout=30000)
            page.screenshot(path=str(CACHE / "review_after_delete.png"))
            b.close()
        assert not errors, errors
        s38 = requests.get(f"{base}/api/exams/{eid}/sheets/S038").json()
        assert s38["missing_pages"] == [] and s38["merged_from"] == ["S076"], s38.get("missing_pages")
        assert [p["template_page"] for p in s38["pages"]] == [0, 1]
        for n in ("20", "21"):
            q = s38["questions"][n]
            assert q["status"] == "done" and q["crops"] and all(c.startswith("work/S038/") for c in q["crops"]), q
        for c in [c for n in ("20", "21") for c in s38["questions"][n]["crops"]] + [p["display"] for p in s38["pages"]] \
                + s38["annotated"]:
            assert requests.get(f"{base}/api/exams/{eid}/file", params={"p": c}).status_code == 200, c
        assert requests.get(f"{base}/api/exams/{eid}/sheets/S076").status_code == 404
        # ---- 删掉的那份（S075）不在了；编号不会被后来的答卷重用
        assert requests.get(f"{base}/api/exams/{eid}/sheets/S075").status_code == 404
        res = requests.get(f"{base}/api/exams/{eid}/results").json()
        ids = [r["id"] for r in res["rows"]]
        assert len(ids) == N_STUDENTS - 1 and "S075" not in ids and "S076" not in ids, ids
        assert not {r["id"]: r for r in res["rows"]}["S038"]["missing_pages"]
        ex = requests.get(f"{base}/api/exams/{eid}").json()
        assert ex["sheet_seq"] == 76
        rep = ex["scan_report"]
        assert rep["partial"] == 0 and rep["notes"][0].startswith("✓ S076") and rep["notes"][0].endswith("（已并进 S038）"), rep
        print("merge/delete OK")
        print("fake stats", requests.get(f"http://127.0.0.1:{FAKE_PORT}/stats").json())
        print("\nBULK OK", tmp)
    finally:
        if app:
            app.terminate()
        fake.terminate()


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:4]] or DEFAULT
    if len(args) != 3 or not all(a.exists() for a in args):
        sys.exit("用法：python tests/test_bulk.py 试卷.pdf 答案.pdf 答题纸.pdf（答题纸要是两页的）")
    main(*args)
