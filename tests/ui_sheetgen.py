"""浏览器测试：导入班级名单 → 新建作业（写题目结构）→ 生成答题纸 → 下载通用版和每人一份 →
模拟学生作答扫描后上传 → 复核页搜索学生（姓名、学号、拼音首字母）、排序、缺交名单、改学号。

python tests/ui_sheetgen.py [截图目录]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_sheetgen as T   # 复用：渲染、模拟作答、模拟扫描

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(tempfile.gettempdir()) / "autograde_ui_sheetgen")
FAKE, APP = 8767, 8012


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ag_uigen_"))
    data = tmp / "data"
    data.mkdir()
    (data / "settings.json").write_text(json.dumps({"api_key": "sk-fake-1234", "base_url": f"http://127.0.0.1:{FAKE}",
                                                    "samples": 1, "max_workers": 4, "scan_dpi": 220}), encoding="utf-8")
    env = dict(os.environ, AUTOGRADE_DATA=str(data), PYTHONPATH=str(ROOT))
    fake = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_deepseek.py"), str(FAKE)], env=env)
    app = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--port", str(APP), "--no-browser"], env=env)
    base = f"http://127.0.0.1:{APP}"
    try:
        for url in (f"http://127.0.0.1:{FAKE}/stats", base + "/"):
            for _ in range(200):
                try:
                    requests.get(url, timeout=1)
                    break
                except Exception:
                    time.sleep(0.1)
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("dialog", lambda d: d.accept())

            # ---- 名单
            page.goto(base + "/#/roster")
            page.wait_for_selector("text=还没有名单")
            page.fill("input[name=name]", "高一5班")
            page.fill("textarea[name=text]", T.ROSTER)
            page.click("button:text-is('导入名单')")
            page.wait_for_selector("text=5 人")
            page.screenshot(path=str(OUT / "01_roster.png"), full_page=True)

            # ---- 新建作业：写题目结构
            page.goto(base + "/#/new")
            page.fill("input[name=title]", "第3周作业")
            page.fill("textarea[name=structure]", T.STRUCT)
            page.click("button[type=submit]")
            page.wait_for_selector("text=选择题 8 道")
            eid = requests.get(base + "/api/exams").json()[0]["id"]
            ex = requests.get(f"{base}/api/exams/{eid}").json()
            for q in ex["questions"]:
                q["answer"] = T.KEY.get(q["number"], "（略）" if q["type"] == "解答题" else "")
            requests.put(f"{base}/api/exams/{eid}/questions", json={"questions": ex["questions"]}).raise_for_status()
            page.reload()
            page.wait_for_selector("text=选择题 8 道")
            page.screenshot(path=str(OUT / "02_questions.png"), full_page=True)

            # ---- 生成答题纸
            page.click("text=答题纸版面")
            page.wait_for_selector("text=按题目生成答题纸")
            page.wait_for_function("document.querySelector('input[name=id_digits]').value === '4'")
            page.screenshot(path=str(OUT / "03_generate_form.png"), full_page=True)
            page.click("button:text-is('生成答题纸')")
            page.wait_for_selector("text=答题纸是按题目生成的", timeout=60000)
            page.wait_for_selector("text=每人一份 PDF")
            page.wait_for_function("[...document.querySelectorAll('.page-view img')].every(i => i.complete)")
            page.screenshot(path=str(OUT / "04_generated.png"), full_page=True)

            # ---- 拖第 13 题方框的下边框（往上拖，变矮）→ 松手后重新排版
            def box_mm(q):
                plan = requests.get(f"{base}/api/exams/{eid}").json()["template"]["generated"]["plan"]
                return sum(b["rect"][3] - b["rect"][1] for pg in plan["pages"] for b in pg["boxes"] if b["q"] == q), plan
            h13, _ = box_mm("13")
            grip = page.locator(".grip[data-q='13'] .bar").first
            grip.scroll_into_view_if_needed()
            bb = grip.bounding_box()
            x, y = bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] / 2
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x, y - 70, steps=8)
            page.wait_for_selector(".grip-label")
            page.screenshot(path=str(OUT / "04b_dragging.png"))
            page.mouse.up()
            page.wait_for_selector("text=已重新排版", timeout=30000)
            h13b, plan_b = box_mm("13")
            assert h13b < h13 - 10 and abs(plan_b["heights"]["13"] - h13b) < 1, (h13, h13b, plan_b["heights"])
            # ---- 侧栏直接填高度：第 14 题 150 毫米
            page.wait_for_function("[...document.querySelectorAll('.page-view img')].every(i => i.complete)")
            page.locator(".region.k-解答题", has_text="第14题").first.click()
            page.wait_for_selector("text=第14题方框高度")
            page.fill(".side input[type=number][max='800']", "150")
            page.click(".side button:text-is('改')")
            page.wait_for_selector("text=第14题方框改成 150 毫米", timeout=30000)
            h14, plan_c = box_mm("14")
            assert abs(h14 - 150) < 1 and plan_c["heights"]["13"] == plan_b["heights"]["13"], (h14, plan_c["heights"])
            print("drag/resize OK:", round(h13), "->", round(h13b), "; 14 ->", round(h14), "pages", len(plan_c["pages"]))
            page.screenshot(path=str(OUT / "04c_resized.png"), full_page=True)
            pdf = requests.get(f"{base}/api/exams/{eid}/answer-sheet.pdf")
            assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF", pdf.status_code
            cid = requests.get(base + "/api/rosters").json()[0]["id"]
            per = requests.get(f"{base}/api/exams/{eid}/answer-sheet.pdf", params={"classes": cid})
            import pymupdf
            n_blank = pymupdf.open(stream=pdf.content, filetype="pdf").page_count
            n_per = pymupdf.open(stream=per.content, filetype="pdf").page_count
            assert n_per == 5 * n_blank, (n_blank, n_per)
            assert "filename*=UTF-8''" in per.headers.get("content-disposition", "")
            print("pdf pages:", n_blank, "personal:", n_per)

            # ---- 先传一页白纸：一份答卷都分不出来 → 留在上传页，把原因摆出来；复核页也说清楚
            T.to_pdf(tmp / "blank.pdf", [np.full((2339, 1654, 3), 250, np.uint8)])
            page.click("text=上传扫描件")
            page.wait_for_selector(".dropzone")
            page.set_input_files(".dropzone input[type=file]", str(tmp / "blank.pdf"))
            page.click("text=开始批改")
            page.wait_for_selector("text=这次上传没有分出答卷", timeout=60000)
            page.wait_for_selector("details.notice.err >> text=是空白页，已跳过")
            assert "#/exam/" in page.url and page.url.endswith("/scans"), page.url
            page.screenshot(path=str(OUT / "04d_nothing_uploaded.png"), full_page=True)
            page.click("text=复核与成绩")
            page.wait_for_selector("h2:text-is('这次上传没有分出答卷')")
            page.wait_for_selector("text=是空白页，已跳过")
            page.wait_for_selector("text=手机拍也可以")

            # ---- 模拟作答、扫描、上传
            ex = requests.get(f"{base}/api/exams/{eid}").json()
            plan = ex["template"]["generated"]["plan"]
            rng = np.random.default_rng(5)
            blank = T.render(pdf.content)
            personal = T.render(per.content)[2 * n_blank: 3 * n_blank]          # 名单第 3 个人（王五）那一份
            sheets = [T.fill_sheet(blank, plan, rng, "2501", {"1": "A", "2": "B", "7": "BD"}, name="张三"),
                      T.fill_sheet(blank, plan, rng, "2502", {"1": "C", "8": "AC"}, name="李四"),
                      T.fill_sheet(personal, plan, rng, "", {"1": "A"}),
                      T.fill_sheet(blank, plan, rng, "2501", {"1": "B"}, name="钱七")]          # 涂成了张三的学号
            scans = [T.scan(img, rng) for pages in sheets for img in pages]
            T.to_pdf(tmp / "scan.pdf", scans)
            page.click("text=上传扫描件")
            page.wait_for_selector(".dropzone")
            page.set_input_files(".dropzone input[type=file]", str(tmp / "scan.pdf"))
            page.wait_for_selector("text=已选择 1 个文件")
            page.click("text=开始批改")
            page.wait_for_selector(".roster", timeout=120000)
            page.wait_for_selector(".roster >> text=张三")
            page.wait_for_function("[...document.images].every(i => i.complete)")
            time.sleep(0.4)
            page.screenshot(path=str(OUT / "05_review.png"), full_page=True)

            # ---- 搜索：拼音首字母、学号、姓名
            page.fill(".roster input[type=search]", "zs")              # 张三，以及涂成张三学号的那份
            page.wait_for_function("document.querySelectorAll('.roster a').length === 2")
            assert all("张三" in t for t in page.eval_on_selector_all(".roster a", "els => els.map(e => e.innerText)"))
            page.screenshot(path=str(OUT / "06_search.png"))
            page.fill(".roster input[type=search]", "2502")
            page.wait_for_function("document.querySelectorAll('.roster a').length === 1 && document.querySelector('.roster a').innerText.includes('李四')")
            page.keyboard.press("Enter")
            page.wait_for_selector(".roster a.on >> text=李四")
            page.fill(".roster input[type=search]", "wang")
            page.wait_for_function("document.querySelectorAll('.roster a').length === 1 && document.querySelector('.roster a').innerText.includes('王五')")
            page.fill(".roster input[type=search]", "没有这个人")
            page.wait_for_selector("text=没有找到")
            page.fill(".roster input[type=search]", "")
            page.select_option(".roster select", "sid")
            names = page.eval_on_selector_all(".roster a b", "els => els.map(e => e.innerText)")
            assert names == ["张三", "张三", "李四", "王五"], names
            # ---- 重号：两份都标“学号待核对”；没交的人：赵六、钱七
            assert page.locator(".roster .chip:text-is('学号待核对')").count() == 2
            page.wait_for_selector("text=名单里还有 2 人没有答卷")
            page.click("text=名单里还有 2 人没有答卷")
            page.wait_for_selector("text=赵六（2504）")
            page.click(".roster a >> text=S004")
            page.wait_for_selector("text=学号待核对：")
            page.screenshot(path=str(OUT / "07_absent.png"))
            # ---- 老师看了答题纸，改成钱七 2505 → 重号和“钱七没交”都消失
            page.fill("input[placeholder=姓名]", "钱七")
            page.fill("input[placeholder=学号]", "2505")
            page.click("button:text-is('保存')")
            page.wait_for_selector("text=名单里还有 1 人没有答卷")
            page.wait_for_function("!document.body.innerText.includes('学号待核对：')")
            assert page.locator(".roster .chip:text-is('学号待核对')").count() == 0
            page.wait_for_selector(".roster >> text=钱七")
            page.screenshot(path=str(OUT / "08_fixed.png"))
            # ---- 用“/”快捷键聚焦搜索框
            page.click("h2")
            page.keyboard.press("/")
            assert page.evaluate("document.activeElement.type") == "search"
            b.close()
        csv = requests.get(f"{base}/api/exams/{eid}/export.csv").content.decode("utf-8-sig")
        assert "赵六" in csv and "没有答卷" in csv and "钱七,2505" in csv, csv
        print(csv)
        print("JS errors:", errors)
        assert not errors, errors
        print("UI SHEETGEN OK", OUT)
    finally:
        app.terminate()
        fake.terminate()


if __name__ == "__main__":
    main()
