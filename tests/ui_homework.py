"""浏览器测试：平时作业。
新建（选“平时作业”，上传试卷照片，不传答案）→ AI 录题并做选择、填空题（第 13 题两遍不一样，标红）→
批量确认、单题确认 → 题目卡片不显示分值 → 生成答题纸（不印分值）→ 模拟作答扫描上传 →
复核页：对/半对/错、错在哪、改判 → 成绩表 → 改成“考试”后按分数重新批。

python tests/ui_homework.py [截图目录]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_sheetgen as T   # 复用：渲染、模拟作答、模拟扫描

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(tempfile.gettempdir()) / "autograde_ui_homework")
FAKE, APP = 8768, 8013


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="ag_uihw_"))
    data = tmp / "data"
    data.mkdir()
    (data / "settings.json").write_text(json.dumps({"api_key": "sk-fake-1234", "base_url": f"http://127.0.0.1:{FAKE}",
                                                    "samples": 1, "max_workers": 4, "scan_dpi": 220}), encoding="utf-8")
    paper = tmp / "作业照片.jpg"            # 内容无所谓：模拟接口按固定的题目录入
    img = np.full((1600, 1200, 3), 255, np.uint8)
    cv2.putText(img, "Homework 5.2", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
    cv2.imwrite(str(paper), img)
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
        requests.post(f"http://127.0.0.1:{FAKE}/state", json={"solve_disagree": ["13"], "hw_verdict": {"15": "半对", "16": "错"}})
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("dialog", lambda d: d.accept())

            # ---- 新建：平时作业
            page.goto(base + "/#/new")
            page.click("label.kind:has-text('平时作业')")
            page.wait_for_function("document.querySelector('input[name=title]').placeholder.includes('作业')")
            assert "AI 会先做选择、填空题" in page.inner_text("form")
            page.fill("input[name=title]", "5.2 导数的运算 作业")
            page.set_input_files("input[name=paper]", str(paper))
            page.screenshot(path=str(OUT / "01_new.png"), full_page=True)
            page.click("button[type=submit]")
            page.wait_for_selector("text=录题和识别版面完成", timeout=120000)
            page.wait_for_selector("text=AI，待核对")
            eid = requests.get(base + "/api/exams").json()[0]["id"]
            ex = requests.get(f"{base}/api/exams/{eid}").json()
            assert ex["kind"] == "homework"
            qs = {q["number"]: q for q in ex["questions"]}
            obj = [n for n, q in qs.items() if q["type"] in ("选择题", "填空题")]
            assert len(obj) == 14 and all(qs[n]["answer_source"] == "ai" and qs[n]["answer"] for n in obj), [(n, qs[n]["answer"]) for n in obj]
            assert qs["13"]["ai_solve"]["agree"] is False
            # 题目卡片：没有“满分”，按钮是“让 AI 重写答案”/“让 AI 写参考答案”
            assert page.locator(".qcard header label:has-text('满分')").count() == 0
            assert page.locator("#objective th:text-is('分值')").count() == 0
            assert page.locator(".kindsel").input_value() == "homework"
            row13 = page.locator("#objective tr", has=page.locator("td.num:text-is('13')"))
            assert "两次独立解答不一样" in row13.inner_text(), row13.inner_text()
            page.evaluate("window.scrollTo(0, document.querySelector('#objective').getBoundingClientRect().top + window.scrollY - 64)")
            time.sleep(0.3)
            page.screenshot(path=str(OUT / "02_ai_answers.png"))

            # ---- 批量确认两遍一致的；第 13 题单独确认
            page.click("button:has-text('确认两遍一致的 13 道')")
            page.wait_for_selector("text=题目已保存")
            page.wait_for_function("document.querySelectorAll('#objective .badge.ok').length === 13")
            row13 = page.locator("#objective tr", has=page.locator("td.num:text-is('13')"))
            row13.locator("button:text-is('确认')").click()
            page.wait_for_function("document.querySelectorAll('#objective .badge.ok').length === 14")
            assert page.locator("button:has-text('AI 重做')").count() == 0
            ex = requests.get(f"{base}/api/exams/{eid}").json()
            assert all(q["confirmed"] for q in ex["questions"] if q["type"] != "解答题")
            assert not any("AI 做的" in w or "两次独立" in w for w in ex["warnings"]), ex["warnings"]
            page.screenshot(path=str(OUT / "03_confirmed.png"), full_page=True)

            # ---- 生成答题纸：不印分值
            page.click("text=答题纸版面")
            page.wait_for_selector("text=按题目生成答题纸")
            assert "按小问数给方框高度" in page.inner_text("form")
            page.click("button:text-is('生成答题纸')")
            page.wait_for_selector("text=答题纸是按题目生成的", timeout=60000)
            ex = requests.get(f"{base}/api/exams/{eid}").json()
            plan = ex["template"]["generated"]["plan"]
            assert plan["scored"] is False
            pdf = requests.get(f"{base}/api/exams/{eid}/answer-sheet.pdf").content
            page.wait_for_function("[...document.querySelectorAll('.page-view img')].every(i => i.complete)")
            assert page.locator(".region text:has-text('分')").count() == 0
            page.screenshot(path=str(OUT / "04_generated.png"), full_page=True)

            # ---- 两个学生作答、扫描、上传
            rng = np.random.default_rng(8)
            blank = T.render(pdf)
            key = {n: qs[n]["answer"] for n in ("11", "12", "13", "14")}
            sheets = [T.fill_sheet(blank, plan, rng, "2501", key),
                      T.fill_sheet(blank, plan, rng, "2502", dict(key, **{"11": "A" if key["11"] != "A" else "B"}))]
            T.to_pdf(tmp / "scan.pdf", [T.scan(im, rng) for pages in sheets for im in pages])
            page.click("text=上传扫描件")
            page.wait_for_selector(".dropzone")
            page.set_input_files(".dropzone input[type=file]", str(tmp / "scan.pdf"))
            page.click("text=开始批改")
            page.wait_for_selector(".roster", timeout=180000)
            page.wait_for_selector(".roster a >> text=对 ")
            chips = page.eval_on_selector_all(".roster a:first-of-type .chip", "els => els.map(e => e.innerText)")
            assert "15题 ½" in chips and "16题 ✗" in chips and "17题 ✓" in chips, chips
            assert page.locator("select[aria-label=排序] option:text-is('做对从多到少')").count() == 1

            # ---- 第 15 题：半对，错在哪，小问逐个判
            page.click(".qtabs button:has-text('第15题')")
            page.wait_for_selector(".verdicts")
            assert page.locator(".score-big").first.inner_text().strip() == "½ 半对"
            txt = page.inner_text(".qview")
            assert "错在哪：" in txt and "第15题第二步计算错误" in txt and "订正提示：" in txt, txt[:500]
            assert page.locator(".verdicts .vb.ai.mark:has-text('半对')").count() == 1
            assert page.locator(".parts-t td:has-text('✓ 对')").count() == 1 and page.locator(".parts-t td:has-text('✗ 错')").count() == 1
            page.wait_for_function("[...document.images].every(i => i.complete)")
            page.screenshot(path=str(OUT / "05_review_half.png"), full_page=True)
            # 改判：第 16 题判对
            page.click(".qtabs button:has-text('第16题')")
            page.wait_for_selector(".score-big:has-text('错')")
            page.click(".verdicts .vb:has-text('对'):not(:has-text('半'))")
            page.wait_for_selector("text=已改判为“对”")
            page.wait_for_selector(".score-big:has-text('✓ 对')")
            assert page.locator(".verdicts .vb.mark:has-text('对')").count() == 1
            page.wait_for_selector(".roster a.on .chip:has-text('16题 ✓')")
            # 选择题：做对几题，不显示分数
            page.click(".qtabs button:has-text('选择题')")
            page.wait_for_selector(".review-objt")
            assert page.inner_text(".score-big").startswith("做对"), page.inner_text(".score-big")
            assert page.locator(".review-objt .verdict small").count() == 0
            page.screenshot(path=str(OUT / "06_review_choice.png"), full_page=True)
            csv = requests.get(f"{base}/api/exams/{eid}/export.csv").content.decode("utf-8-sig")
            assert "做对题数（共17题）" in csv and "半对" in csv, csv

            # ---- 改成“考试”：确认后按分数重新批
            page.select_option(".kindsel", "exam")
            page.wait_for_selector("text=已改成考试")
            page.wait_for_selector("text=改成考试，重新批改完成", timeout=180000)
            page.click("text=复核与成绩")
            page.wait_for_selector(".roster")
            page.click(".qtabs button:has-text('第15题')")
            page.wait_for_selector(".scorebar")
            ex = requests.get(f"{base}/api/exams/{eid}").json()
            assert ex["kind"] == "exam"
            res = requests.get(f"{base}/api/exams/{eid}/results").json()
            assert res["scored"] and all(r["scores"]["15"] == 11 for r in res["rows"]), [r["scores"] for r in res["rows"]]
            page.screenshot(path=str(OUT / "07_exam_mode.png"), full_page=True)
            b.close()
        print("JS errors:", errors)
        assert not errors, errors
        print("UI HOMEWORK OK", OUT)
    finally:
        app.terminate()
        fake.terminate()


if __name__ == "__main__":
    main()
