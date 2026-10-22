"""浏览器冒烟测试：模拟 DeepSeek + 真服务 + 无头 Chromium，走一遍 建考试 → 版面 → 上传扫描件 → 复核改分 → 导出。"""
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
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/work/sample.png"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/claude/work/scratch/ui")
OUT.mkdir(parents=True, exist_ok=True)


def wait_http(url, t=20):
    for _ in range(int(t * 10)):
        try:
            requests.get(url, timeout=1)
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("not up: " + url)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ag_ui_"))
    data = tmp / "data"
    data.mkdir()
    (data / "settings.json").write_text(json.dumps({"api_key": "sk-fake-1234", "base_url": "http://127.0.0.1:8765",
                                                    "samples": 2, "max_workers": 4, "scan_dpi": 150}), encoding="utf-8")
    env = dict(os.environ, AUTOGRADE_DATA=str(data), PYTHONPATH=str(ROOT))
    fake = subprocess.Popen([sys.executable, str(ROOT / "tests" / "fake_deepseek.py"), "8765"], env=env)
    app = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--port", "8011", "--no-browser"], env=env)
    try:
        wait_http("http://127.0.0.1:8765/stats")
        wait_http("http://127.0.0.1:8011/")
        # 准备文件
        paper = tmp / "paper.png"   # 模拟拍照的试卷（模拟服务器不看内容）
        sheet_img = np.full((1600, 1130, 3), 255, np.uint8)
        for k in range(12):
            cv2.putText(sheet_img, f"{k + 1}. question text ......", (80, 120 + k * 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        cv2.imwrite(str(paper), sheet_img)
        img = cv2.imread(SAMPLE)
        cv2.imwrite(str(tmp / "stu1.png"), img)
        cv2.imwrite(str(tmp / "stu2_upside_down.jpg"), cv2.rotate(cv2.resize(img, None, fx=1.3, fy=1.3), cv2.ROTATE_180))

        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.goto("http://127.0.0.1:8011/")
            page.wait_for_selector("text=还没有考试")
            page.screenshot(path=str(OUT / "01_empty.png"))
            page.click("text=新建考试 >> nth=0")
            page.fill("input[name=title]", "高二第二学期数学小练2")
            page.set_input_files("input[name=paper]", str(paper))
            page.set_input_files("input[name=sheet]", SAMPLE)
            page.screenshot(path=str(OUT / "02_new.png"))
            page.click("button[type=submit]")
            page.wait_for_selector("text=第15题", timeout=60000)
            page.wait_for_function("!document.querySelector('#jobbox .progress')", timeout=60000)
            time.sleep(0.5)
            page.screenshot(path=str(OUT / "03_questions.png"), full_page=True)
            # 确认第 15 题
            page.click("section.qcard:not(#objective) >> nth=0 >> button:text-is('确认无误')")
            page.wait_for_selector("text=AI 生成，已确认")
            page.click("text=答题纸版面")
            page.wait_for_selector(".page-view img")
            page.wait_for_function("document.querySelector('.page-view img').complete")
            page.click(".region.k-解答题 >> nth=0")
            time.sleep(0.3)
            page.screenshot(path=str(OUT / "04_layout.png"), full_page=True)
            page.click("text=上传扫描件")
            page.wait_for_selector(".dropzone")
            page.set_input_files(".dropzone input[type=file]", [str(tmp / "stu1.png"), str(tmp / "stu2_upside_down.jpg")])
            page.wait_for_selector("text=已选择 2 个文件")
            page.click("text=开始批改")
            try:
                page.wait_for_selector(".roster", timeout=120000)
            except Exception:
                page.screenshot(path=str(OUT / "fail_scans.png"), full_page=True)
                eid = requests.get("http://127.0.0.1:8011/api/exams").json()[0]["id"]
                ex = requests.get(f"http://127.0.0.1:8011/api/exams/{eid}").json()
                print("LAST JOB:", ex.get("last_job"), "JOB:", ex.get("job"))
                print("JS errors:", errors)
                raise
            page.wait_for_selector(".crop img")
            page.wait_for_function("[...document.querySelectorAll('.crop img')].every(i => i.complete)")
            time.sleep(0.5)
            page.screenshot(path=str(OUT / "05_review.png"), full_page=True)
            # 第 17 题（AI 打了超范围的分，被标记复核）→ 在打分条上改成 10 分
            page.click(".qtabs button >> text=第17题")
            page.wait_for_selector("text=建议复核")
            page.click(".scorebar button[title='给 10 分']")
            page.wait_for_selector("text=老师改分")
            page.keyboard.press("ArrowDown")
            page.wait_for_selector(".roster a.on >> text=S002")
            page.keyboard.press("Enter")
            page.wait_for_selector("text=已确认 AI 分数")
            time.sleep(0.4)
            page.screenshot(path=str(OUT / "06_review_after.png"), full_page=True)
            page.click("text=看批改后的第 1 页")
            page.wait_for_selector(".lightbox img")
            page.wait_for_function("document.querySelector('.lightbox img').complete")
            page.screenshot(path=str(OUT / "07_annotated.png"))
            page.click(".lightbox")
            page.goto("http://127.0.0.1:8011/#/settings")
            page.wait_for_selector("text=保存设置")
            page.screenshot(path=str(OUT / "08_settings.png"), full_page=True)
            # 窄屏（平板竖屏）
            page.set_viewport_size({"width": 820, "height": 1100})
            page.go_back()
            page.wait_for_selector(".roster")
            time.sleep(0.5)
            page.screenshot(path=str(OUT / "09_tablet.png"))
            b.close()
        eid = requests.get("http://127.0.0.1:8011/api/exams").json()[0]["id"]
        csv = requests.get(f"http://127.0.0.1:8011/api/exams/{eid}/export.csv")
        print(csv.content.decode("utf-8-sig"))
        z = requests.get(f"http://127.0.0.1:8011/api/exams/{eid}/export.zip")
        print("zip bytes", len(z.content), z.headers.get("content-disposition"))
        print("fake stats", requests.get("http://127.0.0.1:8765/stats").json())
        print("JS errors:", errors)
    finally:
        app.terminate(); fake.terminate()


if __name__ == "__main__":
    main()
