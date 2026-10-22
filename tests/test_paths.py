"""补充测试：参数回退、模型输出容错、改转写后重评、老师按小问改分。"""
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from autograde.grading import round_step, validate_grade
from autograde.llm import DeepSeek


def check_validate():
    q = {"number": "15", "score": 12, "parts": [{"label": "(1)", "score": 5}, {"label": "(2)", "score": 7}]}
    # 标签写法不同、分数是字符串、超出满分、总分写错
    r = validate_grade({"parts": [{"label": "第(1)问", "score": "4分"}, {"label": "（2）", "score": 9}], "total": 20,
                        "confidence": "high"}, q)
    assert [p["score"] for p in r["parts"]] == [4, 7] and r["score"] == 11 and r["needs_review"], r
    # 只给了一个小问
    r = validate_grade({"parts": [{"label": "(1)", "score": 5}], "confidence": "high"}, q)
    assert r["score"] == 5 and r["needs_review"] and "(2)" in r["review_notes"], r
    # 小问分值没配齐 -> 按全题
    q2 = {"number": "16", "score": 10, "parts": [{"label": "(1)", "score": None}]}
    r = validate_grade({"parts": [{"label": "(1)", "score": 3}, {"label": "(2)", "score": 4}], "confidence": "medium"}, q2)
    assert r["score"] == 7 and len(r["parts"]) == 1, r
    # 0.5 分一档
    assert round_step(3.74, 0.5) == 3.5 and round_step(3.75, 0.5) == 3.5 and round_step(3.76, 0.5) == 4
    assert round_step(7.5, 1) == 7 and round_step(7.51, 1) == 8
    # 完全不是 JSON 对象
    r = validate_grade(["oops"], q)
    assert r["score"] == 0 and r["needs_review"]
    print("validate_grade ok")


def check_param_drop():
    import uvicorn, fake_deepseek
    srv = uvicorn.Server(uvicorn.Config(fake_deepseek.app, host="127.0.0.1", port=8766, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    llm = DeepSeek("sk-x", "http://127.0.0.1:8766")
    res = llm.chat([{"role": "system", "content": "你负责从答题纸上读出考生信息。"},
                    {"role": "user", "content": "只输出 json"}], "deepseek-v4-flash", json_mode=True, thinking=True, effort="high")
    assert "25090315" in res["content"], res
    print("param drop ok (thinking 被拒后自动去掉重试)")
    try:
        DeepSeek("", "http://127.0.0.1:8766").chat([{"role": "user", "content": "hi"}], "deepseek-flash")
        raise AssertionError("should fail without key")
    except Exception as e:
        assert "API Key" in str(e)
    srv.should_exit = True


if __name__ == "__main__":
    check_validate()
    check_param_drop()
