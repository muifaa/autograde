"""本地模拟的 DeepSeek 接口，只用于开发测试（不调用真模型）。

会按官方文档检查请求格式：图片只能放在 user 消息里、图片必须是 data URL 或 https、
JSON 模式下提示词里要有 "json" 字样等；格式不对返回 400。
运行：python tests/fake_deepseek.py 8765
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
_lock = threading.Lock()
STATE = {"calls": 0, "by_task": {}, "fail_once": set()}

SAMPLE_QUESTIONS = [
    {"number": str(i), "type": "填空题", "score": 4 if i <= 4 else 5, "stem": f"第{i}题（填空）……", "parts": [], "figure": "", "continued": False}
    for i in range(1, 11)
] + [
    {"number": str(i), "type": "选择题", "score": 4.5, "stem": f"第{i}题（选择）A. … B. … C. … D. …", "parts": [], "figure": "", "continued": False}
    for i in range(11, 15)
] + [
    {"number": "15", "type": "解答题", "score": 12,
     "stem": "如图，在三棱锥 $P-ABC$ 中，$PA\\perp$ 平面 $ABC$，$AB\\perp AC$。",
     "parts": [{"label": "(1)", "text": "求证：平面 $PAC\\perp$ 平面 $PAB$；", "score": None},
               {"label": "(2)", "text": "若 $AC=4$，$M$ 为 $PC$ 中点，求点 $M$ 到平面 $PAB$ 的距离。", "score": None}],
     "figure": "三棱锥 P-ABC，PA 垂直底面 ABC，∠BAC=90°", "continued": False},
    {"number": "16", "type": "解答题", "score": 12,
     "stem": "在矩形 $ABCD$ 中，$AB=4$，$AD=2$，$E$ 为 $CD$ 中点，将 $\\triangle ADE$ 沿 $AE$ 折起，使 $BD=2\\sqrt{3}$，$G$ 为 $AE$ 中点。",
     "parts": [{"label": "(1)", "text": "求证：$DG\\perp$ 平面 $ABCE$；", "score": 6},
               {"label": "(2)", "text": "求直线 $BD$ 与平面 $ABCE$ 所成角的大小。", "score": 6}],
     "figure": "", "continued": False},
    {"number": "17", "type": "解答题", "score": 12,
     "stem": "四边形 $ABCD$ 与 $ABEF$ 都是边长为 1 的正方形，二面角 $D-AB-E$ 为 $60^\\circ$。",
     "parts": [{"label": "(1)", "text": "求证：$AE$ 与 $BC$ 是异面直线；", "score": 5},
               {"label": "(2)", "text": "求异面直线 $AE$ 与 $BC$ 所成角的大小。", "score": 7}],
     "figure": "", "continued": False},
]

# 用户那份答案的填空/选择答案（模拟“模型看图录对了”）
KEY_OBJ = {"1": "\\sqrt{5}", "2": "5-\\mathrm{i}", "3": "\\frac{\\pi}{2}", "4": "-\\frac{\\sqrt{5}}{5}", "5": "-\\frac{4}{5}",
           "6": "\\frac{\\pi}{2}", "7": "5\\sqrt{7}", "8": "\\frac{1}{3}\\vec{a}-\\frac{1}{2}\\vec{b}", "9": "\\frac{4}{3}",
           "10": "\\sqrt{10}+1", "11": "[2,4]", "12": "-\\frac{16}{15}", "13": "B", "14": "C", "15": "C", "16": "D"}

# 模拟“模型从拼图里认出的学生填空答案”（真模型会看图）
STUDENT_FILL = {"1": "\\sqrt{5}", "2": "5-i", "3": "90^{\\circ}", "4": "-\\frac{\\sqrt5}{5}", "5": "\\frac{4}{5}",
                "6": "\\frac{\\pi}{2}", "7": "5\\sqrt{7}", "8": "\\frac13\\vec a-\\frac12\\vec b", "9": "4/3",
                "10": "1+\\sqrt{10}", "11": "2\\le\\omega\\le4", "12": "-\\frac{16}{15}"}
JUDGE_OK = {"9", "10", "11"}       # 写法不同但等价；3（90°，题目要求 rad）和 5（漏了负号）判错

TRANSCRIPTS = {
    "15": "(1) 证：\n∵ $PA\\perp$ 面 $ABC$\n∴ $PA\\perp AC$\n∵ $AB\\perp AC$ 且 $PA\\cap AB=A$\n∴ $CA\\perp$ 面 $ABP$\n∵ $AC\\subset$ 面 $PAC$\n∴ 面 $PAC\\perp$ 面 $PAB$\n【右栏】\n(2) ……\n∴ 面 $MEF$ 与面 $PAB$ 的距离为 2",
    "16": "(1) 证：∵ $DE=AD=2$ 且 $G$ 为 $AE$ 中点 ∴ $DG\\perp AE$ ……\n(2) $\\angle DBG$ 为 $BD$ 与平面 $ABCE$ 所成角 ……",
    "17": "(1) 证：设 $AE$ 与 $BC$ 不是异面直线 ……\n(2) ∵ $AD\\parallel BC$ ……$\\angle EAD=\\arccos\\frac{\\sqrt{2}}{4}$",
}


def _layer(user: str) -> str:
    """提示里 <<< >>> 之间的文字层，去掉公式乱码（私用区字符）。"""
    m = re.search(r"<<<\n(.*?)\n>>>", user, re.S)
    t = m.group(1) if m else ""
    t = re.sub(r"[\ue000-\uf8ff]", "", t)
    return t.strip()


def _err(msg: str, code: int = 400):
    return JSONResponse({"error": {"message": msg, "type": "invalid_request_error"}}, status_code=code)


def _text_of(msg) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    return "\n".join(p.get("text", "") for p in c if p.get("type") == "text")


def _check(body: dict):
    if not body.get("model"):
        return "model is required"
    if body.get("model") not in ("deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash"):
        return f"Model Not Exist: {body.get('model')}"
    msgs = body.get("messages") or []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if p.get("type") == "image_url":
                    if m.get("role") != "user":
                        return "images are only supported in user messages"
                    url = (p.get("image_url") or {}).get("url", "")
                    if not (url.startswith("data:image/") or url.startswith("https://")):
                        return "invalid image url"
                    if body.get("model") == "deepseek-v4-pro":
                        return "deepseek-v4-pro does not support image input"
                elif p.get("type") != "text":
                    return f"unknown content type {p.get('type')}"
    if body.get("response_format", {}).get("type") == "json_object":
        if not any("json" in _text_of(m).lower() for m in msgs):
            return "response_format json_object requires the word json in the prompt"
    th = body.get("thinking")
    if th is not None and th.get("type") not in ("enabled", "disabled"):
        return "invalid thinking type"
    if body.get("reasoning_effort") not in (None, "low", "high", "max"):
        return "invalid reasoning_effort"
    return None


def _respond(content: str, thinking: bool):
    msg = {"role": "assistant", "content": content}
    if thinking:
        msg["reasoning_content"] = "（模拟的思考过程）"
    return {"id": "fake", "object": "chat.completion", "created": int(time.time()), "model": "deepseek-v4.1-flash",
            "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 400, "prompt_cache_hit_tokens": 800,
                      "prompt_cache_miss_tokens": 400, "completion_tokens_details": {"reasoning_tokens": 300 if thinking else 0}}}


@app.post("/chat/completions")
async def chat(req: Request):
    body = await req.json()
    if body.get("model") == "deepseek-v4-flash" and "thinking" in body:
        return _err("Unrecognized request argument supplied: thinking")   # 模拟不认识 thinking 参数的模型
    problem = _check(body)
    if problem:
        return _err(problem)
    msgs = body["messages"]
    system = _text_of(msgs[0]) if msgs and msgs[0]["role"] == "system" else ""
    user = "\n".join(_text_of(m) for m in msgs if m["role"] == "user")
    thinking = (body.get("thinking") or {}).get("type", "enabled") == "enabled"
    with _lock:
        STATE["calls"] += 1

    if "版面" in system:
        task = "layout"
        ids = re.findall(r"R\d+", user.split("编号写在每个框的右下角：")[-1].split("。")[0])
        sol = re.findall(r"第(\d+)题（解答题", user)
        regions = []
        n_sol = len(sol)
        for k, rid in enumerate(ids):
            if k >= len(ids) - n_sol and n_sol:
                num = sol[k - (len(ids) - n_sol)]
                regions.append({"id": rid, "kind": "解答题", "questions": [num], "full_score": 12, "header": f"{num}(12分)"})
            elif k == 0:
                regions.append({"id": rid, "kind": "填空题", "questions": [str(i) for i in range(1, 11)], "full_score": None, "header": "一、填空题"})
            else:
                regions.append({"id": rid, "kind": "选择题", "questions": ["11", "12", "13", "14"], "full_score": None, "header": "二、选择题"})
        out = json.dumps({"regions": regions, "unboxed": ""}, ensure_ascii=False)
    elif "试卷录入员" in system:
        m = re.search(r"试卷第(\d+)题（已经按题号位置单独裁出来", user)
        if m:   # 逐题录入：模拟“看图录入”——用提示里给的文字层汉字拼出题干（真模型会看图写出公式）
            task = "paper_q"
            num = m.group(1)
            layer = _layer(user)
            stem = re.sub(r"^\s*\d+\s*[\.．]\s*", "", layer)
            parts = re.findall(r"[（(]([1-9])[)）]", stem)
            out = json.dumps({"type": "解答题", "score": None, "stem": stem,
                              "parts": [{"label": f"({k})", "score": None} for k in dict.fromkeys(parts)],
                              "figure": "三角形 ABD，C 在 AB 上，DC⊥AB" if "如图" in stem else ""}, ensure_ascii=False)
        else:   # 整份一次录入（扫描件）
            task = "paper"
            out = json.dumps({"questions": SAMPLE_QUESTIONS, "section_notes": ["三、解答题：共3小题，36分"]}, ensure_ascii=False)
            out = out.replace("\\\\", "\\")   # 模拟模型漏转义的 LaTeX
    elif "参考答案（含评分标准）" in system and "部分的截图" in user:
        task = "answer_obj"
        nums = re.search(r"需要的题号：([\d、]+)", user).group(1).split("、")
        out = json.dumps({"answers": {n: KEY_OBJ[n] for n in nums if n in KEY_OBJ}}, ensure_ascii=False)
    elif "参考答案（含评分标准）" in system:
        m = re.search(r"参考答案中第(\d+)题的部分", user)
        if m:
            task = "answer_q"
            num = m.group(1)
            ans = re.sub(r"^\s*\d+\s*[\.．]\s*", "", _layer(user))
            if num == "18":   # 模拟模型多写了别的题的内容（上一版逐页录入时出现过的情况）
                ans += "\n解：（1）因为 $f(x)=e^x-ax$，所以 $f'(x)=e^x-a$。①当 $a\\le0$ 时，$f'(x)>0$ 恒成立，所以 $f(x)$ 单调递增。（2）由（1）知最小值为 $a-a\\ln a$。"
            out = json.dumps({"answer": ans}, ensure_ascii=False)
        else:
            task = "answers_doc"
            out = json.dumps({"items": []}, ensure_ascii=False)
    elif "阅卷组长" in system:
        task = "gen"
        num = re.search(r"请为第(\d+)题", user).group(1)
        parts = re.findall(r"(\(\d\)) (\?|\d+)分", user)
        m = re.search(r"满分 (\d+) 分", user)
        total = int(m.group(1)) if m else 12          # 平时作业没有分值（“满分 ? 分”）
        if parts and all(p[1] != "?" for p in parts):
            ps = [{"label": l, "score": int(s)} for l, s in parts]
        else:
            ps = [{"label": "(1)", "score": total // 2 - 1}, {"label": "(2)", "score": total - total // 2 + 1}]
        out = json.dumps({"parts": ps, "answer": f"第{num}题参考答案：（1）证明：……$\\frac{{1}}{{2}}$ ……（2）解：……",
                          "rubric": "（1）……（2分）；……\n（2）……", "issues": ""}, ensure_ascii=False)
    elif "录入员" in system and "手写" in system and "填空题" not in system:
        task = "transcribe"
        num = re.search(r"这是第(\w+)题", user).group(1)
        out = TRANSCRIPTS.get(num, "（学生作答）……") + "\n[[清晰度:中]]"
    elif "阅卷老师" in system and "填空题" not in system:
        task = "grade"
        num = re.search(r"【题目】第(\w+)题", user).group(1)
        key = ("grade", num)
        with _lock:
            first = key not in STATE["fail_once"]
            STATE["fail_once"].add(key)
        if first and num == "16":
            return _err("Server overloaded", 503)   # 模拟一次服务繁忙，测试重试
        labels = re.findall(r"(\(\d\)|\(全题\)) (\d+(?:\.\d+)?)分", user.split("【小问及分值】")[1].split("\n")[0])
        parts = []
        for k, (lab, mx) in enumerate(labels):
            mx = float(mx)
            sc = mx if k == 0 else mx - 1
            if num == "17" and k == 1:
                sc = mx + 3      # 故意超出满分，测试截断
            parts.append({"label": lab, "max": mx, "score": sc,
                          "points": [{"item": "关键步骤", "max": mx, "got": sc, "reason": "学生写了 PA⊥AC"}],
                          "comment": "步骤基本完整" if sc >= mx else "缺少关键依据，扣1分"})
        total = sum(p["score"] for p in parts)
        out = json.dumps({"parts": parts, "total": total, "comment": "整体思路正确，注意书写规范。",
                          "confidence": "high", "needs_review": False, "review_notes": ""}, ensure_ascii=False)
    elif "填空题横线上手写" in system:
        task = "fill_read"
        m = re.search(r"题号：([\d、]+)", user)
        nums = m.group(1).split("、") if m else []
        STATE.setdefault("fill_asked", []).append(nums)
        out = json.dumps({"answers": {n: STUDENT_FILL.get(n, "") for n in nums}, "unclear": []}, ensure_ascii=False)
    elif "填空题答案的对错" in system:
        task = "fill_judge"
        rows = json.loads(user[user.index("["):user.index("]\n\n判分规则") + 1])
        res = {}
        for r in rows:
            n = r["题号"]
            res[n] = {"correct": n in JUDGE_OK, "uncertain": False,
                      "reason": "与参考答案等价" if n in JUDGE_OK else "与参考答案不一致"}
        out = json.dumps({"results": res}, ensure_ascii=False)
    elif "选择题的涂卡结果" in system:
        task = "choice_read"
        out = json.dumps({"answers": {}, "unclear": []}, ensure_ascii=False)
    elif "做标准答案" in system:
        # 模拟“模型做选择/填空题”：答案取 STATE["solve_key"]（测试设置）或 KEY_OBJ；
        # STATE["solve_disagree"] 里的题号，第 2 遍故意做出不同的答案
        task = "obj_solve"
        num = re.search(r"请做第(\w+)题", user).group(1)
        attempt = int(re.search(r"独立解答 #(\d+)", user).group(1))
        choice = "这是一道选择题" in user
        ans = (STATE.get("solve_key") or {}).get(num) or KEY_OBJ.get(num) or ""
        if choice and not re.fullmatch(r"[A-H]+", ans):        # KEY_OBJ 是另一份卷子的，题型可能对不上
            ans = "ABCD"[int(num) % 4] if num.isdigit() else "A"
        elif not ans:
            ans = "1"
        if attempt == 2 and num in (STATE.get("solve_disagree") or set()):
            ans = ("C" if ans != "C" else "D") if choice else ans + "+1"
        out = json.dumps({"answer": ans, "brief": f"第{num}题：由条件得……", "confidence": "high"}, ensure_ascii=False)
    elif "批改学生的平时作业" in system:
        task = "homework"     # 模拟“作业判对错”：STATE["hw_verdict"] 按题号给，默认对
        num = re.search(r"【题目】第(\w+)题", user).group(1)
        v = (STATE.get("hw_verdict") or {}).get(num, "对")
        m = re.search(r"label 用 (.+?)）；没有小问", user)
        labels = [l for l in (m.group(1).split("、") if m else []) if l and l != "(全题)"]
        parts = [{"label": l, "verdict": ("对" if k == 0 else "错") if v == "半对" else v, "comment": ""}
                 for k, l in enumerate(labels)] or [{"label": "(全题)", "verdict": v, "comment": ""}]
        out = json.dumps({"verdict": v, "parts": parts, "error": "" if v == "对" else f"第{num}题第二步计算错误",
                          "comment": "做得不错" if v == "对" else "注意检查计算", "confidence": "high",
                          "needs_review": False, "review_notes": ""}, ensure_ascii=False)
    elif "考生信息" in system:
        task = "student"       # 测试可以设 STATE["student_reply"] 模拟“模型认出的手写姓名”
        out = json.dumps(STATE.get("student_reply") or {"name": "文华", "student_id": "25090315", "class": "高二3班"},
                         ensure_ascii=False)
    else:
        return _err("unknown task in fake server")
    with _lock:
        STATE["by_task"][task] = STATE["by_task"].get(task, 0) + 1
    return _respond(out, thinking)


@app.get("/stats")
def stats():
    return {"calls": STATE["calls"], "by_task": STATE["by_task"]}


@app.post("/state")
async def set_state(req: Request):
    """给单独进程跑的模拟服务设测试数据（如 solve_disagree、hw_verdict、student_reply）。"""
    body = await req.json()
    with _lock:
        for k, v in body.items():
            STATE[k] = set(v) if k == "solve_disagree" else v
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8765, log_level="warning")
