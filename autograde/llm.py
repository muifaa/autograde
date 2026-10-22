"""DeepSeek API 客户端（OpenAI 兼容的 /chat/completions），带重试、JSON 解析、本地缓存和用量统计。

参考官方文档（2026-10）：
  - base_url: https://api.deepseek.com
  - deepseek-flash（V4.1 Flash）支持图片输入；deepseek-v4-pro 不支持图片
  - 思考模式：{"thinking": {"type": "enabled"|"disabled"}}，"reasoning_effort": "low"|"high"|"max"
  - JSON 输出：response_format={"type": "json_object"}，提示词里要出现 json 字样并给出示例
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests


class LLMError(Exception):
    pass


class Usage:
    """线程安全的用量累计。"""

    def __init__(self, data: Optional[dict] = None):
        self._lock = threading.Lock()
        self.data = {"calls": 0, "cached_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                     "reasoning_tokens": 0, "cache_hit_tokens": 0}
        if data:
            self.data.update({k: v for k, v in data.items() if k in self.data})

    def add(self, usage: Optional[dict], cached: bool) -> None:
        with self._lock:
            self.data["calls"] += 1
            if cached:
                self.data["cached_calls"] += 1
                return
            u = usage or {}
            self.data["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            self.data["completion_tokens"] += int(u.get("completion_tokens") or 0)
            self.data["cache_hit_tokens"] += int(u.get("prompt_cache_hit_tokens") or 0)
            det = u.get("completion_tokens_details") or {}
            self.data["reasoning_tokens"] += int(det.get("reasoning_tokens") or 0)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.data)


class DeepSeek:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 cache_dir: Optional[str] = None, usage: Optional[Usage] = None,
                 timeout: int = 900, max_retries: int = 4):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.usage = usage or Usage()
        self.timeout = timeout
        self.max_retries = max_retries
        self._local = threading.local()

    # 每个线程一个 Session
    @property
    def session(self) -> requests.Session:
        s = getattr(self._local, "s", None)
        if s is None:
            s = requests.Session()
            self._local.s = s
        return s

    # ------------------------------------------------------------ 缓存
    def _cache_path(self, body: dict) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, body: dict) -> Optional[dict]:
        p = self._cache_path(body)
        if p and p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _cache_put(self, body: dict, result: dict) -> None:
        p = self._cache_path(body)
        if not p:
            return
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    # ------------------------------------------------------------ 请求
    def chat(self, messages: List[dict], model: str, *, json_mode: bool = False,
             thinking: Optional[bool] = None, effort: Optional[str] = None,
             max_tokens: Optional[int] = None, temperature: Optional[float] = None,
             use_cache: bool = True) -> dict:
        """返回 {"content", "reasoning", "usage", "cached", "finish_reason"}。"""
        if not self.api_key:
            raise LLMError("没有配置 DeepSeek API Key（环境变量 DEEPSEEK_API_KEY，或在网页“设置”里填写）")
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if thinking is not None:
            body["thinking"] = {"type": "enabled" if thinking else "disabled"}
        if thinking and effort:
            body["reasoning_effort"] = effort
        if not thinking and temperature is not None:
            body["temperature"] = temperature

        if use_cache:
            hit = self._cache_get(body)
            if hit is not None:
                self.usage.add(None, cached=True)
                hit["cached"] = True
                return hit

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        send = dict(body)
        last_err = ""
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(60, (2 ** attempt) + random.random() * 2))
            try:
                r = self.session.post(url, json=send, headers=headers, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError) as e:
                last_err = f"网络错误：{e}"
                continue
            if r.status_code == 200:
                try:
                    data = r.json()
                    choice = data["choices"][0]
                    msg = choice.get("message") or {}
                except Exception:
                    last_err = f"返回内容无法解析：{r.text[:300]}"
                    continue
                content = msg.get("content") or ""
                if not content.strip():
                    last_err = "模型返回了空内容"
                    continue
                result = {"content": content, "reasoning": msg.get("reasoning_content"),
                          "usage": data.get("usage"), "finish_reason": choice.get("finish_reason"),
                          "model": data.get("model"), "cached": False}
                self.usage.add(result["usage"], cached=False)
                if result["finish_reason"] != "length":
                    self._cache_put(body, result)
                return result
            text = r.text[:500]
            if r.status_code == 401:
                raise LLMError("DeepSeek API Key 无效（401）")
            if r.status_code == 402:
                raise LLMError("DeepSeek 账户余额不足（402）")
            if r.status_code in (400, 422):
                # 某些参数当前模型不支持：去掉后重试一次
                dropped = False
                low = text.lower()
                for k in ("response_format", "thinking", "reasoning_effort", "temperature"):
                    if k in send and (k in low or k.replace("_", " ") in low):
                        send.pop(k)
                        dropped = True
                if dropped:
                    continue
                raise LLMError(f"请求被拒绝（{r.status_code}）：{text}")
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"服务繁忙（{r.status_code}）"
                continue
            raise LLMError(f"请求失败（{r.status_code}）：{text}")
        raise LLMError(f"多次重试仍失败：{last_err}")

    def chat_json(self, messages: List[dict], model: str, **kw) -> Tuple[Any, dict]:
        """要求 JSON 输出并解析；解析失败会追加一轮提示让模型重写一次。"""
        res = self.chat(messages, model, json_mode=True, **kw)
        obj = parse_json(res["content"])
        if obj is not None:
            return obj, res
        retry = messages + [{"role": "assistant", "content": res["content"]},
                            {"role": "user", "content": "上面的输出不是合法的 JSON。请只输出合法的 json 对象，不要任何其他文字。"}]
        res2 = self.chat(retry, model, json_mode=True, **kw)
        obj = parse_json(res2["content"])
        if obj is None:
            raise LLMError("模型输出无法解析为 JSON：" + res2["content"][:300])
        return obj, res2


# ---------------------------------------------------------------- 消息构造

def image_part(data_url: str, detail: str = "high") -> dict:
    return {"type": "image_url", "image_url": {"url": data_url, "detail": detail}}


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


# ---------------------------------------------------------------- JSON 解析（含 LaTeX 反斜杠修复）

_LATEX_BFNRT = {
    "b": "beta bar because bot boxed binom bigcup bigcap big Big bigg bf bm bmod backslash bullet breve bigl bigr",
    "f": "frac forall flat frown fbox",
    "n": "neq ne nabla not notin nu newline nleq ngeq nmid nparallel nsubseteq neg nolimits nleqslant ngeqslant",
    "r": "right rho rangle rfloor rceil rm rightarrow rightleftharpoons rVert rvert",
    "t": "theta times tan triangle text textbf tfrac to top tau therefore tilde textit textrm textstyle "
         "triangleq tanh tbinom textcircled",
}
_LATEX_SET = {w for v in _LATEX_BFNRT.values() for w in v.split()}
_HEX = set("0123456789abcdefABCDEF")


def _fix_backslashes(s: str) -> str:
    """模型在 JSON 里写 LaTeX 时常漏掉转义：把明显是 LaTeX 命令的反斜杠补成双反斜杠。"""
    out = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        nxt = s[i + 1] if i + 1 < n else ""
        if nxt in ('"', "\\", "/"):
            out.append(c + nxt)
            i += 2
            continue
        if nxt.isalpha():
            j = i + 1
            while j < n and s[j].isalpha() and s[j].isascii():
                j += 1
            word = s[i + 1:j]
            if nxt in "bfnrt":
                out.append("\\\\" if word in _LATEX_SET else "\\")
            elif nxt == "u" and len(s[i + 2:i + 6]) == 4 and all(ch in _HEX for ch in s[i + 2:i + 6]):
                out.append("\\")
            else:
                out.append("\\\\")
            i += 1
            continue
        out.append("\\\\")   # \{ \, \; 之类，JSON 里不合法
        i += 1
    return "".join(out)


def parse_json(text: str) -> Any:
    if text is None:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    start = min([p for p in (t.find("{"), t.find("[")) if p >= 0], default=-1)
    if start < 0:
        return None
    end = max(t.rfind("}"), t.rfind("]"))
    if end <= start:
        return None
    t = t[start:end + 1]
    for cand in (_fix_backslashes(t), t):
        try:
            return json.loads(cand, strict=False)
        except Exception:
            continue
    return None
