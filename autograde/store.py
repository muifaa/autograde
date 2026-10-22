"""本地存储：每场考试一个文件夹，配置都在 exam.json 里；全局设置在 data/settings.json。"""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

DEFAULT_SETTINGS = {
    "api_key": "",                      # 留空则读取环境变量 DEEPSEEK_API_KEY
    "base_url": "https://api.deepseek.com",
    "vision_model": "deepseek-flash",   # 看图：录题、版面标注、手写转写（需要支持图片）
    "grading_model": "deepseek-flash",  # 给分；可换成 deepseek-v4-pro（不支持图片，只看转写文本）
    "grading_effort": "high",           # 思考强度 low / high / max
    "grading_attach_image": True,       # 给分时同时附原图，便于模型核对转写
    "answer_model": "deepseek-flash",   # 生成参考答案（附试卷图片，便于看到配图）
    "answer_effort": "max",
    "transcribe_thinking": False,       # 转写时是否开启思考（关闭更“照抄”）
    "samples": 1,                       # 每道题独立批改几次；2 = 双评
    "review_gap": 2,                    # 双评分差超过多少分就标记复核
    "score_step": 1,                    # 给分最小单位：1 或 0.5
    "fill_strict": False,               # 填空题必须是最简形式才算对
    "multi_partial": "proportional",    # 多选题没选全：proportional 按比例 / none 不给分 / 数字 固定给几分
    "judge_effort": "high",             # 填空题判等价时的思考强度
    "identify_student": True,           # 让模型读出姓名/学号
    "max_workers": 16,                  # 并发请求数（DeepSeek 每个账号 deepseek-flash 可以 2500 个并发）
    "scan_dpi": 250,                    # PDF 扫描件渲染分辨率
    "crop_max_side": 2000,              # 裁切图最长边（像素）
}

VISION_FREE_MODELS = {"deepseek-chat", "deepseek-reasoner"}


def model_supports_images(name: str) -> bool:
    n = (name or "").lower()
    return not ("pro" in n or n in VISION_FREE_MODELS)


class Store:
    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or os.environ.get("AUTOGRADE_DATA") or Path.cwd() / "data")
        (self.root / "exams").mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.RLock] = {}
        self._glock = threading.Lock()

    # ------------------------------------------------------------ 设置
    @property
    def settings_path(self) -> Path:
        return self.root / "settings.json"

    def settings(self) -> dict:
        s = copy.deepcopy(DEFAULT_SETTINGS)
        if self.settings_path.exists():
            try:
                s.update(json.loads(self.settings_path.read_text(encoding="utf-8")))
            except Exception:
                pass
        if not s.get("api_key"):
            s["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")
        return s

    def save_settings(self, new: dict) -> dict:
        cur = self.settings()
        for k, v in new.items():
            if k in DEFAULT_SETTINGS:
                cur[k] = v
        _atomic_write(self.settings_path, cur)
        return cur

    # ------------------------------------------------------------ 考试
    def lock(self, exam_id: str) -> threading.RLock:
        with self._glock:
            return self._locks.setdefault(exam_id, threading.RLock())

    def exam_dir(self, exam_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", exam_id or ""):
            raise KeyError(exam_id)
        return self.root / "exams" / exam_id

    def create_exam(self, title: str, pages_per_sheet: int = 1, kind: str = "exam") -> dict:
        exam_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        exam = {
            "id": exam_id, "title": title or "未命名考试", "created_at": _now(),
            "kind": kind if kind in ("exam", "quiz", "homework") else "exam",
            "pages_per_sheet": max(1, int(pages_per_sheet or 1)),
            "files": {"paper": [], "sheet": [], "answers": [], "scans": []},
            "questions": [], "template": None, "sheets": [],
            "status": {"stage": "created", "message": ""},
            "warnings": [], "usage": {},
        }
        d = self.exam_dir(exam_id)
        for sub in ("paper", "sheet", "answers", "scans", "template", "work", "cache"):
            (d / sub).mkdir(parents=True, exist_ok=True)
        self.save_exam(exam)
        return exam

    def list_exams(self) -> List[dict]:
        out = []
        for p in sorted((self.root / "exams").glob("*/exam.json"), reverse=True):
            try:
                e = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append({"id": e["id"], "title": e.get("title"), "created_at": e.get("created_at"), "kind": e.get("kind") or "exam",
                        "n_questions": len([q for q in e.get("questions", []) if q.get("type") == "解答题"]),
                        "n_sheets": len(e.get("sheets", [])), "status": e.get("status", {})})
        return out

    def load_exam(self, exam_id: str) -> dict:
        p = self.exam_dir(exam_id) / "exam.json"
        if not p.exists():
            raise KeyError(exam_id)
        with self.lock(exam_id):
            return json.loads(p.read_text(encoding="utf-8"))

    def save_exam(self, exam: dict) -> None:
        with self.lock(exam["id"]):
            _atomic_write(self.exam_dir(exam["id"]) / "exam.json", exam)

    def update_exam(self, exam_id: str, fn: Callable[[dict], None]) -> dict:
        """读-改-写，加锁，防止后台任务和网页编辑互相覆盖。"""
        with self.lock(exam_id):
            exam = self.load_exam(exam_id)
            fn(exam)
            self.save_exam(exam)
            return exam

    def delete_exam(self, exam_id: str) -> None:
        import shutil
        shutil.rmtree(self.exam_dir(exam_id), ignore_errors=True)

    # ------------------------------------------------------------ 每份答卷的结果
    def sheet_path(self, exam_id: str, sheet_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", sheet_id or ""):
            raise KeyError(sheet_id)
        return self.exam_dir(exam_id) / "work" / sheet_id / "result.json"

    def load_sheet(self, exam_id: str, sheet_id: str) -> dict:
        p = self.sheet_path(exam_id, sheet_id)
        if not p.exists():
            raise KeyError(sheet_id)
        with self.lock(f"{exam_id}/{sheet_id}"):
            return json.loads(p.read_text(encoding="utf-8"))

    def save_sheet(self, exam_id: str, sheet: dict) -> None:
        with self.lock(f"{exam_id}/{sheet['id']}"):
            _atomic_write(self.sheet_path(exam_id, sheet["id"]), sheet)

    def update_sheet(self, exam_id: str, sheet_id: str, fn: Callable[[dict], None]) -> dict:
        with self.lock(f"{exam_id}/{sheet_id}"):
            s = self.load_sheet(exam_id, sheet_id)
            fn(s)
            self.save_sheet(exam_id, s)
            return s

    def list_sheets(self, exam_id: str) -> List[dict]:
        exam = self.load_exam(exam_id)
        out = []
        for sid in exam.get("sheets", []):
            try:
                out.append(self.load_sheet(exam_id, sid))
            except KeyError:
                continue
        return out


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _atomic_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def safe_filename(name: str) -> str:
    base = os.path.basename(name or "file")
    base = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", base).strip(". ") or "file"
    return base[:120]
