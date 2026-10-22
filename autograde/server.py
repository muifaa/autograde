"""网页平台（FastAPI）。也可以作为后端服务，被你自己的平台通过 HTTP 调用。

启动：python run.py   然后浏览器打开 http://127.0.0.1:8000
"""
import os
import re
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import exam_setup, pipeline, report, roster, sheetgen
from .exam_setup import _num, norm_label, norm_number, norm_type
from .imaging import SUPPORTED_EXTS
from .llm import DeepSeek, LLMError, Usage
from .store import DEFAULT_SETTINGS, Store, safe_filename

STATIC = Path(__file__).resolve().parent.parent / "static"
store = Store()
app = FastAPI(title="阅卷助手", version="0.1")


# ================================================================ 后台任务

class Jobs:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}
        self.active: Dict[str, str] = {}   # exam_id -> job_id
        self.lock = threading.Lock()

    def start(self, exam_id: str, kind: str, label: str, fn: Callable[[Callable[[float, str], None]], Any]) -> dict:
        with self.lock:
            jid = self.active.get(exam_id)
            if jid and self.jobs[jid]["status"] == "running":
                raise HTTPException(409, f"这场考试正在{self.jobs[jid]['label']}，完成后再试")
            jid = uuid.uuid4().hex[:10]
            job = {"id": jid, "exam_id": exam_id, "kind": kind, "label": label, "status": "running",
                   "progress": 0.0, "message": "开始…", "error": "", "result": None,
                   "started": time.time(), "finished": None}
            self.jobs[jid] = job
            self.active[exam_id] = jid

        def progress(f: float, msg: str):
            job["progress"] = max(job["progress"], min(1.0, float(f)))
            job["message"] = msg

        def run():
            try:
                job["result"] = fn(progress)
                job["status"] = "done"
                job["progress"] = 1.0
                job["message"] = "完成"
            except Exception as e:  # noqa
                job["status"] = "error"
                job["error"] = str(e) if isinstance(e, (LLMError, ValueError, KeyError)) else f"{type(e).__name__}: {e}"
                job["message"] = "出错了"
                traceback.print_exc()
            finally:
                job["finished"] = time.time()
                try:
                    store.update_exam(exam_id, lambda ex: ex.__setitem__("last_job", {
                        k: job[k] for k in ("id", "kind", "label", "status", "message", "error", "finished")}))
                except Exception:
                    pass

        threading.Thread(target=run, daemon=True).start()
        return job

    def get(self, jid: str) -> dict:
        if jid not in self.jobs:
            raise HTTPException(404, "任务不存在")
        return self.jobs[jid]

    def for_exam(self, exam_id: str) -> Optional[dict]:
        jid = self.active.get(exam_id)
        return self.jobs.get(jid) if jid else None


jobs = Jobs()


def _llm(exam_id: str, require_key: bool = True) -> Optional[DeepSeek]:
    st = store.settings()
    if not st.get("api_key"):
        if require_key:
            raise HTTPException(400, "还没有填写 DeepSeek API Key，请先到“设置”里填写")
        return None
    exam = store.load_exam(exam_id)
    return DeepSeek(st["api_key"], st["base_url"], cache_dir=str(store.exam_dir(exam_id) / "cache"),
                    usage=Usage(exam.get("usage")))


def _save_usage(exam_id: str, llm: Optional[DeepSeek]):
    if llm is not None:
        store.update_exam(exam_id, lambda e: e.__setitem__("usage", llm.usage.snapshot()))


def _exam_or_404(exam_id: str) -> dict:
    try:
        return store.load_exam(exam_id)
    except KeyError:
        raise HTTPException(404, "考试不存在")


async def _save_uploads(exam_id: str, sub: str, files: List[UploadFile], allow_text: bool = False) -> List[str]:
    d = store.exam_dir(exam_id) / sub
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files or []:
        if not f or not f.filename:
            continue
        name = safe_filename(f.filename)
        ext = Path(name).suffix.lower()
        if ext in (".heic", ".heif") and ext not in SUPPORTED_EXTS:
            raise HTTPException(400, f"{name} 是 iPhone 的 HEIC 照片，要先装 pillow-heif 才能读（pip install pillow-heif），"
                                     "或者导出成 JPG 再上传")
        if ext not in SUPPORTED_EXTS and not (allow_text and ext in (".txt", ".md")):
            raise HTTPException(400, f"不支持的文件类型：{name}（支持 PDF、JPG、PNG、TIFF{'、HEIC' if '.heic' in SUPPORTED_EXTS else ''}"
                                     f"{'、TXT' if allow_text else ''}）")
        target = d / name
        k = 1
        while target.exists():
            target = d / f"{Path(name).stem}_{k}{ext}"
            k += 1
        with open(target, "wb") as fh:
            while True:
                chunk = await f.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        out.append(f"{sub}/{target.name}")
    return out


def _exam_view(exam: dict) -> dict:
    job = jobs.for_exam(exam["id"])
    exam = dict(exam)
    exam["job"] = job
    exam["graded_numbers"] = report.graded_numbers(exam)
    st = store.settings()
    exam["has_key"] = bool(st.get("api_key"))
    return exam


# ================================================================ 页面与设置

@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/api/settings")
def get_settings():
    s = store.settings()
    key = s.get("api_key") or ""
    s["api_key"] = ("********" + key[-4:]) if key else ""
    saved = {}
    if store.settings_path.exists():
        import json
        try:
            saved = json.loads(store.settings_path.read_text(encoding="utf-8"))
        except Exception:
            saved = {}
    s["api_key_source"] = "settings" if saved.get("api_key") else ("env" if os.environ.get("DEEPSEEK_API_KEY") else "")
    s["defaults"] = DEFAULT_SETTINGS
    return s


@app.put("/api/settings")
def put_settings(body: dict = Body(...)):
    body = {k: v for k, v in body.items() if k in DEFAULT_SETTINGS}
    if str(body.get("api_key", "")).startswith("********"):
        body.pop("api_key")
    for k in ("samples", "max_workers", "scan_dpi", "crop_max_side"):
        if k in body:
            body[k] = int(body[k])
    if "fill_strict" in body:
        body["fill_strict"] = bool(body["fill_strict"]) and str(body["fill_strict"]).lower() not in ("false", "0", "")
    for k in ("review_gap", "score_step"):
        if k in body:
            body[k] = float(body[k])
    store.save_settings(body)
    return get_settings()


# ================================================================ 考试

@app.get("/api/exams")
def list_exams():
    return store.list_exams()


@app.post("/api/exams")
async def create_exam(title: str = Form(""), pages_per_sheet: int = Form(1), generate_answers: bool = Form(True),
                      paper: List[UploadFile] = File(default=[]), sheet: List[UploadFile] = File(default=[]),
                      answers: List[UploadFile] = File(default=[]), question_text: str = Form(""),
                      structure: str = Form(""), kind: str = Form("exam")):
    parsed: List[dict] = []
    if structure.strip():
        parsed, bad = exam_setup.parse_structure(structure)
        if bad:
            raise HTTPException(400, "题目结构这几行看不懂：" + "；".join(bad[:5]) + "。每行写成“1-8 单选 5分”这样")
    exam = store.create_exam(title.strip() or "未命名考试", pages_per_sheet, kind)
    eid = exam["id"]
    if parsed:
        def put_qs(e):
            e["questions"] = exam_setup.merge_structure([], parsed)
            e["warnings"] = exam_setup.reconcile(e)
        store.update_exam(eid, put_qs)
    pf = await _save_uploads(eid, "paper", paper, allow_text=True)
    if question_text.strip():
        p = store.exam_dir(eid) / "paper" / "pasted_questions.txt"
        p.write_text(question_text, encoding="utf-8")
        pf.append("paper/pasted_questions.txt")
    sf = await _save_uploads(eid, "sheet", sheet)
    af = await _save_uploads(eid, "answers", answers, allow_text=True)
    store.update_exam(eid, lambda e: e["files"].update(paper=pf, sheet=sf, answers=af))
    job = _start_setup(eid, generate_answers)
    return {"exam": _exam_view(store.load_exam(eid)), "job": job}


def _start_setup(eid: str, generate_answers: bool) -> Optional[dict]:
    exam = store.load_exam(eid)
    if not any(exam["files"].get(k) for k in ("paper", "sheet", "answers")):
        return None
    llm = _llm(eid, require_key=False)
    st = store.settings()

    def run(progress):
        try:
            exam_setup.run_setup(store, eid, llm, st, progress, generate_missing=generate_answers)
        finally:
            _save_usage(eid, llm)
        if llm is None:
            return {"note": "未配置 API Key：没有录入题目，版面按规则粗略标注，请手动检查"}
        return {}
    return jobs.start(eid, "setup", "录题和识别版面", run)


@app.post("/api/exams/{eid}/setup")
async def add_setup_files(eid: str, generate_answers: bool = Form(True), paper: List[UploadFile] = File(default=[]),
                          sheet: List[UploadFile] = File(default=[]), answers: List[UploadFile] = File(default=[])):
    _exam_or_404(eid)
    pf = await _save_uploads(eid, "paper", paper, allow_text=True)
    sf = await _save_uploads(eid, "sheet", sheet)
    af = await _save_uploads(eid, "answers", answers, allow_text=True)

    def apply(e):
        if pf:
            e["files"]["paper"] = pf
        if sf:
            e["files"]["sheet"] = sf
        if af:
            e["files"]["answers"] = af
    store.update_exam(eid, apply)
    return {"job": _start_setup(eid, generate_answers)}


@app.get("/api/exams/{eid}")
def get_exam(eid: str):
    return _exam_view(_exam_or_404(eid))


@app.delete("/api/exams/{eid}")
def delete_exam(eid: str):
    _exam_or_404(eid)
    job = jobs.for_exam(eid)
    if job and job["status"] == "running":
        raise HTTPException(409, "正在处理中，不能删除")
    store.delete_exam(eid)
    return {"ok": True}


@app.put("/api/exams/{eid}/kind")
def set_kind(eid: str, body: dict = Body(...)):
    """考试 / 周测 / 平时作业。平时作业不打分，只判对错。
    已经批过答卷、又在“打分”和“不打分”之间切换时：老师改过的分清掉（尺度不一样了），按新的方式重新批一遍。"""
    exam = _exam_or_404(eid)
    _no_running_job(eid, "改类型")
    kind = str(body.get("kind") or "")
    if kind not in exam_setup.EXAM_KINDS:
        raise HTTPException(400, "类型只能是考试、周测或平时作业")
    flip = exam_setup.is_scored(exam) != exam_setup.is_scored(dict(exam, kind=kind))
    sids = list(exam.get("sheets") or []) if flip else []
    llm = _llm(eid) if sids else None            # 要重新批：先确认填了 API Key，再改

    def apply(e):
        e["kind"] = kind
        e["warnings"] = exam_setup.reconcile(e)
    store.update_exam(eid, apply)
    if sids:
        def reset(s):
            for qr in (s.get("questions") or {}).values():
                qr["teacher"] = {}
        for sid in sids:
            try:
                store.update_sheet(eid, sid, reset)
            except KeyError:
                pass
        st = store.settings()

        def run(progress):
            try:
                return pipeline.regrade(store, eid, llm, st, sids, None, False, False, progress)
            finally:
                _save_usage(eid, llm)
        jobs.start(eid, "regrade", f"改成{exam_setup.EXAM_KINDS[kind]}，重新批改", run)
    return _exam_view(store.load_exam(eid))


@app.post("/api/exams/{eid}/objective/solve")
def solve_objective(eid: str, body: dict = Body(default={})):
    """让 AI 做没有答案的选择题、填空题（每题独立解两遍，不一致的标出来）。redo=true 时 AI 做过、老师还没确认的也重做。"""
    _exam_or_404(eid)
    llm = _llm(eid)
    st = store.settings()
    d = store.exam_dir(eid)
    redo = bool(body.get("redo"))

    def run(progress):
        exam = store.load_exam(eid)
        qs = [dict(q) for q in exam["questions"]]
        try:
            log = exam_setup.generate_objective_answers(llm, st, d, qs, redo, progress)
        finally:
            _save_usage(eid, llm)
        done = {q["number"]: q for q in qs if q.get("answer_source") == "ai" and q.get("ai_solve")}

        def apply(e):
            for i, q in enumerate(e["questions"]):     # 只写回 AI 刚做的题（期间老师改过的不覆盖）
                n = q["number"]
                if n in done and (not (q.get("answer") or "").strip() or (q.get("answer_source") == "ai" and not q.get("confirmed"))):
                    e["questions"][i] = dict(q, **{k: done[n][k] for k in ("answer", "answer_source", "confirmed", "ai_solve", "checks")})
            e["warnings"] = exam_setup.reconcile(e)
        store.update_exam(eid, apply)
        return {"log": log}
    return {"job": jobs.start(eid, "solve", "AI 做选择、填空题", run)}


@app.put("/api/exams/{eid}/title")
def rename_exam(eid: str, body: dict = Body(...)):
    _exam_or_404(eid)
    store.update_exam(eid, lambda e: e.__setitem__("title", str(body.get("title") or "").strip() or e["title"]))
    return _exam_view(store.load_exam(eid))


def _clean_question(q: dict, old: Optional[dict]) -> dict:
    """网页提交的题目：只接受可编辑字段，截图、核对结果等由服务器端保留。"""
    out = exam_setup.new_question(norm_number(q.get("number")), norm_type(q.get("type")) or "解答题")
    if not out["number"]:
        raise HTTPException(400, "题号不能为空")
    out["score"] = _num(q.get("score"))
    for k in ("stem", "figure", "answer", "rubric", "notes", "issues"):
        out[k] = str(q.get(k) or "")
    out["parts"] = [{"label": norm_label(p.get("label"), k), "text": str(p.get("text") or ""), "score": _num(p.get("score"))}
                    for k, p in enumerate(q.get("parts") or []) if isinstance(p, dict)]
    exam_setup.fold_parts_into_stem(out)
    if out["type"] == "选择题":
        try:
            n_opt = int(q.get("options") or 0)
        except (TypeError, ValueError):
            n_opt = 0
        if 2 <= n_opt <= 8:
            out["options"] = n_opt
        if q.get("multi"):
            out["multi"] = True
    out["confirmed"] = bool(q.get("confirmed"))
    o = old or {}
    for k in ("paper_crops", "answer_crops", "score_convention", "section", "score_from"):
        if k in o:
            out[k] = o[k]
    if o.get("ai_solve") and out["answer"] == o.get("answer"):
        out["ai_solve"] = o["ai_solve"]          # AI 做选择/填空题时两遍的结果（答案没改就保留）
    checks = list(o.get("checks") or [])
    if out["stem"] != o.get("stem", ""):
        checks = [c for c in checks if c.get("where") != "题目"]     # 老师改过题干：之前的核对提示作废
    if out["answer"] != o.get("answer", ""):
        checks = [c for c in checks if c.get("where") != "答案"]
    if out["confirmed"] and not o.get("confirmed"):
        checks = []                                                   # 老师确认过了
    out["checks"] = checks
    if old:
        src = o.get("answer_source", "none")
        if out["answer"] and (out["answer"] != o.get("answer") or out["rubric"] != o.get("rubric")):
            src = "teacher"   # 老师改过就算老师的
    else:
        src = "teacher" if out["answer"] else "none"
    out["answer_source"] = src if out["answer"] else "none"
    return out


@app.put("/api/exams/{eid}/questions")
def put_questions(eid: str, body: dict = Body(...)):
    _exam_or_404(eid)
    incoming = body.get("questions") or []

    def apply(e):
        old = {q["number"]: q for q in e.get("questions") or []}
        qs = [_clean_question(q, old.get(norm_number(q.get("number")))) for q in incoming]
        nums = [q["number"] for q in qs]
        dup = {n for n in nums if nums.count(n) > 1}
        if dup:
            raise HTTPException(400, f"题号重复：{'、'.join(sorted(dup))}")
        e["questions"] = qs
        e["warnings"] = exam_setup.reconcile(e)
    return _exam_view(store.update_exam(eid, apply))


@app.post("/api/exams/{eid}/questions/quick")
def quick_questions(eid: str, body: dict = Body(...)):
    """按“1-8 单选 5分”这样的写法快速添加题目（没有试卷文件时用）。"""
    _exam_or_404(eid)
    parsed, bad = exam_setup.parse_structure(str(body.get("text") or ""))
    if not parsed:
        raise HTTPException(400, "没有看懂：每行写成“1-8 单选 5分”“9-12 填空 5分”“13 解答 12分 2问”这样")

    def apply(e):
        e["questions"] = exam_setup.merge_structure(e.get("questions") or [], parsed)
        e["warnings"] = exam_setup.reconcile(e)
    exam = store.update_exam(eid, apply)
    return {"exam": _exam_view(exam), "added": len(parsed), "bad": bad}


@app.post("/api/exams/{eid}/answer-sheet")
def make_answer_sheet(eid: str, body: dict = Body(default={})):
    """按题目生成答题纸，并直接设为版面。"""
    exam = _exam_or_404(eid)
    _no_running_job(eid, "生成答题纸")
    old = (exam.get("template") or {}).get("generated") or {}       # 没给的设置沿用上次生成时的
    paper = str(body.get("paper") or old.get("paper") or "A4")
    space = str(body.get("space") or old.get("space") or "normal")
    raw = body.get("id_digits")
    try:
        digits = int(raw if raw not in (None, "") else old.get("id_digits", 4))
    except (TypeError, ValueError):
        raise HTTPException(400, "学号位数要是数字")
    if paper not in sheetgen.PAPERS or space not in sheetgen.SPACES or not 0 <= digits <= 12:
        raise HTTPException(400, "参数不对")
    heights = {} if body.get("reset_heights") else dict((old.get("plan") or {}).get("heights") or {})
    for n, h in (body.get("heights") or {}).items():               # 拖动方框改的高度；null = 恢复自动
        if h in (None, "", 0):
            heights.pop(str(n), None)
        else:
            try:
                heights[str(n)] = max(sheetgen.MIN_BOX, min(800.0, float(h)))
            except (TypeError, ValueError):
                raise HTTPException(400, "方框高度要是数字")
    try:
        exam = exam_setup.generate_sheet(store, eid, paper, digits, space,
                                         str(body.get("title") or old.get("title") or exam["title"]),
                                         bool(body.get("notice", (old.get("plan") or {}).get("notice", True))), heights)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _exam_view(exam)


@app.get("/api/exams/{eid}/answer-sheet.pdf")
def answer_sheet_pdf(eid: str, classes: str = ""):
    """生成的答题纸 PDF。classes 为空：通用版（可以复印）；给了班级：按名单每人一份。"""
    from urllib.parse import quote
    exam = _exam_or_404(eid)
    gen = (exam.get("template") or {}).get("generated")
    if not gen:
        raise HTTPException(400, "这场考试的答题纸不是生成的")
    plan = gen["plan"]
    name = f"{safe_filename(exam['title'])}_答题纸"
    students = None
    if classes.strip():
        cids = [c for c in classes.split(",") if c]
        students = []
        cnames = []
        for cid in cids:
            try:
                c = roster.get_class(store.root, cid)
            except KeyError:
                raise HTTPException(404, "名单不存在")
            cnames.append(c["name"])
            students += [{"sid": s.get("sid", ""), "name": s.get("name", ""), "class": c["name"]} for s in c["students"]]
        if not students:
            raise HTTPException(400, "名单是空的")
        too_long = [s["name"] for s in students if len(re.sub(r"\D", "", s["sid"])) > int(plan["id_digits"])]
        if plan["id_digits"] and too_long:
            raise HTTPException(400, f"有 {len(too_long)} 人的学号比答题纸上的 {plan['id_digits']} 位长（如{too_long[0]}），"
                                     f"请重新生成答题纸时把学号位数调大")
        name += f"_{'_'.join(cnames)}_每人一份"
    pdf = sheetgen.render_pdf(plan, eid, students)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(safe_filename(name) + '.pdf')}"})


@app.post("/api/exams/{eid}/questions/{num}/generate")
def generate_one(eid: str, num: str):
    exam = _exam_or_404(eid)
    q = next((q for q in exam["questions"] if q["number"] == num), None)
    if q is None:
        raise HTTPException(404, "没有这道题")
    if not q.get("stem"):
        raise HTTPException(400, "请先填写题干")
    llm = _llm(eid)
    st = store.settings()
    d = store.exam_dir(eid)

    def run(progress):
        progress(0.1, f"正在为第 {num} 题解题并写评分细则（可能需要一两分钟）…")
        imgs = exam_setup._load_crops(d, q.get("paper_crops"))
        qq = dict(q)
        try:
            exam_setup.generate_answer(llm, st, qq, imgs)
        finally:
            _save_usage(eid, llm)

        def apply(e):
            for i, x in enumerate(e["questions"]):
                if x["number"] == num:
                    e["questions"][i] = qq
            e["warnings"] = exam_setup.reconcile(e)
        store.update_exam(eid, apply)
        return {}
    return {"job": jobs.start(eid, "generate", f"生成第{num}题答案", run)}


@app.post("/api/exams/{eid}/layout")
async def relayout(eid: str, source: str = Form("blank"), files: List[UploadFile] = File(...)):
    _exam_or_404(eid)
    sub = "sheet" if source == "blank" else "sheet_scan"
    saved = await _save_uploads(eid, sub, files)
    if not saved:
        raise HTTPException(400, "请上传答题纸")
    if source == "blank":
        store.update_exam(eid, lambda e: e["files"].update(sheet=saved))
    llm = _llm(eid, require_key=False)
    st = store.settings()
    d = store.exam_dir(eid)

    def run(progress):
        try:
            exam_setup.run_layout(store, eid, [d / f for f in saved], llm, st, source, progress)
        finally:
            _save_usage(eid, llm)
        return {}
    return {"job": jobs.start(eid, "layout", "识别版面", run)}


@app.put("/api/exams/{eid}/template")
def put_template(eid: str, body: dict = Body(...)):
    _exam_or_404(eid)
    try:
        exam = exam_setup.save_template_edit(store, eid, body.get("pages") or [])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _exam_view(exam)


# ================================================================ 批改

@app.post("/api/exams/{eid}/scans")
async def upload_scans(eid: str, files: List[UploadFile] = File(...)):
    exam = _exam_or_404(eid)
    if not exam.get("template"):
        raise HTTPException(400, "请先在“答题纸版面”里识别版面")
    if not report.graded_numbers(exam):
        raise HTTPException(400, "还没有可批改的解答题：请检查题目满分和版面上的题号标注")
    llm = _llm(eid)
    saved = await _save_uploads(eid, "scans", files)
    if not saved:
        raise HTTPException(400, "请选择扫描件")
    st = store.settings()

    def run(progress):
        try:
            return pipeline.grade_scans(store, eid, llm, st, sorted(saved), progress)
        finally:
            _save_usage(eid, llm)
    return {"job": jobs.start(eid, "grade", f"批改 {len(saved)} 个文件", run)}


@app.post("/api/exams/{eid}/regrade")
def regrade(eid: str, body: dict = Body(default={})):
    _exam_or_404(eid)
    llm = _llm(eid)
    st = store.settings()
    sids = body.get("sheet_ids") or None
    nums = body.get("numbers") or None

    def run(progress):
        try:
            return pipeline.regrade(store, eid, llm, st, sids, nums, bool(body.get("retranscribe")),
                                    bool(body.get("fresh")), progress)
        finally:
            _save_usage(eid, llm)
    what = "重新批改" + (f"{len(sids)}份" if sids else "全部")
    return {"job": jobs.start(eid, "regrade", what, run)}


@app.get("/api/exams/{eid}/results")
def get_results(eid: str):
    _exam_or_404(eid)
    return report.results(store, eid)


@app.get("/api/exams/{eid}/sheets/{sid}")
def get_sheet(eid: str, sid: str):
    try:
        return store.load_sheet(eid, sid)
    except KeyError:
        raise HTTPException(404, "没有这份答卷")


@app.put("/api/exams/{eid}/sheets/{sid}/q/{num}/teacher")
def put_teacher(eid: str, sid: str, num: str, body: dict = Body(...)):
    try:
        return pipeline.set_teacher_score(store, eid, sid, num, score=body.get("score"), parts=body.get("parts"),
                                          comment=str(body.get("comment") or ""), accept=bool(body.get("accept")),
                                          clear=bool(body.get("clear")))
    except (ValueError, TypeError) as e:
        raise HTTPException(400, str(e))
    except KeyError:
        raise HTTPException(404, "没有这道题")


@app.put("/api/exams/{eid}/sheets/{sid}/q/{num}/student")
def put_student_answer(eid: str, sid: str, num: str, body: dict = Body(...)):
    """老师更正选择题/填空题的识别结果。选择题立即重新给分；填空题写法和答案不一致时，交给模型重新判。"""
    try:
        s = pipeline.set_student_answer(store, eid, sid, num, str(body.get("answer") or ""))
    except KeyError:
        raise HTTPException(404, "没有这道题")
    qr = s["questions"][num]
    if qr.get("type") == "填空题" and qr.get("status") == "pending":
        llm = _llm(eid)
        st = store.settings()

        def run(progress):
            try:
                return pipeline.regrade(store, eid, llm, st, [sid], [num], False, False, progress)
            finally:
                _save_usage(eid, llm)
        return {"sheet": s, "job": jobs.start(eid, "regrade", f"重新判断 {sid} 第{num}题", run)}
    return {"sheet": s, "job": None}


@app.put("/api/exams/{eid}/sheets/{sid}/q/{num}/transcription")
def put_transcription(eid: str, sid: str, num: str, body: dict = Body(...)):
    try:
        pipeline.set_transcription(store, eid, sid, num, str(body.get("text") or ""))
    except KeyError:
        raise HTTPException(404, "没有这道题")
    if body.get("regrade"):
        llm = _llm(eid)
        st = store.settings()

        def run(progress):
            try:
                return pipeline.regrade(store, eid, llm, st, [sid], [num], False, False, progress)
            finally:
                _save_usage(eid, llm)
        return {"job": jobs.start(eid, "regrade", f"按修改后的转写重评 {sid} 第{num}题", run)}
    return {"job": None}


@app.put("/api/exams/{eid}/sheets/{sid}/student")
def put_student(eid: str, sid: str, body: dict = Body(...)):
    try:
        s = store.update_sheet(eid, sid, lambda s: s["student"].update(
            {k: str(body.get(k) or "").strip()[:40] for k in ("name", "student_id", "class")}, source="teacher", issues=[]))
        report.export_csv(store, eid)
        return s
    except KeyError:
        raise HTTPException(404, "没有这份答卷")


@app.get("/api/exams/{eid}/export.csv")
def export_csv(eid: str):
    exam = _exam_or_404(eid)
    path = report.export_csv(store, eid)
    return FileResponse(path, media_type="text/csv; charset=utf-8", filename=f"{safe_filename(exam['title'])}_成绩.csv")


@app.get("/api/exams/{eid}/export.zip")
def export_zip(eid: str):
    """成绩表 + 每份批改后的答题纸。先写到磁盘上的临时文件再发送（几十上百份也不占内存），发完就删。"""
    exam = _exam_or_404(eid)
    d = store.exam_dir(eid)
    report.export_csv(store, eid)
    path = d / f"export_{uuid.uuid4().hex[:8]}.zip"
    try:
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(d / "scores.csv", "成绩.csv")
            for s in store.list_sheets(eid):
                stu = s.get("student") or {}
                tag = "_".join(x for x in (s["id"], stu.get("student_id"), stu.get("name")) if x)
                for a in s.get("annotated") or []:
                    p = d / a
                    if p.exists():   # JPEG 本身已经压缩过，直接存
                        z.write(p, f"批改后答题纸/{safe_filename(tag)}_{p.name}", compress_type=zipfile.ZIP_STORED)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    name = f"{safe_filename(exam['title'])}_批改结果.zip"
    from urllib.parse import quote
    from starlette.background import BackgroundTask
    return FileResponse(path, media_type="application/zip", background=BackgroundTask(_remove_quietly, path),
                        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(name)}"})


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _no_running_job(eid: str, what: str) -> None:
    job = jobs.for_exam(eid)
    if job and job["status"] == "running":
        raise HTTPException(409, f"正在{job['label']}，完成后再{what}")


@app.delete("/api/exams/{eid}/sheets/{sid}")
def delete_sheet(eid: str, sid: str):
    _exam_or_404(eid)
    _no_running_job(eid, "删除")
    try:
        pipeline.delete_sheet(store, eid, sid)
    except KeyError:
        raise HTTPException(404, "没有这份答卷")
    return {"ok": True}


@app.post("/api/exams/{eid}/sheets/{sid}/merge")
def merge_sheet(eid: str, sid: str, body: dict = Body(...)):
    """把另一份答卷（比如补扫上传的缺页）并进这份。"""
    _exam_or_404(eid)
    _no_running_job(eid, "合并")
    try:
        sheet, todo = pipeline.merge_sheets(store, eid, sid, str(body.get("source") or ""))
    except KeyError:
        raise HTTPException(404, "没有这份答卷")
    except ValueError as e:
        raise HTTPException(400, str(e))
    job = None
    llm = _llm(eid, require_key=False) if todo else None
    if llm is not None:
        st = store.settings()

        def run(progress):
            try:
                return pipeline.regrade(store, eid, llm, st, [sid], todo, False, False, progress)
            finally:
                _save_usage(eid, llm)
        job = jobs.start(eid, "regrade", f"批改 {sid} 补上的题", run)
    return {"sheet": sheet, "job": job, "todo": todo}


# ================================================================ 班级名单

def _reresolve_all(changed: Optional[str] = None) -> None:
    """名单变了：用到这个名单（或没指定名单、自动对照）的考试，重新把答卷对一遍名单。"""
    for e in store.list_exams():
        try:
            ex = store.load_exam(e["id"])
        except KeyError:
            continue
        bound = ex.get("roster") or []
        if ex.get("sheets") and (not bound or changed in bound):
            j = jobs.for_exam(e["id"])
            if not (j and j["status"] == "running"):
                pipeline.resolve_students(store, e["id"])


@app.get("/api/rosters")
def list_rosters():
    return roster.summary(store.root)


@app.get("/api/rosters/{cid}")
def get_roster(cid: str):
    try:
        return roster.get_class(store.root, cid)
    except KeyError:
        raise HTTPException(404, "名单不存在")


@app.post("/api/rosters/import")
async def import_roster(name: str = Form(""), text: str = Form(""), mode: str = Form("replace"),
                        files: List[UploadFile] = File(default=[])):
    """导入名单：粘贴（从 Excel 复制也行）或上传 CSV / Excel / TXT。名单里有“班级”一列时按班级分开。"""
    rows, bad = [], []
    fallback = name.strip()
    if text.strip():
        r, b = roster.parse_text(text)
        rows += r
        bad += b
    tmpdir = store.root / "tmp_upload"
    for f in files or []:
        if not f or not f.filename:
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in (".csv", ".xlsx", ".xlsm", ".txt", ".xls"):
            raise HTTPException(400, f"不支持的名单文件：{f.filename}（支持 .xlsx、.csv、.txt）")
        tmpdir.mkdir(parents=True, exist_ok=True)
        p = tmpdir / f"{uuid.uuid4().hex[:8]}{ext}"
        p.write_bytes(await f.read())
        try:
            r, b = roster.parse_file(p)
        except ValueError as e:
            raise HTTPException(400, str(e))
        finally:
            p.unlink(missing_ok=True)
        fallback = fallback or Path(f.filename).stem
        rows += r
        bad += b
    if not rows:
        raise HTTPException(400, "没有读出学生：每行要有学号和姓名，例如“2507 张三”" + (f"（看不懂的行：{bad[0]}）" if bad else ""))
    groups = roster.group_by_class(rows, fallback or "未命名班级")
    if name.strip() and len(groups) == 1:
        groups = {name.strip(): next(iter(groups.values()))}
    made = [roster.upsert(store.root, cname, studs, mode="append" if mode == "append" else "replace")
            for cname, studs in groups.items()]
    for c in made:
        _reresolve_all(c["id"])
    return {"classes": [{"id": c["id"], "name": c["name"], "count": len(c["students"])} for c in made], "bad": bad[:20]}


@app.put("/api/rosters/{cid}")
def update_roster(cid: str, body: dict = Body(...)):
    try:
        old = roster.get_class(store.root, cid)
    except KeyError:
        raise HTTPException(404, "名单不存在")
    studs = old["students"]
    bad: List[str] = []
    if "text" in body:
        rows, bad = roster.parse_text(str(body.get("text") or ""))
        studs = [{"sid": r["sid"], "name": r["name"]} for r in rows]
    c = roster.upsert(store.root, str(body.get("name") or old["name"]), studs, cid=cid)
    _reresolve_all(cid)
    return {"class": c, "bad": bad[:20]}


@app.delete("/api/rosters/{cid}")
def delete_roster(cid: str):
    try:
        roster.delete(store.root, cid)
    except KeyError:
        raise HTTPException(404, "名单不存在")
    _reresolve_all(cid)
    return {"ok": True}


@app.put("/api/exams/{eid}/roster")
def bind_roster(eid: str, body: dict = Body(...)):
    """这场考试对照哪几个班的名单（空 = 自动：在全部名单里按学号找）。"""
    _exam_or_404(eid)
    _no_running_job(eid, "换名单")
    known = {c["id"] for c in roster.summary(store.root)}
    cids = [c for c in body.get("classes") or [] if c in known]
    store.update_exam(eid, lambda e: e.__setitem__("roster", cids))
    pipeline.resolve_students(store, eid)
    return _exam_view(store.load_exam(eid))


@app.get("/api/exams/{eid}/file")
def get_file(eid: str, p: str):
    d = store.exam_dir(eid).resolve()
    target = (d / p).resolve()
    if d not in target.parents or not target.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(target, headers={"Cache-Control": "no-cache"})


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return jobs.get(jid)


@app.exception_handler(LLMError)
def llm_error(_, exc: LLMError):
    return JSONResponse({"detail": str(exc)}, status_code=502)
