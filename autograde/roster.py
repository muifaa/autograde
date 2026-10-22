"""班级名单：导入（粘贴、CSV、Excel）、按学号/姓名查找、对答卷补全学生信息、拼音首字母（搜索用）。

名单存在 data/rosters.json：{"classes": [{"id", "name", "students": [{"sid", "name"}], "updated_at"}]}。
一场考试可以指定对照哪几个班（exam["roster"]）；不指定时在全部名单里按学号找（只认唯一匹配）。
"""
from __future__ import annotations

import csv
import io
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

_lock = threading.RLock()

SID_HEAD = ("学号", "考号", "学籍号", "编号", "准考证号", "id")
NAME_HEAD = ("姓名", "名字", "学生", "name")
CLASS_HEAD = ("班级", "班别", "行政班", "class")
SEQ_HEAD = ("序号", "座号", "班内序号", "座位号")


# ================================================================ 存取

def _path(root: Path) -> Path:
    return Path(root) / "rosters.json"


def load(root: Path) -> dict:
    p = _path(root)
    with _lock:
        if not p.exists():
            return {"classes": []}
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"classes": []}
    data.setdefault("classes", [])
    return data


def save(root: Path, data: dict) -> None:
    p = _path(root)
    with _lock:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)


def summary(root: Path) -> List[dict]:
    out = []
    for c in load(root)["classes"]:
        lens = sorted({len(s["sid"]) for s in c["students"] if s.get("sid")})
        out.append({"id": c["id"], "name": c["name"], "count": len(c["students"]), "updated_at": c.get("updated_at", ""),
                    "sid_lengths": lens})
    return out


def get_class(root: Path, cid: str) -> dict:
    for c in load(root)["classes"]:
        if c["id"] == cid:
            return c
    raise KeyError(cid)


def upsert(root: Path, name: str, students: List[dict], cid: Optional[str] = None, mode: str = "replace") -> dict:
    """新建或更新一个班。mode: replace 用新名单替换；append 追加（同学号的更新姓名）。"""
    name = (name or "").strip()[:40] or "未命名班级"
    with _lock:
        data = load(root)
        cls = next((c for c in data["classes"] if (cid and c["id"] == cid) or (not cid and c["name"] == name)), None)
        if cls is None:
            cls = {"id": uuid.uuid4().hex[:8], "name": name, "students": []}
            data["classes"].append(cls)
        cls["name"] = name
        if mode == "append":
            by = {s["sid"]: s for s in cls["students"] if s.get("sid")}
            for s in students:
                if s.get("sid") and s["sid"] in by:
                    by[s["sid"]]["name"] = s["name"]
                else:
                    cls["students"].append(s)
        else:
            cls["students"] = list(students)
        cls["students"].sort(key=lambda s: (_sid_key(s.get("sid", "")), s.get("name", "")))
        cls["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save(root, data)
        return cls


def delete(root: Path, cid: str) -> None:
    with _lock:
        data = load(root)
        n = len(data["classes"])
        data["classes"] = [c for c in data["classes"] if c["id"] != cid]
        if len(data["classes"]) == n:
            raise KeyError(cid)
        save(root, data)


def _sid_key(sid: str):
    s = re.sub(r"\D", "", sid or "")
    return (0, int(s), sid) if s else (1, 0, sid or "")


def norm_sid(sid) -> str:
    """比对用：只留数字，去掉前导 0（“07”和“7”是同一个人）。"""
    s = re.sub(r"\D", "", str(sid or ""))
    return s.lstrip("0") or ("0" if s else "")


# ================================================================ 解析（粘贴 / CSV / Excel）

_CJK_NAME = re.compile(r"^[㐀-鿿·•．.A-Za-z ]{2,20}$")
_CLASS = re.compile(r"班$|^(高|初)?[一二三四五六七八九十\d]+[（(]?\d*[)）]?班")


def _cells(line: str) -> List[str]:
    if re.search(r"[\t,，;；|]", line):
        cells = re.split(r"[\t,，;；|]", line)
    else:
        cells = re.split(r"\s+", line)
    return [c.strip().strip('"').strip() for c in cells]


def _header_map(cells: List[str]) -> Optional[Dict[str, int]]:
    low = [c.lower() for c in cells]
    m: Dict[str, int] = {}
    for k, c in enumerate(low):
        if any(h in c for h in SID_HEAD) and "sid" not in m:
            m["sid"] = k
        elif any(h in c for h in NAME_HEAD) and "name" not in m:
            m["name"] = k
        elif any(h in c for h in CLASS_HEAD) and "class" not in m:
            m["class"] = k
        elif any(h in c for h in SEQ_HEAD) and "seq" not in m:
            m["seq"] = k
    if "name" in m and ("sid" in m or "seq" in m):
        if "sid" not in m:
            m["sid"] = m.pop("seq")       # 只有序号：序号就当学号用（答题纸上涂的就是它）
        return m
    return None


def _guess_row(cells: List[str]) -> Optional[Tuple[str, str, str]]:
    nums = [c for c in cells if re.fullmatch(r"\d{1,12}", c)]
    klass = next((c for c in cells if _CLASS.search(c) and not re.fullmatch(r"\d+", c)), "")
    names = [c for c in cells if c and c != klass and not re.fullmatch(r"[\d\-/:.]+", c) and _CJK_NAME.match(c)
             and c not in ("男", "女")]
    if not names or not nums:
        return None
    sid = max(nums, key=len)              # 有序号又有学号时，学号通常更长
    return sid, names[0], klass


def parse_rows(rows: Iterable[List[str]], default_class: str = "") -> Tuple[List[dict], List[str]]:
    """每行若干格 → [{"sid", "name", "class"}]，以及看不懂的行。第一行是表头时按表头取列。"""
    out, bad = [], []
    hmap = None
    for k, cells in enumerate(rows):
        cells = [str(c).strip() if c is not None else "" for c in cells]
        if not any(cells):
            continue
        if hmap is None and k < 3:
            h = _header_map(cells)
            if h:
                hmap = h
                continue
        if hmap:
            get = lambda key: cells[hmap[key]].strip() if key in hmap and hmap[key] < len(cells) else ""
            sid, name, klass = get("sid"), get("name"), get("class")
            sid = re.sub(r"\.0$", "", sid)                  # Excel 把学号存成数字时会多一个 .0
            if not name or not re.fullmatch(r"\d{1,12}", sid):
                if any(cells):
                    bad.append(" ".join(c for c in cells if c))
                continue
        else:
            g = _guess_row(cells)
            if g is None:
                bad.append(" ".join(c for c in cells if c))
                continue
            sid, name, klass = g
        out.append({"sid": sid, "name": re.sub(r"\s+", "", name) if re.search(r"[㐀-鿿]", name) else name.strip(),
                    "class": klass or default_class})
    return out, bad


def parse_text(text: str, default_class: str = "") -> Tuple[List[dict], List[str]]:
    return parse_rows((_cells(line) for line in (text or "").splitlines()), default_class)


def parse_file(path: Path, default_class: str = "") -> Tuple[List[dict], List[str]]:
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        try:
            import openpyxl
        except ImportError:
            raise ValueError("读取 Excel 需要 openpyxl：pip install openpyxl；或者在 Excel 里复制名单，直接粘贴到文本框")
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        try:
            ws = wb.worksheets[0]
            rows = [["" if v is None else (str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)) for v in row]
                    for row in ws.iter_rows(values_only=True)]
        finally:
            wb.close()
        return parse_rows(rows, default_class)
    if ext == ".xls":
        raise ValueError("不支持老的 .xls 格式：请在 Excel 里另存为 .xlsx 或 .csv，或者复制名单直接粘贴")
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gb18030"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("文件编码看不懂，请另存为 UTF-8 的 CSV，或者直接粘贴")
    if ext == ".csv":
        return parse_rows(list(csv.reader(io.StringIO(text))), default_class)
    return parse_text(text, default_class)


def group_by_class(rows: List[dict], fallback: str) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for r in rows:
        out.setdefault(r.get("class") or fallback, []).append({"sid": r["sid"], "name": r["name"]})
    return out


# ================================================================ 查找、对答卷补全学生信息

def index(root: Path, class_ids: Optional[List[str]] = None) -> dict:
    """{"by_sid": {规范学号: [学生]}, "by_name": {姓名: [学生]}, "classes": [班名], "n": 人数}。学生带 class / class_id。"""
    by_sid: Dict[str, List[dict]] = {}
    by_name: Dict[str, List[dict]] = {}
    names = []
    n = 0
    for c in load(root)["classes"]:
        if class_ids and c["id"] not in class_ids:
            continue
        names.append(c["name"])
        for s in c["students"]:
            e = {"sid": s.get("sid", ""), "name": s.get("name", ""), "class": c["name"], "class_id": c["id"]}
            n += 1
            if e["sid"]:
                by_sid.setdefault(norm_sid(e["sid"]), []).append(e)
            if e["name"]:
                by_name.setdefault(e["name"], []).append(e)
    return {"by_sid": by_sid, "by_name": by_name, "classes": names, "n": n}


def exam_index(root: Path, exam: dict) -> dict:
    return index(root, exam.get("roster") or None)


def resolve(stu: dict, idx: dict) -> dict:
    """按名单补全一份答卷的学生信息（改的就是 stu）。
    依据：每人一份答题纸二维码里的学号 > 涂的学号 > AI 认出的手写学号；都没有时按 AI 认出的姓名找。
    老师手动改过的（source=teacher）不动。问题写进 stu["issues"]。"""
    if stu.get("source") == "teacher":
        stu["issues"] = []
        return stu
    ai = stu.get("ai")
    if ai is None and stu.get("source") in ("ai", "", None) and (stu.get("name") or stu.get("student_id")):
        ai = {"name": stu.get("name", ""), "student_id": stu.get("student_id", ""), "class": stu.get("class", "")}
        stu["ai"] = ai                         # 旧版本存的 AI 识别结果
    ai = ai or {}
    omr = stu.get("omr") or {}
    issues: List[str] = list(omr.get("flags") or []) if not stu.get("qr_sid") else []
    omr_id = omr.get("id") or ""
    cands = []
    if stu.get("qr_sid"):
        cands.append(("qr", stu["qr_sid"]))
    if omr_id and omr.get("complete"):
        cands.append(("omr", omr_id))
    if ai.get("student_id"):
        cands.append(("ai", ai["student_id"]))
    if stu.get("qr_sid") and omr_id and omr.get("complete") and norm_sid(omr_id) != norm_sid(stu["qr_sid"]):
        issues.append(f"涂的学号 {omr_id} 和答题纸上印的 {stu['qr_sid']} 不一样（可能拿错了别人的答题纸）")
    have_roster = idx.get("n", 0) > 0
    name, sid, klass, source = "", "", "", ""
    if cands:
        how, sid = cands[0]
        source = how
        if have_roster:
            hits = idx["by_sid"].get(norm_sid(sid)) or []
            if len(hits) == 1:
                name, klass, sid, source = hits[0]["name"], hits[0]["class"], hits[0]["sid"], "roster"
                if ai.get("name") and _clean_name(ai["name"]) and _clean_name(ai["name"]) != hits[0]["name"]:
                    issues.append(f"学号 {sid} 在名单上是{hits[0]['name']}，答题纸上的手写姓名像是“{ai['name']}”")
            elif len(hits) > 1:
                issues.append(f"学号 {sid} 在 {'、'.join(h['class'] for h in hits)} 都有，请在复核页选对照的名单")
            else:
                issues.append(f"学号 {sid} 不在名单里")
    if not name and ai.get("name"):
        name = _clean_name(ai["name"]) or ai["name"]
        klass = klass or ai.get("class") or ""
        if have_roster and not sid:
            hits = idx["by_name"].get(name) or []
            if len(hits) == 1:
                sid, klass, source = hits[0]["sid"], hits[0]["class"], "roster_name"
                issues.append(f"学号没涂清楚，按手写姓名在名单里找到了{name}（{sid}），请确认")
        source = source or "ai"
    if not sid and omr_id and not omr.get("complete") and not issues:
        issues.append(f"学号没涂清楚：{omr_id}")
    stu.update(name=name, student_id=sid, **({"class": klass} if klass or "class" not in stu else {}), source=source,
               issues=list(dict.fromkeys(issues)))
    return stu


def _clean_name(s: str) -> str:
    s = re.sub(r"\s+", "", str(s or ""))
    return s if re.fullmatch(r"[㐀-鿿·]{2,6}", s) else ""


# ================================================================ 拼音（搜索用）

def pinyin_keys(name: str) -> Tuple[str, str]:
    """(全拼, 首字母)，例如 张三 → ("zhangsan", "zs")。没装 pypinyin 时返回空串。"""
    if not name or not re.search(r"[㐀-鿿]", name):
        return "", ""
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return "", ""
    full = lazy_pinyin(name, errors="ignore")
    first = lazy_pinyin(name, style=Style.FIRST_LETTER, errors="ignore")
    return "".join(full).lower(), "".join(first).lower()
