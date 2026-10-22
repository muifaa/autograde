"""名单解析、题目结构解析、学生信息对名单（不需要模型）。python tests/test_roster.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autograde import roster as R
from autograde.exam_setup import merge_structure, new_question, parse_structure


def main():
    tmp = Path(tempfile.mkdtemp(prefix="ag_roster_"))

    # ---- 粘贴：从 Excel 复制（制表符）、空格分隔、逗号分隔、没有表头、序号+学号两列数字
    rows, bad = R.parse_text("学号\t姓名\t性别\n2501\t张三\t男\n2502\t欧阳 娜娜\t女\n\n")
    assert [(r["sid"], r["name"]) for r in rows] == [("2501", "张三"), ("2502", "欧阳娜娜")] and not bad, rows
    rows, bad = R.parse_text("1 25090315 张三 高二3班\n2 25090316 李四 高二3班\n这一行看不懂")
    assert [(r["sid"], r["name"], r["class"]) for r in rows] == [("25090315", "张三", "高二3班"), ("25090316", "李四", "高二3班")], rows
    assert bad == ["这一行看不懂"], bad
    rows, _ = R.parse_text("序号,姓名\n1,王五\n2,赵六")              # 只有序号：序号当学号
    assert [(r["sid"], r["name"]) for r in rows] == [("1", "王五"), ("2", "赵六")], rows
    rows, _ = R.parse_text("姓名，考号，班级\n钱七，0703，高一7班")      # 中文逗号、前导 0
    assert rows == [{"sid": "0703", "name": "钱七", "class": "高一7班"}], rows

    # ---- 文件：GBK 编码的 CSV、Excel（学号存成数字）
    (tmp / "a.csv").write_bytes("班级,学号,姓名,性别\n高一5班,2501,张三,男\n高一6班,2601,李四,女\n".encode("gb18030"))
    rows, bad = R.parse_file(tmp / "a.csv")
    assert [(r["sid"], r["name"], r["class"]) for r in rows] == [("2501", "张三", "高一5班"), ("2601", "李四", "高一6班")], rows
    groups = R.group_by_class(rows, "x")
    assert sorted(groups) == ["高一5班", "高一6班"]
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["序号", "学号", "姓名", "身份证号"])
        ws.append([1, 25090315, "张三", "110101200801011234"])
        ws.append([2, 25090316.0, "李四", "110101200801011235"])
        wb.save(tmp / "b.xlsx")
        rows, bad = R.parse_file(tmp / "b.xlsx")
        assert [(r["sid"], r["name"]) for r in rows] == [("25090315", "张三"), ("25090316", "李四")], (rows, bad)
    except ImportError:
        print("（没装 openpyxl，跳过 Excel）")

    # ---- 存取、查找、对答卷
    c1 = R.upsert(tmp, "高一5班", [{"sid": "2501", "name": "张三"}, {"sid": "2502", "name": "李四"}, {"sid": "07", "name": "孙八"}])
    R.upsert(tmp, "高一6班", [{"sid": "2501", "name": "周九"}])
    assert [c["count"] for c in R.summary(tmp)] == [3, 1]
    idx_all = R.index(tmp)
    idx5 = R.index(tmp, [c1["id"]])
    s = R.resolve({"omr": {"id": "2502", "complete": True, "flags": []}}, idx_all)
    assert (s["name"], s["student_id"], s["class"], s["source"], s["issues"]) == ("李四", "2502", "高一5班", "roster", []), s
    s = R.resolve({"omr": {"id": "2501", "complete": True, "flags": []}}, idx_all)     # 两个班都有 2501
    assert not s["name"] and any("都有" in x for x in s["issues"]), s
    s = R.resolve({"omr": {"id": "2501", "complete": True, "flags": []}}, idx5)        # 指定了班级就唯一
    assert s["name"] == "张三"
    s = R.resolve({"omr": {"id": "0007", "complete": True, "flags": []}}, idx5)        # 前导 0
    assert s["name"] == "孙八", s
    s = R.resolve({"omr": {"id": "2509", "complete": True, "flags": []}}, idx5)
    assert any("不在名单里" in x for x in s["issues"]), s
    s = R.resolve({"omr": {"id": "25?2", "complete": False, "flags": ["第 3 位没涂"]}, "ai": {"name": "李四"}}, idx5)
    assert (s["name"], s["student_id"], s["source"]) == ("李四", "2502", "roster_name") and len(s["issues"]) == 2, s
    s = R.resolve({"qr_sid": "2501", "omr": {"id": "2502", "complete": True, "flags": []}}, idx5)
    assert s["name"] == "张三" and any("不一样" in x for x in s["issues"]), s
    s = R.resolve({"omr": {"id": "2501", "complete": True, "flags": []}, "ai": {"name": "王五"}}, idx5)
    assert s["name"] == "张三" and any("手写姓名像是" in x for x in s["issues"]), s
    s = R.resolve({"name": "老师写的", "source": "teacher", "issues": ["x"]}, idx5)
    assert s["name"] == "老师写的" and s["issues"] == []
    s = R.resolve({"name": "文华", "student_id": "25090315", "class": "高二3班", "source": "ai"}, R.index(tmp / "none"))
    assert (s["name"], s["student_id"], s["source"]) == ("文华", "25090315", "ai"), s   # 旧数据、没有名单：保持原样
    assert R.pinyin_keys("张三") in (("zhangsan", "zs"), ("", ""))

    # ---- 题目结构
    qs, bad = parse_structure("""1-8 单选 每题5分
第9～11题 多选题 共18分
12、13、14 填空 4分
15 解答 13分 2问
16-17 解答题 15分
18 选择 5分 5个选项
随便写的一行""")
    assert bad == ["随便写的一行"], bad
    by = {q["number"]: q for q in qs}
    assert len(qs) == 18 and by["1"]["type"] == "选择题" and by["1"]["score"] == 5 and not by["1"]["multi"]
    assert by["10"]["multi"] and by["10"]["score"] == 6, by["10"]
    assert by["13"]["type"] == "填空题" and by["13"]["score"] == 4
    assert by["15"]["type"] == "解答题" and len(by["15"]["parts"]) == 2 and by["17"]["score"] == 15
    assert by["18"]["options"] == 5
    old = [dict(new_question("15", "解答题"), answer="原来的答案", score=12)]
    merged = merge_structure(old, qs)
    assert len(merged) == 18 and next(q for q in merged if q["number"] == "15")["answer"] == "原来的答案"
    assert next(q for q in merged if q["number"] == "15")["score"] == 13
    print("ROSTER OK", tmp)


if __name__ == "__main__":
    main()
