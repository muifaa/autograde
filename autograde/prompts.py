"""所有提示词集中放在这里，方便按学科、学段调整。

版本号变了会让缓存失效（重新调用模型）。
"""
from __future__ import annotations

import json
import re
from typing import List, Optional

PROMPT_VERSION = "2026-10-08.2"


def question_text(q: dict) -> str:
    """题目全文：题干 + 小问（旧数据里小问文字单独存在 parts[].text，这里补回去）。"""
    stem = (q.get("stem") or "").strip()
    extra = []
    for p in q.get("parts") or []:
        t = (p.get("text") or "").strip()
        if t and t[:12] not in stem:
            extra.append(f"{p.get('label', '')} {t}")
    return (stem + ("\n" + "\n".join(extra) if extra else "")).strip()


CONVENTION_TEXT = {
    "cumulative": "参考答案中步骤后面的“……X分”是累计分：学生做到这一步（且前面的步骤正确），这一小问累计可得 X 分；相邻两个标注之差就是这一步的分值。",
    "cumulative_all": "参考答案中步骤后面的“……X分”是全题累计分：做到这一步，全题累计可得 X 分；相邻两个标注之差就是这一步的分值。",
    "step": "参考答案中步骤后面的“……X分”是这一步本身的分值，同一小问内各步分值相加等于该小问满分。",
}


def _qlist(questions: List[dict]) -> str:
    rows = []
    for q in questions:
        s = f"第{q['number']}题（{q.get('type') or '题型未知'}"
        if q.get("score"):
            s += f"，{q['score']}分"
        rows.append(s + "）")
    return "、".join(rows)


# ================================================================ 版面标注

LAYOUT_SYSTEM = "你是答题卡版面分析助手，负责识别答题纸上各个作答区域对应的题号、题型和分值。"


def layout_user(page_no: int, page_total: int, region_ids: List[str], questions: List[dict]) -> str:
    known = f"本次考试的题目（来自试卷）：{_qlist(questions)}。\n" if questions else ""
    example = {"regions": [
        {"id": "R1", "kind": "选择题", "questions": ["1", "2", "3", "4", "5", "6", "7", "8"], "full_score": None, "header": "一、选择题"},
        {"id": "R2", "kind": "解答题", "questions": ["17"], "full_score": 10, "header": "17.（10分）"}],
        "unboxed": ""}
    return f"""这是一张答题纸（第{page_no}页，共{page_total}页）。程序已检测出 {len(region_ids)} 个封闭的作答区域，在第一张图里用不同颜色的框标出，编号写在每个框的右下角：{'、'.join(region_ids)}。第二张图是每个区域顶部（含框上方一点）的放大图，左边写着区域编号，便于看清印刷的题号和分值。
{known}
请判断每个区域：
- kind：只能是 "解答题"、"填空题"、"选择题"、"学生信息"、"其他" 之一（写作文/作图等也归入"解答题"）
- questions：这个区域对应的题号（字符串列表）。填空题、选择题区域列出包含的全部题号；一个框里有多道解答题就全部列出；写着"接第X题""第X题（续）"的也写上 X
- full_score：框内印刷的分值（如 "15(12分)" 写 12），没有写 null
- header：框内或框正上方印刷的标题文字，没有写空字符串

如果答题纸上有明显的作答区域没被框出来，在 unboxed 里用一句话说明位置。
只输出 json，格式示例：
{json.dumps(example, ensure_ascii=False)}"""


# ================================================================ 试卷录题

PAPER_SYSTEM = ("你是试卷录入员，负责把试卷图片里的题目逐字准确地录入为文本。"
                "你只录入图片里实际印着的内容：不解题、不改写、不补充，图片里没有的题目绝不能写。")

_STEM_RULES = """- stem：从题号后面开始的题目全文，**包括题干和全部小问**。每个小问另起一行，保留原编号，例如 "(1) 设……；"。
  选择题把选项也写进 stem。"（本题满分15分，本题共有2个小题……）"这类分值说明不要写进 stem。
- score：本题满分（试卷上写了才填，否则 null）
- parts：小问编号和分值 [{"label":"(1)","score":6}]，score 只填试卷上写明的分值，没写为 null；没有小问写 []
- figure：配图的文字描述（图形、各点位置关系、图中标注的已知量），没有图写空字符串
- 数学式用 LaTeX，行内用 $...$；"因为/所以"等文字照原样写，不要换成符号；json 字符串里的反斜杠写成两个（如 "\\frac{1}{2}"）"""


def paper_q_user(number: str, n_images: int, text_layer: str = "") -> str:
    """一道题的截图（数字版 PDF 按题号位置裁出来的）。"""
    example = {"type": "解答题", "score": 15, "parts": [{"label": "(1)", "score": 6}, {"label": "(2)", "score": 9}],
               "stem": "已知复数 $z=3+m\\mathrm{i}$，其中 $m\\in\\mathbf{R}$．\n(1) 设 $z_1=(1+3\\mathrm{i})z$，若 $z_1$ 是纯虚数，求实数 $m$ 的值；\n(2) ……",
               "figure": ""}
    imgs = f"下面 {n_images} 张图按顺序拼起来是" if n_images > 1 else "下面这张图是"
    tl = ""
    if text_layer.strip():
        tl = ("\n从 PDF 文字层提取的同一区域文字（汉字准确，公式是乱码，公式以图片为准）：\n<<<\n"
              + text_layer[:4000] + "\n>>>\n")
    return f"""{imgs}试卷第{number}题（已经按题号位置单独裁出来，图里只有这一道题）。{tl}
请把这道题录入为 json：
- type："解答题"/"填空题"/"选择题"/"其他"
{_STEM_RULES}
只录入第{number}题；图的边缘如果带到一点相邻题目的内容，忽略它。只输出 json，格式示例：
{json.dumps(example, ensure_ascii=False)}"""


def paper_doc_user(n_pages: int) -> str:
    """没有可用文字层（扫描件/照片）时，整份试卷一次录入。"""
    example = {"questions": [{"number": "18", "type": "解答题", "score": 15,
                              "parts": [{"label": "(1)", "score": 6}, {"label": "(2)", "score": 9}],
                              "stem": "已知复数 $z=3+m\\mathrm{i}$……\n(1) ……\n(2) ……", "figure": ""}],
               "section_notes": ["三、解答题（本大题共有5题，满分78分）"]}
    return f"""下面 {n_pages} 张图按顺序是一份试卷的全部页面。请录入试卷上的题目，输出 json。
- 解答题：完整录入，字段如下。
- 选择题、填空题：只写 number、type、score，stem 写空字符串（这些题不需要录题干）。
每道解答题：number（题号字符串）、type
{_STEM_RULES}
- 一道题跨页时合并成一道题。大题标题（如"三、解答题（本大题共有5题，满分78分）"）原样写进 section_notes。
只录入图片里实际有的题目。只输出 json，格式示例：
{json.dumps(example, ensure_ascii=False)}"""


def paper_text_user(text: str) -> str:
    """老师直接粘贴/上传了题目文字（txt/md）时用。"""
    return f"""下面是一份试卷（或其中一部分）的文字，请整理为 json：{{"questions":[…],"section_notes":[…]}}。
每道题：number、type（"选择题"/"填空题"/"解答题"/"其他"）
{_STEM_RULES}
只整理原文里有的题，原文照录不要改写。只输出 json。
<<<
{text[:30000]}
>>>"""


# ================================================================ 参考答案/评分标准录入

ANSWERS_SYSTEM = ("你是阅卷组的录入员，负责把参考答案（含评分标准）逐字准确地录入为文本。"
                  "你只录入图片里实际印着的内容：不解题、不改写、不总结，图片里没有的内容绝不能写。")

_ANSWER_RULES = """1. 逐字照录，不改写、不省略、不总结、不补充。
2. 保留小问编号 (1)(2)…… 和原来的分行。
3. 步骤后面的分值标注照录，统一写成"……X分"（例如"……2分"），位置不变。
4. 数学式用 LaTeX（行内 $...$），方程组/分段写成 \\begin{cases}…\\end{cases}；"因为/所以"等文字照原样写。
5. json 字符串里的反斜杠写成两个。"""


def answer_q_user(number: str, n_images: int, text_layer: str = "", stem: str = "") -> str:
    """参考答案里一道题的截图（数字版 PDF 按题号位置裁出来的，跨页的已经拼好顺序）。"""
    imgs = f"下面 {n_images} 张图按顺序连起来是" if n_images > 1 else "下面这张图是"
    tl = ""
    if text_layer.strip():
        tl = ("\n从 PDF 文字层提取的同一区域文字（汉字准确，公式是乱码，公式以图片为准）：\n<<<\n"
              + text_layer[:5000] + "\n>>>\n")
    st = f"\n这道题的题目（只用来帮你辨认符号）：\n{stem[:1200]}\n" if stem.strip() else ""
    return f"""{imgs}参考答案中第{number}题的部分（已经按题号位置单独裁出来）。{tl}{st}
请把图中这部分答案录入为 json：{{"answer": "……"}}
{_ANSWER_RULES}
6. 只录入第{number}题；图的开头或结尾如果带到一点别的题，不要录入。
只输出 json。"""


def answers_doc_user(n_pages: int, numbers: List[str], objective: Optional[List[str]] = None) -> str:
    """没有可用文字层（扫描件/照片）时，整份答案一次录入。"""
    example = {"items": [{"number": "18", "answer": "(1) $z_1=(1+3\\mathrm{i})(3+m\\mathrm{i})=3-3m+(9+m)\\mathrm{i}$，……3分\n因为……，解得 $m=1$；……6分\n(2) ……"}],
               "objective": {"1": "\\sqrt{5}", "13": "B"}}
    need = f"试卷的解答题是第 {'、'.join(numbers)} 题。" if numbers else ""
    if objective:
        need += (f"选择题、填空题是第 {'、'.join(objective)} 题，它们的答案写进 objective（题号: 答案；"
                 "选择题只写字母，填空题写 LaTeX，不加 $；答案里没有的题不要写）。")
    return f"""下面 {n_pages} 张图按顺序是一份参考答案的全部页面。{need}
请录入答案里**实际出现的**每道解答题的参考答案，输出 json。
{_ANSWER_RULES}
6. 一道题的答案跨页时，把后一页开头接续的内容合并到这道题里。
7. 答案里没有的题不要输出，更不要自己解题。
只输出 json，格式示例：
{json.dumps(example, ensure_ascii=False)}"""


def answers_text_user(text: str, numbers: List[str]) -> str:
    need = f"试卷的解答题是第 {'、'.join(numbers)} 题。" if numbers else ""
    return f"""下面是参考答案的文字。{need}请把其中每道解答题的答案整理为 json：{{"items":[{{"number":"18","answer":"……"}}]}}
{_ANSWER_RULES}
6. 原文里没有的题不要输出。
<<<
{text[:30000]}
>>>"""


def objective_answers_user(kind: str, numbers: List[str], n_images: int, text_layer: str = "") -> str:
    """参考答案里“填空题/选择题”那一段的截图（答案常常一行写好几题）。"""
    ex = {"answers": {"1": "\\sqrt{5}", "2": "5-\\mathrm{i}", "3": "\\frac{\\pi}{2}"}} if kind == "填空题" else {"answers": {"13": "B", "14": "C"}}
    imgs = f"下面 {n_images} 张图按顺序是" if n_images > 1 else "下面这张图是"
    tl = ""
    if text_layer.strip():
        tl = "\n从 PDF 文字层提取的同一区域文字（题号准确，公式是乱码，以图片为准）：\n<<<\n" + text_layer[:3000] + "\n>>>\n"
    rule = ("- 只写选项字母（多选题写在一起，如 \"BD\"）。" if kind == "选择题" else
            "- 数学式写成 LaTeX，不要加 $；一题有几个空或几个答案的照原样写（如 \"x=0 或 x=\\frac{\\pi}{4}\"）。")
    return f"""{imgs}参考答案中{kind}部分的截图。{tl}
需要的题号：{'、'.join(numbers)}。请逐题录入图中的答案，输出 json：{{"answers": {{"题号": "答案"}}}}
- 只录入图里写着的答案，不要自己解题；图里找不到的题号不要写。
{rule}
- json 字符串里的反斜杠写成两个。只输出 json，格式示例：
{json.dumps(ex, ensure_ascii=False)}"""


# ================================================================ 填空题：识别学生答案、判对错

FILL_SYSTEM = ("你是阅卷组的录入员，负责把学生在填空题横线上手写的答案如实转写成文本。"
               "你只照抄：不解题、不纠错、不化简、不补全。")


def fill_transcribe_user(numbers: List[str], n_images: int) -> str:
    ex = {"answers": {"1": "\\frac{\\sqrt{5}}{5}", "2": "", "3": "[2,4]"}, "unclear": []}
    return f"""下面 {n_images} 张图，每张由若干横条组成：每条左边灰底方框里是题号（程序标注的，准确无误），右边是学生在这道填空题横线上写的答案（手写）。
题号：{'、'.join(numbers)}。请转写每道题学生写的最终答案，输出 json。
规则：
1. 照学生写的抄，写错了也照抄，不要改正、不要化简、不要补全。
2. 数学式写成 LaTeX，不要加 $，例如 \\frac{{\\sqrt{{5}}}}{{5}}、5-\\mathrm{{i}}、[2,4]、\\arctan\\frac{{5\\sqrt{{2}}}}{{6}}。
3. 学生划掉的内容不要写，只写最后保留的答案；横条里写了几个答案就都写上。
4. 没写（空白）的写空字符串 ""。
5. 看不清的字写 [?]，并把这道题的题号放进 unclear。
6. 忽略印刷的横线、小方框、题号和任何红色批改痕迹。
json 字符串里的反斜杠写成两个。只输出 json，格式示例：
{json.dumps(ex, ensure_ascii=False)}"""


def fill_region_user(numbers: List[str]) -> str:
    """没能定位到每条横线时：整块填空题区域一起识别，题号由模型自己看印刷的题号。"""
    ex = {"answers": {numbers[0] if numbers else "1": "\\sqrt{3}"}, "unclear": []}
    return f"""下面是答题纸上填空题的作答区域，每条横线前面印着题号（第 {'、'.join(numbers)} 题）。
请按印刷的题号，逐题转写学生写在横线上的最终答案，输出 json：{{"answers": {{"题号": "答案"}}, "unclear": []}}。
规则同照抄：不改正、不化简；数学式写成 LaTeX（不加 $）；划掉的不写；空白写 ""；看不清写 [?] 并把题号放进 unclear；忽略红色批改痕迹。
json 字符串里的反斜杠写成两个。只输出 json，格式示例：
{json.dumps(ex, ensure_ascii=False)}"""


JUDGE_SYSTEM = "你是严谨的中学数学阅卷老师，负责判断学生填空题答案的对错。"


def fill_judge_user(items: List[dict], strict: bool = False) -> str:
    """items: [{"number", "question", "reference", "student", "notes"}]"""
    rows = []
    for it in items:
        rows.append({"题号": it["number"], "题目": (it.get("question") or "")[:400], "参考答案": it["reference"],
                     "学生答案": it["student"], **({"老师说明": it["notes"]} if it.get("notes") else {})})
    simp = ("必须化简到最简形式才算对（如 2/4、√8 不算对）。" if strict else
            "没有化简到最简但数值完全相等的也算对（如 2/4 与 1/2；√8 与 2√2）。")
    ex = {"results": {"1": {"correct": True, "uncertain": False, "reason": "与参考答案相同"},
                      "2": {"correct": False, "uncertain": False, "reason": "符号错误，应为 -4/5"}}}
    return f"""判断下面每道填空题学生的答案是否正确。学生答案是手写识别的结果。
{json.dumps(rows, ensure_ascii=False, indent=1)}

判分规则：
1. 学生答案与参考答案数学上等价就算对，写法不同不影响：如 √5 与 5^(1/2)；1+√10 与 √10+1；0.5 与 1/2；区间、集合、不等式表示同一范围也算对（如 [2,4] 与 2≤ω≤4）。{simp}
2. 题目要求了答案形式的（如“用区间表示”“写成弧度”）按要求判；横线后面已经印了单位的，学生不写单位不扣分；角度单位、弧度与度混用要看题目要求。
3. 有多个答案的（如“x=0 或 x=π/4”），必须全部写对、没有多写才算对，顺序不限。
4. 学生答案里有 [?]（识别不清），而看不清的部分会影响对错时，uncertain 设为 true。
5. 拿不准参考答案和学生答案是否等价时，认真推算；仍拿不准就 uncertain 设为 true。
6. 只判对错，不给部分分。老师说明优先。
只输出 json，格式示例：
{json.dumps(ex, ensure_ascii=False)}"""


# ================================================================ 选择题：涂卡区域整体识别（定位不到选项框时）

CHOICE_SYSTEM = "你负责识别答题卡上选择题的涂卡结果。"


def choice_region_user(numbers: List[str]) -> str:
    ex = {"answers": {numbers[0] if numbers else "1": "B"}, "unclear": []}
    return f"""下面是答题卡上选择题的作答区域（第 {'、'.join(numbers)} 题），每道题后面有 A、B、C、D 等选项框，学生把选中的框涂黑。
请按印刷的题号逐题写出被涂黑的选项，输出 json：{{"answers": {{"题号": "字母"}}, "unclear": []}}。
没涂的写空字符串；涂了几个就都写上（如 "BD"）；涂得很浅、有擦痕或拿不准的，把题号放进 unclear。只输出 json，格式示例：
{json.dumps(ex, ensure_ascii=False)}"""


# ================================================================ 生成参考答案与评分细则

GEN_SYSTEM = "你是资深的中学数学教师和阅卷组长，熟悉高考评分标准。你要为解答题写出正确、规范的参考答案和可操作的评分细则。"


def gen_user(q: dict) -> str:
    parts = q.get("parts") or []
    pl = "；".join(f"{p.get('label')} {p.get('score') if p.get('score') is not None else '?'}分" for p in parts) or "无小问"
    example = {"parts": [{"label": "(1)", "score": 5}, {"label": "(2)", "score": 7}],
               "answer": "（1）证明：……\n（2）解：……",
               "rubric": "（1）共5分：① 由……得……（1分）；② ……（2分）；③ 结论（2分）。\n（2）共7分：……\n其他解法：……\n常见错误：……",
               "issues": ""}
    return f"""请为第{q['number']}题写出参考答案和评分细则（将交给 AI 阅卷使用，老师会审核）。
题目（满分 {q.get('score') or '?'} 分；小问分值：{pl}；"?" 表示试卷没写，请按常规合理分配，各小问之和等于满分）：
{question_text(q)}
{('图形说明：' + q['figure']) if q.get('figure') else ''}
（如果附了试卷图片，题目配图以图片为准。）

要求：
1. 先认真解题并检验，保证答案正确；参考答案写出完整、规范的步骤，数学式用 LaTeX（行内 $...$）。
2. 评分细则按小问写，每个小问列出得分点及分值，得分点之和等于该小问分值。体现"踩点给分"：关键依据、关键步骤、结论分别给分；写清楚缺什么扣几分。
3. 列出其他常见的正确解法（如向量法、几何法）及其得分点的对应方式；列出常见错误及扣分办法。
4. 如果题目条件缺失或有歧义，写在 issues 里，否则 issues 为空字符串。
json 字符串里的反斜杠写成两个。只输出 json，格式示例：
{json.dumps(example, ensure_ascii=False)}"""


# ================================================================ 选择题、填空题：没有答案时让模型做

OBJ_SOLVE_SYSTEM = "你是严谨的中学数学老师，正在为试卷上的选择题、填空题做标准答案。你会认真解题、验算后才给出最终答案。"


def obj_solve_user(q: dict, n_images: int, attempt: int = 0) -> str:
    kind = q.get("type") or "选择题"
    stem = question_text(q)
    if kind == "选择题":
        n_opt = q.get("options") or 4
        how = (f"这是一道选择题（选项 A–{'ABCDEFGH'[max(2, min(8, int(n_opt))) - 1]}）。"
               + ("这是多选题，写出全部正确选项（如 \"BD\"）。" if q.get("multi") else
                  "一般只有一个正确选项；如果题目说明是多选题（或“多项选择”“不定项”），写出全部正确选项（如 \"BD\"）。")
               + "answer 只写大写字母。")
        ex = {"answer": "B", "brief": "由……得……，故选 B", "confidence": "high"}
    else:
        how = ("这是一道填空题。answer 写最终结果，数学式用 LaTeX、不加 $（如 \\frac{\\sqrt{5}}{5}）；"
               "按题目要求的形式写（区间、弧度、最简形式等）；一题有几个空的，按顺序用分号隔开；横线后面印好的单位不用写。")
        ex = {"answer": "\\frac{\\sqrt{5}}{5}", "brief": "……", "confidence": "high"}
    img = (f"题目见下面{'这张' if n_images == 1 else f'{n_images} 张'}图（试卷原题截图），以图为准。" if n_images else "")
    txt = f"\n题目文字（可能有识别错误，与图冲突时以图为准）：\n{stem[:1500]}\n" if stem else ""
    return f"""请做第{q['number']}题。{img}{txt}
{how}
要求：认真解题，并代入或用另一种方法验算；不确定时 confidence 写 low。brief 用一两句话写关键思路（数学式用普通文字，不要反斜杠）。
（独立解答 #{attempt + 1}）
json 字符串里的反斜杠写成两个。只输出 json，格式示例：
{json.dumps(ex, ensure_ascii=False)}"""


# ================================================================ 平时作业：只判对错，不打分

HW_SYSTEM = """你是认真负责的中学数学老师，在批改学生的平时作业。作业不打分，只判断做得对不对、错在哪里，方便学生订正、老师讲评。

批改原则：
1. 对：方法正确，过程基本完整，结论正确。书写不规范、省略了显而易见的步骤不算错。
2. 半对：思路基本正确，但有计算错误、关键步骤或依据缺失、结论不完整；或者几个小问有对有错。
3. 错：方法错误、关键步骤错误导致结论错误，或者只写了一个错误的结论；没有作答也算错（error 写“未作答”）。
4. 方法和参考答案不同但正确的，算对。没有参考答案时，自己认真解题后再判断。
5. 学生作答是手写识别的转写，可能有个别识别错误；附有原图时以原图为准。识别可疑、影响判断的，needs_review 设为 true 并在 review_notes 说明。
6. 学生划掉的内容（~~……~~）不算作答。区域里如果还有别的题的作答，只批本题。"""


def hw_question_block(q: dict) -> str:
    blocks = [f"【题目】第{q['number']}题", question_text(q) or "（题目文字没有录入，看图）"]
    if q.get("figure"):
        blocks.append(f"【图形说明】{q['figure']}")
    if (q.get("answer") or "").strip():
        blocks += ["【参考答案】", q["answer"].strip()]
    if q.get("notes"):
        blocks += ["【老师说明】", q["notes"].strip()]
    return "\n".join(blocks)


def hw_student_block(q: dict, transcription: str, legibility: str, with_image: bool) -> str:
    labels = [p.get("label") for p in q.get("parts") or [] if p.get("label")]
    ex = {"verdict": "半对",
          "parts": [{"label": l, "verdict": v, "comment": c} for l, v, c in
                    zip(labels or ["(全题)"], ["对", "错"], ["", "第二步求导时符号错了"])],
          "error": "（2）问求导时 -x 的导数写成了 1，后面都跟着错", "comment": "（1）问正确；（2）问注意复合函数求导的符号。",
          "confidence": "high", "needs_review": False, "review_notes": ""}
    img = "原图附在后面，与转写冲突时以原图为准。" if with_image else ""
    return f"""【学生作答（手写识别转写，可能有个别识别错误）】
{transcription.strip() or '（空白，未作答）'}
（识别清晰度：{legibility or '未知'}）{img}

请判断这道题做得对不对。只输出 json，格式如下：
{json.dumps(ex, ensure_ascii=False)}
说明：
- verdict 只能是 “对”“半对”“错”。有小问的，parts 按小问顺序逐个判断（label 用 {('、'.join(labels)) if labels else '(全题)'}）；没有小问的 parts 只写一项 (全题)。
- error 写第一个出错的地方和原因（对的写空字符串）；comment 是写给学生看的一句话订正提示。
- confidence 取 high / medium / low；没把握或识别存疑时 needs_review 设为 true，并在 review_notes 说明。
- 所有文字用中文；数学式用普通文字和符号书写（如 √2、∠EAD），不要用 LaTeX，不要出现反斜杠。"""


# ================================================================ 手写转写

TRANSCRIBE_SYSTEM = ("你是中学数学阅卷组的录入员，负责把学生手写的作答逐字如实地转写成文本，供阅卷老师评分。"
                     "你只负责照抄：不解题、不评分、不纠错、不补全。")


def transcribe_user(q: dict, n_images: int) -> str:
    multi = f"下面 {n_images} 张图按顺序是同一道题的连续作答区域。" if n_images > 1 else "下面的图是这道题的作答区域。"
    stem = question_text(q)
    stem_txt = f"\n题目（只用来帮助你辨认字母和符号，不要据此修改学生写的内容）：\n{stem[:1500]}\n" if stem else ""
    return f"""{multi}这是第{q['number']}题。{stem_txt}
转写规则：
1. 逐字照写学生写的内容。学生写错的地方（计算错误、符号错误、笔误、逻辑跳步、没写完）一律照写，绝对不要改正、补全或替学生推导。
2. 数学式用 LaTeX，行内用 $...$，例如 $\\angle DBG=\\arctan\\frac{{\\sqrt{{5}}}}{{5}}$；"∵""∴"写成 ∵ ∴。
3. 保持原来的书写顺序、分行和小题编号（如 (1)、(2)）。作答分成左右两栏时，先转写左栏，再单独一行写【右栏】，然后转写右栏。
4. 学生划掉的内容写成 ~~被划掉的内容~~，看不清划掉的是什么就写 ~~[划掉]~~。
5. 看不清的字写 [?]；有两种可能时写成 [?候选1|候选2]。不要因为"按题意应该是"就认定是某个字。
6. 学生画的图写成 [图：图中的点、线和标注]。
7. 忽略印刷的文字和线框（题号、分值栏、页码等）和任何红色批改痕迹。
8. 如果区域里还有别的题的作答，也照样转写，并在那部分前单独一行写【第X题】。
9. 如果学生没有作答（区域里只有印刷内容或一片空白），只输出：（未作答）

只输出转写结果，不要任何解释。最后单独一行写清晰度：[[清晰度:高]]、[[清晰度:中]] 或 [[清晰度:低]]（潦草、大量看不清写低）。"""


# ================================================================ 阅卷给分

GRADE_SYSTEM = """你是一位经验丰富、严谨公正的中学数学阅卷老师。你要根据参考答案和评分细则，对学生解答题的作答逐小问给分。

阅卷原则：
1. 踩点给分：按评分细则的得分点给分，学生写出了对应的正确内容才给该点的分；与得分点无关的内容不额外加分。
2. 其他解法：方法与参考答案不同但正确、完整的，按同类步骤的分值给分，不因方法不同扣分。
3. 关键步骤缺失、依据不足（例如证线面垂直时没有说明"两条相交直线"）按细则扣分；显而易见的计算过程可以省略。
4. 错误传递：某步出错后，依赖这个错误结果的后续步骤一般不给分；与错误无关、独立正确的后续步骤可以给分。
5. 只有结论没有过程，或过程有实质性错误、结论碰巧正确，不能给满分。
6. 学生划掉的内容（~~……~~）不算作答。
7. 不因字迹潦草或格式不规范扣分（细则另有规定的除外）。
8. 学生作答是手写识别的转写，可能有个别识别错误。若附有原图，两者冲突时以原图为准。某处符号可疑、像是识别错误时，不要据此扣分，在 review_notes 里说明并把 needs_review 设为 true。
9. 不要根据参考答案去"脑补"学生没写清楚的内容；确实看不清、又影响给分的，按看不清处理并标记复核。
10. 区域里如果还有别的题的作答，只评本题。
11. 参考答案里步骤后面标的"……X分"就是评分标准，按【分值标注说明】理解（多数答案是累计分）。"""


def grade_question_block(q: dict) -> str:
    parts = q.get("parts") or []
    if parts:
        pl = "；".join(f"{p['label']} {p['score']}分" for p in parts)
    else:
        pl = f"（全题）{q.get('score')}分"
    blocks = [f"【题目】第{q['number']}题（满分 {q.get('score')} 分）", question_text(q)]
    if q.get("figure"):
        blocks.append(f"【图形说明】{q['figure']}")
    blocks += [f"【小问及分值】{pl}", "【参考答案（含评分标准）】", (q.get("answer") or "（无）").strip()]
    conv = q.get("score_convention")
    if conv in CONVENTION_TEXT:
        blocks += ["【分值标注说明】", CONVENTION_TEXT[conv]]
    elif "分" in (q.get("answer") or ""):
        blocks += ["【分值标注说明】", "参考答案中步骤后面的“……X分”如果逐步增大、最后等于该小问满分，就是累计分（相邻两个标注之差是这一步的分值）；否则是这一步本身的分值。"]
    rub = (q.get("rubric") or "").strip()
    if rub:
        blocks += ["【评分细则（补充）】", rub]
    elif not re.search(r"\d\s*分", q.get("answer") or ""):
        blocks += ["【评分细则】", "（参考答案没有标分值：按参考答案的关键步骤合理分配各小问分值）"]
    if q.get("notes"):
        blocks += ["【阅卷补充说明（老师）】", q["notes"].strip()]
    return "\n".join(b for b in blocks if b is not None)


def grade_student_block(q: dict, transcription: str, legibility: str, with_image: bool, step: float) -> str:
    parts = q.get("parts") or [{"label": "(全题)", "score": q.get("score")}]
    ex_parts = []
    for k, p in enumerate(parts[:2]):
        ex_parts.append({"label": p["label"], "max": p["score"], "score": "该小问得分",
                         "points": [{"item": "得分点（简述）", "max": "该点分值", "got": "实得分", "reason": "依据：学生写了……/缺少……"}],
                         "comment": "这一小问的问题"})
    example = {"parts": ex_parts, "total": "各小问得分之和", "comment": "给学生的一两句评语",
               "confidence": "high", "needs_review": False, "review_notes": ""}
    img = "原图附在后面，与转写冲突时以原图为准。" if with_image else ""
    step_txt = "整数" if float(step) == 1.0 else f"{step} 的整数倍"
    return f"""【学生作答（手写识别转写，可能有个别识别错误）】
{transcription.strip() or '（空白，未作答）'}
（识别清晰度：{legibility or '未知'}）{img}

请按阅卷原则逐小问给分。只输出 json，格式如下：
{json.dumps(example, ensure_ascii=False)}
说明：
- parts 按小问顺序全部列出，label 与【小问及分值】一致；score 为 0 到 max 之间的{step_txt}。
- points 列出你用到的每个得分点，got 为实得分，reason 简要说明依据（引用学生写的关键内容）。
- total 等于各小问 score 之和；comment 是写给学生看的简短评语。
- confidence 取 high / medium / low，表示你对分数的把握；没把握或识别存疑时 needs_review 设为 true，并在 review_notes 说明。
- 所有文字用中文；数学式用普通文字和符号书写（如 √2、∠EAD、AE⊥BC、arccos(√2/4)），不要用 LaTeX，不要出现反斜杠。"""


# ================================================================ 学生信息

STUDENT_SYSTEM = "你负责从答题纸上读出考生信息。"

STUDENT_USER = """请读出这张答题纸上的考生信息：姓名、学号或考号、班级（可能是手写，也可能在贴的条形码标签上）。没有或看不清的写空字符串，不要猜。
只输出 json，例如：{"name": "张三", "student_id": "25090315", "class": "高二3班"}"""
