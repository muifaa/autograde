"""其他上传方式：试卷和答案合在一个 PDF；试卷/答案是拍照或扫描的图片（没有文字层）。"""
import shutil, sys, tempfile, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import pymupdf
from autograde import exam_setup
from autograde.imaging import imwrite, load_pages
from autograde.llm import DeepSeek, Usage
from autograde.store import Store

PORT = 8772


def run(store, files, title):
    st = store.save_settings({"api_key": "sk-fake", "base_url": f"http://127.0.0.1:{PORT}", "max_workers": 6})
    e = store.create_exam(title, 1)
    d = store.exam_dir(e["id"])
    rel = {}
    for kind, paths in files.items():
        rel[kind] = []
        for p in paths:
            shutil.copy(p, d / kind / Path(p).name)
            rel[kind].append(f"{kind}/{Path(p).name}")
    store.update_exam(e["id"], lambda x: x["files"].update(rel))
    llm = DeepSeek(st["api_key"], st["base_url"], cache_dir=str(d / "cache"), usage=Usage())
    return exam_setup.run_setup(store, e["id"], llm, st, lambda f, m: None, generate_missing=False)


def main(paper, answers):
    import uvicorn, fake_deepseek
    srv = uvicorn.Server(uvicorn.Config(fake_deepseek.app, host="127.0.0.1", port=PORT, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    while not srv.started:
        time.sleep(0.05)
    tmp = Path(tempfile.mkdtemp(prefix="ag_var_"))
    store = Store(str(tmp / "data"))

    # 1) 试卷 + 答案合成一个 PDF（常见：试卷后面附参考答案）
    comb = tmp / "试卷含答案.pdf"
    doc = pymupdf.open(paper); doc.insert_pdf(pymupdf.open(answers)); doc.save(str(comb))
    fake_deepseek.STATE["by_task"].clear()
    ex = run(store, {"paper": [comb]}, "合并 PDF")
    sol = {q["number"]: q for q in ex["questions"] if q["type"] == "解答题"}
    print("合并PDF：题目", len(ex["questions"]), "解答题", sorted(sol, key=int), "调用", dict(fake_deepseek.STATE["by_task"]))
    print("   18 答案来源", sol["18"]["answer_source"], "小问", [(p["label"], p["score"]) for p in sol["18"]["parts"]],
          "答案截图", len(sol["18"]["answer_crops"]), "日志", ex["setup_log"])
    assert len(ex["questions"]) == 21 and sorted(sol, key=int) == ["17", "18", "19", "20", "21"]
    assert all(sol[n]["answer_source"] == "key" for n in sol), "合并 PDF 里的答案应自动录入"
    assert len(sol["18"]["answer_crops"]) == 2

    # 2) 拍照/扫描的图片（没有文字层）：整份一次录入；空白页跳过
    imgs = []
    for i, img in enumerate(load_pages(paper, dpi=120)):
        p = tmp / f"paper_{i}.jpg"; imwrite(p, img); imgs.append(p)
    aimgs = []
    for i, img in enumerate(load_pages(answers, dpi=120)):
        p = tmp / f"ans_{i}.jpg"; imwrite(p, img); aimgs.append(p)
    fake_deepseek.STATE["by_task"].clear()
    ex = run(store, {"paper": imgs, "answers": aimgs}, "图片试卷")
    print("图片：调用", dict(fake_deepseek.STATE["by_task"]), "题目", [(q["number"], q["type"]) for q in ex["questions"]][:6], "…")
    print("   日志", ex["setup_log"])
    assert fake_deepseek.STATE["by_task"].get("paper") == 1, "图片试卷应整份一次录入"
    assert fake_deepseek.STATE["by_task"].get("answers_doc") == 1, "图片答案应整份一次录入"
    print("VARIANTS OK")
    srv.should_exit = True


if __name__ == "__main__":
    main(*sys.argv[1:3])
