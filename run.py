"""启动阅卷助手：python run.py [--port 8000] [--host 127.0.0.1] [--no-browser]"""
import argparse
import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AUTOGRADE_DATA", str(ROOT / "data"))


def main():
    ap = argparse.ArgumentParser(description="阅卷助手（DeepSeek 批改解答题）")
    ap.add_argument("--host", default="127.0.0.1", help="只在本机用保持默认；要让同一局域网的其他电脑访问可改为 0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    import uvicorn
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    if not args.no_browser:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    print(f"阅卷助手已启动：{url}    数据保存在 {os.environ['AUTOGRADE_DATA']}    按 Ctrl+C 退出")
    uvicorn.run("autograde.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
