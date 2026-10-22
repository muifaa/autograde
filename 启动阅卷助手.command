#!/bin/bash
# 双击启动阅卷助手（Mac）
cd "$(dirname "$0")" || exit 1

# ===== 可以改的设置 =====
HOST="127.0.0.1"      # 让局域网其他电脑访问改成 0.0.0.0
PORT="8000"
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
# ========================

pause_exit() { echo; read -n 1 -s -r -p "按任意键关闭窗口……"; exit 1; }

[ -f run.py ] || { echo "请把这个文件放在 autograde 文件夹里，和 run.py 放在一起。"; pause_exit; }

VPY=".venv/bin/python"
if [ -x "$VPY" ] && ! "$VPY" -c "import sys" >/dev/null 2>&1; then rm -rf .venv; fi

if [ ! -x "$VPY" ]; then
  echo "第一次运行，正在准备运行环境，大约需要几分钟……"
  PY=""
  for c in python3.12 python3.13 python3.11 python3.10 python3.9 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >/dev/null 2>&1; then
      PY="$c"; break
    fi
  done
  [ -n "$PY" ] || { echo "没有找到 Python 3.9 以上版本，请到 https://www.python.org/downloads/ 安装。"; pause_exit; }
  "$PY" -m venv .venv || { echo "创建运行环境失败。"; pause_exit; }
fi

if ! cmp -s requirements.txt .venv/req_installed.txt; then
  echo "正在安装依赖……"
  "$VPY" -m pip install --upgrade pip -i "$MIRROR" >/dev/null 2>&1
  "$VPY" -m pip install -r requirements.txt -i "$MIRROR" || "$VPY" -m pip install -r requirements.txt || { echo "安装依赖失败，请检查网络。"; pause_exit; }
  cp requirements.txt .venv/req_installed.txt
fi

echo
echo "正在启动，浏览器会自动打开。用完关掉这个窗口就退出了。"
echo
"$VPY" run.py --host "$HOST" --port "$PORT" || { echo "程序异常退出。如果提示端口被占用，直接打开 http://127.0.0.1:$PORT 即可。"; pause_exit; }
