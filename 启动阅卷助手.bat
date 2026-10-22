@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title 阅卷助手

rem ===== 可以改的设置 =====
rem 只在本机用：127.0.0.1    让局域网其他电脑也能访问：0.0.0.0
set "HOST=127.0.0.1"
set "PORT=8000"
set "MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple"
rem ========================

if not exist "run.py" goto :wrongdir

set "VPY=.venv\Scripts\python.exe"

rem 已有环境但用不了（比如之前没建成功），删掉重建
if exist "%VPY%" (
  "%VPY%" -c "import sys" >nul 2>&1 || rmdir /s /q .venv
)
if exist "%VPY%" goto :deps

echo 第一次运行，正在准备运行环境，大约需要几分钟……
set "PY="
for %%V in (3.12 3.13 3.11 3.10 3.9 3.14) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY goto :nopython

echo 使用 %PY% 创建运行环境……
%PY% -m venv .venv
if errorlevel 1 goto :venvfail

:deps
fc /b requirements.txt ".venv\req_installed.txt" >nul 2>&1
if not errorlevel 1 goto :run

echo 正在安装依赖……
"%VPY%" -m pip install --upgrade pip -i %MIRROR% >nul 2>&1
"%VPY%" -m pip install -r requirements.txt -i %MIRROR%
if not errorlevel 1 goto :depsok
echo 用国内镜像安装失败，改用官方源再试一次……
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :pipfail
:depsok
copy /y requirements.txt ".venv\req_installed.txt" >nul

:run
echo.
echo 正在启动，浏览器会自动打开。用完关掉这个黑色窗口就退出了。
echo.
"%VPY%" run.py --host %HOST% --port %PORT%
if errorlevel 1 goto :crashed
goto :eof

:wrongdir
echo 请把这个文件放在 autograde 文件夹里，和 run.py 放在一起，再双击。
pause
exit /b 1

:nopython
echo 没有找到可用的 Python。
echo 请到 https://www.python.org/downloads/ 下载安装 Python 3.12，
echo 安装第一页记得勾选 Add python.exe to PATH，装好后再双击本文件。
pause
exit /b 1

:venvfail
echo 创建运行环境失败，请把上面的报错截图发给开发者。
pause
exit /b 1

:pipfail
echo 安装依赖失败，请检查网络，或把上面的报错截图发给开发者。
echo 如果提示需要 Microsoft Visual C++，请改装 Python 3.12 后，删掉 .venv 文件夹再双击。
pause
exit /b 1

:crashed
echo.
echo 程序异常退出。如果提示端口被占用，说明阅卷助手已经开着了，直接在浏览器打开 http://127.0.0.1:%PORT% 即可。
echo 其他报错请把上面的内容截图发给开发者。
pause
exit /b 1
