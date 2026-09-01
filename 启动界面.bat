@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 定位 venv 里的 python(依赖都装在 .venv 里)
REM 用绝对路径，避免后面 cd 到 01_mvp 后相对路径 01_mvp\.venv 失效变成 01_mvp\01_mvp\.venv
set VENV_PY=%~dp001_mvp\.venv\Scripts\python.exe
if not exist "%VENV_PY%" (
    echo [错误] 找不到 %VENV_PY%
    echo 请先确认依赖已安装: cd 01_mvp 然后执行 .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

cd /d "%~dp001_mvp"

echo [1/2] 检查知识库是否已建立...
"%VENV_PY%" -c "from core import config; config.ensure_dirs(); from core.vectorstore import collection_count; import sys; sys.exit(0 if collection_count()>0 else 1)" 2>nul
if errorlevel 1 goto :build_index
goto :start_ui

:build_index
echo       知识库为空，正在自动建档 首次需下载/加载 bge-m3 模型 稍慢...
"%VENV_PY%" -c "from core import config; config.ensure_dirs(); from core.ingest import build_index; print(build_index())"

:start_ui

echo [2/2] 启动智能问答界面...
echo   启动完成后浏览器会自动打开 http://localhost:8501
echo   关掉本窗口即可停止程序。
echo.
"%VENV_PY%" -m streamlit run app.py

pause