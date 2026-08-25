@echo off
chcp 65001 >nul
title 氨电联产 AGC 智能调度平台
cd /d %~dp0

echo ================================================================
echo   氨电联产 AGC 智能调度平台 (FastAPI + PPO)
echo ================================================================
echo.

where python >nul 2>nul || (echo [错误] 未找到 python，请先安装并加入 PATH & pause & exit /b 1)

python -c "import fastapi, uvicorn, pandas, multipart, numpy, torch, openpyxl" 2>nul || (
    echo [提示] 正在安装运行依赖（首次约需数分钟，含 CPU 版 PyTorch 约 200 MB）...
    pip install -r requirements.txt -q
    python -c "import fastapi, uvicorn, pandas, multipart, numpy, torch, openpyxl" 2>nul || (
        echo [错误] 依赖安装失败，请检查网络或 Python 版本(建议 3.10+) & pause & exit /b 1)
)

echo 启动服务: http://127.0.0.1:8000   （关闭本窗口即停止后台引擎）
echo.
start "" http://127.0.0.1:8000
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
pause
