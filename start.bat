@echo off
chcp 65001 >nul
title 氨电联产 AGC 智能调度平台
cd /d %~dp0

echo ================================================================
echo   氨电联产 AGC 智能调度平台 (FastAPI + PPO)
echo ================================================================
echo.

where python >nul 2>nul || (echo [错误] 未找到 python，请先安装并加入 PATH & pause & exit /b 1)

python -c "import fastapi, uvicorn, pandas, multipart" 2>nul || (
    echo [提示] 正在安装依赖...
    pip install -r requirements.txt -q
)

echo.
echo 启动服务: http://127.0.0.1:8000
echo.
start "" http://127.0.0.1:8000
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
pause
