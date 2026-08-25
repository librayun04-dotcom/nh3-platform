@echo off
chcp 65001 >nul
title 打包 氨电联产 AGC 调度平台 为 exe
cd /d %~dp0

echo ================================================================
echo   将平台打包为独立 exe（PyInstaller, 首次需联网装依赖）
echo ================================================================
echo.

where python >nul 2>nul || (echo [错误] 未找到 python & pause & exit /b 1)

python -c "import PyInstaller" 2>nul || pip install pyinstaller -q
python -c "import torch, fastapi" 2>nul || (
    echo [提示] 先安装运行依赖...
    pip install -r requirements.txt -q
)

echo.
echo 开始打包（含 CPU 版推理运行时，产物约 1 GB，请耐心等待）...
python -m PyInstaller --noconfirm nh3_platform.spec
if errorlevel 1 (echo [失败] 请检查上方日志 & pause & exit /b 1)

echo.
echo ================================================================
echo   打包完成: dist\氨电联产AGC调度平台\氨电联产AGC调度平台.exe
echo   双击该 exe 即可启动平台（浏览器自动打开控制台）
echo ================================================================
pause
