#!/usr/bin/env bash
# 氨电联产 AGC 智能调度平台 一键启动（Linux/Mac）
cd "$(dirname "$0")"

if ! command -v python3 &>/dev/null; then
    echo "[错误] 未找到 python3"; exit 1
fi

python3 -c "import fastapi, uvicorn, pandas, multipart, numpy, torch, openpyxl" 2>/dev/null || {
    echo "[提示] 正在安装运行依赖（首次约需数分钟）..."
    pip3 install -r requirements.txt -q
}

echo "启动服务: http://127.0.0.1:8000"
python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8000
