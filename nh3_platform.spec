# -*- mode: python ; coding: utf-8 -*-
# ============================================================
# 氨电联产 AGC 调度平台 — PyInstaller 打包配置
#
# 用法（在仓库根目录）:
#   pip install pyinstaller
#   pyinstaller --noconfirm nh3_platform.spec
#
# 产物: dist/氨电联产AGC调度平台/氨电联产AGC调度平台.exe
# 说明: 内含 CPU 版推理运行时，体积约 1 GB 属正常；首次解包到
#       %TEMP% 需数秒。双击 exe 即启动，浏览器自动打开控制台。
# ============================================================

block_cipher = None

datas = [
    ("config.py", "."),
    ("src", "src"),
    ("server/web", "server/web"),
    ("output/data", "output/data"),
    ("samples", "samples"),
]

hiddenimports = [
    # uvicorn 运行时按字符串动态导入的模块必须显式声明
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # 项目内动态导入
    "config",
    "src",
    "src.models.grid_env",
    "src.agents.ppo",
    # 三方库子模块
    "torch",
    "pandas",
    "openpyxl",
    "multipart",
    "anyio._backends._asyncio",
]

a = Analysis(
    ["server\\main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "jupyter"],
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="氨电联产AGC调度平台",
    console=True,          # 保留控制台便于查看服务日志
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    name="氨电联产AGC调度平台",
    upx=False,
)
