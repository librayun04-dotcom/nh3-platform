# 氨电联产 AGC 智能调度平台（nh3-platform）

基于自研离散 PPO 深度强化学习的风光-火电-氨电联产综合能源系统 AGC 智能调度平台。
上传风光负荷曲线即可获得未来一日 96 个时步（15 分钟分辨率）的电解槽功率调度指令，
并支持多模型对比、在线训练与火电 AGC 实时滚动仿真。

> 主论文与完整实验见主仓库：https://github.com/librayun04-dotcom/nh3

## 功能总览

| 页面 | 功能 |
|------|------|
| 推理预测 | 4 个内置测试案例一键运行；或上传 CSV/Excel 曲线（96 行 15 分钟值，或 24 行小时值自动插值），默认使用本地已训练模型推理一日调度指令 |
| 多模型对比 | 同一曲线对磁盘上全部 checkpoint 批量推理，收益/消纳率/产量横向对比 |
| 在线训练 | 页面配置步数/并行环境数/种子启动后台训练，SSE 每秒推送收敛曲线，完成后新模型自动注册为默认候选 |
| 实时仿真 | WebSocket 驱动的火电 AGC 滚动决策：每 15 分钟下发一次电解槽功率指令，时钟滚动、功率平衡实时追加、支持暂停/单步/重置 |

系统自动选用本地步数最多的 checkpoint 作为默认模型（当前随仓库附带 100 万步训练成果）。

## 快速开始

```bash
# 1. 安装依赖（Python >= 3.10）
pip install -r requirements.txt

# 2. 启动平台
#    Windows: 双击 start.bat，或：
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000

# 3. 浏览器打开 http://127.0.0.1:8000
```

有 NVIDIA GPU 时自动启用 CUDA 加速推理与训练。

## 目录结构

```
nh3-platform/
├── config.py                  # 全局配置（装机参数/经济参数/PPO超参）
├── requirements.txt           # Python 依赖
├── start.bat / start.sh       # 一键启动脚本
├── gen_samples.py             # 示例曲线生成脚本
├── samples/                   # 内置示例曲线（夏季晴日/冬季雨雪）
├── server/
│   ├── main.py                # FastAPI 后端（模型管理/推理/训练/仿真 API）
│   └── web/index.html         # 前端单页应用（ECharts 可视化）
├── src/
│   ├── agents/
│   │   ├── ppo.py             # 自研离散 PPO（Actor/Critic 分离网络）
│   │   └── baseline.py        # 无电解槽/恒功率/贪心消纳基线
│   └── models/
│       ├── grid_env.py        # AGC 调度环境（12维状态/11档动作/综合效益奖励）
│       ├── vector_env.py      # 向量化并行环境（在线训练用）
│       ├── electrolyzer.py    # 碱性电解槽极化曲线 + 哈勃-博世合成氨
│       ├── thermal.py         # 火电机组（阀点效应燃料成本）
│       ├── renewable.py       # 季节x地区x天气风光出力模型
│       └── ammonia_storage.py # 氨库存动态与三条产品路径经济模型
└── output/data/ppo_agent.pt   # 已训练 PPO 模型（100万步，开箱即用）
```

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/models` | 列出可用 checkpoint 与计算设备 |
| GET | `/api/cases` | 内置测试案例列表 |
| POST | `/api/run_case` | 内置案例一键评估（表单 `case_id`） |
| POST | `/api/predict` | 上传曲线推理（表单 `file`，可选 `model`） |
| POST | `/api/compare` | 多模型批量对比（表单 `file`, `models`） |
| POST | `/api/train/start` / `/stop` | 启动/停止后台训练 |
| GET | `/api/train/stream` | SSE 训练进度流 |
| WS | `/ws/simulate` | 实时 AGC 滚动仿真（start/pause/step/reset） |

## MDP 建模要点

- **状态（12 维）**：光伏/风电/负荷标幺、时刻周期编码、电解槽与火电出力、弃电惩罚因子、分时电价、当前弃电潜力、未来 4h 弃电预测、火电爬坡余量
- **动作（11 档）**：电解槽功率 0%~100%，档位式贴合真实 AGC 下发模式
- **奖励**：氢能价值（绿氢 22 元/kg 分级）+ 氨价值 + 副产氧 + 火电调峰收益 − 火电燃料成本 − 外购电 − 弃电惩罚 − 缺电惩罚 − 制氨加热成本（元 → 万元缩放）

## 训练参考结果（RTX 4090）

| 指标 | 数值 |
|------|------|
| 单场景综合收益 | +110.1 万元/日 |
| 新能源消纳率 | 96.2%（贪心基线 75.7%） |
| 12 场景平均收益 | +84.1±62.0 万元/日 |
| 日产氢 / 日产氨 | 95.2 t / 541 t |
