# -*- coding: utf-8 -*-
"""
氨电联产综合能源系统 —— 仿真与AGC/PPO调度优化 全局配置
依据：《基于风光发电与氨电联产的储能与调峰策略》（说明书）理论参数
"""
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ============ 基础单位换算 ============
KG_PER_NM3_H2 = 0.0899          # 氢气密度 kg/Nm3
NM3_PER_KG_H2 = 1.0 / KG_PER_NM3_H2   # 11.12 Nm3/kg
H2_KG_PER_TON_NH3 = 176.0       # 每吨氨消耗氢气 kg（说明书）
COAL_PER_KWH = 0.0001229        # 每kWh电量折标准煤 t（全国平均约0.123 kgce/kWh）

# ============ 时间与规模 ============
HOURS_PER_DAY = 24
SIM_STEPS = 96                   # 一天 96 个时点（15分钟分辨率）
DT_HOURS = HOURS_PER_DAY / SIM_STEPS

# 机组规模（MW）—— 高新能源渗透率场景（消纳压力大，契合说明书应用场景）
PV_CAPACITY = 1000.0             # 光伏装机
WIND_CAPACITY = 700.0            # 风电装机
THERMAL_UNITS = np.array([300.0, 300.0, 300.0])   # 3×300MW 火电机组
THERMAL_PMIN = np.array([90.0, 90.0, 90.0])       # 最小出力
ELECTROLYZER_CAP = 400.0         # 碱性电解槽额定功率 MW
ELECTROLYZER_MIN_RATIO = 0.20    # 最小运行比（调峰能力下限）
LOAD_PEAK = 800.0                # 区域负荷峰值 MW

# ============ 电解槽制氢（说明书：6 kWh/Nm3 综合能耗，效率≈80%） ============
ELY_ENERGY_KWH_NM3 = 6.0         # 综合能耗 kWh/Nm3（含辅机）
ELY_EFFICIENCY = 0.80            # 电解效率
ELY_RAMP_RATE = 0.70             # 每15min功率调整率上限（占额定功率比例）

# ============ 电价与成本（元） ============
ELECTRICITY_PRICE = 0.25         # 平均电价 元/kWh（说明书）
H2_PRICE = 22.0                  # 氢气价值 元/kg（说明书按每吨氢2.2万元测算）
NH3_PRICE = 3100.0               # 氨价 元/t（说明书）
CO2_REDUCTION_PER_KWH = 0.000997 # 每kWh弃电制氢替代煤制氢的CO2减排 t（说明书：3.53e6t/3.54e9kWh）

# 火电燃料成本系数（阀点效应模型，ai Pi^2 + bi Pi + ci + |ei sin(fi(Pimin - Pi))|）
# 三台机组经典阀点效应系数（元/h）
THERMAL_A = np.array([0.00049, 0.00031, 0.00200])
THERMAL_B = np.array([6.60, 7.10, 7.00])
THERMAL_C = np.array([700.0, 450.0, 370.0])
THERMAL_E = np.array([300.0, 200.0, 200.0])
THERMAL_F = np.array([0.035, 0.042, 0.042])
THERMAL_RAMP_UP = np.array([60.0, 60.0, 60.0])    # 每15min爬坡上限 MW
THERMAL_RAMP_DN = np.array([60.0, 60.0, 60.0])

# ============ 弃光弃风惩罚（说明书公式5/6：分段惩罚因子 Cs） ============
# 不同时段弃电惩罚因子 元/MWh —— 午间光伏高峰惩罚高（参考：弃电惩罚≈上网电价 0.3~0.6 元/kWh）
CURTAIL_PENALTY_BY_HOUR = {
    0: 120.0, 1: 120.0, 2: 120.0, 3: 120.0, 4: 120.0, 5: 150.0,
    6: 150.0, 7: 220.0, 8: 300.0, 9: 380.0, 10: 450.0, 11: 500.0,
    12: 520.0, 13: 500.0, 14: 450.0, 15: 380.0, 16: 300.0,
    17: 220.0, 18: 180.0, 19: 150.0, 20: 140.0, 21: 130.0,
    22: 120.0, 23: 120.0,
}

# ============ 合成氨（哈勃-博世，说明书：转化率≈40%，循环反应） ============
NH3_CONVERSION = 0.40            # 单程转化率
NH3_REACTOR_TEMP = 450.0         # 反应温度 ℃
H2_CP = 14.3                     # 氢气比热容 kJ/kg·K
N2_CP = 1.04                     # 氮气比热容 kJ/kg·K
TEMP_HEAT_UP = 370.0             # 30℃→400℃ 升温
HEAT_COST_PER_GJ = 45.0          # 加热成本 元/GJ（标准煤折算）
HEATING_GJ_PER_TON_NH3 = 1.863   # 每吨氨加热原料气需能量 GJ（说明书理论计算）
NH3_PRICE = 3100.0               # 氨价 元/t（说明书：3100元/t）
H2_PRICE_GREEN = 22.0            # 绿氢价值 元/kg（弃电制氢，说明书）
H2_PRICE_GREY = 8.0              # 灰氢价值 元/kg（外购电制氢，边际价值）
# 火电调峰收益系数：电解槽减少火电深度调峰/爬坡，按减少的爬坡成本折算
THERMAL_PEAK_SHED_VALUE = 400.0  # 每MW·h电解槽调峰贡献的调峰价值 元/MWh
# 废热耦合：火电烟气/乏汽废热可满足合成氨加热需求的比例（说明书核心创新）
WASTE_HEAT_RATIO = 0.80          # 废热利用率（0~1），越高外购加热成本越低
# 副产氧价值（说明书：每吨氨产氧 2816 kg）
O2_PRICE_PER_T = 500.0           # 氧气售价 元/t

# ============ 氨库存动态模型（AmmoniaStorage） ============
# 液氨储罐：-33.4℃常压或 1MPa 加压，跨日库存管理
NH3_STORAGE_CAPACITY_T = 5000.0      # 最大储氨容量 吨（约10天日产）
NH3_START_INVENTORY_T = 0.0           # 起始库存 吨
NH3_BOILOFF_RATE_PER_DAY = 0.00002    # 自放电/泄漏 日损失率（年<1%）
NH3_TANK_ENERGY_KWH_PER_KG_DAY = 0.4  # 储罐制冷能耗 kWh/kg/日

# ============ 氨→电回电效率模型（AmmoniaRePower） ============
# 回电路径选择："sell" / "decompose_fc" / "decompose_ice" / "direct_gt" / "direct_st"
NH3_REPOWER_PATH = "sell"             # 默认化工原料外售（不计回电）
# 氨→电效率参数（各路径，详见 ammonia_storage.py）
NH3_DECOMPOSE_EFFICIENCY = 0.85       # 氨分解制氢效率
NH3_SOFC_EFFICIENCY = 0.60            # SOFC燃料电池回电效率
NH3_ICE_EFFICIENCY = 0.40             # 内燃机回电效率
NH3_DIRECT_GT_EFFICIENCY = 0.38       # 纯氨燃气轮机回电效率
NH3_DIRECT_ST_EFFICIENCY = 0.35       # 纯氨蒸汽轮机回电效率
# 回电电价（氨→电上网电价，元/kWh）
NH3_REPOWER_PRICE_PER_KWH = 0.35      # 按平段电价
NH3_REPOWER_ENABLE_CAPACITY_CREDIT = False  # 是否计入储能容量价值

# ============ 氨运行模式 ============
# 氨产品路径：控制氨是外售、储存还是回电
#   "sell"    — 外售（化工原料，不计回电）
#   "store"   — 全部储存（允许库存累积）
#   "repower" — 储存+回电
#   "hybrid"  — 混合：按 NH3_REPOWER_RATIO 比例回电 vs 外售
NH3_OPERATE_MODE = "sell"
NH3_REPOWER_RATIO = 0.0               # 混合模式下回电比例（0~1）

# ============ 场景生成（季节/地区/天气） ============
SEASON_MONTH = 6                 # 默认季节月份（6=夏季，12=冬季）
REGION = "northwest"             # 默认地区（northwest/northeast/northchina）
WEATHER_MODE = "auto"            # 天气模式（auto/晴sunny/多云cloudy/阴overcast/雨rain）
# 状态前瞻窗口：提供给智能体的未来弃电预测时长（小时）
LOOKAHEAD_HOURS = 4              # 未来4小时弃电预测（说明书AGC应基于预测调度）

# ============ 方案H：动作掩码（领域知识注入，PPO+动作掩码组合） ============
# 原理：峰段（11:00-14:00、18:00-21:00）且当前无弃电潜力时，屏蔽高功率档位，
#       直接消除"峰段高价外购电制氢"的次优行为（已观测：恒定70%运行的诱因），
#       缩小搜索空间、加速收敛。通过将非法档位 logits 置 -inf 实现。
ACTION_MASK = True               # 是否启用峰段功率档位掩码（方案H开关）
MASK_PEAK_LEVEL = 0.7            # 峰段且无弃电时屏蔽 ≥ 该功率（标幺）的档位

# ============ 方案B：自适应熵温度（MaxEnt-PPO，PPO+SAC自动温度） ============
# 将固定熵系数替换为可学习温度 α，按"当前策略熵 vs 目标熵 -ln(K)"梯度更新，
# 避免策略过早确定性化（已观测：恒定70%运行）。默认关闭，--maxent 开启。
AUTO_ENTROPY = False
ENTROPY_TARGET_COEF = 1.0        # 目标熵 = -ln(n_actions) * 该系数
ENTROPY_ALPHA_LR = 3e-4          # 温度 α 的学习率
ENTROPY_ALPHA_CLIP = (1e-3, 10.0)  # α 裁剪范围

# ============ 方案C：MPC 滚动修正（PPO+模型预测控制，评估期叠加） ============
# 原理：评估时对 PPO 建议动作做前瞻滚动优化修正（受爬坡/能量平衡硬约束），
#       消除缺电、降低外购电成本。训练无需改动，仅评估叠加。--mpc 开启。
MPC_ENABLED = False
MPC_HORIZON = 8                  # 前瞻时步数（8×15min = 2h）
MPC_RANDOM_SEEDS = [0]           # MPC 评估固定场景种子

# ============ 方案D：ES 进化超参（CMA-ES风格×PPO，独立脚本 evolution.py） ============
# 外层随机搜索 PPO 关键超参（熵系数/学习率/clip），内层短训+固定场景评估，
# 以收益为适应度选择精英。参数见 src/evolution.py。

# ============ 方案E：时序状态窗口（帧堆叠，PPO+时间相关性） ============
# 将最近 WINDOW_STEPS 个时点的状态帧拼接为输入（捕获天气持续性/爬坡时序），
# 是 LSTM 的轻量替代。1=关闭（单帧，与原版一致）；>1 开启。--window N。
WINDOW_STEPS = 1

# ============ 方案F：MAPPO 多智能体（电解槽+火电分工，CTDE） ============
# 双 Actor（电解槽11档 + 每台火电机组11档）共享 Critic。独立脚本 train_mappo.py。
# --mappo 开启后动作通道变为 4 维（电解槽 + 3台火电）。
MAPPO = False

# ============ 方案G：PER 优先经验回放（样本效率） ============
# 按 |GAE优势| 优先级采样关键时段样本（弃电高峰/峰谷切换），加速收敛。
# 注意：PPO 为 on-policy 算法，PER 会破坏其 i.i.d./on-policy 假设（理论不严格），
# 属实验性组合，需与基线对比验证。--per 开启。
PER = False
PER_ALPHA = 0.6                  # 优先级指数（0=均匀）
PER_BETA_INIT = 0.4              # 重要性采样权重指数初值
PER_BETA_FINAL = 1.0             # 重要性采样权重指数终值
PER_EPS = 1e-3                   # 优先级平滑项

# ============ PPO 超参数 ============
PPO = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "entropy_coef": 0.01,
    "vf_coef": 0.5,
    "epochs": 10,
    "batch_size": 256,
    "hidden_dim": 256,
    "n_actions": 11,             # 离散动作档位数（0/10%/.../100%功率）
    "total_timesteps": 4_000_000,
    "n_envs": 16,                # 并行环境数（4090 建议 16~32）
    "reward_scale": 1e4,         # 奖励缩放：元 -> 万元（稳定价值网络训练）
    "seed": 42,
    "lr_decay": True,            # 学习率线性衰减至 10%
    "save_interval": 500_000,    # checkpoint 保存间隔（步）
}

# ============ 数据导出 ============
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DASHBOARD_DATA_DIR = os.path.join(PROJECT_ROOT, "dashboard", "data")
