# -*- coding: utf-8 -*-
"""
风光出力模型 —— 季节 / 天气 / 地区差异化
================================================================
改进（对应原模型弱点）：
  1. 季节因子：冬夏日照时长/强度差异（说明书应用场景：西北/华北/东北）
  2. 天气类型：晴/多云/阴/雨随机切换（天气系统持续性）
  3. 地区参数：可配置区域（西北=风强光强、东北=冬季风强、华北=光为主）
  4. 负荷曲线：冬夏用电差异（冬季取暖负荷高）

模型依据：
- 光伏：晴空模型(日出日落) × 季节日长 × 天气衰减
- 风电：威布尔风速 × 季节风速差异 × 功率曲线
- 负荷：基础双峰 + 季节偏移
"""
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config as cfg

# 地区配置：光伏/风电资源系数、负荷特征
REGIONS = {
    "northwest": {"pv_scale": 1.15, "wind_scale": 1.25, "load_winter_bias": 1.05, "label": "西北"},
    "northeast": {"pv_scale": 0.90, "wind_scale": 1.40, "load_winter_bias": 1.15, "label": "东北"},
    "northchina": {"pv_scale": 1.00, "wind_scale": 0.90, "load_winter_bias": 1.10, "label": "华北"},
}

# 季节：日长/强度缩放（month: 1-12）
# 日长因子（日照小时比例），夏至最长
SEASON_DAYLENGTH = {1: 0.72, 2: 0.80, 3: 0.92, 4: 1.02, 5: 1.08, 6: 1.10,
                    7: 1.09, 8: 1.03, 9: 0.94, 10: 0.84, 11: 0.75, 12: 0.70}
SEASON_WIND = {1: 1.30, 2: 1.25, 3: 1.10, 4: 0.95, 5: 0.85, 6: 0.75,
               7: 0.72, 8: 0.78, 9: 0.90, 10: 1.05, 11: 1.18, 12: 1.28}

# 天气类型：对光伏的衰减系数（晴/多云/阴/雨）
WEATHER_PV = {"sunny": 1.00, "cloudy": 0.65, "overcast": 0.35, "rain": 0.20}
WEATHER_WIND = {"sunny": 0.90, "cloudy": 1.00, "overcast": 1.10, "rain": 1.20}


def _weather_sequence(days: int, steps_per_day: int, seed: int) -> np.ndarray:
    """生成天气类型序列（0晴/1多云/2阴/3雨），天气系统持续6~18小时"""
    rng = np.random.default_rng(seed + 500)
    n = days * steps_per_day
    types = ["sunny", "cloudy", "overcast", "rain"]
    # 初始天气 + 马尔可夫切换（保持持续性）
    seq = []
    cur = rng.choice(4, p=[0.45, 0.30, 0.15, 0.10])
    t = 0
    while t < n:
        dur = rng.integers(6, 19) * (steps_per_day // 24)   # 6~18小时
        dur = max(1, dur)
        seq.extend([cur] * dur)
        t += dur
        # 转移概率（晴->多云最常见）
        trans = np.array([[0.55, 0.30, 0.10, 0.05],
                          [0.35, 0.35, 0.20, 0.10],
                          [0.30, 0.30, 0.25, 0.15],
                          [0.25, 0.30, 0.25, 0.20]])
        cur = rng.choice(4, p=trans[cur])
    return np.array([types[i] for i in seq[:n]])


def pv_output(days: int = 1, seed: int = 0, season_month: int = 6,
              region: str = "northwest", weather: str = "auto") -> np.ndarray:
    """光伏出力（标幺值 0~1），含季节/天气/地区差异"""
    rng = np.random.default_rng(seed)
    steps_per_day = cfg.SIM_STEPS
    n = days * steps_per_day
    t = np.arange(n) / steps_per_day
    t_day = (t % 1.0) * 24.0
    reg = REGIONS.get(region, REGIONS["northwest"])

    # 日出日落随季节变化（夏季 5:00-19:00，冬季 7:30-16:30）
    daylen = SEASON_DAYLENGTH.get(season_month, 0.9)
    sunrise = 12.0 - 6.0 * daylen
    sunset = 12.0 + 6.0 * daylen
    clear = np.clip(np.sin(np.pi * (t_day - sunrise) / max(1e-6, (sunset - sunrise))), 0, 1)

    # 天气衰减
    if weather == "auto":
        wtypes = _weather_sequence(days, steps_per_day, seed)
        wfac = np.array([WEATHER_PV[w] for w in wtypes], dtype=float)
    else:
        wfac = np.full(n, WEATHER_PV.get(weather, 1.0))

    # 云量慢变 + 随机扰动
    cloud = 0.85 + 0.15 * np.sin(2 * np.pi * t / (10.0 + rng.uniform(0, 4))) \
                 + rng.normal(0, 0.05, n)
    cloud = np.clip(cloud, 0.4, 1.05)

    # 季节强度：冬季太阳高度角低 -> 峰值强度降
    season_peak = 0.75 + 0.25 * daylen
    return np.clip(clear * wfac * cloud * season_peak * reg["pv_scale"], 0, 1)


def wind_output(days: int = 1, seed: int = 1, season_month: int = 6,
                region: str = "northwest", weather: str = "auto") -> np.ndarray:
    """风电出力（标幺值 0~1），含季节/天气/地区差异"""
    rng = np.random.default_rng(seed)
    steps_per_day = cfg.SIM_STEPS
    n = days * steps_per_day
    t = np.arange(n) / steps_per_day
    t_day = (t % 1.0) * 24.0
    reg = REGIONS.get(region, REGIONS["northwest"])

    # 季节风速差异（冬季风大）
    season_w = SEASON_WIND.get(season_month, 1.0)

    # 天气对风的影响（阴雨天风通常更大）
    if weather == "auto":
        wtypes = _weather_sequence(days, steps_per_day, seed)
        wfac = np.array([WEATHER_WIND[w] for w in wtypes], dtype=float)
    else:
        wfac = np.full(n, WEATHER_WIND.get(weather, 1.0))

    # 威布尔风速 + 日夜差异
    v_base = rng.weibull(2.0, n) * 7.5 * season_w * reg["wind_scale"]
    diurnal = 1.0 + 0.15 * np.cos(2 * np.pi * (t_day - 4.0) / 24.0)
    v = v_base * diurnal * wfac
    # 平滑（风团持续性）
    kernel = np.ones(6) / 6.0
    v = np.convolve(v, kernel, mode="same")
    v = np.clip(v, 0, 25.0)
    # 风机功率曲线
    v_in, v_rated, v_out = 3.0, 12.0, 25.0
    p = np.zeros_like(v)
    p[v >= v_in] = (v[v >= v_in] ** 3) / (v_rated ** 3)
    p[v >= v_rated] = 1.0
    p[v >= v_out] = 0.0
    return np.clip(p, 0, 1)


def load_curve(days: int = 1, seed: int = 2, season_month: int = 6,
               region: str = "northwest") -> np.ndarray:
    """负荷曲线（标幺值 0~1），早晚双峰 + 冬季取暖偏置"""
    rng = np.random.default_rng(seed)
    steps_per_day = cfg.SIM_STEPS
    n = days * steps_per_day
    t_day = (np.arange(n) % steps_per_day) / steps_per_day * 24.0
    reg = REGIONS.get(region, REGIONS["northwest"])
    # 冬季取暖：冬季负荷整体抬升
    winter_bias = 1.0 + 0.08 * (SEASON_DAYLENGTH.get(season_month, 0.9) < 0.9)
    base = (0.62
            + 0.16 * np.exp(-((t_day - 9.0) ** 2) / 6.0)
            + 0.20 * np.exp(-((t_day - 19.5) ** 2) / 8.0))
    noise = 1.0 + rng.normal(0, 0.015, n)
    return np.clip(base * noise * winter_bias * reg["load_winter_bias"], 0.35, 1.15)


def gen_day_data(seed: int = 0, days: int = 1, season_month: int = 6,
                 region: str = "northwest", weather: str = "auto"):
    """生成一天的完整出力数据（MW 绝对量）"""
    pv_pu = pv_output(days, seed, season_month, region, weather)
    wind_pu = wind_output(days, seed + 100, season_month, region, weather)
    load_pu = load_curve(days, seed + 200, season_month, region)
    pv = pv_pu * cfg.PV_CAPACITY
    wind = wind_pu * cfg.WIND_CAPACITY
    load = load_pu * cfg.LOAD_PEAK
    return {"pv": pv, "wind": wind, "load": load,
            "pv_pu": pv_pu, "wind_pu": wind_pu, "load_pu": load_pu}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.font_setup import set_plot_font
    set_plot_font()
    # 对比：夏/冬、晴/雨、西北/东北
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    cases = [("夏季·西北·晴", 6, "northwest", "sunny"),
             ("冬季·西北·多云", 12, "northwest", "cloudy"),
             ("夏季·东北·晴", 6, "northeast", "sunny"),
             ("冬季·东北·阴雨", 12, "northeast", "rain")]
    for ax, (title, mon, reg, w) in zip(axes.ravel(), cases):
        d = gen_day_data(season_month=mon, region=reg, weather=w)
        t = np.arange(cfg.SIM_STEPS) / cfg.SIM_STEPS * 24
        ax.plot(t, d["pv"], label="光伏", color="#f5b942")
        ax.plot(t, d["wind"], label="风电", color="#5aa9e6")
        ax.plot(t, d["load"], label="负荷", color="k")
        ax.set_title(f"{title}（{REGIONS[reg]['label']}）")
        ax.set_xlabel("小时"); ax.set_ylabel("MW")
        ax.legend(fontsize=14); ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "figures"), exist_ok=True)
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, "figures", "renewable_season_weather.png"), dpi=150)
    print("已生成 output/figures/renewable_season_weather.png")
