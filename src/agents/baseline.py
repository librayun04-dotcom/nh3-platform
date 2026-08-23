# -*- coding: utf-8 -*-
"""
基线策略 —— 启发式AGC调度（无强化学习）

三种基线：
  1. curtail_first : 优先消纳弃电（电解槽功率 = 弃电潜力，贪心）
  2. flat          : 电解槽恒定功率运行（无调峰意识）
  3. no_ely        : 不配置电解槽（纯弃电，参照系）
"""
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config as cfg
from src.models.grid_env import AGCEnv


def run_baseline(env: AGCEnv, mode: str = "curtail_first"):
    """在给定环境中运行基线策略，返回 summary"""
    env.reset()
    done = False
    while not done:
        hour = (env.t % cfg.SIM_STEPS) / cfg.SIM_STEPS * 24.0
        pv = env.data["pv"][env.t]
        wind = env.data["wind"][env.t]
        load = env.data["load"][env.t]
        if mode == "curtail_first":
            # 弃电潜力全部用于制氢（限电解槽容量）
            surplus = max(0.0, pv + wind - load)
            action = np.array([min(1.0, surplus / env.ely.capacity)])
        elif mode == "flat":
            # 恒定 60% 额定功率
            action = np.array([0.6])
        elif mode == "no_ely":
            action = np.array([0.0])
        else:
            raise ValueError(mode)
        _, _, done, _ = env.step(action)
    return env.summary()


if __name__ == "__main__":
    env = AGCEnv(seed=0)
    for m in ["curtail_first", "flat", "no_ely"]:
        s = run_baseline(env, m)
        print(f"[{m}] 消纳率 {s['renewable_utilization']*100:.1f}% | "
              f"弃电 {s['curtail_mwh']:.0f} MWh | 产氢 {s['h2_kg']:.0f} kg | "
              f"收益 {s['total_reward']:.0f} 元")
