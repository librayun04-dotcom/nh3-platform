# -*- coding: utf-8 -*-
"""
火电机组模型 —— 阀点效应燃料成本 + 爬坡约束

依据说明书公式 (7)(8)(9)：
- 出力约束：Pimin <= Pi <= Pimax
- 爬坡约束：-D <= dPi <= U
- 燃料成本（含阀点效应）：f1 = sum(ai*Pi^2 + bi*Pi + ci + |ei*sin(fi*(Pimin - Pi))|)
"""
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config as cfg


class ThermalPlant:
    """多机组火电厂（含阀点效应的经济调度）"""

    def __init__(self, units: np.ndarray = None):
        self.units = units if units is not None else cfg.THERMAL_UNITS
        self.n = len(self.units)
        self.pmin = np.array(cfg.THERMAL_PMIN, dtype=float)
        self.pmax = np.array(cfg.THERMAL_UNITS, dtype=float)
        self.ramp_up = np.array(cfg.THERMAL_RAMP_UP, dtype=float)
        self.ramp_dn = np.array(cfg.THERMAL_RAMP_DN, dtype=float)
        self.a = np.array(cfg.THERMAL_A, dtype=float)
        self.b = np.array(cfg.THERMAL_B, dtype=float)
        self.c = np.array(cfg.THERMAL_C, dtype=float)
        self.e = np.array(cfg.THERMAL_E, dtype=float)
        self.f = np.array(cfg.THERMAL_F, dtype=float)
        self.power = self.pmin.copy()          # 当前各机组出力
        self.cost_total = 0.0                  # 累计燃料成本（元）

    def fuel_cost(self, power: np.ndarray) -> float:
        """公式(9)：含阀点效应的燃料成本 元/h"""
        p = np.asarray(power, dtype=float)
        base = self.a * p ** 2 + self.b * p + self.c
        valve = self.e * np.abs(np.sin(self.f * (self.pmin - p)))
        return float(np.sum(base + valve))

    def step(self, target: np.ndarray, dt_h: float) -> tuple:
        """
        设定目标出力，考虑爬坡约束后执行。
        返回 (实际出力, 本时段燃料成本元, 是否越限)
        """
        target = np.asarray(target, dtype=float)
        # 爬坡约束
        p_prev = self.power.copy()
        p = np.clip(target, p_prev - self.ramp_dn * (dt_h * 4),
                    p_prev + self.ramp_up * (dt_h * 4))
        # 出力上下限约束
        p = np.clip(p, self.pmin, self.pmax)
        # 单时段燃料成本 = 元/h * 时长h
        cost = self.fuel_cost(p) * dt_h
        self.cost_total += cost
        self.power = p
        return p, cost, bool(np.any(p != target))

    def total_power(self) -> float:
        return float(np.sum(self.power))


if __name__ == "__main__":
    tp = ThermalPlant()
    p, cost, flag = tp.step(np.array([250.0, 240.0, 230.0]), 0.25)
    print(f"出力: {p}, 单时段成本: {cost:.1f} 元, 越限: {flag}")
