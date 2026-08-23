# -*- coding: utf-8 -*-
"""
碱性电解槽制氢 + 合成氨模型（改进版）
================================================================
改进（对应原模型弱点）：
  1. 电解槽负载-效率曲线：碱性电解槽在低负载时效率显著下降
     （极化曲线简化：η(P) = η0 - k*(1 - P/Pmax)^2）
  2. 合成氨单程转化率 40%（说明书），未反应气体循环
  3. 废热耦合：合成氨加热优先利用火电烟气废热（说明书核心创新），
     废热利用率高则外购加热成本低

依据说明书：
  - 碱性电解池综合能耗 6 kWh/Nm3，额定效率约 80%
  - 每吨氨需 176 kg 氢气；单程转化率约 40%
  - 合成氨反应 N2 + 3H2 -> 2NH3，温度 400-500℃（火电烟气废热驱动）
"""
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config as cfg


class Electrolyzer:
    """碱性电解水制氢系统（可变功率负荷，含负载-效率曲线）"""

    def __init__(self, capacity_mw: float = None):
        self.capacity = capacity_mw or cfg.ELECTROLYZER_CAP
        self.min_power = self.capacity * cfg.ELECTROLYZER_MIN_RATIO
        self.max_power = self.capacity * 1.05          # 允许短时过载5%
        self.power = self.min_power                    # 当前功率 MW
        self.h2_produced_kg = 0.0                      # 累计产氢 kg
        self.h2_flow_kg = 0.0                          # 当前产氢速率 kg/h
        self.efficiency_eff = 0.0                      # 当前效率

    @staticmethod
    def efficiency(power_mw: float, capacity: float) -> float:
        """负载-效率曲线（极化曲线简化模型）
        额定(100%负载)效率 η0=0.80；低负载效率下降（欧姆损耗占比上升）
        η(P) = η0 * (1 - k*(1 - P/Pmax)^2)，k=0.15 使 20%负载时效率约 70%
        """
        if power_mw <= 0:
            return 0.0
        ratio = min(power_mw / capacity, 1.0)
        eta0 = cfg.ELY_EFFICIENCY
        k = 0.15
        return max(0.3, eta0 * (1 - k * (1 - ratio) ** 2))

    def step(self, power_mw: float, dt_h: float):
        """设定功率并结算产氢量（考虑爬坡约束在环境层处理）"""
        self.power = float(np.clip(power_mw, 0, self.max_power))
        self.efficiency_eff = self.efficiency(self.power, self.capacity)
        # 有效电解能耗：实际耗电 / 效率
        energy_kwh = self.power * 1000 * dt_h          # kWh（总耗电）
        eff = max(0.3, self.efficiency_eff)
        # 产氢：6 kWh/Nm3 是额定能耗，按效率折算
        h2_nm3 = (energy_kwh * eff) / cfg.ELY_ENERGY_KWH_NM3
        self.h2_flow_kg = h2_nm3 * cfg.KG_PER_NM3_H2 / dt_h  # kg/h
        self.h2_produced_kg += h2_nm3 * cfg.KG_PER_NM3_H2
        return self.h2_flow_kg


class AmmoniaPlant:
    """哈勃-博世合成氨系统（单程转化率40% + 未反应气体循环 + 废热耦合）

    依据说明书：
      - 单程转化率约 40%，未反应混合气（H2+N2）循环继续反应
      - 反应温度 400-500℃，利用火电烟气/乏汽废热驱动（废热耦合）
      - 净消耗：每吨氨 176 kg 氢（循环不改变净消耗，只体现工艺描述）
    """

    def __init__(self, waste_heat_ratio: float = 0.8):
        self.nh3_produced_t = 0.0
        self.waste_heat_ratio = waste_heat_ratio   # 火电废热可满足加热需求的比例

    @staticmethod
    def h2_to_nh3(h2_kg: float) -> float:
        """按每吨氨 176 kg 氢折算氨产量（吨）"""
        return h2_kg / cfg.H2_KG_PER_TON_NH3

    def step(self, h2_kg: float):
        """反应器结算：单程转化 40%，未反应气体循环
        返回 (本时段产氨, 单程消耗氢, 循环氢)
        """
        nh3_t = self.h2_to_nh3(h2_kg)
        self.nh3_produced_t += nh3_t
        # 单程转化率：每次通过反应器只有 40% 的氢转化为氨
        conversion = cfg.NH3_CONVERSION
        h2_single_pass = h2_kg / conversion          # 单程处理氢量
        h2_recycled = h2_single_pass - h2_kg         # 未反应循环氢
        return nh3_t, h2_single_pass, h2_recycled

    @staticmethod
    def heating_energy_gj(h2_kg: float) -> float:
        """将混合原料气（H2:N2=3:1，30℃→400℃）加热所需能量 GJ
        依据说明书：加热每吨氨原料气约 1.863e9 J = 1.863 GJ
        注：循环气已处于高温，仅补充新进料加热，故按净消耗计"""
        per_ton_nh3_gj = cfg.HEATING_GJ_PER_TON_NH3
        nh3_t = h2_kg / cfg.H2_KG_PER_TON_NH3
        return nh3_t * per_ton_nh3_gj

    def heating_cost(self, h2_kg: float, waste_heat_ratio: float = None) -> float:
        """加热成本（元）：总加热能量 × 废热未覆盖比例 × 热价
        废热利用比例越高（说明书：火电烟气废热驱动），外购加热成本越低"""
        whr = waste_heat_ratio if waste_heat_ratio is not None else self.waste_heat_ratio
        return self.heating_energy_gj(h2_kg) * (1.0 - whr) * cfg.HEAT_COST_PER_GJ


def oxygen_byproduct(h2_kg: float) -> float:
    """电解水副产氧气 kg（每吨氨产氧约 2816 kg）"""
    nh3_t = h2_kg / cfg.H2_KG_PER_TON_NH3
    return nh3_t * 2816.0


def oxygen_value(h2_kg: float, price_per_t: float = 500.0) -> float:
    """副产氧价值（元）：氧气售价约 500 元/t"""
    return oxygen_byproduct(h2_kg) / 1000.0 * price_per_t


if __name__ == "__main__":
    ely = Electrolyzer()
    # 验证效率曲线：不同负载下产氢量
    for ratio in [0.2, 0.5, 0.8, 1.0]:
        p = ely.capacity * ratio
        h2 = ely.step(p, 0.25)
        print(f"负载{ratio*100:.0f}%: 功率{p:.0f}MW 效率{ely.efficiency_eff*100:.1f}% 产氢{ely.h2_flow_kg:.0f} kg/h")
    nh3 = AmmoniaPlant()
    h2_kg = 1000.0
    nh3_t, h2_pass, h2_recycle = nh3.step(h2_kg)
    print(f"\n1000kg氢 -> 氨 {nh3_t:.1f} t")
    print(f"单程转化率{cfg.NH3_CONVERSION*100:.0f}%：单程处理氢 {h2_pass:.0f} kg，循环氢 {h2_recycle:.0f} kg")
    print(f"加热总能耗 {nh3.heating_energy_gj(h2_kg):.1f} GJ")
    print(f"废热利用率80%时外购加热成本 {nh3.heating_cost(h2_kg):.0f} 元")
    print(f"副产氧 {oxygen_byproduct(h2_kg):.0f} kg，价值 {oxygen_value(h2_kg):.0f} 元")
