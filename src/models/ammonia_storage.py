# -*- coding: utf-8 -*-
"""
氨库存动态模型 + 氨→电回电效率模型
=============================================
氨库存动态
  - 液氨储罐(常压-33.4℃ 或 1MPa加压)：日累计、储存损失(自放电+泄漏)、储罐能耗
  - 支持跨日库存滚动（seasonal storage）
  - 三条互斥路径：氨外售(化工原料)、氨分解制氢+回电、氨直接燃烧回电

回电效率模型
  - 路径A（化工原料）：氨直接外售，不计回电效率
  - 路径B（氨分解→H₂→燃料电池/燃机回电）：分解效率+发电效率
  - 路径C（氨直接燃烧→燃机/锅炉回电）：纯氨燃烧效率

依据：
  - 氨低位热值 18.6 MJ/kg
  - 液氨体积能量密度 ~12.7 GJ/m³（vs 液氢 ~8.5~10.1 GJ/m³）
  - SOFC回电效率 ~60%，内燃机/燃气轮机 ~35%~45%
  - 氨分解制氢：分解效率约 80%~90%（催化分解）
"""

import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config as cfg


class AmmoniaStorage:
    """液氨储罐动态模型（跨日库存管理 + 储存损耗 + 储罐能耗）

    核心特性：
      - 自放电趋近于零：满罐年损失 <1%（约 0.002%/日）
      - 储罐能耗：低温常压储罐需制冷维持 -33.4℃，能耗约 0.3~0.5 kWh/kg-NH₃/日
      - 支持最大库存容量上限
      - 支持跨日库存滚动（初始化可设起始库存）
    """

    def __init__(self, capacity_t: float = 5000.0, start_inventory_t: float = 0.0):
        """
        Args:
            capacity_t: 最大储氨容量，吨（默认 5000 t，约 10 天日产）
            start_inventory_t: 起始库存（吨）
        """
        self.capacity_t = capacity_t
        self.inventory_t = start_inventory_t      # 当前库存 t
        self.total_produced_t = 0.0               # 累计产氨
        self.total_sold_t = 0.0                    # 累计外售
        self.total_repower_t = 0.0                 # 累计回电消耗
        self.total_boiloff_t = 0.0                 # 累计自然损失
        self.total_tank_energy_kwh = 0.0           # 累计储罐能耗

    def step(self, nh3_produced_t: float, dt_days: float = 1.0,
             operate_mode: str = "sell",
             repower_ratio: float = 0.0) -> dict:
        """单日库存操作

        Args:
            nh3_produced_t: 本日产氨量（吨）
            dt_days: 步长时间（日，默认 1 日）
            operate_mode: 运行模式
                "sell"      — 氨全部外售（化工原料路径，不计回电）
                "repower"   — 氨全部用于回电（储+发电）
                "hybrid"    — 按 repower_ratio 比例分配回电 vs 外售
            repower_ratio: 回电比例（仅 hybrid 模式有效，0~1）

        Returns:
            dict: {
                "inventory_end": 期末库存 t,
                "sold_t": 本日外售 t,
                "repower_t": 本日回电消耗 t,
                "boiloff_t": 本日自放电损失 t,
                "tank_energy_kwh": 本日储罐能耗 kWh,
                "overflow_t": 超出容量溢出（无法储存/需弃氨）t,
            }
        """
        # 1. 产氨入库
        self.total_produced_t += nh3_produced_t
        available_t = self.inventory_t + nh3_produced_t

        # 2. 自放电/泄漏损失（液氨年损失 <1%，日损失约 0.002%）
        boiloff_rate = 0.00002  # 每日约 0.002%
        boiloff_t = available_t * boiloff_rate * dt_days
        self.total_boiloff_t += boiloff_t
        available_t -= boiloff_t

        # 3. 储罐能耗（维持 -33.4℃ 制冷，约 0.4 kWh/kg-NH₃/日）
        # 实际能耗与储罐规模/绝热/环境温度相关，此处取典型值
        tank_energy_per_kg_per_day = 0.4  # kWh/kg-氨/日
        tank_energy_kwh = available_t * 1000 * tank_energy_per_kg_per_day * dt_days
        self.total_tank_energy_kwh += tank_energy_kwh

        # 4. 路径分配（三条互斥路径）
        sold_t = 0.0
        repower_t = 0.0
        overflow_t = 0.0

        if operate_mode == "sell":
            # 路径A：氨全部外售，仅扣除不可售部分（损耗），不留库存（简化模型）
            sold_t = available_t
            self.total_sold_t += sold_t
            available_t = 0.0

        elif operate_mode == "repower":
            # 路径B/C：氨全部储存并用于回电，不外售
            # 限制不超过容量
            if available_t > self.capacity_t:
                overflow_t = available_t - self.capacity_t
                available_t = self.capacity_t
            self.inventory_t = available_t
            repower_t = nh3_produced_t  # 本日产氨量用于回电（库存消耗单独处理）

        elif operate_mode == "hybrid":
            # 混合模式：按比例分配
            repower_t = nh3_produced_t * repower_ratio
            surplus_for_sale = available_t - repower_t
            if surplus_for_sale < 0:
                repower_t = available_t
                surplus_for_sale = 0.0
            sold_t = surplus_for_sale
            self.total_sold_t += sold_t
            # 回电部分消耗库存
            available_t -= repower_t + sold_t
            if available_t > self.capacity_t:
                overflow_t = available_t - self.capacity_t
                available_t = self.capacity_t
            self.inventory_t = available_t

        # 5. 回电消耗累计
        if operate_mode in ("repower", "hybrid"):
            # 实际回电时从库存取氨，此处按"本日用于回电的氨"记录
            # 在 repower 模式下，本日产氨全部进入库存等待回电；
            # 回电时从库存扣除，跨日循环
            self.total_repower_t += repower_t

        return {
            "inventory_end": self.inventory_t,
            "sold_t": sold_t,
            "repower_t": repower_t,
            "boiloff_t": boiloff_t,
            "tank_energy_kwh": tank_energy_kwh,
            "overflow_t": overflow_t,
        }

    def summary(self) -> dict:
        return {
            "capacity_t": self.capacity_t,
            "inventory_t": self.inventory_t,
            "total_produced_t": self.total_produced_t,
            "total_sold_t": self.total_sold_t,
            "total_repower_t": self.total_repower_t,
            "total_boiloff_t": self.total_boiloff_t,
            "total_tank_energy_kwh": self.total_tank_energy_kwh,
        }

    def reset(self, start_inventory_t: float = 0.0):
        self.inventory_t = start_inventory_t
        self.total_produced_t = 0.0
        self.total_sold_t = 0.0
        self.total_repower_t = 0.0
        self.total_boiloff_t = 0.0
        self.total_tank_energy_kwh = 0.0


class AmmoniaRePower:
    """氨→电回电效率模型（氨作为储能介质的能量释放环节）

    三条路径的能量效率：
      - 路径A（化工原料外售）：不计回电效率，以氨市场价直接计值
      - 路径B（氨分解→氢气→燃料电池/燃机）：
          氨分解制氢效率：80%~90%（催化分解）
          SOFC回电效率：60%（最高）
          综合效率：0.85 × 0.60 ≈ 51%（理论最高）
          —或— 内燃机/燃气轮机：35%~45%
          综合效率：0.85 × 0.40 ≈ 34%（典型）
      - 路径C（氨直接燃烧→蒸汽轮机/燃气轮机）：
          纯氨燃烧回电效率：35%~40%（与天然气掺混可略高）
          综合效率：约 35%~40%（无分解损耗）

    氨物理常数：
      - 氨低位热值 (LHV)：18.6 MJ/kg = 5.167 kWh/kg
      - 液氨密度：0.682 kg/L（-33.4℃）
    """

    def __init__(self, repower_path: str = "sell"):
        """
        Args:
            repower_path: 回电路径选择
                "sell"      — 化工原料外售，不计回电（默认）
                "decompose_fc" — 氨分解→H₂→燃料电池回电
                "decompose_ice" — 氨分解→H₂→内燃机回电
                "direct_gt"  — 氨直接燃烧→燃气轮机回电
                "direct_st"  — 氨直接燃烧→蒸汽轮机回电
        """
        self.repower_path = repower_path

    @property
    def round_trip_efficiency(self) -> float:
        """电→氨→电全程往返效率"""
        # 电→氨：电解效率80% × 合成氨循环效率约90% ≈ 72%
        power_to_nh3 = 0.80 * 0.90
        # 氨→电：取决于路径
        nh3_to_power = self.repower_efficiency
        return power_to_nh3 * nh3_to_power

    @property
    def repower_efficiency(self) -> float:
        """氨→电效率（仅回电环节，不含电→氨）"""
        eff_map = {
            "sell": 0.0,              # 外售不计回电
            "decompose_fc": 0.51,     # 分解0.85 × SOFC 0.60
            "decompose_ice": 0.34,    # 分解0.85 × 内燃机0.40
            "direct_gt": 0.38,        # 纯氨燃气轮机
            "direct_st": 0.35,        # 纯氨蒸汽轮机
        }
        return eff_map.get(self.repower_path, 0.0)

    @staticmethod
    def nh3_lhv_kwh_per_kg() -> float:
        """氨低位热值 kWh/kg"""
        return 18.6 / 3.6  # 18.6 MJ/kg ÷ 3.6 MJ/kWh ≈ 5.167 kWh/kg

    def repower_value(self, nh3_t: float, electricity_price_per_kwh: float = 0.35,
                      include_capacity_credit: bool = False) -> dict:
        """计算氨回电的经济价值

        Args:
            nh3_t: 用于回电的氨量（吨）
            electricity_price_per_kwh: 回电的电价 元/kWh（默认 0.35 元/kWh = 350 元/MWh）
            include_capacity_credit: 是否计入储能容量价值（额外补偿）

        Returns:
            dict: {
                "power_output_kwh": 回电量 kWh,
                "power_value": 回电价值 元,
                "nh3_consumed_kg": 消耗氨量 kg,
                "efficiency": 回电效率,
            }
        """
        if self.repower_path == "sell":
            return {"power_output_kwh": 0.0, "power_value": 0.0,
                    "nh3_consumed_kg": nh3_t * 1000, "efficiency": 0.0}

        nh3_kg = nh3_t * 1000
        # 氨的总化学能 kWh
        total_energy_kwh = nh3_kg * self.nh3_lhv_kwh_per_kg()
        # 回电输出
        power_output_kwh = total_energy_kwh * self.repower_efficiency
        power_value = power_output_kwh * electricity_price_per_kwh

        # 储能容量价值（可选）：氨储能为电网提供了长周期储能容量，
        # 按储能容量市场补偿计算（此处简化为回电价值的 10%）
        capacity_credit = power_value * 0.10 if include_capacity_credit else 0.0

        return {
            "power_output_kwh": power_output_kwh,
            "power_value": power_value + capacity_credit,
            "nh3_consumed_kg": nh3_kg,
            "efficiency": self.repower_efficiency,
        }

    @staticmethod
    def nh3_value_as_chemical(nh3_t: float, nh3_price_per_t: float = 3100.0) -> float:
        """氨作为化工原料的价值（路径A）"""
        return nh3_t * nh3_price_per_t


# ===== 工具函数 =====
def nh3_energy_content(nh3_t: float) -> dict:
    """计算氨的化学能含量"""
    mj_per_kg = 18.6
    kwh_per_kg = mj_per_kg / 3.6
    total_mj = nh3_t * 1000 * mj_per_kg
    total_kwh = nh3_t * 1000 * kwh_per_kg
    return {
        "energy_mj": total_mj,
        "energy_kwh": total_kwh,
        "energy_mwh_th": total_kwh / 1000,
        "energy_gwh_th": total_kwh / 1e6,
    }


if __name__ == "__main__":
    # 快速验证
    storage = AmmoniaStorage(capacity_t=5000.0, start_inventory_t=0.0)

    # 模拟 10 天日产 541 t 氨，混合模式 50% 回电
    for day in range(10):
        res = storage.step(nh3_produced_t=541.0, operate_mode="hybrid", repower_ratio=0.5)
        print(f"Day {day+1}: 产氨541t → 期末库存{res['inventory_end']:.1f}t "
              f"外售{res['sold_t']:.1f}t 回电{res['repower_t']:.1f}t "
              f"自放电{res['boiloff_t']:.3f}t 储罐能耗{res['tank_energy_kwh']:.0f}kWh")

    print(f"\n累计: 产{storage.total_produced_t:.0f}t 售{storage.total_sold_t:.0f}t "
          f"回电{storage.total_repower_t:.0f}t 自放电{storage.total_boiloff_t:.2f}t")

    # 验证回电效率
    for path in ["sell", "decompose_fc", "decompose_ice", "direct_gt", "direct_st"]:
        rp = AmmoniaRePower(repower_path=path)
        val = rp.repower_value(1.0)  # 1吨氨回电
        print(f"\n路径 {path}: 回电效率 {rp.repower_efficiency:.1%} "
              f"全程效率 {rp.round_trip_efficiency:.1%} "
              f"回电量 {val['power_output_kwh']:.0f} kWh "
              f"价值 {val['power_value']:.0f} 元")

    # 氨化学能
    eng = nh3_energy_content(541.0)
    print(f"\n日产 541 t 氨化学能: {eng['energy_mwh_th']:.0f} MWh_th = {eng['energy_gwh_th']:.2f} GWh_th")
