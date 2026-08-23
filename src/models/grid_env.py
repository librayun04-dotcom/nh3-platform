# -*- coding: utf-8 -*-
"""
AGC 电网调度环境（Gymnasium 风格，无外部依赖）
================================================================
落地说明书公式：
  (4) 目标函数：min f1(火电成本) + f2(光伏成本) - f3(电解水消纳减少成本)
  (5) 弃光惩罚：Cs * (Pt - P)  （Pt风光发电功率，P实际接入功率）
  (6) Cs 分段惩罚因子（按时段）
  (7) 火电出力约束：Pimin <= Pi <= Pimax
  (8) 火电爬坡约束：-D <= dPi/dt <= U
  (9) 阀点效应燃料成本：f1 = sum(ai*Pi^2+bi*Pi+ci+|ei*sin(fi*(Pimin-Pi))|)
  (10)电解水系统降低的成本（消纳收益 + 减少火电调峰）

调度逻辑：
  1. 智能体（AGC）决策：电解槽功率设定值 ely_target（用电端调峰）
  2. 新能源优先上网，余电优先供给电解槽，超出部分弃电（按 Cs 惩罚）
  3. 剩余缺口由火电按阀点效应经济调度补足（含爬坡/出力约束）
  4. 奖励 = 氢能价值 - 火电燃料成本 - 弃电惩罚 - 外购电价成本
"""
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config as cfg
from src.models.renewable import gen_day_data
from src.models.electrolyzer import Electrolyzer, AmmoniaPlant, oxygen_byproduct
from src.models.thermal import ThermalPlant
from src.models.ammonia_storage import AmmoniaStorage, AmmoniaRePower, nh3_energy_content


class AGCEnv:
    """AGC 调度环境：动作=电解槽功率（标幺 0~1），状态=调度态势"""

    def __init__(self, seed: int = 0, days: int = 1):
        self.seed = seed
        self.days = days
        self.ely = Electrolyzer()
        self.thermal = ThermalPlant()
        self.ammonia = AmmoniaPlant()
        self.nh3_storage = AmmoniaStorage(
            capacity_t=cfg.NH3_STORAGE_CAPACITY_T,
            start_inventory_t=cfg.NH3_START_INVENTORY_T
        )
        self.nh3_repower = AmmoniaRePower(repower_path=cfg.NH3_REPOWER_PATH)
        self.reset()

    # ---------- 状态与动作空间 ----------
    @property
    def state_dim(self):
        # 方案E：时序窗口堆叠后维度 = 12 × WINDOW_STEPS
        return 12 * int(getattr(cfg, "WINDOW_STEPS", 1))

    @property
    def action_dim(self):
        return 1

    # ---------- 峰谷分时电价（电解槽外购电价，元/kWh） ----------
    @staticmethod
    def tou_price(hour: float) -> float:
        if 23.0 <= hour or hour < 7.0:
            return 0.15        # 谷段
        if 11.0 <= hour < 14.0 or 18.0 <= hour < 21.0:
            return 0.55        # 峰段
        return 0.35            # 平段

    @staticmethod
    def curtail_factor(hour: float) -> float:
        """分段弃电惩罚因子 Cs（元/MWh），查表+插值"""
        h = int(np.floor(hour)) % 24
        return float(cfg.CURTAIL_PENALTY_BY_HOUR.get(h, 300.0))

    # ---------- 方案H：领域知识动作掩码 ----------
    def action_mask(self) -> np.ndarray:
        """返回 11 档动作掩码（1=合法，0=非法）。
        规则：峰段（11:00-14:00、18:00-21:00）且当前无弃电潜力时，
        屏蔽 ≥ MASK_PEAK_LEVEL（默认70%）的功率档位，
        消除"峰段高价外购电制氢"的次优行为，缩小搜索空间。"""
        n_actions = cfg.PPO.get("n_actions", 11)
        mask = np.ones(n_actions, dtype=np.float32)
        if not cfg.ACTION_MASK:
            return mask
        hour = (self.t % cfg.SIM_STEPS) / cfg.SIM_STEPS * 24.0
        surplus = max(0.0, self.data["pv"][self.t] + self.data["wind"][self.t]
                      - self.data["load"][self.t])
        is_peak = (11.0 <= hour < 14.0) or (18.0 <= hour < 21.0)
        if is_peak and surplus <= 1e-6:
            levels = np.linspace(0.0, 1.0, n_actions)
            mask[levels >= cfg.MASK_PEAK_LEVEL] = 0.0
        return mask

    # ---------- 环境主流程 ----------
    def reset(self, seed: int = None):
        if seed is not None:
            self.seed = seed
        self.data = gen_day_data(seed=self.seed, days=self.days,
                                 season_month=getattr(cfg, "SEASON_MONTH", 6),
                                 region=getattr(cfg, "REGION", "northwest"),
                                 weather=getattr(cfg, "WEATHER_MODE", "auto"))
        self.t = 0                                   # 当前时点
        self.ely.power = self.ely.min_power
        self.ely.h2_produced_kg = 0.0
        # 火电预热：初始出力置于首时刻平衡点附近，避免冷启动爬坡惩罚
        init_load = self.data["load"][0]
        init_ren = self.data["pv"][0] + self.data["wind"][0]
        init_th = float(np.clip(init_load - init_ren, 0, self.thermal.pmax.sum()))
        self.thermal.power = np.clip(np.full(self.thermal.n, init_th / self.thermal.n),
                                     self.thermal.pmin, self.thermal.pmax)
        self.thermal.cost_total = 0.0
        self.ammonia.nh3_produced_t = 0.0
        self.curtail_mwh = 0.0
        self.ely_mwh = 0.0
        self.thermal_mwh = 0.0
        self.ren_used_mwh = 0.0
        self.grid_buy_mwh = 0.0
        self.total_reward = 0.0
        self.co2_reduction_t = 0.0
        self.nh3_storage.reset(start_inventory_t=cfg.NH3_START_INVENTORY_T)
        self.nh3_repower = AmmoniaRePower(repower_path=cfg.NH3_REPOWER_PATH)
        self.history = []                            # 每步明细
        # 方案E：时序窗口历史帧（WINDOW_STEPS>1 时启用，首帧重复填充）
        self.state_hist = np.tile(self._raw_state(), (int(getattr(cfg, "WINDOW_STEPS", 1)), 1))
        return self._state()

    def _raw_state(self) -> np.ndarray:
        """当前时点单帧状态（12维，窗口堆叠的基础帧）"""
        hour = (self.t % cfg.SIM_STEPS) / cfg.SIM_STEPS * 24.0
        surplus_mw = max(0.0, self.data["pv"][self.t] + self.data["wind"][self.t]
                         - self.data["load"][self.t])
        # 未来前瞻：未来 LOOKAHEAD_HOURS 小时平均弃电潜力（AGC预测调度）
        look_steps = max(1, int(cfg.LOOKAHEAD_HOURS / cfg.DT_HOURS))
        fut = min(self.t + look_steps, cfg.SIM_STEPS * self.days)
        fut_surplus = np.mean([
            max(0.0, self.data["pv"][j] + self.data["wind"][j] - self.data["load"][j])
            for j in range(self.t, fut)
        ])
        # 火电爬坡余量：当前出力距上限的爬坡空间（归一化）
        ramp_margin = (self.thermal.pmax.sum() - self.thermal.total_power()) / self.thermal.pmax.sum()
        s = np.array([
            self.data["pv_pu"][self.t],
            self.data["wind_pu"][self.t],
            self.data["load_pu"][self.t],
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            self.ely.power / self.ely.capacity,
            self.thermal.total_power() / self.thermal.pmax.sum(),
            self.curtail_factor(hour) / 520.0,
            self.tou_price(hour) / 0.55,
            surplus_mw / self.ely.capacity,        # 当前弃电潜力（标幺）
            fut_surplus / self.ely.capacity,       # 未来弃电潜力（标幺）
            ramp_margin,                           # 火电爬坡余量
        ], dtype=np.float32)
        return s

    def _state(self) -> np.ndarray:
        """方案E：返回最近 WINDOW_STEPS 帧堆叠后的状态"""
        if int(getattr(cfg, "WINDOW_STEPS", 1)) <= 1:
            return self._raw_state()
        return self.state_hist.reshape(-1).astype(np.float32)

    def step(self, action: np.ndarray):
        """执行一个时点（15min）的调度，返回 (state, reward, done, info)"""
        dt_h = cfg.DT_HOURS
        hour = (self.t % cfg.SIM_STEPS) / cfg.SIM_STEPS * 24.0

        # ---- 1. 电解槽功率（用电端调峰决策） ----
        ely_norm = float(np.clip(action[0], 0.0, 1.0))
        ely_target = ely_norm * self.ely.capacity
        # 爬坡约束（电解槽功率调整率）
        ramp_max = self.ely.capacity * cfg.ELY_RAMP_RATE
        ely_target = np.clip(ely_target, self.ely.power - ramp_max,
                             self.ely.power + ramp_max)
        # 最低运行功率约束
        if ely_target < self.ely.min_power * 0.5:
            ely_target = 0.0
        h2_flow = self.ely.step(ely_target, dt_h)
        h2_kg = h2_flow * dt_h

        # ---- 2. 电力平衡与消纳 ----
        pv = self.data["pv"][self.t]
        wind = self.data["wind"][self.t]
        load = self.data["load"][self.t]
        renewable = pv + wind
        net_load = load + self.ely.power

        # 新能源优先供电，缺口由火电补足（受爬坡/出力约束）
        th_need = max(0.0, net_load - renewable)
        th_target = np.full(self.thermal.n, th_need / self.thermal.n)
        th_target = np.clip(th_target, self.thermal.pmin, self.thermal.pmax)
        th_actual, th_cost, _ = self.thermal.step(th_target, dt_h)
        th_sum = float(th_actual.sum())

        # 能量守恒：总发电(renewable+火电)超出用电的部分必须弃掉，先弃新能源
        generation = renewable + th_sum
        curtail_mw = max(0.0, generation - net_load)
        ren_used = min(renewable, renewable - curtail_mw)
        ren_used = max(0.0, ren_used)
        # 缺口（供不应求，惩罚）
        unserved_mw = max(0.0, net_load - generation)

        # ---- 3. 经济结算（元） ----
        curtail_mwh = curtail_mw * dt_h
        ely_mwh = self.ely.power * dt_h
        # 电解槽用电力来源：优先免费消纳弃电，不足部分按分时电价外购
        surplus_mw = max(0.0, renewable - load)      # 弃电潜力
        ely_from_surplus = min(self.ely.power, surplus_mw)
        ely_from_grid = max(0.0, self.ely.power - ely_from_surplus)
        grid_buy_mwh = ely_from_grid * dt_h
        cost_grid = grid_buy_mwh * 1000 * self.tou_price(hour)
        cost_curtail = curtail_mwh * self.curtail_factor(hour)
        cost_unserved = unserved_mw * dt_h * 3000.0       # 缺电损失 3000 元/MWh

        # ---- 优化2：分级氢价值（绿氢/灰氢） ----
        # 弃电制氢 = 绿氢（22元/kg，零碳溢价）；外购电制氢 = 灰氢（8元/kg，边际价值）
        h2_from_surplus = h2_kg * (ely_from_surplus / max(1e-6, self.ely.power)) if self.ely.power > 0 else 0.0
        h2_from_grid = h2_kg - h2_from_surplus
        value_h2 = h2_from_surplus * cfg.H2_PRICE_GREEN + h2_from_grid * cfg.H2_PRICE_GREY

        # ---- 优化1：氨电联产经济闭环（氨价值 − 制氨加热成本 + 副产氧价值） ----
        # 每吨氨 176 kg 氢 -> 产氨量；氨价 3100 元/t
        # 加热成本：总能耗 × 废热未覆盖比例 × 热价（废热耦合，说明书核心创新）
        nh3_t = self.ammonia.h2_to_nh3(h2_kg)

        # ===== 氨库存动态与价值计算（三条互斥路径） =====
        # 根据运行模式决定氨是否库存/外售/回电
        operate_mode = getattr(cfg, "NH3_OPERATE_MODE", "sell")
        if operate_mode == "sell":
            # 路径A：化工原料外售（原模式，不计回电）
            value_nh3 = nh3_t * cfg.NH3_PRICE
        elif operate_mode in ("store", "repower", "hybrid"):
            # 路径B/C：进入库存管理，按库存动态计算期末用量
            repower_ratio = getattr(cfg, "NH3_REPOWER_RATIO", 0.0)
            store_result = self.nh3_storage.step(
                nh3_produced_t=nh3_t,
                operate_mode=operate_mode,
                repower_ratio=repower_ratio
            )
            # 外售部分：按氨价计价
            value_nh3_sold = store_result["sold_t"] * cfg.NH3_PRICE

            # 回电部分：按回电效率折算价值
            repower_t = store_result["repower_t"]
            if repower_t > 0 and self.nh3_repower.repower_path != "sell":
                rp_val = self.nh3_repower.repower_value(
                    repower_t,
                    electricity_price_per_kwh=getattr(cfg, "NH3_REPOWER_PRICE_PER_KWH", 0.35),
                    include_capacity_credit=getattr(cfg, "NH3_REPOWER_ENABLE_CAPACITY_CREDIT", False)
                )
                value_nh3_repower = rp_val["power_value"]
            else:
                value_nh3_repower = 0.0

            # 储罐能耗成本
            tank_energy_kwh = store_result["tank_energy_kwh"]
            cost_tank_energy = tank_energy_kwh * self.tou_price(23)  # 谷段电价（储罐制冷持续运行）

            # 综合：外售价值 + 回电价值 - 储罐能耗成本
            value_nh3 = value_nh3_sold + value_nh3_repower - cost_tank_energy
        else:
            value_nh3 = nh3_t * cfg.NH3_PRICE  # 默认走外售

        cost_nh3_heat = self.ammonia.heating_cost(h2_kg, cfg.WASTE_HEAT_RATIO)
        # 副产氧价值（说明书：每吨氨产氧2816kg）
        value_o2 = oxygen_byproduct(h2_kg) / 1000.0 * cfg.O2_PRICE_PER_T

        # ---- 优化3：火电调峰收益（电解槽消纳弃电 = 减少火电深度调峰） ----
        # 电解槽每消纳 1 MWh 弃电，火电减少相应调峰出力，按调峰价值折算收益
        peak_shed_value = ely_from_surplus * dt_h * cfg.THERMAL_PEAK_SHED_VALUE

        # ---- 4. 奖励（利润最大 = 成本最小） ----
        reward = (value_h2 + value_nh3 + value_o2 + peak_shed_value
                  - th_cost - cost_grid - cost_curtail - cost_unserved - cost_nh3_heat)
        self.total_reward += reward

        # ---- 5. 统计 ----
        self.curtail_mwh += curtail_mwh
        self.ely_mwh += ely_mwh
        self.thermal_mwh += th_sum * dt_h
        self.ren_used_mwh += ren_used * dt_h
        self.grid_buy_mwh += grid_buy_mwh
        self.ammonia.step(h2_kg)

        # 氨库存每日结束时记录（如果启用库存模式）
        if getattr(cfg, "NH3_OPERATE_MODE", "sell") != "sell":
            self.nh3_storage_inventory_t = self.nh3_storage.inventory_t
            self.nh3_storage_sold_t = self.nh3_storage.total_sold_t
            self.nh3_storage_repower_t = self.nh3_storage.total_repower_t
            self.nh3_storage_tank_kwh = self.nh3_storage.total_tank_energy_kwh

        # CO2减排：电解槽消纳的弃电替代煤制氢（每 kWh 减排量）
        ely_surplus_mwh = ely_from_surplus * dt_h
        self.co2_reduction_t += ely_surplus_mwh * 1000 * cfg.CO2_REDUCTION_PER_KWH

        self.history.append({
            "t": self.t, "hour": hour, "pv": pv, "wind": wind, "load": load,
            "ely": self.ely.power, "h2_kg": h2_kg, "nh3_t": nh3_t,
            "th_sum": th_sum, "th_cost": th_cost, "ren_used": ren_used,
            "curtail_mw": curtail_mw, "unserved_mw": unserved_mw,
            "cost_grid": cost_grid, "cost_curtail": cost_curtail,
            "value_h2": value_h2, "value_nh3": value_nh3, "value_o2": value_o2,
            "cost_nh3_heat": cost_nh3_heat, "peak_shed_value": peak_shed_value,
            "reward": reward,
        })

        self.t += 1
        done = self.t >= cfg.SIM_STEPS * self.days
        # 方案E：滚动更新时序窗口（仅未终止时，终止后立即 reset）
        if int(getattr(cfg, "WINDOW_STEPS", 1)) > 1 and not done:
            self.state_hist = np.roll(self.state_hist, -1, axis=0)
            self.state_hist[-1] = self._raw_state()
        return self._state() if not done else None, reward, done, {
            "h2_kg": h2_kg, "nh3_t": nh3_t, "th_cost": th_cost,
            "curtail_mw": curtail_mw, "ely_mw": self.ely.power,
            "value_h2": value_h2, "value_nh3": value_nh3,
            "cost_nh3_heat": cost_nh3_heat, "peak_shed_value": peak_shed_value,
        }

    # ---------- 结果汇总 ----------
    def summary(self) -> dict:
        ren_total = float((self.data["pv"] + self.data["wind"]).sum() * cfg.DT_HOURS)
        load_total = float(self.data["load"].sum() * cfg.DT_HOURS)
        return {
            "curtail_mwh": self.curtail_mwh,
            "ely_mwh": self.ely_mwh,
            "thermal_mwh": self.thermal_mwh,
            "ren_used_mwh": self.ren_used_mwh,
            "grid_buy_mwh": self.grid_buy_mwh,
            "load_mwh": load_total,
            "renewable_mwh": ren_total,
            "curtail_rate": self.curtail_mwh / ren_total if ren_total else 0,
            "renewable_utilization": 1.0 - (self.curtail_mwh / ren_total if ren_total else 0),
            "h2_kg": self.ely.h2_produced_kg,
            "nh3_t": self.ammonia.nh3_produced_t,
            "h2_t": self.ely.h2_produced_kg / 1000.0,
            "thermal_cost": self.thermal.cost_total,
            "grid_cost": self.grid_buy_mwh * 1000 * 0.35,   # 近似外购成本
            "curtail_cost": self.curtail_mwh * 800.0,       # 近似惩罚
            "total_reward": self.total_reward,
            "co2_reduction_t": self.co2_reduction_t,
            "nh3_inventory_t": getattr(self, "nh3_storage_inventory_t", 0.0),
            "nh3_sold_t": getattr(self, "nh3_storage_sold_t", 0.0),
            "nh3_repower_t": getattr(self, "nh3_storage_repower_t", 0.0),
            "nh3_tank_kwh": getattr(self, "nh3_storage_tank_kwh", 0.0),
            "history": self.history,
        }
