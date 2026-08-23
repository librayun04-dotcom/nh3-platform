# -*- coding: utf-8 -*-
"""
向量化 AGC 环境（VectorAGCEnv）—— 为 4090 训练设计
================================================================
并行运行 N 个子环境（每个子环境独立的风光/负荷日曲线），
一次 step 批量推进所有环境，配合 PPO 批量推理，最大化 GPU 利用率。

物理模型与 AGCEnv 完全一致（说明书公式 4-10）：
  - 电解槽功率（动作，0~1 标幺）→ 用电端调峰
  - 新能源优先上网，弃电按分段惩罚因子 Cs 计罚
  - 火电按阀点效应经济调度补足缺口（爬坡/出力约束）
  - 奖励 = 氢能价值 − 火电成本 − 外购电 − 弃电惩罚 − 缺电惩罚
"""
import numpy as np

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config as cfg
from src.models.renewable import gen_day_data


def _tou_price_vec(hour: np.ndarray) -> np.ndarray:
    """分时电价（元/kWh）向量化"""
    price = np.full_like(hour, 0.35)
    valley = (hour >= 23.0) | (hour < 7.0)
    peak = ((hour >= 11.0) & (hour < 14.0)) | ((hour >= 18.0) & (hour < 21.0))
    price[valley] = 0.15
    price[peak] = 0.55
    return price


def _curtail_factor_vec(hour: np.ndarray) -> np.ndarray:
    """分段弃电惩罚因子 Cs（元/MWh）向量化"""
    h = np.floor(hour).astype(int) % 24
    table = np.array([cfg.CURTAIL_PENALTY_BY_HOUR.get(i, 300.0) for i in range(24)])
    return table[h]


def _action_mask_vec(hour: np.ndarray, surplus: np.ndarray) -> np.ndarray:
    """方案H：领域知识动作掩码（向量化）。
    规则：峰段（11:00-14:00、18:00-21:00）且当前无弃电潜力时，
    屏蔽 ≥ MASK_PEAK_LEVEL（默认70%）的功率档位。
    返回 (n_envs, n_actions) 的 0/1 掩码。"""
    n_actions = cfg.PPO.get("n_actions", 11)
    mask = np.ones((len(hour), n_actions), dtype=np.float32)
    if not cfg.ACTION_MASK:
        return mask
    is_peak = ((hour >= 11.0) & (hour < 14.0)) | ((hour >= 18.0) & (hour < 21.0))
    block = is_peak & (surplus <= 1e-6)
    if block.any():
        levels = np.linspace(0.0, 1.0, n_actions)
        mask[block] = np.where(levels >= cfg.MASK_PEAK_LEVEL, 0.0, 1.0)
    return mask


class VectorAGCEnv:
    """N 个并行 AGC 调度环境（全向量化，供训练）"""

    def __init__(self, n_envs: int = 8, seed: int = 0):
        self.n_envs = n_envs
        self.base_seed = seed
        self.dt_h = cfg.DT_HOURS
        # 方案E：时序窗口堆叠（1=单帧）
        self.window = int(getattr(cfg, "WINDOW_STEPS", 1))
        self.state_dim = 12 * self.window
        self.action_dim = 1
        self.ely_cap = cfg.ELECTROLYZER_CAP
        self.ely_min = cfg.ELECTROLYZER_CAP * cfg.ELECTROLYZER_MIN_RATIO
        self.ely_ramp = cfg.ELECTROLYZER_CAP * cfg.ELY_RAMP_RATE
        self.n_units = len(cfg.THERMAL_UNITS)
        self.pmin = np.array(cfg.THERMAL_PMIN, dtype=np.float32)
        self.pmax = np.array(cfg.THERMAL_UNITS, dtype=np.float32)
        self.ramp_up = np.array(cfg.THERMAL_RAMP_UP, dtype=np.float32)
        self.ramp_dn = np.array(cfg.THERMAL_RAMP_DN, dtype=np.float32)
        self.a = np.array(cfg.THERMAL_A, dtype=np.float32)
        self.b = np.array(cfg.THERMAL_B, dtype=np.float32)
        self.c = np.array(cfg.THERMAL_C, dtype=np.float32)
        self.e = np.array(cfg.THERMAL_E, dtype=np.float32)
        self.f = np.array(cfg.THERMAL_F, dtype=np.float32)
        self.reset()

    # ---------- 重置 ----------
    def reset(self, seeds: np.ndarray = None):
        if seeds is None:
            seeds = self.base_seed + np.arange(self.n_envs)
        self.seeds = np.asarray(seeds)
        self.t = np.zeros(self.n_envs, dtype=int)
        self.ely_power = np.full(self.n_envs, self.ely_min, dtype=np.float32)
        self.th_power = np.tile(self.pmin[None, :], (self.n_envs, 1)).astype(np.float32)
        self.th_cost_total = np.zeros(self.n_envs, dtype=np.float32)
        self.h2_total = np.zeros(self.n_envs, dtype=np.float32)
        self.nh3_total = np.zeros(self.n_envs, dtype=np.float32)
        self.curtail_mwh = np.zeros(self.n_envs, dtype=np.float32)
        self.ely_mwh = np.zeros(self.n_envs, dtype=np.float32)
        self.reward_total = np.zeros(self.n_envs, dtype=np.float32)
        self.co2_total = np.zeros(self.n_envs, dtype=np.float32)
        self._load_day_data()
        # 火电预热：置于首时刻平衡点
        ren0 = self.pv[:, 0] + self.wind[:, 0]
        load0 = self.load[:, 0]
        init_th = np.clip(load0 - ren0, 0, self.pmax.sum())
        self.th_power = np.clip(np.tile((init_th / self.n_units)[:, None], (1, self.n_units)),
                                self.pmin, self.pmax).astype(np.float32)
        # 方案E：时序窗口历史帧（首帧重复填充）
        if self.window > 1:
            self.hist = np.tile(self._raw_state()[:, None, :], (1, self.window, 1))
        else:
            self.hist = None
        return self._state()

    def _load_day_data(self):
        n = self.n_envs
        steps = cfg.SIM_STEPS
        self.pv = np.zeros((n, steps), dtype=np.float32)
        self.wind = np.zeros((n, steps), dtype=np.float32)
        self.load = np.zeros((n, steps), dtype=np.float32)
        self.pv_pu = np.zeros((n, steps), dtype=np.float32)
        self.wind_pu = np.zeros((n, steps), dtype=np.float32)
        self.load_pu = np.zeros((n, steps), dtype=np.float32)
        for i in range(n):
            d = gen_day_data(seed=int(self.seeds[i]), days=1,
                             season_month=getattr(cfg, "SEASON_MONTH", 6),
                             region=getattr(cfg, "REGION", "northwest"),
                             weather=getattr(cfg, "WEATHER_MODE", "auto"))
            self.pv[i] = d["pv"]; self.wind[i] = d["wind"]; self.load[i] = d["load"]
            self.pv_pu[i] = d["pv_pu"]; self.wind_pu[i] = d["wind_pu"]; self.load_pu[i] = d["load_pu"]

    # ---------- 状态 ----------
    def _raw_state(self) -> np.ndarray:
        """当前时点单帧状态（12维）"""
        hour = (self.t % cfg.SIM_STEPS) / cfg.SIM_STEPS * 24.0
        idx = np.arange(self.n_envs)
        surplus = np.maximum(0.0, self.pv[idx, self.t] + self.wind[idx, self.t]
                             - self.load[idx, self.t])
        # 未来前瞻平均弃电潜力（向量化）
        look_steps = max(1, int(cfg.LOOKAHEAD_HOURS / cfg.DT_HOURS))
        fut = np.minimum(self.t + look_steps, cfg.SIM_STEPS - 1)
        fut_surplus = np.zeros(self.n_envs, dtype=np.float32)
        for i in range(self.n_envs):
            if fut[i] > self.t[i]:
                fut_surplus[i] = np.mean(np.maximum(0.0,
                    self.pv[i, self.t[i]:fut[i]] + self.wind[i, self.t[i]:fut[i]]
                    - self.load[i, self.t[i]:fut[i]]))
        ramp_margin = (self.pmax.sum() - self.th_power.sum(axis=1)) / self.pmax.sum()
        s = np.stack([
            self.pv_pu[idx, self.t],
            self.wind_pu[idx, self.t],
            self.load_pu[idx, self.t],
            np.sin(2 * np.pi * hour / 24.0),
            np.cos(2 * np.pi * hour / 24.0),
            self.ely_power / self.ely_cap,
            self.th_power.sum(axis=1) / self.pmax.sum(),
            _curtail_factor_vec(hour) / 520.0,
            _tou_price_vec(hour) / 0.55,
            surplus / self.ely_cap,                  # 当前弃电潜力（标幺）
            fut_surplus / self.ely_cap,              # 未来弃电潜力（标幺）
            ramp_margin,                             # 火电爬坡余量
        ], axis=1).astype(np.float32)
        return s

    def _state(self) -> np.ndarray:
        """方案E：返回最近 WINDOW_STEPS 帧堆叠后的状态"""
        if self.window <= 1:
            return self._raw_state()
        return self.hist.reshape(self.n_envs, -1).astype(np.float32)

    # ---------- 方案H：领域知识动作掩码（向量化） ----------
    def action_mask(self) -> np.ndarray:
        """返回 (n_envs, n_actions) 动作掩码（1=合法，0=非法），
        规则见 _action_mask_vec：峰段且无弃电时屏蔽高功率档位"""
        hour = (self.t % cfg.SIM_STEPS) / cfg.SIM_STEPS * 24.0
        idx = np.arange(self.n_envs)
        surplus = np.maximum(0.0, self.pv[idx, self.t] + self.wind[idx, self.t]
                             - self.load[idx, self.t])
        return _action_mask_vec(hour, surplus)

    # ---------- 单步 ----------
    def step(self, action: np.ndarray):
        """action: (n_envs, action_dim) ∈ [0,1] -> (state, reward, done, info)"""
        dt = self.dt_h
        idx = np.arange(self.n_envs)
        hour = (self.t % cfg.SIM_STEPS) / cfg.SIM_STEPS * 24.0
        pv = self.pv[idx, self.t]; wind = self.wind[idx, self.t]; load = self.load[idx, self.t]
        renewable = pv + wind

        # ---- 1. 电解槽功率（爬坡约束 + 最低运行） ----
        ely_norm = np.clip(action[:, 0], 0.0, 1.0)
        ely_target = ely_norm * self.ely_cap
        ely_target = np.clip(ely_target, self.ely_power - self.ely_ramp,
                             self.ely_power + self.ely_ramp)
        ely_target[ely_target < self.ely_min * 0.5] = 0.0
        self.ely_power = ely_target
        # 产氢（含负载-效率曲线）
        from src.models.electrolyzer import Electrolyzer
        eff = np.array([Electrolyzer.efficiency(p, self.ely_cap) for p in self.ely_power], dtype=np.float32)
        energy_kwh = self.ely_power * 1000 * dt
        h2_nm3 = (energy_kwh * eff) / cfg.ELY_ENERGY_KWH_NM3
        h2_kg = h2_nm3 * cfg.KG_PER_NM3_H2
        self.h2_total += h2_kg

        # ---- 2. 电力平衡 ----
        net_load = load + self.ely_power
        if getattr(cfg, "MAPPO", False):
            # 方案F：火电出力由智能体直接决策（动作 a[:,1:] 标幺 -> MW）
            th_target = np.clip(action[:, 1:], 0.0, 1.0) * self.pmax[None, :]
        else:
            th_need = np.maximum(0.0, net_load - renewable)
            th_target = np.clip(np.tile((th_need / self.n_units)[:, None], (1, self.n_units)),
                                self.pmin, self.pmax)
        # 爬坡约束
        p_prev = self.th_power
        th_target = np.clip(th_target, p_prev - self.ramp_dn * (dt * 4),
                            p_prev + self.ramp_up * (dt * 4))
        th_target = np.clip(th_target, self.pmin, self.pmax)
        self.th_power = th_target
        th_sum = th_target.sum(axis=1)
        # 阀点效应成本（元/h * dt）
        base = self.a * th_target ** 2 + self.b * th_target + self.c
        valve = self.e * np.abs(np.sin(self.f * (self.pmin - th_target)))
        th_cost = (base + valve).sum(axis=1) * dt
        self.th_cost_total += th_cost

        # 能量守恒：总发电超出用电 → 弃电（先弃新能源）
        generation = renewable + th_sum
        curtail_mw = np.maximum(0.0, generation - net_load)
        ren_used = np.minimum(renewable, np.maximum(0.0, renewable - curtail_mw))
        unserved_mw = np.maximum(0.0, net_load - generation)

        # ---- 3. 经济结算 ----
        curtail_mwh = curtail_mw * dt
        ely_mwh = self.ely_power * dt
        surplus_mw = np.maximum(0.0, renewable - load)
        ely_from_surplus = np.minimum(self.ely_power, surplus_mw)
        ely_from_grid = np.maximum(0.0, self.ely_power - ely_from_surplus)
        grid_buy_mwh = ely_from_grid * dt
        cost_grid = grid_buy_mwh * 1000 * _tou_price_vec(hour)
        cost_curtail = curtail_mwh * _curtail_factor_vec(hour)
        cost_unserved = unserved_mw * dt * 3000.0

        # 优化2：分级氢价值（弃电制氢=绿氢22元/kg，外购电制氢=灰氢8元/kg）
        surpl_frac = ely_from_surplus / np.maximum(1e-6, self.ely_power)
        h2_from_surplus = h2_kg * surpl_frac
        h2_from_grid = h2_kg - h2_from_surplus
        value_h2 = h2_from_surplus * cfg.H2_PRICE_GREEN + h2_from_grid * cfg.H2_PRICE_GREY

        # 优化1：氨电联产闭环（氨价值 − 制氨加热成本 + 副产氧价值）
        nh3_t = h2_kg / cfg.H2_KG_PER_TON_NH3
        value_nh3 = nh3_t * cfg.NH3_PRICE
        # 废热耦合：加热成本按废热利用率折扣（说明书核心创新）
        cost_nh3_heat = nh3_t * cfg.HEATING_GJ_PER_TON_NH3 * (1 - cfg.WASTE_HEAT_RATIO) * cfg.HEAT_COST_PER_GJ
        value_o2 = (nh3_t * 2816.0) / 1000.0 * cfg.O2_PRICE_PER_T

        # 优化3：火电调峰收益（电解槽消纳弃电 = 减少火电深度调峰）
        peak_shed_value = ely_from_surplus * dt * cfg.THERMAL_PEAK_SHED_VALUE

        reward = (value_h2 + value_nh3 + value_o2 + peak_shed_value
                  - th_cost - cost_grid - cost_curtail - cost_unserved - cost_nh3_heat)
        self.reward_total += reward

        # ---- 4. 统计 ----
        self.curtail_mwh += curtail_mwh
        self.ely_mwh += ely_mwh
        self.nh3_total += nh3_t
        self.co2_total += (ely_from_surplus * dt) * 1000 * cfg.CO2_REDUCTION_PER_KWH

        self.t += 1
        done = self.t >= cfg.SIM_STEPS
        # 方案E：滚动更新时序窗口（仅未终止环境，终止后立即 reset）
        if self.window > 1 and not done.all():
            self.hist = np.roll(self.hist, -1, axis=1)
            self.hist[:, -1, :] = self._raw_state()
        ns = None
        if not done.all():
            valid = np.where(~done)[0]
            ns = np.zeros((self.n_envs, self.state_dim), dtype=np.float32)
            if self.window > 1:
                # 方案E：窗口模式下子环境状态直接取已滚动的 hist
                ns[valid] = self.hist[valid].reshape(len(valid), -1)
            else:
                fut_v = np.minimum(self.t[valid] + max(1, int(cfg.LOOKAHEAD_HOURS / cfg.DT_HOURS)), cfg.SIM_STEPS - 1)
                fut_surp_v = np.zeros(len(valid), dtype=np.float32)
                for k, i in enumerate(valid):
                    if fut_v[k] > self.t[i]:
                        fut_surp_v[k] = np.mean(np.maximum(0.0,
                            self.pv[i, self.t[i]:fut_v[k]] + self.wind[i, self.t[i]:fut_v[k]]
                            - self.load[i, self.t[i]:fut_v[k]]))
                ramp_v = (self.pmax.sum() - self.th_power[valid].sum(axis=1)) / self.pmax.sum()
                sub = np.stack([
                    self.pv_pu[valid, self.t[valid]],
                    self.wind_pu[valid, self.t[valid]],
                    self.load_pu[valid, self.t[valid]],
                    np.sin(2 * np.pi * (self.t[valid] % cfg.SIM_STEPS) / cfg.SIM_STEPS * 24.0 / 24.0),
                    np.cos(2 * np.pi * (self.t[valid] % cfg.SIM_STEPS) / cfg.SIM_STEPS * 24.0 / 24.0),
                    self.ely_power[valid] / self.ely_cap,
                    self.th_power[valid].sum(axis=1) / self.pmax.sum(),
                    _curtail_factor_vec(hour[valid]) / 520.0,
                    _tou_price_vec(hour[valid]) / 0.55,
                    np.maximum(0.0, self.pv[valid, self.t[valid]] + self.wind[valid, self.t[valid]]
                               - self.load[valid, self.t[valid]]) / self.ely_cap,
                    fut_surp_v / self.ely_cap,
                    ramp_v,
                ], axis=1).astype(np.float32)
                ns[valid] = sub
        return ns, reward.astype(np.float32), done, {
            "h2_kg": h2_kg, "curtail_mw": curtail_mw, "th_cost": th_cost,
            "value_h2": value_h2, "value_nh3": value_nh3, "value_o2": value_o2,
            "cost_nh3_heat": cost_nh3_heat, "peak_shed_value": peak_shed_value,
            "reward": reward,
        }

    # ---------- 结果 ----------
    def summary(self, env_idx: int = 0) -> dict:
        ren_total = float((self.pv[env_idx] + self.wind[env_idx]).sum() * self.dt_h)
        return {
            "curtail_mwh": float(self.curtail_mwh[env_idx]),
            "ely_mwh": float(self.ely_mwh[env_idx]),
            "renewable_mwh": ren_total,
            "curtail_rate": float(self.curtail_mwh[env_idx] / ren_total) if ren_total else 0,
            "renewable_utilization": 1.0 - (float(self.curtail_mwh[env_idx] / ren_total) if ren_total else 0),
            "h2_kg": float(self.h2_total[env_idx]),
            "h2_t": float(self.h2_total[env_idx] / 1000.0),
            "nh3_t": float(self.nh3_total[env_idx]),
            "thermal_cost": float(self.th_cost_total[env_idx]),
            "total_reward": float(self.reward_total[env_idx]),
            "co2_reduction_t": float(self.co2_total[env_idx]),
        }


if __name__ == "__main__":
    env = VectorAGCEnv(n_envs=4, seed=0)
    s = env.reset()
    print("初始状态 shape:", s.shape)
    a = np.full((4, 1), 0.5, dtype=np.float32)
    ns, r, done, info = env.step(a)
    print("一步后 reward:", r, "done:", done)
    sm = env.summary(0)
    print("环境0汇总:", {k: round(v, 2) if isinstance(v, float) else v for k, v in sm.items() if k != "history"})
