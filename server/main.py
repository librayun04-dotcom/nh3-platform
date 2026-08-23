# -*- coding: utf-8 -*-
"""
氨电联产 AGC 智能调度平台 —— FastAPI 后端
================================================
功能：
  1. 模型管理     GET  /api/models            列出所有可用 checkpoint（refresh=1 重扫）
  2. 曲线推理预测 POST /api/predict           上传风光负荷曲线(CSV/XLSX) → PPO 推理 → 96 时步调度指令
  3. 多模型对比   POST /api/compare           同一曲线多 checkpoint 批量推理
  4. 示例曲线     GET  /api/sample_curve      一键生成演示曲线
  5. 在线训练     POST /api/train/start       后台线程训练（与 src/train.py 同一套路）
                  GET  /api/train/stream      SSE 实时推送收敛曲线
                  POST /api/train/stop        终止训练
  6. 实时 AGC 仿真 WS   /ws/simulate          WebSocket 滚动下发电解槽功率指令（15min/步）
  7. 前端托管     GET  /                      静态单页

启动：
    python server/main.py          # http://127.0.0.1:8000
"""
import os
import sys
import json
import time
import asyncio
import threading
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg  # noqa: E402

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ============================================================
# 全局状态
# ============================================================
app = FastAPI(title="氨电联产 AGC 智能调度平台", version="1.0")
TRAIN_STATE: Dict = {
    "running": False, "should_stop": False, "thread": None,
    "log": [],            # 每幕一条 {step, reward_mean, util_pct, h2_t}
    "config": {}, "started_at": None,
    "finished": False, "error": None, "final_model": None,
}


def _torch_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ============================================================
# 模型注册与管理
# ============================================================
MODEL_REGISTRY: Dict[str, dict] = {}


def scan_models():
    """动态扫描项目根下所有 */data/*.pt（含在线训练新产出的目录）"""
    MODEL_REGISTRY.clear()
    import torch
    for d in sorted(PROJECT_ROOT.glob("*/data")):
        if not d.is_dir() or d.name.startswith(("_", ".")):
            continue
        for p in sorted(d.glob("*.pt")):
            label = f"{d.parent.name}/{p.stem}"
            steps = 0
            try:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
                steps = int(ckpt.get("steps_done", 0)) if isinstance(ckpt, dict) else 0
                del ckpt
            except Exception:
                pass
            MODEL_REGISTRY[label] = {
                "name": label, "path": str(p.relative_to(PROJECT_ROOT)),
                "steps": steps, "size_kb": round(p.stat().st_size / 1024, 1),
                "mtime": p.stat().st_mtime,
            }
    return MODEL_REGISTRY


_AGENT_CACHE: Dict[str, object] = {}


def resolve_default_model() -> str:
    """默认模型：本地已训练 checkpoint 中步数最多者（并列取最新）"""
    if not MODEL_REGISTRY:
        scan_models()
    if not MODEL_REGISTRY:
        raise HTTPException(404, "未找到任何已训练模型，请先运行 src/train.py 或在「在线训练」页启动训练")
    best = max(MODEL_REGISTRY.values(), key=lambda m: (m["steps"], m["mtime"]))
    return best["name"]


def load_agent(model_name: str):
    """按名称加载 PPO 智能体（LRU=1，只缓存最近一个）"""
    from src.agents.ppo import PPO

    info = MODEL_REGISTRY.get(model_name)
    if not info:
        scan_models()
        info = MODEL_REGISTRY.get(model_name)
    if not info:
        raise HTTPException(404, f"模型「{model_name}」不存在。可用: {list(MODEL_REGISTRY)[:8]}")
    path = PROJECT_ROOT / info["path"]
    cached = _AGENT_CACHE.get("_cur")
    if cached is not None and cached[0] == str(path):
        return cached[1]
    n_actions = cfg.PPO["n_actions"]
    agent = PPO(cfg.PPO.get("state_dim", 12 * int(getattr(cfg, "WINDOW_STEPS", 1))),
                n_actions, seed=cfg.PPO.get("seed", 42))
    agent.load(str(path))
    _AGENT_CACHE["_cur"] = (str(path), agent)
    return agent


def _single_action(agent, state: np.ndarray) -> int:
    """单状态贪心推理 → 离散档位索引"""
    a, _, _ = agent.select_action_batch(np.asarray(state, dtype=np.float32)[None, ...],
                                        greedy=True, mask=None)
    return int(a[0])


# ============================================================
# 曲线注入环境（复用真实物理模型）
# ============================================================
def make_env_with_curve(curve: dict, seed: int = 0):
    """创建 AGCEnv 并注入用户上传的 96 时步曲线"""
    from src.models.grid_env import AGCEnv

    pv = np.asarray(curve["pv"], dtype=float)
    wind = np.asarray(curve["wind"], dtype=float)
    load = np.asarray(curve["load"], dtype=float)

    env = AGCEnv(seed=seed)
    env.reset()
    env.data.update({
        "pv": pv, "wind": wind, "load": load,
        "pv_pu": pv / cfg.PV_CAPACITY,
        "wind_pu": wind / cfg.WIND_CAPACITY,
        "load_pu": load / max(load.max(), 1e-6),
    })
    return env


def parse_curve_csv(content: bytes, filename: str) -> dict:
    """解析上传曲线 → 恒定 96 时步。

    支持 CSV/XLSX；表头含 pv/solar/光伏、wind/风电、load/demand/负荷（顺序不限）；
    无表头按 pv,wind,load 列序；24 行小时值自动插值到 15min。
    """
    import io
    import pandas as pd

    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(content), encoding="gbk")
    except Exception as e:
        raise HTTPException(400, f"文件解析失败: {e}")

    df.columns = [str(c).strip().lower() for c in df.columns]
    ALIAS = {"pv": ("pv", "solar", "光伏", "光"), "wind": ("wind", "风电"),
             "load": ("load", "demand", "负荷")}
    col_map = {}
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for std, aliases in ALIAS.items():
        hit = next((c for c in df.columns if any(a in c for a in aliases)), None)
        col_map[std] = hit
    # 无表头回退：按列序 pv,wind,load
    fallback = {"pv": 0, "wind": 1, "load": 2}
    series = {}
    for std in ("pv", "wind", "load"):
        c = col_map.get(std)
        if c is not None:
            series[std] = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(dtype=float)
        elif len(numeric_cols) > fallback[std]:
            series[std] = pd.to_numeric(df[numeric_cols[fallback[std]]],
                                        errors="coerce").fillna(0).to_numpy(dtype=float)
        else:
            series[std] = (np.zeros(len(df)) if std == "pv"
                           else np.full(len(df), 300.0 if std == "wind" else 500.0))

    n = len(series["pv"])
    if len(series["wind"]) != n or len(series["load"]) != n:
        raise HTTPException(400, "pv/wind/load 列长度不一致")

    def to96(arr: np.ndarray) -> list:
        if n == 96:
            return arr.tolist()
        xi = np.linspace(0, n - 1, 96)
        return np.interp(xi, np.arange(n), arr).tolist()

    return {"hour": [round(i * 0.25, 2) for i in range(96)],
            "pv": to96(series["pv"]), "wind": to96(series["wind"]),
            "load": to96(series["load"])}


# ============================================================
# 推理：PPO 一日调度 / 基线一日调度
# ============================================================
def run_dispatch(model_name: str, curve: dict) -> dict:
    from src.agents.ppo import PPO  # noqa: F401  (确保类已定义)

    agent = load_agent(model_name)
    env = make_env_with_curve(curve)
    s = env._state()
    steps = []
    t_idx = 0
    while True:
        idx = _single_action(agent, s)
        power = float(agent.action_to_power(np.array([idx]))[0])
        s2, r, done, info = env.step(np.array([power], dtype=np.float32))
        i = t_idx % cfg.SIM_STEPS
        ren = env.data["pv"][i] + env.data["wind"][i]
        steps.append({
            "t": t_idx, "hour": round(i * 0.25, 2),
            "action_level": idx,
            "ely_mw": round(env.ely.power, 1),
            "th_mw": round(env.thermal.total_power(), 1),
            "curtail_mw": round(max(0.0, ren - env.data["load"][i] - env.ely.power), 1),
            "reward": round(r / agent.reward_scale, 3),
        })
        t_idx += 1
        if done or s2 is None or t_idx >= cfg.SIM_STEPS * env.days:
            break
        s = s2

    sm = env.summary()
    scale = agent.reward_scale
    return {
        "model": model_name,
        "steps": steps,
        "summary": {
            "reward_wan": round(sm["total_reward"] / scale, 2),
            "utilization_pct": round(sm["renewable_utilization"] * 100, 1),
            "curtail_rate_pct": round(sm["curtail_rate"] * 100, 1),
            "h2_t": round(sm["h2_kg"] / 1000, 2),
            "nh3_t": round(sm["nh3_t"], 2),
            "co2_reduction_t": round(sm["co2_reduction_t"], 1),
            "ely_energy_mwh": round(sm["ely_mwh"], 1),
            "thermal_energy_mwh": round(sm["thermal_mwh"], 1),
        },
    }


def run_baseline_on_curve(kind: str, curve: dict) -> dict:
    from src.agents.baseline import run_baseline

    env = make_env_with_curve(curve)
    # run_baseline 内部会调用 env.reset() 重生成数据 —— 用 no-op 覆盖以保留注入曲线
    env.reset = lambda seed=None: None
    sm = run_baseline(env, kind)
    tr = sm["total_reward"]
    return {
        "strategy": kind,
        "summary": {
            "reward_wan": round(tr / cfg.PPO["reward_scale"], 2),
            "utilization_pct": round(sm["renewable_utilization"] * 100, 1),
            "curtail_rate_pct": round(sm["curtail_rate"] * 100, 1),
            "h2_t": round(sm["h2_kg"] / 1000, 2),
            "nh3_t": round(sm["nh3_t"], 2),
        },
    }


# ============================================================
# API：模型管理
# ============================================================
@app.get("/api/models")
def api_models(refresh: int = 0):
    if refresh or not MODEL_REGISTRY:
        scan_models()
    return {"models": sorted(MODEL_REGISTRY.values(), key=lambda m: -m["mtime"]),
            "device": _torch_device(), "n_models": len(MODEL_REGISTRY)}


@app.post("/api/models/rescan")
def api_rescan():
    _AGENT_CACHE.clear()
    scan_models()
    return {"ok": True, "n_models": len(MODEL_REGISTRY)}


# ============================================================
# API：推理预测
# ============================================================
@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...), model: str = Form(""),
                      with_baselines: int = Form(1)):
    content = await file.read()
    curve = parse_curve_csv(content, file.filename)
    model_name = model.strip() or resolve_default_model()
    result = run_dispatch(model_name, curve)
    if with_baselines:
        result["baselines"] = {k: run_baseline_on_curve(k, curve)["summary"]
                               for k in ("no_ely", "flat", "curtail_first")}
    result["curve"] = curve
    result["filename"] = file.filename
    return JSONResponse(result)


@app.post("/api/compare")
async def api_compare(file: UploadFile = File(...), models: str = Form("")):
    content = await file.read()
    curve = parse_curve_csv(content, file.filename)
    names = [m.strip() for m in models.split(",") if m.strip()]
    if not names:
        names = sorted(MODEL_REGISTRY.keys())
    results = []
    for nm in names[:6]:
        try:
            r = run_dispatch(nm, curve)
            results.append({"model": nm, "summary": r["summary"]})
        except HTTPException as e:
            results.append({"model": nm, "error": str(e.detail)})
    return {"filename": file.filename, "results": results}


@app.get("/api/sample_curve")
def api_sample(season: str = "summer", weather: str = "sunny"):
    from src.models.renewable import gen_day_data

    month = {"spring": 4, "summer": 6, "autumn": 10, "winter": 12}.get(season, 6)
    wmode = {"sunny": "sunny", "cloudy": "cloudy", "overcast": "overcast",
             "rain": "rain"}.get(weather, "sunny")
    d = gen_day_data(seed=7, days=1, season_month=month, region="northwest", weather=wmode)
    return {"hour": [round(i * 0.25, 2) for i in range(96)],
            "pv": [round(float(x), 1) for x in d["pv"]],
            "wind": [round(float(x), 1) for x in d["wind"]],
            "load": [round(float(x), 1) for x in d["load"]]}


# ============================================================
# 内置测试案例：一键用默认模型运行完整评估
# ============================================================
BUILTIN_CASES: List[dict] = [
    {"id": "summer_sunny", "name": "夏季晴日", "tagline": "标准工况 · 光伏大发",
     "desc": "6月西北晴天，光照资源最优，午间光伏大发形成显著弃电高峰，检验电解槽消纳能力。",
     "season_month": 6, "weather": "sunny", "region_key": "northwest", "region": "西北地区", "seed": 7},
    {"id": "winter_rain", "name": "冬季雨雪", "tagline": "最不利工况 · 资源匮乏",
     "desc": "12月东北雨雪天，光伏大幅衰减、供暖负荷抬升，检验策略在资源匮乏期的调度韧性。",
     "season_month": 12, "weather": "rain", "region_key": "northeast", "region": "东北地区", "seed": 11},
    {"id": "autumn_cloudy", "name": "秋季多云", "tagline": "温和波动 · 精细调节",
     "desc": "10月华北多云，中等光照叠加平稳负荷，考察精细化调节与外购电成本控制。",
     "season_month": 10, "weather": "cloudy", "region_key": "northchina", "region": "华北地区", "seed": 13},
    {"id": "spring_wind", "name": "春季大风", "tagline": "极限工况 · 双高峰",
     "desc": "4月西北大风期，风电持续高位与光伏午峰叠加形成双高峰，全天消纳压力最大。",
     "season_month": 4, "weather": "sunny", "region_key": "northwest", "region": "西北地区", "seed": 17},
]


def _case_curve(case: dict) -> dict:
    from src.models.renewable import gen_day_data

    d = gen_day_data(seed=case["seed"], days=1, season_month=case["season_month"],
                     region=case["region_key"], weather=case["weather"])
    return {"hour": [round(i * 0.25, 2) for i in range(96)],
            "pv": np.asarray(d["pv"], dtype=float).tolist(),
            "wind": np.asarray(d["wind"], dtype=float).tolist(),
            "load": np.asarray(d["load"], dtype=float).tolist()}


@app.get("/api/cases")
def api_cases():
    return {"cases": [{k: c[k] for k in ("id", "name", "tagline", "desc", "season_month",
                                         "region")} for c in BUILTIN_CASES]}


@app.post("/api/run_case")
async def api_run_case(case_id: str = Form(...)):
    """内置案例一键运行：默认模型推理 + 三基线对比 + 完整曲线"""
    case = next((c for c in BUILTIN_CASES if c["id"] == case_id), None)
    if not case:
        raise HTTPException(404, f"案例 {case_id} 不存在")
    curve = _case_curve(case)
    model_name = resolve_default_model()
    result = run_dispatch(model_name, curve)
    result["baselines"] = {k: run_baseline_on_curve(k, curve)["summary"]
                           for k in ("no_ely", "flat", "curtail_first")}
    result["curve"] = curve
    result["case"] = {k: case[k] for k in ("id", "name", "tagline", "desc", "region")}
    result["model"] = model_name
    return JSONResponse(result)


# ============================================================
# 在线训练（后台线程 + SSE）—— 与 src/train.py 同一训练范式
# ============================================================
def _train_worker(steps_total: int, n_envs: int, seed: int, outdir: str):
    from src.models.vector_env import VectorAGCEnv
    from src.agents.ppo import PPO, PPOBuffer

    try:
        TRAIN_STATE.update(running=True, should_stop=False, finished=False, error=None,
                           started_at=time.time(), log=[],
                           config={"steps": steps_total, "n_envs": n_envs,
                                   "seed": seed, "outdir": outdir})
        n_actions = cfg.PPO["n_actions"]
        env = VectorAGCEnv(n_envs=n_envs, seed=seed)
        ppo = PPO(env.state_dim, n_actions, seed=seed)
        buf_cap = n_envs * cfg.SIM_STEPS
        buf = PPOBuffer(env.state_dim, env.action_dim, buf_cap,
                        n_envs=n_envs, n_actions=n_actions)

        s = env.reset()
        ep_rew = np.zeros(n_envs)
        steps, ep_cnt = 0, 0
        ckpt_dir = PROJECT_ROOT / outdir / "data"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        final_path = ckpt_dir / "ppo_agent.pt"

        while steps < steps_total and not TRAIN_STATE["should_stop"]:
            mask = env.action_mask() if getattr(cfg, "ACTION_MASK", False) else None
            a, logp, v = ppo.select_action_batch(s, mask=mask)
            a_power = ppo.action_to_power(a)
            ns, r, done, info = env.step(a_power.reshape(-1, 1))
            r_scaled = r / ppo.reward_scale
            buf.store(s, a, r_scaled.astype(np.float32), done.astype(np.float32),
                      logp.astype(np.float32), v.astype(np.float32), mask)

            ep_rew += r
            steps += n_envs
            ppo.steps_done = steps

            if done.all():
                ep_cnt += 1
                utils, h2s = [], []
                for i in range(n_envs):
                    smi = env.summary(i)
                    utils.append(smi["renewable_utilization"])
                    h2s.append(smi["h2_t"])
                r_mean_wan = float(ep_rew.mean() / 1e4)
                TRAIN_STATE["log"].append({
                    "episode": ep_cnt, "step": steps,
                    "reward_mean": round(r_mean_wan, 2),
                    "util_pct": round(100 * float(np.mean(utils)), 1),
                    "h2_t": round(float(np.mean(h2s)), 2),
                    "lr": float(ppo.opt.param_groups[0]["lr"]),
                })
                ep_rew[:] = 0.0
                new_seeds = seed + np.random.randint(0, 10 ** 9, size=n_envs)
                cfg.SEASON_MONTH = int(np.random.choice([1, 4, 6, 7, 10, 12]))
                cfg.REGION = str(np.random.choice(["northwest", "northeast", "northchina"]))
                cfg.WEATHER_MODE = str(np.random.choice(["sunny", "cloudy", "overcast", "rain"]))
                s = env.reset(seeds=new_seeds)

            if buf.size >= buf_cap:
                last_v = np.zeros(n_envs, dtype=np.float32)
                last_d = np.ones(n_envs, dtype=np.float32)
                ppo.update(buf, last_v, last_d)
                buf = PPOBuffer(env.state_dim, env.action_dim, buf_cap,
                                n_envs=n_envs, n_actions=n_actions)

        ppo.save(str(final_path))
        _AGENT_CACHE.clear()
        scan_models()
        TRAIN_STATE.update(running=False, finished=True,
                           final_model=str(final_path.relative_to(PROJECT_ROOT)),
                           last_step=steps)
    except Exception as e:
        TRAIN_STATE.update(running=False, finished=True,
                           error=f"{e}\n{traceback.format_exc()[-800:]}")


@app.post("/api/train/start")
def api_train_start(steps: int = Form(200000), n_envs: int = Form(16),
                    seed: int = Form(42), outdir: str = Form("output_web")):
    if TRAIN_STATE["running"]:
        raise HTTPException(409, "已有训练任务在运行，请先停止或等待完成")
    outdir = "".join(ch for ch in outdir if ch.isalnum() or ch in "_-/") or "output_web"
    t = threading.Thread(target=_train_worker, args=(int(steps), int(n_envs), int(seed), outdir),
                         daemon=True)
    TRAIN_STATE["thread"] = t
    t.start()
    return {"ok": True, "msg": f"训练已启动: {steps} 步 × {n_envs} 环境 → {outdir}/data/ppo_agent.pt"}


@app.post("/api/train/stop")
def api_train_stop():
    if not TRAIN_STATE["running"]:
        return {"ok": False, "msg": "没有正在运行的训练"}
    TRAIN_STATE["should_stop"] = True
    return {"ok": True, "msg": "已发送停止信号，本幕结束后保存并退出"}


@app.get("/api/train/status")
def api_train_status():
    return {k: v for k, v in TRAIN_STATE.items() if k != "thread"}


@app.get("/api/train/stream")
async def api_train_stream():
    async def gen():
        sent = 0
        while True:
            payload = {
                "running": TRAIN_STATE["running"],
                "finished": TRAIN_STATE["finished"],
                "error": TRAIN_STATE["error"],
                "config": TRAIN_STATE["config"],
                "final_model": TRAIN_STATE["final_model"],
                "new_points": TRAIN_STATE["log"][sent:],
            }
            sent = len(TRAIN_STATE["log"])
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if TRAIN_STATE["finished"]:
                yield "event: end\ndata: {}\n\n"
                break
            await asyncio.sleep(1.0)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ============================================================
# 实时 AGC 仿真（WebSocket）
# ============================================================
class SimSession:
    """单客户端实时仿真会话：AGCEnv + 训练策略，逐 15min 滚动决策"""

    def __init__(self, model_name: str, seed: int = 3):
        from src.models.grid_env import AGCEnv

        self.agent = load_agent(model_name)
        self.env = AGCEnv(seed=seed)
        self.env.reset()
        self.obs = self.env._state()
        self.done = False
        self.t = 0

    def step(self) -> dict:
        if self.done:
            return {"done": True}
        idx = _single_action(self.agent, self.obs)
        power = float(self.agent.action_to_power(np.array([idx]))[0])
        ns, r, done, info = self.env.step(np.array([power], dtype=np.float32))
        self.t += 1
        self.done = bool(done)
        e = self.env
        i = (self.t - 1) % cfg.SIM_STEPS
        ren = e.data["pv"][i] + e.data["wind"][i]
        used = max(ren - max(0.0, ren - e.data["load"][i] - e.ely.power), 1e-9)
        out = {
            "done": False,
            "t": self.t - 1, "hour": round(i * 0.25, 2),
            "action_level": idx, "action_power": round(power * 100, 1),
            "pv": round(float(e.data["pv"][i]), 1),
            "wind": round(float(e.data["wind"][i]), 1),
            "load": round(float(e.data["load"][i]), 1),
            "ely_mw": round(e.ely.power, 1),
            "th_mw": round(e.thermal.total_power(), 1),
            "curtail_mw": round(max(0.0, ren - e.data["load"][i] - e.ely.power), 1),
            "step_reward_wan": round(r / self.agent.reward_scale, 3),
            "cum_reward_wan": round(e.total_reward / self.agent.reward_scale, 2),
            "cum_h2_t": round(e.ely.h2_produced_kg / 1000, 3),
            "cum_nh3_t": round(e.ammonia.nh3_produced_t, 3),
            "utilization": round(min(1.0, used / max(ren, 1e-9)), 4),
        }
        if self.done:
            out["done"] = True
            sm = e.summary()
            out["summary"] = {"total_reward_wan": round(sm["total_reward"] / self.agent.reward_scale, 2),
                              "h2_t": round(sm["h2_kg"] / 1000, 2),
                              "nh3_t": round(sm["nh3_t"], 2),
                              "utilization_pct": round(sm["renewable_utilization"] * 100, 1)}
        else:
            self.obs = ns
        return out


@app.websocket("/ws/simulate")
async def ws_simulate(ws: WebSocket):
    """协议：
       ← {"cmd":"start","model":"..","seed":3,"interval_ms":150} | pause | resume | step | reset | stop
       → SimSession.step() 输出 + {"type":"started|paused|reset"}
    """
    await ws.accept()
    session: SimSession | None = None
    auto_task = None

    async def auto_loop(interval_ms: int):
        try:
            while session and not session.done:
                await ws.send_json(session.step())
                await asyncio.sleep(max(0.02, interval_ms / 1000))
            if session and session.done:
                await ws.send_json(session.step())   # 带 summary 的终帧
        except Exception:
            pass

    try:
        while True:
            msg = await ws.receive_json()
            cmd = msg.get("cmd")
            if cmd == "start":
                if auto_task:
                    auto_task.cancel()
                model = msg.get("model") or resolve_default_model()
                session = SimSession(model, int(msg.get("seed", 3)))
                interval = int(msg.get("interval_ms", 150))
                auto_task = asyncio.create_task(auto_loop(interval))
                await ws.send_json({"type": "started", "model": model, "interval_ms": interval})
            elif cmd == "pause":
                if auto_task:
                    auto_task.cancel(); auto_task = None
                await ws.send_json({"type": "paused"})
            elif cmd == "resume":
                if session and auto_task is None and not session.done:
                    interval = int(msg.get("interval_ms", 150))
                    auto_task = asyncio.create_task(auto_loop(interval))
            elif cmd == "step":
                if session:
                    await ws.send_json(session.step())
            elif cmd == "reset":
                if auto_task:
                    auto_task.cancel(); auto_task = None
                session = SimSession(resolve_default_model(),
                                     int(msg.get("seed", int(time.time()) % 99991)))
                await ws.send_json({"type": "reset"})
            elif cmd == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if auto_task:
            auto_task.cancel()


# ============================================================
# 前端托管与启动
# ============================================================
WEB_DIR = PROJECT_ROOT / "server" / "web"


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB_DIR / "index.html").read_text(encoding="utf-8")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.on_event("startup")
def startup():
    scan_models()
    dev = _torch_device()
    print(f"[platform] 已注册 {len(MODEL_REGISTRY)} 个模型 | 设备: {dev}")
    print("[platform] 打开 http://127.0.0.1:8000 进入调度平台")


if __name__ == "__main__":
    uvicorn.run("server.main:app", host="127.0.0.1", port=8000, reload=False)
