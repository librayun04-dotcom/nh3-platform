# -*- coding: utf-8 -*-
"""
PPO（Proximal Policy Optimization）—— 离散动作版本（推荐用于 AGC 调度）
================================================================
为什么用离散动作：
  - AGC 对电解槽功率的下发本来就是档位式指令（如 0/25%/50%/75%/100%）
  - 最优调度策略含"开关"行为（谷段/弃电时段开，峰段停）
  - 连续动作策略（高斯/Beta）在有限训练下难以表达"开关"，易陷于"全天开"局部最优
  - Categorical 策略天然支持离散档位选择，能直接学出"峰段停机"

动作空间：K 档功率 [0, 1/(K-1), ..., 1]（默认 5 档：0/25%/50%/75%/100%）

依据说明书："利用深度强化学习近端策略优化（PPO）算法对AGC进行优化"
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as D

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import config as cfg


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Categorical Actor-Critic：动作 = 功率档位索引
    注意：Actor 与 Critic 使用独立网络，避免价值梯度淹没策略梯度
    （共享特征网络会导致特征坍缩——价值损失远大于策略损失时）"""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.n_actions = action_dim
        # Actor 网络（独立）
        self.actor = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
        )
        self.policy_head = layer_init(nn.Linear(hidden, action_dim), std=0.01)
        # Critic 网络（独立，价值量级大，单独学习）
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden), std=1.0), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden), std=1.0), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden), std=1.0), nn.Tanh(),
        )
        self.v_head = layer_init(nn.Linear(hidden, 1), std=1.0)

    def get_dist(self, x, mask=None):
        logits = self.policy_head(self.actor(x))
        if mask is not None:
            # 方案H：动作掩码——非法档位 logits 置 -1e8（softmax 后概率≈0）
            logits = logits.masked_fill(mask <= 0.5, -1e8)
        return D.Categorical(logits=logits)

    def act(self, x, greedy: bool = False, mask=None):
        dist = self.get_dist(x, mask)
        v = self.v_head(self.critic(x)).squeeze(-1)
        a = dist.probs.argmax(dim=-1) if greedy else dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v

    def evaluate(self, x, a, mask=None):
        dist = self.get_dist(x, mask)
        v = self.v_head(self.critic(x)).squeeze(-1)
        logp = dist.log_prob(a)
        entropy = dist.entropy()
        return logp, entropy, v


class PPOBuffer:
    """批量经验缓冲区（按时间步批量 push，GAE 向量化）
    方案H扩展：额外存储每个样本的动作掩码 mask（(capacity, n_actions)），
    保证更新时 evaluate 使用与采样时相同的分布（否则 logp 不一致导致概率比错误）"""

    def __init__(self, state_dim: int, action_dim: int, capacity: int, n_envs: int = 1,
                 n_actions: int = None):
        self.cap = capacity
        self.n_envs = n_envs
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((capacity, action_dim), dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.logp = np.zeros(capacity, dtype=np.float32)
        self.v = np.zeros(capacity, dtype=np.float32)
        self.mask = np.zeros((capacity, n_actions or action_dim), dtype=np.float32)
        self.prio = np.ones(capacity, dtype=np.float32)   # 方案G：优先级
        self.ptr = 0
        self.size = 0

    def store(self, s, a, r, d, logp, v, mask=None):
        n = len(s)
        idx = (self.ptr + np.arange(n)) % self.cap
        self.s[idx] = s
        self.a[idx] = a.reshape(-1, 1) if a.ndim == 1 else a
        self.r[idx] = r; self.d[idx] = d
        self.logp[idx] = logp; self.v[idx] = v
        if mask is not None:
            self.mask[idx] = mask
        if self.prio is not None:
            self.prio[idx] = np.ones(n, dtype=np.float32)   # 新样本默认高优先级
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def finish_path(self, last_v: np.ndarray, gamma=0.99, lam=0.95):
        """GAE 优势估计（按时间步向量化）"""
        n_envs = self.n_envs
        T = self.size // n_envs
        if T * n_envs != self.size or T == 0:
            T = self.size
            r = self.r[:self.size]; v = self.v[:self.size]; d = self.d[:self.size]
            adv = np.zeros(T, dtype=np.float32)
            last_adv = np.zeros(1, dtype=np.float32)
            last_d = np.ones(1, dtype=np.float32)
            for t in reversed(range(T)):
                next_v = v[t + 1] if t + 1 < T else last_v
                next_d = d[t + 1] if t + 1 < T else last_d
                mask = 1.0 - d[t]
                delta = r[t] + gamma * next_v * (1 - next_d) - v[t]
                adv[t] = last_adv = (delta + gamma * lam * mask * last_adv) * mask
                last_d = d[t]
            return adv
        r = self.r[:self.size].reshape(T, n_envs)
        v = self.v[:self.size].reshape(T, n_envs)
        d = self.d[:self.size].reshape(T, n_envs)
        adv = np.zeros((T, n_envs), dtype=np.float32)
        last_adv = np.zeros(n_envs, dtype=np.float32)
        last_d = np.ones(n_envs, dtype=np.float32)
        for t in reversed(range(T)):
            next_v = v[t + 1] if t + 1 < T else last_v
            next_d = d[t + 1] if t + 1 < T else last_d
            mask = 1.0 - d[t]
            delta = r[t] + gamma * next_v * (1 - next_d) - v[t]
            adv[t] = last_adv = (delta + gamma * lam * mask * last_adv) * mask
            last_d = d[t]
        return adv.reshape(-1)


class PPO:
    def __init__(self, state_dim: int, action_dim: int, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        p = cfg.PPO
        self.gamma, self.lam = p["gamma"], p["gae_lambda"]
        self.clip, self.ent_coef, self.vf_coef = p["clip_ratio"], p["entropy_coef"], p["vf_coef"]
        self.epochs, self.batch = p["epochs"], p["batch_size"]
        self.reward_scale = p.get("reward_scale", 1e4)
        self.n_actions = action_dim
        # 离散档位功率（动作索引 -> 0~1 标幺）
        self.actions_map = np.linspace(0.0, 1.0, action_dim, dtype=np.float32)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = ActorCritic(state_dim, action_dim, p["hidden_dim"]).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=p["lr"], eps=1e-5)
        self.lr_decay = p.get("lr_decay", False)
        self.lr_init = p["lr"]
        self.steps_done = 0
        self.total_steps = p["total_timesteps"]
        self.log = {"update": [], "loss_pi": [], "loss_v": [], "entropy": [],
                    "kl": [], "lr": [], "reward_mean": []}
        self.update_count = 0
        # ---- 方案B：自适应熵温度（MaxEnt-PPO）----
        self.auto_entropy = bool(getattr(cfg, "AUTO_ENTROPY", False))
        if self.auto_entropy:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.target_ent = float(-np.log(self.n_actions) * cfg.ENTROPY_TARGET_COEF)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=cfg.ENTROPY_ALPHA_LR)
            self.alpha_clip = cfg.ENTROPY_ALPHA_CLIP
            self.log["alpha"] = []
        # ---- 方案G：优先经验回放（PER）----
        self.per = bool(getattr(cfg, "PER", False))
        self.per_alpha = getattr(cfg, "PER_ALPHA", 0.6)
        self.per_beta_init = getattr(cfg, "PER_BETA_INIT", 0.4)
        self.per_beta = self.per_beta_init
        self.per_beta_final = getattr(cfg, "PER_BETA_FINAL", 1.0)
        print(f"PPO(离散{action_dim}档) 网络初始化于 {self.device}（{sum(p.numel() for p in self.net.parameters())} 参数）"
              + (" | 方案B自适应熵" if self.auto_entropy else "")
              + (" | 方案G-PER" if self.per else ""))

    def select_action_batch(self, s: np.ndarray, greedy=False, mask=None) -> tuple:
        """返回 (档位索引, logp, v)；功率值 = actions_map[idx]
        mask: (batch, n_actions) 0/1，方案H领域知识动作掩码"""
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        mask_t = None
        if mask is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            # 防御：全 0 掩码的样本退化为无掩码（避免 logits 全 -inf -> NaN）
            zero_rows = mask_t.sum(dim=-1) <= 0.5
            if zero_rows.any():
                mask_t = mask_t.clone()
                mask_t[zero_rows] = 1.0
        with torch.no_grad():
            a, logp, v = self.net.act(s_t, greedy, mask_t)
        return a.cpu().numpy().astype(np.int64), logp.cpu().numpy(), v.cpu().numpy()

    def action_to_power(self, a: np.ndarray) -> np.ndarray:
        """档位索引 -> 功率标幺 [0,1]"""
        return self.actions_map[np.asarray(a)]

    def value(self, s: np.ndarray) -> np.ndarray:
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.net.v_head(self.net.critic(s_t)).squeeze(-1)
        return v.cpu().numpy()

    def _decay_lr(self):
        if not self.lr_decay or self.total_steps <= 0:
            return
        frac = max(0.1, 1.0 - self.steps_done / self.total_steps)
        for g in self.opt.param_groups:
            g["lr"] = self.lr_init * frac

    def update(self, buf: PPOBuffer, last_v: np.ndarray, last_d: np.ndarray):
        self._decay_lr()
        t0 = time.time()
        adv = buf.finish_path(last_v, self.gamma, self.lam)
        ret = adv + buf.v[:buf.size]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        s = torch.as_tensor(buf.s[:buf.size], device=self.device)
        a = torch.as_tensor(buf.a[:buf.size].reshape(-1), device=self.device, dtype=torch.long)
        old_logp = torch.as_tensor(buf.logp[:buf.size], device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)
        mask_t = torch.as_tensor(buf.mask[:buf.size], device=self.device) if buf.mask is not None else None

        # ---- 方案G：PER 优先级（按 |优势| 更新）与加权采样 ----
        if self.per and buf.prio is not None:
            buf.prio[:buf.size] = (np.abs(adv) + cfg.PER_EPS) ** self.per_alpha
            probs = buf.prio[:buf.size] / buf.prio[:buf.size].sum()
            is_weight = (1.0 / (buf.size * probs + 1e-12)) ** self.per_beta
            is_weight = torch.as_tensor(is_weight / is_weight.max(), dtype=torch.float32, device=self.device)
        else:
            is_weight = None

        idx = np.arange(buf.size)
        kl_sum, n_batches = 0.0, 0
        for _ in range(self.epochs):
            if self.per:
                idx = np.random.choice(buf.size, size=buf.size, replace=False, p=probs)
            else:
                np.random.shuffle(idx)
            for start in range(0, buf.size, self.batch):
                b = idx[start:start + self.batch]
                bm = mask_t[b] if mask_t is not None else None
                logp, ent, v = self.net.evaluate(s[b], a[b], bm)
                ratio = (logp - old_logp[b]).exp()
                clip_adv = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[b]
                loss_pi = -(torch.min(ratio * adv_t[b], clip_adv)).mean()
                loss_v = ((v - ret_t[b]) ** 2).mean()
                # ---- 方案B：自适应熵温度（替换固定熵系数）----
                if self.auto_entropy:
                    alpha = self.log_alpha.exp().clamp(*self.alpha_clip).detach()
                    ent_loss = -alpha * ent.mean()
                else:
                    ent_loss = -self.ent_coef * ent.mean()
                loss = loss_pi + self.vf_coef * loss_v + ent_loss
                if is_weight is not None:
                    # PER 重要性采样权重修正（作用于策略与价值损失）
                    wb = is_weight[b]
                    loss = (loss_pi * wb).mean() + self.vf_coef * (loss_v * wb).mean() + ent_loss
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                with torch.no_grad():
                    kl_sum += (old_logp[b] - logp).mean().item()
                n_batches += 1

        # ---- 方案B：更新温度 α（目标熵正则）----
        if self.auto_entropy:
            with torch.no_grad():
                ent_mean = self.net.evaluate(s, a, mask_t)[1].mean()
            alpha_loss = -(self.log_alpha * (ent_mean - self.target_ent).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.data.clamp_(np.log(self.alpha_clip[0]), np.log(self.alpha_clip[1]))
            cur_alpha = float(self.log_alpha.exp().item())

        # ---- 方案G：PER beta 退火 ----
        if self.per:
            frac = min(1.0, self.steps_done / max(1, self.total_steps))
            self.per_beta = self.per_beta_final * frac + self.per_beta_init * (1 - frac)

        self.update_count += 1
        cur_lr = self.opt.param_groups[0]["lr"]
        self.log["update"].append(self.update_count)
        self.log["loss_pi"].append(loss_pi.item())
        self.log["loss_v"].append(loss_v.item())
        self.log["entropy"].append(ent.mean().item())
        self.log["kl"].append(kl_sum / max(1, n_batches))
        self.log["lr"].append(cur_lr)
        self.log["reward_mean"].append(float(buf.r[:buf.size].mean()))
        if self.auto_entropy:
            self.log["alpha"].append(cur_alpha)
        if self.update_count % 5 == 0 or self.update_count == 1:
            extra = f" alpha={cur_alpha:.4f}" if self.auto_entropy else ""
            print(f"  [更新 {self.update_count}] loss_pi={loss_pi.item():.4f} "
                  f"loss_v={loss_v.item():.1f} ent={ent.mean().item():.3f} "
                  f"kl={kl_sum/max(1,n_batches):.4f} lr={cur_lr:.2e} "
                  f"mean_r={buf.r[:buf.size].mean():.2f}{extra} ({time.time()-t0:.1f}s)")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ckpt = {
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "steps_done": self.steps_done,
            "update_count": self.update_count,
            "log": self.log,
        }
        if self.auto_entropy:
            ckpt["log_alpha"] = self.log_alpha.detach().cpu().item()
        torch.save(ckpt, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])
        self.steps_done = ckpt.get("steps_done", 0)
        self.update_count = ckpt.get("update_count", 0)
        self.log = ckpt.get("log", self.log)
        if self.auto_entropy and "log_alpha" in ckpt:
            with torch.no_grad():
                self.log_alpha.fill_(float(ckpt["log_alpha"]))
        print(f"已恢复 checkpoint: {path}（steps={self.steps_done}）")

    def pretrain_bc(self, s: np.ndarray, a: np.ndarray, mask: np.ndarray = None,
                    epochs: int = 100, lr: float = 1e-3, batch: int = 256,
                    log_every: int = 10) -> list:
        """方案A：行为克隆（BC）预训练。
        以专家轨迹 (s, a) 为监督标签，最小化 Actor 的负对数似然（交叉熵），
        将策略初始化为接近专家先验，随后由 PPO 微调超越专家。
        仅更新 Actor 网络，使用独立优化器，预训练结束后不影响 PPO 主优化器状态。
        返回各 epoch 的平均 BC 损失。"""
        s_t = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(a, dtype=torch.long, device=self.device)
        mask_t = None
        if mask is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.float32, device=self.device)
            zero_rows = mask_t.sum(dim=-1) <= 0.5
            if zero_rows.any():
                mask_t = mask_t.clone()
                mask_t[zero_rows] = 1.0
        bc_opt = optim.Adam(self.net.actor.parameters(), lr=lr)
        losses = []
        n = len(s)
        for epoch in range(epochs):
            idx = np.random.permutation(n)
            ep_loss = 0.0
            n_b = 0
            for start in range(0, n, batch):
                b = idx[start:start + batch]
                logits = self.net.policy_head(self.net.actor(s_t[b]))
                if mask_t is not None:
                    logits = logits.masked_fill(mask_t[b] <= 0.5, -1e8)
                loss = F.cross_entropy(logits, a_t[b])
                bc_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.actor.parameters(), 1.0)
                bc_opt.step()
                ep_loss += loss.item()
                n_b += 1
            mean_loss = ep_loss / max(1, n_b)
            losses.append(mean_loss)
            if (epoch + 1) % log_every == 0 or epoch == 0:
                # 专家动作复现率（argmax 一致比例）
                with torch.no_grad():
                    logits = self.net.policy_head(self.net.actor(s_t))
                    if mask_t is not None:
                        logits = logits.masked_fill(mask_t <= 0.5, -1e8)
                    acc = (logits.argmax(dim=-1) == a_t).float().mean().item()
                print(f"  [BC预训练 {epoch+1}/{epochs}] loss={mean_loss:.4f} "
                      f"专家动作复现率={acc*100:.1f}% lr={lr:.1e}")
        # 预训练后重置主优化器状态（PPO 微调从干净状态开始）
        self.opt = optim.Adam(self.net.parameters(), lr=self.lr_init, eps=1e-5)
        return losses
