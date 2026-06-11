"""Model-based RL for DeepTraffic under the SAME I/O as the DQN benchmark.

Experimental design
-------------------
From a single, budgeted stream of environment interaction we learn:

  * ``QNet``      -- a model-free action-value function (a DQN on the real data),
  * ``DynModel``  -- a learned forward model  g(s,a) -> (Δs, r).

We then evaluate two policies built from these *same* learned components:

  * ``pi_Q``   : reactive  argmax_a Q(s,a)                       (no planning)
  * ``pi_MPC`` : receding-horizon planning in the LEARNED model,
                 scoring action sequences by  Σ γ^t r̂_t + γ^H max_a Q(s_H,a).

The difference ``pi_MPC - pi_Q`` isolates the contribution of *lookahead* under
identical observation space, action space, and data budget. The external anchor
is the real top DeepTraffic submission, which scores **74.05 mph** on this engine.

Hard constraints (the scientific claim depends on these):
  * Only the .s() observation (here 315-dim) and the 5 discrete actions are used.
  * Planning happens ONLY in the learned model -- env.step is never called to look
    ahead. The env is used solely to (a) collect data and (b) final evaluate().

CPU-only, deterministic seeding throughout.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deeptraffic.env import DeepTrafficEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # --- fixed env config (matches the 74.05 top submission, single-ego) ---
    lanes_side: int = 3
    patches_ahead: int = 40
    patches_behind: int = 5
    other_agents: int = 0
    frames: int = 2000

    # --- interaction budget (env decision-steps; 1 step = 30 frames) ---
    budget: int = 60_000
    warmup_steps: int = 4_000          # random data before learning kicks in
    steps_per_iter: int = 4_000        # env steps collected per outer iteration

    # --- Q (DQN) ---
    q_hidden: Tuple[int, ...] = (256, 256)
    q_lr: float = 1e-3
    gamma: float = 0.95
    q_batch: int = 256
    q_updates_per_iter: int = 2_000
    target_update_every: int = 500     # gradient steps between target syncs

    # --- dynamics model ---
    m_hidden: Tuple[int, ...] = (512, 512)
    m_lr: float = 1e-3
    m_batch: int = 256
    m_updates_per_iter: int = 2_000

    # --- planner ---
    horizon: int = 4                   # MPC depth (exhaustive 5^H sequences)

    # --- exploration ---
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay_steps: int = 40_000

    # --- eval ---
    eval_runs: int = 30
    eval_every_iters: int = 1

    # --- misc ---
    buffer_size: int = 400_000
    seed: int = 0
    out: str = "runs/mbrl/exp"

    @property
    def obs_dim(self) -> int:
        return (self.lanes_side * 2 + 1) * (self.patches_ahead + self.patches_behind)

    n_actions: int = 5


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def mlp(sizes: List[int], act=nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: Tuple[int, ...]):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class DynModel(nn.Module):
    """g(s,a) -> (Δs, r). Targets are standardized; ``predict`` returns real units."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: Tuple[int, ...]):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.trunk = mlp([obs_dim + n_actions, *hidden])
        self.head_delta = nn.Linear(hidden[-1], obs_dim)
        self.head_r = nn.Linear(hidden[-1], 1)
        # standardization buffers (set from data before training)
        self.register_buffer("delta_mean", torch.zeros(obs_dim))
        self.register_buffer("delta_std", torch.ones(obs_dim))
        self.register_buffer("r_mean", torch.zeros(1))
        self.register_buffer("r_std", torch.ones(1))

    def _trunk(self, s: torch.Tensor, a_onehot: torch.Tensor) -> torch.Tensor:
        return F.relu(self.trunk(torch.cat([s, a_onehot], dim=-1)))

    def forward(self, s, a_onehot):  # standardized outputs (for training)
        h = self._trunk(s, a_onehot)
        return self.head_delta(h), self.head_r(h)

    @torch.no_grad()
    def predict(self, s, a_onehot):  # real-unit (Δs, r)
        d_std, r_std = self.forward(s, a_onehot)
        delta = d_std * self.delta_std + self.delta_mean
        r = (r_std * self.r_std + self.r_mean).squeeze(-1)
        return delta, r


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.idx = 0
        self.full = False

    def add(self, s, a, r, s2, d):
        i = self.idx
        self.s[i] = s; self.a[i] = a; self.r[i] = r; self.s2[i] = s2; self.d[i] = d
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, n, rng: np.random.Generator):
        hi = len(self)
        idx = rng.integers(0, hi, size=n)
        return (self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx])

    def all_indices(self):
        return np.arange(len(self))


# ---------------------------------------------------------------------------
# Agent: holds Q + model, exposes reactive and MPC policies
# ---------------------------------------------------------------------------
class Agent:
    def __init__(self, cfg: Config, device="cpu"):
        self.cfg = cfg
        self.device = device
        self.q = QNet(cfg.obs_dim, cfg.n_actions, cfg.q_hidden).to(device)
        self.q_target = QNet(cfg.obs_dim, cfg.n_actions, cfg.q_hidden).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.model = DynModel(cfg.obs_dim, cfg.n_actions, cfg.m_hidden).to(device)
        # precompute all action sequences for exhaustive MPC: [A^H, H]
        self.seqs = self._make_sequences(cfg.n_actions, cfg.horizon).to(device)
        # one-hot cache for sequence columns
        self._eye = torch.eye(cfg.n_actions, device=device)

    @staticmethod
    def _make_sequences(n_actions: int, horizon: int) -> torch.Tensor:
        grids = torch.meshgrid(*[torch.arange(n_actions) for _ in range(horizon)],
                               indexing="ij")
        return torch.stack([g.reshape(-1) for g in grids], dim=1)  # [A^H, H]

    # --- reactive policy ---
    @torch.no_grad()
    def act_q(self, obs: np.ndarray) -> int:
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.q(s).argmax(dim=1).item())

    # --- MPC policy (plans only in the learned model) ---
    @torch.no_grad()
    def act_mpc(self, obs: np.ndarray) -> int:
        cfg = self.cfg
        N = self.seqs.shape[0]
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        s = s.unsqueeze(0).expand(N, -1).contiguous()          # [N, obs]
        ret = torch.zeros(N, device=self.device)
        g = 1.0
        for t in range(cfg.horizon):
            a = self.seqs[:, t]                                # [N]
            a_oh = self._eye[a]                                # [N, A]
            delta, r = self.model.predict(s, a_oh)
            ret = ret + g * r
            s = s + delta
            g *= cfg.gamma
        qN = self.q(s).max(dim=1).values                       # terminal value
        ret = ret + g * qN
        best = int(ret.argmax().item())
        return int(self.seqs[best, 0].item())

    def eps_greedy(self, obs: np.ndarray, eps: float, rng: np.random.Generator) -> int:
        if rng.random() < eps:
            return int(rng.integers(0, self.cfg.n_actions))
        return self.act_q(obs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def set_model_norm(model: DynModel, buf: ReplayBuffer):
    """Set standardization buffers from the current replay data."""
    n = len(buf)
    s = buf.s[:n]; s2 = buf.s2[:n]; r = buf.r[:n]
    delta = s2 - s
    dm = delta.mean(axis=0); ds = delta.std(axis=0) + 1e-6
    rm = float(r.mean()); rs = float(r.std()) + 1e-6
    model.delta_mean.copy_(torch.as_tensor(dm, dtype=torch.float32))
    model.delta_std.copy_(torch.as_tensor(ds, dtype=torch.float32))
    model.r_mean.copy_(torch.as_tensor([rm], dtype=torch.float32))
    model.r_std.copy_(torch.as_tensor([rs], dtype=torch.float32))


def train_model(agent: Agent, buf: ReplayBuffer, opt, rng, updates: int) -> float:
    cfg = agent.cfg
    m = agent.model
    eye = torch.eye(cfg.n_actions)
    set_model_norm(m, buf)
    last = 0.0
    for _ in range(updates):
        s, a, r, s2, d = buf.sample(cfg.m_batch, rng)
        s_t = torch.as_tensor(s); s2_t = torch.as_tensor(s2)
        a_oh = eye[torch.as_tensor(a)]
        delta_tgt = (s2_t - s_t - m.delta_mean) / m.delta_std
        r_tgt = (torch.as_tensor(r).unsqueeze(-1) - m.r_mean) / m.r_std
        d_pred, r_pred = m(s_t, a_oh)
        loss = F.mse_loss(d_pred, delta_tgt) + F.mse_loss(r_pred, r_tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
    return last


def train_q(agent: Agent, buf: ReplayBuffer, opt, rng, updates: int, grad_step0: int) -> float:
    cfg = agent.cfg
    q, qt = agent.q, agent.q_target
    last = 0.0
    for u in range(updates):
        s, a, r, s2, d = buf.sample(cfg.q_batch, rng)
        s_t = torch.as_tensor(s); s2_t = torch.as_tensor(s2)
        a_t = torch.as_tensor(a); r_t = torch.as_tensor(r); d_t = torch.as_tensor(d)
        qsa = q(s_t).gather(1, a_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double-DQN target
            a2 = q(s2_t).argmax(dim=1, keepdim=True)
            q2 = qt(s2_t).gather(1, a2).squeeze(1)
            tgt = r_t + cfg.gamma * (1.0 - d_t) * q2
        loss = F.smooth_l1_loss(qsa, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.item())
        if (grad_step0 + u + 1) % cfg.target_update_every == 0:
            qt.load_state_dict(q.state_dict())
    return last


def collect(env: DeepTrafficEnv, agent: Agent, buf: ReplayBuffer, n_steps: int,
            step0: int, cfg: Config, rng: np.random.Generator,
            np_seed_rng: np.random.Generator) -> int:
    """Collect ~n_steps env decision-steps with epsilon-greedy on Q. Returns steps used."""
    used = 0
    while used < n_steps:
        obs = env.reset(deterministic=False, seed=int(np_seed_rng.integers(1, 2**31 - 1)))
        done = False
        while not done:
            eps = max(cfg.eps_end,
                      cfg.eps_start - (cfg.eps_start - cfg.eps_end)
                      * (step0 + used) / max(1, cfg.eps_decay_steps))
            a = agent.eps_greedy(obs, eps, rng)
            obs2, r, done, info = env.step(a)
            buf.add(obs, a, r, obs2, float(done))
            obs = obs2
            used += 1
            if used >= n_steps:
                break
    return used


def evaluate_policy(env: DeepTrafficEnv, policy, runs: int, frames: int) -> dict:
    return env.evaluate(policy, runs=runs, frames=frames, deterministic=True)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(cfg: Config) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    seed_rng = np.random.default_rng(cfg.seed + 12345)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    os.makedirs(cfg.out, exist_ok=True)
    log_path = os.path.join(cfg.out, "log.jsonl")
    logf = open(log_path, "a")

    env = DeepTrafficEnv(lanes_side=cfg.lanes_side, patches_ahead=cfg.patches_ahead,
                         patches_behind=cfg.patches_behind, other_agents=cfg.other_agents,
                         frames=cfg.frames)
    assert env.num_inputs == cfg.obs_dim, (env.num_inputs, cfg.obs_dim)

    agent = Agent(cfg)
    buf = ReplayBuffer(cfg.buffer_size, cfg.obs_dim)
    q_opt = torch.optim.Adam(agent.q.parameters(), lr=cfg.q_lr)
    m_opt = torch.optim.Adam(agent.model.parameters(), lr=cfg.m_lr)

    steps = 0
    grad_steps = 0
    best = {"pi_q": -1.0, "pi_mpc": -1.0}
    t0 = time.time()

    # warmup: random data
    steps += collect(env, agent, buf, cfg.warmup_steps, steps, cfg, rng, seed_rng)
    print(f"[warmup] collected {steps} steps, buffer={len(buf)}", flush=True)

    it = 0
    while steps < cfg.budget:
        it += 1
        # collect on-policy(ish)
        used = collect(env, agent, buf,
                       min(cfg.steps_per_iter, cfg.budget - steps),
                       steps, cfg, rng, seed_rng)
        steps += used
        # learn
        m_loss = train_model(agent, buf, m_opt, rng, cfg.m_updates_per_iter)
        q_loss = train_q(agent, buf, q_opt, rng, cfg.q_updates_per_iter, grad_steps)
        grad_steps += cfg.q_updates_per_iter

        rec = {"iter": it, "steps": steps, "buffer": len(buf),
               "q_loss": round(q_loss, 5), "m_loss": round(m_loss, 5),
               "elapsed": round(time.time() - t0, 1)}

        if it % cfg.eval_every_iters == 0:
            rq = evaluate_policy(env, agent.act_q, cfg.eval_runs, cfg.frames)
            rm = evaluate_policy(env, agent.act_mpc, cfg.eval_runs, cfg.frames)
            rec["pi_q_median"] = rq["median"]; rec["pi_q_mean"] = round(rq["mean"], 2)
            rec["pi_mpc_median"] = rm["median"]; rec["pi_mpc_mean"] = round(rm["mean"], 2)
            if rq["median"] > best["pi_q"]:
                best["pi_q"] = rq["median"]
            if rm["median"] > best["pi_mpc"]:
                best["pi_mpc"] = rm["median"]
                torch.save({"cfg": asdict(cfg), "q": agent.q.state_dict(),
                            "model": agent.model.state_dict()},
                           os.path.join(cfg.out, "best_mpc.pt"))
            print(f"[it {it:3d}] steps={steps:>7} | q_loss={q_loss:.4f} m_loss={m_loss:.4f} "
                  f"| pi_Q={rq['median']:.2f} pi_MPC={rm['median']:.2f} "
                  f"(best Q={best['pi_q']:.2f} MPC={best['pi_mpc']:.2f}) "
                  f"| {rec['elapsed']:.0f}s", flush=True)
        else:
            print(f"[it {it:3d}] steps={steps:>7} | q_loss={q_loss:.4f} m_loss={m_loss:.4f}",
                  flush=True)

        logf.write(json.dumps(rec) + "\n"); logf.flush()

    logf.close()
    summary = {"best": best, "anchor_dqn": 74.05, "steps": steps,
               "elapsed": round(time.time() - t0, 1), "out": cfg.out}
    with open(os.path.join(cfg.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] best pi_Q={best['pi_q']:.2f}  best pi_MPC={best['pi_mpc']:.2f}  "
          f"(DQN anchor 74.05)", flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run")
    p.add_argument("--budget", type=int)
    p.add_argument("--horizon", type=int)
    p.add_argument("--gamma", type=float)
    p.add_argument("--eval-runs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--out", type=str)
    args = p.parse_args()

    cfg = Config()
    if args.smoke:
        cfg.budget = 4_000
        cfg.warmup_steps = 1_500
        cfg.steps_per_iter = 2_500
        cfg.q_updates_per_iter = 400
        cfg.m_updates_per_iter = 400
        cfg.horizon = 3
        cfg.eval_runs = 8
        cfg.out = "runs/mbrl/smoke"
    if args.budget is not None: cfg.budget = args.budget
    if args.horizon is not None: cfg.horizon = args.horizon
    if args.gamma is not None: cfg.gamma = args.gamma
    if args.eval_runs is not None: cfg.eval_runs = args.eval_runs
    if args.seed is not None: cfg.seed = args.seed
    if args.out is not None: cfg.out = args.out

    print("config:", json.dumps(asdict(cfg), default=str), flush=True)
    train(cfg)


if __name__ == "__main__":
    main()
