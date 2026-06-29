import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from deeptraffic.env import DeepTrafficEnv
from deeptraffic.llm.heuristic import make_policy_cfg
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5, other_agents=10)
base = dict(strategy="inner_outer", rear_safety=True, drift=False)  # = H4
configs = {
  "bg12 sm3 (=H4)":   dict(block_gap=12, switch_margin=3),
  "bg16 sm3":         dict(block_gap=16, switch_margin=3),
  "bg20 sm3":         dict(block_gap=20, switch_margin=3),
  "bg16 sm2":         dict(block_gap=16, switch_margin=2),
  "bg20 sm2 rear50":  dict(block_gap=20, switch_margin=2, rear_fast=50.0),
}
for name, extra in configs.items():
    pol = make_policy_cfg(3, 40, 5, **{**base, **extra})
    t = time.time(); r = env.evaluate(pol, runs=RUNS, frames=2000, deterministic=True)
    print("%-20s median=%.2f mean=%.2f min=%.2f max=%.2f (%.0fs)" % (
        name, r["median"], r["mean"], r["min"], r["max"], time.time()-t), flush=True)
print("ref: DQN 74.13 | leaderboard#1 75.66 | H4 76.02 | ceiling 76.3")
