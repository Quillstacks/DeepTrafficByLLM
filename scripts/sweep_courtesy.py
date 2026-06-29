import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from deeptraffic.env import DeepTrafficEnv
from deeptraffic.llm.heuristic import make_policy_cfg
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5, other_agents=10)
H4 = dict(strategy="inner_outer", rear_safety=True, drift=False)
configs = {
  "H4 (no courtesy)":            dict(),
  "Y1 yield g4 f60":             dict(yield_neighbor=True, yield_gap=4, yield_fast=60),
  "Y2 yield g5 f55":             dict(yield_neighbor=True, yield_gap=5, yield_fast=55),
  "Y3 yield g5 f55 req-block":   dict(yield_neighbor=True, yield_gap=5, yield_fast=55, yield_require_block=True),
  "Y4 yield g6 f50 (aggressive)":dict(yield_neighbor=True, yield_gap=6, yield_fast=50),
}
for name, extra in configs.items():
    pol = make_policy_cfg(3, 40, 5, **{**H4, **extra})
    t = time.time(); r = env.evaluate(pol, runs=RUNS, frames=2000, deterministic=True)
    print("%-30s median=%.2f mean=%.2f min=%.2f max=%.2f (%.0fs)" % (
        name, r["median"], r["mean"], r["min"], r["max"], time.time()-t), flush=True)
print("ref: H4 76.06 | DQN 74.13 | ceiling 76.3 | per-car max 80")
