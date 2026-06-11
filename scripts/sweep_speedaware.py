import sys, time
sys.path.insert(0, "/Users/markschutera/Documents/Prof/Research/MITDeepDrive/src")
from deeptraffic.env import DeepTrafficEnv
from deeptraffic.llm.heuristic import make_policy_cfg
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5, other_agents=10)
base = dict(strategy="inner_outer", rear_safety=True, drift=False)
configs = {
  "H4 baseline (no SA)":   dict(),
  "H5 speed-aware t70":     dict(speed_aware=True, slow_thresh=70),
  "H5 speed-aware t72":     dict(speed_aware=True, slow_thresh=72),
  "H5 speed-aware t66":     dict(speed_aware=True, slow_thresh=66),
  "H5 SA t72 + block16":    dict(speed_aware=True, slow_thresh=72, block_gap=16),
}
for name, extra in configs.items():
    pol = make_policy_cfg(3, 40, 5, **{**base, **extra})
    t = time.time(); r = env.evaluate(pol, runs=RUNS, frames=2000, deterministic=True)
    print("%-26s median=%.2f mean=%.2f min=%.2f max=%.2f (%.0fs)" % (
        name, r["median"], r["mean"], r["min"], r["max"], time.time()-t), flush=True)
print("ref: H4 76.06 | DQN 74.13 | leaderboard#1 75.66 | CEILING 76.3 | max 80")
