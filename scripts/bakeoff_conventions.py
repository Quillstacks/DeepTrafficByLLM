import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from deeptraffic.env import DeepTrafficEnv
from deeptraffic.llm.heuristic import make_policy_cfg
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5, other_agents=10)
configs = {
  "H1 inner_outer (baseline)":    dict(strategy="inner_outer", rear_safety=False, drift=True),
  "H2 inner_outer + rear-safety": dict(strategy="inner_outer", rear_safety=True,  drift=True),
  "H3 fastest_lane + rear-safety":dict(strategy="fastest_lane", rear_safety=True),
  "H4 inner_outer+rear, NO drift":dict(strategy="inner_outer", rear_safety=True,  drift=False),
}
for name, cfg in configs.items():
    pol = make_policy_cfg(3, 40, 5, **cfg)
    t = time.time(); r = env.evaluate(pol, runs=RUNS, frames=2000, deterministic=True)
    print("%-32s median=%.2f mean=%.2f min=%.2f max=%.2f (%.0fs)" % (
        name, r["median"], r["mean"], r["min"], r["max"], time.time()-t), flush=True)
print("ref: DQN 74.13 | leaderboard#1 75.66 | ceiling 76.3")
