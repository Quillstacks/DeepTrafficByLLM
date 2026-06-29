import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from deeptraffic.env import DeepTrafficEnv
from deeptraffic.llm.heuristic import make_policy_cfg
env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5, other_agents=10)
base = dict(strategy="inner_outer", rear_safety=True, drift=False)
for name, extra in {
  "H5 SA t70 (500)": dict(speed_aware=True, slow_thresh=70),
  "H5 SA t72 (500)": dict(speed_aware=True, slow_thresh=72),
}.items():
    pol = make_policy_cfg(3,40,5, **{**base, **extra})
    t=time.time(); r=env.evaluate(pol, runs=500, frames=2000, deterministic=True)
    print("%-18s median=%.2f mean=%.2f min=%.2f max=%.2f (%.0fs)"%(name,r["median"],r["mean"],r["min"],r["max"],time.time()-t),flush=True)
print("ref: H4 76.06 | leaderboard#1 75.66 | CEILING 76.3")
