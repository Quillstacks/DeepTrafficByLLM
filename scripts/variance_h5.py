import sys, statistics
sys.path.insert(0, "/Users/markschutera/Documents/Prof/Research/MITDeepDrive/src")
from deeptraffic.env import DeepTrafficEnv
from deeptraffic.llm.heuristic import make_policy_cfg
env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5, other_agents=10)
pol = make_policy_cfg(3,40,5, strategy="inner_outer", rear_safety=True, drift=False, speed_aware=True, slow_thresh=72)
meds = []
for base in [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]:
    env._eng.r = base; env._eng.t = base; env._eng.restore_fresh_load()
    r = env.evaluate(pol, runs=500, frames=2000, deterministic=False)
    meds.append(r["median"]); print("seed-block %5d: median=%.2f mean=%.2f" % (base, r["median"], r["mean"]), flush=True)
print("medians:", [round(m,2) for m in meds])
print("median spread: min=%.2f max=%.2f std=%.3f  (ceiling 76.3)" % (min(meds), max(meds), statistics.pstdev(meds)))
