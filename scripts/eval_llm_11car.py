import sys, time, json
sys.path.insert(0, "/Users/markschutera/Documents/Prof/Research/MITDeepDrive/src")
from deeptraffic.env import DeepTrafficEnv
from deeptraffic.llm.llm_policy import LLMPolicy

SYS_A=("You control one car on a 7-lane highway; other cars (some copies of you) share the road. "
 "Your score is your AVERAGE SPEED over time, so keep moving as fast as possible. Key facts:\n"
 "- Your car auto-drives at the fastest SAFE speed for its lane and cannot pass through the car ahead. "
 "So if a car ahead in your lane is slow, ACCELERATING does nothing useful - you stay stuck behind it.\n"
 "- To go faster when stuck, move to an adjacent lane that has more open road ahead. Lanes 1 and 7 are OUTER; "
 "lane 4 is the centre. Prefer passing via centre-ward lanes.\n"
 "- When your lane is clear far ahead, accelerate. If you are inner and an outer lane is open, drift outward so the road stays uncongested.\n"
 "- Only change lanes for a clear speed gain. Reason briefly, then choose ONE available action.")

RUNS = int(sys.argv[1]) if len(sys.argv)>1 else 3
env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5, other_agents=10)
pol = LLMPolicy(3,40,5, cache=True, system=SYS_A)
t=time.time()
r = env.evaluate(pol, runs=RUNS, frames=2000, deterministic=True)
dt=time.time()-t
print(json.dumps({"variant":"SYS_A","runs":RUNS,"median":r["median"],"mean":round(r["mean"],2),
  "min":r["min"],"max":r["max"],"sec":round(dt), **pol.stats()}))
print("ref: DQN 74.13 | heuristic 75.36 | ceiling 76.3 | do-nothing ~60")
