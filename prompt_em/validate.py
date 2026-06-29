"""Validate a saved system prompt at the FULL protocol (default 2000 frames,
24 runs) and bootstrap the median's CI vs the 76.3 ceiling. Use to turn a
proxy-optimized best prompt into the headline number.

Usage: PYTHONPATH=../src:. python validate.py <prompt_file> [RUNS=24] [FRAMES=2000]
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prompt_em.runner import evaluate_prompt

prompt = open(sys.argv[1]).read()
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 24
frames = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
eng = dict(lanes_side=3, patches_ahead=40, patches_behind=5, other_agents=10)


def prog(g, n, sc):
    s = sorted(sc)
    print("  run %2d/%d score=%.2f running_median=%.2f" % (g, n, sc[-1], s[len(s) // 2]),
          flush=True)


res = evaluate_prompt(prompt, model="qwen2.5:3b", engine_cfg=eng, runs=runs,
                      frames=frames, num_predict=96, progress=prog)
sc = res["scores"]
# bootstrap the median
rng = random.Random(42)
meds = []
for _ in range(20000):
    samp = [sc[rng.randrange(len(sc))] for _ in range(len(sc))]
    meds.append(sorted(samp)[len(samp) // 2])
meds.sort()
lo, hi = meds[500], meds[19500]
p_above = sum(1 for m in meds if m > 76.3) / len(meds)
out = {"runs": runs, "frames": frames, "median": res["median"], "mean": res["mean"],
       "std": res["std"], "min": res["min"], "max": res["max"], "scores": sc,
       "bootstrap_ci95": [lo, hi], "p_median_gt_76_3": round(p_above, 3),
       "action_fractions": res["action_fractions"], "parse_fail": res["parse_fail"]}
json.dump(out, open(sys.argv[1] + ".validation.json", "w"), indent=2)
print("\nVALIDATION: median=%.2f mean=%.2f  95%% CI [%.2f, %.2f]  P(med>76.3)=%.0f%%"
      % (res["median"], res["mean"], lo, hi, 100 * p_above), flush=True)
print("ref: DQN 74.13 | ceiling 76.3 | H6 76.40 | hand-tuned P4 76.50", flush=True)
