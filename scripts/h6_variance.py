"""Statistical-significance check for the H6 convention vs the 76.3 ceiling.

The official protocol (deterministic, 500x2000) yields ONE median (76.51 for the
H6 winner). The question "does H6 beat 76.3 significantly?" is really: how much
does the 500-run median move across INDEPENDENT seed blocks? If the across-block
std is small relative to the 0.21 margin, the win is significant.

Method (mirrors the repo's variance_h5.py): for each seed-block base, set the
engine's r/t to that base, restore the fresh-load u/v stream, and run 500
non-deterministic runs -> one median per block. base=0 reproduces the canonical
deterministic 76.51 (a consistency check). Report mean/std/min/max of the
block medians and the z-score vs 76.3.

CPU-only, parallel over blocks.

Usage: PYTHONPATH=src python3 scripts/h6_variance.py [RUNS=500] [N_BLOCKS=10]
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

H6 = dict(bg_base=14, bg_slope=0.0, slow_thresh=70, horizon=16, margin_mph=2,
          rear_gap=5, rear_fast=62, prefer_centre=True, boxed_action=2)
H5 = dict(strategy="inner_outer", rear_safety=True, drift=False,
          speed_aware=True, slow_thresh=70)
CEILING = 76.3


def eval_block(job):
    name, base, runs = job
    from deeptraffic.env import DeepTrafficEnv
    from deeptraffic.llm.heuristic2 import make_policy_h6
    from deeptraffic.llm.heuristic import make_policy_cfg
    env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5,
                         other_agents=10)
    pol = (make_policy_h6(3, 40, 5, **H6) if name == "H6"
           else make_policy_cfg(3, 40, 5, **H5))
    eng = env._eng
    eng.r = base
    eng.t = base
    eng.restore_fresh_load()
    r = env.evaluate(pol, runs=runs, frames=2000, deterministic=(base == 0))
    return name, base, float(r["median"]), float(r["mean"])


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    nblocks = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    bases = [b * 500 for b in range(nblocks)]
    jobs = ([("H6", b, runs) for b in bases] +
            [("H5", b, runs) for b in bases])

    t0 = time.time()
    res = {"H6": [], "H5": []}
    with Pool(min(12, len(jobs))) as p:
        for name, base, med, mean in p.imap_unordered(eval_block, jobs):
            res[name].append((base, med, mean))
            print("  %s base=%-5d median=%.3f mean=%.3f (%.0fs)" % (
                name, base, med, mean, time.time() - t0), flush=True)

    print("\n=== %d blocks x %d runs ===" % (nblocks, runs))
    out = {}
    for name in ("H6", "H5"):
        meds = sorted(m for _, m, _ in res[name])
        mu = statistics.mean(meds)
        sd = statistics.pstdev(meds)
        z = (mu - CEILING) / sd if sd > 0 else float("inf")
        out[name] = dict(blocks=meds, mean=round(mu, 3), std=round(sd, 4),
                         min=min(meds), max=max(meds),
                         z_vs_ceiling=round(z, 2))
        print("%s: mean=%.3f std=%.4f min=%.2f max=%.2f  | vs %.1f: +%.3f = %.1f sigma"
              % (name, mu, sd, min(meds), max(meds), CEILING, mu - CEILING, z),
              flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "..", "results", "h6_variance.json"), "w"),
              indent=2)


if __name__ == "__main__":
    main()
