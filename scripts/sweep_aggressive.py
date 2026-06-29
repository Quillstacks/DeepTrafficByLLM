"""Is there headroom ABOVE H6 (76.40) from bolder overtaking? (CPU, parallel.)

P8's LLM already beats H6 in spots by overtaking into open lanes where H6's
fixed `margin_mph` makes it wait. This sweep tests whether a less conservative
convention scores higher than H6 at the official protocol -- which would prove
real headroom above the oracle and give the LLM prompt better target behaviour.

Axes (around the H6 winner): margin_mph (lower = bolder overtakes), horizon
(how far ahead counts as a usable lane), bg_base (how early a slow car blocks).
Stage 1: all combos @300 runs (parallel). Stage 2: top 6 @500 runs.

Appends every result to results/aggressive_sweep.jsonl.

Usage: PYTHONPATH=src python3 scripts/sweep_aggressive.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from itertools import product
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL = os.path.join(ROOT, "results", "aggressive_sweep.jsonl")
WORKERS = 11   # leave cores for the GPU eval's python client + system

H6 = dict(bg_base=14, bg_slope=0.0, slow_thresh=70, horizon=16, margin_mph=2,
          rear_gap=5, rear_fast=62, prefer_centre=True, boxed_action=2)

GRID = dict(
    margin_mph=[0, 1, 2, 3],
    horizon=[12, 16, 20, 24],
    bg_base=[12, 14, 16, 18],
)


def eval_one(job):
    cfg, runs = job
    from deeptraffic.env import DeepTrafficEnv
    from deeptraffic.llm.heuristic2 import make_policy_h6
    env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5,
                         other_agents=10)
    pol = make_policy_h6(3, 40, 5, **cfg)
    r = env.evaluate(pol, runs=runs, frames=2000, deterministic=True)
    return cfg, runs, float(r["median"]), float(r["mean"])


def main():
    combos = []
    for m, h, b in product(GRID["margin_mph"], GRID["horizon"], GRID["bg_base"]):
        c = dict(H6); c["margin_mph"] = m; c["horizon"] = h; c["bg_base"] = b
        combos.append(c)
    # dedupe
    seen, uniq = set(), []
    for c in combos:
        k = (c["margin_mph"], c["horizon"], c["bg_base"])
        if k not in seen:
            seen.add(k); uniq.append(c)

    fh = open(JSONL, "a")
    print("stage1: %d configs @300 runs on %d workers" % (len(uniq), WORKERS),
          flush=True)
    t0 = time.time()
    res = []
    with Pool(WORKERS) as p:
        for cfg, runs, med, mean in p.imap_unordered(
                eval_one, [(c, 300) for c in uniq]):
            res.append((med, mean, cfg))
            fh.write(json.dumps(dict(config=cfg, runs=runs, median=med,
                                     mean=mean)) + "\n"); fh.flush()
    res.sort(reverse=True)
    print("stage1 done (%.0fs). top 8 @300:" % (time.time() - t0), flush=True)
    for med, mean, cfg in res[:8]:
        print("  %.3f (mean %.3f)  margin=%s horizon=%s bg=%s"
              % (med, mean, cfg["margin_mph"], cfg["horizon"], cfg["bg_base"]),
              flush=True)

    top = [cfg for _, _, cfg in res[:6]]
    print("\nstage2: top 6 @500 runs", flush=True)
    best = None
    with Pool(min(WORKERS, 6)) as p:
        for cfg, runs, med, mean in p.imap_unordered(
                eval_one, [(c, 500) for c in top]):
            fh.write(json.dumps(dict(config=cfg, runs=runs, median=med,
                                     mean=mean)) + "\n"); fh.flush()
            print("  @500 median=%.3f mean=%.3f  margin=%s horizon=%s bg=%s"
                  % (med, mean, cfg["margin_mph"], cfg["horizon"], cfg["bg_base"]),
                  flush=True)
            if best is None or med > best[0]:
                best = (med, cfg)
    fh.close()
    print("\nBEST @500: median=%.3f  %s  (H6=76.40, ceiling=76.3)"
          % (best[0], json.dumps(best[1])), flush=True)


if __name__ == "__main__":
    main()
