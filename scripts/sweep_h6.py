"""Staged parameter sweep of make_policy_h6 (heuristic2.py).

Stage 1: coordinate descent from the default config, 2-3 passes, 100-run medians.
Stage 2: ~80 random configs jittered +-1 grid step around the incumbent.
Stage 3: top 15 configs re-evaluated at 300 runs.
Stage 4: top 5 configs at the official 500 runs.

Every evaluation is appended as a JSON line to results/h6_sweep.jsonl.
Policies are constructed INSIDE each worker from a plain config dict.
"""
import json
import os
import random
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

JSONL = os.path.join(ROOT, "results", "h6_sweep.jsonl")
SUMMARY = os.path.join(ROOT, "results", "h6_sweep_summary.md")
WORKERS = 12

DEFAULT = dict(bg_base=12, bg_slope=0.0, slow_thresh=70, horizon=16,
               margin_mph=4, rear_gap=5, rear_fast=55,
               prefer_centre=True, boxed_action=2)

GRID = dict(
    bg_base=[8, 10, 12, 14, 16],
    bg_slope=[0.0, 0.15, 0.3, 0.5],
    slow_thresh=[66, 68, 70, 72, 74],
    horizon=[12, 16, 20, 24, 28],
    margin_mph=[2, 3, 4, 6, 8],
    rear_gap=[3, 4, 5, 6],
    rear_fast=[50, 55, 58, 62],
    prefer_centre=[True, False],
    boxed_action=[2, 0],
)
AXES = list(GRID)


def ckey(cfg):
    return tuple((k, cfg[k]) for k in AXES)


def eval_one(job):
    cfg, runs = job
    from deeptraffic.env import DeepTrafficEnv
    from deeptraffic.llm.heuristic2 import make_policy_h6
    env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5,
                         other_agents=10)
    pol = make_policy_h6(3, 40, 5, **cfg)
    r = env.evaluate(pol, runs=runs, frames=2000, deterministic=True)
    return cfg, runs, float(r["median"]), float(r["mean"])


class Sweep:
    def __init__(self):
        self.cache = {}          # (ckey, runs) -> (median, mean)
        self.pool = Pool(WORKERS)
        self.fh = open(JSONL, "a")
        self.n_eval = 0

    def batch(self, configs, runs, tag=""):
        """Evaluate configs (deduped, cached) at `runs`; return ckey->median."""
        seen, jobs = set(), []
        for cfg in configs:
            k = ckey(cfg)
            if (k, runs) in self.cache or k in seen:
                continue
            seen.add(k)
            jobs.append((dict(cfg), runs))
        if jobs:
            t0 = time.time()
            for cfg, r, med, mean in self.pool.imap_unordered(eval_one, jobs):
                self.cache[(ckey(cfg), r)] = (med, mean)
                self.fh.write(json.dumps(dict(config=cfg, runs=r,
                                              median=med, mean=mean)) + "\n")
                self.fh.flush()
                self.n_eval += 1
            print("[%s] %d evals @%d runs in %.0fs (total %d)" % (
                tag, len(jobs), runs, time.time() - t0, self.n_eval), flush=True)
        return {ckey(c): self.cache[(ckey(c), runs)][0] for c in configs}

    def median(self, cfg, runs):
        return self.cache[(ckey(cfg), runs)][0]


def main():
    random.seed(20260611)
    sw = Sweep()
    incumbent = dict(DEFAULT)

    # ---------------- Stage 1: coordinate descent ----------------
    for p in range(1, 4):
        cand = [dict(incumbent)]
        for ax in AXES:
            for v in GRID[ax]:
                if v != incumbent[ax]:
                    c = dict(incumbent); c[ax] = v; cand.append(c)
        sw.batch(cand, 100, tag="pass%d" % p)
        base_med = sw.median(incumbent, 100)
        new = dict(incumbent)
        for ax in AXES:
            best_v, best_m = incumbent[ax], base_med
            for v in GRID[ax]:
                c = dict(incumbent); c[ax] = v
                m = sw.median(c, 100)
                if m > best_m + 1e-9:
                    best_v, best_m = v, m
            new[ax] = best_v
        # combined greedy update may interact badly: compare against the best
        # single-axis config seen so far before accepting.
        sw.batch([new], 100, tag="pass%d-combined" % p)
        best_cfg, best_m = dict(incumbent), base_med
        for c in cand + [new]:
            m = sw.median(c, 100)
            if m > best_m + 1e-9:
                best_cfg, best_m = dict(c), m
        print("pass %d incumbent: %.2f %s" % (p, best_m, best_cfg), flush=True)
        if ckey(best_cfg) == ckey(incumbent):
            print("pass %d: no improvement, stopping descent" % p, flush=True)
            incumbent = best_cfg
            break
        incumbent = best_cfg

    # ---------------- Stage 2: random jitter around incumbent ----------------
    rand_cfgs, tries = [], 0
    seen = {k for (k, r) in sw.cache if r == 100}
    while len(rand_cfgs) < 80 and tries < 4000:
        tries += 1
        c = dict(incumbent)
        for ax in AXES:
            g = GRID[ax]
            i = g.index(c[ax]) if c[ax] in g else min(
                range(len(g)), key=lambda j: abs(float(g[j]) - float(c[ax])))
            j = max(0, min(len(g) - 1, i + random.choice([-1, 0, 1])))
            c[ax] = g[j]
        k = ckey(c)
        if k not in seen:
            seen.add(k)
            rand_cfgs.append(c)
    sw.batch(rand_cfgs, 100, tag="random")

    # ---------------- Stage 3: top 15 at 300 runs ----------------
    all100 = [(med, k) for (k, r), (med, _) in sw.cache.items() if r == 100]
    all100.sort(reverse=True)
    top15 = [dict(k) for _, k in all100[:15]]
    sw.batch(top15, 300, tag="top15@300")

    # ---------------- Stage 4: top 5 at 500 runs ----------------
    top15.sort(key=lambda c: sw.median(c, 300), reverse=True)
    top5 = top15[:5]
    sw.batch(top5, 500, tag="top5@500")
    top5.sort(key=lambda c: sw.median(c, 500), reverse=True)

    # ---------------- Summary ----------------
    best = top5[0]
    lines = ["# H6 sweep summary (%s)" % time.strftime("%Y-%m-%d %H:%M"), "",
             "Baseline H5: 76.18 median @500 runs. Target ceiling: 76.3.", "",
             "| rank | median@100 | median@300 | median@500 | config |",
             "|---|---|---|---|---|"]
    for i, c in enumerate(top5, 1):
        lines.append("| %d | %.2f | %.2f | %.2f | `%s` |" % (
            i, sw.median(c, 100), sw.median(c, 300), sw.median(c, 500),
            json.dumps(c)))
    lines += ["", "Best config:", "", "```json", json.dumps(best, indent=2),
              "```", "", "Best 500-run median: **%.2f**" % sw.median(best, 500),
              ""]
    with open(SUMMARY, "w") as f:
        f.write("\n".join(lines))
    print("BEST", json.dumps(best), "median500=%.2f" % sw.median(best, 500),
          flush=True)
    sw.pool.close()
    sw.pool.join()


def smoke():
    with Pool(2) as p:
        jobs = [(dict(DEFAULT), 2), ({**DEFAULT, "horizon": 20}, 2)]
        for cfg, runs, med, mean in p.imap_unordered(eval_one, jobs):
            print("smoke ok: runs=%d median=%.2f mean=%.2f horizon=%s" % (
                runs, med, mean, cfg["horizon"]), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        smoke()
    else:
        main()
