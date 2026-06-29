"""Faster, paired prompt/model screen (v2 of probe_prompt.py).

Two phases:
  build  -- replay the H6 winner (76.51 @500), record every DISTINCT decision
            state with its VISIT COUNT and H6's action, save the top-K
            most-visited states to results/h6_states.json. Model-independent and
            deterministic, so it is built ONCE and reused by every candidate ->
            all candidates are scored on the IDENTICAL state set (paired).
  probe  -- for a candidate (prompt + model), query the LLM on each saved state
            and report VISIT-WEIGHTED agreement with H6 (a state visited 200x
            counts 200x). Weighted agreement approximates the decision-frequency
            the full sim sees, so it tracks the real score far better than
            uniform-over-distinct agreement, at a fraction of the cost.

Because the score-relevant mass is concentrated in a few hundred common states,
K=400 covers most decisions; one candidate ~= 6-10 min vs ~90 min full eval.

Usage:
    PYTHONPATH=src python3 scripts/probe2.py build [RUNS=4] [K=400]
    PYTHONPATH=src python3 scripts/probe2.py probe <variant>
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                       # noqa: E402

from deeptraffic.env import DeepTrafficEnv               # noqa: E402
from deeptraffic.llm.state_tool import decode            # noqa: E402
from deeptraffic.llm.heuristic2 import make_policy_h6    # noqa: E402
from deeptraffic.llm.llm_policy import LLMPolicy, MODEL, SCHEMA  # noqa: E402
import prompt_lab as pl                                  # noqa: E402

NAMES = {0: "maintain", 1: "accelerate", 2: "decelerate", 3: "left", 4: "right"}
H6 = dict(bg_base=14, bg_slope=0.0, slow_thresh=70, horizon=16, margin_mph=2,
          rear_gap=5, rear_fast=62, prefer_centre=True, boxed_action=2)
STATES_FILE = os.path.join(pl.RESULTS, "h6_states.json")


def build(runs: int, k: int) -> None:
    """Reservoir-sample `k` actual decisions from H6's trajectory. The raw obs
    space is ~continuous (states almost never repeat), so we do NOT dedup -- a
    uniform reservoir sample reflects the true decision-FREQUENCY distribution,
    making agreement on the sample an unbiased estimate of agreement over all
    decisions. Deterministic (seeded) so it is built once and reused by every
    candidate -> a clean paired comparison."""
    import random
    rng = random.Random(20260615)
    ref = make_policy_h6(3, 40, 5, **H6)
    env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5,
                         other_agents=10)
    eng = env._eng
    eng.set_deterministic_seeds()

    sample = []          # reservoir of (obs_list, ref_action)
    n_seen = 0

    def collect(state, reward):
        nonlocal n_seen
        obs = np.asarray(state, dtype=np.float32)
        a_ref = ref(obs)
        n_seen += 1
        if len(sample) < k:
            sample.append((obs.tolist(), a_ref))
        else:
            j = rng.randint(0, n_seen - 1)
            if j < k:
                sample[j] = (obs.tolist(), a_ref)
        return a_ref

    t0 = time.time()
    for _ in range(runs):
        eng.reset()
        for _f in range(2000):
            eng.V(collect)
    states = [{"obs": o, "ref": a, "visits": 1} for o, a in sample]
    json.dump(states, open(STATES_FILE, "w"))
    mix = Counter(NAMES[a] for _, a in sample)
    print("reservoir-sampled %d decisions of %d total from %d runs (%.0fs) -> %s"
          % (len(sample), n_seen, runs, time.time() - t0, STATES_FILE),
          flush=True)
    print("H6 action mix over sample:",
          {a: round(100.0 * n / len(sample), 1) for a, n in mix.most_common()})


def probe(variant: str) -> None:
    v = pl.VARIANTS[variant]
    states = json.load(open(STATES_FILE))
    pol = LLMPolicy(3, 40, 5, cache="exact", system=v["system"],
                    format_user=v.get("fmt"), model=v.get("model", MODEL),
                    num_predict=v.get("num_predict", 96),
                    schema=v.get("schema", SCHEMA))
    cache_file = os.path.join(pl.RESULTS, f"promptlab_{variant}.json")
    if os.path.exists(cache_file):
        pol.cache.update(json.load(open(cache_file)))

    t0 = time.time()
    wagree = wtot = 0
    by_status = defaultdict(lambda: [0, 0])   # status -> [agree_w, tot_w]
    mism = Counter()
    for i, s in enumerate(states):
        obs = np.asarray(s["obs"], dtype=np.float32)
        w = s["visits"]
        a_ref = s["ref"]
        a_llm = pol(obs)
        text = (v.get("fmt") or pol.format_user)(decode(obs, 3, 40, 5))
        status = "blocked" if "BLOCKED" in text else "open"
        ok = (a_llm == a_ref)
        wtot += w
        by_status[status][1] += w
        if ok:
            wagree += w
            by_status[status][0] += w
        else:
            mism[(status, NAMES[a_ref], NAMES[a_llm])] += w
        if (i + 1) % 100 == 0:
            print("  %d/%d  w-agree=%.1f%% (%.0fs)" % (
                i + 1, len(states), 100.0 * wagree / wtot, time.time() - t0),
                flush=True)
    json.dump(pol.cache, open(cache_file, "w"))

    out = {"variant": variant, "model": v.get("model", MODEL),
           "states": len(states),
           "weighted_agreement": round(100.0 * wagree / wtot, 2),
           "open_agree": round(100.0 * by_status["open"][0] /
                               max(1, by_status["open"][1]), 2),
           "blocked_agree": round(100.0 * by_status["blocked"][0] /
                                  max(1, by_status["blocked"][1]), 2),
           "parse_fail": pol.parse_fail, "sec": round(time.time() - t0)}
    print(json.dumps(out), flush=True)
    print("  top disagreements (visit-weighted, status, H6 -> LLM):")
    for (st, ah, al), c in mism.most_common(8):
        print("    w%-4d [%s] %s -> %s" % (c, st, ah, al))
    with open(os.path.join(pl.RESULTS, "probe2_results.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build(int(sys.argv[2]) if len(sys.argv) > 2 else 4,
              int(sys.argv[3]) if len(sys.argv) > 3 else 400)
    elif cmd == "probe":
        probe(sys.argv[2])
    else:
        print("usage: probe2.py build [RUNS] [K] | probe <variant>")
