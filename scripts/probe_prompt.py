"""Fast prompt/model screen: how often does a candidate LLM policy AGREE with a
known-good convention on the states that convention actually visits?

The H6 sweep winner scores 76.51 @500. If a prompt makes the LLM reproduce H6's
decision on H6's own trajectory distribution, the LLM will score near H6 too.
Replaying H6 over a few runs visits a few hundred DISTINCT semantic states; we
query the candidate LIVE on each distinct state once (memoized to the variant's
cache, so this also warms the cache for a later full eval) and report agreement.

This is a ~5-minute proxy for a ~90-minute full eval — use it to iterate prompts
and screen models, then confirm the winner with the real eval (prompt_lab.py).

Usage:
    PYTHONPATH=src python3 scripts/probe_prompt.py <variant> [RUNS=3]
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                       # noqa: E402

from deeptraffic.env import DeepTrafficEnv               # noqa: E402
from deeptraffic.llm.state_tool import decode, ACTION_TO_ID  # noqa: E402
from deeptraffic.llm.heuristic2 import make_policy_h6    # noqa: E402
from deeptraffic.llm.llm_policy import LLMPolicy, MODEL, SCHEMA  # noqa: E402
import prompt_lab as pl                                  # noqa: E402

NAMES = {0: "maintain", 1: "accelerate", 2: "decelerate", 3: "left", 4: "right"}

H6 = dict(bg_base=14, bg_slope=0.0, slow_thresh=70, horizon=16, margin_mph=2,
          rear_gap=5, rear_fast=62, prefer_centre=True, boxed_action=2)


def main() -> None:
    variant = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    v = pl.VARIANTS[variant]

    pol = LLMPolicy(3, 40, 5, cache="exact", system=v["system"],
                    format_user=v.get("fmt"), model=v.get("model", MODEL),
                    num_predict=v.get("num_predict", 96),
                    schema=v.get("schema", SCHEMA))
    cache_file = os.path.join(pl.RESULTS, f"promptlab_{variant}.json")
    if os.path.exists(cache_file):
        pol.cache.update(json.load(open(cache_file)))

    ref = make_policy_h6(3, 40, 5, **H6)
    env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5,
                         other_agents=10)
    eng = env._eng
    eng.set_deterministic_seeds()

    # Pass 1: collect distinct states H6 visits and H6's action there.
    seen = {}  # state_key(text) -> (obs, ref_action)

    def collect(state, reward):
        obs = np.asarray(state, dtype=np.float32)
        a_ref = ref(obs)
        ds = decode(obs, 3, 40, 5)
        text = (v.get("fmt") or pol.format_user)(ds)
        if text not in seen:
            seen[text] = (obs, a_ref)
        return a_ref

    for _ in range(runs):
        eng.reset()
        for _f in range(2000):
            eng.V(collect)

    # Pass 2: query the candidate LIVE on each distinct state.
    t0 = time.time()
    agree = 0
    by_status = Counter()        # (status, agree?) buckets
    mismatch = Counter()
    items = list(seen.items())
    for i, (text, (obs, a_ref)) in enumerate(items):
        a_llm = pol(obs)
        status = "blocked" if "BLOCKED" in text else "open"
        ok = (a_llm == a_ref)
        agree += ok
        by_status[(status, ok)] += 1
        if not ok:
            mismatch[(status, NAMES[a_ref], NAMES[a_llm])] += 1
        if (i + 1) % 50 == 0:
            print("  probed %d/%d agree=%.1f%% (%.0fs)" % (
                i + 1, len(items), 100.0 * agree / (i + 1), time.time() - t0),
                flush=True)

    json.dump(pol.cache, open(cache_file, "w"))
    n = len(items)
    print("\n[%s] model=%s distinct_states=%d  AGREEMENT=%.1f%%  (%.0fs, "
          "parse_fail=%d)" % (variant, v.get("model", MODEL), n,
                              100.0 * agree / max(1, n), time.time() - t0,
                              pol.parse_fail))
    for status in ("open", "blocked"):
        a = by_status[(status, True)]
        b = by_status[(status, False)]
        tot = a + b
        if tot:
            print("  %-7s states=%-4d agreement=%.1f%%" % (
                status, tot, 100.0 * a / tot))
    print("  top disagreements (status, H6 -> LLM):")
    for (st, ah, al), c in mismatch.most_common(8):
        print("    x%-3d [%s] %s -> %s" % (c, st, ah, al))


if __name__ == "__main__":
    main()
