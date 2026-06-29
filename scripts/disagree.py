"""Disagreement analysis: where does a prompt variant's (cached) LLM policy
differ from the H5 heuristic convention?

Replays the official deterministic seeds driving every car with H5, and at each
decision ALSO renders the variant's user text and looks the LLM's answer up in
the variant's memo cache (no LLM calls; states not yet in the cache are
counted as 'unseen'). Prints the most frequent disagreement patterns.

Usage:
    PYTHONPATH=src python3 scripts/disagree.py <variant> [RUNS=4] [TOP=15]
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                    # noqa: E402

from deeptraffic.env import DeepTrafficEnv            # noqa: E402
from deeptraffic.llm.state_tool import decode         # noqa: E402
from deeptraffic.llm.heuristic import make_policy_cfg  # noqa: E402
from deeptraffic.llm import llm_policy as lp          # noqa: E402
import prompt_lab as pl                               # noqa: E402

NAMES = {0: "maintain", 1: "accelerate", 2: "decelerate", 3: "left", 4: "right"}


def main() -> None:
    variant = sys.argv[1]
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    v = pl.VARIANTS[variant]
    fmt = v.get("fmt") or lp._format_user
    cache = json.load(open(os.path.join(pl.RESULTS, f"promptlab_{variant}.json")))

    ref = os.environ.get("DT_REF", "h6")
    if ref == "h6":
        # the E4 sweep winner (76.51 @500)
        from deeptraffic.llm.heuristic2 import make_policy_h6
        h5 = make_policy_h6(3, 40, 5, bg_base=14, bg_slope=0.0, slow_thresh=70,
                            horizon=16, margin_mph=2, rear_gap=5, rear_fast=62,
                            prefer_centre=True, boxed_action=2)
    else:
        h5 = make_policy_cfg(3, 40, 5, strategy="inner_outer", rear_safety=True,
                             drift=False, speed_aware=True, slow_thresh=70)

    env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5,
                         other_agents=10)
    eng = env._eng
    eng.set_deterministic_seeds()

    n = agree = unseen = 0
    mismatch = Counter()

    def wrapped(state, reward):
        nonlocal n, agree, unseen
        obs = np.asarray(state, dtype=np.float32)
        a_h5 = h5(obs)
        text = fmt(decode(obs, 3, 40, 5))
        a_llm = cache.get(text)
        n += 1
        if a_llm is None:
            unseen += 1
        elif a_llm == a_h5:
            agree += 1
        else:
            mismatch[(text, NAMES[a_h5], NAMES[a_llm])] += 1
        return a_h5

    for _ in range(runs):
        eng.reset()
        for _f in range(2000):
            eng.V(wrapped)

    seen = n - unseen
    print(f"[{variant}] decisions={n} unseen={unseen} "
          f"agreement={agree}/{seen} ({100.0 * agree / max(1, seen):.1f}%)")
    print(f"top {top} disagreements (count, H5 -> LLM):\n")
    for (text, ah, al), c in mismatch.most_common(top):
        print(f"--- x{c}  H5={ah}  LLM={al}")
        print(text)
        print()


if __name__ == "__main__":
    main()
