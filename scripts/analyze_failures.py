"""Deep failure analysis for a candidate prompt+model, focused on the
score-critical BLOCKED states.

Replays the H6 winner (76.40 expected) as the driver, samples blocked-state
decisions, and queries the candidate LIVE -- capturing the model's own one-line
REASON on every decision. Then it clusters the WRONG decisions (LLM != H6) by
(H6 action -> LLM action) and prints, for each cluster, representative prompts
together with the model's reason. The reasons reveal *why* the prose is being
misread, which is what we fix.

Also dumps every wrong (prompt, reason, h6, llm) to results/fail_<variant>.jsonl
for offline reading.

Usage:
    PYTHONPATH=src python3 scripts/analyze_failures.py <variant> [N_BLOCKED=160] [RUNS=12]
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
from deeptraffic.llm import llm_policy as lp             # noqa: E402
import prompt_lab as pl                                  # noqa: E402

NAMES = {0: "maintain", 1: "accelerate", 2: "decelerate", 3: "left", 4: "right"}
H6 = dict(bg_base=14, bg_slope=0.0, slow_thresh=70, horizon=16, margin_mph=2,
          rear_gap=5, rear_fast=62, prefer_centre=True, boxed_action=2)


def query_full(model, system, user, num_predict, schema):
    """Like lp._query but returns the parsed dict (reason + action)."""
    content = lp._query(model, system, user, num_predict=num_predict,
                        schema=schema)
    try:
        d = json.loads(content)
    except Exception:
        d = {"reason": content[:160], "action": lp.parse_action(content)}
    return d


def main():
    variant = sys.argv[1]
    n_blocked = int(sys.argv[2]) if len(sys.argv) > 2 else 160
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    v = pl.VARIANTS[variant]
    fmt = v.get("fmt") or lp._format_user
    model = v.get("model", lp.MODEL)
    system = v["system"]
    num_predict = v.get("num_predict", 96)
    schema = v.get("schema", lp.SCHEMA)

    ref = make_policy_h6(3, 40, 5, **H6)
    env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5,
                         other_agents=10)
    eng = env._eng
    eng.set_deterministic_seeds()

    # collect distinct blocked-state prompts H6 visits (+ H6 action)
    blocked = {}   # user_text -> (obs, h6_action)
    def collect(state, reward):
        obs = np.asarray(state, dtype=np.float32)
        a_ref = ref(obs)
        ds = decode(obs, 3, 40, 5)
        text = fmt(ds)
        if "BLOCKED" in text and text not in blocked:
            blocked[text] = (obs, a_ref)
        return a_ref
    for _ in range(runs):
        eng.reset()
        for _f in range(2000):
            eng.V(collect)
        if len(blocked) >= n_blocked:
            break

    items = list(blocked.items())[:n_blocked]
    print("collected %d distinct BLOCKED prompts; querying %s live...\n"
          % (len(items), model), flush=True)

    t0 = time.time()
    clusters = defaultdict(list)       # (h6,llm) -> [(text, reason)]
    cnt = Counter()
    agree = 0
    fh = open(os.path.join(pl.RESULTS, f"fail_{variant}.jsonl"), "w")
    for i, (text, (obs, a_ref)) in enumerate(items):
        d = query_full(model, system, text, num_predict, schema)
        act = d.get("action")
        reason = (d.get("reason") or "").strip()
        a_llm = lp.ACTION_TO_ID.get(act, 0) if act in lp.ACTION_ENUM else 0
        rec = dict(prompt=text, h6=NAMES[a_ref], llm=NAMES.get(a_llm, "?"),
                   reason=reason)
        fh.write(json.dumps(rec) + "\n")
        if a_llm == a_ref:
            agree += 1
        else:
            clusters[(NAMES[a_ref], NAMES.get(a_llm, "?"))].append((text, reason))
            cnt[(NAMES[a_ref], NAMES.get(a_llm, "?"))] += 1
        if (i + 1) % 40 == 0:
            print("  %d/%d  blocked-agree=%.1f%% (%.0fs)" % (
                i + 1, len(items), 100.0 * agree / (i + 1), time.time() - t0),
                flush=True)
    fh.close()

    n = len(items)
    print("\n[%s] BLOCKED agreement with H6: %d/%d = %.1f%%\n"
          % (variant, agree, n, 100.0 * agree / max(1, n)), flush=True)
    for (ah, al), c in cnt.most_common():
        print("############ x%d  H6=%s  ->  LLM=%s" % (c, ah, al))
        for text, reason in clusters[(ah, al)][:3]:
            short = " | ".join(l.strip() for l in text.splitlines()
                               if l.startswith(("Your lane", "LEFT", "RIGHT")))
            print("   STATE: %s" % short)
            print("   REASON: %s" % reason[:200])
        print(flush=True)


if __name__ == "__main__":
    main()
