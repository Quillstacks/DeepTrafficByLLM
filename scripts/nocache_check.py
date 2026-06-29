"""Integrity check: the exact-text cache must equal the PURE (uncached) policy.

At temperature 0 the LLM is a deterministic function of the prompt, so
cache="exact" only memoises identical prompts -- it cannot change any decision.
This script proves it: run P4 with cache=False (every decision queried live) on
the first official seeds and compare to the cached scores recorded earlier.

Usage: PYTHONPATH=src python3 scripts/nocache_check.py [N_SEEDS=3]
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deeptraffic.env import DeepTrafficEnv          # noqa: E402
from deeptraffic.llm.llm_policy import LLMPolicy     # noqa: E402
import prompt_lab as pl                              # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
# P4 cached per-run scores on the official deterministic seeds (from e12 run).
CACHED = [75.95, 75.72, 77.18, 76.77, 76.8][:N]

env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5,
                     other_agents=10, frames=2000)
pol = LLMPolicy(3, 40, 5, cache=False, system=pl.SYS_P4, format_user=pl.fmt_p4)
eng = env._eng
eng.set_deterministic_seeds()


def wrapped(state, reward):
    return pol(np.asarray(state, dtype=np.float32))


ok = True
for g in range(N):
    eng.reset()
    O = 0.0
    for _ in range(2000):
        eng.V(wrapped)
        na = eng.nOtherAgents
        for B in range(na + 1):
            O += max(0.0, eng.z[B].c * eng.z[B].a) / (na + 1)
    s = math.floor(O / 2000 * 2e3) / 100
    match = abs(s - CACHED[g]) < 0.01
    ok = ok and match
    print("seed %d: nocache=%.2f  cached=%.2f  match=%s"
          % (g, s, CACHED[g], match), flush=True)

print("\nPURE no-cache run: llm_calls=%d parse_fail=%d" % (pol.calls, pol.parse_fail))
print("RESULT: %s -- exact-cache %s the pure policy"
      % ("PASS" if ok else "FAIL", "equals" if ok else "DIFFERS FROM"), flush=True)
