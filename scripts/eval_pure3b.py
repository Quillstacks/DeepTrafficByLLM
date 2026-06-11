"""Pure 3B-LLM driving evaluation at 11 cars (no cache; the LLM is queried for
EVERY decision of every controlled car). This is the honest "LLM orchestrator"
benchmark -- slow on CPU, fast on a GPU box running ollama.

Usage:
    PYTHONPATH=src python3 scripts/eval_pure3b.py [RUNS] [FRAMES]
    # defaults: RUNS=500 FRAMES=2000 (the official protocol)

Requires: `ollama serve` running with the model pulled:  ollama pull qwen2.5:3b
Prints per-run progress (score + running median + #LLM calls) so you can watch.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from deeptraffic.env import DeepTrafficEnv          # noqa: E402
from deeptraffic.llm.llm_policy import LLMPolicy, MODEL  # noqa: E402

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5,
                     other_agents=10, frames=FRAMES)
pol = LLMPolicy(3, 40, 5)  # cache=False, speed-aware SYSTEM, semantic state tool
eng = env._eng
eng.set_deterministic_seeds()  # official deterministic seed sequence


def wrapped(state, reward):
    return pol(np.asarray(state, dtype=np.float32))


scores = []
t0 = time.time()
for g in range(RUNS):
    eng.reset()
    O = 0.0
    for _ in range(FRAMES):
        eng.V(wrapped)
        na = eng.nOtherAgents
        for B in range(na + 1):
            O += max(0.0, eng.z[B].c * eng.z[B].a) / (na + 1)
    s = math.floor(O / FRAMES * 2e3) / 100
    scores.append(s)
    sc = sorted(scores)
    print("run %3d/%d  score=%.2f  running_median=%.2f  llm_calls=%d  parse_fail=%d  %.0fs"
          % (g + 1, RUNS, s, sc[len(sc) // 2], pol.calls, pol.parse_fail, time.time() - t0),
          flush=True)

sc = sorted(scores)
print("\nPURE-3B LLM (%s) @11cars %dx%d: median=%.2f mean=%.2f min=%.2f max=%.2f"
      % (MODEL, RUNS, FRAMES, sc[len(sc) // 2], sum(sc) / len(sc), sc[0], sc[-1]))
print("ref: DQN 74.13 | heuristic H5 76.18 | all-time ceiling 76.3 | per-car max 80")
