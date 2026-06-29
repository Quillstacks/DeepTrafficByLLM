"""Evaluate a driving SYSTEM prompt on the DeepTraffic engine.

Runs ``runs`` deterministic episodes with the LLM driving every controlled car
under the given system prompt, and returns the official score (median fleet mph)
plus the mean and the per-run scores. Uses the exact-text cache WITHIN one
evaluation: at temperature 0 the policy is a deterministic function of the
prompt, so memoising identical prompts is decision-identical and just faster.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from deeptraffic.env import DeepTrafficEnv
from deeptraffic.llm.llm_policy import LLMPolicy

from .render import render_state


def evaluate_prompt(system_prompt: str, *, model: str, engine_cfg: Dict,
                    runs: int = 10, frames: int = 2000,
                    num_predict: int = 96, progress=None) -> Dict:
    env = DeepTrafficEnv(lanes_side=engine_cfg["lanes_side"],
                         patches_ahead=engine_cfg["patches_ahead"],
                         patches_behind=engine_cfg["patches_behind"],
                         other_agents=engine_cfg["other_agents"], frames=frames)
    pol = LLMPolicy(engine_cfg["lanes_side"], engine_cfg["patches_ahead"],
                    engine_cfg["patches_behind"], model=model, cache="exact",
                    system=system_prompt,
                    format_user=lambda ds: render_state(ds),
                    num_predict=num_predict)
    eng = env._eng
    eng.set_deterministic_seeds()

    # action distribution -> behavioural signal for the paper (e.g. does a
    # high-overtake-weight prompt actually produce more lane changes?)
    ACTION = {0: "maintain", 1: "accelerate", 2: "decelerate", 3: "left", 4: "right"}
    action_counts = {v: 0 for v in ACTION.values()}

    def wrapped(state, reward):
        a = pol(np.asarray(state, dtype=np.float32))
        action_counts[ACTION.get(a, "maintain")] += 1
        return a

    scores: List[float] = []
    for g in range(runs):
        eng.reset()
        O = 0.0
        for _ in range(frames):
            eng.V(wrapped)
            na = eng.nOtherAgents
            for B in range(na + 1):
                O += max(0.0, eng.z[B].c * eng.z[B].a) / (na + 1)
        scores.append(math.floor(O / frames * 2e3) / 100)
        if progress:
            progress(g + 1, runs, scores)

    ordered = sorted(scores)
    n_dec = sum(action_counts.values()) or 1
    return {
        "median": float(ordered[len(ordered) // 2]),
        "mean": round(float(sum(ordered) / len(ordered)), 3),
        "std": round(float(np.std(scores)), 3),
        "min": float(ordered[0]), "max": float(ordered[-1]),
        "scores": scores,                              # per-run, in seed order
        "action_counts": action_counts,
        "action_fractions": {k: round(v / n_dec, 4)
                             for k, v in action_counts.items()},
        "decisions": n_dec,
        "llm_calls": pol.calls, "cache_hits": pol.cache_hits,
        "parse_fail": pol.parse_fail,
    }
