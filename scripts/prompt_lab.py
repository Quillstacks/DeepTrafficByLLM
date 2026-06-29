"""Prompt lab: compare system-prompt / state-rendering variants of the 3B-LLM
policy on the official deterministic seed sequence.

Uses cache='exact' (memoize on the exact user-message text). At temperature=0
the model is a deterministic function of the prompt, so the scores are
bit-identical to the pure no-cache benchmark -- we just never ask the same
question twice. Each variant's decision cache is persisted to
results/promptlab_<variant>.json so later (longer) evals resume cheaply.

Usage:
    PYTHONPATH=src python3 scripts/prompt_lab.py [RUNS] [variant ...]
    # default RUNS=30; default variants: all
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from deeptraffic.env import DeepTrafficEnv                       # noqa: E402
from deeptraffic.llm.llm_policy import (                          # noqa: E402
    LLMPolicy, MODEL, SYSTEM, SCHEMA, ACTION_ENUM)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------- variants

# P1: the H5 convention written out as an explicit, numeric decision procedure.
SYS_P1 = (
    "You drive ONE car on a 7-lane highway. Several other cars run the same rules as "
    "you; the score is the fleet's AVERAGE SPEED, so drive fast and never slow a "
    "teammate down. Top speed is 80 mph; you always auto-drive as fast as the car "
    "ahead allows.\n"
    "Decide with these rules, in order:\n"
    "1. You are BLOCKED only if the car ahead in YOUR lane is within 12 positions "
    "AND slower than 70 mph. A fast car (70+) ahead never blocks you.\n"
    "2. If NOT blocked: accelerate. Stay in your lane.\n"
    "3. If BLOCKED, overtake: pick an adjacent lane that is clearly better - its "
    "road ahead must be open at least 3 positions further than yours, and it must "
    "NOT have a slow (under 70) car within 12 positions. Never cut into a lane "
    "where a fast car is close behind (within 5) - you would slow a teammate. "
    "If both lanes qualify, prefer the one toward the centre (lane 4).\n"
    "4. If BLOCKED and neither lane qualifies: decelerate and wait - do NOT force "
    "a bad lane change.\n"
    "Think in one short sentence, then choose one action."
)


def fmt_neutral(ds) -> str:
    """Neutral fact sheet: no BLOCKED/not-blocked verdict baked in -- the system
    rules do all the judging. Gaps and speeds are stated plainly."""
    def lane_line(ln):
        if ln is None or not ln.on_road:
            return "no lane there (off road)"
        if ln.gap_ahead is None:
            return "open ahead (no car in sight)"
        return f"nearest car {ln.gap_ahead} positions ahead at {ln.speed_ahead:.0f} mph"

    def rear_line(ln):
        if ln is None or not ln.on_road or ln.gap_behind is None:
            return None
        return f"{ln.gap_behind} behind at {ln.speed_behind:.0f} mph"

    ego, left, right = ds.ego, ds.lane_by_rel(-1), ds.lane_by_rel(+1)
    lines = [
        f"You are in lane {ds.ego_lane} of 7.",
        f"YOUR lane: {lane_line(ego)}.",
        f"LEFT lane: {lane_line(left)}.",
        f"RIGHT lane: {lane_line(right)}.",
    ]
    rl, rr = rear_line(left), rear_line(right)
    if rl:
        lines.append(f"Car behind in LEFT lane: {rl}.")
    if rr:
        lines.append(f"Car behind in RIGHT lane: {rr}.")
    lines.append("Available actions: " + ", ".join(ds.legal_actions()))
    return "\n".join(lines)


def _coarse_speed(mph: float) -> int:
    return int(round(mph / 5.0)) * 5


def _coarse_gap(gap: int) -> str:
    """Exact small gaps (the decision-relevant range), bucketed far gaps."""
    if gap <= 16:
        return str(gap)
    if gap <= 24:
        return "about 20"
    if gap <= 34:
        return "about 30"
    return "about 40"


def fmt_coarse(ds) -> str:
    """Neutral fact sheet with COARSE numbers: speeds to the nearest 5 mph,
    far gaps bucketed. Removes precision a 3B model cannot use, and collapses
    the distinct-prompt space (better memoization, more consistent policy)."""
    def lane_line(ln):
        if ln is None or not ln.on_road:
            return "no lane there (off road)"
        if ln.gap_ahead is None:
            return "open ahead (no car in sight)"
        return (f"nearest car {_coarse_gap(ln.gap_ahead)} positions ahead "
                f"at {_coarse_speed(ln.speed_ahead)} mph")

    ego, left, right = ds.ego, ds.lane_by_rel(-1), ds.lane_by_rel(+1)
    lines = [
        f"You are in lane {ds.ego_lane} of 7.",
        f"YOUR lane: {lane_line(ego)}.",
        f"LEFT lane: {lane_line(left)}.",
        f"RIGHT lane: {lane_line(right)}.",
    ]
    for nm, ln in (("LEFT", left), ("RIGHT", right)):
        if (ln is not None and ln.on_road and ln.gap_behind is not None
                and ln.gap_behind <= 5 and (ln.speed_behind or 0.0) >= 55):
            lines.append(f"A fast car is close behind in the {nm} lane "
                         f"({ln.gap_behind} positions, {_coarse_speed(ln.speed_behind)} mph).")
    lines.append("Available actions: " + ", ".join(ds.legal_actions()))
    return "\n".join(lines)


# Action-only schema: no chain-of-thought, tiny generation budget.
SCHEMA_ACTION_ONLY = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ACTION_ENUM}},
    "required": ["action"],
}

SYS_P1_NOCOT = SYS_P1.replace(
    "Think in one short sentence, then choose one action.",
    "Apply the rules and output the action.")

# ---- P4: P0's shape (prose convention + semantic verdicts), fixed with the
# disagreement data (E5) and the H6-sweep thresholds (E4):
#   * blocked = slow (<70) car within 14 ahead (H6: bg_base 14, slow_thresh 70)
#   * rear-veto only for fast (>=62) cars within 5 behind (H6: rear_fast 62)
#   * top P0 errors fixed in prose: never 'maintain' when open (gas recovers
#     +0.02/decision only while accelerating), never change lanes unless
#     overtaking (P0 drifted centre-ward and yielded to cars behind).

P4_FAST = 70.0
P4_NEAR = 14
P4_REAR_NEAR = 5
P4_REAR_FAST = 62.0

SYS_P4 = (
    "You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED "
    "(top speed 80 mph). You auto-drive as fast as the road ahead allows.\n"
    "Your default action is ACCELERATE. Whenever your lane is OPEN or the car "
    "ahead is fast, accelerate - never 'maintain' (it wastes speed recovery) and "
    "never change lanes (a lane change costs speed unless it actually gets you "
    "past a slow car). Ignore cars behind you; never move aside for them.\n"
    "Change lanes ONLY to overtake when you are BLOCKED (a SLOW car CLOSE ahead "
    "in your lane). Pick an adjacent lane that is OPEN or whose nearest car is "
    "fast or far; NEVER move into a lane that also has a slow car close ahead, "
    "and never cut into a lane where a fast car is close behind. If both sides "
    "qualify, prefer the lane closer to the centre (lane 4).\n"
    "If you are blocked and neither side qualifies, ease off briefly and wait "
    "for a gap.\n"
    "Think in one short sentence, then choose one action."
)


def _p4_ahead(ln) -> str:
    if ln is None or not ln.on_road:
        return "no lane there"
    if ln.gap_ahead is None:
        return "OPEN (clear far ahead)"
    spd = ln.speed_ahead or 0.0
    if spd >= P4_FAST:
        return f"fast car {ln.gap_ahead}p ahead ({spd:.0f} mph) - not blocking"
    label = "SLOW car CLOSE" if ln.gap_ahead < P4_NEAR else "slow car far"
    return f"{label} {ln.gap_ahead}p ahead ({spd:.0f} mph)"


def fmt_p4(ds) -> str:
    ego = ds.ego
    left, right = ds.lane_by_rel(-1), ds.lane_by_rel(+1)
    blocked = (ego.gap_ahead is not None and (ego.speed_ahead or 99.0) < P4_FAST
               and ego.gap_ahead < P4_NEAR)
    status = ("BLOCKED by a slow car ahead" if blocked
              else "your lane is clear/fast (NOT blocked)")
    lines = [
        f"You are in lane {ds.ego_lane} of 7. STATUS: {status}.",
        f"Your lane: {_p4_ahead(ego)}.",
        f"LEFT lane: {_p4_ahead(left)}.",
        f"RIGHT lane: {_p4_ahead(right)}.",
    ]
    for nm, ln in (("LEFT", left), ("RIGHT", right)):
        if (ln is not None and ln.on_road and ln.gap_behind is not None
                and ln.gap_behind <= P4_REAR_NEAR
                and (ln.speed_behind or 0.0) >= P4_REAR_FAST):
            lines.append(f"Fast car {ln.gap_behind}p behind in the {nm} lane "
                         f"({ln.speed_behind:.0f} mph) - do not cut in front of it.")
    lines.append("Available actions: " + ", ".join(ds.legal_actions()))
    return "\n".join(lines)


# ---- P5: P4 + the E5 disagreement fixes. P4 followed the convention 87.8% of
# the time; the residual loss was almost entirely (a) maintain / spurious lane
# changes when NOT blocked, and (b) diving into a slow-close lane when blocked.
# Fix: bind the decision to the (reliable) STATUS verdict — NOT blocked means
# accelerate and ignore every lane detail; only read lanes when BLOCKED — and
# state the overtake target test as a strict comparison against the blocker.

SYS_P5 = (
    "You drive ONE car on a 7-lane highway. Score = your AVERAGE SPEED (top "
    "80 mph); the car auto-drives as fast as the road right ahead allows.\n"
    "FIRST read the STATUS line. It is the whole decision:\n"
    "* STATUS NOT blocked -> ACCELERATE. Always. Ignore every other lane, ignore "
    "cars ahead that are fast or far, ignore all cars behind. Do NOT maintain "
    "(you lose speed you could be regaining) and do NOT change lanes (a needless "
    "lane change only costs speed).\n"
    "* STATUS BLOCKED (a slow car is close ahead in your lane) -> overtake. Move "
    "one lane to a side that is genuinely better: that lane must be OPEN, or its "
    "nearest car must be FAST or FARTHER ahead than the slow car blocking you. "
    "Never move into a lane whose nearest car is also slow and close - that is "
    "not an escape. Never cut in front of a fast car close behind in that lane. "
    "If neither side is genuinely better, ease off and wait. If both sides are "
    "equally good, take the one toward the centre (lane 4).\n"
    "Think in one short sentence, then choose one action."
)


def _p5_ahead(ln) -> str:
    if ln is None or not ln.on_road:
        return "no lane there"
    if ln.gap_ahead is None:
        return "OPEN (clear far ahead)"
    spd = ln.speed_ahead or 0.0
    if spd >= P4_FAST:
        return f"fast car {ln.gap_ahead}p ahead ({spd:.0f} mph, not an obstacle)"
    label = "SLOW car CLOSE" if ln.gap_ahead < P4_NEAR else "slow car far"
    return f"{label} {ln.gap_ahead}p ahead ({spd:.0f} mph)"


def fmt_p5(ds) -> str:
    ego = ds.ego
    left, right = ds.lane_by_rel(-1), ds.lane_by_rel(+1)
    blocked = (ego.gap_ahead is not None and (ego.speed_ahead or 99.0) < P4_FAST
               and ego.gap_ahead < P4_NEAR)
    status = ("BLOCKED (slow car close ahead in your lane)" if blocked
              else "NOT blocked (your lane is clear or the car ahead is fast)")
    lines = [
        f"STATUS: {status}.",
        f"You are in lane {ds.ego_lane} of 7.",
        f"Your lane: {_p5_ahead(ego)}.",
        f"LEFT lane: {_p5_ahead(left)}.",
        f"RIGHT lane: {_p5_ahead(right)}.",
    ]
    for nm, ln in (("LEFT", left), ("RIGHT", right)):
        if (ln is not None and ln.on_road and ln.gap_behind is not None
                and ln.gap_behind <= P4_REAR_NEAR
                and (ln.speed_behind or 0.0) >= P4_REAR_FAST):
            lines.append(f"Fast car {ln.gap_behind}p behind in the {nm} lane "
                         f"({ln.speed_behind:.0f} mph) - do not cut in front of it.")
    lines.append("Available actions: " + ", ".join(ds.legal_actions()))
    return "\n".join(lines)


# ---- P6: a MINIMAL delta on P4 (P5 taught us rewrites regress at 3B). P4's
# open-state wording is kept VERBATIM (qwen3:1.7b already scores 100% there);
# only the BLOCKED branch is strengthened, targeting the bake-off's dominant
# failure: when blocked, qwen3 keeps ACCELERATING into the slow car instead of
# overtaking. Make explicit that accelerating-while-blocked is futile.
SYS_P6 = SYS_P4.replace(
    "Change lanes ONLY to overtake when you are BLOCKED (a SLOW car CLOSE ahead "
    "in your lane). Pick",
    "When you are BLOCKED (a SLOW car CLOSE ahead in your lane), ACCELERATING is "
    "useless - you are already going as fast as that slow car, so you stay stuck "
    "at its speed. You MUST instead change lanes to overtake. Pick")

# ---- P7: surgical fix to qwen2.5/P4's BLOCKED-state timidity (analyze_failures
# showed the model reads "OPEN (clear far ahead)" as "the gap is far AWAY / not
# usable" and decelerates/maintains instead of taking a valid open/fast lane).
# Two minimal changes, open-state wording untouched (P5 regressed by touching it):
#   1. fmt_p7: unambiguous lane labels ("OPEN (no car ahead at all)", "far, not
#      blocking") so the model stops misreading good lanes as unusable.
#   2. SYS_P7: make the blocked branch DECISIVE — if either side is open or
#      fast/far you MUST move; decelerate only when BOTH sides are slow-close.
SYS_P7 = SYS_P4.replace(
    "Change lanes ONLY to overtake when you are BLOCKED (a SLOW car CLOSE ahead "
    "in your lane). Pick an adjacent lane that is OPEN or whose nearest car is "
    "fast or far; NEVER move into a lane that also has a slow car close ahead, "
    "and never cut into a lane where a fast car is close behind. If both sides "
    "qualify, prefer the lane closer to the centre (lane 4).\n"
    "If you are blocked and neither side qualifies, ease off briefly and wait "
    "for a gap.",
    "When you are BLOCKED (a SLOW car CLOSE ahead in your lane) you are stuck at "
    "that slow car's speed, so you MUST overtake. If EITHER side lane is OPEN or "
    "its nearest car is fast or far, MOVE INTO IT NOW - moving past the slow car "
    "is always better than staying stuck, so do NOT decelerate and do NOT "
    "maintain. A side lane is unusable ONLY if it also has a SLOW car CLOSE "
    "ahead, or a fast car close behind. Decelerate ONLY when BOTH side lanes are "
    "unusable. If both sides are usable, prefer the one closer to the centre "
    "(lane 4).")


def _p7_ahead(ln) -> str:
    if ln is None or not ln.on_road:
        return "no lane there"
    if ln.gap_ahead is None:
        return "OPEN (no car ahead at all - a totally clear lane)"
    spd = ln.speed_ahead or 0.0
    if spd >= P4_FAST:
        return f"fast car {ln.gap_ahead}p ahead ({spd:.0f} mph) - far, not blocking"
    label = "SLOW car CLOSE" if ln.gap_ahead < P4_NEAR else "slow car far"
    return f"{label} {ln.gap_ahead}p ahead ({spd:.0f} mph)"


def fmt_p7(ds) -> str:
    ego = ds.ego
    left, right = ds.lane_by_rel(-1), ds.lane_by_rel(+1)
    blocked = (ego.gap_ahead is not None and (ego.speed_ahead or 99.0) < P4_FAST
               and ego.gap_ahead < P4_NEAR)
    status = ("BLOCKED by a slow car ahead" if blocked
              else "your lane is clear/fast (NOT blocked)")
    lines = [
        f"You are in lane {ds.ego_lane} of 7. STATUS: {status}.",
        f"Your lane: {_p7_ahead(ego)}.",
        f"LEFT lane: {_p7_ahead(left)}.",
        f"RIGHT lane: {_p7_ahead(right)}.",
    ]
    for nm, ln in (("LEFT", left), ("RIGHT", right)):
        if (ln is not None and ln.on_road and ln.gap_behind is not None
                and ln.gap_behind <= P4_REAR_NEAR
                and (ln.speed_behind or 0.0) >= P4_REAR_FAST):
            lines.append(f"Fast car {ln.gap_behind}p behind in the {nm} lane "
                         f"({ln.speed_behind:.0f} mph) - do not cut in front of it.")
    lines.append("Available actions: " + ", ".join(ds.legal_actions()))
    return "\n".join(lines)


# ---- P8: P7 (decisive overtake, fixed timidity) + two fixes from P7's reasons:
#   1. Rear HALLUCINATION: qwen2.5 kept inventing "a fast car close behind" to
#      mark a good lane unusable. fmt_p8 states the rear situation EXPLICITLY -
#      either the real threat, or "no car is close behind you" - so it stops
#      guessing. Also a system line: trust the descriptions, don't invent cars.
#   2. LEFT bias / wrong-faster-side: P7 defaulted left and sometimes took the
#      slower side. SYS_P8 tie-break = move to the FASTER/more-open side; centre
#      only on a true tie. (Fleet-safety: a shared left bias congests left lanes.)
SYS_P8 = SYS_P7.replace(
    "If both sides are usable, prefer the one closer to the centre (lane 4).",
    "If both sides are usable, move to the FASTER side - the one that is OPEN, or "
    "whose nearest car is faster or farther ahead. Only if the two sides are "
    "equally clear, take the side toward the centre (lane 4). Trust the lane "
    "descriptions exactly: a side lane has a car behind it ONLY if a 'behind' "
    "line says so - do not assume one.")


def fmt_p8(ds) -> str:
    ego = ds.ego
    left, right = ds.lane_by_rel(-1), ds.lane_by_rel(+1)
    blocked = (ego.gap_ahead is not None and (ego.speed_ahead or 99.0) < P4_FAST
               and ego.gap_ahead < P4_NEAR)
    status = ("BLOCKED by a slow car ahead" if blocked
              else "your lane is clear/fast (NOT blocked)")
    lines = [
        f"You are in lane {ds.ego_lane} of 7. STATUS: {status}.",
        f"Your lane: {_p7_ahead(ego)}.",
        f"LEFT lane: {_p7_ahead(left)}.",
        f"RIGHT lane: {_p7_ahead(right)}.",
    ]
    # explicit rear status for the on-road side lanes -> kills the "fast car
    # close behind" hallucination that P7 used to veto good lanes.
    threats = []
    for nm, ln in (("LEFT", left), ("RIGHT", right)):
        if (ln is not None and ln.on_road and ln.gap_behind is not None
                and ln.gap_behind <= P4_REAR_NEAR
                and (ln.speed_behind or 0.0) >= P4_REAR_FAST):
            threats.append(f"a fast car {ln.gap_behind}p behind in the {nm} lane "
                           f"({ln.speed_behind:.0f} mph) - do not cut in front of it")
    if threats:
        lines.append("Behind: " + "; ".join(threats) + ".")
    else:
        lines.append("Behind: no car is close behind you in either side lane, so "
                     "moving over will not cut anyone off.")
    lines.append("Available actions: " + ", ".join(ds.legal_actions()))
    return "\n".join(lines)


VARIANTS = {
    # name: dict(system=..., fmt=..., num_predict=..., schema=...)
    "P0_current": dict(system=SYSTEM),
    "P4_convention": dict(system=SYS_P4, fmt=fmt_p4),
    "P7_decisive": dict(system=SYS_P7, fmt=fmt_p7),
    "P8_faster_side": dict(system=SYS_P8, fmt=fmt_p8),
    "P6_overtake": dict(system=SYS_P6, fmt=fmt_p4),
    "P6_qwen3_17": dict(system=SYS_P6, fmt=fmt_p4, model="qwen3:1.7b"),
    "P6_llama32": dict(system=SYS_P6, fmt=fmt_p4, model="llama3.2:3b"),
    "P5_status": dict(system=SYS_P5, fmt=fmt_p5),
    "P5_llama32": dict(system=SYS_P5, fmt=fmt_p5, model="llama3.2:3b"),
    "P5_qwen3_17": dict(system=SYS_P5, fmt=fmt_p5, model="qwen3:1.7b"),
    "P4_llama32": dict(system=SYS_P4, fmt=fmt_p4, model="llama3.2:3b"),
    "P4_qwen3_17": dict(system=SYS_P4, fmt=fmt_p4, model="qwen3:1.7b"),
    "P1_h5rules_neutral": dict(system=SYS_P1, fmt=fmt_neutral),
    "P1_h5rules_semantic": dict(system=SYS_P1),
    "P2_h5rules_coarse": dict(system=SYS_P1, fmt=fmt_coarse),
    "P3_actiononly_coarse": dict(system=SYS_P1_NOCOT, fmt=fmt_coarse,
                                 num_predict=12, schema=SCHEMA_ACTION_ONLY),
}

# ---------------------------------------------------------------- harness


def run_variant(name: str, runs: int, frames: int = 2000) -> dict:
    v = VARIANTS[name]
    cache_file = os.path.join(RESULTS, f"promptlab_{name}.json")
    env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5,
                         other_agents=10, frames=frames)
    pol = LLMPolicy(3, 40, 5, cache="exact", system=v["system"],
                    format_user=v.get("fmt"),
                    model=v.get("model", MODEL),
                    num_predict=v.get("num_predict", 96),
                    schema=v.get("schema", SCHEMA))
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            pol.cache.update(json.load(f))
        print(f"[{name}] loaded {len(pol.cache)} cached decisions", flush=True)

    import math

    import numpy as np

    eng = env._eng
    eng.set_deterministic_seeds()

    def wrapped(state, reward):
        return pol(np.asarray(state, dtype=np.float32))

    t0 = time.time()
    scores = []  # seed order
    for g in range(runs):
        eng.reset()
        O = 0.0
        for _ in range(frames):
            eng.V(wrapped)
            na = eng.nOtherAgents
            for B in range(na + 1):
                O += max(0.0, eng.z[B].c * eng.z[B].a) / (na + 1)
        scores.append(math.floor(O / frames * 2e3) / 100)
        sc = sorted(scores)
        print("[%s] run %3d/%d score=%.2f running_median=%.2f calls=%d hits=%d "
              "fail=%d cache=%d %.0fs"
              % (name, g + 1, runs, scores[-1], sc[len(sc) // 2], pol.calls,
                 pol.cache_hits, pol.parse_fail, len(pol.cache), time.time() - t0),
              flush=True)
        with open(cache_file, "w") as f:
            json.dump(pol.cache, f)
    dt = time.time() - t0

    sc = sorted(scores)
    out = {"variant": name, "runs": runs, "median": sc[len(sc) // 2],
           "mean": round(sum(sc) / len(sc), 2), "min": sc[0], "max": sc[-1],
           "sec": round(dt), "scores_seed_order": scores, **pol.stats()}
    print(json.dumps(out), flush=True)
    with open(os.path.join(RESULTS, "promptlab_results.jsonl"), "a") as f:
        f.write(json.dumps(out) + "\n")
    return out


if __name__ == "__main__":
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    names = sys.argv[2:] or list(VARIANTS)
    print(f"prompt lab: runs={runs} variants={names}", flush=True)
    print("ref: DQN 74.13 | H5 heuristic 76.18 | ceiling 76.3", flush=True)
    for n in names:
        run_variant(n, runs)
