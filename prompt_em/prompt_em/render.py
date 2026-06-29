"""Semantic state renderer -- the USER message the driving LLM sees each step.

This is held FIXED across the experiment (it is the proven P4/P8 rendering): the
EM loop optimises the SYSTEM prompt (the driving rules), not the state encoding.
It is a pure function of the same observation the DQN sees -- it adds no
information, only translates the speed-grid into the features a driver reasons
over (open / fast-far / slow-close, plus an explicit rear status).
"""
from __future__ import annotations

from deeptraffic.llm.state_tool import decode, DecodedState  # noqa: F401

FAST_MPH = 70.0     # at/above this (or no car), a lane ahead is "not blocking"
NEAR = 14           # patches: a slow car within this is "close"
REAR_NEAR = 5       # patches: a car within this behind is "close"
REAR_FAST = 62.0    # mph: a close car behind this fast is a real threat


def _ahead(ln) -> str:
    if ln is None or not ln.on_road:
        return "no lane there"
    if ln.gap_ahead is None:
        return "OPEN (no car ahead at all - a totally clear lane)"
    spd = ln.speed_ahead or 0.0
    if spd >= FAST_MPH:
        return f"fast car {ln.gap_ahead}p ahead ({spd:.0f} mph) - far, not blocking"
    label = "SLOW car CLOSE" if ln.gap_ahead < NEAR else "slow car far"
    return f"{label} {ln.gap_ahead}p ahead ({spd:.0f} mph)"


def render_state(ds: DecodedState) -> str:
    ego = ds.ego
    left, right = ds.lane_by_rel(-1), ds.lane_by_rel(+1)
    blocked = (ego.gap_ahead is not None and (ego.speed_ahead or 99.0) < FAST_MPH
               and ego.gap_ahead < NEAR)
    status = ("BLOCKED by a slow car ahead" if blocked
              else "your lane is clear/fast (NOT blocked)")
    lines = [
        f"You are in lane {ds.ego_lane} of 7. STATUS: {status}.",
        f"Your lane: {_ahead(ego)}.",
        f"LEFT lane: {_ahead(left)}.",
        f"RIGHT lane: {_ahead(right)}.",
    ]
    threats = []
    for nm, ln in (("LEFT", left), ("RIGHT", right)):
        if (ln is not None and ln.on_road and ln.gap_behind is not None
                and ln.gap_behind <= REAR_NEAR
                and (ln.speed_behind or 0.0) >= REAR_FAST):
            threats.append(f"a fast car {ln.gap_behind}p behind in the {nm} lane "
                           f"({ln.speed_behind:.0f} mph)")
    if threats:
        lines.append("Behind: " + "; ".join(threats)
                     + " - do not cut in front of it.")
    else:
        lines.append("Behind: no car is close behind you in either side lane.")
    lines.append("Available actions: " + ", ".join(ds.legal_actions()))
    return "\n".join(lines)
