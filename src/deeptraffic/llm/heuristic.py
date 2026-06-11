"""The driving-convention encoded as a hand-written policy (the ABLATION baseline).

This isolates the *convention* from the *LLM*: if these greedy clones beat the DQN
(74.13 @ 11 cars), the lane-discipline convention itself is the win; if the LLM
later beats this, the LLM's per-state judgment adds value on top.

Convention (same as we'll prompt the LLM with):
  * go as fast as possible; if a car is close ahead, OVERTAKE by moving to the
    clearest adjacent lane (only if clearly better), preferring a move toward the
    centre ("inner") lanes for the pass;
  * when NOT blocked, drift back toward an OUTER lane (1 or 7) so the fleet spreads
    out and leaves the inner lanes free for overtaking;
  * never change lanes without a clear benefit (lane changes cost time / can be
    vetoed by the safety system).

Uses ONLY the decoded obs (same I/O as the DQN). Returns an engine action id
(0 maintain, 1 accel, 2 decel, 3 left, 4 right).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .state_tool import decode, DecodedState

CLEAR = 10_000          # "no car in view" clearance
BLOCK_GAP = 12          # car within this many patches ahead -> consider overtaking
SWITCH_MARGIN = 3       # only switch lanes if neighbour is clearer by this much
CENTRE_LANE = 4         # 1..7; lanes <4 are left half, >4 right half


def _clearance(ln) -> int:
    if ln is None or not ln.on_road:
        return -1                       # cannot move there
    return ln.gap_ahead if ln.gap_ahead is not None else CLEAR


def heuristic_action(ds: DecodedState) -> int:
    ego = ds.ego
    left = ds.lane_by_rel(-1)
    right = ds.lane_by_rel(+1)
    cur = _clearance(ego)
    blocked = ego.gap_ahead is not None and ego.gap_ahead < BLOCK_GAP

    if blocked:
        # OVERTAKE: choose the clearly-clearer neighbour; prefer moving toward centre.
        opts = []
        if _clearance(left) > cur + SWITCH_MARGIN:
            # moving left is "toward centre" if ego is on the right half
            inner = ds.ego_lane > CENTRE_LANE
            opts.append((_clearance(left), inner, 3))
        if _clearance(right) > cur + SWITCH_MARGIN:
            inner = ds.ego_lane < CENTRE_LANE
            opts.append((_clearance(right), inner, 4))
        if opts:
            # rank by (clearance, prefer-inner)
            opts.sort(key=lambda x: (x[0], x[1]), reverse=True)
            return opts[0][2]
        return 2  # decelerate; let the safety system manage following distance

    # NOT blocked: drift toward an outer lane if that lane is at least as clear.
    if ds.ego_lane < CENTRE_LANE:
        outer = left, 3          # move left toward lane 1
    elif ds.ego_lane > CENTRE_LANE:
        outer = right, 4         # move right toward lane 7
    else:
        outer = None, None       # centre: just go fast
    if outer[0] is not None and _clearance(outer[0]) >= cur:
        return outer[1]
    return 1  # accelerate


def make_policy(lanes_side: int, patches_ahead: int, patches_behind: int) -> Callable[[np.ndarray], int]:
    def policy(obs: np.ndarray) -> int:
        ds = decode(obs, lanes_side, patches_ahead, patches_behind)
        return heuristic_action(ds)
    return policy


def make_policy_cfg(lanes_side: int, patches_ahead: int, patches_behind: int, *,
                    strategy: str = "inner_outer", rear_safety: bool = False,
                    block_gap: int = BLOCK_GAP, switch_margin: int = SWITCH_MARGIN,
                    rear_gap: int = 5, rear_fast: float = 55.0, drift: bool = True,
                    yield_neighbor: bool = False, yield_gap: int = 4, yield_fast: float = 60.0,
                    yield_require_block: bool = False,
                    speed_aware: bool = False, slow_thresh: float = 70.0
                    ) -> Callable[[np.ndarray], int]:
    """Configurable convention, for searching what beats the inner/outer baseline.

    strategy: 'inner_outer' (overtake clearer adjacent, prefer inner; drift outer when clear)
              or 'fastest_lane' (move toward the clearest adjacent lane; no outer bias).
    rear_safety: forbid changing into a lane with a fast car close behind (don't cut
                 off / slow a (team)mate -- directly protects the fleet-average score).
    drift: (inner_outer only) whether to drift toward outer lanes when clear.
    """
    def rear_unsafe(ln) -> bool:
        return (ln is not None and ln.on_road and ln.gap_behind is not None
                and ln.gap_behind <= rear_gap and (ln.speed_behind or 0.0) >= rear_fast)

    def policy(obs: np.ndarray) -> int:
        ds = decode(obs, lanes_side, patches_ahead, patches_behind)
        ego = ds.ego
        left, right = ds.lane_by_rel(-1), ds.lane_by_rel(+1)
        cur = _clearance(ego)
        # speed-aware: a car ahead at/near top speed does NOT block you (you can't
        # exceed it anyway), so only count it as a block if it is genuinely slow.
        ego_slow = (not speed_aware) or (ego.speed_ahead is not None and ego.speed_ahead < slow_thresh)
        blocked = ego.gap_ahead is not None and ego.gap_ahead < block_gap and ego_slow

        def usable(ln) -> bool:
            return ln is not None and ln.on_road and not (rear_safety and rear_unsafe(ln))

        def good_target(ln) -> bool:
            if not usable(ln) or _clearance(ln) <= cur + switch_margin:
                return False
            # speed-aware: don't dive into a lane whose near car is slow AND close
            # (the classic "clear-for-13-then-48mph" trap).
            if (speed_aware and ln.gap_ahead is not None
                    and (ln.speed_ahead or 0.0) < slow_thresh and ln.gap_ahead < block_gap):
                return False
            return True

        if blocked:
            opts = []
            if good_target(left):
                opts.append((_clearance(left), ds.ego_lane > CENTRE_LANE, 3))
            if good_target(right):
                opts.append((_clearance(right), ds.ego_lane < CENTRE_LANE, 4))
            if opts:
                opts.sort(key=lambda x: (x[0], x[1]), reverse=True)
                return opts[0][2]
            return 2  # decelerate

        # ---- not blocked: optional COURTESY (clear for a neighbour-behind) ----
        # A fast car approaching behind in a NEIGHBOURING lane is NOT speed-capped
        # to us. If it will want our lane, vacate to our other (equally clear,
        # rear-safe) lane so it can merge and keep speed -- a cooperative move that
        # can lift the FLEET average even though we gain nothing ourselves.
        if yield_neighbor:
            for fast_ln, tgt_ln, tgt_act in ((left, right, 4), (right, left, 3)):
                if (fast_ln is not None and fast_ln.on_road
                        and fast_ln.gap_behind is not None
                        and fast_ln.gap_behind <= yield_gap
                        and (fast_ln.speed_behind or 0.0) >= yield_fast):
                    if yield_require_block and fast_ln.gap_ahead is None:
                        continue  # that lane is open ahead; neighbour won't need to merge
                    if (tgt_ln is not None and tgt_ln.on_road
                            and _clearance(tgt_ln) >= cur
                            and not (rear_safety and rear_unsafe(tgt_ln))):
                        return tgt_act

        if strategy == "fastest_lane":
            best = None
            for ln, act in ((left, 3), (right, 4)):
                if usable(ln) and _clearance(ln) > cur + 8:
                    if best is None or _clearance(ln) > best[0]:
                        best = (_clearance(ln), act)
            return best[1] if best else 1

        # inner_outer
        if drift:
            if ds.ego_lane < CENTRE_LANE:
                outer = (left, 3)
            elif ds.ego_lane > CENTRE_LANE:
                outer = (right, 4)
            else:
                outer = (None, None)
            if outer[0] is not None and usable(outer[0]) and _clearance(outer[0]) >= cur:
                return outer[1]
        return 1  # accelerate

    return policy
