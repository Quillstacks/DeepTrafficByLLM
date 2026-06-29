"""H6 candidate conventions: speed/time-aware refinements over heuristic.py.

Two ideas H5's binary thresholds ignore:

1. TIME-TO-CATCH blocking. H5 says blocked iff (gap < 12 AND speed < 70).
   But a 50 mph car 14 ahead is a worse problem than a 68 mph car 11 ahead.
   Here: blocked iff gap < bg_base + bg_slope * (80 - speed_ahead) -- the
   slower the car ahead, the earlier we react (bg_slope=0, bg_base=12,
   slow_thresh=70 reduces exactly to H5's rule).

2. LANE UTILITY in mph, not patches. The lane you settle into is worth the
   speed of its nearest car if that car is within `horizon`, else ~80 (open).
   Overtake into the lane with the highest utility, if it beats staying by
   `margin_mph`. Clearance-in-patches (H5) is a proxy for exactly this.

Everything else (rear-safety, centre preference, decelerate-when-boxed-in,
no drift) follows H5, which the repo's sweeps already found best-in-class.

All functions are pure obs -> action (same I/O as the DQN).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .state_tool import decode

TOP = 80.0
CENTRE_LANE = 4


def make_policy_h6(lanes_side: int, patches_ahead: int, patches_behind: int, *,
                   bg_base: float = 12.0, bg_slope: float = 0.0,
                   slow_thresh: float = 70.0,
                   horizon: int = 16, margin_mph: float = 4.0,
                   rear_gap: int = 5, rear_fast: float = 55.0,
                   prefer_centre: bool = True,
                   boxed_action: int = 2,
                   ) -> Callable[[np.ndarray], int]:
    """Configurable H6. Defaults ~= H5 with utility-based target selection."""

    def utility(ln) -> float:
        """mph you would settle at in that lane."""
        if ln is None or not ln.on_road:
            return -1.0
        if ln.gap_ahead is None or ln.gap_ahead >= horizon:
            return TOP
        return float(ln.speed_ahead or 0.0)

    def rear_unsafe(ln) -> bool:
        return (ln is not None and ln.on_road and ln.gap_behind is not None
                and ln.gap_behind <= rear_gap
                and (ln.speed_behind or 0.0) >= rear_fast)

    def policy(obs: np.ndarray) -> int:
        ds = decode(obs, lanes_side, patches_ahead, patches_behind)
        ego = ds.ego
        left, right = ds.lane_by_rel(-1), ds.lane_by_rel(+1)

        spd = ego.speed_ahead if ego.gap_ahead is not None else None
        blocked = (ego.gap_ahead is not None and spd is not None
                   and spd < slow_thresh
                   and ego.gap_ahead < bg_base + bg_slope * (TOP - spd))
        if not blocked:
            return 1  # accelerate

        u_stay = utility(ego)
        opts = []
        for ln, act, centre_ward in (
                (left, 3, ds.ego_lane > CENTRE_LANE),
                (right, 4, ds.ego_lane < CENTRE_LANE)):
            if ln is None or not ln.on_road or rear_unsafe(ln):
                continue
            u = utility(ln)
            if u > u_stay + margin_mph:
                opts.append((u, centre_ward if prefer_centre else False, act))
        if opts:
            opts.sort(reverse=True)
            return opts[0][2]
        return boxed_action

    return policy
