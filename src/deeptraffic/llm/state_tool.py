"""Decode the DeepTraffic ``.s()`` observation into a structured, legible state.

This is the **state tool**: it turns the raw 315-cell speed-grid (exactly what the
DQN sees) into per-lane semantics a human/LLM can reason over — *without adding any
information*. It is a PURE function of the obs vector, so any policy built on it
uses the **same I/O** as the DQN (fair comparison).

Observation encoding (from gameopt.js ``Map.s``):
  flat[cols*row + col] = H_cell / 100, where cols = 2*lanesSide+1, rows = pa+pb.
  * empty road  -> 1.0
  * off the road (no lane) -> 0.0
  * occupied by a car -> (c*a)/100 in (0, 0.04];  that car's mph = value * 2000
Columns are ego-relative lanes (left->right); the ego is column ``lanesSide``.
Rows 0..pa-1 are AHEAD (row pa-1 nearest), row pa is the ego's level, rows
pa+1.. are BEHIND.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

EMPTY_THRESH = 0.5      # value > this  -> empty road (empty == 1.0)
OFFGRID_THRESH = 1e-6   # value ~ 0     -> off the road
MPH_SCALE = 2000.0      # occupied value * 2000 == occupant mph (c*a/100 * 2000)
N_LANES = 7
ACTIONS = ["accelerate", "decelerate", "left", "right", "maintain"]
# engine action ids: 1 accelerate, 2 decelerate, 3 left, 4 right, 0 maintain
ACTION_TO_ID = {"accelerate": 1, "decelerate": 2, "left": 3, "right": 4, "maintain": 0}


@dataclass
class LaneInfo:
    rel: int                       # relative offset from ego (-lanesSide..+lanesSide)
    abs_lane: Optional[int]        # 1..7 absolute (1 = left outer), None if off-road
    on_road: bool
    gap_ahead: Optional[int]       # patches to nearest car ahead (None = clear)
    speed_ahead: Optional[float]   # that car's mph
    gap_behind: Optional[int]
    speed_behind: Optional[float]


@dataclass
class DecodedState:
    lanes_side: int
    patches_ahead: int
    patches_behind: int
    ego_lane: Optional[int]        # 1..7 absolute
    lanes: List[LaneInfo]          # left -> right, length 2*lanes_side+1

    # ---- convenience ----
    @property
    def ego(self) -> LaneInfo:
        return self.lanes[self.lanes_side]

    def lane_by_rel(self, rel: int) -> Optional[LaneInfo]:
        idx = rel + self.lanes_side
        return self.lanes[idx] if 0 <= idx < len(self.lanes) else None

    def legal_actions(self) -> List[str]:
        """Actions that are not obviously useless (left/right only if that lane is
        on the road). Acceleration/decel/maintain always allowed (the engine's
        safety layer handles the rest)."""
        acts = ["accelerate", "decelerate", "maintain"]
        left = self.lane_by_rel(-1)
        right = self.lane_by_rel(+1)
        if left is not None and left.on_road:
            acts.append("left")
        if right is not None and right.on_road:
            acts.append("right")
        return acts

    def summary(self) -> Dict:
        lanes = {}
        for ln in self.lanes:
            if not ln.on_road:
                continue
            key = f"lane{ln.abs_lane}" + ("(you)" if ln.rel == 0 else "")
            if ln.gap_ahead is None:
                ahead = "clear"
            else:
                ahead = f"car {ln.gap_ahead} ahead @ {ln.speed_ahead:.0f}mph"
            lanes[key] = ahead
        return {
            "your_lane": self.ego_lane,
            "total_lanes": N_LANES,
            "your_lane_is_outer": self.ego_lane in (1, N_LANES),
            "lanes_ahead": lanes,
            "legal_actions": self.legal_actions(),
        }

    def grid_text(self) -> str:
        """Compact ASCII map: columns = lanes (abs), rows = patches (top = far
        ahead). '.' empty, '##' off-road, two-digit speed for cars, 'EG' ego."""
        header = " ".join(
            (f"L{ln.abs_lane}" if ln.on_road else "XX") for ln in self.lanes
        )
        return f"lanes: {header}\n(ego in L{self.ego_lane}; cars show mph; ahead summarized above)"


def decode(obs, lanes_side: int, patches_ahead: int, patches_behind: int) -> DecodedState:
    cols = 2 * lanes_side + 1
    rows = patches_ahead + patches_behind
    g = np.asarray(obs, dtype=float).reshape(rows, cols)  # g[row, col]

    # which columns are on the road (any non-offgrid cell)
    on_road_col = (g > OFFGRID_THRESH).any(axis=0)
    # ego absolute lane: leftmost on-road col maps to the lowest visible abs lane.
    # ego.b (0-indexed) = lanes_side - (#off-road cols to the left of ego)
    first_on = int(np.argmax(on_road_col)) if on_road_col.any() else lanes_side
    ego_b = lanes_side - first_on  # 0-indexed absolute lane of the ego
    ego_lane_abs = ego_b + 1       # 1-indexed

    ahead_rows = range(patches_ahead - 1, -1, -1)   # nearest-first
    behind_rows = range(patches_ahead + 1, rows)    # nearest-first

    lanes: List[LaneInfo] = []
    for col in range(cols):
        rel = col - lanes_side
        on = bool(on_road_col[col])
        abs_lane = ego_b + rel + 1 if on else None
        gap_a = spd_a = gap_b = spd_b = None
        if on:
            for r in ahead_rows:
                v = g[r, col]
                if OFFGRID_THRESH < v <= EMPTY_THRESH:  # a car
                    gap_a = patches_ahead - r
                    spd_a = v * MPH_SCALE
                    break
            for r in behind_rows:
                v = g[r, col]
                if OFFGRID_THRESH < v <= EMPTY_THRESH:
                    gap_b = r - patches_ahead
                    spd_b = v * MPH_SCALE
                    break
        lanes.append(LaneInfo(rel, abs_lane, on, gap_a, spd_a, gap_b, spd_b))

    return DecodedState(lanes_side, patches_ahead, patches_behind, ego_lane_abs, lanes)
