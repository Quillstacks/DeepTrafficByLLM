# prompt-EM report — my_run

- objective: **median fleet mph** (target: beat 76.3 ceiling)
- iterations: 16 x 10 runs
- policy model: qwen2.5:3b; synthesis: template; co-optimising emphasis + sequence

## Trajectory (median & mean fleet mph per iteration)

| iter | median | mean | min | max | parse_fail |
|---|---|---|---|---|---|
| 0 | **73.65** | 73.52 | 71.44 | 75.29 | 27 |
| 1 | **74.44** | 74.01 | 72.06 | 75.53 | 4 |
| 2 | **72.13** | 72.14 | 69.41 | 75.24 | 11 |
| 3 | **72.03** | 67.65 | 45.86 | 74.98 | 3 |
| 4 | **74.90** | 74.78 | 73.34 | 76.48 | 1 |
| 5 | **62.70** | 62.49 | 58.91 | 66.42 | 0 |
| 6 | **74.75** | 74.75 | 72.10 | 77.35 | 0 |
| 7 | **75.43** | 75.08 | 72.83 | 77.22 | 2 |
| 8 | **74.11** | 72.88 | 69.91 | 75.09 | 11 |
| 9 | **72.55** | 72.27 | 67.29 | 76.39 | 5 |
| 10 | **75.36** | 75.55 | 74.64 | 76.69 | 4 |
| 11 | **75.75** | 75.44 | 74.33 | 76.83 | 22 |
| 12 | **74.30** | 73.33 | 66.35 | 76.24 | 15 |
| 13 | **74.95** | 74.43 | 71.10 | 75.79 | 4 |
| 14 | **75.53** | 75.45 | 73.92 | 76.71 | 8 |
| 15 | **75.73** | 74.30 | 67.65 | 77.01 | 44 |

**Best iteration: 11 — median 75.75, mean 75.44.**

## Emphasis contributions (higher weight => helps the objective)

| heuristic | weight-contribution | best-iter weight |
|---|---|---|
| pick_faster_side | +1.553 | 0.169 |
| avoid_edge_lanes | +1.071 | 0.095 |
| fast_or_far_not_blocking | +1.017 | 0.104 |
| top_speed_when_open | +0.801 | 0.116 |
| accelerate_when_clear | +0.764 | 0.090 |
| commit_after_change | +0.567 | 0.078 |
| best_of_blocked | +0.506 | 0.071 |
| keep_following_gap | +0.323 | 0.112 |
| overtake_early | -0.052 | 0.000 |
| rear_safety | -0.149 | 0.018 |
| wait_if_boxed_in | -0.385 | 0.000 |
| prefer_centre_on_tie | -0.838 | 0.024 |
| ignore_rear_when_not_overtaking | -1.434 | 0.000 |
| avoid_slow_close_lane | -1.659 | 0.079 |
| spread_to_outer_when_clear | -2.083 | 0.044 |

## Sequence contributions (higher => helps to state EARLIER)

| heuristic | order-contribution |
|---|---|
| ignore_rear_when_not_overtaking | +1.572 |
| best_of_blocked | +1.199 |
| accelerate_when_clear | +0.871 |
| wait_if_boxed_in | +0.654 |
| avoid_slow_close_lane | +0.306 |
| commit_after_change | +0.263 |
| spread_to_outer_when_clear | -0.015 |
| rear_safety | -0.040 |
| keep_following_gap | -0.060 |
| avoid_edge_lanes | -0.531 |
| top_speed_when_open | -0.533 |
| overtake_early | -0.575 |
| pick_faster_side | -0.809 |
| prefer_centre_on_tie | -0.870 |
| fast_or_far_not_blocking | -1.430 |

## Best prompt — sequence (heuristics in order)

ignore_rear_when_not_overtaking -> best_of_blocked -> keep_following_gap -> fast_or_far_not_blocking -> overtake_early -> accelerate_when_clear -> wait_if_boxed_in -> commit_after_change -> avoid_slow_close_lane -> pick_faster_side -> spread_to_outer_when_clear -> top_speed_when_open -> prefer_centre_on_tie -> rear_safety -> avoid_edge_lanes

## Best system prompt

```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road right ahead allows. Follow these driving rules:
- (Also) If every lane has a slow car close ahead, move toward the lane whose nearest car is the fastest -- take the least-bad option rather than staying stuck behind the slowest.
- (Important) Keep a small following gap and do not tailgate the car directly ahead.
- (Important) A car ahead that is near top speed or far away does NOT block you -- you could not go faster anyway, so do not slow down, maintain, or change lanes for it; keep accelerating.
- (Important) Your DEFAULT action is to ACCELERATE: whenever your lane is open or the car ahead is fast, accelerate and stay in your lane -- never just maintain, because maintaining wastes the speed you could be regaining.
- (Also) Right after you change lanes, accelerate and hold the new lane; do not immediately change again or oscillate between lanes.
- (Also) NEVER move into a lane that itself has a slow car close ahead -- that is not a real escape, it only trades one block for another.
- (ALWAYS) When overtaking, move to the side lane that is open, or whose nearest car is faster or farther ahead than the car blocking you -- pick the genuinely faster side.
- (Also) When you are not blocked you may drift toward an open outer lane so the fleet spreads out, but only if it does not cost you any speed.
- (Important) When the road far ahead in your lane is clear, accelerate all the way to top speed -- an empty lane is the fastest lane, so do not hold back.
- (Important) Avoid the outermost lanes (1 and 7) when an inner lane is equally open.
Think in one short sentence, then choose one action.
```
