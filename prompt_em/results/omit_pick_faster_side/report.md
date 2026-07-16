# prompt-EM report — omit_pick_faster_side

- objective: **median fleet mph** (target: beat 76.3 ceiling)
- iterations: 16 x 10 runs
- policy model: qwen2.5:3b; synthesis: template; co-optimising emphasis + sequence

## Trajectory (median & mean fleet mph per iteration)

| iter | median | mean | min | max | parse_fail |
|---|---|---|---|---|---|
| 0 | **74.11** | 73.83 | 71.42 | 75.58 | 14 |
| 1 | **74.21** | 74.15 | 72.56 | 76.25 | 0 |
| 2 | **72.66** | 71.70 | 67.15 | 75.07 | 14 |
| 3 | **72.79** | 69.21 | 50.76 | 74.40 | 1 |
| 4 | **74.64** | 74.28 | 71.64 | 75.89 | 3 |
| 5 | **64.28** | 63.33 | 60.80 | 65.18 | 2 |
| 6 | **74.76** | 74.04 | 69.74 | 75.82 | 0 |
| 7 | **74.55** | 73.52 | 70.71 | 76.01 | 1 |
| 8 | **73.69** | 73.34 | 68.63 | 75.67 | 0 |
| 9 | **75.66** | 75.61 | 74.10 | 77.07 | 0 |
| 10 | **74.63** | 74.30 | 71.95 | 76.82 | 0 |
| 11 | **74.69** | 74.82 | 73.00 | 77.14 | 3 |
| 12 | **74.77** | 74.80 | 73.36 | 76.37 | 2 |
| 13 | **75.51** | 74.62 | 69.74 | 76.53 | 2 |
| 14 | **74.65** | 74.49 | 70.47 | 77.29 | 0 |
| 15 | **75.38** | 74.98 | 72.71 | 76.79 | 1 |

**Best iteration: 9 — median 75.66, mean 75.61.**

## Emphasis contributions (higher weight => helps the objective)

| heuristic | weight-contribution | best-iter weight |
|---|---|---|
| overtake_when_blocked | +1.965 | 0.282 |
| top_speed_when_open | +0.799 | 0.103 |
| avoid_edge_lanes | +0.750 | 0.097 |
| best_of_blocked | +0.571 | 0.075 |
| overtake_early | +0.471 | 0.125 |
| commit_after_change | +0.230 | 0.033 |
| fast_or_far_not_blocking | +0.222 | 0.000 |
| wait_if_boxed_in | +0.202 | 0.070 |
| accelerate_when_clear | +0.040 | 0.000 |
| keep_following_gap | -0.176 | 0.017 |
| rear_safety | -0.267 | 0.000 |
| prefer_centre_on_tie | -0.655 | 0.024 |
| ignore_rear_when_not_overtaking | -0.952 | 0.080 |
| avoid_slow_close_lane | -1.264 | 0.092 |
| spread_to_outer_when_clear | -1.935 | 0.000 |

## Sequence contributions (higher => helps to state EARLIER)

| heuristic | order-contribution |
|---|---|
| ignore_rear_when_not_overtaking | +0.906 |
| rear_safety | +0.856 |
| wait_if_boxed_in | +0.723 |
| accelerate_when_clear | +0.692 |
| spread_to_outer_when_clear | +0.555 |
| avoid_slow_close_lane | +0.450 |
| commit_after_change | +0.307 |
| best_of_blocked | +0.248 |
| avoid_edge_lanes | -0.051 |
| keep_following_gap | -0.432 |
| prefer_centre_on_tie | -0.483 |
| overtake_when_blocked | -0.579 |
| top_speed_when_open | -0.741 |
| overtake_early | -0.850 |
| fast_or_far_not_blocking | -1.602 |

## Best prompt — sequence (heuristics in order)

rear_safety -> wait_if_boxed_in -> spread_to_outer_when_clear -> accelerate_when_clear -> ignore_rear_when_not_overtaking -> avoid_edge_lanes -> commit_after_change -> overtake_when_blocked -> prefer_centre_on_tie -> avoid_slow_close_lane -> best_of_blocked -> fast_or_far_not_blocking -> overtake_early -> keep_following_gap -> top_speed_when_open

## Best system prompt

```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road right ahead allows. Follow these driving rules:
- (Also) If you are blocked and neither side lane is genuinely better, ease off briefly and wait for a gap rather than forcing a bad lane change.
- (Also) When you are not overtaking, ignore the cars behind you completely and never move aside for them.
- (Also) Avoid the outermost lanes (1 and 7) when an inner lane is equally open.
- (ALWAYS) When a SLOW car is close ahead in your lane you are stuck at its speed and accelerating is useless, so you MUST change to an adjacent lane to overtake instead of staying behind it.
- (Also) NEVER move into a lane that itself has a slow car close ahead -- that is not a real escape, it only trades one block for another.
- (Also) If every lane has a slow car close ahead, move toward the lane whose nearest car is the fastest -- take the least-bad option rather than staying stuck behind the slowest.
- (Also) Begin overtaking as soon as a slow car is close ahead; do not wait until you are right behind it, by which time the adjacent lane may have filled.
- (Also) When the road far ahead in your lane is clear, accelerate all the way to top speed -- an empty lane is the fastest lane, so do not hold back.
Think in one short sentence, then choose one action.
```
