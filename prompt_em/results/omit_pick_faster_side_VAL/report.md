# prompt-EM report — omit_pick_faster_side_VAL

- objective: **median fleet mph** (target: beat 76.3 ceiling)
- iterations: 16 x 10 runs
- policy model: qwen2.5:3b; synthesis: template; co-optimising emphasis + sequence

## Trajectory (median & mean fleet mph per iteration)

| iter | median | mean | min | max | parse_fail |
|---|---|---|---|---|---|
| 0 | **74.02** | 72.66 | 67.68 | 75.34 | 9 |
| 1 | **74.70** | 74.19 | 70.57 | 76.92 | 2 |
| 2 | **68.13** | 66.39 | 48.07 | 74.51 | 8 |
| 3 | **73.16** | 67.97 | 45.50 | 74.16 | 2 |
| 4 | **75.62** | 75.11 | 73.30 | 77.15 | 3 |
| 5 | **63.89** | 63.59 | 61.47 | 65.33 | 0 |
| 6 | **73.81** | 74.25 | 73.26 | 76.47 | 1 |
| 7 | **74.14** | 73.36 | 69.67 | 75.62 | 4 |
| 8 | **75.48** | 75.25 | 73.93 | 76.86 | 5 |
| 9 | **75.49** | 74.76 | 70.45 | 76.46 | 5 |
| 10 | **74.45** | 74.14 | 71.80 | 75.60 | 18 |
| 11 | **73.88** | 73.86 | 72.04 | 76.13 | 12 |
| 12 | **72.87** | 72.82 | 70.71 | 76.72 | 7 |
| 13 | **75.35** | 74.49 | 71.62 | 76.23 | 2 |
| 14 | **74.41** | 73.79 | 70.89 | 75.79 | 2 |
| 15 | **75.31** | 74.42 | 65.77 | 76.77 | 4 |

**Best iteration: 4 — median 75.62, mean 75.11.**

## Emphasis contributions (higher weight => helps the objective)

| heuristic | weight-contribution | best-iter weight |
|---|---|---|
| top_speed_when_open | +1.453 | 0.140 |
| avoid_edge_lanes | +1.367 | 0.111 |
| fast_or_far_not_blocking | +1.206 | 0.154 |
| overtake_when_blocked | +0.945 | 0.037 |
| accelerate_when_clear | +0.637 | 0.084 |
| best_of_blocked | +0.629 | 0.103 |
| overtake_early | +0.296 | 0.068 |
| wait_if_boxed_in | -0.039 | 0.122 |
| commit_after_change | -0.329 | 0.000 |
| keep_following_gap | -0.525 | 0.004 |
| prefer_centre_on_tie | -0.534 | 0.020 |
| rear_safety | -0.724 | 0.049 |
| ignore_rear_when_not_overtaking | -1.162 | 0.005 |
| avoid_slow_close_lane | -1.269 | 0.064 |
| spread_to_outer_when_clear | -1.950 | 0.037 |

## Sequence contributions (higher => helps to state EARLIER)

| heuristic | order-contribution |
|---|---|
| wait_if_boxed_in | +1.557 |
| commit_after_change | +1.405 |
| accelerate_when_clear | +1.187 |
| ignore_rear_when_not_overtaking | +0.393 |
| spread_to_outer_when_clear | +0.295 |
| avoid_slow_close_lane | +0.292 |
| best_of_blocked | +0.062 |
| rear_safety | -0.158 |
| avoid_edge_lanes | -0.329 |
| top_speed_when_open | -0.409 |
| overtake_when_blocked | -0.535 |
| prefer_centre_on_tie | -0.601 |
| overtake_early | -0.711 |
| keep_following_gap | -1.112 |
| fast_or_far_not_blocking | -1.336 |

## Best prompt — sequence (heuristics in order)

wait_if_boxed_in -> accelerate_when_clear -> avoid_edge_lanes -> commit_after_change -> fast_or_far_not_blocking -> top_speed_when_open -> overtake_early -> prefer_centre_on_tie -> ignore_rear_when_not_overtaking -> avoid_slow_close_lane -> rear_safety -> keep_following_gap -> best_of_blocked -> overtake_when_blocked -> spread_to_outer_when_clear

## Best system prompt

```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road right ahead allows. Follow these driving rules:
- (Important) If you are blocked and neither side lane is genuinely better, ease off briefly and wait for a gap rather than forcing a bad lane change.
- (Important) Your DEFAULT action is to ACCELERATE: whenever your lane is open or the car ahead is fast, accelerate and stay in your lane -- never just maintain, because maintaining wastes the speed you could be regaining.
- (Important) Avoid the outermost lanes (1 and 7) when an inner lane is equally open.
- (ALWAYS) A car ahead that is near top speed or far away does NOT block you -- you could not go faster anyway, so do not slow down, maintain, or change lanes for it; keep accelerating.
- (ALWAYS) When the road far ahead in your lane is clear, accelerate all the way to top speed -- an empty lane is the fastest lane, so do not hold back.
- (Also) Begin overtaking as soon as a slow car is close ahead; do not wait until you are right behind it, by which time the adjacent lane may have filled.
- (Also) NEVER move into a lane that itself has a slow car close ahead -- that is not a real escape, it only trades one block for another.
- (Also) Do not change into a lane where a fast car is close behind you; you would cut it off and lose speed.
- (Important) If every lane has a slow car close ahead, move toward the lane whose nearest car is the fastest -- take the least-bad option rather than staying stuck behind the slowest.
Think in one short sentence, then choose one action.
```
