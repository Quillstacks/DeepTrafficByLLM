# prompt-EM report — omit_accelerate_when_clear

- objective: **median fleet mph** (target: beat 76.3 ceiling)
- iterations: 16 x 10 runs
- policy model: qwen2.5:3b; synthesis: template; co-optimising emphasis + sequence

## Trajectory (median & mean fleet mph per iteration)

| iter | median | mean | min | max | parse_fail |
|---|---|---|---|---|---|
| 0 | **74.25** | 73.55 | 68.76 | 75.67 | 1 |
| 1 | **76.09** | 75.52 | 73.28 | 76.83 | 2 |
| 2 | **68.72** | 66.11 | 45.73 | 74.06 | 4 |
| 3 | **72.03** | 67.65 | 45.86 | 74.98 | 3 |
| 4 | **73.89** | 74.09 | 72.49 | 76.19 | 0 |
| 5 | **64.40** | 63.66 | 58.87 | 66.62 | 0 |
| 6 | **74.87** | 74.44 | 70.24 | 76.22 | 2 |
| 7 | **73.85** | 73.96 | 70.73 | 76.68 | 4 |
| 8 | **70.50** | 70.50 | 68.12 | 73.53 | 0 |
| 9 | **75.35** | 73.76 | 68.38 | 76.32 | 4 |
| 10 | **74.68** | 74.51 | 71.97 | 76.63 | 10 |
| 11 | **73.33** | 73.56 | 68.78 | 77.48 | 0 |
| 12 | **75.48** | 75.35 | 74.10 | 76.41 | 1 |
| 13 | **74.40** | 73.88 | 70.76 | 75.46 | 2 |
| 14 | **74.42** | 72.96 | 64.68 | 75.89 | 0 |
| 15 | **73.99** | 73.38 | 71.48 | 74.87 | 14 |

**Best iteration: 1 — median 76.09, mean 75.52.**

## Emphasis contributions (higher weight => helps the objective)

| heuristic | weight-contribution | best-iter weight |
|---|---|---|
| avoid_edge_lanes | +1.312 | 0.197 |
| best_of_blocked | +1.149 | 0.171 |
| pick_faster_side | +0.994 | 0.023 |
| top_speed_when_open | +0.900 | 0.059 |
| fast_or_far_not_blocking | +0.537 | 0.137 |
| overtake_when_blocked | +0.467 | 0.091 |
| overtake_early | +0.370 | 0.076 |
| commit_after_change | -0.160 | 0.014 |
| prefer_centre_on_tie | -0.203 | 0.069 |
| keep_following_gap | -0.341 | 0.027 |
| wait_if_boxed_in | -0.386 | 0.071 |
| rear_safety | -0.588 | 0.015 |
| ignore_rear_when_not_overtaking | -0.730 | 0.033 |
| avoid_slow_close_lane | -1.290 | 0.006 |
| spread_to_outer_when_clear | -2.031 | 0.010 |

## Sequence contributions (higher => helps to state EARLIER)

| heuristic | order-contribution |
|---|---|
| spread_to_outer_when_clear | +1.085 |
| rear_safety | +0.882 |
| commit_after_change | +0.637 |
| wait_if_boxed_in | +0.608 |
| avoid_slow_close_lane | +0.525 |
| ignore_rear_when_not_overtaking | +0.345 |
| best_of_blocked | +0.342 |
| fast_or_far_not_blocking | +0.323 |
| top_speed_when_open | +0.272 |
| pick_faster_side | -0.471 |
| prefer_centre_on_tie | -0.591 |
| avoid_edge_lanes | -0.707 |
| keep_following_gap | -0.824 |
| overtake_early | -0.869 |
| overtake_when_blocked | -1.557 |

## Best prompt — sequence (heuristics in order)

top_speed_when_open -> best_of_blocked -> commit_after_change -> spread_to_outer_when_clear -> rear_safety -> avoid_slow_close_lane -> wait_if_boxed_in -> keep_following_gap -> pick_faster_side -> avoid_edge_lanes -> prefer_centre_on_tie -> overtake_early -> fast_or_far_not_blocking -> overtake_when_blocked -> ignore_rear_when_not_overtaking

## Best system prompt

```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road right ahead allows. Follow these driving rules:
- (Also) When the road far ahead in your lane is clear, accelerate all the way to top speed -- an empty lane is the fastest lane, so do not hold back.
- (ALWAYS) If every lane has a slow car close ahead, move toward the lane whose nearest car is the fastest -- take the least-bad option rather than staying stuck behind the slowest.
- (Also) If you are blocked and neither side lane is genuinely better, ease off briefly and wait for a gap rather than forcing a bad lane change.
- (ALWAYS) Avoid the outermost lanes (1 and 7) when an inner lane is equally open.
- (Also) When two side lanes are equally good for overtaking, prefer the one toward the centre of the road.
- (Also) Begin overtaking as soon as a slow car is close ahead; do not wait until you are right behind it, by which time the adjacent lane may have filled.
- (Important) A car ahead that is near top speed or far away does NOT block you -- you could not go faster anyway, so do not slow down, maintain, or change lanes for it; keep accelerating.
- (Also) When a SLOW car is close ahead in your lane you are stuck at its speed and accelerating is useless, so you MUST change to an adjacent lane to overtake instead of staying behind it.
Think in one short sentence, then choose one action.
```
