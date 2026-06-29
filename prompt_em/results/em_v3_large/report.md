# prompt-EM report — em_v3_large

- objective: **median fleet mph** (target: beat 76.3 ceiling)
- iterations: 16 x 10 runs
- policy model: qwen2.5:3b; synthesis: template; co-optimising emphasis + sequence

## Trajectory (median & mean fleet mph per iteration)

| iter | median | mean | min | max | parse_fail |
|---|---|---|---|---|---|
| 0 | **73.09** | 73.30 | 71.88 | 75.04 | 6 |
| 1 | **73.00** | 72.27 | 67.07 | 75.91 | 0 |
| 2 | **74.41** | 73.38 | 66.67 | 77.15 | 15 |
| 3 | **73.63** | 72.99 | 69.66 | 76.49 | 7 |
| 4 | **73.42** | 72.04 | 57.70 | 76.49 | 1 |
| 5 | **65.07** | 64.83 | 57.12 | 72.53 | 1 |
| 6 | **75.94** | 75.39 | 73.42 | 76.34 | 12 |
| 7 | **74.55** | 74.11 | 69.65 | 76.33 | 41 |
| 8 | **75.03** | 75.08 | 74.31 | 76.07 | 0 |
| 9 | **74.48** | 74.23 | 71.09 | 76.94 | 0 |
| 10 | **75.77** | 75.52 | 73.88 | 77.59 | 1 |
| 11 | **74.56** | 74.47 | 72.26 | 76.75 | 1 |
| 12 | **75.05** | 74.51 | 71.72 | 76.06 | 0 |
| 13 | **75.90** | 75.72 | 74.13 | 76.59 | 0 |
| 14 | **73.90** | 74.12 | 71.00 | 77.36 | 12 |
| 15 | **75.46** | 75.33 | 72.89 | 78.03 | 0 |

**Best iteration: 6 — median 75.94, mean 75.39.**

## Emphasis contributions (higher weight => helps the objective)

| heuristic | weight-contribution | best-iter weight |
|---|---|---|
| overtake_when_blocked | +1.911 | 0.274 |
| top_speed_when_open | +1.082 | 0.170 |
| keep_following_gap | +0.671 | 0.127 |
| best_of_blocked | +0.619 | 0.064 |
| spread_to_outer_when_clear | +0.135 | 0.044 |
| overtake_early | +0.018 | 0.021 |
| commit_after_change | -0.028 | 0.000 |
| avoid_slow_close_lane | -0.044 | 0.000 |
| wait_if_boxed_in | -0.073 | 0.000 |
| fast_or_far_not_blocking | -0.114 | 0.000 |
| accelerate_when_clear | -0.206 | 0.004 |
| avoid_edge_lanes | -0.328 | 0.073 |
| rear_safety | -0.339 | 0.144 |
| ignore_rear_when_not_overtaking | -0.658 | 0.008 |
| pick_faster_side | -0.998 | 0.068 |
| prefer_centre_on_tie | -1.648 | 0.001 |

## Sequence contributions (higher => helps to state EARLIER)

| heuristic | order-contribution |
|---|---|
| rear_safety | +0.935 |
| wait_if_boxed_in | +0.761 |
| keep_following_gap | +0.678 |
| avoid_slow_close_lane | +0.475 |
| pick_faster_side | +0.470 |
| overtake_early | +0.469 |
| accelerate_when_clear | +0.430 |
| prefer_centre_on_tie | +0.410 |
| ignore_rear_when_not_overtaking | +0.150 |
| avoid_edge_lanes | -0.121 |
| commit_after_change | -0.269 |
| best_of_blocked | -0.333 |
| overtake_when_blocked | -0.636 |
| spread_to_outer_when_clear | -0.849 |
| top_speed_when_open | -0.916 |
| fast_or_far_not_blocking | -1.654 |

## Best prompt — sequence (heuristics in order)

ignore_rear_when_not_overtaking -> avoid_slow_close_lane -> keep_following_gap -> rear_safety -> accelerate_when_clear -> wait_if_boxed_in -> prefer_centre_on_tie -> pick_faster_side -> avoid_edge_lanes -> top_speed_when_open -> overtake_early -> overtake_when_blocked -> spread_to_outer_when_clear -> fast_or_far_not_blocking -> commit_after_change -> best_of_blocked

## Best system prompt

```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road right ahead allows. Follow these driving rules:
- (Also) Keep a small following gap and do not tailgate the car directly ahead.
- (Important) Do not change into a lane where a fast car is close behind you; you would cut it off and lose speed.
- (Also) When overtaking, move to the side lane that is open, or whose nearest car is faster or farther ahead than the car blocking you -- pick the genuinely faster side.
- (Also) Avoid the outermost lanes (1 and 7) when an inner lane is equally open.
- (Important) When the road far ahead in your lane is clear, accelerate all the way to top speed -- an empty lane is the fastest lane, so do not hold back.
- (ALWAYS) When a SLOW car is close ahead in your lane you are stuck at its speed and accelerating is useless, so you MUST change to an adjacent lane to overtake instead of staying behind it.
- (Also) When you are not blocked you may drift toward an open outer lane so the fleet spreads out, but only if it does not cost you any speed.
- (Also) If every lane has a slow car close ahead, move toward the lane whose nearest car is the fastest -- take the least-bad option rather than staying stuck behind the slowest.
Think in one short sentence, then choose one action.
```
