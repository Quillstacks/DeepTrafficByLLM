# prompt-EM report — freeze_order

- objective: **median fleet mph** (target: beat 76.3 ceiling)
- iterations: 16 x 10 runs
- policy model: qwen2.5:3b; synthesis: template; co-optimising emphasis + sequence

## Trajectory (median & mean fleet mph per iteration)

| iter | median | mean | min | max | parse_fail |
|---|---|---|---|---|---|
| 0 | **74.16** | 73.92 | 70.64 | 76.31 | 1 |
| 1 | **72.94** | 72.07 | 64.80 | 74.07 | 6 |
| 2 | **73.00** | 72.10 | 66.92 | 75.61 | 14 |
| 3 | **75.54** | 75.39 | 73.02 | 77.17 | 6 |
| 4 | **71.80** | 72.02 | 66.40 | 76.31 | 18 |
| 5 | **50.30** | 50.84 | 46.59 | 60.33 | 12 |
| 6 | **75.14** | 74.98 | 71.90 | 76.85 | 6 |
| 7 | **75.09** | 74.84 | 72.01 | 77.42 | 26 |
| 8 | **75.08** | 74.22 | 71.00 | 76.21 | 4 |
| 9 | **74.49** | 74.39 | 70.64 | 76.94 | 3 |
| 10 | **75.51** | 75.20 | 72.47 | 76.49 | 1 |
| 11 | **75.09** | 74.47 | 68.41 | 76.29 | 2 |
| 12 | **74.94** | 74.72 | 71.00 | 76.31 | 3 |
| 13 | **74.88** | 75.05 | 73.50 | 76.71 | 1 |
| 14 | **74.79** | 74.92 | 73.53 | 76.38 | 4 |
| 15 | **75.57** | 75.29 | 72.45 | 77.36 | 3 |

**Best iteration: 15 — median 75.57, mean 75.29.**

## Emphasis contributions (higher weight => helps the objective)

| heuristic | weight-contribution | best-iter weight |
|---|---|---|
| top_speed_when_open | +2.272 | 0.212 |
| overtake_when_blocked | +2.262 | 0.108 |
| best_of_blocked | +2.205 | 0.110 |
| keep_following_gap | +0.934 | 0.052 |
| spread_to_outer_when_clear | +0.827 | 0.070 |
| commit_after_change | +0.737 | 0.050 |
| overtake_early | +0.540 | 0.055 |
| avoid_slow_close_lane | +0.349 | 0.024 |
| fast_or_far_not_blocking | +0.031 | 0.026 |
| accelerate_when_clear | -0.061 | 0.000 |
| ignore_rear_when_not_overtaking | -0.275 | 0.022 |
| wait_if_boxed_in | -0.470 | 0.017 |
| rear_safety | -1.251 | 0.097 |
| avoid_edge_lanes | -1.439 | 0.017 |
| pick_faster_side | -2.822 | 0.116 |
| prefer_centre_on_tie | -3.839 | 0.025 |

## Sequence contributions (higher => helps to state EARLIER)

| heuristic | order-contribution |
|---|---|
| ignore_rear_when_not_overtaking | +0.000 |
| wait_if_boxed_in | +0.000 |
| overtake_early | +0.000 |
| overtake_when_blocked | +0.000 |
| top_speed_when_open | +0.000 |
| keep_following_gap | +0.000 |
| prefer_centre_on_tie | +0.000 |
| spread_to_outer_when_clear | +0.000 |
| fast_or_far_not_blocking | +0.000 |
| commit_after_change | +0.000 |
| avoid_edge_lanes | -0.000 |
| rear_safety | -0.000 |
| best_of_blocked | -0.000 |
| avoid_slow_close_lane | -0.000 |
| accelerate_when_clear | -0.000 |
| pick_faster_side | -0.000 |

## Best prompt — sequence (heuristics in order)

accelerate_when_clear -> overtake_when_blocked -> fast_or_far_not_blocking -> pick_faster_side -> avoid_slow_close_lane -> top_speed_when_open -> ignore_rear_when_not_overtaking -> wait_if_boxed_in -> overtake_early -> rear_safety -> best_of_blocked -> commit_after_change -> prefer_centre_on_tie -> keep_following_gap -> avoid_edge_lanes -> spread_to_outer_when_clear

## Best system prompt

```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road right ahead allows. Follow these driving rules:
- (Important) When a SLOW car is close ahead in your lane you are stuck at its speed and accelerating is useless, so you MUST change to an adjacent lane to overtake instead of staying behind it.
- (Important) When overtaking, move to the side lane that is open, or whose nearest car is faster or farther ahead than the car blocking you -- pick the genuinely faster side.
- (ALWAYS) When the road far ahead in your lane is clear, accelerate all the way to top speed -- an empty lane is the fastest lane, so do not hold back.
- (Also) Begin overtaking as soon as a slow car is close ahead; do not wait until you are right behind it, by which time the adjacent lane may have filled.
- (Also) Do not change into a lane where a fast car is close behind you; you would cut it off and lose speed.
- (Important) If every lane has a slow car close ahead, move toward the lane whose nearest car is the fastest -- take the least-bad option rather than staying stuck behind the slowest.
- (Also) Right after you change lanes, accelerate and hold the new lane; do not immediately change again or oscillate between lanes.
- (Also) Keep a small following gap and do not tailgate the car directly ahead.
- (Also) When you are not blocked you may drift toward an open outer lane so the fleet spreads out, but only if it does not cost you any speed.
Think in one short sentence, then choose one action.
```
