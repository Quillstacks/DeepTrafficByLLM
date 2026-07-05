# prompt-EM report — omit_accelerate_when_clear_VAL

- objective: **median fleet mph** (target: beat 76.3 ceiling)
- iterations: 16 x 10 runs
- policy model: qwen2.5:3b; synthesis: template; co-optimising emphasis + sequence

## Trajectory (median & mean fleet mph per iteration)

| iter | median | mean | min | max | parse_fail |
|---|---|---|---|---|---|
| 0 | **71.75** | 71.59 | 69.21 | 74.39 | 2 |
| 1 | **75.37** | 74.14 | 71.92 | 76.35 | 2 |
| 2 | **60.06** | 57.95 | 38.01 | 73.17 | 8 |
| 3 | **71.74** | 66.22 | 43.33 | 73.68 | 2 |
| 4 | **74.27** | 74.04 | 70.48 | 75.99 | 3 |
| 5 | **63.28** | 63.23 | 61.59 | 65.09 | 1 |
| 6 | **74.34** | 74.38 | 71.08 | 77.08 | 1 |
| 7 | **75.35** | 75.11 | 73.25 | 76.44 | 3 |
| 8 | **73.88** | 73.89 | 71.91 | 75.49 | 12 |
| 9 | **73.22** | 71.20 | 60.04 | 74.92 | 3 |
| 10 | **74.72** | 74.79 | 72.74 | 76.70 | 7 |
| 11 | **73.71** | 73.32 | 69.25 | 75.42 | 6 |
| 12 | **75.40** | 74.42 | 68.78 | 76.35 | 6 |
| 13 | **74.87** | 74.69 | 73.18 | 75.69 | 26 |
| 14 | **72.71** | 72.05 | 69.60 | 73.57 | 2 |
| 15 | **74.44** | 74.16 | 71.97 | 75.42 | 3 |

**Best iteration: 12 — median 75.40, mean 74.42.**

## Emphasis contributions (higher weight => helps the objective)

| heuristic | weight-contribution | best-iter weight |
|---|---|---|
| avoid_edge_lanes | +2.639 | 0.174 |
| pick_faster_side | +1.478 | 0.034 |
| top_speed_when_open | +1.429 | 0.086 |
| overtake_when_blocked | +1.334 | 0.163 |
| fast_or_far_not_blocking | +1.258 | 0.085 |
| best_of_blocked | +0.753 | 0.116 |
| overtake_early | +0.555 | 0.088 |
| prefer_centre_on_tie | -0.014 | 0.078 |
| ignore_rear_when_not_overtaking | -0.705 | 0.040 |
| keep_following_gap | -0.737 | 0.024 |
| commit_after_change | -0.862 | 0.019 |
| avoid_slow_close_lane | -1.101 | 0.021 |
| rear_safety | -1.762 | 0.031 |
| wait_if_boxed_in | -1.770 | 0.040 |
| spread_to_outer_when_clear | -2.495 | 0.002 |

## Sequence contributions (higher => helps to state EARLIER)

| heuristic | order-contribution |
|---|---|
| spread_to_outer_when_clear | +1.734 |
| commit_after_change | +1.600 |
| wait_if_boxed_in | +1.317 |
| top_speed_when_open | +0.637 |
| fast_or_far_not_blocking | +0.591 |
| avoid_slow_close_lane | +0.464 |
| best_of_blocked | +0.446 |
| ignore_rear_when_not_overtaking | +0.256 |
| rear_safety | +0.079 |
| pick_faster_side | -0.126 |
| prefer_centre_on_tie | -0.806 |
| overtake_early | -0.922 |
| overtake_when_blocked | -1.672 |
| avoid_edge_lanes | -1.684 |
| keep_following_gap | -1.913 |

## Best prompt — sequence (heuristics in order)

top_speed_when_open -> rear_safety -> spread_to_outer_when_clear -> commit_after_change -> avoid_slow_close_lane -> best_of_blocked -> wait_if_boxed_in -> pick_faster_side -> fast_or_far_not_blocking -> keep_following_gap -> overtake_early -> overtake_when_blocked -> ignore_rear_when_not_overtaking -> prefer_centre_on_tie -> avoid_edge_lanes

## Best system prompt

```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road right ahead allows. Follow these driving rules:
- (Also) When the road far ahead in your lane is clear, accelerate all the way to top speed -- an empty lane is the fastest lane, so do not hold back.
- (Important) If every lane has a slow car close ahead, move toward the lane whose nearest car is the fastest -- take the least-bad option rather than staying stuck behind the slowest.
- (Also) A car ahead that is near top speed or far away does NOT block you -- you could not go faster anyway, so do not slow down, maintain, or change lanes for it; keep accelerating.
- (Important) Begin overtaking as soon as a slow car is close ahead; do not wait until you are right behind it, by which time the adjacent lane may have filled.
- (ALWAYS) When a SLOW car is close ahead in your lane you are stuck at its speed and accelerating is useless, so you MUST change to an adjacent lane to overtake instead of staying behind it.
- (Also) When you are not overtaking, ignore the cars behind you completely and never move aside for them.
- (Also) When two side lanes are equally good for overtaking, prefer the one toward the centre of the road.
- (ALWAYS) Avoid the outermost lanes (1 and 7) when an inner lane is equally open.
Think in one short sentence, then choose one action.
```
