# prompt-EM report — omit_prefer_centre_on_tie

- objective: **median fleet mph** (target: beat 76.3 ceiling)
- iterations: 16 x 10 runs
- policy model: qwen2.5:3b; synthesis: template; co-optimising emphasis + sequence

## Trajectory (median & mean fleet mph per iteration)

| iter | median | mean | min | max | parse_fail |
|---|---|---|---|---|---|
| 0 | **73.42** | 73.69 | 70.84 | 76.66 | 7 |
| 1 | **74.77** | 74.01 | 69.15 | 76.73 | 0 |
| 2 | **74.63** | 74.22 | 71.48 | 76.48 | 4 |
| 3 | **74.84** | 73.98 | 65.93 | 76.18 | 4 |
| 4 | **75.65** | 75.46 | 73.39 | 77.14 | 62 |
| 5 | **38.25** | 39.87 | 32.16 | 47.54 | 1 |
| 6 | **74.62** | 74.41 | 71.97 | 75.83 | 9 |
| 7 | **74.38** | 74.31 | 71.21 | 76.12 | 5 |
| 8 | **74.79** | 74.55 | 72.53 | 77.02 | 7 |
| 9 | **75.13** | 74.93 | 73.07 | 76.43 | 4 |
| 10 | **73.68** | 72.57 | 65.57 | 74.97 | 3 |
| 11 | **74.46** | 73.03 | 57.63 | 76.80 | 1 |
| 12 | **74.41** | 73.31 | 65.83 | 77.41 | 1 |
| 13 | **75.52** | 75.27 | 73.06 | 76.87 | 11 |
| 14 | **73.84** | 72.77 | 65.58 | 76.23 | 0 |
| 15 | **73.94** | 73.74 | 70.17 | 77.24 | 5 |

**Best iteration: 4 — median 75.65, mean 75.46.**

## Emphasis contributions (higher weight => helps the objective)

| heuristic | weight-contribution | best-iter weight |
|---|---|---|
| top_speed_when_open | +2.417 | 0.140 |
| avoid_edge_lanes | +2.415 | 0.111 |
| overtake_when_blocked | +2.363 | 0.037 |
| avoid_slow_close_lane | +2.346 | 0.122 |
| fast_or_far_not_blocking | +2.270 | 0.154 |
| best_of_blocked | +2.184 | 0.103 |
| overtake_early | +1.323 | 0.068 |
| commit_after_change | +1.277 | 0.000 |
| accelerate_when_clear | +0.802 | 0.084 |
| keep_following_gap | +0.026 | 0.004 |
| wait_if_boxed_in | -0.063 | 0.049 |
| ignore_rear_when_not_overtaking | -1.878 | 0.020 |
| rear_safety | -3.830 | 0.005 |
| pick_faster_side | -5.508 | 0.064 |
| spread_to_outer_when_clear | -6.142 | 0.037 |

## Sequence contributions (higher => helps to state EARLIER)

| heuristic | order-contribution |
|---|---|
| commit_after_change | +2.251 |
| best_of_blocked | +1.956 |
| rear_safety | +1.909 |
| accelerate_when_clear | +1.843 |
| pick_faster_side | +1.794 |
| avoid_slow_close_lane | +1.773 |
| wait_if_boxed_in | +1.460 |
| avoid_edge_lanes | +1.077 |
| spread_to_outer_when_clear | +0.685 |
| keep_following_gap | -0.603 |
| ignore_rear_when_not_overtaking | -1.809 |
| top_speed_when_open | -1.828 |
| overtake_when_blocked | -2.351 |
| overtake_early | -2.909 |
| fast_or_far_not_blocking | -5.247 |

## Best prompt — sequence (heuristics in order)

avoid_slow_close_lane -> accelerate_when_clear -> avoid_edge_lanes -> commit_after_change -> fast_or_far_not_blocking -> top_speed_when_open -> overtake_early -> ignore_rear_when_not_overtaking -> rear_safety -> pick_faster_side -> wait_if_boxed_in -> keep_following_gap -> best_of_blocked -> overtake_when_blocked -> spread_to_outer_when_clear

## Best system prompt

```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road right ahead allows. Follow these driving rules:
- (Important) NEVER move into a lane that itself has a slow car close ahead -- that is not a real escape, it only trades one block for another.
- (Important) Your DEFAULT action is to ACCELERATE: whenever your lane is open or the car ahead is fast, accelerate and stay in your lane -- never just maintain, because maintaining wastes the speed you could be regaining.
- (Important) Avoid the outermost lanes (1 and 7) when an inner lane is equally open.
- (ALWAYS) A car ahead that is near top speed or far away does NOT block you -- you could not go faster anyway, so do not slow down, maintain, or change lanes for it; keep accelerating.
- (ALWAYS) When the road far ahead in your lane is clear, accelerate all the way to top speed -- an empty lane is the fastest lane, so do not hold back.
- (Also) Begin overtaking as soon as a slow car is close ahead; do not wait until you are right behind it, by which time the adjacent lane may have filled.
- (Also) When overtaking, move to the side lane that is open, or whose nearest car is faster or farther ahead than the car blocking you -- pick the genuinely faster side.
- (Also) If you are blocked and neither side lane is genuinely better, ease off briefly and wait for a gap rather than forcing a bad lane change.
- (Important) If every lane has a slow car close ahead, move toward the lane whose nearest car is the fastest -- take the least-bad option rather than staying stuck behind the slowest.
Think in one short sentence, then choose one action.
```
