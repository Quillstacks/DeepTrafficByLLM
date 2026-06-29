# prompt-EM report — em_v1

- objective: **median fleet mph** (target: beat 76.3 ceiling)
- iterations: 8 x 10 runs
- policy model: qwen2.5:3b; synthesis: llm; co-optimising emphasis + sequence

## Trajectory (median & mean fleet mph per iteration)

| iter | median | mean | min | max | parse_fail |
|---|---|---|---|---|---|
| 0 | **68.64** | 66.99 | 61.79 | 70.98 | 1 |
| 1 | **39.75** | 35.91 | 21.99 | 45.47 | 0 |
| 2 | **73.61** | 73.06 | 67.91 | 76.20 | 0 |
| 3 | **67.64** | 67.51 | 63.06 | 71.15 | 0 |
| 4 | **61.74** | 61.67 | 56.29 | 65.92 | 0 |
| 5 | **63.59** | 63.62 | 58.30 | 71.04 | 0 |
| 6 | **66.97** | 64.73 | 43.55 | 71.14 | 0 |
| 7 | **53.53** | 51.78 | 38.63 | 61.79 | 0 |

**Best iteration: 2 — median 73.61, mean 73.06.**

## Emphasis contributions (higher weight => helps the objective)

| heuristic | weight-contribution | best-iter weight |
|---|---|---|
| avoid_slow_close_lane | +3.358 | 0.237 |
| pick_faster_side | +1.639 | 0.048 |
| prefer_centre_on_tie | +1.549 | 0.084 |
| wait_if_boxed_in | +1.230 | 0.228 |
| fast_or_far_not_blocking | +1.003 | 0.100 |
| rear_safety | +0.949 | 0.028 |
| accelerate_when_clear | +0.944 | 0.118 |
| ignore_rear_when_not_overtaking | -1.961 | 0.061 |
| spread_to_outer_when_clear | -2.790 | 0.096 |
| overtake_when_blocked | -5.922 | 0.000 |

## Sequence contributions (higher => helps to state EARLIER)

| heuristic | order-contribution |
|---|---|
| wait_if_boxed_in | +2.846 |
| prefer_centre_on_tie | +2.616 |
| overtake_when_blocked | +1.645 |
| spread_to_outer_when_clear | +1.100 |
| avoid_slow_close_lane | -0.375 |
| rear_safety | -1.082 |
| fast_or_far_not_blocking | -1.262 |
| accelerate_when_clear | -1.361 |
| ignore_rear_when_not_overtaking | -1.926 |
| pick_faster_side | -2.202 |

## Best prompt — sequence (heuristics in order)

spread_to_outer_when_clear -> avoid_slow_close_lane -> overtake_when_blocked -> ignore_rear_when_not_overtaking -> prefer_centre_on_tie -> wait_if_boxed_in -> accelerate_when_clear -> pick_faster_side -> rear_safety -> fast_or_far_not_blocking

## Best system prompt

```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road right ahead allows. Follow these driving rules:
When driving, prioritize spreading out in clear outer lanes to keep inner lanes free for passing (rule 1). Avoid moving into a lane with a slow car ahead as it's not an escape route (rule 2). If you're not overtaking and behind cars, stay put rather than changing lanes unnecessarily (rule 3). When choosing between side lanes, opt for the one closer to the center of the road if they are equally good for passing (rule 4). In situations where you're blocked, wait patiently in your lane or move aside only when a better option is available (rule 5). Accelerate and stay in your current lane if it's open or if the car ahead is fast (rule 6). When overtaking, choose the side lane that offers more space or has a faster or farther-ahead car nearby (rule 7). Finally, don't slow down or change lanes just because a car ahead is moving at top speed or far away as this won’t help you (rule 8).
Think in one short sentence, then choose one action.
```
