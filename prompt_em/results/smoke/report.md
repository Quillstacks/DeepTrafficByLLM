# prompt-EM report — smoke

- objective: **median fleet mph** (target: beat 76.3 ceiling)
- iterations: 2 x 2 runs
- policy model: qwen2.5:3b; synthesis: llm; co-optimising emphasis + sequence

## Trajectory (median & mean fleet mph per iteration)

| iter | median | mean | min | max | parse_fail |
|---|---|---|---|---|---|
| 0 | **66.05** | 65.30 | 64.54 | 66.05 | 0 |
| 1 | **62.72** | 62.60 | 62.49 | 62.72 | 0 |

**Best iteration: 0 — median 66.05, mean 65.30.**

## Emphasis contributions (higher weight => helps the objective)

| heuristic | weight-contribution | best-iter weight |
|---|---|---|
| overtake_when_blocked | +0.194 | 0.138 |
| ignore_rear_when_not_overtaking | +0.135 | 0.092 |
| wait_if_boxed_in | +0.132 | 0.092 |
| rear_safety | +0.108 | 0.083 |
| accelerate_when_clear | +0.064 | 0.147 |
| pick_faster_side | +0.053 | 0.119 |
| fast_or_far_not_blocking | -0.016 | 0.128 |
| avoid_slow_close_lane | -0.036 | 0.119 |
| prefer_centre_on_tie | -0.192 | 0.055 |
| spread_to_outer_when_clear | -0.441 | 0.028 |

## Sequence contributions (higher => helps to state EARLIER)

| heuristic | order-contribution |
|---|---|
| avoid_slow_close_lane | +0.179 |
| fast_or_far_not_blocking | +0.165 |
| accelerate_when_clear | +0.161 |
| prefer_centre_on_tie | +0.081 |
| rear_safety | +0.078 |
| wait_if_boxed_in | +0.058 |
| ignore_rear_when_not_overtaking | +0.026 |
| spread_to_outer_when_clear | +0.023 |
| overtake_when_blocked | -0.348 |
| pick_faster_side | -0.422 |

## Best prompt — sequence (heuristics in order)

accelerate_when_clear -> overtake_when_blocked -> fast_or_far_not_blocking -> pick_faster_side -> avoid_slow_close_lane -> wait_if_boxed_in -> ignore_rear_when_not_overtaking -> rear_safety -> prefer_centre_on_tie -> spread_to_outer_when_clear

## Best system prompt

```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road right ahead allows. Follow these driving rules:
When driving, prioritize staying in your lane if open or when ahead is fast. For slow cars close ahead, consider moving to an adjacent lane to overtake instead of maintaining speed. Do not slow down for a car far away that's near top speed. When overtaking, choose lanes with more space or faster/longer leads. Avoid changing into lanes with slow cars behind you as it can block your path. If stuck in a blocked lane without better options, ease off and wait for a gap rather than forcing a risky lane change. Do not move aside for cars behind unless they are blocking your way. Finally, when two side lanes are equally good for overtaking, choose the one closer to the center of the road.
Think in one short sentence, then choose one action.
```
