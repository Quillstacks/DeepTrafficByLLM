# prompt-EM report — diag_template

- objective: **median fleet mph** (target: beat 76.3 ceiling)
- iterations: 1 x 6 runs
- policy model: qwen2.5:3b; synthesis: template; co-optimising emphasis + sequence

## Trajectory (median & mean fleet mph per iteration)

| iter | median | mean | min | max | parse_fail |
|---|---|---|---|---|---|
| 0 | **74.64** | 74.07 | 70.94 | 75.85 | 5 |

**Best iteration: 0 — median 74.64, mean 74.07.**

## Best prompt — sequence (heuristics in order)

accelerate_when_clear -> overtake_when_blocked -> fast_or_far_not_blocking -> pick_faster_side -> avoid_slow_close_lane -> wait_if_boxed_in -> ignore_rear_when_not_overtaking -> rear_safety -> prefer_centre_on_tie -> spread_to_outer_when_clear

## Best system prompt

```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road right ahead allows. Follow these driving rules:
- (ALWAYS) Your DEFAULT action is to ACCELERATE: whenever your lane is open or the car ahead is fast, accelerate and stay in your lane -- never just maintain, because maintaining wastes the speed you could be regaining.
- (ALWAYS) When a SLOW car is close ahead in your lane you are stuck at its speed and accelerating is useless, so you MUST change to an adjacent lane to overtake instead of staying behind it.
- (ALWAYS) A car ahead that is near top speed or far away does NOT block you -- you could not go faster anyway, so do not slow down, maintain, or change lanes for it; keep accelerating.
- (Important) When overtaking, move to the side lane that is open, or whose nearest car is faster or farther ahead than the car blocking you -- pick the genuinely faster side.
- (Important) NEVER move into a lane that itself has a slow car close ahead -- that is not a real escape, it only trades one block for another.
- (Important) If you are blocked and neither side lane is genuinely better, ease off briefly and wait for a gap rather than forcing a bad lane change.
- (Important) When you are not overtaking, ignore the cars behind you completely and never move aside for them.
- (Important) Do not change into a lane where a fast car is close behind you; you would cut it off and lose speed.
- (Also) When two side lanes are equally good for overtaking, prefer the one toward the centre of the road.
Think in one short sentence, then choose one action.
```
