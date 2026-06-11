# DeepTraffic v2.0 engine — decoded mechanics (authoritative reference)

Source of truth: `reference/original_js/gameopt.js` (the full simulation, minified) and
`reference/original_js/eval_webworker.js` (the scoring protocol). The README in
`/tmp/deeptraffic_src` gives the high-level spec. Where this file and the minified
code disagree, **the code wins** — verify against a Node reference harness.

## Coordinate system
- 7 lanes, each 20 px wide. Lane index `b ∈ {0..6}`. Car x-pixel `= 20*b + 4`.
- Vertical "patches" 10 px each. Road map `H` is **7 cols × 70 rows** (700 px tall).
  Patch row `= floor(y/10)`.
- Cars are 15×34 px. Ego (`z[0]`) starts lane `b=3`, `y=525`, gas `a=2`.
- 20 cars total (`z[0..19]`). `z[0]` is ego. `z[1..nOtherAgents]` run the user policy
  (controlled). The rest are "game" cars with fixed random gas.

## RNG (deterministic — replicate exactly)
- `p(seed)`: `g = seed % 2147483647; if g<=0 g += 2147483646`.
- `next()`: `g = 16807*g % 2147483647`. `q(stream) = (next()-1)/2147483646` ∈ [0,1).
- Two streams: `u` (map init, seed `r`), `v` (car behavior, seed `t`). Start `r=t=1`;
  each `reset()` does `r+=1; t+=1; u=new p(r); v=new p(r)` (note: BOTH reseeded from r).
- **Deterministic eval** sets `r=t=0` and `brain.reset_seed(0)` before the runs.

## Car state (`D`)
- `x` (px), `y` (px), `b` (lane), `a` (gas ∈[0,2]), `c` (safety factor ∈[0,2]),
  `h` (60-length ring buffer of `c*a*20` for mph display), `f` (controlled? bool).
- **Speed in mph = `c*a*20`** (so `c*a ∈ [0,4]` maps to 0..80 mph).
- Spawn `j()`: lane `aRnd = floor(140*q(v)/20)` (0..6); sets a `gasPedalModulator`;
  if not controlled: `a = 1 + 0.7*gasPedalModulator()` (≈1.0..1.7 → 40..68 mph);
  `y = 10*floor(700*Math.random()/10)`. NOTE: spawn y uses `Math.random()` (nondeterministic!)
  but lane/gas use `q(v)`. Confirm behavior under eval in the Node harness.
- `move()`: advances y by global scroll `E`, handles passing counter `F`, wraps off-road
  cars (y<-68 → 734 & respawn; y>768 → -34 & respawn), eases x toward target lane.
- `l()`: stamps the car footprint into `H` with value `c*a` (its speed/20). Empty H = 100.
- `u()` (safety): `c` starts 2; scan 4 cells ahead (b=1..4) at own column; if cell occupied
  (`H<100`): `c = min(c, 0.5*(b-1))` and `c = min(c, leadVal/own_a)`. So a car 1 cell
  ahead → `c=0` (stop); farther → partial. This is the collision-avoidance override.
- `i(dir)`: attempt lane change; succeeds only if target lane cells are clear (gap check).
- `m(action)`: `1`→accelerate `a+=0.02` (cap 2); `2`→decelerate `a-=0.02` (floor 0);
  `3`→lane left `i(-1)`; `4`→lane right `i(+1)`; `0`/other→nothing.

## Observation (the policy input — "same I/O")
- Built by `H.o(carIdx, K)`: for `g ∈ [-lanesSide, lanesSide]`, `h ∈ [-patchesAhead, patchesBehind)`,
  `K[g+lanesSide][h+patchesAhead] = H.get(z[idx].b + g, floor(z[idx].y)/10 + h, 0)`.
- Grid shape: `(2*lanesSide+1) × (patchesAhead+patchesBehind)`.
- `.s()` flattens **column-major** and **divides by 100**: empty→1.0, occupied→`c*a/100` (≈0..0.04).
  This `.s()` vector is exactly what the net receives. (README's "80 / speed" is a simplification;
  the actual fed values are H/100.)
- `temporal_window` (ConvNetJS deepqlearn) stacks past states+actions; `network_basic` uses 0.

## Timing & control loop (`V()`, one call = one frame)
- Every frame: `H.reset()` → all cars `move()`+`l()` → recompute `E` from ego y →
  all cars `u()` (safety) → game cars random lane-change with prob `>.99+.004*c`.
- **Controlled others** `z[1..nOtherAgents]`: decide on frame `G%30 == 3*a` (staggered per agent).
- **Ego** `z[0]`: decide on `G%30 == 0`: build obs `K`, call `learn(K.s(), reward)`,
  `reward = (N-60)/20` where `N = Σ c*a` over the last 30 frames (reset after).
- IMPORTANT SUBTLETY: the chosen action `d` appears to persist (global) and `z[0].m(d)` is
  called **every frame**, so e.g. "accelerate" ramps gas +0.02 *per frame* until the next
  decision changes it. **Confirm exact persistence semantics in the Node harness** — it
  strongly affects dynamics.

## Scoring (`doEvalRun`, eval_webworker.js)
- Defaults: `runs=500`, `frames=2000`, `deterministic=true`.
- Per frame: `O += Σ_{B=0..nOtherAgents} max(0, z[B].c*z[B].a) / (nOtherAgents+1)`.
- Per run score `= floor(O/frames * 2000)/100  == 20*mean_frames(mean_controlled(c*a))` (≈ mph, 2-dp).
- **Final benchmark = MEDIAN over the 500 runs.** (Not mean.)
- `network_basic.js` (10-neuron FC, lanesSide=1, patchesAhead=10) → ~66.8 mph per README.

## Fidelity validation plan
1. Run ORIGINAL `gameopt.js` headless in Node (stub `self/window/document`, set
   `headless=true`, load convnetjs) → reference scores for: (a) `network_basic` brain,
   (b) fixed policy "always accelerate" (action 1), (c) "do nothing" (action 0).
2. Drive BOTH the JS engine and the Python port with an **identical scripted action sequence**
   (bypassing any net) and assert per-frame `(b, a, c, y)` of every car match — this isolates
   engine fidelity from the neural net.
3. With deterministic seeds the Python port should match the JS to ~floating point.
   Acceptance: |Python − JS| ≤ 0.1 mph on the three fixed policies.
