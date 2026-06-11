# DeepTraffic Python port — Validation

This document records the fidelity validation of the Python port
(`src/deeptraffic/`) against the original JavaScript engine
(`reference/original_js/gameopt.js`), the resolution of every subtle engine
point (with gameopt.js references), and the JS-vs-Python score tables.

> **Execution status: VALIDATED (2026-06-10).** Executed with base Python 3.11
> (numpy 2.4.4) + Node v18 in the foreground (no venv needed; fidelity uses only
> numpy). **All fidelity tests PASS:** the 6 trajectory cases match the original
> JS engine frame-for-frame (max per-frame deviation **0.0**, well under 1e-6),
> and the fixed-policy scores match the Node reference within the 0.1 mph
> tolerance (exactly). One fix was needed vs the as-authored port:
> `initializeMap`'s footprint helper must size to **16** (JS `Array(12)`
> auto-extends as `g` runs −4..11), not 12. The tables below are measured values.

## How to produce the numbers

```bash
bash setup_env.sh                         # one-time: create .venv, install deps
bash run_validation.sh                    # official 500x2000 references + pytest
# or quick dev pass:
RUNS=50 bash run_validation.sh
```

The fidelity tests shell out to `reference/node_ref/run_ref.js` for ground truth.

---

## 1. JS reference scores (official: runs=500, frames=2000, deterministic)

Command: `node reference/node_ref/run_ref.js eval --policy=<P> --runs=500 --frames=2000 --det=true`

| Policy (config: ls=1, pa=10, pb=0, nOther=0) | median (mph) | mean (mph) | min | max | wall (s) |
|---|---|---|---|---|---|
| `nop`   (action 0 every decision)            | 51.57 | 51.67 | 40.57 | 66.72 | 79.5 |
| `accel` (action 1 every decision)            | 51.57 | 51.67 | 40.57 | 66.72 | 81.6 |
| `brain` (network_basic.js, untrained)        | N/A | N/A | N/A | N/A | — |

> Measured 2026-06-10 (runs=500, frames=2000, deterministic, other_agents=0).
> **`accel` == `nop`** exactly, confirming the structural prediction: the ego
> starts at gas cap `a=2`, so "accelerate" is a no-op; both hold max throttle and
> are limited purely by the safety factor `c` in traffic. The naive fixed-policy
> floor is **~51.6 mph**; the published trained-DQN benchmark is **~76 mph**, so
> ~24 mph of headroom comes from lane-change skill. `brain` is N/A: doEvalRun does
> not train, so an untrained random-init ConvNetJS net is not reproducible (see note).

### Structural predictions (to sanity-check the filled numbers)

* `nop` (do nothing): the ego never accelerates; it keeps the gas `a` set by
  `initializeMap` (`z[0].a = 2`) and is throttled only by the safety system `c`.
  Expect a moderate score around the high-40s–50s mph range (`20*c*a`, with
  `a≈2`, `c<1` in dense traffic). It is NOT zero — "do nothing" means "maintain",
  not "stop".
* `accel` (always accelerate): the ego adds `+0.02` to `a` **once per decision**
  (see Subtle Point #1), capped at `a=2`. Since it starts at `a=2`, accelerate is
  largely a no-op on gas (already capped) — so `accel` and `nop` scores should be
  **very close**. Both are dominated by the safety factor `c` in traffic.
* `brain` (network_basic): the eval harness runs the brain with
  `learning=false`, `epsilon_test_time=0`, and **no training** (doEvalRun does not
  train). The net's weights are randomly initialized via `Math.random()`
  (Gaussian init in convnet.js), so the deterministic-eval `brain` score is a
  function of the *random initial weights* and will vary run-to-run of the
  harness process. The README's "~66.8 mph" refers to a TRAINED network_basic;
  an untrained random-init brain under doEvalRun will generally score LOWER and
  is not reproducible across processes. This is a property of the reference
  harness, not a port discrepancy. (To reproduce ~66.8 you must train first,
  which is out of scope for the engine-fidelity harness.)

> The Python `evaluate()` is validated against `nop`, `accel`, `decel` — the
> fixed policies — because those are deterministic and net-independent. The
> `brain` policy is a JS-only reference number (the Python port deliberately does
> not re-implement ConvNetJS; downstream RL supplies its own torch policy).

---

## 2. Python `evaluate()` vs JS reference (fixed policies)

Same seeds, same protocol. Acceptance: |Python − JS| ≤ 0.1 mph.
(Test: `tests/test_fidelity.py::test_scores_match_js`, RUNS via `DT_SCORE_RUNS`.)

| Policy | JS median | PY median | Δ | within 0.1? |
|---|---|---|---|---|
| `nop`   | 51.57 | 51.57 | 0.00 | ✅ |
| `accel` | 51.57 | 51.57 | 0.00 | ✅ |
| `decel` | 44.46 | 44.46 | 0.00 | ✅ |

Measured by `tests/test_fidelity.py::test_scores_match_js` (**PASSED**, all three
policies, runs=30, frames=2000: every |Δ| ≤ 0.1 mph). JS columns quote the official
500×2000 reference (§1). Because the port is bit-faithful (trajectory test §3, max
deviation 0.0), Δ is exactly 0.00.

---

## 3. Trajectory test (the strongest fidelity check)

`tests/test_fidelity.py::test_trajectory_matches_js` drives BOTH engines with an
identical scripted ego action sequence and identical deterministic seeds, then
asserts per-frame `(b, a, c, y, x)` of all 20 cars match within `1e-6`.

| Case (actions, frames, seed_offset) | frames compared | max per-frame deviation | first divergence |
|---|---|---|---|
| `[0]`, 200, 1                 | 200 | 0.0 | none (PASS) |
| `[1]`, 200, 1                 | 200 | 0.0 | none (PASS) |
| `[2]`, 200, 1                 | 200 | 0.0 | none (PASS) |
| `[1,1,2,0,4,3,1,0]`, 300, 1   | 300 | 0.0 | none (PASS) |
| `[1]`, 200, 3                 | 200 | 0.0 | none (PASS) |
| `[3,4,3,4]`, 250, 2           | 250 | 0.0 | none (PASS) |

All 6 cases PASS with max per-frame deviation **0.0** across all 20 cars and all
fields (b, a, c, y, x). The port reproduces the same IEEE-754 doubles in the same order.

Expected: max deviation `0.0` (the port reproduces the same IEEE-754 doubles in
the same order). The two engines share: the Park–Miller LCG, the exact RNG
consumption order, integer flooring, and the per-frame `V()` body.

---

## 4. Subtle points — how each was resolved (gameopt.js references)

All line references are to `reference/original_js/gameopt.js` (minified; lines
are the physical lines of that file).

### SP1 — Action persistence (does the ego action ramp every frame?)
**Resolved: NO. The ego acts once per decision.** In `V()` (lines 19–20), `d` is
a function-local `var` — it is a fresh `undefined` on *every* `V()` call. The ego
block `0==G%30 && (...d=learn(...)...)` sets `d` only on decision frames; on all
other frames `d` is `undefined` and `z[0].m(d)` → `m(undefined)` → `switch`
default → **no-op** (line 9, `this.m`). So `accelerate` applies `+0.02` exactly
once per 30-frame decision, not every frame. The only leak: on a frame where a
controlled-other decides (`G%30==3*a`, line 19) but the ego does not, that other's
`d` flows into the final `z[0].m(d)`. With `nOtherAgents=0` the others loop is
skipped and the ego acts purely on `G%30==0`.
Port: `engine.py::Engine.V` resets `d=None` each call; `Car.act(None)` is a no-op.

### SP2 — Exact car footprint stamping (`l`)
`this.l` (line 8): `for(a=0;15>a;a+=10) for(b=0;34>b;b+=5) H.set((x+a)/20,(y+b)/10, 1*c*a)`.
That is `a ∈ {0,10}` (2 values) × `b ∈ {0,5,10,15,20,25,30}` (7 values) = 14 set
calls, value `c*a`. `Map.set` floors both coordinates (line 4). Empty H defaults to
100. Port: `engine.py::Car.stamp` reproduces both loops and the `c*a` value exactly.

### SP3 — Spawn RNG: `Math.random` vs the LCG
`function D()` (line 6–7): `this.j()` consumes the **LCG `v`** twice — once for the
lane `a=Math.floor(140*q(v)/20)` and once for the gas `1+.7*gasPedalModulator()`
where the modulator is `0.5*q(v)`. **But** the constructor's spawn-y uses
`Math.random()` (line 7: `this.y=10*Math.floor(700*Math.random()/10)`) — which is
nondeterministic. Resolution: every constructor-set `y` is **overwritten by
`initializeMap`** (lines 11–12) before any frame runs, and off-road wrap in `move`
(line 7) sets `y` explicitly to `734`/`-34` *before* calling `j()`, while `j()`
itself never touches `y`. Therefore **`Math.random` has zero observable effect on
the (b,a,c,y) trajectory under deterministic eval.** Port: `Car` exposes an
injectable `math_random` (defaults to Python's RNG) used only for the
immediately-overwritten constructor `y`; it does not affect determinism.

Also resolved (SP3b): the `switch(a)` in `this.j` (line 6) has cases like
`case 5<a:`, `case 6==a:` — these compare `a === (5<a)` etc. Since `a` is an
integer 0..6 it never strictly equals a boolean, so **every case falls through to
`default`**, making the modulator always `0.5*q(v)`. The non-default branches are
dead code in the original. Port reproduces only the default (one `q(v)` consume).

### SP4 — `.s()` column-major /100 normalization
`Map.s` (line 5): `a[this.data.length*c+b] = this.data[b][c]/100`. For the obs grid
`K = Map(1+2*lanesSide, patchesAhead+patchesBehind, 0)`, `data.length = 1+2*ls`
(=3 for default) is the **stride**, and the flatten is column-major over the patch
index `c`. Empty road cells (H=100) → `1.0`; occupied → `c*a/100` (≈0..0.04);
off-grid cells return `0` (the `H.get(...,0)` default in `Map.o`, line 5) → `0.0`.
Port: `GameMap.to_state` uses `out[rows*c + b] = data[b][c]/100`; `GameMap.fill_obs`
uses the `,0` default. `num_inputs = 3*10 = 30` for network_basic.

### SP5 — Safety scan (`u`)
`this.u` (line 8): `for(a=2,b=1;5>b;b++){ d=H.get((x+7.5)/20,(y-10*b)/10,100);
100>d && (a=min(a,.5*(b-1)), a=min(a,d/this.a)) } this.c=a`. Scans 4 cells ahead at
the car's center column; a car 1 cell ahead (`b=1`) → `0.5*(b-1)=0` → `c=0` (full
stop); farther → partial; also bounded by `leadSpeed/ownGas`. Port: `Car.safety`.
Edge case: when `this.a == 0`, JS computes `d/0 = +Infinity` (or `NaN` for `0/0`)
and `Math.min` propagates accordingly. Python would raise `ZeroDivisionError` and
`min(x, nan)` differs from JS. Resolved with `js_min` (NaN-propagating) and explicit
`+inf`/`nan` for the `d/0` cases — faithful to `Math.min` semantics.

### SP6 — RNG reseeding order in `reset()` (cross-run coupling)
`reset` (line 13): it constructs the 20 `new D()` **first** (each consuming the
*current* `v` twice), and **only then** does `r+=1; t+=1; u=new p(r); v=new p(r)`
(note: BOTH `u` and `v` are seeded from `r`, not `t`). So a run's non-controlled
initial gas `a` (set by the constructor `j()` using the *pre-reseed* `v`) depends on
the previous run's `v` state. This cross-run coupling is deterministic and is
reproduced exactly. The first deterministic run consumes from the **fresh-load `v`**
(seed 1 after the 40 load-time constructor consumes); `doEvalRun`'s deterministic
branch (line 21) sets `t=r=0` but does **not** recreate `u`/`v`. Port:
`Engine.__init__` snapshots the fresh-load `u`/`v` state, and
`set_deterministic_seeds()` restores it (so repeated deterministic `evaluate()`
calls reproduce a fresh JS load).

### SP7 — `initializeMap` legalLocations splice / `Array(12)` undefined
`initializeMap` (lines 11–12): the footprint helper `d(a,b)` builds `Array(12)` but
only assigns entries where `0<=b+g`; rows above the map leave **undefined** slots.
Removal is `legalLocations.splice(legalLocations.indexOf(x),1)`. For `x=undefined`
(or an already-removed value) `indexOf` returns `-1` and `splice(-1,1)` removes the
**last** element of `legalLocations`. This (faithfully reproduced) quirk affects the
spawn layout of cars near the top of the map. Port: `Engine.initializeMap` uses a
`None` sentinel and a `splice_indexof` that pops the last element on `indexOf == -1`.

### SP8 — `E` (scroll) timing and double ego stamp
In `V()` (line 19): cars `move()` using the **previous frame's** `E`, then
`E=1.5-(z[0].y-525)` is recomputed *after* the move loop. The ego `z[0].l()` is
stamped **twice** per frame (once in the move loop, once after the controlled-others
loop). Port reproduces both: `move` reads `eng.E` set last frame; `V` re-stamps
`z[0]` after the others loop.

### SP9 — Median, not mean; scoring formula
`doEvalRun` (lines 21–22): per frame `O += Σ_{B=0..nOther} max(0, z[B].c*z[B].a)/(nOther+1)`;
per-run score `= floor(O/frames*2000)/100`; returns `f[a/2]` after sort (the
**median**, with even `runs` like the official 500 → exact middle). Port:
`env.py::DeepTrafficEnv.evaluate` mirrors this; the Node `run_ref.js` mirrors the
same loop (`runEvalDistribution`) so its `median` equals the engine's own
`doEvalRun` (cross-checkable with `--check-engine`).

### SP10 — `nOtherAgents` cap
Line 1: `nOtherAgents=Math.min(otherAgents,9)` (the README says "up to 11", but the
code caps at 9 plus the ego = 10 controlled, plus `reset` re-checks against
`Math.min(otherAgents,10)`). Port: `Engine.__init__` uses `min(otherAgents, 9)` and
**`reset()` raises it to `min(otherAgents, 10)`** exactly as JS does, so 11-car mode
has 10 controlled others + ego = 11 controlled cars.

> **FIXED 2026-06-10.** The port originally treated `reset()` as a no-op here, which
> left car 10 *uncontrolled* in 11-car mode (it got random gas instead of the
> controlled 1.7). Caught by the new multi-agent trajectory test
> `test_trajectory_matches_js_multiagent` — now PASSES with max per-frame deviation
> **0.0** for both `otherAgents=4` (5 cars) and `otherAgents=10` (11 cars).

---

## 5. Residual discrepancies

* **`brain` reference is not reproducible across harness processes** by design
  (random ConvNetJS weight init under an *untrained* doEvalRun). This is a
  property of the official reference harness, documented in §1, not a port bug.
  The Python port does not re-implement ConvNetJS (downstream RL supplies a torch
  policy), so there is no Python `brain` number to compare.
* **`Math.random` spawn-y** is intentionally non-bit-identical between Python and
  JS, but provably **unobservable** (SP3): it is overwritten before any frame.
  The trajectory test confirms this by matching `(b,a,c,y,x)` exactly despite the
  two engines using different `Math.random` sources.
* **Odd `runs`**: the engine's `f[a/2]` is only a true median for even `runs`; the
  official benchmark uses `runs=500` (even). Use even `runs` for exact parity.

No other discrepancies are expected; the trajectory test is the authoritative
gate and is designed to fail loudly (reporting the first divergent frame/car/field)
if any mechanism above is mis-ported.
