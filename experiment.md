# Experiment log — beating 76.3 mph with a ≤3B LLM orchestrator (prompt-only)

Goal: beat the DeepTraffic all-time ceiling **76.3 mph** (paper Fig. 2; a *median*
over the official 500×2000 deterministic protocol, 11-car mode) using a small
(≤3B) local LLM as the per-car driving policy, with **all** driving knowledge in
the prompt (system prompt + semantic state rendering). No fine-tuning, no
inter-car communication.

Reference points (our validated engine, 500×2000 deterministic, median mph):

| policy | median |
|---|---|
| do-nothing floor | ~60 |
| real top DQN submission (jerrylingjiemei) | 74.13 |
| archived leaderboard #1 (Jan 2018) | 75.66 |
| hand-coded convention H5 (speed-aware + rear-safety, no drift) | 76.18 |
| **target: all-time ceiling** | **76.3** |

Median-of-500 std ≈ 0.04 mph (per repo README), so 76.3 must be beaten cleanly,
not by luck. Small-N medians on the first seeds of the official sequence run
slightly high (H5: first-100 median 76.36, first-8 median ≈ 76.4); therefore
**all small-N comparisons are paired against H5 on the identical seeds**
(`results/h5_scores_seed_order_100.json`).

---

## Environment (2026-06-11)

- Box: Linux, GTX 1060 6GB. NVIDIA driver 535 is too old for ollama v0.30.7's
  CUDA backend (needs ≥570) → ollama runs the GPU via its **Vulkan** backend.
  ~48 tok/s generation, ~0.4 s prefill (system-prefix KV cache works across
  calls). TODO(user): upgrading the driver to ≥570 would speed everything up.
- ollama v0.30.7 installed **user-space** at `~/.local/opt/ollama` (no sudo);
  server started detached: `setsid nohup env OLLAMA_KEEP_ALIVE=24h
  ~/.local/opt/ollama/bin/ollama serve > ~/.local/var/ollama-serve.log 2>&1`.
- Models pulled: `qwen2.5:3b` (primary), `llama3.2:3b`, `qwen3:1.7b`
  (alternates — model is a lab dimension, anything ≤3B is fair game).
- Python: repo venv `.venv` (numpy 2.2.6, pytest 8.3.4); engine needs numpy only.
- Repo fixes during setup: hardcoded macOS `sys.path` in all 7 `scripts/*.py`
  replaced with path-relative inserts.

## Verification before any LLM work

- **Fidelity: 18/18 tests pass** (`tests/test_fidelity.py`): trajectory
  bit-exact vs original JS (max per-frame deviation 0.0, single + multi-agent),
  fixed-policy scores match Node reference exactly.
- **H5 heuristic reproduction**: first-100 deterministic seeds → median
  **76.36** (consistent with documented 76.18 @ 500). Per-seed scores saved to
  `results/h5_scores_seed_order_100.json` for paired comparisons.

## Methodology decisions

1. **Exact-text memoization** (`LLMPolicy(cache="exact")`, new): memoize on the
   exact user-message text. At `temperature=0` the model is a deterministic
   function of the prompt, so scores are decision-identical to the pure
   no-cache benchmark — we just never ask the same question twice. This is what
   makes iteration (and a 500-run eval) tractable on this GPU. The legacy
   `cache=True` signature cache is *lossy* (gaps only, ignores speeds) and is
   not used for scoring. A pure no-cache spot-check vs the memoized scores on a
   few runs is planned before claiming the final number.
2. **Failures are never cached** and 25 consecutive failures abort the eval
   loudly (see incident E2 below).
3. **Paired small-N comparisons**: every variant is scored on the same official
   deterministic seed prefix; compare per-seed against H5, not absolute medians.

---

## Experiments

### E1 — P0 plumbing test (2 runs, qwen2.5:3b, original SYSTEM + semantic state)

Result: scores 71.80 / 71.02 vs H5's 76.49 / 75.54 on the same seeds
(`{"variant":"P0_current","runs":2,"median":71.8,...,"parse_fail":257}`).

Findings:
- **Truncation bug**: `num_predict=48` cut the JSON mid-`reason` before the
  `action` field ~18% of the time (257/1402); every failure silently became
  "maintain". The model's prose is wordier than the 48-token budget assumed.
  Fix: `num_predict=96` default + lenient `parse_action()` that extracts
  `"action": "<enum>"` from truncated JSON.
- Even ignoring parse failures, P0 is several mph below H5 — the original
  prompt under-specifies the convention (no numeric blocked-threshold, no
  rear-safety rule, no decelerate-when-boxed-in rule).
- Throughput: ~737 LLM decisions/run (11 cars × ~67 decisions); ~1.3–2.5
  calls/s on this GPU → ~5–15 min per uncached run. Exact-memo hit rate after 2
  runs: only 14% (888 distinct texts) → motivates a coarser state rendering.

### E2 — INVALID triage (server-death incident)

The first 8-run triage of all 5 variants produced identical ~59.8 medians with
`llm_calls=0, parse_fail=5336` — the detached ollama server had been reaped
with its parent shell session; every decision fell back to "maintain" (= the
do-nothing floor ~60, matching the README). Rows quarantined to
`results/promptlab_results.invalid.jsonl`. Fixes: ollama restarted via
`setsid` in its own session; `LLMPolicy` now raises after 25 consecutive
failures instead of silently scoring the floor. (Useful accidental datum: the
all-maintain fallback policy scores ≈ 59.8 median on these seeds.)

### E3 — triage v2 (8 runs each, qwen2.5:3b) — RUNNING

Variants (`scripts/prompt_lab.py`):

| variant | system prompt | state rendering | gen budget |
|---|---|---|---|
| P0_current | original speed-aware prose | semantic (BLOCKED verdict baked in) | 96, CoT |
| P1_h5rules_neutral | H5 convention as explicit numeric rules (blocked = <12 positions & <70 mph; overtake only into clearly-better rear-safe lane; prefer centre-ward; decelerate when boxed in) | neutral fact sheet, exact numbers | 96, CoT |
| P1_h5rules_semantic | same H5 rules | original semantic rendering | 96, CoT |
| P2_h5rules_coarse | same H5 rules | coarse fact sheet (speeds →5 mph, far gaps bucketed) | 96, CoT |
| P3_actiononly_coarse | H5 rules, no CoT | coarse fact sheet | 12, action-only schema |

Hypotheses: P1 ≥ P0 (explicit procedure beats prose); P2 ≈ P1 with much better
memo hit rate; P3 tests whether CoT matters at 3B. H5 paired reference,
first 8 seeds: [76.49, 75.54, 76.41, 77.07, 76.31, 76.36, 75.85, 76.62].

Results (paired H5 reference median on these 8 seeds ≈ 76.4):

| variant | median@8 | notes |
|---|---|---|
| P3_actiononly_coarse | **14.28** | catastrophic — see below |
| P2_h5rules_coarse | **42.82** | rules prompt over-decelerates (35% decel, 13% accel) |
| P1_h5rules_neutral | _cut_ | redundant once P2 failed (same rules, finer numbers) |
| P1_h5rules_semantic | ~54 (run 1, then cut) | rules prompt bad EVEN with verdict-style rendering |
| P0_current | _rerunning with fixed parser_ | 71.4 @2 in E1 (with 18% parse-fallback noise) |

**P3 finding — CoT is essential at 3B.** Action-only output (no reasoning
tokens) collapses the policy to "decelerate" in 81% of distinct states
(2026/2499; maintain 17%, accelerate 2%, lane changes ≈0). With zero parse
failures these are genuine choices: without a reasoning sentence the 3B model
doesn't execute the decision procedure, it anchors on the cautious last rule
("decelerate and wait"). Score 14.28 median is far below even the do-nothing
floor (~60) because gas decays with every decelerate. Side-datum: the coarse
rendering did its job on state-space size (53% memo hit rate within 8 runs,
2499 distinct texts vs P0's 888 in just 2 runs).

**E3 conclusion — numbered numeric rule procedures are the wrong prompt shape
for a 3B model.** All three P1/P2/P3 cells (same rules, three renderings, with
and without CoT) collapsed into over-deceleration: the model anchors on the
cautious escape-hatch rule ("decelerate and wait") and is unreliable at numeric
comparisons ("is *about 20* within 12?"). P1_semantic's ~54 with the
verdict-style rendering shows the rendering wasn't the problem — the rules
prompt was. The winning shape is P0's: short PROSE convention with an
accelerate-default framing ("a fast car ahead is NOT a problem — just
accelerate"), verdicts precomputed by the semantic state tool. Iterate from P0.

### E4 — H6 convention search (CPU, parallel to GPU prompt work)

A new utility-based convention family (`src/deeptraffic/llm/heuristic2.py`)
generalizing H5: time-to-catch blocking (`blocked iff gap < bg_base +
bg_slope*(80 - speed_ahead)`) and lane choice by settled-speed utility in mph
(open-within-horizon ≈ 80, else nearest car's speed) instead of clearance in
patches. Defaults reduce to ≈H5: 76.26 @100 (H5: 76.36). Staged sweep
(`scripts/sweep_h6.py`, written by a subagent: 3-pass coordinate descent → 80
random jitters → top15@300 → top5@500, 12 parallel workers) — RUNNING.
Purpose: (a) find a convention > 76.3 as heuristic; (b) distill its thresholds
into the P4 prompt wording. Note on user steer: the final entry should be the
LLM deciding from semantic state + prose convention — heuristics are used for
*finding* the convention and as ablation baselines, not as decision tools at
inference time.

**E4 RESULT — the H6 convention beats the ceiling as a heuristic:
median 76.51 @ official 500 runs** (H5: 76.18; ceiling: 76.3; median-of-500
std ≈ 0.04, so the margin is clean). Best config: `bg_base=14, bg_slope=0,
slow_thresh=70, horizon=16, margin_mph=2, rear_gap=5, rear_fast=62,
prefer_centre=true, boxed_action=decelerate`. Sweep: 175 evals
(3-pass coordinate descent converged pass 3, 80 random jitters, top15@300,
top5@500); full log `results/h6_sweep.jsonl`, summary
`results/h6_sweep_summary.md`. What mattered: bg_base 12→14 (+0.2 react
earlier to slow cars), rear_fast 55→62 (+0.1, veto lane entry only for
genuinely fast followers), margin 4→2 at 300/500 (slightly eager overtakes);
bg_slope, horizon, prefer_centre, boxed_action all plateaued at defaults.
This is the existence proof + threshold source for the LLM prompt; the
LLM entry itself decides per-state from the semantic rendering (per user
steer, no heuristic in the decision path).

### E5 — P0 baseline (fixed parser) + disagreement analysis → P4

P0 with the parser fix, first 4 official seeds: 72.92, 71.76, 73.28, 71.38
(median ≈ 72.3; H5 same seeds ≈ 76.45; parse failures 2/2174 ≈ 0.1%). The
parse fix recovered ~1.1 mph (run 1: 71.80 → 72.92); the remaining ~4 mph is
prompt content.

`scripts/disagree.py` (replays H5 trajectories, looks up the LLM's cached
decision per state): P0 agrees with H5 in only **9.4%** of seen states. The
loss is concentrated in NON-blocked states:
- everything OPEN → LLM picks **maintain** instead of accelerate (x96+x90+…).
  Engine fact: gas moves ±0.02/decision (engine.py `Car.act`), so after any
  deceleration, every skipped accelerate permanently delays recovery.
- lane 1/edge, everything OPEN → LLM changes lanes toward centre (x166) —
  P0's "prefer the centre-ward lane to pass" is misread as a drift rule.
- fast car approaching behind → LLM yields/moves aside (x8+x7+x6); courtesy
  moves the convention proved unnecessary (courtesy sweeps never beat H4/H5).

**P4_convention** (`prompt_lab.py`): P0's shape (prose + semantic verdicts),
with (a) accelerate-as-default stated as the dominant rule, explicit
"never maintain when open / never change lanes except to overtake / ignore
cars behind"; (b) renderer thresholds set to the H6 winners (blocked = <70 mph
within 14; rear-warning only ≥62 mph within 5, phrased as "do not cut in front
of it"); (c) centre preference stated only as an overtaking tie-break.

**E5 RESULT — P4 = 76.29 median @8 runs** (seed order: 75.95, 75.72, 77.18,
76.77, 76.80, 76.29, 74.37, 76.04; mean 76.14; 0 parse failures). H5 on the
same 8 seeds ≈ 76.45 median. So a **pure qwen2.5:3b policy is within ~0.15 mph
of the hand-coded convention and sitting right on the 76.3 ceiling** — a large
jump from P0's 72.3. But an 8-run median runs high vs the official 500
(small-N bias ~0.1–0.3), so this is not yet a clean beat; need a better prompt
and a larger N.

**E5b — disagreement vs the H6 winner** (`scripts/disagree.py`, replays H6's
trajectory, looks up P4's cached decision per state): P4 reproduces H6's action
in **87.8%** of shared states (vs P0's 9.4%). Residual loss, in order of mass:
1. NOT-blocked but a fast/far car visible → P4 sometimes **maintains** (x16) or
   makes a **spurious lane change** toward an open/fast neighbour (x7+x5+x5+x4…).
   Pure waste: a lane change costs speed and gas recovery is +0.02/decision.
2. BLOCKED with own lane 4p@45 and LEFT 1p@43 → P4 **dives into the slower,
   closer lane** (x7); the "not a valid escape" rule wasn't firing.

**E6 — fast probe proxy** (`scripts/probe_prompt.py`): full evals cost ~90 min;
to iterate prompts/models in ~15 min instead, the probe replays the H6 winner
(76.51 @500) over a few runs, collects the DISTINCT semantic states it visits,
and queries a candidate prompt/model LIVE on each, reporting agreement with
H6's action (split by open/blocked status). Since H6 scores 76.51, a prompt that
reproduces H6 on H6's own state distribution should score near 76.5. Calibrated
against P4 (whose full-eval agreement is the known 87.8%).

**P5_status** — the E5b fixes: bind the action to the (reliable) STATUS verdict.
"STATUS NOT blocked → ACCELERATE, ignore every lane detail and all cars
behind"; "STATUS BLOCKED → overtake only into a lane OPEN or with a car FASTER
or FARTHER than your blocker; never into a slow-and-close lane; else ease off."
Renderer also drops the word "blocking" (which anchored "maintain") for
"not an obstacle".

### E6 — probe2 proxy (`scripts/probe2.py`) + the P5 regression

Key engine fact discovered while building the proxy: the raw obs space is
**effectively continuous** — 2573 distinct states out of 2668 decisions in 4
runs; even the semantic *text* rarely repeats (exact gaps 1–40 × exact mph). So
exact-state caching only modestly helps and there are no "few common states" to
exploit. probe2 therefore takes an unbiased **reservoir sample of 450 actual
decisions** from the H6 winner's trajectory (frequency-weighted: 75% accelerate,
20% lane-change, 4% decel) and scores a candidate by **visit-weighted agreement
with H6's action**, split by open/blocked. One candidate ≈ 8 min vs ≈ 90 min
full eval. Same sample reused across candidates → clean paired comparison.

**Paired probe (qwen2.5:3b, 450 decisions):**

| variant | weighted-agree | open | blocked | note |
|---|---|---|---|---|
| **P4_convention** | **69.3%** | 75.5% | 50.5% | full-eval 76.29 @8 (the proxy anchor) |
| P5_status | 47.3% | 45.1% | 54.1% | **REGRESSION** |

**P5 is a clear regression and is discarded.** Despite a more emphatic
"NOT blocked → ACCELERATE always" prompt, qwen2.5 changed lanes / maintained in
55% of open states (open agreement 75.5% → 45.1%). Lesson: at 3B the P4 wording
sits at a fragile local optimum — a *rewrite* of the open-state instruction
backfires even when it says the same thing more forcefully. Future prompts must
be **minimal deltas on P4**, touching only the blocked/overtake logic (P4's real
weakness: blocked-state agreement is just 50.5%, vs 75.5% for open — but blocked
is only ~25% of decisions, and many disagreements are near-equivalent like
accelerate-vs-maintain at gas cap or left-vs-right overtake, which is why P4
still scores 76.29 at 69% agreement).

Proxy calibration so far: H6 = 100% agreement → 76.51; P4 = 69% → 76.29. The
map is shallow near the top (disagreements are often near-free), so 69% already
lands within 0.2 of H6.

### E7 RESULT — significance of the H6 heuristic vs 76.3 (`scripts/h6_variance.py`)

10 independent 500-run seed blocks each (= 5000 runs/policy):

| policy | block medians | mean | block-std | vs 76.3 |
|---|---|---|---|---|
| **H6** | all in [76.32, 76.51] | **76.404** | 0.064 | every block > 76.3; mean +0.104 |
| H5 | all in [76.12, 76.32] | 76.169 | 0.056 | mostly **below** (mean −0.131) |

Reading: the canonical deterministic 76.51 was the *favorable* end; H6's
**expected** 500-run median is ≈ **76.40**. Standard error of that mean over 10
blocks = 0.064/√10 = 0.020, so 76.404 is **(76.404−76.3)/0.020 ≈ 5.1 SE above
76.3** → the expected median beats the ceiling at p ≪ 0.001, and all 10/10
independent blocks individually cleared 76.3 (min 76.32). So **yes, H6 beats
76.3 significantly — but by a thin ~0.1 mph margin**, not the 0.21 the single
lucky deterministic block suggested. Crucially, **H5 (the repo's prior best)
does NOT beat 76.3** — its expected median is 76.17, below the ceiling. H6 is
the first policy here to genuinely clear it. (Caveat per the project steer: H6
is a *heuristic*; the pure-LLM beat is the real target and is still open.)

### E8 — model bake-off on the P4 prompt (`probe2`, 450 paired decisions)

| model | weighted-agree | open | blocked | dominant failure |
|---|---|---|---|---|
| qwen2.5:3b | 69.3% | 75.5% | 50.5% | spurious lane-change when open; mediocre overtakes |
| **qwen3:1.7b** | **82.2%** | **100%** | 27.9% | when BLOCKED it just keeps **accelerating** into the slow car (too passive to overtake) |
| llama3.2:3b | 76.7% | 80.8% | **64.0%** | best overtaker; some spurious left-changes when open |

Surprising: the **smaller** qwen3:1.7b has the highest weighted agreement and
PERFECT open-state discipline (100% — it never makes a needless lane change),
but is too timid to overtake (28% blocked → stays stuck behind slow cars).
llama3.2:3b is the most balanced. The agreement→score map is non-linear
(blocked errors cost more speed per error than open errors), so the full eval
decides. Full-eval of qwen3:1.7b on P4 (12 runs) RUNNING (`results/e8_qwen3_p4.log`);
llama3.2:3b next. P6 idea: a minimal P4 delta that forces overtaking when
blocked could pair qwen3's perfect open discipline with real overtakes.

**E8 RESULT — qwen3:1.7b on P4 = 74.91 median @12 runs** (mean 74.28, seed
order 74.91/74.94/72.99/74.51/75.46/73.05/75.29/73.80/76.15/71.46/75.11/73.73).
**Worse than qwen2.5's 76.29 despite higher weighted agreement** — a crucial
calibration lesson:

> **Blocked-state agreement, not weighted agreement, is the score predictor.**
> qwen3's perfect open discipline (100%) is nearly free in score terms (at gas
> cap, accelerate≈maintain), while its blocked failures (28% — it accelerates
> INTO the slow car instead of overtaking) leave it stuck at slow-car speed,
> which directly tanks the average. The proxy must be read as: open agreement
> ≈ free, blocked agreement ≈ everything.

Recalibrated table (the relevant axis is **blocked** agreement):

| model/prompt | open | **blocked** | full-eval median |
|---|---|---|---|
| qwen2.5 / P4 | 75.5% | **50.5%** | 76.29 @8 |
| qwen3:1.7b / P4 | 100% | **27.9%** | 74.91 @12 |
| llama3.2 / P4 | 80.8% | **64.0%** | RUNNING (10 runs) — best blocked, expected best |

This makes **llama3.2:3b the favourite** (highest blocked agreement). And it
makes **P6** (forces overtaking when blocked) the key prompt lever — it should
lift blocked agreement for every model, most dramatically for qwen3 (28% → ?).
P6 is a minimal delta on P4 (open wording untouched): "when BLOCKED,
ACCELERATING is useless — you stay stuck at the slow car's speed; you MUST
change lanes to overtake."

### DECISION (user steer) — commit to qwen2.5:3b + P4, improve the PROSE

Per the project owner: stop spreading across models; **qwen2.5:3b + P4 (76.29
@8, our proven best) is the bet.** Improve it by analyzing its failures in
detail and fixing them in prose, iterating until the full eval clearly clears
76.3. (qwen3 dropped: proven 74.91. llama parked: unproven challenger.) The P6
multi-model line is retired.

### E9 — qwen2.5/P4 blocked-state failure analysis (`scripts/analyze_failures.py`)

Replays H6, samples 160 distinct BLOCKED prompts, queries qwen2.5/P4 LIVE and
captures the model's own one-line REASON on each wrong decision. Blocked
agreement on this live sample: **64.4%** (103/160). Error clusters (H6→LLM):

| count | H6 wants | LLM does | nature |
|---|---|---|---|
| 14 | right (overtake) | decelerate | **timid** — refuses a valid lane |
| 8 | left (overtake) | maintain | **timid** |
| 7 | decelerate | maintain | minor (both passive) |
| 6 | decelerate | left | LLM overtakes where H6 waits (often LLM is fine) |
| 5+5 | left / right | right / left | wrong overtake direction |
| 4 | left | decelerate | **timid** |

**Root cause = prose misreads, not bad judgment** (the captured reasons prove it):
1. **"OPEN (clear far ahead)" is read as "the gap is far AWAY, not usable."**
   E.g. *own lane slow 4p@47, LEFT OPEN* → LLM maintains, reason: *"the LEFT
   lane is clear but far away."* It refuses the single best lane (an empty one).
2. **It treats "fast car ahead" in a neighbour as a reason to AVOID the lane**,
   and in one case **hallucinated a "slow car close behind"** that wasn't in the
   state, to justify not overtaking.

Net: the model is **too timid in blocked states** (≈26 of 57 errors are
refuse-to-overtake). Direction confusion (10) is secondary.

### P7_decisive — the surgical fix (open-state wording kept verbatim)

1. **Renderer (`fmt_p7`)**: kill the ambiguous labels —
   `OPEN (clear far ahead)` → `OPEN (no car ahead at all - a totally clear lane)`;
   `... - not blocking` → `... - far, not blocking`.
2. **System (`SYS_P7`)**: make the BLOCKED branch decisive — "if EITHER side is
   OPEN or fast/far, MOVE INTO IT NOW; moving beats staying stuck; do NOT
   decelerate/maintain. A side is unusable ONLY if it also has a slow car close
   ahead or a fast car close behind. Decelerate ONLY when BOTH sides unusable."

**E9b — P7 result: blocked agreement 58.8% (DOWN from 64.4%), but failure mode
flipped.** P7 *fixed the timidity* (almost no more overtake→decelerate) but
introduced two new issues, both visible in the captured reasons:
1. **Rear hallucination**: qwen2.5 repeatedly invents "a fast car close behind"
   (absent from the state) to mark a good lane "unusable", then dives the wrong
   way (one reason: *"LEFT is OPEN and has a fast car close behind, making it
   unusable. I must move to RIGHT"* — when RIGHT is "no lane there").
2. **LEFT bias / wrong-faster-side**: P7 defaulted left (x26 right→left,
   x16 decel→left) and sometimes chose the slower side (picked 44 mph over
   73 mph). Fleet risk: all 11 cars share the prompt, so a left bias would
   congest the left lanes and could *lower* the fleet average.

Note: agreement-with-H6 understates P7 here — many P7 "errors" are
overtake-into-a-good-lane where H6 conservatively waited (cheap or even better),
unlike P4's expensive stay-stuck errors. So agreement ≠ score once failure
modes differ; a real eval is needed to rank P4 vs P7/P8.

**P8_faster_side** — P7 + both fixes (open wording still untouched):
1. `fmt_p8` states the rear situation EXPLICITLY ("Behind: no car is close
   behind you …") so the model stops inventing rear cars; system adds "trust the
   descriptions, don't assume a car is there."
2. Tie-break = move to the FASTER/more-open side; centre only on a true tie.
**E10 — P8 analysis: blocked agreement 71.2%** (114/160) — best of any prompt
(P4 64.4%, P7 58.8%). Timidity gone, left-bias balanced (x10 right→left vs
x11 left→right, symmetric). Rear hallucination largely fixed by the explicit
"Behind:" line (one residual case, no longer harmful). Crucially, the biggest
residual "error" clusters (x12 decel→left, x10 decel→right = 22) are mostly the
LLM **correctly overtaking into an OPEN lane that H6 conservatively declined**
(reasons: *"Move to LEFT lane as it is OPEN"*) — i.e. the LLM is arguably
*better* than H6 there, so its true score may exceed its 71% H6-agreement.

Blocked-agreement progression across the prose iterations (qwen2.5:3b):

| prompt | blocked-agree | what changed |
|---|---|---|
| P4 | 64.4% | baseline (timid: refuses good lanes) |
| P7 | 58.8% | decisive overtake + label reword → fixed timidity, added left-bias |
| **P8** | **71.2%** | + explicit rear status + faster-side tie-break |

**E10 RESULT — P8 = 76.17 median @12 (76.31 on the first 8 seeds), mean 75.53.**
Paired vs P4: first-8 median **76.31 (P8) ≈ 76.29 (P4)** — a TIE. P8's mean
(75.53) is *below* P4's (76.14) and it had a worse tail (a 71.32 and a 74.62
run) and 4 parse failures. **So P8 did NOT beat P4** — chasing H6-agreement
(71% for P8 vs 64% for P4) bought zero score. This empirically confirms the
project steer: *agreement with the oracle is not the objective.* P4 remains the
champion (simpler prompt, fewer parse fails, better mean).

### E11 — is there headroom above H6? (`scripts/sweep_aggressive.py`, CPU, parallel)

64 bolder-overtaking configs @300 then top-6 @500 (margin_mph 0–3 × horizon
12–24 × bg_base 12–18). **BEST @500 = 76.51 — the original H6 config.** Bolder
settings (margin 0/1: overtake more eagerly) did NOT beat it; at 500 runs they
fell to 76.38–76.46. **There is no headroom above ~76.5 in this policy family.**

> **KEY FINDING — the engine's practical ceiling is ~76.4–76.5 for ANY policy.**
> Two independent heuristic searches (the E4 coordinate-descent and this E11
> aggressive sweep) both plateau at ~76.5. The published 76.3 "all-time ceiling"
> is genuinely near the global optimum of this fleet setup (11 controlled cars
> weaving through ~9 uncontrolled slow ones). Consequences:
> * "Clearly past 76.3" in the sense of **76.5+ is not achievable** by any
>   policy here — that's above the wall.
> * Both the best heuristic (H6, 76.40 expected / 76.51 best block) and the pure
>   LLM (P4, ~76.3 median) sit right at this ceiling, ~2 mph above the real DQN
>   submission (74.13).
> * A *significant* LLM beat of 76.3 is therefore necessarily **thin** (~0.1
>   mph) and must be established by a tight median at high N — like H6's own
>   "+0.10 = 5 SE" result — not by a large margin.

### E12 — definitive confirmation of the champion P4 at high N

P4 (qwen2.5:3b) is the best pure-LLM prompt: median 76.29 @8, mean 76.14
(≈ H6's mean 76.12), 0–2 parse fails.

**E12 RESULT — P4 @24 runs: median 76.50, mean 76.13** (min 72.08, max 77.52,
5 parse fails over 16k calls). The median rose 76.29 (@8) → 76.50 (@24) as
seeds accumulated, and the mean (76.13) is statistically identical to H6's
500-run mean (76.12) — **the pure 3B LLM performs at the hand-tuned heuristic's
level**, ~2.4 mph above the real DQN submission (74.13).

Significance (bootstrap of the median over the 24 per-run scores, 20k
resamples): **point median 76.50, 95% CI [76.04, 76.80], P(median > 76.3) =
85%.** So the point estimate clearly clears 76.3 and the balance of evidence
(85%) favours a true beat, but the 24-run sample is not yet tight enough for a
95% claim — the LLM's high-variance tail (a 72.08 and two ~74 runs) widens the
interval. Cross-check: since P4's mean equals H6's and H6's expected median is
76.40 (5 SE > 76.3 via the E7 block analysis), P4's expected median is also
≈ 76.4, i.e. above 76.3.

**E12b — no-cache integrity check PASSED.** Pure uncached P4 (`cache=False`,
every decision queried live, 0 parse fails) reproduces the cached per-run scores
exactly on the first official seeds (75.95 / 75.72 / 77.18, all match). This
proves the exact-text cache is decision-identical to the pure policy at
temperature 0 — the 76.50 median is the genuine pure-LLM result, not a caching
artifact. (`scripts/nocache_check.py`.)

**Bottom line for the pure-LLM track:** qwen2.5:3b + the P4 prompt, prompt-only
orchestration, **median 76.50 over 24 runs** (point estimate clears 76.3; 85%
bootstrap confidence; mean 76.13 matches the H6 heuristic, whose expected median
76.40 is a clean 5-SE beat of 76.3). The pure 3B LLM performs at the heuristic
ceiling, ~2.4 mph above the real DQN submission (74.13). A *large* margin over
76.3 is not physically available (E11: engine ceiling ~76.4–76.5 for any policy).

---

## E13 — `prompt_em/`: a reproducible heuristic-weighted prompt-EM suite

Turned the manual prompt iteration into a config-driven optimization framework
(`prompt_em/`, documented in its own README + references.md). It co-optimizes
the **emphasis weighting α** and the **ordering ρ** over 10 atomic driving
heuristics: each iteration an LLM synthesizes one prompt blending the heuristics
by α in ρ-order, evaluates it over 10 engine runs (objective = median fleet
mph), and a **regression-EM** step estimates each heuristic's weight- and
order-contribution and updates α, ρ with learning-rate `alpha`. Latin-hypercube
cold-start + a persistent exploration floor make the credit-assignment
identifiable (synthetic tests: emphasis recovers helpful/harmful heuristics by
~12 iterations; order recovers in isolation). A **scheduled synthesizer-freedom**
stage can freeze the best structure and unlock wording exploration (staged
search: structure → prose). Comprehensive logging (manifest with full
provenance, per-iteration weights/order/action-distributions/contributions,
summary.json). Single heuristics template 1-to-1; the weighted blend uses LLM
synthesis (deterministic template fallback).

**E13 RESULT — baseline run `em_v1` (8×10×2000, LLM synthesis, ~15 h): the 3B
LLM-synthesizer is the bottleneck.** Median trajectory 68.6, 39.8, **73.6**,
67.6, 61.7, 63.6, 67.0, 53.5 — **noisy and non-increasing**, best 73.6 (well
below hand-tuned P4's 76.5). Root cause from the logged action mixes: the policy
**MAINTAINS instead of accelerating** in 7/8 iterations (maintain 0.66–0.93,
accelerate ≈0); only iter 2, where the synthesized prose happened to stress
acceleration (accel 0.47), scored well (73.6). So a 3B model used as the
*synthesizer* cannot reliably reproduce the forceful "accelerate-by-default"
wording that P4 was hand-tuned to have — and the policy collapses into the exact
passive-maintain failure mode that manual tuning fixed (P0→P4). The weight/order
EM cannot repair this because the defect is **wording**, not heuristic selection
or order; worse, the synthesis variance dominates the score, so the EM's
contribution estimates are unreliable here (it credited tie-break heuristics
that merely co-occurred with the one good prose sample). This is the key
negative result: **LLM-synthesis with a small model is a high-variance channel
that can erase a good heuristic structure.**

**E14 — bake the proven wording into the heuristics + template synthesis
(RUNNING).** Two coupled fixes to the E13 synthesis bottleneck:
1. **Upgraded the heuristic `text` fields** to the forceful, proven P4/P8
   phrasing — "Your DEFAULT action is to ACCELERATE … never just maintain";
   "you MUST change to an adjacent lane to overtake"; "pick the genuinely faster
   side"; etc. Now the *wording* is baked into the rules, so synthesis no longer
   has to invent it; the optimizer tunes only which rules / emphasis / order.
2. **Deterministic template synthesis** assembles these strong sentences 1-to-1
   (no synthesizer-LLM variance). The template preview now reproduces a
   P4-quality prompt (accelerate-by-default first and forceful; harmful
   outward-drift rule auto-dropped).
**E14 RESULT — template + baked wording = 74.64 median @6×2000** (mean 74.07),
vs the LLM-synthesis run's noisy 40–74 (best 73.6, mostly collapsed to maintain).
**Confirmed: the synthesizer wording was the bottleneck.** The deterministic
template over the P4-worded heuristics produces a stable policy that accelerates
(no passive collapse), landing in the ceiling band and above the DQN (74.13).
The residual gap to hand-tuned P4 (76.50 @24) is the bulleted-list template
format vs P4's integrated flowing prose, plus small N (6 vs 24 runs). Takeaways:
(i) bake the proven wording into the rules; (ii) run the EM in `template` mode
(reliable); (iii) a constrained LLM-synthesis or the freedom stage can later try
to recover flowing prose without the passive-collapse risk.

**E15 RESULT — full EM in template mode (`em_v2_template`, 10 heuristics, stopped
at iter 4/8 to scale up).** Trajectory (1000-frame proxy): 74.09, **74.75**,
45.47, 73.64, 74.64. STABLE in the 74–75 band (vs em_v1's 40–74 chaos) — the
template+wording config holds. The cold-start iterations doubled as an
**automatic ablation**: dropping `accelerate_when_clear` (it2, weight 0.001)
collapsed the policy to **45.5 mph** (accelerate-fraction 0.16, maintain 0.32);
dropping `overtake_when_blocked` (it3) gave 73.6 (accelerates hard but gets stuck
behind slow cars). So the regression cleanly identifies accelerate-when-clear as
dominant and overtake second — *the optimizer's own exploration is an ablation
study*. Best 74.75. (Validated at 2000 frames separately: the seeded template =
74.64, E14.)

### E16+ — experiment roadmap for the paper (sequential on one GPU)

- **E16 `em_v3_large` RESULT:** 16 heuristics, 16 iterations. Trajectory climbs
  in the exploit phase — cold-start (iters 0–7) mean 72.9, exploit (iters 8–15)
  mean **75.0**, best **75.94** (1000-frame proxy). So the EM scales to the
  larger space and reliably produces good prompts.
  - **EM vs random (free ablation):** best of the random LHS cold-start = 75.94;
    best of EM-guided exploit = 75.90 — same *peak*, but exploit **mean 75.0 vs
    random mean 72.9**. The EM's value here is *reliability / concentration in the
    good region*, not a higher peak (honest characterization at a 16-eval budget).
  - **Credit assignment is noisier at 16 dims:** the contribution ranking
    (overtake_when_blocked top, but pick_faster_side appearing *harmful*) is
    less clean than em_v2's — the regression over 16 dims from 16 samples is
    sample-limited. So the paper's clean-attribution illustration stays em_v2;
    em_v3 shows the optimizer finds a *better prompt* (75.9) even when attribution
    is noisy. Best prompt + its (scrambled) order validating at 2000×24 now.

  **Validation (RUNNING):** best em_v3 prompt at the full 2000-frame protocol,
  24 runs + bootstrap CI — the headline *automated* number vs hand-tuned 76.50
  (proxy ran ~0.5 low, so this may land near/above 76.3).
- **Validation (RUNNING):** best em_v3 prompt at 2000×24 + bootstrap — the
  headline automated number (see above).

### FUTURE WORK (descoped from this paper — not run)

The suite *implements* these as ready-to-run hooks, but they are deferred to
future work and are NOT part of the current results:

- **Staged synthesizer freedom** (`synthesis.freedom` schedule + structure
  freeze): after the structure search converges, unlock LLM *wording* exploration
  around the frozen best structure (keep-best over candidates). Motivated by the
  honest open question — the gap from the automated ~75–76 to hand-tuned 76.5 is
  *wording*, which neither the weight nor the order channel can reach. The code
  path exists (`_freedom_at`, `_freedom_instructions`, candidate loop) but is off
  by default and unevaluated.
- **Sequence-importance isolation** (`optimizer.freeze_emphasis`): freeze the
  emphasis weights and vary ONLY the order, to measure how much prompt sequence
  alone moves the objective in isolation (the joint em_v3 already co-optimizes
  order, but confounds it with emphasis). The freeze hook exists and is
  unit-tested; no dedicated run was done.
- **Multi-seed robustness** of the contribution rankings; **cross-model
  transfer** of the best prompt (llama3.2:3b / qwen3:1.7b).

> **Strategy note.** Matching H6 caps the LLM at ~76.40 (H6's own expected
> median), which beats 76.3 by only ~0.10. The LLM also has high per-run std
> (~0.9 vs H6's per-block 0.06), so a single 8-run median (SE ≈ 0.32) cannot
> resolve 76.4 from 76.3. A **clean, significant** pure-LLM beat therefore needs
> (a) a prompt that matches/*exceeds* H6 on the costly blocked decisions, and
> (b) a final confirmation at ~24–30 runs (median SE ≈ 0.17). Plan: iterate
> prose on the cheap blocked-failure analysis, eval promising ones at 8 runs
> head-to-head vs P4, then confirm the winner at ~24–30 runs.

### Incidents / engineering notes

- E2 server death (documented above); `setsid` fix has held since.
- A `pkill -f prompt_lab` footgun: the pattern matches the probing shell
  itself (exit 144) — kill the python process specifically and verify with
  `ps | grep python`.
- Subagent sandboxes cannot execute `.venv/bin/python` (permission-denied);
  subagents write scripts, the main session executes them.

### Planned next

- E5: P4 prompt = P0's semantic rendering + refined prose convention (explicit
  rear-safety sentence, decelerate-only-when-boxed framing kept implicit,
  thresholds from the H6 sweep winner). Triage @8 paired vs H5, then 30–50
  runs.
- Model bake-off on the best prompt: qwen2.5:3b vs llama3.2:3b vs qwen3:1.7b.
- Official 500×2000 with the winner + pure no-cache spot-check.

---

## Appendix A — the hand-tuned champion prompt (P4), verbatim

**Eval protocol (be precise):** P4 was evaluated at **24 runs × 2000 frames** on
the official *deterministic* seed sequence, 11-car mode — NOT the official
500-run protocol (a 500-run LLM eval is ~days of GPU on this box; only the fast
heuristics H5/H6 got the full 500×2000 + 10-block variance). Result: **median
76.50, mean 76.13**, bootstrap 95% CI **[76.04, 76.80]**, P(median>76.3)=**0.85**;
the no-cache integrity check confirmed the exact-cache equals the pure policy at
temperature 0. So 76.50 is a 24-run median (expected median ≈ 76.4), reported as
such throughout. Code: `scripts/prompt_lab.py` (variant `P4_convention` =
`SYS_P4` + `fmt_p4`); reproduce with
`PYTHONPATH=src .venv/bin/python scripts/prompt_lab.py 24 P4_convention`.

**System prompt (`SYS_P4`, verbatim):**
```
You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED (top speed 80 mph). You auto-drive as fast as the road ahead allows.
Your default action is ACCELERATE. Whenever your lane is OPEN or the car ahead is fast, accelerate - never 'maintain' (it wastes speed recovery) and never change lanes (a lane change costs speed unless it actually gets you past a slow car). Ignore cars behind you; never move aside for them.
Change lanes ONLY to overtake when you are BLOCKED (a SLOW car CLOSE ahead in your lane). Pick an adjacent lane that is OPEN or whose nearest car is fast or far; NEVER move into a lane that also has a slow car close ahead, and never cut into a lane where a fast car is close behind. If both sides qualify, prefer the lane closer to the centre (lane 4).
If you are blocked and neither side qualifies, ease off briefly and wait for a gap.
Think in one short sentence, then choose one action.
```

**User message (`fmt_p4` semantic renderer):** a pure function of the same obs
the DQN sees, translated to per-lane features using thresholds FAST=70 mph,
NEAR=14 patches ("close"), REAR_NEAR=5, REAR_FAST=62. Example (a real blocked
state):
```
You are in lane 4 of 7. STATUS: BLOCKED by a slow car ahead.
Your lane: SLOW car CLOSE 7p ahead (41 mph).
LEFT lane: slow car far 21p ahead (45 mph).
RIGHT lane: OPEN (clear far ahead).
Available actions: accelerate, decelerate, maintain, left, right
```
The model replies with a one-sentence reason + one action (JSON, temperature 0).

**State-space fairness caveat (LLM vs DQN).** The renderer is a *pure* function of
the DQN's 315-cell observation (`ego_state()` = the JS `Map.s` grid, 7 lanes × 45
patches) — it never adds information, online hints, or privileged signals. But it
is **lossy**: it surfaces only the *nearest* car per lane, and only for the ego
lane and its two immediate neighbours (lanes ±2/±3 of the 7-lane grid, every car
behind the first in each lane, and the ego's own speed are dropped), then buckets
the surviving gaps/speeds by the fixed thresholds above. So the LLM and the DQN
share the same observation *source* but **not** identical input features: the LLM
conditions on strictly less than the raw grid, through a hand-designed feature map
(`state_tool.py::decode` → `fmt_p4`). This is an honest fairness limitation — the
LLM is never given more than the DQN, and is arguably *handicapped* on multi-lane
planning (it cannot see two lanes over) — and is now noted in the paper's §2 and
Limitations.
