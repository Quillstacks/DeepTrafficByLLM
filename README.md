# MIT DeepTraffic — beating the benchmark with a small (3B) LLM orchestrator

A faithful, headless re-implementation of MIT's **DeepTraffic** (6.S094) competition,
plus an investigation into whether a **small local LLM (qwen2.5:3b)**, used as the
per-car driving policy, can beat the crowdsourced DQN benchmark.

## Status / results (11-car mode, our validated engine; 500×2000 deterministic, median mph)

| Policy | median mph | notes |
|---|---|---|
| do-nothing floor | ~60 | no lane changes |
| **DQN — real top submission** | **74.13** | the actual `jerrylingjiemei` 74.05 entry, run on our engine |
| archived leaderboard #1 (Jan 2018) | 75.66 | from Wayback |
| hand-coded convention (heuristic H5) | 76.18 | speed-aware overtake + rear-safety, no drift |
| **all-time ceiling** (paper Fig 2) | **76.3** | the number to beat (a *median*, not a lucky run; our median std ≈ 0.04) |
| 3B-LLM orchestrator | _measuring_ | pure per-decision qwen2.5:3b |

The engine is validated **frame-for-frame** against the original JavaScript (`reference/`),
and cross-checked against the real 74.05 submission's trained weights.

## Layout
- `src/deeptraffic/engine.py`, `env.py` — faithful headless port (frozen Gym-like API).
- `src/deeptraffic/llm/` — the LLM-orchestrator track: `state_tool.py` (decode obs →
  semantic state), `llm_policy.py` (qwen2.5:3b policy + speed-aware system prompt),
  `heuristic.py` (reference conventions, used only for analysis/baselines).
- `reference/` — original JS (`original_js/gameopt.js`), Node ground-truth harness,
  the paper PDF, the real 74.05 submission, `ENGINE_NOTES.md`, `VALIDATION.md`.
- `scripts/` — eval/sweep entry points.
- `tests/test_fidelity.py` — JS-vs-Python fidelity (single + multi-agent).

## Setup
Python (engine + LLM eval need only **numpy**; torch optional for the MBRL extra):
```bash
python3 -m pip install numpy
```
Local LLM via **ollama** (this is what a GPU box accelerates):
```bash
ollama serve &            # start the daemon
ollama pull qwen2.5:3b    # ~1.9 GB
```
Node 18+ (optional, only for the fidelity tests).

## Run
```bash
# Fidelity (needs node):         PYTHONPATH=src python3 -m pytest tests/test_fidelity.py -q
# Heuristic convention baseline: PYTHONPATH=src python3 scripts/bakeoff_conventions.py 500
# >>> THE LLM ORCHESTRATOR <<<   PYTHONPATH=src python3 scripts/eval_pure3b.py 500
```
`eval_pure3b.py` queries qwen2.5:3b for **every** decision (no cache) and prints per-run
progress. CPU: ~7-10 min/run; a GPU should be far faster — the reason this repo exists.

## The experiment
Each of 11 cars independently runs the *same* qwen2.5:3b policy (no inter-car
communication). The model receives a **semantic rendering of the same observation the
DQN sees** ("SLOW car close ahead", "fast car — not blocking", "approaching from
behind") plus a fixed **speed-aware driving convention** in the system prompt; it
reasons in one sentence and picks one of 5 actions. Question: can this beat the DQN
(74.13) and approach/break the 76.3 ceiling?

The driving knowledge lives entirely in `SYSTEM` and the semantic state tool — see
`src/deeptraffic/llm/llm_policy.py`. Iterate the prompt there.
