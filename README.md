# DeepTraffic by LLM — a small-LLM driving policy + heuristic-guided prompt-EM

Two things live in this repo:

1. **The result.** A faithful, headless re-implementation of MIT's **DeepTraffic**
   (6.S094) benchmark, and the finding that a **small local LLM (qwen2.5:3b)**,
   used as the per-car driving policy with *all* knowledge in the prompt, reaches
   the crowd-sourced benchmark ceiling — a **76.50 mph** 24-run median, matching
   the best hand-tuned heuristic and beating the top real DQN submission (74.13)
   by ~2.4 mph.
2. **The method.** `prompt_em/` — a config-driven optimizer ("prompt-EM") that
   turns the manual prompt engineering into a reproducible loop: it co-optimizes
   the **weighting and ordering** of a small set of human driving **heuristics**,
   synthesized into one prompt, against the simulator's median speed, with a
   regression/EM credit-assignment step. There is a 1-page paper in `paper/`.

> **New here? Read this file, then [`experiment.md`](experiment.md)** (the full,
> chronological lab log with every result and why), then
> [`prompt_em/README.md`](prompt_em/README.md) (the suite). Handover notes and
> open threads are at the bottom of this file.

## Results at a glance

Official 11-car protocol, 500×2000 deterministic, score = **median fleet mph**.

| policy | median | notes |
|---|---|---|
| do-nothing floor | ~60 | |
| real top DQN submission (`jerrylingjiemei`) | 74.13 | the actual trained entry on our engine |
| archived leaderboard #1 (Jan 2018) | 75.66 | from Wayback |
| repo's prior-best heuristic (H5) | 76.17 expected | actually **below** the ceiling |
| **all-time ceiling** (paper Fig. 2) | **76.3** | the number to beat |
| our heuristic sweep (H6) | **76.40** expected | 5σ above 76.3 over 10 seed-blocks |
| **hand-tuned pure LLM (P4), qwen2.5:3b** | **76.50** @24 runs | point est.; P(med>76.3)=85%; no-cache verified |
| automated prompt-EM (em_v3, 16 heuristics) | 75.94 @proxy | best validated at 2000 frames — see experiment.md |

Two structural findings bound everything: (a) the engine's practical ceiling is
**~76.4–76.5 for *any* policy** (two independent sweeps plateau there), so 76.3
is near-optimal and margins are thin; (b) for a 3B model, *prose* conventions
work and numbered numeric rule-lists fail, and **wording is the largest lever**
(naive→tuned was 72→76.5 with the same rules).

## Repository layout

```
README.md            <- you are here (orientation + handover)
experiment.md        <- THE lab log: every experiment, result, and rationale
paper/paper.tex      <- 1-page NeurIPS-style extended abstract (self-contained)
src/deeptraffic/     <- the engine (frame-exact port of the original JS)
  engine.py env.py     faithful headless DeepTraffic; frozen Gym-like API
  llm/                 state_tool.py (obs->semantic state), llm_policy.py
                       (the qwen2.5:3b policy), heuristic.py / heuristic2.py
                       (hand-coded conventions = baselines/oracle)
reference/           <- original JS, Node ground-truth harness, paper PDF, notes
tests/test_fidelity  <- JS-vs-Python fidelity (frame-for-frame)
scripts/             <- the engine/LLM-track experiments (sweeps, evals, probes)
prompt_em/           <- the prompt-EM optimizer suite (its own README + paper refs)
  config/*.yaml         experiment + heuristic configs
  prompt_em/*.py        optimizer, synthesize, render, runner, experiment, cli
  validate.py           validate a prompt at the full 2000-frame protocol
  results/<name>/        manifest.json + iterations.jsonl + summary.json + report.md
results/             <- scratch outputs from the scripts/ track
```

## Setup

Engine + LLM eval need only **numpy**; the suite adds **pyyaml**.

```bash
# 1) Python venv (already present as .venv; to recreate):
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.2.6 pyyaml pytest tqdm

# 2) Local LLM via ollama (installed user-space at ~/.local/opt/ollama):
~/.local/opt/ollama/bin/ollama serve &            # start daemon (keep running)
~/.local/opt/ollama/bin/ollama pull qwen2.5:3b    # the driving model (~1.9 GB)

# 3) Node 18+ is only needed for the fidelity tests.
```

> **GPU note.** This box has a GTX 1060 6 GB. Its NVIDIA driver (535) is too old
> for ollama's CUDA backend (needs ≥570), so ollama runs the GPU via **Vulkan**
> (~48 tok/s). Upgrading the driver to ≥570 would speed up every LLM run
> substantially — the single biggest practical win for iteration speed.

## Reproduce the key results

```bash
# Engine fidelity (needs node):
PYTHONPATH=src .venv/bin/python -m pytest tests/test_fidelity.py -q

# Heuristic baselines / the H6 oracle sweep:
PYTHONPATH=src .venv/bin/python scripts/sweep_aggressive.py     # ceiling ~76.5
PYTHONPATH=src .venv/bin/python scripts/h6_variance.py 500 10   # H6 significance

# The hand-tuned pure-LLM policy (P4 = 76.50). Needs `ollama serve` + qwen2.5:3b:
PYTHONPATH=src .venv/bin/python scripts/prompt_lab.py 24 P4_convention
PYTHONPATH=src .venv/bin/python scripts/nocache_check.py 3      # cache==pure proof
```

## Run your own prompt-EM experiments

See [`prompt_em/README.md`](prompt_em/README.md) for the full method and options.
Quickstart:

```bash
cd prompt_em
# preview the prompt the heuristics+weights synthesize to (no GPU):
PYTHONPATH=../src:. ../.venv/bin/python -m prompt_em.cli synth --mode template
# run an experiment (edit config/default.yaml + config/heuristics.yaml first):
PYTHONPATH=../src:. ../.venv/bin/python -m prompt_em.cli run --config config/default.yaml
# validate the winning prompt at the full 2000-frame protocol:
PYTHONPATH=../src:. ../.venv/bin/python validate.py results/my_run/... 24 2000
```

To experiment: edit **`config/heuristics.yaml`** (add/edit/weight the atomic
rules — keep the forceful wording; that is what works) and **`config/default.yaml`**
(iterations, runs, synthesis mode, optimizer). Every run writes a reproducibility
`manifest.json`, a per-iteration `iterations.jsonl`, a `summary.json` for
plotting, and a human `report.md`.

## Handover notes / status

**Done & solid:** the engine (frame-exact, 18/18 tests), all baselines, the
hand-tuned 76.50 pure-LLM result (no-cache verified), the H6 oracle + ceiling
finding, the prompt-EM suite (unit-tested optimizer, full logging), and the
experiment arc em_v1 (LLM-synthesis collapses) → diagnostic (template + baked
wording = 74.64) → em_v2 (clean automatic ablation) → em_v3 (large 16-heuristic
space, exploit climbs to 75.94).

**In flight (as of this commit):** the best em_v3 prompt is being validated at
the full 2000-frame protocol (24 runs + bootstrap CI) — the headline *automated*
number. It takes ~2.5–4 h on this GPU (24 same-prompt runs; the exact-state cache
warms as it goes, so later runs speed up). The paper currently cites the
1000-frame **proxy** median (75.9, clearly labeled); the validated 2000-frame
median + CI will be filled into `paper.tex` and this README in a follow-up
commit. To run/repeat it yourself:
```bash
cd prompt_em
# best prompt text lives in results/em_v3_large/summary.json -> best_system_prompt
PYTHONPATH=../src:. ../.venv/bin/python validate.py <best_prompt_file> 24 2000
```

**Known gotchas:**
- Each full-protocol eval is ~2 h on this GPU; the suite uses a 1000-frame proxy
  for the loop and validates the winner at 2000. The driver upgrade (above) is
  the main speedup.
- A 3B model is a **weak prompt *synthesizer*** (em_v1): keep the proven wording
  baked into `heuristics.yaml` and use `mode: template`, or constrain LLM
  synthesis to preserve the "accelerate-by-default" imperative.
- Credit-assignment over many heuristics from few evals is sample-limited; em_v2
  (10 rules) gives the cleanest attribution, em_v3 (16) is noisier.

**Future work (descoped; hooks exist in the code, unevaluated):** staged
synthesizer *freedom* to explore wording around a frozen structure (the gap from
~75.9 to 76.5 is wording); sequence-importance *isolation* (`freeze_emphasis`);
multi-seed robustness; cross-model transfer. See the "FUTURE WORK" section of
`experiment.md`.

**Git:** the new files (`experiment.md`, `paper/`, `prompt_em/`, the new
`scripts/*.py`) are untracked and a few `scripts/*.py` are modified (macOS paths
fixed). Nothing has been committed — review and commit as you see fit.
