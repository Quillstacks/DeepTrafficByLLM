# prompt-EM — optimizing an LLM driving prompt as a weighted mix of heuristics

This suite turns the hand-tuned prompt engineering that reached the DeepTraffic
ceiling (~76.3–76.5 mph, see `../experiment.md`) into a **reproducible,
config-driven optimization loop**. You give it a set of atomic driving
**heuristics**; it searches for the **weighting AND ordering** of those
heuristics whose LLM-synthesized prompt maximizes the fleet's median speed —
i.e. it co-optimizes *what to say* (emphasis) and *in what sequence* (order),
and logs both so sequence importance can be analyzed.

It is an instance of automatic prompt optimization / LLM-as-optimizer applied to
an LLM-as-driving-policy — see `references.md` for the related literature and how
this differs from it (the short version: the search space is an interpretable
*weight simplex over human heuristics*, updated by a *regression/EM* step).

---

## The loop

```
        ┌──────────────────────────────────────────────────────────┐
        │  weights α  (one per heuristic, on the simplex)           │
        └──────────────────────────────────────────────────────────┘
                     │ synthesize (LLM or template)
                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │  one driving SYSTEM prompt blending the heuristics by α    │
        └──────────────────────────────────────────────────────────┘
                     │ evaluate: 10 engine runs, all cars run this prompt
                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │  objective = MEDIAN fleet mph   (+ mean, reported)         │
        └──────────────────────────────────────────────────────────┘
                     │ regression-EM credit assignment
                     ▼
        ┌──────────────────────────────────────────────────────────┐
        │  E: ridge-regress score on past α → each heuristic's       │
        │     marginal contribution (its "responsibility")           │
        │  M: α ← (1−alpha)·α_best + alpha·responsibility + noise     │
        └──────────────────────────────────────────────────────────┘
                     └────────── next iteration ─────────────────────┘
```

Every iteration logs the weights, the synthesized prompt, the **median and mean**
fleet mph, and the per-run scores. The final report shows the whole trajectory,
the best prompt found, and the regression's estimate of which heuristics help.

## Install / prerequisites

```bash
# from the repo root (DeepTrafficByLLM/)
.venv/bin/python -m pip install numpy pyyaml          # engine needs only numpy
ollama serve &                                        # local LLM daemon
ollama pull qwen2.5:3b                                # the driving + synth model
```
The DeepTraffic engine is imported from `../src` (the CLI adds it to the path).

## Usage

```bash
cd prompt_em

# preview the prompt the current heuristics+weights synthesize to (no eval):
PYTHONPATH=../src:. ../.venv/bin/python -m prompt_em.cli synth --mode template
PYTHONPATH=../src:. ../.venv/bin/python -m prompt_em.cli synth   # llm mode

# run the full experiment (writes results/<name>/):
PYTHONPATH=../src:. ../.venv/bin/python -m prompt_em.cli run --config config/default.yaml

# reprint a finished run's report:
PYTHONPATH=../src:. ../.venv/bin/python -m prompt_em.cli report --name em_v1
```

Outputs in `results/<name>/`:
- `manifest.json` — full provenance for reproducibility: the exact config, all
  heuristic texts + priors, seeds, the policy/synthesis models, engine config,
  and tool/library versions (python, numpy, pyyaml, ollama, GPU+driver) + the
  repo git commit and dirty flag.
- `iterations.jsonl` — one record per iteration with **everything**: the
  emphasis `weights`, the order `priority` and resulting `order`, `dropped`
  heuristics, optimizer `diagnostics` (phase, exploration scale, EM
  responsibilities), the objective + `median`/`mean`/`std`/`min`/`max`, the
  per-run `scores` (seed order), the policy's `action_fractions`/`action_counts`
  (behavioural signal), `parse_fail`/`llm_calls`/`cache_hits`, the
  `weight_contributions` and `order_contributions` (regression credit
  assignment), the full `system_prompt`, and timestamps.
- `summary.json` — machine-readable trajectories (median/mean/objective per
  iteration), best iteration, action-fraction trajectory, final contributions,
  costs, and reference ceilings — ready for plotting.
- `report.md` — human trajectory table (median & mean per iteration), best
  prompt + its sequence, and the emphasis and sequence contribution tables.

## Configuration

Two YAML files under `config/` — everything a human edits:

- **`heuristics.yaml`** — the atomic rules. Each has an `id`, a `text` (its 1-to-1
  prose form), and an `init_weight`. Add/remove/edit freely; the loop adapts.
- **`default.yaml`** — the experiment:
  - `experiment.iterations`, `runs_per_iter` (the "10 runs"), `frames`.
  - `objective: median` (the official benchmark; mean is always reported too).
  - `policy.model` — the LLM that drives.
  - `synthesis.mode` — `llm` (a model writes the blended prompt) or `template`
    (deterministic 1-to-1 assembly, no LLM, fully reproducible).
  - `optimizer.alpha` — the learning rate that folds the EM update into the next
    weights; plus `ridge_lambda`, `explore_std`/`explore_decay`,
    `cold_start_rounds`, `seed`.
  - `engine.*` — DeepTraffic config (defaults = the 11-car benchmark).

## 1-to-1 vs LLM synthesis (the design question)

A **single** heuristic → a prompt sentence is **1-to-1** and needs no LLM (that's
the `text` field, used verbatim by `template` mode). The part that benefits from
an LLM is **combining the weighted heuristics into one fluent prompt** — but even
that has a deterministic fallback. So:
- `synthesis.mode: template` — reproducible, no second model, slightly stilted.
- `synthesis.mode: llm` — fluent, but adds nondeterminism (mitigated by temp 0)
  and one extra model call per iteration. The `llm` path always falls back to the
  template if the synthesis call fails, so a run never dies on it.

The **state rendering** the driver sees each step (`render.py`) is held fixed
across the experiment — we optimize the *driving rules*, not the state encoding.

## How the EM credit-assignment works (and its limits)

Each round we fit a ridge regression of the objective on the weight vectors tried
so far; the coefficient for heuristic *i* estimates its marginal contribution.
The M-step moves the weights toward the positive-contribution heuristics (a
*mix*, not the single best — a one-rule prompt underperforms), blended with
`alpha` and anchored on the best weights seen so far, plus annealed exploration.

**Sample efficiency.** Attributing a fleet score to 10 heuristics from a handful
of combined evaluations is genuinely hard (the weight vectors are collinear on
the simplex). Two design choices address this: (1) the first `cold_start_rounds`
use a **Latin-hypercube** design that varies every heuristic independently, so
the regression is identifiable; (2) a **persistent exploration floor** keeps any
heuristic from being permanently excluded. In synthetic tests the emphasis
channel recovers the true helpful/harmful heuristics by ~10–12 iterations. With
fewer iterations the attribution is approximate — but the loop always keeps the
**best prompt found**, so the reported best never regresses. More iterations →
sharper attribution; each costs one 10-run evaluation (~30–60 min on a GTX 1060).

## Sequence (order) as a co-optimized dimension

LLMs — especially small ones — are sensitive to instruction *order* (primacy: a
3B model anchors on the first/most-salient rule). So the suite optimizes a
second channel, `priority`, with the same regression-EM: heuristics are placed
in the prompt sorted by descending priority, and the regression of the objective
on the priority vector estimates each heuristic's **order-contribution** (the
value of stating it *earlier*). Each iteration logs the `order`, the `priority`
vector, and the `order_contributions`, so you can analyze **sequence importance**
post-hoc — or isolate it by fixing the emphasis weights (`optimizer.explore_std`
small) and letting only the order vary. In synthetic isolation tests the order
channel recovers the true order-importance; under joint search with emphasis it
is noisier (the two channels confound), so clean sequence claims warrant a
dedicated order-only run. The `init_order` field in `heuristics.yaml` (defaults
to `init_weight`) seeds the starting sequence.

## What "good" looks like

The benchmark to beat is the DeepTraffic **76.3 mph** all-time ceiling. Hand
tuning reached a 24-run median of **76.5** (`../experiment.md`); the engine's
practical optimum for any policy is ~76.4–76.5, so a strong run lands the median
in the mid-76s. The report's trajectory should show the median trending up across
iterations and the contribution table should rank the core heuristics
(accelerate-when-clear, overtake-when-blocked, pick-faster-side,
avoid-slow-close-lane) above the optional ones (fleet-spreading, courtesy).

## Files

```
config/heuristics.yaml   the 10 atomic driving heuristics (edit me)
config/default.yaml      the experiment config (edit me)
prompt_em/heuristics.py  load yaml
prompt_em/render.py      fixed semantic state renderer (the driver's user msg)
prompt_em/synthesize.py  weights -> system prompt (llm | template)
prompt_em/optimizer.py   regression-EM over the weights
prompt_em/runner.py      evaluate a prompt over N engine runs
prompt_em/experiment.py  the iteration loop + logging + report
prompt_em/cli.py         command line (run | synth | report)
references.md            related literature
```
