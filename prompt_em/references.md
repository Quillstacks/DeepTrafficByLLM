# References — related literature

**Our method, in one paragraph.** We use a small local LLM (qwen2.5:3b) as the
per-car *driving policy* in the MIT DeepTraffic highway simulator; all driving
knowledge lives in the prompt. The `prompt_em` suite then *optimizes that
prompt*: it starts from ~10 atomic, human-written driving **heuristics**, gives
each a weight (α), has an LLM **synthesize** one coherent system-prompt that
blends the heuristics in proportion to their weights, **evaluates** it by running
the simulator (objective = median fleet mph over 10 runs), and uses a
**regression / EM** step to estimate each heuristic's contribution and update the
weights — iterating prompt → score → reweight. So we sit at the intersection of
*automatic prompt optimization*, *LLM-as-optimizer*, and *LLM-as-policy for
sequential control*, with a black-box simulator as the reward.

Closeness scale: **very close** (same problem + similar mechanism) ·
**close** (same problem, different mechanism) · **adjacent** (one half of our
setup) · **background** (foundational/contextual).

Citations below were spot-verified against arXiv (title/authors/year/venue).

---

## 1. Automatic prompt optimization & LLM-as-optimizer

- **Zhou, Muresanu, Han, Paster, Pitis, Chan, Ba (2022). "Large Language Models
  Are Human-Level Prompt Engineers" (APE).** arXiv:2211.01910 (ICLR 2023).
  *Gist:* an LLM proposes instruction candidates and searches over them to
  maximize a score function — automatic instruction induction.
  *Closeness:* **close.** Same idea of an LLM-generated prompt scored by a task
  metric and selected/iterated; we differ by optimizing *continuous weights over
  fixed heuristics* rather than searching free-form instructions, and our score
  comes from a control simulator, not a QA benchmark.

- **Yang, Wang, Lu, Liu, Le, Zhou, Chen (2023). "Large Language Models as
  Optimizers" (OPRO).** arXiv:2309.03409 (ICLR 2024).
  *Gist:* describe an optimization task in natural language and let the LLM
  iteratively propose new solutions conditioned on a trajectory of past
  (solution, score) pairs.
  *Closeness:* **very close.** Our loop is an OPRO-style trajectory of
  (prompt, score) pairs feeding the next proposal — but we structure the search
  space as weighted heuristics and use an explicit regression-EM update instead
  of letting the LLM freely re-propose.

- **Pryzant, Iter, Li, Lee, Zhu, Zeng (2023). "Automatic Prompt Optimization
  with 'Gradient Descent' and Beam Search" (APO; often nicknamed ProTeGi).**
  arXiv:2305.03495 (EMNLP 2023).
  *Gist:* use LLM textual feedback as a pseudo-gradient to iteratively rewrite a
  prompt, with beam search over candidates.
  *Closeness:* **close.** Same iterative "evaluate → critique → revise" prompt
  loop; we replace the textual-gradient rewrite with a numeric regression-EM
  update over heuristic weights, which is more interpretable but less free-form.

- **Guo, Wang, Guo, Li, Song, Tan, Liu, Bian, Yang (2023). "EvoPrompt:
  Connecting LLMs with Evolutionary Algorithms Yields Powerful Prompt
  Optimizers."** arXiv:2309.08532 (ICLR 2024).
  *Gist:* evolutionary search (mutation/crossover) over a population of prompts,
  with the LLM as the variation operator, selected by task score.
  *Closeness:* **close.** Population-based black-box prompt search like ours in
  spirit; we keep a single weighted prompt and do gradient-like reweighting
  rather than maintaining a population with evolutionary operators.

- **Fernando, Banarse, Michalewski, Osindero, Rocktäschel (2023).
  "Promptbreeder: Self-Referential Self-Improvement via Prompt Evolution."**
  arXiv:2309.16797.
  *Gist:* evolves both task-prompts and the mutation-prompts that modify them, a
  self-referential improvement loop.
  *Closeness:* **close.** Shares evolve-and-score; our "genome" is the heuristic
  weight vector (compact, human-readable) instead of free-text prompts, and we
  do not evolve the optimizer itself.

- **Yuksekgonul, Bianchi, Boen, Liu, Huang, Guestrin, Zou (2024). "TextGrad:
  Automatic 'Differentiation' via Text."** arXiv:2406.07496.
  *Gist:* backpropagates natural-language feedback through a compound AI system
  to improve each component.
  *Closeness:* **close.** Same "optimize a component of an LLM system against a
  downstream metric" framing; our gradient is a literal least-squares fit of
  score on heuristic weights rather than a textual gradient.

- **Khattab, Singhvi, Maheshwari, Zhang, Santhanam, Vardhamanan, Haq, Sharma,
  Joshi, Moazam, Miller, Zaharia, Potts (2023). "DSPy: Compiling Declarative
  Language Model Calls into Self-Improving Pipelines."** arXiv:2310.03714.
  *Gist:* a framework that compiles/optimizes prompts and few-shot demos for
  modular LLM programs against a metric.
  *Closeness:* **adjacent→close.** Same goal of metric-driven prompt
  optimization as infrastructure; DSPy optimizes demonstrations/instructions for
  pipelines, while we optimize a weighting over hand-authored control heuristics.

- **Sun, Shao, Qian, Huang, Qiu (2022). "Black-Box Tuning for Language-Model-
  as-a-Service" (BBT).** arXiv:2201.03514 (ICML 2022).
  *Gist:* derivative-free (CMA-ES) optimization of continuous prompt tokens
  through black-box LLM API access.
  *Closeness:* **adjacent.** Shares black-box, derivative-free optimization of a
  prompt parameterization; theirs is continuous soft-prompt vectors, ours is a
  weight simplex over discrete human heuristics with a simulator reward.

- **Chen, Zhang, Zhang, Su, Zhu (2023). "InstructZero: Efficient Instruction
  Optimization for Black-Box Large Language Models."** arXiv:2306.03082.
  *Gist:* Bayesian optimization over a low-dimensional soft prompt that is
  decoded by an open LLM into instructions for a black-box LLM.
  *Closeness:* **adjacent.** Black-box, surrogate-model optimization of prompts,
  like our regression surrogate over weights; different parameterization and no
  control/agent setting.

## 2. LLM as policy / decision-maker for (autonomous) driving

- **Wen, Fu, Li, Cai, Ma, Cai, Dou, Shi, He, Qiao (2023). "DiLu: A
  Knowledge-Driven Approach to Autonomous Driving with Large Language Models."**
  arXiv:2309.16292 (ICLR 2024).
  *Gist:* an LLM makes driving decisions via reasoning + reflection memory,
  injecting common-sense knowledge into closed-loop highway control.
  *Closeness:* **very close (problem), adjacent (method).** Nearest neighbor on
  the *task* — LLM as a closed-loop highway driving policy — but DiLu improves
  via a memory of reflections, whereas we optimize the system-prompt itself with
  an outer EM loop.

- **Mao, Qian, Ye, Zhao, Wang (2023). "GPT-Driver: Learning to Drive with
  GPT."** arXiv:2310.01415.
  *Gist:* reformulates motion planning as language modeling; GPT-3.5 emits
  trajectories from tokenized scene inputs (evaluated on nuScenes).
  *Closeness:* **adjacent.** LLM-as-driving-policy like ours, but open-loop
  trajectory generation on a dataset rather than closed-loop fleet simulation,
  and no prompt optimization loop.

- **Sha, Mu, Jiang, Chen, Xu, Luo, Li, Tomizuka, Zhan, Ding (2023).
  "LanguageMPC: Large Language Models as Decision Makers for Autonomous
  Driving."** arXiv:2310.03026.
  *Gist:* LLM does high-level reasoning ("cognitive pathways") and its outputs
  are converted into low-level control via a parameter matrix.
  *Closeness:* **adjacent.** LLM as a driving decision-maker feeding a
  controller; we instead have the LLM emit the discrete action directly and
  focus on optimizing its prompt.

- **Fu, Li, Wen, Dou, Cai, Shi, Qiao (2023). "Drive Like a Human: Rethinking
  Autonomous Driving with Large Language Models."** arXiv:2307.07162.
  *Gist:* argues for and demonstrates human-like, closed-loop reasoning by an
  LLM driver in simulation, with reflection on mistakes.
  *Closeness:* **adjacent.** Same motivation (LLM common-sense driving in
  closed loop); qualitative/reflective rather than an optimization framework.

## 3. Iterative self-improvement loops (the outer-loop pattern)

- **Shinn, Cassano, Berman, Gopinath, Narasimhan, Yao (2023). "Reflexion:
  Language Agents with Verbal Reinforcement Learning."** arXiv:2303.11366
  (NeurIPS 2023).
  *Gist:* an agent verbally reflects on task feedback and stores reflections in
  memory to improve over trials — RL-like improvement without weight updates.
  *Closeness:* **adjacent→close.** Same "evaluate → reflect → improve" outer
  loop; our "reflection" is a quantitative regression over heuristic weights
  rather than natural-language self-critique.

- **Madaan, Tandon, Gupta, Hallinan, Gao, Wiegreffe, Alon, Dziri, Prabhumoye,
  Yang, Gupta, Majumder, Hermann, Welleck, Yazdanbakhsh, Clark (2023).
  "Self-Refine: Iterative Refinement with Self-Feedback."** arXiv:2303.17651
  (NeurIPS 2023).
  *Gist:* a single LLM iteratively critiques and refines its own output without
  any external supervision.
  *Closeness:* **background→adjacent.** The iterate-on-feedback motif underlies
  our loop; we externalize the feedback as a simulator score and a numeric
  weight update rather than LLM self-critique.

---

## Where our work sits

The **nearest neighbors** are **OPRO** (LLM-as-optimizer over a (solution, score)
trajectory) and the prompt-evolution line (**EvoPrompt**, **Promptbreeder**,
**APO/ProTeGi**) on the optimization side, and **DiLu** on the task side
(LLM as a closed-loop highway driving policy). What appears **novel in
combination** is: (1) the search space is a **weight simplex over a small set of
human-authored, interpretable driving heuristics** (not free-form prompt text or
soft-prompt vectors), (2) the update is an explicit **regression/EM credit-
assignment** over those weights — giving a readable account of *which* heuristics
the score rewards — while (3) an LLM performs the **synthesis** from weights to
fluent prompt, and (4) the reward is the **median fleet speed of a multi-agent
traffic simulator**, i.e. a closed-loop control objective rather than a static
benchmark. In short: OPRO/EvoPrompt-style black-box prompt optimization, but with
an interpretable heuristic-weight genome and an EM update, applied to a DiLu-style
LLM driving policy.
