"""Regression-EM optimiser over the heuristic weights (the "alphas").

We are NOT looking for the single best heuristic; we are looking for the best
WEIGHTED MIX. So the state is a weight vector w on the simplex (one weight per
heuristic), and we search for the w whose synthesised prompt maximises the
objective (median fleet mph).

Each round:
  * E-step  -- regress the observed objective on the weight vectors tried so far
               (ridge regression). The coefficient for heuristic i estimates its
               marginal contribution to the score: its "responsibility".
  * M-step  -- turn the responsibilities into a target distribution and move the
               weights toward it with learning rate ``alpha``:
                   w <- (1 - alpha) * w_base + alpha * responsibility
               anchored on the best-scoring weights seen so far, plus annealed
               exploration noise.

Cold start: with few observations the regression is underdetermined, so the
first ``cold_start_rounds`` use deliberately diverse weight vectors (uniform +
large noise) to make the regression identifiable before we start exploiting it.
Ridge regularisation keeps the early estimates from blowing up.

Pure numpy; deterministic given the seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


def _normalise(w: np.ndarray) -> np.ndarray:
    w = np.clip(w, 0.0, None)
    s = w.sum()
    return np.full_like(w, 1.0 / len(w)) if s <= 1e-12 else w / s


@dataclass
class RegressionEM:
    n: int                       # number of heuristics
    alpha: float = 0.35          # learning rate (weight blend factor)
    ridge_lambda: float = 1.0    # L2 for the contribution regression
    explore_std: float = 0.12    # exploration noise std (on the simplex)
    explore_decay: float = 0.6   # noise *= decay each round
    cold_start_rounds: int = 3
    seed: int = 0
    init_weights: Optional[np.ndarray] = None
    frozen: bool = False         # if True, always return the init weights (no
                                 # exploration/exploit) -- for isolating the
                                 # OTHER channel (e.g. order-only experiments)

    _rng: np.random.Generator = field(init=False)
    _round: int = field(default=0, init=False)
    W: List[np.ndarray] = field(default_factory=list, init=False)   # tried weights
    y: List[float] = field(default_factory=list, init=False)        # scores

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        if self.init_weights is None:
            self._base = np.full(self.n, 1.0 / self.n)
        else:
            self._base = _normalise(np.asarray(self.init_weights, float))
        self._lhs = self._latin_hypercube(max(self.cold_start_rounds, 1))

    def _latin_hypercube(self, m: int) -> np.ndarray:
        """m space-filling points in the n-simplex. Independent per-dimension
        stratification makes the cold-start regression IDENTIFIABLE -- every
        heuristic's weight is varied independently, so collinear pairs can be
        separated (random jitter around uniform cannot do this)."""
        pts = np.empty((m, self.n))
        for j in range(self.n):
            strata = (self._rng.permutation(m) + self._rng.random(m)) / m
            pts[:, j] = strata
        # sharpen so points are genuinely diverse on the simplex, then normalise
        pts = pts ** 2
        return np.vstack([_normalise(p) for p in pts])

    # ---- E-step: contribution of each heuristic ----
    def responsibilities(self) -> np.ndarray:
        """Ridge regression of score on the weight vectors -> per-heuristic
        marginal contribution, mapped to a TARGET MIX. We use the positive part
        of the contributions (share proportional to how much each heuristic
        helps) plus a uniform floor, rather than a softmax: we want the best
        *blend* of helpful heuristics, not to collapse onto the single best one
        (a one-rule prompt underperforms)."""
        beta = self.contributions()
        if beta is None or beta.std() < 1e-9:
            return self._base.copy()
        pos = np.clip(beta, 0.0, None)
        if pos.sum() <= 1e-9:                 # nothing clearly helps -> stay broad
            return self._base.copy()
        share = pos / pos.sum()
        floor = 1.0 / self.n                  # keep every heuristic alive a bit
        return _normalise(share + 0.5 * floor)

    # ---- propose the weights to try this round ----
    def next_weights(self) -> np.ndarray:
        if self.frozen:
            return self._base.copy()
        # persistent exploration floor: never let the noise fully vanish, or a
        # heuristic excluded early can never be revisited.
        noise_scale = self.explore_std * max(self.explore_decay ** self._round, 0.25)
        if self._round == 0:
            w = self._base.copy()
        elif self._round < self.cold_start_rounds:
            # cold start: space-filling Latin-hypercube design (identifiable)
            w = self._lhs[self._round % len(self._lhs)].copy()
        else:
            # exploit: move best-so-far toward the EM responsibilities
            base = self.best()[0] if self.W else self._base
            target = self.responsibilities()
            w = (1 - self.alpha) * base + self.alpha * target
            w = w + self._rng.normal(0, noise_scale, self.n)
        return _normalise(w)

    def observe(self, weights: np.ndarray, score: float) -> None:
        self.W.append(np.asarray(weights, float))
        self.y.append(float(score))
        self._round += 1

    def explore_scale(self) -> float:
        return self.explore_std * max(self.explore_decay ** self._round, 0.25)

    # ---- introspection ----
    def best(self) -> Tuple[np.ndarray, float]:
        i = int(np.argmax(self.y))
        return self.W[i], self.y[i]

    def contributions(self) -> Optional[np.ndarray]:
        """Raw ridge coefficients (signed marginal value), or None if too few
        observations. Positive => this heuristic's weight helps the objective."""
        if len(self.y) < 2:
            return None
        X = np.vstack(self.W)
        y = np.asarray(self.y, float)
        Xc = X - X.mean(axis=0, keepdims=True)
        A = Xc.T @ Xc + self.ridge_lambda * np.eye(self.n)
        return np.linalg.solve(A, Xc.T @ (y - y.mean()))


class JointOptimizer:
    """Co-optimise BOTH the emphasis weights AND the sequence of the heuristics.

    Two channels, each its own regression-EM over the simplex:
      * ``emphasis``  -- how strongly each heuristic is stated (and whether it is
                         included at all). This controls the prompt's *content*.
      * ``priority``  -- the ORDER: heuristics are placed in the prompt sorted by
                         descending priority, so this controls the *sequence*.
    Both are proposed (and explored) independently each round and regressed
    against the same objective, so the loop searches content and order jointly
    and logs each separately -- enabling post-hoc "sequence importance" analysis
    (e.g. regress the objective on the order channel alone).
    """

    def __init__(self, n: int, *, alpha, ridge_lambda, explore_std,
                 explore_decay, cold_start_rounds, seed,
                 init_weights=None, init_priority=None,
                 freeze_emphasis=False, freeze_order=False):
        self.n = n
        self.emphasis = RegressionEM(
            n=n, alpha=alpha, ridge_lambda=ridge_lambda, explore_std=explore_std,
            explore_decay=explore_decay, cold_start_rounds=cold_start_rounds,
            seed=seed, init_weights=init_weights, frozen=freeze_emphasis)
        self.priority = RegressionEM(
            n=n, alpha=alpha, ridge_lambda=ridge_lambda, explore_std=explore_std,
            explore_decay=explore_decay, cold_start_rounds=cold_start_rounds,
            seed=seed + 9973, init_weights=init_priority, frozen=freeze_order)
        self._scores: List[float] = []
        self._wp: List[Tuple[np.ndarray, np.ndarray]] = []

    def propose(self) -> Tuple[np.ndarray, np.ndarray, list]:
        """Return (emphasis_weights, priority_vector, order_indices)."""
        w = self.emphasis.next_weights()
        p = self.priority.next_weights()
        order = [int(i) for i in np.argsort(-p)]   # highest priority first
        return w, p, order

    def observe(self, weights, priority, score: float) -> None:
        self.emphasis.observe(weights, score)
        self.priority.observe(priority, score)
        self._scores.append(float(score))
        self._wp.append((np.asarray(weights, float), np.asarray(priority, float)))

    def best(self) -> Tuple[np.ndarray, np.ndarray, list, float]:
        i = int(np.argmax(self._scores))
        w, p = self._wp[i]
        order = [int(j) for j in np.argsort(-p)]
        return w, p, order, self._scores[i]

    def contributions(self):
        """(emphasis_contributions, order_contributions) -- signed marginal value
        of each heuristic's weight and of placing it earlier."""
        return self.emphasis.contributions(), self.priority.contributions()

    def diagnostics(self) -> dict:
        """Optimiser internal state for logging the optimisation dynamics."""
        return {
            "phase": ("cold_start" if self.emphasis._round
                      < self.emphasis.cold_start_rounds else "exploit"),
            "explore_scale": round(self.emphasis.explore_scale(), 4),
            "emphasis_responsibility":
                [round(float(x), 4) for x in self.emphasis.responsibilities()],
            "priority_responsibility":
                [round(float(x), 4) for x in self.priority.responsibilities()],
        }
