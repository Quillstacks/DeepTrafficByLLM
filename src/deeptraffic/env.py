"""Frozen public API for the DeepTraffic Python port.

``DeepTrafficEnv`` wraps :class:`deeptraffic.engine.Engine` and exposes a small,
stable, Gym-like interface. The engine itself is a verbatim port of
``reference/original_js/gameopt.js`` (see that module's docstring for the
fidelity rules).

step() semantics
----------------
``step()`` is **decision-level**: each call advances the simulation by exactly
30 internal frames (one ego decision interval), applying ``action`` at the
decision boundary. Internally it runs the frame-accurate ``Engine.V`` loop 30
times, so the per-frame dynamics are identical to the JS engine; only the agent
*observes / acts* at the 30-frame cadence (matching how the DQN agent acts).

The reward returned is the engine's own ego reward ``(N - 60) / 20`` where ``N``
is the sum of ``c*a`` over the 30 frames of the interval -- exactly the value the
JS engine passes to ``learn(state, reward)``.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

import numpy as np

from .engine import Engine, ParkMillerLCG  # noqa: F401 (re-export LCG)

Policy = Callable[[np.ndarray], int]


class DeepTrafficEnv:
    """Faithful headless DeepTraffic v2.0 environment.

    Parameters mirror the network config knobs. Defaults reproduce the
    ``network_basic.js`` configuration (lanesSide=1, patchesAhead=10,
    patchesBehind=0, temporal_window=0, otherAgents=0) which scores ~66.8 mph
    with the trained baseline brain.
    """

    DECISION_INTERVAL = 30  # frames per ego decision (G % 30 == 0)

    def __init__(
        self,
        lanes_side: int = 1,
        patches_ahead: int = 10,
        patches_behind: int = 0,
        temporal_window: int = 0,
        other_agents: int = 0,
        seed: Optional[int] = None,
        frames: int = 2000,
    ) -> None:
        if temporal_window != 0:
            # The engine port models the value-function I/O; temporal stacking is
            # a ConvNetJS Brain concern (deepqlearn.getNetInput) and is not part
            # of the simulation dynamics. We expose num_inputs honoring it but do
            # not stack states inside the env (the JS .s() vector is unstacked).
            pass
        self.lanes_side = lanes_side
        self.patches_ahead = patches_ahead
        self.patches_behind = patches_behind
        self.temporal_window = temporal_window
        self.other_agents = other_agents
        self.frames = frames
        self._seed = seed

        self._eng = Engine(
            lanes_side=lanes_side,
            patches_ahead=patches_ahead,
            patches_behind=patches_behind,
            other_agents=other_agents,
        )
        # frame counter within the current episode (independent of engine.G,
        # which the engine resets to 0 in reset()).
        self._frame = 0
        self._done = False
        # last action persisted to the engine for non-decision frames; the
        # engine handles persistence itself, this is only for info reporting.
        self._last_action = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_inputs(self) -> int:
        """Length of the ``.s()`` observation vector fed to the policy.

        Matches ``(lanesSide*2+1) * (patchesAhead+patchesBehind)`` -- i.e. the
        un-stacked value-function input. For network_basic (ls=1, pa=10, pb=0)
        this is ``3 * 10 = 30``.
        """
        return (self.lanes_side * 2 + 1) * (self.patches_ahead + self.patches_behind)

    @property
    def num_actions(self) -> int:
        return 5

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, deterministic: bool = False) -> np.ndarray:
        """Reset and return the initial ego observation (the ``.s()`` vector).

        deterministic=True mirrors doEvalRun's deterministic branch: it sets the
        engine RNG anchor ``r=t=0`` BEFORE the reset (so the first reset uses
        r=t=1 internally), exactly like the official eval. After that anchor,
        each ``reset()`` advances ``r,t`` by +1, producing the deterministic
        sequence of layouts the official benchmark uses.

        deterministic=False seeds ``r,t`` from ``seed`` (or the constructor seed,
        or leaves the engine's running streams alone if both are None).
        """
        if deterministic:
            self._eng.set_deterministic_seeds()  # r=t=0
        else:
            s = seed if seed is not None else self._seed
            if s is not None:
                # Mimic the non-deterministic eval anchor: use the seed as the
                # r,t base. (reset() will +1 before building, like JS.)
                self._eng.r = int(s)
                self._eng.t = int(s)
        self._eng.reset()
        self._frame = 0
        self._done = False
        self._last_action = 0
        return np.asarray(self._eng.ego_state(), dtype=np.float32)

    def step(self, action: int):
        """Advance ONE decision (30 frames). Returns (obs, reward, done, info).

        ``action`` in {0,1,2,3,4}. The engine applies the ego action on its own
        decision frame (G % 30 == 0), exactly once per decision -- it does NOT
        ramp every frame (see engine.V() action-persistence note).

        ``reward`` is the engine's own ego reward ``(N-60)/20`` for the decision
        that opened this 30-frame block (N = Σ c*a over the prior interval). This
        is the value the JS engine passes to ``learn(state, reward)``.
        """
        if self._done:
            raise RuntimeError("step() called on a done env; call reset() first")
        if not (0 <= action < self.num_actions):
            raise ValueError(f"action {action} out of range 0..{self.num_actions-1}")

        self._last_action = action

        def _policy(_state: List[float], _reward: float) -> int:
            # Returns `action` whenever the engine asks (ego decision frame, and
            # controlled-other frames if other_agents>0). The reward is read from
            # engine.last_ego_reward below, which is unambiguous for the ego.
            return action

        # Run exactly DECISION_INTERVAL frames so the next observation is again on
        # a decision boundary (G % 30 == 0). The engine's V() applies the ego
        # action on its decision frame.
        for _ in range(self.DECISION_INTERVAL):
            self._eng.V(_policy)
            self._frame += 1
            if self._frame >= self.frames:
                self._done = True
                break

        ego = self._eng.z[0]
        speed = ego.c * ego.a  # c*a in [0,4]
        obs = np.asarray(self._eng.ego_state(), dtype=np.float32)
        reward = self._eng.last_ego_reward
        info = {
            "mph": 20.0 * speed,       # current 20*c*a of ego
            "speed": speed,            # c*a
            "lane": ego.b,
            "frame": self._frame,
            "G": self._eng.G,
            "c": ego.c,
            "a": ego.a,
        }
        return obs, reward, self._done, info

    # ------------------------------------------------------------------
    # Frame-accurate evaluation (doEvalRun port)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        policy: Policy,
        runs: int = 500,
        frames: int = 2000,
        deterministic: bool = True,
    ) -> Dict[str, object]:
        """Port of ``doEvalRun``: run ``runs`` episodes of ``frames`` frames each
        and return the MEDIAN (the official benchmark) plus mean and the full
        per-run distribution.

        ``policy`` is a callable ``policy(obs: np.ndarray) -> int`` evaluated at
        the engine's decision cadence. For the ego the engine calls it on
        ``G % 30 == 0``; for controlled others on ``G % 30 == 3*idx``.

        Scoring (verbatim from doEvalRun):
            per frame: O += sum_{B=0..nOther} max(0, z[B].c*z[B].a)/(nOther+1)
            per run score = floor(O/frames * 2000)/100
            benchmark = median over runs (f[floor(runs/2)] after sorting)
        """
        eng = self._eng
        if deterministic:
            eng.set_deterministic_seeds()  # t=r=0

        def _wrapped(state: List[float], reward: float) -> int:
            return policy(np.asarray(state, dtype=np.float32))

        scores: List[float] = []
        nOther = eng.nOtherAgents
        for _g in range(runs):
            eng.reset()
            O = 0.0
            for _P in range(frames):
                eng.V(_wrapped)
                na = eng.nOtherAgents  # Math.floor already applied
                for B in range(na + 1):
                    ca = eng.z[B].c * eng.z[B].a
                    O += max(0.0, ca) / (na + 1)
            # floor(O/frames * 2000)/100
            scores.append(math.floor(O / frames * 2e3) / 100)
        eng.reset()

        ordered = sorted(scores)
        median = ordered[len(ordered) // 2]  # f[floor(runs/2)]
        mean = sum(ordered) / len(ordered)
        return {
            "median": float(median),
            "mean": float(mean),
            "min": float(ordered[0]),
            "max": float(ordered[-1]),
            "runs": ordered,
        }

    # ------------------------------------------------------------------
    # Trajectory dump (for fidelity testing against the Node reference)
    # ------------------------------------------------------------------
    def dump_trajectory(
        self,
        actions: List[int],
        frames: int,
        seed_offset: int = 1,
    ) -> Dict[str, object]:
        """Drive the engine with a scripted ego action list (one action per ego
        decision, cycled) and record per-frame (b, a, c, y, x) of all 20 cars.

        Mirrors ``reference/node_ref/run_ref.js`` ``dump`` mode: set r=t=0, then
        call reset() ``seed_offset`` times, then run ``frames`` frames.
        """
        eng = self._eng
        eng.set_deterministic_seeds()  # t=r=0
        for _ in range(seed_offset):
            eng.reset()

        dec = {"i": 0}

        def _policy(state: List[float], reward: float) -> int:
            a = actions[dec["i"] % len(actions)]
            dec["i"] += 1
            return a

        def capture():
            cars = [[c.b, c.a, c.c, c.y, c.x] for c in eng.z]
            return {"cars": cars, "G": eng.G, "E": eng.E, "N": eng.N}

        states = []
        initial = capture()
        for _ in range(frames):
            eng.V(_policy)
            states.append(capture())
        return {"frames": frames, "seedOffset": seed_offset, "actions": actions,
                "initial": initial, "states": states}
