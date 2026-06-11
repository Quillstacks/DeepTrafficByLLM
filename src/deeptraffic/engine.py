"""Verbatim Python port of the MIT DeepTraffic v2.0 engine (``gameopt.js``).

This module is a *line-for-line* translation of the minified simulation in
``reference/original_js/gameopt.js``. Variable names follow the original where
useful for cross-referencing; comments cite the relevant gameopt.js construct.

Fidelity rules followed here (do NOT "improve"):

* All arithmetic uses Python floats (IEEE-754 double), matching JS Number.
* ``Math.floor`` -> :func:`math.floor` (floors toward -inf, like JS).
* Integer division of pixel coordinates is done via float then ``floor`` inside
  :class:`GameMap` ``get``/``set`` exactly as ``Map.set``/``Map.get`` do.
* The two RNG streams ``u`` (map init) and ``v`` (car behavior) are persistent
  stateful objects, reseeded only where ``reset()`` reseeds them. The ordering of
  RNG consumption is reproduced exactly (this matters: see ``reset`` notes).
* Spawn ``y`` in the ``D`` constructor / ``j`` uses ``Math.random`` in the JS;
  during ``initializeMap`` every car's ``y`` is overwritten, and on off-road
  wrap ``j`` reuses ``Math.random`` only for an immediately-overwritten field in
  the constructor case (in ``j`` itself the ``y`` is NOT touched). We model the
  nondeterministic ``Math.random`` via an injectable callback so the trajectory
  test can drive it from the JS reference if needed; under deterministic eval it
  has no observable effect on (b, a, c, y) after ``initializeMap`` because all
  positions are overwritten and wraps set ``y`` explicitly (734 / -34) *before*
  calling ``j`` (and ``j`` does not modify ``y``). See ``Car.j`` / ``Car.move``.
"""

from __future__ import annotations

import math
import random as _pyrandom
from typing import Callable, List, Optional


def js_min(x: float, y: float) -> float:
    """``Math.min(x, y)`` with JS NaN semantics: if either is NaN -> NaN.

    Python's builtin ``min`` returns the first argument when a NaN is involved,
    which differs from JS. The DeepTraffic safety scan relies on Math.min, so we
    mirror JS exactly (relevant only in the pathological 0/0 gas case).
    """
    if x != x or y != y:  # NaN check
        return float("nan")
    return x if x < y else y


# ---------------------------------------------------------------------------
# RNG -- Park-Miller minimal standard LCG. Mirrors `function p(a)` + `q`.
# ---------------------------------------------------------------------------
class ParkMillerLCG:
    """Exact port of gameopt.js ``function p(a){this.g=a%2147483647;...}``.

    ``next()`` -> ``g = 16807*g % 2147483647``. ``q(stream)`` is implemented as
    :meth:`nextq` -> ``(next()-1)/2147483646`` in [0, 1).
    """

    __slots__ = ("g",)

    def __init__(self, seed: int) -> None:
        self.g = seed % 2147483647
        if self.g <= 0:
            self.g += 2147483646

    def next(self) -> int:
        self.g = 16807 * self.g % 2147483647
        return self.g

    def nextq(self) -> float:
        """``q(a) = (a.next()-1)/2147483646``."""
        return (self.next() - 1) / 2147483646


# ---------------------------------------------------------------------------
# Map -- the 2D grid. Mirrors `function Map(a,b,d)`.
# ---------------------------------------------------------------------------
class GameMap:
    """Port of ``function Map(rows, cols, defaultValue)``.

    NOTE on indexing: the JS ``Map`` is ``data[i][j]`` where ``i`` ranges over
    the *first* dimension (``a`` cols = 7 for ``H``) and ``j`` over the second
    (``b`` rows = 70 for ``H``). We keep the same orientation: ``data[i][j]``.
    ``set(a,b,c)`` floors ``a`` and ``b`` and bounds-checks.
    """

    __slots__ = ("data", "defaultValue", "_rows", "_cols")

    def __init__(self, rows: int, cols: int, default_value: float) -> None:
        # JS: for c in 0..rows: push row of length cols filled with default.
        self.defaultValue = float(default_value)
        self._rows = rows
        self._cols = cols
        self.data: List[List[float]] = [
            [self.defaultValue for _ in range(cols)] for _ in range(rows)
        ]

    def reset(self) -> None:
        dv = self.defaultValue
        for row in self.data:
            for j in range(len(row)):
                row[j] = dv

    def set(self, a: float, b: float, c: float) -> None:
        a = math.floor(a)
        b = math.floor(b)
        if 0 <= a < len(self.data) and 0 <= b < len(self.data[a]):
            self.data[a][b] = c

    def get(self, a: float, b: float, c: Optional[float] = None) -> float:
        a = math.floor(a)
        b = math.floor(b)
        if 0 <= a < len(self.data) and 0 <= b < len(self.data[a]):
            return self.data[a][b]
        return self.defaultValue if c is None else c

    # this.o = function(a,b){ ... } -- fill grid `b` (=K) for car index `a`.
    def fill_obs(self, car_idx: int, dest: "GameMap", engine: "Engine") -> None:
        c = engine.lanesSide
        d = engine.patchesAhead
        f = engine.patchesBehind
        if car_idx == 0:
            engine.C = engine.z[0].b  # global C (cutout-view column)
        z = engine.z
        g = -c
        while g <= c:
            h = -d
            while h < f:
                dest.data[g + c][h + d] = self.get(
                    z[car_idx].b + g, math.floor(z[car_idx].y) / 10 + h, 0
                )
                h += 1
            g += 1

    # this.s = function(){...} -- column-major flatten, /100.
    def to_state(self) -> List[float]:
        rows = len(self.data)
        cols = len(self.data[0])
        out = [0.0] * (rows * cols)
        for b in range(rows):
            for c in range(cols):
                out[rows * c + b] = self.data[b][c] / 100
        return out


# ---------------------------------------------------------------------------
# Car -- mirrors `function D()`.
# ---------------------------------------------------------------------------
class Car:
    """Port of ``function D()`` (a vehicle).

    Fields: x, y (px), b (lane), a (gas in [0,2]), c (safety factor in [0,2]),
    h (60-len ring buffer of c*a*20 for mph display), f (controlled? bool).
    """

    __slots__ = ("x", "y", "a", "c", "b", "h", "f", "_engine", "_gas_mod")

    def __init__(self, engine: "Engine") -> None:
        self._engine = engine
        # this.y=this.x=0; this.a=this.c=1; this.b=0; this.h=Array(60);
        self.x = 0.0
        self.y = 0.0
        self.a = 1.0
        self.c = 1.0
        self.b = 0
        self.h = [0.0] * 60
        # IMPORTANT: in JS `this.j()` runs BEFORE `this.f=!1`, so during the
        # constructor's j() call `this.f` is undefined -> falsy. We replicate by
        # leaving f unset until after j().
        self.f = False  # placeholder; gets its constructor semantics below
        self._gas_mod: Callable[[], float] = lambda: 0.0
        # Run j() with f acting as falsy (undefined). We set a temporary flag.
        self._j(constructor=True)
        # this.y = 10*Math.floor(700*Math.random()/10);  (Math.random!)
        self.y = 10 * math.floor(700 * engine.math_random() / 10)
        # this.f=!1;
        self.f = False

    # this.j = function(){ ... }
    def _j(self, constructor: bool = False) -> None:
        eng = self._engine
        a = math.floor(140 * eng.v.nextq() / 20)  # lane index 0..6
        self.x = 20 * a + 4
        # switch(a){ ... } sets the gasPedalModulator closure. NOTE the JS switch
        # uses `case 5<a:` etc. -- these are boolean cases compared against `a`
        # in a `switch(a)`. Since `a` is an integer 0..6 it never equals `true`,
        # so EVERY explicit case falls through and ONLY `default` is selected.
        # (This is a quirk in the original; reproduced verbatim: modulator is
        #  always the default `0.5*q(v)`.) See gameopt.js function D / this.j.
        def default_mod() -> float:
            return 0.5 * eng.v.nextq()

        self._gas_mod = default_mod
        eng.gasPedalModulator = self._gas_mod
        # this.f || (this.a = 1 + .7*gasPedalModulator());
        controlled = self.f if not constructor else False
        if not controlled:
            self.a = 1 + 0.7 * eng.gasPedalModulator()
        self.b = a

    # this.move = function(a){ ... }  (a = isOther flag = (index != 0))
    def move(self, is_other: bool) -> None:
        eng = self._engine
        b = self.y - (self.c * self.a - eng.E)
        # passing counter F
        if is_other and self.y > 525 and 525 <= b:
            eng.F += 1
        elif is_other and self.y < 525 and 525 >= b:
            eng.F -= 1
        self.y = b
        # this.h[G%60] = c*a*20
        self.h[eng.G % len(self.h)] = self.c * self.a * 20
        # x easing toward target lane center 20*b+4 by step 20/30
        a = 20 * self.b + 4 - self.x
        if abs(a) < 20 / 30:
            self.x = 20 * self.b + 4
        elif a > 0:
            self.x = self.x + 20 / 30
        else:
            self.x = self.x - 20 / 30
        # off-road wrap + respawn
        if self.y + 68 < 0:
            self.y = 734
            self._j()
        if self.y - 68 > 700:
            self.y = -34
            self._j()

    # this.l = function(){ stamp footprint into H with value c*a }
    def stamp(self) -> None:
        eng = self._engine
        a = 0
        while a < 15:
            b = 0
            while b < 34:
                eng.H.set((self.x + a) / 20, (self.y + b) / 10, 1 * self.c * self.a)
                b += 5
            a += 10

    # this.u = function(){ safety scan -> set c }
    def safety(self) -> None:
        eng = self._engine
        a = 2.0
        b = 1
        while b < 5:
            d = eng.H.get((self.x + 7.5) / 20, (self.y - 10 * b) / 10, 100)
            if d < 100:
                a = js_min(a, 0.5 * (b - 1))
                # JS: a = Math.min(a, d/this.a). When this.a == 0, JS yields
                # d/0 = +Infinity (d>0) or NaN (d==0). js_min replicates both.
                if self.a != 0:
                    ratio = d / self.a
                else:
                    ratio = float("nan") if d == 0 else float("inf")
                a = js_min(a, ratio)
            b += 1
        self.c = a

    # this.i = function(a){ attempt lane change by `a` (-1 left, +1 right) }
    def lane_change(self, a: int) -> bool:
        eng = self._engine
        b = (self.x + 7.5) / 20 + a
        d = self.y / 10
        # c starts as: is the car centered in its lane? (|x - (20*b+4)| < 0.5)
        c = abs(self.x - (20 * self.b + 4)) < 0.5
        f = 3 * -self.a
        while f < 4:
            c = c and (100 <= eng.H.get(b, d + f, 0))
            f += 1
        if c:
            self.b += a
        return c

    # this.m = function(a){ apply action a }
    def act(self, action: int) -> None:
        if action == 1:
            if self.a < 2:
                self.a += 0.02
        elif action == 2:
            if self.a > 0:
                self.a -= 0.02
        elif action == 3:
            if self.lane_change(-1):
                self._engine.J = 0
        elif action == 4:
            if self.lane_change(1):
                self._engine.J = 0
        # action 0 / other -> nothing

    # this.w = function(){ floor(mean(h)) for mph display }
    def avg_mph(self) -> int:
        return math.floor(sum(self.h) / len(self.h))


# ---------------------------------------------------------------------------
# Engine -- holds globals (z, H, I, K, G, E, N, J, F, u, v, ...) and V()/reset().
# ---------------------------------------------------------------------------
class Engine:
    """Port of the gameopt.js module globals + ``reset()``, ``initializeMap()``,
    ``V()``, and ``doEvalRun`` scoring loop.

    A policy is provided as a callable taking the ``.s()`` state vector and the
    last reward, returning an integer action in {0..4}. This mirrors the
    ``learn(state, lastReward)`` hook the engine calls for the ego.
    """

    def __init__(
        self,
        lanes_side: int = 1,
        patches_ahead: int = 10,
        patches_behind: int = 0,
        other_agents: int = 0,
        math_random: Optional[Callable[[], float]] = None,
    ) -> None:
        self.lanesSide = lanes_side
        self.patchesAhead = patches_ahead
        self.patchesBehind = patches_behind
        self.otherAgents = math.floor(other_agents)
        # nOtherAgents = Math.min(otherAgents, 9)  (line 1 of gameopt.js)
        self.nOtherAgents = min(self.otherAgents, 9)

        # Injectable Math.random source. Default = Python's RNG. Under
        # deterministic eval its result is overwritten by initializeMap, so it
        # does not affect observable (b,a,c,y) trajectories.
        self._math_random = math_random or _pyrandom.random

        self.n = [0, 1, 2, 3, 4]

        # RNG streams. JS: r=1,t=1,u=new p(r),v=new p(t).
        self.r = 1
        self.t = 1
        self.u = ParkMillerLCG(self.r)
        self.v = ParkMillerLCG(self.t)

        # gasPedalModulator is a single mutable global closure.
        self.gasPedalModulator: Callable[[], float] = lambda: 0.0

        # Maps. H=7x70 default 100, I=7x70 default 100, K=(2*ls+1)x(pa+pb) def 0.
        self.H = GameMap(7, 70, 100)
        self.I = GameMap(7, 70, 100)
        self.K = GameMap(1 + 2 * lanes_side, patches_ahead + patches_behind, 0)

        self.C = 0
        # z = 20 cars (the constructors consume v exactly like JS load order).
        self.z: List[Car] = [Car(self) for _ in range(20)]

        # globals: G=0,E=1.5,M=0,N=0,J=0,F=0,Q=false
        self.G = 0
        self.E = 1.5
        self.M = 0.0
        self.N = 0.0
        self.J = 0
        self.F = 0
        self.last_ego_reward = 0.0  # set on each ego decision frame (env helper)

        # The module body ends with `initializeMap(u)`.
        self.initializeMap(self.u)

        # Snapshot the RNG/state anchor that exactly matches a FRESH gameopt.js
        # load (v = p(1) after the 20 constructor consumes; u = p(1) after
        # initializeMap). doEvalRun's deterministic branch only sets t=r=0 and
        # does NOT recreate u/v, so the very first run consumes from this exact
        # state. We restore it in deterministic evaluate()/dump so repeated calls
        # reproduce a fresh-load run (true fidelity). See `restore_fresh_load`.
        self._fresh_u_g = self.u.g
        self._fresh_v_g = self.v.g

    def math_random(self) -> float:
        return self._math_random()

    # initializeMap = function(a){ ... }   (a = the `u` stream)
    def initializeMap(self, a: ParkMillerLCG) -> None:
        # JS uses `legalLocations.splice(legalLocations.indexOf(x), 1)`. When `x`
        # is `undefined` (footprint cells with row+g < 0 are left undefined in
        # `Array(12)`), indexOf returns -1 and splice(-1, 1) removes the LAST
        # element. We replicate this exact (buggy-but-faithful) behavior. We use
        # `None` as the JS `undefined` sentinel inside footprints.
        SENTINEL = None

        def pick(arr: List[int]) -> int:
            c = math.floor(len(arr) * a.nextq())
            return arr[c]

        def footprint(col: int, row: int):
            # JS d(a,b): `for(var c=Array(12),d=0,g=-4;12>g;g++)` runs g=-4..11
            # (16 iterations, d=0..15). JS `Array(12)` auto-extends to length 16 as
            # high indices are assigned; slots where row+g<0 stay `undefined` (holes),
            # which later make indexOf return -1 -> splice(-1,1) removes the last legal
            # cell. So size to 16 and keep None holes. col%7 == col since col in 0..6.
            cells = [SENTINEL] * 16
            d = 0
            g = -4
            while g < 12:
                if 0 <= row + g:
                    cells[d] = 7 * (row + g) + col % 7
                g += 1
                d += 1
            return cells

        def splice_indexof(arr: List[int], x) -> None:
            # arr.splice(arr.indexOf(x), 1)
            try:
                idx = arr.index(x)  # JS indexOf
            except ValueError:
                idx = -1
            if not arr:
                return
            if idx == -1:
                arr.pop()  # splice(-1, 1) removes the last element
            else:
                arr.pop(idx)

        z = self.z
        z[0].y = 525
        z[0].x = 64
        z[0].b = 3
        legal = list(range(490))  # Array(490).fill().map((a,b)=>b)
        c = math.floor(z[0].x / 20)
        f = math.floor(z[0].y / 10 + 4)
        l = footprint(c, f)
        z[0].a = 2
        for v in l:
            splice_indexof(legal, v)
        g = 1
        while g < len(z):
            c = pick(legal)
            f = math.floor(c / 7)
            c = c % 7
            l = footprint(c, f)
            for v in l:
                splice_indexof(legal, v)
            z[g].x = math.floor(20 * c + 4)
            z[g].y = math.floor(f / 70 * 700)
            z[g].b = c
            if z[g].f:
                z[g].a = 1.7
            g += 1
        z[0].a = 2

    # reset = function(){ ... }
    def reset(self) -> None:
        # JS reset() raises the cap from 9 (load-time, gameopt.js line 1) to 10:
        #   nOtherAgents != Math.min(otherAgents,10) && (nOtherAgents=Math.min(otherAgents,10), brains=y(), w=false)
        # So after the first reset there are min(otherAgents,10) controlled OTHERS
        # plus the ego -> up to 11 controlled cars. (brains=y()/w is ConvNetJS
        # brain-clone bookkeeping; our port drives controlled others through
        # `policy` directly, so only the controlled-car COUNT matters here.)
        self.nOtherAgents = min(self.otherAgents, 10)
        self.H = GameMap(7, 70, 100)
        self.I = GameMap(7, 70, 100)
        self.K = GameMap(
            1 + 2 * self.lanesSide, self.patchesAhead + self.patchesBehind, 0
        )
        # z=[]; for a in 0..19: z.push(new D); if a<nOtherAgents+1: z[a].f=true
        # NOTE: `new D()` consumes the CURRENT v stream (pre-reseed); only AFTER
        # constructing all 20 does reset reseed u,v. We mirror that order exactly.
        self.z = []
        for a in range(20):
            self.z.append(Car(self))
            if a < self.nOtherAgents + 1:
                self.z[a].f = True
        # r+=1; t+=1; u=new p(r); v=new p(r);   (BOTH from r!)
        self.r += 1
        self.t += 1
        self.u = ParkMillerLCG(self.r)
        self.v = ParkMillerLCG(self.r)
        self.initializeMap(self.u)
        self.G = 0
        self.E = 1.5
        self.F = self.J = 0
        self.N = 0.0
        self.M = 0.0

    def restore_fresh_load(self) -> None:
        """Restore the u/v RNG streams to their exact FRESH-load state.

        This makes a deterministic evaluate()/dump reproduce a fresh gameopt.js
        load even if reset()/V() have been called before on this Engine. After
        this, the next reset()'s `new Car()` constructors consume from the same v
        state the JS engine has at doEvalRun entry. (gameopt.js's deterministic
        branch sets t=r=0 but never recreates u/v; the fresh-load v carries 40
        constructor consumes from seed 1.)
        """
        self.u.g = self._fresh_u_g
        self.v.g = self._fresh_v_g

    def set_deterministic_seeds(self) -> None:
        """Mirror doEvalRun's deterministic branch: ``t=r=0`` and restore the
        fresh-load u/v state so the first run is bit-identical to a fresh JS load.

        Brain ``reset_seed(0)`` is the policy's concern (handled by the caller).
        """
        self.t = self.r = 0
        self.restore_fresh_load()

    # V() -- one frame.  policy(state_vector, last_reward) -> action int.
    def V(self, policy: Callable[[List[float], float], int]) -> None:
        z = self.z
        nOther = self.nOtherAgents
        self.H.reset()
        # move + stamp all cars. move(0!=a) -> is_other = (a != 0).
        for a in range(len(z)):
            z[a].move(a != 0)
            z[a].stamp()
        # E = 1.5 - (z[0].y - 525)
        self.E = 1.5 - (z[0].y - 525)
        # safety for all; game cars random lane change with prob >.99+.004*c
        for a in range(len(z)):
            z[a].safety()
            if a > nOther and self.v.nextq() > 0.99 + 0.004 * z[a].c:
                bdir = -1 if 0.5 < self.v.nextq() else 1
                z[a].lane_change(bdir)

        # Controlled others z[1..nOtherAgents]: decide on G%30 == 3*a (staggered).
        #
        # ACTION-PERSISTENCE RESOLUTION (subtle point from ENGINE_NOTES):
        # In gameopt.js, `d` is a function-local `var` in V(), so it is a FRESH
        # `undefined` on EVERY frame (it does NOT persist across frames). The ego
        # therefore acts ONLY on its decision frame (G%30==0); on all other
        # frames `z[0].m(d)` is `m(undefined)` -> switch default -> no-op. So
        # "accelerate" applies +0.02 ONCE per decision, NOT ramping every frame.
        # The one exception: on a frame where a controlled-other decides
        # (G%30==3*a) but the ego does not, that other's `d` leaks into the final
        # `z[0].m(d)` -- a faithful quirk we preserve. With nOtherAgents=0 this
        # loop is skipped entirely and the ego only acts on G%30==0.
        d: Optional[int] = None
        for a in range(1, nOther + 1):
            if self.G % 30 == 3 * a:
                kmap = GameMap(
                    1 + 2 * self.lanesSide,
                    self.patchesAhead + self.patchesBehind,
                    0,
                )
                self.H.fill_obs(a, kmap, self)
                # The controlled-other brains run the SAME policy in the JS
                # (cloned brains). We approximate with the same policy callable.
                dd = policy(kmap.to_state(), 0.0)
                d = dd if (0 <= dd < len(self.n)) else self.J
            if d is not None:
                z[a].act(d)

        # ego stamp again, safety-view (skipped: needs k flag, headless=false UI)
        z[0].stamp()
        # N += z[0].c*z[0].a
        self.N += z[0].c * z[0].a
        # ego decision on G%30==0
        if self.G % 30 == 0:
            self.H.fill_obs(0, self.K, self)
            reward = (self.N - 60) / 20
            self.last_ego_reward = reward  # exposed for the env wrapper
            dd = policy(self.K.to_state(), reward)
            d = dd if (0 <= dd < len(self.n)) else self.J
            self.N = 0.0
        # z[0].m(d). `d` is None on non-decision frames (unless a controlled
        # other set it this frame) -> act(None) is a no-op, matching JS
        # m(undefined). So the ego applies its action exactly once per decision.
        if d is not None:
            z[0].act(d)
        self.G += 1

    # ----------------------------------------------------------------
    # Convenience: the .s() observation for the ego at the current frame.
    # ----------------------------------------------------------------
    def ego_state(self) -> List[float]:
        self.H.fill_obs(0, self.K, self)
        return self.K.to_state()
