"""Fidelity tests: Python port vs. the ORIGINAL JavaScript engine.

Two tests, in order of strength:

1. ``test_trajectory_matches_js`` -- the STRONGEST test. Drives BOTH engines with
   an identical scripted ego action sequence and identical deterministic seeds,
   then asserts per-frame (b, a, c, y) of all 20 cars match within 1e-6. This
   isolates engine fidelity from the neural net.

2. ``test_scores_match_js`` -- the Python ``evaluate()`` for the three fixed
   policies (do-nothing, always-accelerate, decelerate) matches the Node
   reference within <= 0.1 mph.

Both tests shell out to the Node reference harness
(``reference/node_ref/run_ref.js``) to obtain ground truth. They are skipped if
``node`` is not available on PATH.

Run with:
    .venv/bin/python -m pytest tests/test_fidelity.py -v
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

# Make src importable without installation.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from deeptraffic.env import DeepTrafficEnv  # noqa: E402

RUN_REF = os.path.join(ROOT, "reference", "node_ref", "run_ref.js")
NODE = shutil.which("node")

requires_node = pytest.mark.skipif(NODE is None, reason="node not on PATH")


def _node_json(args):
    """Run the Node harness and parse its JSON stdout (last JSON line)."""
    proc = subprocess.run(
        [NODE, RUN_REF] + args,
        capture_output=True,
        text=True,
        timeout=1200,
        cwd=os.path.join(ROOT, "reference", "node_ref"),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"node harness failed (args={args}):\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    # The dump mode writes a single JSON blob to stdout. Eval mode may print
    # progress lines first; take the last non-empty line that parses as JSON.
    out = proc.stdout.strip()
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        for line in reversed(out.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise


# ---------------------------------------------------------------------------
# Trajectory test
# ---------------------------------------------------------------------------
TRAJ_CASES = [
    # (actions, frames, seed_offset)
    ([0], 200, 1),            # do nothing, layout #1
    ([1], 200, 1),            # always accelerate
    ([2], 200, 1),            # always decelerate
    ([1, 1, 2, 0, 4, 3, 1, 0], 300, 1),  # mixed scripted sequence
    ([1], 200, 3),            # accelerate, a different layout
    ([3, 4, 3, 4], 250, 2),   # lane-change churn
]


@requires_node
@pytest.mark.parametrize("actions,frames,seed_offset", TRAJ_CASES)
def test_trajectory_matches_js(actions, frames, seed_offset):
    """Per-frame (b, a, c, y) of all 20 cars must match the JS engine <= 1e-6."""
    js = _node_json(
        [
            "dump",
            "--actions=" + ",".join(str(a) for a in actions),
            f"--frames={frames}",
            f"--seed-offset={seed_offset}",
        ]
    )

    env = DeepTrafficEnv()  # network_basic config (ls=1, pa=10, pb=0, nOther=0)
    py = env.dump_trajectory(actions=actions, frames=frames, seed_offset=seed_offset)

    assert len(js["states"]) == len(py["states"]) == frames

    max_dev = 0.0
    first_div = None
    for fi in range(frames):
        jcars = js["states"][fi]["cars"]
        pcars = py["states"][fi]["cars"]
        for ci in range(20):
            jb, ja, jc, jy, jx = jcars[ci]
            pb, pa, pc, py_, px = pcars[ci]
            # Compare b (lane, int), a (gas), c (safety), y (px). x compared too.
            for jv, pv, name in (
                (jb, pb, "b"), (ja, pa, "a"), (jc, pc, "c"),
                (jy, py_, "y"), (jx, px, "x"),
            ):
                dev = abs(float(jv) - float(pv))
                if dev > max_dev:
                    max_dev = dev
                if dev > 1e-6 and first_div is None:
                    first_div = (fi, ci, name, jv, pv)

    assert first_div is None, (
        f"first divergence frame={first_div[0]} car={first_div[1]} "
        f"field={first_div[2]} js={first_div[3]} py={first_div[4]} "
        f"(max_dev={max_dev:.3e})"
    )
    assert max_dev <= 1e-6, f"max per-frame deviation {max_dev:.3e} > 1e-6"


# ---------------------------------------------------------------------------
# Multi-agent (controlled clones) trajectory fidelity
# ---------------------------------------------------------------------------
# The otherAgents cap is 9 at load (gameopt.js line 1) but raised to 10 in
# reset() -> up to 11 controlled cars (ego + 10). Config = the 74.05 submission.
MULTI_CASES = [
    (10, [1], 200, 1),
    (10, [0], 200, 1),
    (10, [1, 1, 2, 0, 4, 3, 1, 0], 250, 2),
    (4, [1], 200, 1),
    (4, [3, 4, 3, 4], 200, 1),
]


@requires_node
@pytest.mark.parametrize("other_agents,actions,frames,seed_offset", MULTI_CASES)
def test_trajectory_matches_js_multiagent(other_agents, actions, frames, seed_offset):
    """Per-frame (b,a,c,y,x) of all 20 cars must match JS in multi-agent mode."""
    js = _node_json([
        "dump", "--lanes-side=3", "--patches-ahead=40", "--patches-behind=5",
        f"--other-agents={other_agents}",
        "--actions=" + ",".join(str(a) for a in actions),
        f"--frames={frames}", f"--seed-offset={seed_offset}",
    ])
    env = DeepTrafficEnv(lanes_side=3, patches_ahead=40, patches_behind=5,
                         other_agents=other_agents)
    py = env.dump_trajectory(actions=actions, frames=frames, seed_offset=seed_offset)
    max_dev = 0.0
    first_div = None
    for fi in range(frames):
        jcars = js["states"][fi]["cars"]
        pcars = py["states"][fi]["cars"]
        for ci in range(20):
            for k, name in enumerate(("b", "a", "c", "y", "x")):
                dev = abs(float(jcars[ci][k]) - float(pcars[ci][k]))
                if dev > max_dev:
                    max_dev = dev
                if dev > 1e-6 and first_div is None:
                    first_div = (fi, ci, name, jcars[ci][k], pcars[ci][k])
    assert first_div is None, f"divergence {first_div} (max_dev={max_dev:.3e})"


# ---------------------------------------------------------------------------
# Score test
# ---------------------------------------------------------------------------
# Use a modest runs/frames count for CI speed; the relationship between the two
# engines is deterministic so a small sample is a valid fidelity check. Increase
# for the official numbers (runs=500, frames=2000).
SCORE_RUNS = int(os.environ.get("DT_SCORE_RUNS", "30"))
SCORE_FRAMES = int(os.environ.get("DT_SCORE_FRAMES", "2000"))

SCORE_POLICIES = {
    "nop": 0,
    "accel": 1,
    "decel": 2,
}


@requires_node
@pytest.mark.parametrize("name,action", list(SCORE_POLICIES.items()))
def test_scores_match_js(name, action):
    """Python evaluate() median & mean within 0.1 mph of the Node reference."""
    js = _node_json(
        [
            "eval",
            f"--policy={name}",
            f"--runs={SCORE_RUNS}",
            f"--frames={SCORE_FRAMES}",
            "--det=true",
        ]
    )

    env = DeepTrafficEnv()
    res = env.evaluate(
        policy=lambda obs, _a=action: _a,
        runs=SCORE_RUNS,
        frames=SCORE_FRAMES,
        deterministic=True,
    )

    assert abs(res["median"] - js["median"]) <= 0.1, (
        f"{name}: median py={res['median']} js={js['median']}"
    )
    assert abs(res["mean"] - js["mean"]) <= 0.1, (
        f"{name}: mean py={res['mean']} js={js['mean']}"
    )


# ---------------------------------------------------------------------------
# Cheap unit checks that do NOT require node (sanity / smoke).
# ---------------------------------------------------------------------------
def test_num_inputs_network_basic():
    env = DeepTrafficEnv(lanes_side=1, patches_ahead=10, patches_behind=0)
    assert env.num_inputs == 30
    assert env.num_actions == 5


def test_reset_returns_obs_vector():
    env = DeepTrafficEnv()
    obs = env.reset(deterministic=True)
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (env.num_inputs,)
    # empty cells normalize to 100/100 = 1.0; values are in {.. , 1.0}.
    assert obs.max() <= 1.0 + 1e-9


def test_step_decision_level():
    env = DeepTrafficEnv(frames=2000)
    env.reset(deterministic=True)
    obs, reward, done, info = env.step(1)
    assert obs.shape == (env.num_inputs,)
    assert "mph" in info and "speed" in info and "lane" in info and "frame" in info
    assert info["frame"] == 30  # one decision = 30 frames
    assert not done


def test_lcg_matches_reference_sequence():
    """Spot-check the Park-Miller LCG against hand-computed values."""
    from deeptraffic.env import ParkMillerLCG

    g = ParkMillerLCG(1)
    # seed 1 -> first next() = 16807*1 % 2147483647 = 16807
    assert g.next() == 16807
    # second: 16807*16807 % 2147483647 = 282475249
    assert g.next() == 282475249
    g2 = ParkMillerLCG(1)
    q0 = g2.nextq()
    assert abs(q0 - (16807 - 1) / 2147483646) < 1e-15
