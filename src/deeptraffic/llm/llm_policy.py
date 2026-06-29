"""Local-LLM policy (qwen2.5:3b): each car independently follows a fixed SYSTEM
prompt of driving rules, reasoning over a CLEAN rendering of the obs (no online
help — the per-call message contains only a faithful description of the same
state the DQN sees; all intelligence lives in the system prompt).

No inter-car communication. The only "coordination" is that all cars share the
same convention, so they self-organize. Chain-of-thought is allowed (reason-first
schema). A discretized decision cache memoizes recurring states for tractable eval.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Callable, Optional

import numpy as np

from .state_tool import decode, DecodedState, ACTION_TO_ID

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen2.5:3b"

# Driving rules go ENTIRELY in the system prompt (the thing we optimize).
SYSTEM = (
    "You drive ONE car on a 7-lane highway and want the highest AVERAGE SPEED. "
    "You cannot pass through cars, and you already go as fast as the car right ahead "
    "of you allows.\n"
    "KEY: a car ahead going FAST (near 80 mph) is NOT a problem - you couldn't go "
    "faster anyway, so just ACCELERATE and stay in your lane. Only a SLOWER car close "
    "ahead (under ~70 mph) actually holds you up.\n"
    "When a slow car is close ahead, OVERTAKE: switch to an adjacent lane that is clear "
    "or whose nearest car is fast/far - NEVER into a lane that also has a slow car close "
    "ahead. Prefer the centre-ward lane to pass.\n"
    "Otherwise just accelerate and keep your lane; do not change lanes without a clear, "
    "lasting speed gain.\n"
    "Think in one short sentence, then choose one action."
)

# reason-first (chain-of-thought); 'accelerate' placed LAST to avoid first-option bias.
ACTION_ENUM = ["left", "right", "decelerate", "maintain", "accelerate"]
SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "action": {"type": "string", "enum": ACTION_ENUM},
    },
    "required": ["reason", "action"],
}


FAST_MPH = 72.0   # at/above this, a car ahead is not really blocking you
NEAR = 14         # patches: "close" ahead
REAR_NEAR = 5     # patches: "close" behind
REAR_FAST = 58.0  # mph: a behind car this fast is "approaching"


def _describe_ahead(ln) -> str:
    """Semantic label for a lane's nearest car ahead (derived purely from the obs)."""
    if ln is None or not ln.on_road:
        return "no lane there"
    if ln.gap_ahead is None:
        return "OPEN (clear far ahead)"
    spd = ln.speed_ahead or 0.0
    if spd >= FAST_MPH:
        return f"fast car {ln.gap_ahead}p ahead ({spd:.0f} mph) - not blocking"
    label = "SLOW car CLOSE" if ln.gap_ahead < NEAR else "slow car far"
    return f"{label} {ln.gap_ahead}p ahead ({spd:.0f} mph)"


def _format_user(ds: DecodedState) -> str:
    """SEMANTIC state: same information as the obs, translated into the features a
    driver cares about (slow/fast, blocked, approaching-from-behind). Salient
    features only -- it does NOT suggest an action; the model still decides."""
    ego = ds.ego
    left, right = ds.lane_by_rel(-1), ds.lane_by_rel(+1)
    blocked = (ego.gap_ahead is not None and (ego.speed_ahead or 99.0) < FAST_MPH
               and ego.gap_ahead < NEAR)
    status = "BLOCKED by a slow car ahead" if blocked else "your lane is clear/fast (NOT blocked)"

    def rear(ln):
        if ln is None or not ln.on_road or ln.gap_behind is None:
            return None
        if (ln.speed_behind or 0.0) >= REAR_FAST and ln.gap_behind <= REAR_NEAR:
            return f"{ln.gap_behind}p behind ({ln.speed_behind:.0f} mph, approaching)"
        return None

    lines = [
        f"You are in lane {ds.ego_lane} of 7. STATUS: {status}.",
        f"Your lane: {_describe_ahead(ego)}.",
        f"LEFT lane: {_describe_ahead(left)}.",
        f"RIGHT lane: {_describe_ahead(right)}.",
    ]
    rl, rr = rear(left), rear(right)
    if rl or rr:
        bits = []
        if rl:
            bits.append(f"fast car in LEFT lane {rl}")
        if rr:
            bits.append(f"fast car in RIGHT lane {rr}")
        lines.append("Watch behind: " + "; ".join(bits) + ".")
    lines.append("Available actions: " + ", ".join(ds.legal_actions()))
    return "\n".join(lines)


def _query(model: str, system: str, user: str, timeout: float = 40.0,
           num_predict: int = 96, schema: dict = SCHEMA) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": schema,
        "options": {"num_predict": num_predict, "temperature": 0.0},
    }
    if "qwen3" in model:
        # qwen3 emits <think> blocks by default; the structured 'reason' field
        # is our CoT budget -- disable native thinking for comparable latency.
        payload["think"] = False
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r.get("message", {}).get("content", "")


_ACTION_RE = None


def parse_action(content: str) -> Optional[str]:
    """Extract the chosen action from the model output. Strict JSON first; if
    the JSON was truncated by the token budget, fall back to scanning for the
    "action": "<enum>" pair (the schema guarantees the field name)."""
    global _ACTION_RE
    try:
        act = json.loads(content).get("action")
        if act in ACTION_ENUM:
            return act
    except Exception:
        pass
    if _ACTION_RE is None:
        import re
        _ACTION_RE = re.compile(r'"action"\s*:\s*"(%s)"?' % "|".join(ACTION_ENUM))
    m = _ACTION_RE.search(content)
    return m.group(1) if m else None


class LLMPolicy:
    """callable(obs) -> engine action id (0..4)."""

    def __init__(self, lanes_side: int, patches_ahead: int, patches_behind: int,
                 model: str = MODEL, cache=False,
                 system: str = SYSTEM, format_user=None,
                 num_predict: int = 96, schema: dict = SCHEMA):
        """cache=False: query every decision (the honest pure benchmark).
        cache='exact': memoize on the EXACT user-message text. At temperature=0
            the model is a deterministic function of the prompt, so this returns
            bit-identical decisions to cache=False -- it only avoids re-asking
            the same question twice. Safe for evaluation.
        cache=True: legacy lossy signature (gaps only, no speeds) -- fast but
            NOT faithful to the pure policy; analysis only."""
        self.ls, self.pa, self.pb = lanes_side, patches_ahead, patches_behind
        self.model = model
        self.system = system
        self.format_user = format_user or _format_user
        self.cache_mode = cache
        self.cache: Optional[dict] = {} if cache else None
        self.num_predict = num_predict
        self.schema = schema
        self.calls = 0
        self.cache_hits = 0
        self.parse_fail = 0
        self._consec_fail = 0

    def _signature(self, ds: DecodedState):
        parts = [ds.ego_lane]
        for ln in ds.lanes:
            if not ln.on_road:
                parts.append("x")
            elif ln.gap_ahead is None:
                parts.append("c")
            else:
                parts.append(min(ln.gap_ahead // 4, 9))
        return tuple(parts)

    def __call__(self, obs: np.ndarray) -> int:
        ds = decode(obs, self.ls, self.pa, self.pb)
        legal = set(ds.legal_actions())
        user = self.format_user(ds)
        if self.cache_mode == "exact":
            sig = user
        elif self.cache is not None:
            sig = self._signature(ds)
        else:
            sig = None
        if sig is not None and sig in self.cache:
            self.cache_hits += 1
            return self.cache[sig]
        ok = True
        try:
            content = _query(self.model, self.system, user,
                             num_predict=self.num_predict, schema=self.schema)
            self.calls += 1
            act = parse_action(content)
            if act is None:
                self.parse_fail += 1
                act = "maintain"
                ok = False
        except Exception:
            self.parse_fail += 1
            act = "maintain"
            ok = False
        self._consec_fail = 0 if ok else self._consec_fail + 1
        if self._consec_fail >= 25:
            # 25 failures in a row means the LLM server is down, not that the
            # model is misbehaving -- abort loudly instead of silently scoring
            # the do-nothing floor with "maintain" fallbacks.
            raise RuntimeError(
                "LLM backend unreachable/failing (25 consecutive failures); "
                "is `ollama serve` running with %s pulled?" % self.model)
        if act not in legal:
            act = "maintain"
        aid = ACTION_TO_ID.get(act, 0)
        if sig is not None and ok:
            self.cache[sig] = aid
        return aid

    def stats(self) -> dict:
        total = self.calls + self.cache_hits
        return {"llm_calls": self.calls, "cache_hits": self.cache_hits,
                "cache_hit_rate": round(self.cache_hits / max(1, total), 3),
                "parse_fail": self.parse_fail,
                "cache_size": len(self.cache) if self.cache is not None else 0}


def make_policy(lanes_side: int, patches_ahead: int, patches_behind: int,
                model: str = MODEL, cache: bool = False,
                system: str = SYSTEM, format_user=None) -> Callable[[np.ndarray], int]:
    return LLMPolicy(lanes_side, patches_ahead, patches_behind, model, cache, system, format_user)
