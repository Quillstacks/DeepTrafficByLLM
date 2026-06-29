"""Turn a weighted set of heuristics into one driving SYSTEM prompt.

Two modes (config: synthesis.mode):
  * template -- deterministic 1-to-1 assembly. Heuristics are sorted by weight,
                grouped into emphasis tiers, and concatenated as prose. No LLM,
                fully reproducible. This is also the fallback if the LLM
                synthesiser fails.
  * llm      -- an LLM rewrites the weighted heuristics into one fluent prompt,
                told to stress the high-weight rules and drop the low-weight
                ones. Adds nondeterminism (mitigated by temperature 0).

Both return the SYSTEM-prompt string the driving policy will use. A fixed
preamble (the task framing) and a fixed closing instruction (reason then act)
bracket the synthesised rules so every prompt is well-formed.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Dict, List

import numpy as np

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

PREAMBLE = (
    "You drive ONE car on a 7-lane highway; your score is your AVERAGE SPEED "
    "(top speed 80 mph). You auto-drive as fast as the road right ahead allows. "
    "Follow these driving rules:"
)
CLOSING = "Think in one short sentence, then choose one action."


def _kept(heuristics: List[Dict], weights: np.ndarray, order, drop_below: float):
    """Keep heuristics whose weight share clears ``drop_below``, placed in the
    given ORDER (a list of heuristic indices, highest-priority first). Order
    (sequence) is decoupled from weight (emphasis): order comes from ``order``,
    the emphasis tier from ``weights``."""
    if order is None:
        order = list(np.argsort(-weights))
    kept = [(heuristics[i], float(weights[i])) for i in order
            if weights[i] >= drop_below]
    if not kept:                       # never emit an empty prompt
        i = int(np.argmax(weights))
        kept = [(heuristics[i], float(weights[i]))]
    return kept


def template_synthesis(heuristics, weights, order=None, drop_below=0.04) -> str:
    kept = _kept(heuristics, weights, order, drop_below)
    hi = max(w for _, w in kept)
    lines = [PREAMBLE]
    for h, w in kept:
        share = w / hi
        if share >= 0.85:
            tag = "ALWAYS"
        elif share >= 0.5:
            tag = "Important"
        else:
            tag = "Also"
        lines.append(f"- ({tag}) {h['text'].strip()}")
    lines.append(CLOSING)
    return "\n".join(lines)


def _freedom_instructions(freedom: float) -> str:
    """How much latitude the synthesizer gets, scheduled by ``freedom`` in [0,1].
    At 0 it faithfully blends the given sentences in the given order; as freedom
    rises it may rephrase, sharpen, and add clarifying detail, exploring the
    WORDING space around the fixed rules."""
    if freedom < 0.15:
        return ("KEEP THE GIVEN ORDER (state rule 1 first, then 2, ...) and keep "
                "the wording faithful. Do NOT add new information.")
    if freedom < 0.5:
        return ("Keep the given order. You MAY lightly rephrase for clarity and "
                "force, but preserve each rule's meaning and add no new rules.")
    if freedom < 0.8:
        return ("Keep the overall order. You MAY rephrase freely, sharpen the "
                "wording, merge related rules, and add brief clarifying detail, "
                "as long as you preserve the rules' intent.")
    return ("Rewrite this into the MOST EFFECTIVE driving system prompt you can. "
            "You may rephrase, reorganise locally, and add clarifying detail; "
            "preserve the intent of the high-weight rules above all.")


def _llm_rewrite(heuristics, weights, order, model, temperature, drop_below,
                 freedom=0.0, variant=0) -> str:
    kept = _kept(heuristics, weights, order, drop_below)
    total = sum(w for _, w in kept)
    bullet = "\n".join(
        f"{k+1}. weight {round(100 * w / total)}%: {h['text'].strip()}"
        for k, (h, w) in enumerate(kept))
    sys = (
        "You are a prompt engineer. You will be given a NUMBERED, ORDERED list "
        "of driving rules, each with a WEIGHT (its importance). Write ONE concise "
        "system prompt for a driver that follows these rules, stressing the "
        "high-weight rules while only briefly mentioning (or omitting) low-weight "
        "ones. Keep it to a few sentences of plain driving advice. Do NOT mention "
        "weights or numbers. Output ONLY the prompt text.\n"
        + _freedom_instructions(freedom))
    user = ("Driving rules in order, with weights:\n" + bullet +
            "\n\nWrite the single driving system prompt now.")
    if variant:                     # vary candidates at temperature > 0
        user += f"\n(Draft variant {variant}.)"
    # freedom raises the sampling temperature -> more wording exploration
    temp = max(temperature, round(0.9 * float(freedom), 3))
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": sys},
                     {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temp, "num_predict": 320},
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=120))
    body = (r.get("message", {}).get("content") or "").strip()
    if not body:
        raise RuntimeError("empty synthesis")
    # bracket the model's rules with the fixed framing so it is always well-formed
    return f"{PREAMBLE}\n{body}\n{CLOSING}"


def synthesize(heuristics, weights, order=None, *, mode="llm",
               model="qwen2.5:3b", temperature=0.0, drop_below=0.04,
               freedom=0.0, variant=0) -> str:
    """Return the SYSTEM prompt for the given heuristic weights and ORDER.

    ``order`` is a list of heuristic indices (highest-priority first); if None,
    order falls back to descending weight. ``freedom`` in [0,1] is the
    synthesizer's latitude to explore WORDING (0 = faithful blend; higher =
    rephrase/sharpen/elaborate, with a raised sampling temperature). ``variant``
    diversifies candidates when freedom > 0."""
    weights = np.asarray(weights, float)
    if mode == "template":
        return template_synthesis(heuristics, weights, order, drop_below)
    try:
        return _llm_rewrite(heuristics, weights, order, model, temperature,
                            drop_below, freedom=freedom, variant=variant)
    except Exception:
        # robust fallback: never let a synthesis hiccup kill the experiment
        return template_synthesis(heuristics, weights, order, drop_below)
