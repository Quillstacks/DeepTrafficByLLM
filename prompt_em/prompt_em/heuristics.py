"""Load the heuristic set and the experiment config from YAML."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import yaml


def load_heuristics(path: str) -> Tuple[List[Dict], np.ndarray, np.ndarray]:
    """Return (heuristics, init_weights, init_order_priority), both normalised.
    init_order defaults to init_weight (order follows emphasis at the start)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    hs = data["heuristics"]
    for h in hs:
        if "id" not in h or "text" not in h:
            raise ValueError(f"heuristic missing id/text: {h}")
    w = np.array([float(h.get("init_weight", 1.0)) for h in hs], dtype=float)
    p = np.array([float(h.get("init_order", h.get("init_weight", 1.0)))
                  for h in hs], dtype=float)
    return hs, w / w.sum(), p / p.sum()


def load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)
