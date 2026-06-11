"""LLM-as-policy for DeepTraffic (11-car mode, independent clones, no coordination).

Pieces:
* ``state_tool``  -- decode the .s() obs into a structured, legible state (same I/O).
* ``heuristic``   -- the driving-convention encoded as code (the ablation baseline).
* ``llm_policy``  -- a local-LLM policy over the same state tool, schema-constrained.
"""
