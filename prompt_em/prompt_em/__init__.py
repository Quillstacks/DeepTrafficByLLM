"""prompt_em -- optimise an LLM driving prompt as a weighted mixture of atomic
driving heuristics, via a regression/EM loop against a simulator objective.

See README.md for the method and usage; references.md for related literature.
"""

__all__ = ["optimizer", "synthesize", "render", "runner", "experiment", "heuristics"]
