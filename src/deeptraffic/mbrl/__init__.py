"""Model-based RL (MBRL) for the faithful DeepTraffic environment.

See :mod:`deeptraffic.mbrl.core` for the implementation: a learned dynamics
model + a model-free Q-function, with two policies (reactive argmax-Q and
receding-horizon MPC) built from the same data so their difference isolates the
contribution of lookahead. Planning happens only in the learned model.
"""

from .core import Config, Agent, train

__all__ = ["Config", "Agent", "train"]
