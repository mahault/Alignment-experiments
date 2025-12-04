"""
Environment modules for path flexibility experiments.

This package contains:
- lava_corridor: LavaCorridorEnv and generative model builder
- rollout_lava: Multi-agent rollout functions for Experiments 1 and 2
"""

from .lava_corridor import (
    LavaCorridorEnv,
    LavaCorridorConfig,
    build_generative_model_for_env,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    STAY,
    ACTIONS,
    ACTION_NAMES,
)

from .rollout_lava import (
    rollout_multi_agent_lava,
    rollout_exp1,
    rollout_exp2,
)

__all__ = [
    # Environment
    "LavaCorridorEnv",
    "LavaCorridorConfig",
    "build_generative_model_for_env",
    # Actions
    "UP",
    "DOWN",
    "LEFT",
    "RIGHT",
    "STAY",
    "ACTIONS",
    "ACTION_NAMES",
    # Rollout functions
    "rollout_multi_agent_lava",
    "rollout_exp1",
    "rollout_exp2",
]
