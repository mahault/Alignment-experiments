# src/common/types.py

"""
Shared data types and configurations for path flexibility experiments.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class FlexibilityMetrics:
    """Path flexibility metrics for a single policy."""

    policy_idx: int
    empowerment: float          # E(π)
    returnability: float        # R(π)
    overlap: float              # O(π)
    flexibility: float          # F(π) = λE·E + λR·R + λO·O

    # Raw components
    empowerment_per_step: Optional[List[float]] = None
    returnability_per_step: Optional[List[float]] = None
    overlap_per_step: Optional[List[float]] = None


@dataclass
class PolicyMetrics:
    """Complete metrics for a policy including EFE and flexibility."""

    policy_idx: int

    # Expected Free Energy
    G_i: float                  # Agent i's EFE
    G_j: float                  # Agent j's EFE
    G_joint: float              # G_i + G_j

    # Path Flexibility
    F_i: float                  # Agent i's flexibility
    F_j: float                  # Agent j's flexibility
    F_joint: float              # F_i + F_j

    # Detailed flexibility breakdown
    flex_i: Optional[FlexibilityMetrics] = None
    flex_j: Optional[FlexibilityMetrics] = None

    # Decision variable (for Exp 2)
    J_i: Optional[float] = None  # J_i = G_i + α·G_j - (κ/γ)·(F_i + β·F_j)

    # Policy posterior
    q_pi: Optional[float] = None


@dataclass
class EpisodeMetrics:
    """Aggregated metrics for an entire episode."""

    episode_id: int

    # All policies considered during the episode
    all_policies: List[PolicyMetrics]

    # Selected policy at each timestep
    selected_policies: List[int]

    # Behavioral outcomes
    collision: bool
    success_agent_i: bool
    success_agent_j: bool
    num_timesteps: int

    # Trajectories
    positions_i: List[tuple]
    positions_j: List[tuple]
    actions_i: List[int]
    actions_j: List[int]


@dataclass
class ExperimentConfig:
    """Configuration for path flexibility experiments."""

    # Environment
    env_height: int = 7
    env_width: int = 11
    slip_prob: float = 0.05
    max_steps: int = 50

    # Agent parameters
    alpha: float = 0.5          # Empathy weight
    gamma: float = 16.0         # Precision (inverse temperature)
    policy_len: int = 5         # Planning horizon

    # Flexibility computation
    lambda_E: float = 1.0       # Weight on empowerment
    lambda_R: float = 1.0       # Weight on returnability
    lambda_O: float = 1.0       # Weight on overlap

    # Shared outcome set for returnability
    shared_outcomes: Optional[List[int]] = None

    # Experiment 2 parameters (F-prior)
    kappa: float = 0.0          # Flexibility prior strength
    beta: float = 1.0           # Weight on other's flexibility

    # Simulation
    num_episodes: int = 100
    seed: int = 42


@dataclass
class ToMState:
    """State of Theory of Mind tree at a timestep."""

    tree: Any                   # ToM tree structure
    other_tree: Any             # Other agent's tree (from ToM perspective)
    qs_i: np.ndarray           # Agent i's belief state
    qs_j: np.ndarray           # Agent j's belief state
    policies_i: List[Any]       # Agent i's candidate policies
    policies_j: List[Any]       # Agent j's candidate policies
