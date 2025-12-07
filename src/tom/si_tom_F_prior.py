# src/tom/si_tom_F_prior.py

"""
F-aware prior extension for ToM policy search (Experiment 2).

This module extends the basic ToM step from tom/si_tom.py to include
path flexibility in the policy prior.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from pymdp.agent import Agent

# Import from existing modules
import sys
from pathlib import Path

# Add parent directories to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.tom.si_tom import run_tom_step
from src.metrics.path_flexibility import (
    compute_F_arrays_for_policies,
    compute_q_pi_with_F_prior,
)

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass
class ToMPolicyConfig:
    """Configuration for ToM policy search with optional F-aware prior."""

    # Basic parameters
    horizon: int = 5
    gamma: float = 16.0

    # Empathy + F-prior parameters
    alpha_empathy: float = 1.0  # α: weight on other's EFE
    kappa_prior: float = 0.0  # κ: flexibility prior strength (0 = no F-prior)
    beta_joint_flex: float = 1.0  # β: weight on other's flexibility

    # Flexibility computation
    flex_lambdas: Tuple[float, float, float] = (1.0, 1.0, 1.0)  # (λ_E, λ_R, λ_O)
    shared_outcome_set: Optional[List[int]] = None  # Safe observations for returnability


def run_tom_step_with_F_prior(
    agents: List[Agent],
    o: np.ndarray,
    qs_prev: List[Dict[str, Any]] | None,
    t: int,
    learn: bool,
    agent_num: int,
    B_self: np.ndarray,
    config: ToMPolicyConfig,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Run ToM step with optional F-aware policy prior.

    This function wraps run_tom_step and adds F-aware prior when kappa > 0.

    For Experiment 1 (kappa=0):
        Uses standard q(π) = softmax(-γ [G_i + α·G_j])

    For Experiment 2 (kappa>0):
        Computes F_i(π), F_j(π) for all policies
        Uses q(π) = softmax(-γ J_i) where:
            J_i(π) = G_i + α·G_j - (κ/γ)[F_i + β·F_j]

    Parameters
    ----------
    agents : List[Agent]
        PyMDP agents for ToM
    o : np.ndarray
        Observations [K]
    qs_prev : List[Dict] | None
        Previous beliefs
    t : int
        Timestep
    learn : bool
        Whether to update B matrices
    agent_num : int
        Real agent index
    B_self : np.ndarray
        Self B-matrix for learning
    config : ToMPolicyConfig
        Configuration with α, κ, β parameters

    Returns
    -------
    tom_results : List[Dict]
        ToM results with modified q_pi when kappa > 0
    EFE_arr : np.ndarray
        EFE per policy [K, num_policies]
    Emp_arr : np.ndarray
        Placeholder empowerment array [K, num_policies]
    """
    K = len(agents)

    LOGGER.debug(
        f"Running ToM step with F-prior: t={t}, K={K}, "
        f"α={config.alpha_empathy}, κ={config.kappa_prior}, β={config.beta_joint_flex}"
    )

    # 1) Run standard ToM step to get EFE and initial q_pi
    tom_results, EFE_arr, Emp_arr = run_tom_step(
        agents=agents,
        o=o,
        qs_prev=qs_prev,
        t=t,
        learn=learn,
        agent_num=agent_num,
        B_self=B_self,
    )

    # 2) If kappa == 0, return as-is (Experiment 1)
    if config.kappa_prior == 0.0:
        LOGGER.debug("κ=0: Using standard policy posterior (no F-prior)")
        return tom_results, EFE_arr, Emp_arr

    # 3) If kappa > 0, recompute q_pi with F-aware prior (Experiment 2)
    LOGGER.info(f"κ={config.kappa_prior}: Computing F-aware policy prior")

    # Check if shared_outcome_set is provided
    if config.shared_outcome_set is None:
        LOGGER.warning("shared_outcome_set not provided, using empty set (R=0)")
        config.shared_outcome_set = []

    # For each agent, recompute q_pi with F-prior
    for k in range(K):
        # Get policy IDs (assume agents have same policy library for now)
        num_policies = len(tom_results[k]["G"])
        policy_ids = list(range(num_policies))

        # Get G_i, G_j
        # Here we assume k=0 is focal agent, k=1 is other agent
        # Adjust based on actual setup
        if k == 0:
            focal_idx = 0
            other_idx = 1 if K > 1 else 0
        else:
            focal_idx = k
            other_idx = 0

        G_i = tom_results[focal_idx]["G"]
        G_j = tom_results[other_idx]["G"] if K > 1 else G_i.copy()

        # Get current belief states
        current_qs_i = tom_results[focal_idx]["qs"]
        current_qs_j = tom_results[other_idx]["qs"] if K > 1 else current_qs_i

        # Handle factorized beliefs (take first factor)
        if isinstance(current_qs_i, (list, tuple)):
            current_qs_i = current_qs_i[0]
        if isinstance(current_qs_j, (list, tuple)):
            current_qs_j = current_qs_j[0]

        try:
            # Compute F_i and F_j for all policies
            F_i, F_j = compute_F_arrays_for_policies(
                policies=policy_ids,
                focal_agent_model=agents[focal_idx],
                other_agent_model=agents[other_idx] if K > 1 else agents[focal_idx],
                shared_outcome_set=config.shared_outcome_set,
                horizon=config.horizon,
                lambdas=config.flex_lambdas,
                current_qs_i=current_qs_i,
                current_qs_j=current_qs_j,
            )

            # Recompute q_pi with F-aware prior
            q_pi_new = compute_q_pi_with_F_prior(
                G_i=G_i,
                G_j=G_j,
                F_i=F_i,
                F_j=F_j,
                gamma=config.gamma,
                alpha=config.alpha_empathy,
                kappa=config.kappa_prior,
                beta=config.beta_joint_flex,
            )

            # Update tom_results with new q_pi
            tom_results[k]["q_pi"] = q_pi_new

            LOGGER.debug(
                f"  Agent {k}: Recomputed q_pi with F-prior, "
                f"F_i range=[{F_i.min():.3f}, {F_i.max():.3f}], "
                f"F_j range=[{F_j.min():.3f}, {F_j.max():.3f}]"
            )

        except Exception as e:
            LOGGER.error(f"Failed to compute F-aware prior for agent {k}: {e}")
            LOGGER.warning(f"Falling back to standard q_pi for agent {k}")
            # Keep original q_pi

    return tom_results, EFE_arr, Emp_arr
