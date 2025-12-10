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
from src.config import use_jax

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# Conditional JAX import
_jax_available = False
_jax_warmup_done = False

try:
    from src.metrics.jax_path_flexibility import compute_F_arrays_for_policies_jax
    import jax.numpy as jnp
    _jax_available = True
    LOGGER.info("JAX path flexibility module loaded successfully")
except ImportError as e:
    LOGGER.warning(f"JAX path flexibility module not available: {e}")
    LOGGER.warning("Will use NumPy implementation")


def _warmup_jax_if_needed(agents, config):
    """
    Warm up JAX JIT compilation with a small dummy computation.

    This avoids compilation overhead during the first real computation.
    """
    global _jax_warmup_done

    if _jax_warmup_done or not _jax_available or not use_jax():
        return

    try:
        LOGGER.info("Warming up JAX JIT compilation...")

        # Extract model matrices
        agent = agents[0]
        A = jnp.array(agent.A[0] if isinstance(agent.A, list) else agent.A)
        B_raw = jnp.array(agent.B[0] if isinstance(agent.B, list) else agent.B)
        D = jnp.array(agent.D[0] if isinstance(agent.D, list) else agent.D)

        # Normalize B to [num_actions, num_states, num_states]
        if B_raw.shape[0] == B_raw.shape[1] and B_raw.shape[2] < B_raw.shape[0]:
            B = jnp.transpose(B_raw, (2, 0, 1))
        else:
            B = B_raw

        # Get a few dummy policies for warmup
        num_warmup_policies = min(5, len(agent.policies))
        dummy_policies = jnp.array(agent.policies[:num_warmup_policies], dtype=jnp.int32)

        # Dummy warmup call
        _ = compute_F_arrays_for_policies_jax(
            policies=dummy_policies,
            A_i=A, B_i=B, D_i=D,
            A_j=A, B_j=B, D_j=D,
            shared_outcome_set=config.shared_outcome_set or [0],
            lambdas=config.flex_lambdas,
        )

        _jax_warmup_done = True
        LOGGER.info("JAX warmup complete!")

    except Exception as e:
        LOGGER.warning(f"JAX warmup failed: {e}")
        LOGGER.warning("Will compile on first use")


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

    # Warm up JAX if enabled (only happens once)
    if use_jax() and _jax_available:
        _warmup_jax_if_needed(agents, config)

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
            # Use JAX if enabled and available, otherwise fall back to NumPy
            if use_jax() and _jax_available:
                # JAX path (60-100x faster!)
                LOGGER.debug(f"  Using JAX-accelerated flexibility computation")

                # Extract model matrices
                focal_agent = agents[focal_idx]
                other_agent = agents[other_idx] if K > 1 else agents[focal_idx]

                A_i = jnp.array(focal_agent.A[0] if isinstance(focal_agent.A, list) else focal_agent.A)
                B_i_raw = jnp.array(focal_agent.B[0] if isinstance(focal_agent.B, list) else focal_agent.B)
                D_i = jnp.array(focal_agent.D[0] if isinstance(focal_agent.D, list) else focal_agent.D)

                A_j = jnp.array(other_agent.A[0] if isinstance(other_agent.A, list) else other_agent.A)
                B_j_raw = jnp.array(other_agent.B[0] if isinstance(other_agent.B, list) else other_agent.B)
                D_j = jnp.array(other_agent.D[0] if isinstance(other_agent.D, list) else other_agent.D)

                # Normalize B matrices to [num_actions, num_states, num_states]
                if B_i_raw.shape[0] == B_i_raw.shape[1] and B_i_raw.shape[2] < B_i_raw.shape[0]:
                    B_i = jnp.transpose(B_i_raw, (2, 0, 1))
                else:
                    B_i = B_i_raw

                if B_j_raw.shape[0] == B_j_raw.shape[1] and B_j_raw.shape[2] < B_j_raw.shape[0]:
                    B_j = jnp.transpose(B_j_raw, (2, 0, 1))
                else:
                    B_j = B_j_raw

                # Get policies as JAX array
                policies_jax = jnp.array(focal_agent.policies, dtype=jnp.int32)

                # Convert current beliefs to JAX
                current_qs_i_jax = jnp.array(current_qs_i) if current_qs_i is not None else None
                current_qs_j_jax = jnp.array(current_qs_j) if current_qs_j is not None else None

                # JAX-accelerated computation
                F_i, F_j = compute_F_arrays_for_policies_jax(
                    policies=policies_jax,
                    A_i=A_i, B_i=B_i, D_i=D_i,
                    A_j=A_j, B_j=B_j, D_j=D_j,
                    shared_outcome_set=config.shared_outcome_set,
                    lambdas=config.flex_lambdas,
                    current_qs_i=current_qs_i_jax,
                    current_qs_j=current_qs_j_jax,
                )
                # F_i, F_j are already NumPy arrays (converted inside JAX function)

            else:
                # NumPy path (original implementation)
                LOGGER.debug(f"  Using NumPy flexibility computation")

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
