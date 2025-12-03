# agents/empowerment.py

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def estimate_empowerment_one_step(
    transition_logits: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute a simple one-step empowerment estimate:
        Emp = max_{p(a)} I(A; O_next)
    using a discrete action set and discrete observations.

    Parameters
    ----------
    transition_logits : np.ndarray
        Array of shape [num_actions, num_observations]
        representing p(o_next | a) up to normalization; computed from the agent's A/B
        under its current beliefs.

        This is a *local*, subjective approximation to empowerment:
        Emp ≈ max_{p(a)} I(A; O_next).
    eps : float
        Small constant for numerical stability

    Returns
    -------
    float
        Empowerment value (mutual information between actions and observations)
    """
    # Normalize over observations
    probs = transition_logits / (transition_logits.sum(axis=1, keepdims=True) + eps)
    num_actions, num_obs = probs.shape

    LOGGER.debug(f"Computing one-step empowerment: num_actions={num_actions}, num_obs={num_obs}")

    # If degenerate, return 0
    if num_actions <= 1 or num_obs <= 1:
        LOGGER.warning(f"Degenerate case: num_actions={num_actions}, num_obs={num_obs}. Returning 0.")
        return 0.0

    # Brute-force over a fixed small family of p(a).
    # For now: uniform prior over actions.
    p_a = np.full(num_actions, 1.0 / num_actions)

    # Compute p(o_next) = sum_a p(a)*p(o|a)
    p_o = (p_a[:, None] * probs).sum(axis=0)

    # I(A;O) = sum_{a,o} p(a,o) log( p(o|a) / p(o) )
    p_ao = p_a[:, None] * probs
    ratio = probs / (p_o[None, :] + eps)
    mi = (p_ao * (np.log(ratio + eps))).sum()

    LOGGER.debug(f"Computed empowerment (MI): {mi:.4f}")
    return float(mi)


def estimate_empowerment_over_rollout(
    transition_logits_seq: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> float:
    """
    Given a sequence of local transition logits arrays (one per timestep),
    estimate average empowerment over the rollout.

    Parameters
    ----------
    transition_logits_seq : List[np.ndarray]
        List of [num_actions, num_obs] arrays, one per timestep.
    weights : Optional[List[float]]
        Optional list of weights per timestep (sums to 1).

    Returns
    -------
    float
        Weighted average empowerment over the rollout
    """
    if not transition_logits_seq:
        LOGGER.warning("Empty transition_logits_seq provided. Returning 0.")
        return 0.0

    T = len(transition_logits_seq)
    LOGGER.debug(f"Computing empowerment over rollout: T={T} timesteps")

    if weights is None:
        weights = [1.0 / T] * T

    emps = []
    for t, logits in enumerate(transition_logits_seq):
        emp = estimate_empowerment_one_step(logits)
        emps.append(emp)
        LOGGER.debug(f"  t={t}: empowerment={emp:.4f}")

    emps = np.array(emps)
    weights = np.array(weights)
    weighted_emp = float((weights * emps).sum())

    LOGGER.info(f"Weighted average empowerment over rollout: {weighted_emp:.4f}")
    return weighted_emp
