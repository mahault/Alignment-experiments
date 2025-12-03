# agents/empowerment.py

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np


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

    # If degenerate, return 0
    if num_actions <= 1 or num_obs <= 1:
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
        return 0.0

    T = len(transition_logits_seq)
    if weights is None:
        weights = [1.0 / T] * T

    emps = []
    for logits in transition_logits_seq:
        emps.append(estimate_empowerment_one_step(logits))

    emps = np.array(emps)
    weights = np.array(weights)
    return float((weights * emps).sum())
