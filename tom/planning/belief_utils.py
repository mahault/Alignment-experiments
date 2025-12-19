"""
Utility functions for belief updates with defensive error handling.

In active inference, belief updates compute:
- Posterior belief q(s|o) via Bayes rule
- VFE (Variational Free Energy) = -log p(o) = surprise at observation
"""

import numpy as np
from typing import Tuple, Union


def safe_belief_update(
    obs_dict, A, prior, agent_name="agent", verbose=False, return_vfe=False
) -> Union[Tuple[np.ndarray, bool], Tuple[np.ndarray, bool, float]]:
    """
    Perform safe Bayesian belief update with NaN protection.

    In active inference, this computes:
    - Posterior: q(s|o) = p(o|s) * p(s) / p(o)
    - VFE: -log p(o) where p(o) = sum_s p(o|s) * p(s)

    VFE measures "surprise" - how unexpected the observation is given beliefs.
    Lower VFE = better model fit = less surprise.

    Parameters
    ----------
    obs_dict : dict
        Observation dictionary with "location_obs" key
    A : np.ndarray
        Observation model (num_obs, num_states)
    prior : np.ndarray
        Prior belief state (num_states,)
    agent_name : str
        Agent name for logging
    verbose : bool
        Whether to print warnings
    return_vfe : bool
        If True, also return VFE (variational free energy)

    Returns
    -------
    qs : np.ndarray
        Posterior belief state (num_states,)
    is_valid : bool
        Whether update was valid (no NaNs)
    vfe : float (only if return_vfe=True)
        Variational free energy = -log p(o) = surprise
    """
    obs = int(np.asarray(obs_dict["location_obs"])[0])

    # Check observation is in valid range
    if obs < 0 or obs >= len(A):
        if verbose:
            print(f"  [ERROR] {agent_name}: obs_idx={obs} out of range [0, {len(A)})")
        if return_vfe:
            return prior.copy(), False, float('inf')  # Infinite surprise for invalid obs
        return prior.copy(), False

    likelihood = A[obs]  # p(o|s)
    unnorm = likelihood * prior  # p(o,s)
    denom = unnorm.sum()  # p(o) = model evidence

    if denom > 1e-10:
        qs = unnorm / denom
        # VFE = -log p(o) = surprise at observation
        vfe = -np.log(denom)
        if return_vfe:
            return qs, True, float(vfe)
        return qs, True
    else:
        if verbose:
            print(f"  [WARN] {agent_name}: Bayes denom={denom:.6f} for obs={obs}, using prior")
            print(f"    likelihood.sum()={likelihood.sum():.6f}, prior.sum()={prior.sum():.6f}")
        if return_vfe:
            return prior.copy(), False, float('inf')  # Infinite surprise
        return prior.copy(), False


def compute_vfe(qs: np.ndarray, A: np.ndarray, obs: int, eps: float = 1e-16) -> float:
    """
    Compute VFE for a specific observation given beliefs.

    VFE = -log p(o) where p(o) = sum_s p(o|s) * q(s)

    Parameters
    ----------
    qs : np.ndarray
        Current belief over states
    A : np.ndarray
        Observation model (num_obs, num_states)
    obs : int
        Observed state index
    eps : float
        Numerical floor

    Returns
    -------
    vfe : float
        Variational free energy (surprise)
    """
    # p(o) = sum_s p(o|s) * q(s)
    p_o = np.dot(A[obs], qs)
    p_o = max(p_o, eps)
    return -np.log(p_o)


