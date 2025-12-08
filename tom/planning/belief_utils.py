"""
Utility functions for belief updates with defensive error handling.
"""

import numpy as np


def safe_belief_update(obs_dict, A, prior, agent_name="agent", verbose=False):
    """
    Perform safe Bayesian belief update with NaN protection.

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

    Returns
    -------
    qs : np.ndarray
        Posterior belief state (num_states,)
    is_valid : bool
        Whether update was valid (no NaNs)
    """
    obs = int(np.asarray(obs_dict["location_obs"])[0])

    # Check observation is in valid range
    if obs < 0 or obs >= len(A):
        if verbose:
            print(f"  [ERROR] {agent_name}: obs_idx={obs} out of range [0, {len(A)})")
        return prior.copy(), False

    likelihood = A[obs]  # p(o|s)
    unnorm = likelihood * prior  # p(o,s)
    denom = unnorm.sum()

    if denom > 1e-10:
        qs = unnorm / denom
        return qs, True
    else:
        if verbose:
            print(f"  [WARN] {agent_name}: Bayes denom={denom:.6f} for obs={obs}, using prior")
            print(f"    likelihood.sum()={likelihood.sum():.6f}, prior.sum()={prior.sum():.6f}")
        return prior.copy(), False
