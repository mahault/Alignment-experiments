"""
Path flexibility metrics module.

Provides both NumPy (original) and JAX-accelerated implementations.
"""

# NumPy implementations (original)
from .empowerment import (
    estimate_empowerment_one_step,
    estimate_empowerment_over_rollout,
)

from .path_flexibility import (
    compute_empowerment_along_rollout,
    compute_returnability_from_rollout,
    compute_overlap_from_two_rollouts,
    compute_F_arrays_for_policies,
    compute_q_pi_with_F_prior,
    rollout_beliefs_and_obs,
)

# JAX implementations (optional, only if JAX is available)
try:
    from .jax_path_flexibility import (
        estimate_empowerment_one_step_jax,
        compute_returnability_jax,
        compute_overlap_jax,
        rollout_beliefs_and_obs_jax,
        compute_empowerment_along_rollout_jax,
        compute_F_arrays_for_policies_jax,
    )
    __has_jax__ = True
except ImportError:
    __has_jax__ = False

__all__ = [
    # NumPy
    "estimate_empowerment_one_step",
    "estimate_empowerment_over_rollout",
    "compute_empowerment_along_rollout",
    "compute_returnability_from_rollout",
    "compute_overlap_from_two_rollouts",
    "compute_F_arrays_for_policies",
    "compute_q_pi_with_F_prior",
    "rollout_beliefs_and_obs",
    # JAX (if available)
    "estimate_empowerment_one_step_jax",
    "compute_returnability_jax",
    "compute_overlap_jax",
    "rollout_beliefs_and_obs_jax",
    "compute_empowerment_along_rollout_jax",
    "compute_F_arrays_for_policies_jax",
]
