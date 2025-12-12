"""
JAX-accelerated empathic active inference planner for Lava corridor.

This module provides a JAX reimplementation of compute_empathic_G from si_empathy_lava.py,
replacing triple nested Python loops with compiled JAX operations:

- vmap over i policies (125-625 policies)
- lax.scan over horizon (3-5 timesteps)
- vmap over j actions (5 primitive actions)

Expected speedup: 50-100x

Performance comparison:
- NumPy (horizon=3, 125 policies): ~45-60s
- JAX (horizon=3, 125 policies): ~0.5-1s → 50-100x faster!
"""

import jax
import jax.numpy as jnp
from jax import lax, vmap
from typing import Tuple
import numpy as np


# =============================================================================
# LEVEL 1: Low-level JAX primitives (JIT-compiled)
# =============================================================================

@jax.jit
def propagate_belief_jax(
    qs: jnp.ndarray,
    B: jnp.ndarray,
    action: int,
    qs_other: jnp.ndarray = None,
    eps: float = 1e-16,
) -> jnp.ndarray:
    """
    JAX-compiled belief propagation: qs_next = B @ qs

    Supports both 3D [s', s, a] and 4D [s', s, s_other, a] B matrices.

    Parameters
    ----------
    qs : jnp.ndarray
        Current belief [num_states]
    B : jnp.ndarray
        Transition model [s', s, a] or [s', s, s_other, a]
    action : int
        Action index
    qs_other : jnp.ndarray, optional
        Belief over other agent's position (required for 4D B)
    eps : float
        Numerical stability constant

    Returns
    -------
    qs_next : jnp.ndarray
        Next belief [num_states]
    """
    # Handle 3D vs 4D B matrix
    if B.ndim == 3:
        # 3D: B[s', s, a] → simple matrix-vector product
        qs_next = B[:, :, action] @ qs
    elif B.ndim == 4:
        # 4D: B[s', s, s_other, a] → marginalize over other's position
        # qs_next[s'] = Σ_{s,s_other} B[s', s, s_other, a] * qs[s] * qs_other[s_other]
        B_action = B[:, :, :, action]  # [s', s, s_other]
        # Vectorized computation using einsum
        qs_next = jnp.einsum('ijk,j,k->i', B_action, qs, qs_other)
    else:
        raise ValueError(f"B must be 3D or 4D, got shape {B.shape}")

    # Normalize
    qs_next = qs_next / (qs_next.sum() + eps)
    return qs_next


@jax.jit
def expected_pragmatic_utility_jax(
    qs_self_current: jnp.ndarray,
    qs_other_current: jnp.ndarray,
    qs_self_next: jnp.ndarray,
    qs_other_next: jnp.ndarray,
    action_self: int,
    action_other: int,
    A_loc: jnp.ndarray,
    C_loc: jnp.ndarray,
    A_edge: jnp.ndarray,
    C_edge: jnp.ndarray,
    A_cell_collision: jnp.ndarray,
    C_cell_collision: jnp.ndarray,
    A_edge_collision: jnp.ndarray,
    C_edge_collision: jnp.ndarray,
) -> float:
    """
    JAX-compiled pragmatic utility over ALL observation modalities.

    Modalities:
    1. location_obs: E[C_loc(o)] - uses NEXT state
    2. edge_obs: E[C_edge(e)] - uses NEXT state
    3. cell_collision_obs: Marginalize A over joint NEXT state beliefs
    4. edge_collision_obs: Marginalize A over joint CURRENT state beliefs + actions

    NOTE: Edge collision depends on CURRENT states + actions (which edges are traversed),
    while cell collision depends on NEXT states (where agents end up).

    Parameters
    ----------
    qs_self_current : jnp.ndarray
        Belief over own CURRENT position [num_states]
    qs_other_current : jnp.ndarray
        Belief over other's CURRENT position [num_states]
    qs_self_next : jnp.ndarray
        Belief over own NEXT position [num_states]
    qs_other_next : jnp.ndarray
        Belief over other's NEXT position [num_states]
    action_self : int
        Own action
    action_other : int
        Other's action
    A_loc : jnp.ndarray
        Location observation model [num_obs, num_states]
    C_loc : jnp.ndarray
        Location preferences [num_obs]
    A_edge : jnp.ndarray
        Edge observation model [num_edges + 1, num_states, num_actions]
    C_edge : jnp.ndarray
        Edge preferences [num_edges + 1]
    A_cell_collision : jnp.ndarray
        Cell collision observation model [2, num_states, num_states]
    C_cell_collision : jnp.ndarray
        Cell collision preferences [2]
    A_edge_collision : jnp.ndarray
        Edge collision observation model [2, num_states, num_states, num_actions, num_actions]
    C_edge_collision : jnp.ndarray
        Edge collision preferences [2]

    Returns
    -------
    total_pragmatic : float
        Expected utility
    """
    # 1. Location utility - uses NEXT state
    obs_dist = A_loc @ qs_self_next
    location_utility = (obs_dist * C_loc).sum()

    # 2. Edge utility - uses NEXT state
    edge_dist = A_edge[:, :, action_self] @ qs_self_next
    edge_utility = (edge_dist * C_edge).sum()

    # 3. Cell collision utility - uses NEXT states
    # p(o | qs_self_next, qs_other_next) = Σ_{s_i, s_j} A[o, s_i, s_j] * qs_self_next[s_i] * qs_other_next[s_j]
    cell_obs_dist = jnp.einsum('oij,i,j->o', A_cell_collision, qs_self_next, qs_other_next)
    cell_collision_utility = (cell_obs_dist * C_cell_collision).sum()

    # 4. Edge collision utility - uses CURRENT states + actions
    # p(o | qs_self_current, qs_other_current, a_self, a_other) = Σ_{s_i, s_j} A[o, s_i, s_j, a_self, a_other] * qs_self_current[s_i] * qs_other_current[s_j]
    A_edge_coll_slice = A_edge_collision[:, :, :, action_self, action_other]
    edge_coll_obs_dist = jnp.einsum('oij,i,j->o', A_edge_coll_slice, qs_self_current, qs_other_current)
    edge_collision_utility = (edge_coll_obs_dist * C_edge_collision).sum()

    # Total pragmatic utility
    total = location_utility + edge_utility + cell_collision_utility + edge_collision_utility
    return total


@jax.jit
def epistemic_info_gain_jax(
    qs_prior: jnp.ndarray,
    A: jnp.ndarray,
    eps: float = 1e-16,
) -> float:
    """
    Compute epistemic value (expected information gain).

    I = H[qs_prior] - E_o[H[qs_posterior(o)]]

    where qs_posterior(o) ∝ A[o, :] * qs_prior

    Parameters
    ----------
    qs_prior : jnp.ndarray
        Prior belief [num_states]
    A : jnp.ndarray
        Observation model [num_obs, num_states]
    eps : float
        Numerical stability

    Returns
    -------
    info_gain : float
        Expected information gain
    """
    # Prior entropy
    H_prior = -(qs_prior * jnp.log(qs_prior + eps)).sum()

    # Predicted observation distribution
    obs_dist = A @ qs_prior

    # Expected posterior entropy
    # For each observation o, compute posterior and its entropy
    def posterior_entropy(o_idx):
        likelihood = A[o_idx, :]
        posterior_unnorm = likelihood * qs_prior
        posterior = posterior_unnorm / (posterior_unnorm.sum() + eps)
        H_post = -(posterior * jnp.log(posterior + eps)).sum()
        return H_post

    # Vectorized over observations
    H_posteriors = vmap(posterior_entropy)(jnp.arange(A.shape[0]))
    H_post_expected = (obs_dist * H_posteriors).sum()

    info_gain = H_prior - H_post_expected
    return info_gain




# =============================================================================
# LEVEL 2: One-step EFE computation (vectorized over j actions)
# =============================================================================

@jax.jit
def compute_j_best_response_jax(
    qs_j: jnp.ndarray,
    qs_i: jnp.ndarray,
    qs_i_next: jnp.ndarray,
    action_i: int,
    B_j: jnp.ndarray,
    A_j_loc: jnp.ndarray,
    C_j_loc: jnp.ndarray,
    A_j_edge: jnp.ndarray,
    C_j_edge: jnp.ndarray,
    A_j_cell_collision: jnp.ndarray,
    C_j_cell_collision: jnp.ndarray,
    A_j_edge_collision: jnp.ndarray,
    C_j_edge_collision: jnp.ndarray,
    actions_j: jnp.ndarray,
    epistemic_scale: float,
    alpha_other: float,
    eps: float = 1e-16,
) -> Tuple[float, int]:
    """
    Compute j's best response to i's predicted position.

    For each action j can take:
        1. Propagate j's belief: qs_j' = B_j @ qs_j
        2. Compute G_j(a_j) = -pragmatic - epistemic - collision
        3. Pick best action: a_j* = argmin G_j(a_j)

    **KEY OPTIMIZATION:** Uses vmap over all j actions (vectorized, not loop!)

    Parameters
    ----------
    qs_j : jnp.ndarray
        j's current belief [num_states]
    qs_i : jnp.ndarray
        i's current belief (before i moved) [num_states]
    qs_i_next : jnp.ndarray
        i's predicted next belief [num_states]
    action_i : int
        i's committed action
    B_j : jnp.ndarray
        j's transition model
    A_j_loc, A_j_edge, A_j_cell_collision, A_j_edge_collision : jnp.ndarray
        j's observation models
    C_j_loc, C_j_edge, C_j_cell_collision, C_j_edge_collision : jnp.ndarray
        j's preferences
    actions_j : jnp.ndarray
        Array of j's primitive actions [num_actions]
    epistemic_scale : float
        Weight on epistemic term
    alpha_other : float
        The other agent's (j's) empathy level. Scales j's collision costs by
        (1 + alpha_other) to model that empathic agents feel both their own
        and the other's collision pain.
    eps : float
        Numerical stability

    Returns
    -------
    G_j_best : float
        Best EFE value for j
    best_action : int
        Best action index for j
    """
    # Scale j's collision preferences by (1 + alpha_other) to model j's empathy
    # An empathic j (alpha_other=1) feels collision costs for BOTH agents
    # A selfish j (alpha_other=0) only feels its own collision cost
    collision_scale = 1.0 + alpha_other
    C_j_cell_collision_scaled = C_j_cell_collision * collision_scale
    C_j_edge_collision_scaled = C_j_edge_collision * collision_scale

    # Compute G_j for each action (vectorized!)
    def compute_G_for_action(action_j):
        # Propagate j's belief
        qs_j_pred = propagate_belief_jax(qs_j, B_j, action_j, qs_other=qs_i_next, eps=eps)

        # Pragmatic utility (all modalities including edge collision)
        # CRITICAL: Use CURRENT states for edge collision, NEXT states for cell collision
        # Use SCALED collision preferences to model j's empathy level
        pragmatic = expected_pragmatic_utility_jax(
            qs_self_current=qs_j,         # j's CURRENT state for edge collision
            qs_other_current=qs_i,        # i's CURRENT state (before i moved) for edge collision
            qs_self_next=qs_j_pred,       # j's NEXT state for location/cell collision
            qs_other_next=qs_i_next,      # i's NEXT state for cell collision
            action_self=action_j,
            action_other=action_i,
            A_loc=A_j_loc,
            C_loc=C_j_loc,
            A_edge=A_j_edge,
            C_edge=C_j_edge,
            A_cell_collision=A_j_cell_collision,
            C_cell_collision=C_j_cell_collision_scaled,  # Scaled by j's empathy
            A_edge_collision=A_j_edge_collision,
            C_edge_collision=C_j_edge_collision_scaled,  # Scaled by j's empathy
        )

        # Epistemic value
        epistemic = epistemic_info_gain_jax(qs_j, A_j_loc, eps=eps)

        # EFE
        G_j = -pragmatic - epistemic_scale * epistemic

        return G_j

    # Vectorize over all j actions (THIS IS THE KEY OPTIMIZATION!)
    G_j_all = vmap(compute_G_for_action)(actions_j)

    # Best response
    best_idx = jnp.argmin(G_j_all)
    G_j_best = G_j_all[best_idx]
    best_action = actions_j[best_idx]

    return G_j_best, best_action


# =============================================================================
# LEVEL 3: Horizon rollout using lax.scan
# =============================================================================

@jax.jit
def rollout_one_policy_jax(
    policy_i: jnp.ndarray,
    qs_i_init: jnp.ndarray,
    qs_j_init: jnp.ndarray,
    B_i: jnp.ndarray,
    A_i_loc: jnp.ndarray,
    C_i_loc: jnp.ndarray,
    A_i_edge: jnp.ndarray,
    C_i_edge: jnp.ndarray,
    A_i_cell_collision: jnp.ndarray,
    C_i_cell_collision: jnp.ndarray,
    A_i_edge_collision: jnp.ndarray,
    C_i_edge_collision: jnp.ndarray,
    B_j: jnp.ndarray,
    A_j_loc: jnp.ndarray,
    C_j_loc: jnp.ndarray,
    A_j_edge: jnp.ndarray,
    C_j_edge: jnp.ndarray,
    A_j_cell_collision: jnp.ndarray,
    C_j_cell_collision: jnp.ndarray,
    A_j_edge_collision: jnp.ndarray,
    C_j_edge_collision: jnp.ndarray,
    actions_j: jnp.ndarray,
    epistemic_scale: float,
    alpha_other: float,
    eps: float = 1e-16,
) -> Tuple[float, float]:
    """
    Rollout a single i-policy over the horizon using lax.scan.

    **KEY OPTIMIZATION:** Replaces Python for-loop over timesteps with lax.scan.

    NumPy version (SLOW):
        for t in range(horizon):
            # i takes action
            qs_i_next = propagate_belief(...)
            # Compute i's EFE
            G_i_step = ...
            # j best-responds
            for a_j in actions_j:  # Python loop
                ...
            # Update beliefs
            ...

    JAX version (FAST):
        def step_fn(carry, action_i):
            # All operations in JAX
            # Inner loop over j actions uses vmap!
            ...
        lax.scan(step_fn, init_carry, policy_i)

    Parameters
    ----------
    policy_i : jnp.ndarray
        i's action sequence [horizon, num_factors]
    qs_i_init : jnp.ndarray
        i's initial belief [num_states]
    qs_j_init : jnp.ndarray
        j's initial belief [num_states]
    B_i : jnp.ndarray
        i's transition model
    A_i_loc, C_i_loc, A_i_edge, C_i_edge : jnp.ndarray
        i's observation models and preferences
    A_i_cell_collision, C_i_cell_collision : jnp.ndarray
        i's cell collision model and preferences
    A_i_edge_collision, C_i_edge_collision : jnp.ndarray
        i's edge collision model and preferences
    B_j : jnp.ndarray
        j's transition model
    A_j_loc, C_j_loc, A_j_edge, C_j_edge : jnp.ndarray
        j's observation models and preferences
    A_j_cell_collision, C_j_cell_collision : jnp.ndarray
        j's cell collision model and preferences
    A_j_edge_collision, C_j_edge_collision : jnp.ndarray
        j's edge collision model and preferences
    actions_j : jnp.ndarray
        j's primitive actions [num_actions]
    epistemic_scale : float
        Weight on epistemic term
    eps : float
        Numerical stability

    Returns
    -------
    G_i_avg : float
        Average EFE for i over horizon
    G_j_avg : float
        Average EFE for j (best-response) over horizon
    """

    # Extract action sequence (handle [H, 1] or [H] shapes)
    if policy_i.ndim == 2:
        action_seq = policy_i[:, 0].astype(jnp.int32)
    else:
        action_seq = policy_i.astype(jnp.int32)

    # Define step function for scan
    def step_fn(carry, action_i):
        qs_i, qs_j, total_G_i, total_G_j = carry

        # --- i takes action ---
        qs_i_next = propagate_belief_jax(qs_i, B_i, action_i, qs_other=qs_j, eps=eps)

        # --- i's EFE components (without edge collision - added after j's response) ---
        # Edge collision requires knowing BOTH agents' actions, so we compute it
        # after determining j's best response via ToM
        pragmatic_i_partial = expected_pragmatic_utility_jax(
            qs_self_current=qs_i,
            qs_other_current=qs_j,
            qs_self_next=qs_i_next,
            qs_other_next=qs_j,  # j hasn't moved yet, so next = current
            action_self=action_i,
            action_other=4,  # Dummy - edge collision computed separately below
            A_loc=A_i_loc,
            C_loc=C_i_loc,
            A_edge=A_i_edge,
            C_edge=C_i_edge,
            A_cell_collision=A_i_cell_collision,
            C_cell_collision=C_i_cell_collision,
            A_edge_collision=A_i_edge_collision,
            C_edge_collision=jnp.array([0.0, 0.0]),  # Computed separately below
        )
        epistemic_i = epistemic_info_gain_jax(qs_i, A_i_loc, eps=eps)

        # --- j best-responds (vectorized over all j actions!) ---
        # alpha_other is i's observation/belief about j's empathy level
        # This is derived from the empathy_obs modality (identity mapping for now)
        G_j_best, best_action_j = compute_j_best_response_jax(
            qs_j, qs_i, qs_i_next, action_i,
            B_j, A_j_loc, C_j_loc, A_j_edge, C_j_edge,
            A_j_cell_collision, C_j_cell_collision,
            A_j_edge_collision, C_j_edge_collision,
            actions_j, epistemic_scale, alpha_other, eps
        )

        # --- NOW compute i's edge collision since we know j's best response ---
        # Edge collision uses CURRENT states + BOTH agents' committed actions
        A_edge_coll_slice = A_i_edge_collision[:, :, :, action_i, best_action_j]
        edge_coll_obs_dist = jnp.einsum('oij,i,j->o', A_edge_coll_slice, qs_i, qs_j)
        edge_collision_utility_i = (edge_coll_obs_dist * C_i_edge_collision).sum()

        # Complete pragmatic utility for i (now includes edge collision with j's response)
        pragmatic_i = pragmatic_i_partial + edge_collision_utility_i

        G_i_step = -pragmatic_i - epistemic_scale * epistemic_i

        # Update j's belief with best action
        qs_j_next = propagate_belief_jax(qs_j, B_j, best_action_j, qs_other=qs_i_next, eps=eps)

        # Accumulate EFEs
        total_G_i_new = total_G_i + G_i_step
        total_G_j_new = total_G_j + G_j_best

        # Return (new_carry, output_to_stack)
        new_carry = (qs_i_next, qs_j_next, total_G_i_new, total_G_j_new)
        return new_carry, None

    # Initial carry
    init_carry = (qs_i_init, qs_j_init, 0.0, 0.0)

    # Scan over action sequence (THIS REPLACES THE PYTHON FOR LOOP!)
    final_carry, _ = lax.scan(step_fn, init_carry, action_seq)

    _, _, total_G_i, total_G_j = final_carry

    # Average over horizon
    horizon = len(action_seq)
    G_i_avg = total_G_i / horizon
    G_j_avg = total_G_j / horizon

    return G_i_avg, G_j_avg


# =============================================================================
# LEVEL 4: Vectorized over all i policies using vmap
# =============================================================================

# Vectorized version over all i policies
rollout_all_policies_jax = jax.jit(vmap(
    rollout_one_policy_jax,
    in_axes=(
        0,     # policy_i - vmap over policies
        None,  # qs_i_init
        None,  # qs_j_init
        None,  # B_i
        None,  # A_i_loc
        None,  # C_i_loc
        None,  # A_i_edge
        None,  # C_i_edge
        None,  # A_i_cell_collision
        None,  # C_i_cell_collision
        None,  # A_i_edge_collision
        None,  # C_i_edge_collision
        None,  # B_j
        None,  # A_j_loc
        None,  # C_j_loc
        None,  # A_j_edge
        None,  # C_j_edge
        None,  # A_j_cell_collision
        None,  # C_j_cell_collision
        None,  # A_j_edge_collision
        None,  # C_j_edge_collision
        None,  # actions_j
        None,  # epistemic_scale
        None,  # alpha_other - observed/inferred empathy of other agent
    )
    # Note: eps parameter uses default value, so not included in in_axes
))


def compute_empathic_G_jax(
    qs_i: np.ndarray,
    B_i: np.ndarray,
    A_i_loc: np.ndarray,
    C_i_loc: np.ndarray,
    A_i_edge: np.ndarray,
    C_i_edge: np.ndarray,
    A_i_cell_collision: np.ndarray,
    C_i_cell_collision: np.ndarray,
    A_i_edge_collision: np.ndarray,
    C_i_edge_collision: np.ndarray,
    policies_i: np.ndarray,
    qs_j: np.ndarray,
    B_j: np.ndarray,
    A_j_loc: np.ndarray,
    C_j_loc: np.ndarray,
    A_j_edge: np.ndarray,
    C_j_edge: np.ndarray,
    A_j_cell_collision: np.ndarray,
    C_j_cell_collision: np.ndarray,
    A_j_edge_collision: np.ndarray,
    C_j_edge_collision: np.ndarray,
    policies_j: np.ndarray,
    alpha: float,
    alpha_other: float,
    epistemic_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    JAX-accelerated empathic EFE computation.

    **MAIN API FUNCTION** - Drop-in replacement for compute_empathic_G from si_empathy_lava.py

    **KEY OPTIMIZATION:** Triple nested loops → fully vectorized JAX:
    - Outer loop (policies): vmap → batched computation
    - Middle loop (horizon): lax.scan → compiled sequential ops
    - Inner loop (j actions): vmap → batched computation

    NumPy version (SLOW):
        for policy_i in policies_i:  # 125-625 policies
            for t in range(horizon):  # 3-5 timesteps
                for action_j in actions_j:  # 5 actions
                    # Compute G_j(action_j)
                # Pick best j action
            # Accumulate G_i, G_j

    JAX version (FAST):
        G_i, G_j = vmap(
            lambda policy: lax.scan(
                lambda carry, action: vmap(compute_G_j)(actions_j)
            )
        )(policies_i)

    Expected speedup: 50-100x for horizon=3, 125 policies

    Parameters
    ----------
    qs_i, qs_j : np.ndarray
        Beliefs over own/other positions
    B_i, B_j : np.ndarray
        Transition models
    A_i_loc, A_j_loc : np.ndarray
        Location observation models
    C_i_loc, C_j_loc : np.ndarray
        Location preferences
    A_i_edge, A_j_edge : np.ndarray
        Edge observation models
    C_i_edge, C_j_edge : np.ndarray
        Edge preferences
    A_i_cell_collision, A_j_cell_collision : np.ndarray
        Cell collision observation models
    C_i_cell_collision, C_j_cell_collision : np.ndarray
        Cell collision preferences
    A_i_edge_collision, A_j_edge_collision : np.ndarray
        Edge collision observation models
    C_i_edge_collision, C_j_edge_collision : np.ndarray
        Edge collision preferences
    policies_i, policies_j : np.ndarray
        Policy sets
    alpha : float
        Empathy weight ∈ [0, 1]
    epistemic_scale : float
        Weight on epistemic value term

    Returns
    -------
    G_i : np.ndarray
        Agent i's EFE for each policy [num_policies]
    G_j_best_response : np.ndarray
        j's best-response EFE for each i-policy [num_policies]
    G_social : np.ndarray
        Empathy-weighted social EFE [num_policies]
    """
    # Convert to JAX arrays
    qs_i_jax = jnp.array(qs_i)
    qs_j_jax = jnp.array(qs_j)
    B_i_jax = jnp.array(B_i)
    B_j_jax = jnp.array(B_j)

    # Agent i A matrices and preferences
    A_i_loc_jax = jnp.array(A_i_loc)
    C_i_loc_jax = jnp.array(C_i_loc)
    A_i_edge_jax = jnp.array(A_i_edge)
    C_i_edge_jax = jnp.array(C_i_edge)
    A_i_cell_collision_jax = jnp.array(A_i_cell_collision)
    C_i_cell_collision_jax = jnp.array(C_i_cell_collision)
    A_i_edge_collision_jax = jnp.array(A_i_edge_collision)
    C_i_edge_collision_jax = jnp.array(C_i_edge_collision)

    # Agent j A matrices and preferences
    A_j_loc_jax = jnp.array(A_j_loc)
    C_j_loc_jax = jnp.array(C_j_loc)
    A_j_edge_jax = jnp.array(A_j_edge)
    C_j_edge_jax = jnp.array(C_j_edge)
    A_j_cell_collision_jax = jnp.array(A_j_cell_collision)
    C_j_cell_collision_jax = jnp.array(C_j_cell_collision)
    A_j_edge_collision_jax = jnp.array(A_j_edge_collision)
    C_j_edge_collision_jax = jnp.array(C_j_edge_collision)

    policies_i_jax = jnp.array(policies_i, dtype=jnp.int32)

    # j's primitive actions (only consider first timestep of each policy)
    actions_j_jax = jnp.array([p[0, 0] for p in policies_j], dtype=jnp.int32)

    # **VECTORIZED COMPUTATION OVER ALL POLICIES**
    # This is where the magic happens - single vmap replaces 125+ Python iterations!
    G_i_jax, G_j_jax = rollout_all_policies_jax(
        policies_i_jax,
        qs_i_jax,
        qs_j_jax,
        B_i_jax,
        A_i_loc_jax,
        C_i_loc_jax,
        A_i_edge_jax,
        C_i_edge_jax,
        A_i_cell_collision_jax,
        C_i_cell_collision_jax,
        A_i_edge_collision_jax,
        C_i_edge_collision_jax,
        B_j_jax,
        A_j_loc_jax,
        C_j_loc_jax,
        A_j_edge_jax,
        C_j_edge_jax,
        A_j_cell_collision_jax,
        C_j_cell_collision_jax,
        A_j_edge_collision_jax,
        C_j_edge_collision_jax,
        actions_j_jax,
        epistemic_scale,
        alpha_other,  # Observed/inferred empathy of other agent
    )

    # Empathy-weighted social EFE
    G_social_jax = G_i_jax + alpha * G_j_jax

    # Convert back to NumPy for compatibility
    G_i = np.array(G_i_jax)
    G_j_best_response = np.array(G_j_jax)
    G_social = np.array(G_social_jax)

    return G_i, G_j_best_response, G_social
