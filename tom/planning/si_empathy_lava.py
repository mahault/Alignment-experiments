"""
Multi-agent empathic active inference planner for Lava corridor (Phase 2).

This module extends the single-agent planner by adding empathy and full EFE:

For agent i:
- G_i(π) includes:
    - pragmatic value (preferences C over location_obs)
    - epistemic value (information gain about hidden states)
    - collision aversion via C["relation_obs"][2] and p(collision)
- G_social^i(π) = G_i(π) + α·G_j(π | i's actions)

Regime B:
- Collisions are physically possible.
- Collision avoidance is implemented via relational preferences C["relation_obs"].
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass

from tom.models import LavaAgent
from tom.planning.si_lava import compute_full_G

# Recursive ToM depth - how many levels of "I think you think I think..."
TOM_DEPTH = 2


def compute_other_agent_G(
    qs_j: np.ndarray,
    B_j: np.ndarray,
    C_j_loc: np.ndarray,
    policies_j: np.ndarray,
    A_j: np.ndarray = None,
    qs_i: np.ndarray = None,
    epistemic_scale: float = 1.0,
) -> np.ndarray:
    """
    Backwards-compatible helper to compute the other agent's EFE
    for their policy set.

    This wraps compute_full_G from si_lava, using:
      - qs_j as j's belief about its own state
      - B_j as j's transition model (3D or 4D)
      - C_j_loc as j's location_obs preferences
      - policies_j as j's policy set
      - A_j as j's observation model
      - qs_i as "other" belief if B_j is 4D

    Parameters
    ----------
    qs_j : (num_states,)
        Other agent's belief state.
    B_j : np.ndarray
        Other agent's transition model.
    C_j_loc : (num_obs,)
        Other agent's preferences over location observations.
    policies_j : np.ndarray
        Other agent's policy set.
    A_j : np.ndarray, optional
        Other agent's observation model.
    qs_i : (num_states,), optional
        Belief about i's position (for 4D B_j).
    epistemic_scale : float
        Weight on epistemic term in G.

    Returns
    -------
    G_j : (num_policies,)
        Other agent's full EFE for each of their policies.
    """
    return compute_full_G(
        qs_j,
        B_j,
        C_j_loc,
        policies_j,
        A=A_j,
        qs_other=qs_i,
        epistemic_scale=epistemic_scale,
    )


def _propagate_belief(
    qs: np.ndarray,
    B: np.ndarray,
    action: int,
    qs_other: np.ndarray = None,
) -> np.ndarray:
    """
    Propagate belief one step under action using 3D or 4D B.

    Parameters
    ----------
    qs : (num_states,)
        Current belief over own location.
    B : np.ndarray
        Transition model: 3D [s', s, a] or 4D [s', s, s_other, a].
    action : int
        Chosen action index.
    qs_other : (num_states,), optional
        Belief over other agent's location (required if B is 4D).

    Returns
    -------
    qs_next : (num_states,)
        Predicted belief after taking action.
    """
    if B.ndim == 3:
        return B[:, :, action] @ qs
    elif B.ndim == 4:
        if qs_other is None:
            raise ValueError("qs_other is required for 4D B matrices")
        qs_next = np.zeros_like(qs)
        for s_other in range(len(qs_other)):
            qs_next += B[:, :, s_other, action] @ qs * qs_other[s_other]
        return qs_next
    else:
        raise ValueError(f"B must be 3D or 4D, got shape {B.shape}")


def _epistemic_info_gain(
    qs_prior: np.ndarray,
    A: np.ndarray,
    eps: float = 1e-16,
) -> float:
    """
    Compute expected information gain (epistemic value) for a single time step.

    Approximates:
        E_o[ D_KL(q(s|o) || q(s)) ]
    using the current prior qs_prior and observation model A.

    Parameters
    ----------
    qs_prior : (num_states,)
        Prior belief over states before seeing o.
    A : (num_obs, num_states)
        Observation model mapping states → observations.
    eps : float
        Numerical floor for logs.

    Returns
    -------
    info_gain : float
        Expected information gain for this step.
    """
    prior = np.clip(qs_prior, eps, 1.0)
    prior = prior / prior.sum()

    # Predict observations
    obs_dist = A @ prior
    obs_dist = np.clip(obs_dist, eps, 1.0)
    obs_dist = obs_dist / obs_dist.sum()

    # Entropy of prior
    H_prior = -np.sum(prior * np.log(prior))

    # Expected entropy after observing o
    H_post_weighted = 0.0
    num_obs = obs_dist.shape[0]

    for o in range(num_obs):
        p_o = obs_dist[o]
        if p_o < eps:
            continue

        likelihood = np.clip(A[o, :], eps, 1.0)
        numer = likelihood * prior
        if numer.sum() <= eps:
            post = prior
        else:
            post = numer / numer.sum()

        H_post = -np.sum(post * np.log(post))
        H_post_weighted += p_o * H_post

    info_gain = H_prior - H_post_weighted
    return float(info_gain)


def _expected_pragmatic_utility(
    qs_self_current: np.ndarray,
    qs_other_current: np.ndarray,
    qs_self_next: np.ndarray,
    qs_other_next: np.ndarray,
    action_self: int,
    action_other: int,
    A_loc: np.ndarray,
    C_loc: np.ndarray,
    A_edge: np.ndarray,
    C_edge: np.ndarray,
    A_cell_collision: np.ndarray,
    C_cell_collision: np.ndarray,
    A_edge_collision: np.ndarray,
    C_edge_collision: np.ndarray,
) -> float:
    """
    Compute expected pragmatic utility over ALL observation modalities.

    In active inference, pragmatic value = Σ_m E[C_m(o_m)]
    where we sum expected utility over all modalities.

    Modalities:
    1. location_obs: E[C_loc(o)] = (A_loc @ qs_self_next) · C_loc
    2. edge_obs: E[C_edge(e)] = (A_edge[:,:,action_self] @ qs_self_next) · C_edge
    3. cell_collision_obs: Marginalize A over joint NEXT state beliefs
    4. edge_collision_obs: Marginalize A over joint CURRENT state beliefs + actions

    NOTE: Edge collision depends on CURRENT states + actions (which edges are traversed),
    while cell collision depends on NEXT states (where agents end up).

    Parameters
    ----------
    qs_self_current : (num_states,)
        Belief over own CURRENT position
    qs_other_current : (num_states,)
        Belief over other's CURRENT position
    qs_self_next : (num_states,)
        Belief over own NEXT position
    qs_other_next : (num_states,)
        Belief over other's NEXT position
    action_self : int
        Own action
    action_other : int
        Other's action
    A_loc : (num_obs, num_states)
        Location observation model
    C_loc : (num_obs,)
        Location preferences
    A_edge : (num_edges + 1, num_states, num_actions)
        Edge observation model for self
    C_edge : (num_edges + 1,)
        Edge preferences
    A_cell_collision : (2, num_states, num_states)
        Cell collision observation model
    C_cell_collision : (2,)
        Cell collision preferences [no_collision, collision]
    A_edge_collision : (2, num_states, num_states, num_actions, num_actions)
        Edge collision observation model
    C_edge_collision : (2,)
        Edge collision preferences [no_edge_collision, edge_collision]

    Returns
    -------
    total_pragmatic : float
        Sum of expected utilities over all modalities
    """
    # 1. Location utility: E[C_loc(o)] - uses NEXT state
    obs_dist = A_loc @ qs_self_next
    location_utility = float((obs_dist * C_loc).sum())

    # 2. Edge utility: E[C_edge(e)] - uses NEXT state
    edge_dist = A_edge[:, :, action_self] @ qs_self_next
    edge_utility = float((edge_dist * C_edge).sum())

    # 3. Cell collision utility: E[C_cell_collision(o)] - uses NEXT states
    # p(o | qs_self_next, qs_other_next) = Σ_{s_i, s_j} A[o, s_i, s_j] * qs_self_next[s_i] * qs_other_next[s_j]
    # Using einsum for efficient marginalization
    cell_obs_dist = np.einsum('oij,i,j->o', A_cell_collision, qs_self_next, qs_other_next)
    cell_collision_utility = float((cell_obs_dist * C_cell_collision).sum())

    # 4. Edge collision utility: E[C_edge_collision(o)] - uses CURRENT states + actions
    # p(o | qs_self_current, qs_other_current, a_self, a_other) = Σ_{s_i, s_j} A[o, s_i, s_j, a_self, a_other] * qs_self_current[s_i] * qs_other_current[s_j]
    # Extract the slice for the specific actions
    A_edge_coll_slice = A_edge_collision[:, :, :, action_self, action_other]
    edge_coll_obs_dist = np.einsum('oij,i,j->o', A_edge_coll_slice, qs_self_current, qs_other_current)
    edge_collision_utility = float((edge_coll_obs_dist * C_edge_collision).sum())

    # Total pragmatic utility (sum over all modalities)
    total = location_utility + edge_utility + cell_collision_utility + edge_collision_utility
    return float(total)


def predict_other_action_recursive(
    qs_other: np.ndarray,
    qs_self: np.ndarray,
    B_other: np.ndarray,
    B_self: np.ndarray,
    A_other_loc: np.ndarray,
    C_other_loc: np.ndarray,
    A_other_edge: np.ndarray,
    C_other_edge: np.ndarray,
    A_other_cell_collision: np.ndarray,
    C_other_cell_collision: np.ndarray,
    A_other_edge_collision: np.ndarray,
    C_other_edge_collision: np.ndarray,
    A_self_loc: np.ndarray,
    C_self_loc: np.ndarray,
    A_self_edge: np.ndarray,
    C_self_edge: np.ndarray,
    A_self_cell_collision: np.ndarray,
    C_self_cell_collision: np.ndarray,
    A_self_edge_collision: np.ndarray,
    C_self_edge_collision: np.ndarray,
    policies_other: np.ndarray,
    policies_self: np.ndarray,
    alpha_other: float,
    alpha_self: float,
    depth: int = TOM_DEPTH,
    epistemic_scale: float = 1.0,
) -> Tuple[int, np.ndarray]:
    """
    Predict what the other agent will do using recursive ToM.

    BOTH agents are IDENTICAL except for alpha:
    1. Predict other's action (using ToM)
    2. Use other's predicted position for collision
    3. Compute G_social = G_self + alpha * G_other
    4. Choose action minimizing G_social

    depth=0: Base case, assume opponent stays in place
    depth=1: Predict opponent assuming they use depth=0
    depth=2: Predict opponent assuming they use depth=1

    Parameters
    ----------
    qs_other : np.ndarray
        Other agent's current belief state
    qs_self : np.ndarray
        Self's current belief state
    B_other, B_self : np.ndarray
        Transition models
    A_*_loc, C_*_loc, etc : np.ndarray
        Observation models and preferences for both agents
    policies_other, policies_self : np.ndarray
        Policy sets for both agents
    alpha_other : float
        Other agent's empathy level
    alpha_self : float
        Self's empathy level
    depth : int
        Recursion depth for ToM reasoning
    epistemic_scale : float
        Weight on epistemic term

    Returns
    -------
    action : int
        Predicted action for other agent
    G_social : np.ndarray
        Other agent's G_social values for each action
    """
    if depth == 0:
        # Base case: assume opponent (self) stays in place
        # Other computes their G_social with us at current position
        G_other, G_self_sim, G_social = compute_empathic_G(
            qs_other, B_other,
            A_other_loc, C_other_loc, A_other_edge, C_other_edge,
            A_other_cell_collision, C_other_cell_collision,
            A_other_edge_collision, C_other_edge_collision,
            policies_other,
            qs_self, B_self,
            A_self_loc, C_self_loc, A_self_edge, C_self_edge,
            A_self_cell_collision, C_self_cell_collision,
            A_self_edge_collision, C_self_edge_collision,
            policies_self,
            alpha_other, alpha_self,
            epistemic_scale=epistemic_scale,
            qs_other_predicted=None,  # Self stays in place
        )
        best_action = int(np.argmin(G_social))
        return best_action, G_social

    # Recursive case: predict what other predicts we'll do
    # Other uses depth-1 to predict us
    our_predicted_action, _ = predict_other_action_recursive(
        qs_self, qs_other,  # Swapped: from other's perspective, we are "other"
        B_self, B_other,
        A_self_loc, C_self_loc, A_self_edge, C_self_edge,
        A_self_cell_collision, C_self_cell_collision,
        A_self_edge_collision, C_self_edge_collision,
        A_other_loc, C_other_loc, A_other_edge, C_other_edge,
        A_other_cell_collision, C_other_cell_collision,
        A_other_edge_collision, C_other_edge_collision,
        policies_self, policies_other,
        alpha_self, alpha_other,  # Swapped
        depth=depth - 1,
        epistemic_scale=epistemic_scale,
    )

    # Compute our predicted position after taking predicted action
    qs_self_predicted = _propagate_belief(qs_self, B_self, our_predicted_action, qs_other=qs_other)

    # Other computes their G_social using our predicted position
    G_other, G_self_sim, G_social = compute_empathic_G(
        qs_other, B_other,
        A_other_loc, C_other_loc, A_other_edge, C_other_edge,
        A_other_cell_collision, C_other_cell_collision,
        A_other_edge_collision, C_other_edge_collision,
        policies_other,
        qs_self, B_self,
        A_self_loc, C_self_loc, A_self_edge, C_self_edge,
        A_self_cell_collision, C_self_cell_collision,
        A_self_edge_collision, C_self_edge_collision,
        policies_self,
        alpha_other, alpha_self,
        epistemic_scale=epistemic_scale,
        qs_other_predicted=qs_self_predicted,  # Use our predicted position
    )

    best_action = int(np.argmin(G_social))
    return best_action, G_social


def compute_empathic_G(
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
    qs_other_predicted: np.ndarray = None,
    empathy_mode: str = "additive",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empathy-weighted EFE for agent i.

    empathy_mode:
    - "additive": G_social = G_self + α * G_other (default)
    - "weighted": G_social = (1-α) * G_self + α * G_other (Sanjeev's)

    Note: VFE (Variational Free Energy = -log p(o) = surprise) is computed
    during belief updates in safe_belief_update(), not during planning.
    VFE can be used for emotional state analysis (valence = -VFE) but is
    separate from EFE-based action selection.

    For each candidate policy π_i:

        Initialise beliefs:
            q_i^0 = qs_i, q_j^0 = qs_j (or qs_other_predicted for step 0)

        For t = 0..H-1:
            1. i takes action a_i^t (from π_i):
               q_i^{t+1} = B_i(q_i^t, q_j^t, a_i^t)

            2. Compute i's EFE:
               - pragmatic: Σ_modality E[C(o)] over location, edge, cell_collision (no edge collision)
               - epistemic: expected information gain from A_i_loc
               - G_i_step = -pragmatic - epistemic_scale * epistemic

            3. Theory of Mind: j best-responds to i's predicted move.
               For each primitive action a_j:
                   q_j' = B_j(q_j^t, q_i^{t+1}, a_j)
                   pragmatic_j = Σ_modality E[C(o)] over ALL modalities (including edge collision)
                   epistemic_j = info gain from A_j_loc
                   G_j(a_j) = -pragmatic_j - epistemic_scale * epistemic_j

               Choose best a_j^t = argmin_a_j G_j(a_j).
               Update q_j^{t+1} using that best action.

        Then:
            G_i(π_i) = (Σ_t G_i_step^t) / H
            G_j_best(π_i) = (Σ_t G_j_best_step^t) / H

        Empathy:
            G_social(π_i) = G_i(π_i) + α · G_j_best(π_i)

    Parameters
    ----------
    qs_i, qs_j : np.ndarray
        Beliefs over own/other agent's position
    B_i, B_j : np.ndarray
        Transition models
    A_i_loc, A_j_loc : np.ndarray
        Location observation models
    C_i_loc, C_j_loc : np.ndarray
        Location preferences
    A_i_edge, A_j_edge : np.ndarray
        Edge observation models (num_edges + 1, num_states, num_actions)
    C_i_edge, C_j_edge : np.ndarray
        Edge preferences
    A_i_cell_collision, A_j_cell_collision : np.ndarray
        Cell collision observation models (2, num_states, num_states)
    C_i_cell_collision, C_j_cell_collision : np.ndarray
        Cell collision preferences (binary)
    A_i_edge_collision, A_j_edge_collision : np.ndarray
        Edge collision observation models (2, num_states, num_states, num_actions, num_actions)
    C_i_edge_collision, C_j_edge_collision : np.ndarray
        Edge collision preferences (binary)
    policies_i, policies_j : np.ndarray
        Policy sets
    alpha : float
        Empathy weight ∈ [0, 1]
    epistemic_scale : float
        Weight on epistemic value term
    qs_other_predicted : np.ndarray, optional
        If provided, use this as j's predicted position for step 0 collision checking.
        This enables using ToM prediction: if we predict j will move, check collision
        against their PREDICTED position, not current position.

    Returns
    -------
    G_i : (num_policies,)
        Agent i's full EFE for each policy.
    G_j_best_response : (num_policies,)
        j's best-response EFE for each i-policy.
    G_social : (num_policies,)
        Empathy-weighted social EFE: G_social = G_i + α·G_j_best_response.
    """
    num_policies_i = len(policies_i)
    G_i = np.zeros(num_policies_i)
    G_j_best_response = np.zeros(num_policies_i)

    num_states = qs_i.shape[0]
    eps = 1e-16

    # Scale j's collision preferences by (1 + alpha_other) to model j's empathy
    # An empathic j (alpha_other=1) feels collision costs for BOTH agents (2x cost)
    # A selfish j (alpha_other=0) only feels its own collision cost (1x cost)
    collision_scale = 1.0 + alpha_other
    C_j_cell_collision_scaled = C_j_cell_collision * collision_scale
    C_j_edge_collision_scaled = C_j_edge_collision * collision_scale

    # For step 0, use predicted j position if provided (for ToM-based collision check)
    qs_j_step0 = qs_other_predicted if qs_other_predicted is not None else qs_j

    for i_policy_idx, policy_i in enumerate(policies_i):
        action_seq_i = policy_i[:, 0].astype(int)  # (horizon,)
        horizon = len(action_seq_i)

        # Initial beliefs for this rollout
        qs_i_t = qs_i.copy()
        qs_j_t = qs_j.copy()

        total_G_i = 0.0
        total_G_j = 0.0

        for t in range(horizon):
            a_i_t = int(action_seq_i[t])

            # For step 0, use predicted j position for collision checking
            # This is the key fix: in simultaneous play, check collision against
            # where j WILL BE (predicted), not where j currently is
            qs_j_for_collision = qs_j_step0 if t == 0 else qs_j_t

            # --- i's action and belief propagation ---
            qs_i_next = _propagate_belief(qs_i_t, B_i, a_i_t, qs_other=qs_j_t)

            # --- i's pragmatic utility (without edge collision - added after j's response) ---
            # Edge collision requires knowing BOTH agents' actions, so we compute it
            # after determining j's best response via ToM
            pragmatic_i_partial = _expected_pragmatic_utility(
                qs_self_current=qs_i_t,
                qs_other_current=qs_j_t,
                qs_self_next=qs_i_next,
                qs_other_next=qs_j_for_collision,  # Use predicted position for step 0
                action_self=a_i_t,
                action_other=4,  # Dummy - edge collision computed separately below
                A_loc=A_i_loc,
                C_loc=C_i_loc,
                A_edge=A_i_edge,
                C_edge=C_i_edge,
                A_cell_collision=A_i_cell_collision,
                C_cell_collision=C_i_cell_collision,
                A_edge_collision=A_i_edge_collision,
                C_edge_collision=np.array([0.0, 0.0]),  # Computed separately below
            )

            # --- i's epistemic value (info gain) ---
            info_gain_i = _epistemic_info_gain(qs_i_t, A_i_loc, eps=eps)

            # --- Theory of Mind: j best-responds to i's predicted move ---
            # For j, we consider one-step policies given current beliefs.
            # Here we CAN include edge collision because we know both i's action (a_i_t) and j's candidate action (a_j)
            G_j_actions = []
            for policy_j in policies_j:
                a_j = int(policy_j[0, 0])

                # Propagate j under candidate action a_j, conditioned on i's new position
                qs_j_pred = _propagate_belief(qs_j_t, B_j, a_j, qs_other=qs_i_next)

                # Pragmatic utility for j (includes ALL modalities including edge collision)
                # Now we know both actions (i's committed a_i_t and j's candidate a_j)
                # CRITICAL: Use CURRENT states for edge collision (where agents ARE + actions)
                #           Use NEXT states for cell collision (where agents END UP)
                # Use SCALED collision preferences to model j's empathy level
                pragmatic_j = _expected_pragmatic_utility(
                    qs_self_current=qs_j_t,      # j's CURRENT state for edge collision
                    qs_other_current=qs_i_t,     # i's CURRENT state for edge collision (before i moved)
                    qs_self_next=qs_j_pred,      # j's NEXT state for location/cell collision
                    qs_other_next=qs_i_next,     # i's NEXT state for cell collision
                    action_self=a_j,
                    action_other=a_i_t,          # We know i's committed action
                    A_loc=A_j_loc,
                    C_loc=C_j_loc,
                    A_edge=A_j_edge,
                    C_edge=C_j_edge,
                    A_cell_collision=A_j_cell_collision,
                    C_cell_collision=C_j_cell_collision_scaled,  # Scaled by j's empathy
                    A_edge_collision=A_j_edge_collision,
                    C_edge_collision=C_j_edge_collision_scaled,  # Scaled by j's empathy
                )

                # Epistemic value for j
                info_gain_j = _epistemic_info_gain(qs_j_t, A_j_loc, eps=eps)

                # EFE for this action
                G_j_a = -pragmatic_j - epistemic_scale * info_gain_j
                G_j_actions.append(G_j_a)

            G_j_actions = np.asarray(G_j_actions)
            best_j_idx = int(np.argmin(G_j_actions))
            best_j_action = int(policies_j[best_j_idx, 0, 0])
            G_j_best_step = float(G_j_actions[best_j_idx])
            total_G_j += G_j_best_step

            # --- NOW compute i's edge collision since we know j's best response ---
            # Edge collision uses CURRENT states + BOTH agents' committed actions
            A_edge_coll_slice = A_i_edge_collision[:, :, :, a_i_t, best_j_action]
            edge_coll_obs_dist = np.einsum('oij,i,j->o', A_edge_coll_slice, qs_i_t, qs_j_t)
            edge_collision_utility_i = float((edge_coll_obs_dist * C_i_edge_collision).sum())

            # Complete pragmatic utility for i (now includes edge collision with j's response)
            pragmatic_i = pragmatic_i_partial + edge_collision_utility_i

            # One-step EFE contribution for i
            G_i_step = -pragmatic_i - epistemic_scale * info_gain_i
            total_G_i += G_i_step

            # Update j's belief with best-response action
            qs_j_next = _propagate_belief(qs_j_t, B_j, best_j_action, qs_other=qs_i_next)

            # Update beliefs for next timestep
            qs_i_t = qs_i_next
            qs_j_t = qs_j_next

        if horizon > 0:
            G_i[i_policy_idx] = total_G_i / horizon
            G_j_best_response[i_policy_idx] = total_G_j / horizon
        else:
            G_i[i_policy_idx] = 0.0
            G_j_best_response[i_policy_idx] = 0.0

    # Empathy-weighted social EFE
    # - "additive": G_social = G_self + α * G_other (default)
    # - "weighted": G_social = (1-α) * G_self + α * G_other (Sanjeev's)
    if empathy_mode == "weighted":
        G_social = (1 - alpha) * G_i + alpha * G_j_best_response
    else:  # "additive" (default)
        G_social = G_i + alpha * G_j_best_response
    return G_i, G_j_best_response, G_social


def efe_empathic(
    qs_i: np.ndarray,
    B_i: np.ndarray,
    C_i_loc: np.ndarray,
    C_i_rel: np.ndarray,
    policies_i: np.ndarray,
    qs_j: np.ndarray,
    B_j: np.ndarray,
    C_j_loc: np.ndarray,
    C_j_rel: np.ndarray,
    policies_j: np.ndarray,
    gamma: float,
    alpha: float,
    A_i: np.ndarray,
    A_j: np.ndarray,
    epistemic_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empathic full EFE and policy posterior.

    Parameters
    ----------
    qs_i, B_i, C_i_loc, C_i_rel, policies_i, A_i
        Agent i's components
    qs_j, B_j, C_j_loc, C_j_rel, policies_j, A_j
        Agent j's components
    gamma : float
        Inverse temperature for policy selection
    alpha : float
        Empathy weight ∈ [0, 1]
    epistemic_scale : float
        Weight on epistemic term

    Returns
    -------
    G_i : (num_policies,)
        Agent i's full EFE
    G_j : (num_policies,)
        Agent j's best-response EFE
    G_social : (num_policies,)
        Empathy-weighted EFE
    q_pi : (num_policies,)
        Policy posterior based on G_social
    """
    G_i, G_j, G_social = compute_empathic_G(
        qs_i, B_i, C_i_loc, C_i_rel, policies_i,
        qs_j, B_j, C_j_loc, C_j_rel, policies_j,
        alpha, A_i, A_j,
        epistemic_scale=epistemic_scale,
    )

    # Policy posterior: q(π) ∝ exp(-γ * G_social)
    log_q_pi = -gamma * G_social
    log_q_pi = log_q_pi - log_q_pi.max()  # Numerical stability
    q_pi = np.exp(log_q_pi)
    q_pi = q_pi / q_pi.sum()

    return G_i, G_j, G_social, q_pi


@dataclass
class EmpathicLavaPlanner:
    """
    Multi-agent empathic active inference planner for Lava corridor.

    This planner uses full Expected Free Energy (risk + epistemic +
    collision) for the focal agent, and weights the other agent's
    best-response EFE via an empathy parameter α.

    Supports both NumPy (default) and JAX (50-100x faster) backends.

    Attributes
    ----------
    agent_i : LavaAgent
        Focal agent (the one making decisions)
    agent_j : LavaAgent
        Other agent (whose EFE is considered via empathy)
    alpha : float
        Empathy weight ∈ [0, 1] (0 = selfish, 1 = fully prosocial)
    epistemic_scale : float
        Weight on epistemic value term (1.0 = full exploration)
    use_jax : bool
        If True, use JAX-accelerated empathy computation (50-100x faster).
        Falls back to NumPy if JAX is not available.
    empathy_mode : str
        Formula for combining self and other EFE:
        - "additive": G_social = G_self + α * G_other (default, current)
        - "weighted": G_social = (1-α) * G_self + α * G_other (Sanjeev's)
    tom_mode : str
        How to predict other's action:
        - "deterministic": argmin(G_j) - assume other takes best action (default)
        - "probabilistic": softmax(-γ * G_j) - distribution over actions
    """
    agent_i: LavaAgent
    agent_j: LavaAgent
    alpha: float = 0.5
    alpha_other: float = 0.0  # Observed/inferred empathy of other agent (for ToM)
    epistemic_scale: float = 1.0
    use_jax: bool = True  # Default to JAX for performance
    empathy_mode: str = "additive"  # "additive" or "weighted"
    tom_mode: str = "deterministic"  # "deterministic" or "probabilistic"

    def plan(
        self,
        qs_i: np.ndarray,
        qs_j: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Select action for agent i based on empathic full EFE.

        Automatically uses JAX-accelerated computation if use_jax=True and JAX is available,
        otherwise falls back to NumPy.

        Parameters
        ----------
        qs_i : np.ndarray
            Agent i's current belief state (num_states,)
        qs_j : np.ndarray
            Agent j's current belief state (num_states,)

        Returns
        -------
        G_i : np.ndarray
            Agent i's EFE for each policy (num_policies,)
        G_j : np.ndarray
            Agent j's EFE (best response per i-policy)
        G_social : np.ndarray
            Empathy-weighted EFE (num_policies,)
        q_pi : np.ndarray
            Policy posterior (num_policies,)
        action : int
            Selected action (first action of chosen policy)
        """
        # Extract agent i's model components
        B_i = np.asarray(self.agent_i.B["location_state"])
        A_i_loc = np.asarray(self.agent_i.A["location_obs"])
        C_i_loc = np.asarray(self.agent_i.C["location_obs"])
        A_i_edge = np.asarray(self.agent_i.A["edge_obs"])
        C_i_edge = np.asarray(self.agent_i.C["edge_obs"])
        A_i_cell_collision = np.asarray(self.agent_i.A["cell_collision_obs"])
        C_i_cell_collision = np.asarray(self.agent_i.C["cell_collision_obs"])
        A_i_edge_collision = np.asarray(self.agent_i.A["edge_collision_obs"])
        C_i_edge_collision = np.asarray(self.agent_i.C["edge_collision_obs"])
        policies_i = np.asarray(self.agent_i.policies)
        gamma = self.agent_i.gamma

        # Extract agent j's model components
        B_j = np.asarray(self.agent_j.B["location_state"])
        A_j_loc = np.asarray(self.agent_j.A["location_obs"])
        C_j_loc = np.asarray(self.agent_j.C["location_obs"])
        A_j_edge = np.asarray(self.agent_j.A["edge_obs"])
        C_j_edge = np.asarray(self.agent_j.C["edge_obs"])
        A_j_cell_collision = np.asarray(self.agent_j.A["cell_collision_obs"])
        C_j_cell_collision = np.asarray(self.agent_j.C["cell_collision_obs"])
        A_j_edge_collision = np.asarray(self.agent_j.A["edge_collision_obs"])
        C_j_edge_collision = np.asarray(self.agent_j.C["edge_collision_obs"])
        policies_j = np.asarray(self.agent_j.policies)

        # === RECURSIVE ToM: Predict j's action before computing EFE ===
        # This enables checking collision against j's PREDICTED position, not current position
        if self.use_jax:
            try:
                from tom.planning.jax_si_empathy_lava import predict_other_action_recursive_jax, TOM_DEPTH as JAX_TOM_DEPTH, TOM_HORIZON

                # Use JAX-accelerated ToM prediction (much faster!)
                predicted_j_action, _ = predict_other_action_recursive_jax(
                    qs_j, qs_i,  # From j's perspective: j is "self", i is "other"
                    self.alpha_other, self.alpha,  # j's alpha, our alpha
                    B_j, B_i,
                    A_j_loc, C_j_loc, A_j_edge, C_j_edge,
                    A_j_cell_collision, C_j_cell_collision,
                    A_i_loc, C_i_loc, A_i_edge, C_i_edge,
                    A_i_cell_collision, C_i_cell_collision,
                    depth=JAX_TOM_DEPTH,
                    horizon=TOM_HORIZON,
                    tom_mode=self.tom_mode,
                    gamma=gamma,
                )
            except ImportError:
                # Fall back to NumPy if JAX not available
                predicted_j_action, _ = predict_other_action_recursive(
                    qs_j, qs_i, B_j, B_i,
                    A_j_loc, C_j_loc, A_j_edge, C_j_edge,
                    A_j_cell_collision, C_j_cell_collision,
                    A_j_edge_collision, C_j_edge_collision,
                    A_i_loc, C_i_loc, A_i_edge, C_i_edge,
                    A_i_cell_collision, C_i_cell_collision,
                    A_i_edge_collision, C_i_edge_collision,
                    policies_j, policies_i,
                    self.alpha_other, self.alpha,
                    depth=TOM_DEPTH, epistemic_scale=self.epistemic_scale,
                )
        else:
            predicted_j_action, _ = predict_other_action_recursive(
                qs_j, qs_i, B_j, B_i,
                A_j_loc, C_j_loc, A_j_edge, C_j_edge,
                A_j_cell_collision, C_j_cell_collision,
                A_j_edge_collision, C_j_edge_collision,
                A_i_loc, C_i_loc, A_i_edge, C_i_edge,
                A_i_cell_collision, C_i_cell_collision,
                A_i_edge_collision, C_i_edge_collision,
                policies_j, policies_i,
                self.alpha_other, self.alpha,
                depth=TOM_DEPTH, epistemic_scale=self.epistemic_scale,
            )

        # Compute j's predicted position after taking predicted action
        qs_j_predicted = _propagate_belief(qs_j, B_j, predicted_j_action, qs_other=qs_i)

        # Compute empathic EFE (dispatch to JAX or NumPy)
        # Pass qs_j_predicted for step 0 collision checking (simultaneous play)
        if self.use_jax:
            try:
                from tom.planning.jax_si_empathy_lava import compute_empathic_G_jax

                # Use JAX-accelerated version (50-100x faster)
                G_i, G_j, G_social = compute_empathic_G_jax(
                    qs_i, B_i,
                    A_i_loc, C_i_loc, A_i_edge, C_i_edge,
                    A_i_cell_collision, C_i_cell_collision,
                    A_i_edge_collision, C_i_edge_collision,
                    policies_i,
                    qs_j, B_j,
                    A_j_loc, C_j_loc, A_j_edge, C_j_edge,
                    A_j_cell_collision, C_j_cell_collision,
                    A_j_edge_collision, C_j_edge_collision,
                    policies_j,
                    self.alpha,
                    self.alpha_other,  # Observed empathy of other agent
                    epistemic_scale=self.epistemic_scale,
                    qs_other_predicted=qs_j_predicted,  # Use predicted j position for step 0
                    empathy_mode=self.empathy_mode,
                )
            except ImportError as e:
                # Fall back to NumPy if JAX not available
                import warnings
                warnings.warn(
                    f"JAX not available ({e}), falling back to NumPy. "
                    "For 50-100x speedup, install JAX: pip install jax",
                    RuntimeWarning
                )
                G_i, G_j, G_social = compute_empathic_G(
                    qs_i, B_i,
                    A_i_loc, C_i_loc, A_i_edge, C_i_edge,
                    A_i_cell_collision, C_i_cell_collision,
                    A_i_edge_collision, C_i_edge_collision,
                    policies_i,
                    qs_j, B_j,
                    A_j_loc, C_j_loc, A_j_edge, C_j_edge,
                    A_j_cell_collision, C_j_cell_collision,
                    A_j_edge_collision, C_j_edge_collision,
                    policies_j,
                    self.alpha,
                    self.alpha_other,  # Observed empathy of other agent
                    epistemic_scale=self.epistemic_scale,
                    qs_other_predicted=qs_j_predicted,  # Use predicted j position for step 0
                    empathy_mode=self.empathy_mode,
                )
        else:
            # Use NumPy version explicitly
            G_i, G_j, G_social = compute_empathic_G(
                qs_i, B_i,
                A_i_loc, C_i_loc, A_i_edge, C_i_edge,
                A_i_cell_collision, C_i_cell_collision,
                A_i_edge_collision, C_i_edge_collision,
                policies_i,
                qs_j, B_j,
                A_j_loc, C_j_loc, A_j_edge, C_j_edge,
                A_j_cell_collision, C_j_cell_collision,
                A_j_edge_collision, C_j_edge_collision,
                policies_j,
                self.alpha,
                self.alpha_other,  # Observed empathy of other agent
                epistemic_scale=self.epistemic_scale,
                qs_other_predicted=qs_j_predicted,  # Use predicted j position for step 0
                empathy_mode=self.empathy_mode,
            )

        # Compute policy posterior: q(π) ∝ exp(-γ * G_social)
        log_q_pi = -gamma * G_social
        log_q_pi = log_q_pi - log_q_pi.max()  # Numerical stability
        q_pi = np.exp(log_q_pi)
        q_pi = q_pi / q_pi.sum()

        # Select best policy (argmax for now)
        best_policy_idx = int(np.argmax(q_pi))

        # Extract first action from best policy
        best_policy = policies_i[best_policy_idx]  # (horizon, num_state_factors)
        action = int(best_policy[0, 0])           # First timestep, first state factor

        return G_i, G_j, G_social, q_pi, action
