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


def _expected_location_utility(
    qs: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
) -> float:
    """
    Compute expected pragmatic utility from location_obs at one step:

        E_o[C(o)] with o ~ A @ qs

    Parameters
    ----------
    qs : (num_states,)
        Belief over states.
    A : (num_obs, num_states)
        Observation model for location_obs.
    C : (num_obs,)
        Preferences over location observations.

    Returns
    -------
    expected_utility : float
    """
    obs_dist = A @ qs
    return float((obs_dist * C).sum())


def _expected_collision_utility(
    qs_self: np.ndarray,
    qs_other: np.ndarray,
    C_relation: np.ndarray,
) -> float:
    """
    Compute expected relational utility due to collision at one step.

    We approximate the joint belief as factorised:
        q(s_self, s_other) ≈ q_self(s_self) * q_other(s_other)

    Then:
        p_same_cell = sum_s q_self(s) * q_other(s)

    We currently use the C_relation[2] (same cell) category only; C_relation[1]
    (same row) could be added similarly if desired.

    Parameters
    ----------
    qs_self : (num_states,)
        Belief over own position.
    qs_other : (num_states,)
        Belief over other's position.
    C_relation : (3,)
        Relational preferences; index 2 corresponds to collision.

    Returns
    -------
    expected_collision_utility : float
    """
    # Probability that both agents occupy the same cell
    p_same_cell = float(np.dot(qs_self, qs_other))
    # Collision utility
    collision_utility = C_relation[2] * p_same_cell
    # (We ignore same-row penalties for now; can be added later.)
    return collision_utility


def compute_empathic_G(
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
    alpha: float,
    A_i: np.ndarray,
    A_j: np.ndarray,
    epistemic_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empathy-weighted EFE for agent i.

    For each candidate policy π_i:

        Initialise beliefs:
            q_i^0 = qs_i
            q_j^0 = qs_j

        For t = 0..H-1:
            1. i takes action a_i^t (from π_i):
               q_i^{t+1} = B_i(q_i^t, q_j^t, a_i^t)

            2. Compute i's pragmatic + epistemic + collision contributions for this step:
               - pragmatic: E_o[C_i_loc(o)] where o ~ A_i @ q_i^{t+1}
               - epistemic: expected information gain from A_i (prior q_i^t)
               - collision: C_i_rel[2] * p(collision) with p(collision) ≈ Σ_s q_i^{t+1}(s) q_j^t(s)

            3. Theory of Mind: j best-responds my predicted move.
               For each primitive action a_j:
                   q_j' = B_j(q_j^t, q_i^{t+1}, a_j)
                   pragmatic_j = E_o[C_j_loc(o)] where o ~ A_j @ q_j'
                   epistemic_j = info gain from A_j (prior q_j^t)
                   collision_j = C_j_rel[2] * p(collision) with same p(collision)
                   G_j(a_j) = -pragmatic_j - epistemic_scale * epistemic_j - collision_j

               Choose best a_j^t = argmin_a_j G_j(a_j) and accumulate G_j_best += G_j(a_j^t).
               Update q_j^{t+1} using that best action.

        Then:
            G_i(π_i) = (Σ_t G_i_step^t) / H
            G_j_best(π_i) = (Σ_t G_j_best_step^t) / H

        Empathy:
            G_social(π_i) = G_i(π_i) + α · G_j_best(π_i)

    Parameters
    ----------
    qs_i, B_i, C_i_loc, C_i_rel, policies_i, A_i
        Agent i's components
    qs_j, B_j, C_j_loc, C_j_rel, policies_j, A_j
        Agent j's components
    alpha : float
        Empathy weight ∈ [0, 1]
    epistemic_scale : float
        Weight on epistemic value term (η)

    Returns
    -------
    G_i : (num_policies,)
        Agent i's full EFE (risk + epistemic + collision) for each policy.
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

            # --- i's action and belief propagation ---
            qs_i_next = _propagate_belief(qs_i_t, B_i, a_i_t, qs_other=qs_j_t)

            # --- i's pragmatic utility from location_obs ---
            pragmatic_i = _expected_location_utility(qs_i_next, A_i, C_i_loc)

            # --- i's epistemic value (info gain) from A_i ---
            info_gain_i = _epistemic_info_gain(qs_i_t, A_i, eps=eps)

            # --- i's expected collision utility ---
            collision_utility_i = _expected_collision_utility(qs_i_next, qs_j_t, C_i_rel)

            # One-step EFE contribution for i:
            # G_i_step = -pragmatic_i - epistemic_scale * info_gain_i - collision_utility_i
            G_i_step = (
                -pragmatic_i
                - epistemic_scale * info_gain_i
                - collision_utility_i
            )
            total_G_i += G_i_step

            # --- Theory of Mind: j best-responds to i's predicted move ---
            # For j, we consider one-step policies given current beliefs.
            G_j_actions = []
            for policy_j in policies_j:
                a_j = int(policy_j[0, 0])

                # Propagate j under candidate action a_j, conditioned on i's new position
                qs_j_pred = _propagate_belief(qs_j_t, B_j, a_j, qs_other=qs_i_next)

                # Pragmatic utility for j
                pragmatic_j = _expected_location_utility(qs_j_pred, A_j, C_j_loc)

                # Epistemic value for j
                info_gain_j = _epistemic_info_gain(qs_j_t, A_j, eps=eps)

                # Collision utility for j (same p(collision) approximation)
                collision_utility_j = _expected_collision_utility(qs_j_pred, qs_i_next, C_j_rel)

                G_j_a = (
                    -pragmatic_j
                    - epistemic_scale * info_gain_j
                    - collision_utility_j
                )
                G_j_actions.append(G_j_a)

            G_j_actions = np.asarray(G_j_actions)
            best_j_idx = int(np.argmin(G_j_actions))
            best_j_action = int(policies_j[best_j_idx, 0, 0])
            G_j_best_step = float(G_j_actions[best_j_idx])
            total_G_j += G_j_best_step

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

    Attributes
    ----------
    agent_i : LavaAgent
        Focal agent (the one making decisions)
    agent_j : LavaAgent
        Other agent (whose EFE is considered via empathy)
    alpha : float
        Empathy weight ∈ [0, 1] (0 = selfish, 1 = fully prosocial)
    epistemic_scale : float
        Weight on epistemic value term
    """
    agent_i: LavaAgent
    agent_j: LavaAgent
    alpha: float = 0.5
    epistemic_scale: float = 1.0

    def plan(
        self,
        qs_i: np.ndarray,
        qs_j: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Select action for agent i based on empathic full EFE.

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
        C_i_loc = np.asarray(self.agent_i.C["location_obs"])
        C_i_rel = np.asarray(self.agent_i.C["relation_obs"])
        A_i = np.asarray(self.agent_i.A["location_obs"])
        policies_i = np.asarray(self.agent_i.policies)
        gamma = self.agent_i.gamma

        # Extract agent j's model components
        B_j = np.asarray(self.agent_j.B["location_state"])
        C_j_loc = np.asarray(self.agent_j.C["location_obs"])
        C_j_rel = np.asarray(self.agent_j.C["relation_obs"])
        A_j = np.asarray(self.agent_j.A["location_obs"])
        policies_j = np.asarray(self.agent_j.policies)

        # Compute empathic EFE and policy posterior
        G_i, G_j, G_social, q_pi = efe_empathic(
            qs_i, B_i, C_i_loc, C_i_rel, policies_i,
            qs_j, B_j, C_j_loc, C_j_rel, policies_j,
            gamma, self.alpha, A_i, A_j,
            epistemic_scale=self.epistemic_scale,
        )

        # Select best policy (argmax for now)
        best_policy_idx = int(np.argmax(q_pi))

        # Extract first action from best policy
        best_policy = policies_i[best_policy_idx]  # (horizon, num_state_factors)
        action = int(best_policy[0, 0])           # First timestep, first state factor

        return G_i, G_j, G_social, q_pi, action
