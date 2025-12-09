"""
Multi-agent empathic active inference planner for Lava corridor (Phase 2).

This module extends the single-agent planner from Phase 1 by adding empathy:
Agent i weights agent j's EFE when selecting policies.

Decision rule: G_social^i(π) = G_i(π) + α·G_j(π)
"""

import jax.numpy as jnp
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

from tom.models import LavaAgent
from tom.planning.si_lava import compute_risk_G, efe_risk_only


def compute_other_agent_G(
    qs_j: np.ndarray,
    B_j: np.ndarray,
    C_j: np.ndarray,
    policies_j: np.ndarray,
    A_j: np.ndarray = None,
    qs_i: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the other agent's EFE for their policy set.

    This is Theory of Mind: agent i simulates agent j's decision-making.

    Parameters
    ----------
    qs_j : np.ndarray
        Other agent's belief state (num_states,)
    B_j : np.ndarray
        Other agent's transition model. Can be 3D or 4D.
    C_j : np.ndarray
        Other agent's preference vector (num_obs,)
    policies_j : np.ndarray
        Other agent's policy set (num_policies, horizon, num_state_factors)
    A_j : np.ndarray, optional
        Other agent's observation model (num_obs, num_states)
    qs_i : np.ndarray, optional
        Agent i's position (for conditioning 4D B matrix)

    Returns
    -------
    G_j : np.ndarray
        Other agent's EFE for each of their policies (num_policies,)
    """
    return compute_risk_G(qs_j, B_j, C_j, policies_j, A_j, qs_other=qs_i)


def compute_empathic_G(
    qs_i: np.ndarray,
    B_i: np.ndarray,
    C_i: np.ndarray,
    policies_i: np.ndarray,
    qs_j: np.ndarray,
    B_j: np.ndarray,
    C_j: np.ndarray,
    policies_j: np.ndarray,
    alpha: float,
    A_i: np.ndarray = None,
    A_j: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empathy-weighted EFE for agent i.

    CORRECTED MODEL (sophisticated inference rollout):

    Before taking any action, agent i performs counterfactual rollouts:

    For each candidate action k ∈ {UP, DOWN, LEFT, RIGHT, STAY}:
        1. Rollout i's action: simulate i taking k → i at new position (qs_i_next)
        2. Theory of Mind rollout: j observes i's new position
           - j computes G_j for ALL their actions, CONDITIONED on qs_i_next
           - j selects best_action = argmin(G_j | qs_i_next)
        3. Empathy: G_social[k] = G_i[k] + α * G_j[best_action | qs_i_next]

    Then i selects: argmin(G_social)

    Key fixes:
    - j makes independent choice, not same indexed action as i
    - G_j is conditioned on i's predicted next position (not current)
    - Works with both 3D and 4D B matrices

    Parameters
    ----------
    qs_i : np.ndarray
        Agent i's belief state (num_states,)
    B_i, C_i, policies_i, A_i : np.ndarray
        Agent i's model components
    qs_j : np.ndarray
        Agent j's belief state (num_states,)
    B_j, C_j, policies_j, A_j : np.ndarray
        Agent j's model components
    alpha : float
        Empathy weight ∈ [0, 1]

    Returns
    -------
    G_i : np.ndarray
        Agent i's EFE for each policy (num_policies,)
    G_j_simulated : np.ndarray
        Agent j's EFE conditioned on i's actions (num_policies,)
    G_social : np.ndarray
        Empathy-weighted EFE: G_social = G_i + α·G_j(given i's action)
    """
    # Compute agent i's own EFE for each of their policies (selfish cost)
    # i's EFE conditions on j's current position if B_i is 4D
    G_i = compute_risk_G(qs_i, B_i, C_i, policies_i, A_i, qs_other=qs_j)

    # Multi-step counterfactual rollout: for each policy i considers,
    # simulate full horizon with j's best responses at each timestep
    num_policies_i = len(policies_i)
    G_j_best_response = np.zeros(num_policies_i)

    for i_policy_idx, policy_i in enumerate(policies_i):
        # Extract action sequence for this policy
        action_seq_i = policy_i[:, 0].astype(int)  # (horizon,)
        horizon = len(action_seq_i)

        # Initialize beliefs for this rollout
        qs_i_t = qs_i.copy()
        qs_j_t = qs_j.copy()

        # Accumulate j's EFE over the horizon
        total_G_j = 0.0

        # Roll out over the full horizon
        for t in range(horizon):
            action_i_t = action_seq_i[t]

            # ROLLOUT STEP 1: Simulate i taking action at timestep t
            # Predict i's next position, conditioning on j's current position
            if B_i.ndim == 3:
                qs_i_next = B_i[:, :, action_i_t] @ qs_i_t
            elif B_i.ndim == 4:
                # Marginalize over j's current position
                qs_i_next = np.zeros_like(qs_i_t)
                for s_j in range(len(qs_j_t)):
                    qs_i_next += B_i[:, :, s_j, action_i_t] @ qs_i_t * qs_j_t[s_j]
            else:
                raise ValueError(f"B_i must be 3D or 4D, got shape {B_i.shape}")

            # ROLLOUT STEP 2: Theory of Mind - j observes i's new position and responds
            # Compute j's EFE CONDITIONED on i's predicted next position
            G_j_all_actions = compute_other_agent_G(
                qs_j_t, B_j, C_j, policies_j, A_j, qs_i=qs_i_next
            )

            # ROLLOUT STEP 3: j selects their best action (minimum EFE) at this timestep
            best_j_action_idx = int(np.argmin(G_j_all_actions))
            best_j_action = int(policies_j[best_j_action_idx, 0, 0])

            # Accumulate j's EFE for this timestep
            total_G_j += G_j_all_actions[best_j_action_idx]

            # ROLLOUT STEP 4: Simulate j taking their best action
            # Update j's belief state for next iteration
            if B_j.ndim == 3:
                qs_j_next = B_j[:, :, best_j_action] @ qs_j_t
            elif B_j.ndim == 4:
                # Marginalize over i's NEXT position (after i's action)
                qs_j_next = np.zeros_like(qs_j_t)
                for s_i in range(len(qs_i_next)):
                    qs_j_next += B_j[:, :, s_i, best_j_action] @ qs_j_t * qs_i_next[s_i]
            else:
                raise ValueError(f"B_j must be 3D or 4D, got shape {B_j.shape}")

            # Update beliefs for next timestep
            qs_i_t = qs_i_next
            qs_j_t = qs_j_next

        # Store total EFE over horizon for this policy
        # Average over horizon to keep scale comparable across different H
        G_j_best_response[i_policy_idx] = total_G_j / horizon if horizon > 0 else 0.0

        # Note: Collision is now handled in B matrix (blocked transitions)
        # AND in C preferences (penalties for attempting blocked moves)

    # Empathy-weighted social EFE
    # G_social[k] = G_i[k] + α * G_j[best_response to i taking k]
    # If i's action k forces j into a bad outcome, empathic i will avoid k
    G_social = G_i + alpha * G_j_best_response

    return G_i, G_j_best_response, G_social


def efe_empathic(
    qs_i: np.ndarray,
    B_i: np.ndarray,
    C_i: np.ndarray,
    policies_i: np.ndarray,
    qs_j: np.ndarray,
    B_j: np.ndarray,
    C_j: np.ndarray,
    policies_j: np.ndarray,
    gamma: float,
    alpha: float,
    A_i: np.ndarray = None,
    A_j: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empathic EFE and policy posterior.

    Parameters
    ----------
    qs_i, B_i, C_i, policies_i, A_i : np.ndarray
        Agent i's components
    qs_j, B_j, C_j, policies_j, A_j : np.ndarray
        Agent j's components
    gamma : float
        Inverse temperature for policy selection
    alpha : float
        Empathy weight ∈ [0, 1]

    Returns
    -------
    G_i : np.ndarray
        Agent i's EFE (num_policies,)
    G_j : np.ndarray
        Agent j's EFE (num_policies,)
    G_social : np.ndarray
        Empathy-weighted EFE (num_policies,)
    q_pi : np.ndarray
        Policy posterior based on G_social (num_policies,)
    """
    # Compute empathic EFE
    G_i, G_j, G_social = compute_empathic_G(
        qs_i, B_i, C_i, policies_i,
        qs_j, B_j, C_j, policies_j,
        alpha, A_i, A_j
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

    This planner extends LavaPlanner by allowing agent i to weight
    agent j's EFE when selecting policies.

    Attributes
    ----------
    agent_i : LavaAgent
        Focal agent (the one making decisions)
    agent_j : LavaAgent
        Other agent (whose EFE is considered via empathy)
    alpha : float
        Empathy weight ∈ [0, 1] (0 = selfish, 1 = fully prosocial)
    """
    agent_i: LavaAgent
    agent_j: LavaAgent
    alpha: float = 0.5

    def plan(
        self,
        qs_i: np.ndarray,
        qs_j: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Select action for agent i based on empathic EFE.

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
            Agent j's EFE for each policy (num_policies,)
        G_social : np.ndarray
            Empathy-weighted EFE (num_policies,)
        q_pi : np.ndarray
            Policy posterior (num_policies,)
        action : int
            Selected action (first action of chosen policy)
        """
        # Extract agent i's model components
        B_i = np.asarray(self.agent_i.B["location_state"])
        C_i = np.asarray(self.agent_i.C["location_obs"])
        A_i = np.asarray(self.agent_i.A["location_obs"])
        policies_i = np.asarray(self.agent_i.policies)
        gamma = self.agent_i.gamma

        # Extract agent j's model components
        B_j = np.asarray(self.agent_j.B["location_state"])
        C_j = np.asarray(self.agent_j.C["location_obs"])
        A_j = np.asarray(self.agent_j.A["location_obs"])
        policies_j = np.asarray(self.agent_j.policies)

        # Compute empathic EFE and policy posterior
        G_i, G_j, G_social, q_pi = efe_empathic(
            qs_i, B_i, C_i, policies_i,
            qs_j, B_j, C_j, policies_j,
            gamma, self.alpha, A_i, A_j
        )

        # Select best policy (argmax for now)
        best_policy_idx = np.argmax(q_pi)

        # Extract first action from best policy
        best_policy = policies_i[best_policy_idx]  # (horizon, num_state_factors)
        action = int(best_policy[0, 0])  # First timestep, first state factor

        return G_i, G_j, G_social, q_pi, action
