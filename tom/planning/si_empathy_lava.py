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
) -> np.ndarray:
    """
    Compute the other agent's EFE for their policy set.

    This is Theory of Mind: agent i simulates agent j's decision-making.

    Parameters
    ----------
    qs_j : np.ndarray
        Other agent's belief state (num_states,)
    B_j : np.ndarray
        Other agent's transition model (num_states, num_states, num_actions)
    C_j : np.ndarray
        Other agent's preference vector (num_obs,)
    policies_j : np.ndarray
        Other agent's policy set (num_policies, horizon, num_state_factors)
    A_j : np.ndarray, optional
        Other agent's observation model (num_obs, num_states)

    Returns
    -------
    G_j : np.ndarray
        Other agent's EFE for each of their policies (num_policies,)
    """
    return compute_risk_G(qs_j, B_j, C_j, policies_j, A_j)


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
    G_j : np.ndarray
        Agent j's EFE (as simulated by i) (num_policies,)
    G_social : np.ndarray
        Empathy-weighted EFE: G_social = G_i + α·G_j
    """
    # Compute agent i's own EFE
    G_i = compute_risk_G(qs_i, B_i, C_i, policies_i, A_i)

    # Compute agent j's EFE (Theory of Mind)
    G_j = compute_other_agent_G(qs_j, B_j, C_j, policies_j, A_j)

    # Empathy-weighted social EFE
    G_social = G_i + alpha * G_j

    return G_i, G_j, G_social


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
