"""
Single-agent active inference planner for Lava corridor (TOM-style).

This module provides EFE-based planning for LavaAgent without empathy or
flexibility priors. It supports:

- Risk-only Expected Free Energy (pragmatic value)
- Optional epistemic value (information gain about hidden states)

and is the foundation for more complex multi-agent planning.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from tom.models import LavaAgent


def propagate_state(
    qs0: np.ndarray,
    B: np.ndarray,
    actions: np.ndarray,
    qs_other: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Propagate belief state forward under a sequence of actions.

    Parameters
    ----------
    qs0 : np.ndarray
        Initial belief state (num_states,)
    B : np.ndarray
        Transition model. Can be 3D (num_states, num_states, num_actions) for
        single-agent, or 4D (num_states, num_states, num_states, num_actions)
        for multi-agent conditioning on other's position.
    actions : np.ndarray
        Sequence of actions (horizon,)
    qs_other : np.ndarray, optional
        Belief about other agent's position (num_states,).
        Required if B is 4D.

    Returns
    -------
    qs_final : np.ndarray
        Final belief state after applying all actions (num_states,)
    """
    qs = qs0.copy()

    # Check if B is 3D (single-agent) or 4D (multi-agent)
    if B.ndim == 3:
        # Single-agent: B[s', s, a]
        for action in actions:
            qs = B[:, :, action] @ qs
    elif B.ndim == 4:
        # Multi-agent: B[s', s, s_other, a]
        # Need to marginalize over other agent's position
        if qs_other is None:
            raise ValueError("qs_other required for 4D B matrix")

        for action in actions:
            # qs_next[s'] = sum_{s, s_other} B[s', s, s_other, a] * qs[s] * qs_other[s_other]
            # Compute as: qs_next = sum_{s_other} (B[:, :, s_other, action] @ qs) * qs_other[s_other]
            qs_next = np.zeros_like(qs)
            for s_other in range(len(qs_other)):
                qs_next += B[:, :, s_other, action] @ qs * qs_other[s_other]
            qs = qs_next
    else:
        raise ValueError(f"B must be 3D or 4D, got shape {B.shape}")

    return qs


def compute_risk_G(
    qs: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    policies: np.ndarray,
    A: Optional[np.ndarray] = None,
    qs_other: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute risk-based Expected Free Energy for each policy.

    This is the "pragmatic value" component: expected utility under preferences C:
        G(π) = -E_q(o|π)[ C(o) ]

    Parameters
    ----------
    qs : np.ndarray
        Current belief state (num_states,)
    B : np.ndarray
        Transition model. Can be 3D or 4D (if conditioning on other agent).
    C : np.ndarray
        Preference vector over observations (num_obs,)
    policies : np.ndarray
        Policy set (num_policies, horizon, num_state_factors)
    A : np.ndarray, optional
        Observation model (num_obs, num_states). If None, assumes identity.
    qs_other : np.ndarray, optional
        Belief about other agent's position. Required if B is 4D.

    Returns
    -------
    G : np.ndarray
        Risk-based EFE for each policy (num_policies,)
    """
    num_policies = len(policies)
    num_states = len(qs)

    # Default to identity observation model if not provided
    if A is None:
        A = np.eye(num_states)

    G = np.zeros(num_policies)

    for pi_idx, policy in enumerate(policies):
        # Policy shape: (horizon, num_state_factors)
        # For lava: num_state_factors = 1, so squeeze
        action_seq = policy[:, 0].astype(int)  # (horizon,)

        qs_t = qs.copy()
        expected_utility = 0.0

        for action in action_seq:
            # Predict next state (handles both 3D and 4D B)
            if B.ndim == 3:
                qs_t = B[:, :, action] @ qs_t
            elif B.ndim == 4:
                if qs_other is None:
                    raise ValueError("qs_other required for 4D B matrix")
                qs_t_next = np.zeros_like(qs_t)
                for s_other in range(len(qs_other)):
                    qs_t_next += B[:, :, s_other, action] @ qs_t * qs_other[s_other]
                qs_t = qs_t_next
            else:
                raise ValueError(f"B must be 3D or 4D, got shape {B.shape}")

            # Predict observations
            obs_dist = A @ qs_t  # p(o|qs_t)
            # (Assume A is a proper observation model; no extra normalization here)

            # Expected utility: E_o[C(o)]
            expected_utility += (obs_dist * C).sum()

        # G is negative utility (we minimize G, which maximizes utility)
        G[pi_idx] = -expected_utility

    return G


def compute_full_G(
    qs: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    policies: np.ndarray,
    A: Optional[np.ndarray] = None,
    qs_other: Optional[np.ndarray] = None,
    epistemic_scale: float = 1.0,
) -> np.ndarray:
    """
    Compute full Expected Free Energy (risk + epistemic) for each policy.

    G(π) = -E_o[C(o)] - epistemic_scale * E_o[ D_KL(q(s|o,π) || q(s|π)) ]

    This mirrors the standard pymdp-style decomposition:
      - risk term favours preferred outcomes (C)
      - epistemic term favours information gain about hidden states

    Parameters
    ----------
    qs : (num_states,)
        Current belief state.
    B : np.ndarray
        Transition model (3D or 4D).
    C : np.ndarray
        Preference vector over observations (num_obs,).
    policies : np.ndarray
        Policy set (num_policies, horizon, num_state_factors).
    A : np.ndarray, optional
        Observation model (num_obs, num_states). If None, uses identity.
    qs_other : np.ndarray, optional
        Belief about other agent (needed if B is 4D).
    epistemic_scale : float
        Weight η on the epistemic term (η=0 → risk-only).

    Returns
    -------
    G : np.ndarray
        Full EFE (risk + epistemic) for each policy (num_policies,).
    """
    num_policies = len(policies)
    num_states = len(qs)

    if A is None:
        A = np.eye(num_states)

    G = np.zeros(num_policies)
    eps = 1e-16  # numerical floor

    for pi_idx, policy in enumerate(policies):
        action_seq = policy[:, 0].astype(int)
        qs_t = qs.copy()

        expected_utility = 0.0
        expected_info_gain = 0.0

        for action in action_seq:
            # --- state prediction (3D or 4D B) ---
            if B.ndim == 3:
                qs_t = B[:, :, action] @ qs_t
            elif B.ndim == 4:
                if qs_other is None:
                    raise ValueError("qs_other required for 4D B matrix")
                qs_next = np.zeros_like(qs_t)
                for s_other in range(len(qs_other)):
                    qs_next += B[:, :, s_other, action] @ qs_t * qs_other[s_other]
                qs_t = qs_next
            else:
                raise ValueError(f"B must be 3D or 4D, got shape {B.shape}")

            # --- predict observations ---
            obs_dist = A @ qs_t
            obs_dist = np.clip(obs_dist, eps, 1.0)
            obs_dist = obs_dist / obs_dist.sum()

            # --- risk term: E_o[C(o)] ---
            expected_utility += (obs_dist * C).sum()

            # --- epistemic term: expected information gain ---
            prior = np.clip(qs_t, eps, 1.0)
            prior = prior / prior.sum()
            H_prior = -np.sum(prior * np.log(prior))

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
            expected_info_gain += info_gain

        # Minimize G: risk - η * epistemic_value
        G[pi_idx] = -expected_utility - epistemic_scale * expected_info_gain

    return G


def efe_risk_only(
    qs: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    policies: np.ndarray,
    gamma: float,
    A: Optional[np.ndarray] = None,
    qs_other: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute risk-only EFE and policy posterior.

    Parameters
    ----------
    qs : np.ndarray
        Current belief state (num_states,)
    B : np.ndarray
        Transition model (3D or 4D)
    C : np.ndarray
        Preference vector (num_obs,)
    policies : np.ndarray
        Policy set (num_policies, horizon, num_state_factors)
    gamma : float
        Policy precision (inverse temperature)
    A : np.ndarray, optional
        Observation model (num_obs, num_states)
    qs_other : np.ndarray, optional
        Belief about other agent (for 4D B)

    Returns
    -------
    G : np.ndarray
        Expected Free Energy for each policy (num_policies,)
    q_pi : np.ndarray
        Policy posterior (num_policies,)
    """
    G = compute_risk_G(qs, B, C, policies, A=A, qs_other=qs_other)

    # Policy posterior: q(π) ∝ exp(-γ * G)
    log_q_pi = -gamma * G
    log_q_pi = log_q_pi - log_q_pi.max()  # Numerical stability
    q_pi = np.exp(log_q_pi)
    q_pi = q_pi / q_pi.sum()

    return G, q_pi


def efe_with_epistemic(
    qs: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    policies: np.ndarray,
    gamma: float,
    A: Optional[np.ndarray] = None,
    qs_other: Optional[np.ndarray] = None,
    epistemic_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute full EFE (risk + epistemic) and policy posterior.

    Parameters
    ----------
    qs : np.ndarray
        Current belief state (num_states,)
    B : np.ndarray
        Transition model (3D or 4D)
    C : np.ndarray
        Preference vector (num_obs,)
    policies : np.ndarray
        Policy set (num_policies, horizon, num_state_factors)
    gamma : float
        Policy precision (inverse temperature)
    A : np.ndarray, optional
        Observation model (num_obs, num_states)
    qs_other : np.ndarray, optional
        Belief about other agent (for 4D B)
    epistemic_scale : float
        Weight η on the epistemic term

    Returns
    -------
    G : np.ndarray
        Full EFE for each policy
    q_pi : np.ndarray
        Policy posterior q(π) ∝ exp(-γ G)
    """
    G = compute_full_G(
        qs,
        B,
        C,
        policies,
        A=A,
        qs_other=qs_other,
        epistemic_scale=epistemic_scale,
    )

    log_q_pi = -gamma * G
    log_q_pi = log_q_pi - log_q_pi.max()
    q_pi = np.exp(log_q_pi)
    q_pi = q_pi / q_pi.sum()

    return G, q_pi


@dataclass
class LavaPlanner:
    """
    Single-agent active inference planner for Lava corridor.

    This planner can use:
      - risk-based EFE (pragmatic value only), or
      - full EFE (risk + epistemic information gain).

    No empathy, no flexibility prior.

    Attributes
    ----------
    agent : LavaAgent
        Agent with generative model (A, B, C, D) and policies
    use_epistemic : bool
        If True, use full EFE with epistemic term; otherwise risk-only.
    epistemic_scale : float
        Weight η on the epistemic term when use_epistemic is True.
    """
    agent: LavaAgent
    use_epistemic: bool = False
    epistemic_scale: float = 1.0

    def plan(self, qs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Select action based on current belief state.

        Parameters
        ----------
        qs : np.ndarray
            Current belief state (num_states,)

        Returns
        -------
        G : np.ndarray
            Expected Free Energy for each policy (num_policies,)
        q_pi : np.ndarray
            Policy posterior (num_policies,)
        action : int
            Selected action (first action of chosen policy)
        """
        # Extract model components
        B = np.asarray(self.agent.B["location_state"])  # (num_states, num_states, num_actions) or 4D
        C = np.asarray(self.agent.C["location_obs"])    # (num_obs,)
        A = np.asarray(self.agent.A["location_obs"])    # (num_obs, num_states)
        policies = np.asarray(self.agent.policies)      # (num_policies, horizon, num_state_factors)
        gamma = self.agent.gamma

        # For single-agent planning, we ignore qs_other
        if self.use_epistemic:
            G, q_pi = efe_with_epistemic(
                qs,
                B,
                C,
                policies,
                gamma,
                A=A,
                qs_other=None,
                epistemic_scale=self.epistemic_scale,
            )
        else:
            G, q_pi = efe_risk_only(
                qs,
                B,
                C,
                policies,
                gamma,
                A=A,
                qs_other=None,
            )

        # Select best policy (can sample from q_pi, but for now take argmax)
        best_policy_idx = int(np.argmax(q_pi))

        # Extract first action from best policy
        best_policy = policies[best_policy_idx]  # (horizon, num_state_factors)
        action = int(best_policy[0, 0])          # First timestep, first state factor

        return G, q_pi, action
