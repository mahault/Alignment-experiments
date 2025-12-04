# src/metrics/path_flexibility.py

"""
Path Flexibility Metrics: Empowerment, Returnability, Overlap

F(π) = λ_E · E(π) + λ_R · R(π) + λ_O · O(π)

Where:
- E(π): Expected empowerment along policy trajectory
- R(π): Returnability—probability of reaching shared safe outcomes
- O(π): Outcome overlap—overlap in predicted outcome distributions
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from src.metrics.empowerment import estimate_empowerment_one_step
from src.common.types import FlexibilityMetrics

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def compute_empowerment_along_path(
    agent_model: Any,
    policy: Any,
    horizon: int,
    current_qs: np.ndarray,
) -> Tuple[float, List[float]]:
    """
    Compute expected empowerment along a policy trajectory.

    For each timestep t in the policy:
    1. Predict next state distribution q(s_{t+1} | π, s_t)
    2. Compute p(o_{t+1} | a_t, s_{t+1}) from A matrix
    3. Marginalize to get p(o_{t+1} | a_t) = sum_s p(o|a,s) q(s|a)
    4. Compute empowerment E_t = I(A_t; O_{t+1})
    5. Return average empowerment over trajectory

    Parameters
    ----------
    agent_model : Any
        Agent's generative model (A, B matrices, policies)
    policy : Any
        Policy (sequence of actions) to evaluate
    horizon : int
        Number of timesteps to roll out
    current_qs : np.ndarray
        Current belief state q(s_t)

    Returns
    -------
    avg_empowerment : float
        Average empowerment over the trajectory
    empowerment_per_step : List[float]
        Empowerment at each timestep
    """
    LOGGER.debug(f"Computing empowerment along path: horizon={horizon}")

    empowerment_per_step = []
    qs = current_qs.copy()

    for t in range(horizon):
        # Get action at this timestep
        if isinstance(policy, (list, tuple)):
            action = policy[t] if t < len(policy) else policy[-1]
        else:
            # Policy is an index, get action sequence from agent
            action = agent_model.policies[policy][t] if t < len(agent_model.policies[policy]) else agent_model.policies[policy][-1]

        # Compute transition probabilities p(o_{t+1} | a_t)
        # This requires marginalizing over states:
        # p(o | a) = sum_s' p(o | s') * p(s' | s, a)
        transition_logits = _compute_transition_probabilities(
            agent_model, action, qs
        )

        # Compute empowerment for this step
        emp_t = estimate_empowerment_one_step(transition_logits)
        empowerment_per_step.append(emp_t)

        LOGGER.debug(f"  t={t}: empowerment={emp_t:.4f}")

        # Update belief state for next step
        # q(s_{t+1}) = sum_s_t B(s_{t+1} | s_t, a_t) * q(s_t)
        if hasattr(agent_model, 'B') and len(agent_model.B) > 0:
            B_matrix = agent_model.B[0]  # Assuming single state factor
            qs = B_matrix[:, :, action] @ qs
            qs = qs / (qs.sum() + 1e-12)  # Normalize

    avg_empowerment = float(np.mean(empowerment_per_step))
    LOGGER.debug(f"Average empowerment: {avg_empowerment:.4f}")

    return avg_empowerment, empowerment_per_step


def compute_returnability(
    agent_model: Any,
    policy: Any,
    shared_outcome_set: List[int],
    horizon: int,
    current_qs: np.ndarray,
) -> Tuple[float, List[float]]:
    """
    Compute returnability: probability of reaching shared "safe" outcomes.

    R(π) = sum_t Pr(O_t in shared_outcomes | π)

    Parameters
    ----------
    agent_model : Any
        Agent's generative model
    policy : Any
        Policy to evaluate
    shared_outcome_set : List[int]
        Set of observation indices considered "safe" or "returnable"
    horizon : int
        Number of timesteps
    current_qs : np.ndarray
        Current belief state

    Returns
    -------
    total_returnability : float
        Sum of probabilities of being in shared outcomes over time
    returnability_per_step : List[float]
        Probability at each timestep
    """
    LOGGER.debug(f"Computing returnability: shared_outcomes={shared_outcome_set}")

    returnability_per_step = []
    qs = current_qs.copy()

    for t in range(horizon):
        # Get action
        if isinstance(policy, (list, tuple)):
            action = policy[t] if t < len(policy) else policy[-1]
        else:
            action = agent_model.policies[policy][t] if t < len(agent_model.policies[policy]) else agent_model.policies[policy][-1]

        # Predict observation distribution p(o_t | π)
        # p(o) = sum_s p(o | s) * q(s)
        A_matrix = agent_model.A[0] if hasattr(agent_model, 'A') and len(agent_model.A) > 0 else np.eye(len(qs))
        p_o = A_matrix @ qs

        # Probability of being in shared outcome set
        prob_shared = sum(p_o[o] for o in shared_outcome_set if o < len(p_o))
        returnability_per_step.append(float(prob_shared))

        LOGGER.debug(f"  t={t}: returnability={prob_shared:.4f}")

        # Update belief state
        if hasattr(agent_model, 'B') and len(agent_model.B) > 0:
            B_matrix = agent_model.B[0]
            qs = B_matrix[:, :, action] @ qs
            qs = qs / (qs.sum() + 1e-12)

    total_returnability = float(np.sum(returnability_per_step))
    LOGGER.debug(f"Total returnability: {total_returnability:.4f}")

    return total_returnability, returnability_per_step


def compute_overlap(
    agent_i_model: Any,
    agent_j_model: Any,
    policy_i: Any,
    policy_j: Any,
    horizon: int,
    current_qs_i: np.ndarray,
    current_qs_j: np.ndarray,
) -> Tuple[float, List[float]]:
    """
    Compute outcome overlap between two agents' predicted trajectories.

    O_ij(π) = sum_t sum_o min(p_i(o|t,π), p_j(o|t,π))

    Measures how much agents' predicted outcome distributions overlap.

    Parameters
    ----------
    agent_i_model : Any
        Agent i's generative model
    agent_j_model : Any
        Agent j's generative model
    policy_i : Any
        Agent i's policy
    policy_j : Any
        Agent j's policy
    horizon : int
        Number of timesteps
    current_qs_i : np.ndarray
        Agent i's current belief state
    current_qs_j : np.ndarray
        Agent j's current belief state

    Returns
    -------
    total_overlap : float
        Sum of outcome overlaps over time
    overlap_per_step : List[float]
        Overlap at each timestep
    """
    LOGGER.debug("Computing outcome overlap between agents")

    overlap_per_step = []
    qs_i = current_qs_i.copy()
    qs_j = current_qs_j.copy()

    for t in range(horizon):
        # Get actions
        action_i = _get_action_at_timestep(agent_i_model, policy_i, t)
        action_j = _get_action_at_timestep(agent_j_model, policy_j, t)

        # Predict observation distributions
        A_i = agent_i_model.A[0] if hasattr(agent_i_model, 'A') and len(agent_i_model.A) > 0 else np.eye(len(qs_i))
        A_j = agent_j_model.A[0] if hasattr(agent_j_model, 'A') and len(agent_j_model.A) > 0 else np.eye(len(qs_j))

        p_o_i = A_i @ qs_i
        p_o_j = A_j @ qs_j

        # Compute overlap: sum_o min(p_i(o), p_j(o))
        num_obs = min(len(p_o_i), len(p_o_j))
        overlap_t = sum(min(p_o_i[o], p_o_j[o]) for o in range(num_obs))
        overlap_per_step.append(float(overlap_t))

        LOGGER.debug(f"  t={t}: overlap={overlap_t:.4f}")

        # Update belief states
        if hasattr(agent_i_model, 'B') and len(agent_i_model.B) > 0:
            B_i = agent_i_model.B[0]
            qs_i = B_i[:, :, action_i] @ qs_i
            qs_i = qs_i / (qs_i.sum() + 1e-12)

        if hasattr(agent_j_model, 'B') and len(agent_j_model.B) > 0:
            B_j = agent_j_model.B[0]
            qs_j = B_j[:, :, action_j] @ qs_j
            qs_j = qs_j / (qs_j.sum() + 1e-12)

    total_overlap = float(np.sum(overlap_per_step))
    LOGGER.debug(f"Total overlap: {total_overlap:.4f}")

    return total_overlap, overlap_per_step


def compute_path_flexibility(
    agent_i_model: Any,
    agent_j_model: Any,
    policy_i: Any,
    policy_j: Any,
    horizon: int,
    current_qs_i: np.ndarray,
    current_qs_j: np.ndarray,
    shared_outcome_set: List[int],
    lambda_E: float = 1.0,
    lambda_R: float = 1.0,
    lambda_O: float = 1.0,
) -> Tuple[FlexibilityMetrics, FlexibilityMetrics]:
    """
    Compute complete path flexibility metrics for both agents.

    F(π) = λ_E · E(π) + λ_R · R(π) + λ_O · O(π)

    Parameters
    ----------
    agent_i_model, agent_j_model : Any
        Generative models for both agents
    policy_i, policy_j : Any
        Policies to evaluate
    horizon : int
        Planning horizon
    current_qs_i, current_qs_j : np.ndarray
        Current belief states
    shared_outcome_set : List[int]
        Shared "safe" outcomes for returnability
    lambda_E, lambda_R, lambda_O : float
        Weights for empowerment, returnability, overlap

    Returns
    -------
    flex_i : FlexibilityMetrics
        Agent i's flexibility metrics
    flex_j : FlexibilityMetrics
        Agent j's flexibility metrics
    """
    LOGGER.info("Computing path flexibility for both agents")

    # Compute empowerment for each agent
    E_i, emp_steps_i = compute_empowerment_along_path(
        agent_i_model, policy_i, horizon, current_qs_i
    )
    E_j, emp_steps_j = compute_empowerment_along_path(
        agent_j_model, policy_j, horizon, current_qs_j
    )

    # Compute returnability for each agent
    R_i, ret_steps_i = compute_returnability(
        agent_i_model, policy_i, shared_outcome_set, horizon, current_qs_i
    )
    R_j, ret_steps_j = compute_returnability(
        agent_j_model, policy_j, shared_outcome_set, horizon, current_qs_j
    )

    # Compute overlap (same for both agents, it's a joint metric)
    O_ij, overlap_steps = compute_overlap(
        agent_i_model, agent_j_model,
        policy_i, policy_j,
        horizon, current_qs_i, current_qs_j
    )

    # Combined flexibility metrics
    F_i = lambda_E * E_i + lambda_R * R_i + lambda_O * O_ij
    F_j = lambda_E * E_j + lambda_R * R_j + lambda_O * O_ij

    LOGGER.info(f"  Agent i: E={E_i:.4f}, R={R_i:.4f}, O={O_ij:.4f} → F={F_i:.4f}")
    LOGGER.info(f"  Agent j: E={E_j:.4f}, R={R_j:.4f}, O={O_ij:.4f} → F={F_j:.4f}")

    # Package into FlexibilityMetrics dataclasses
    flex_i = FlexibilityMetrics(
        policy_idx=policy_i if isinstance(policy_i, int) else -1,
        empowerment=E_i,
        returnability=R_i,
        overlap=O_ij,
        flexibility=F_i,
        empowerment_per_step=emp_steps_i,
        returnability_per_step=ret_steps_i,
        overlap_per_step=overlap_steps,
    )

    flex_j = FlexibilityMetrics(
        policy_idx=policy_j if isinstance(policy_j, int) else -1,
        empowerment=E_j,
        returnability=R_j,
        overlap=O_ij,
        flexibility=F_j,
        empowerment_per_step=emp_steps_j,
        returnability_per_step=ret_steps_j,
        overlap_per_step=overlap_steps,
    )

    return flex_i, flex_j


# =============================================================================
# Helper functions
# =============================================================================

def _compute_transition_probabilities(
    agent_model: Any,
    action: int,
    qs: np.ndarray,
) -> np.ndarray:
    """
    Compute p(o | a) by marginalizing over states.

    p(o | a) = sum_s' p(o | s') * p(s' | s, a)
             = sum_s' A(o | s') * [B(s' | s, a) @ q(s)]

    Returns
    -------
    transition_logits : np.ndarray
        Array of shape [num_actions, num_observations]
        (For empowerment, we evaluate multiple hypothetical actions)
    """
    A = agent_model.A[0] if hasattr(agent_model, 'A') and len(agent_model.A) > 0 else np.eye(len(qs))
    B = agent_model.B[0] if hasattr(agent_model, 'B') and len(agent_model.B) > 0 else np.eye(len(qs))

    num_actions = B.shape[2] if len(B.shape) > 2 else 1
    num_obs = A.shape[0]

    transition_logits = np.zeros((num_actions, num_obs))

    for a in range(num_actions):
        # Next state distribution: q(s' | s, a)
        qs_next = B[:, :, a] @ qs
        qs_next = qs_next / (qs_next.sum() + 1e-12)

        # Observation distribution: p(o | s')
        p_o_given_a = A @ qs_next
        transition_logits[a] = p_o_given_a

    return transition_logits


def _get_action_at_timestep(agent_model: Any, policy: Any, t: int) -> int:
    """Extract action at timestep t from policy."""
    if isinstance(policy, (list, tuple)):
        return policy[t] if t < len(policy) else policy[-1]
    else:
        # Policy is an index
        policy_seq = agent_model.policies[policy]
        return policy_seq[t] if t < len(policy_seq) else policy_seq[-1]


# =============================================================================
# ToM Tree Integration (for experiments)
# =============================================================================

def compute_path_flexibility_for_tree(
    focal_tree: Any,
    other_tree: Any,
    focal_agent_idx: int,
    other_agent_idx: int,
    shared_outcome_set: List[int],
    horizon: int,
    lambdas: Tuple[float, float, float],
) -> List[Dict]:
    """
    Extract EFE and compute path flexibility for all policies in ToM tree.

    This is the main integration function called by experiments.
    Given a ToM tree (output of si_policy_search_tom), extract:
    - G_i(π), G_j(π) for each policy π
    - Compute F_i(π), F_j(π) using path flexibility metrics

    Parameters
    ----------
    focal_tree : Any
        ToM tree for focal agent (contains policies, EFEs, beliefs)
    other_tree : Any
        ToM tree for other agent
    focal_agent_idx : int
        Index of focal agent (typically 0)
    other_agent_idx : int
        Index of other agent (typically 1)
    shared_outcome_set : List[int]
        Observation indices considered "safe" for returnability
    horizon : int
        Planning horizon
    lambdas : Tuple[float, float, float]
        (λ_E, λ_R, λ_O) weights for flexibility components

    Returns
    -------
    metrics_per_policy : List[Dict]
        For each policy π in the tree:
        {
            "policy_id": int,
            "G_i": float,           # Focal agent's EFE
            "G_j": float,           # Other agent's EFE
            "G_joint": float,       # G_i + G_j
            "F_i": float,           # Focal agent's flexibility
            "F_j": float,           # Other agent's flexibility
            "F_joint": float,       # F_i + F_j
            "E_i": float,           # Focal agent's empowerment
            "E_j": float,           # Other agent's empowerment
            "R_i": float,           # Focal agent's returnability
            "R_j": float,           # Other agent's returnability
            "O_ij": float,          # Outcome overlap
        }
    """
    LOGGER.info("Computing path flexibility for all policies in ToM tree")

    # TODO: This is a STUB that needs to be implemented based on your ToM tree structure
    # The actual implementation depends on how si_tom.py stores policies and EFEs
    #
    # Expected ToM tree attributes (based on si_tom code):
    # - focal_tree.policies: list of policy sequences
    # - focal_tree.G: array of EFE values per policy [num_policies]
    # - focal_tree.qs: belief state distribution
    # - focal_tree.A: observation likelihood matrix
    # - focal_tree.B: transition matrix
    #
    # other_tree should have similar structure for the other agent

    LOGGER.warning("compute_path_flexibility_for_tree is currently a STUB")
    LOGGER.warning("TODO: Implement based on actual ToM tree structure from si_tom.py")

    # STUB: Return placeholder metrics
    # In reality, you would:
    # 1. Extract num_policies from focal_tree
    # 2. For each policy π:
    #    a. Extract G_i from focal_tree.G[π]
    #    b. Extract G_j from other_tree.G[π] (via ToM)
    #    c. Compute F_i, F_j using compute_path_flexibility()
    # 3. Package into list of dicts

    num_policies = 10  # placeholder
    lambda_E, lambda_R, lambda_O = lambdas

    metrics_per_policy = []
    for policy_idx in range(num_policies):
        # TODO: Replace with actual extraction from tree
        G_i = 0.0  # focal_tree.G[policy_idx]
        G_j = 0.0  # other_tree.G[policy_idx]

        # TODO: Replace with actual flexibility computation
        # flex_i, flex_j = compute_path_flexibility(
        #     agent_i_model=focal_tree,
        #     agent_j_model=other_tree,
        #     policy_i=policy_idx,
        #     policy_j=policy_idx,  # Assuming same policy index
        #     horizon=horizon,
        #     current_qs_i=focal_tree.qs,
        #     current_qs_j=other_tree.qs,
        #     shared_outcome_set=shared_outcome_set,
        #     lambda_E=lambda_E,
        #     lambda_R=lambda_R,
        #     lambda_O=lambda_O,
        # )

        # Placeholder
        E_i, E_j = 0.0, 0.0
        R_i, R_j = 0.0, 0.0
        O_ij = 0.0
        F_i = lambda_E * E_i + lambda_R * R_i + lambda_O * O_ij
        F_j = lambda_E * E_j + lambda_R * R_j + lambda_O * O_ij

        metrics_per_policy.append({
            "policy_id": policy_idx,
            "G_i": float(G_i),
            "G_j": float(G_j),
            "G_joint": float(G_i + G_j),
            "F_i": float(F_i),
            "F_j": float(F_j),
            "F_joint": float(F_i + F_j),
            "E_i": float(E_i),
            "E_j": float(E_j),
            "R_i": float(R_i),
            "R_j": float(R_j),
            "O_ij": float(O_ij),
        })

    LOGGER.info(f"Computed metrics for {len(metrics_per_policy)} policies")
    return metrics_per_policy
