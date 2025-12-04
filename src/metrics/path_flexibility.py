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

def root_idx(tree: Any) -> int:
    """
    Find the root node index of the ToM tree.

    This is typically stored in the tree structure or is node 0.
    """
    # TODO: Adjust based on actual tree structure
    # For now assume root is at index 0
    return 0


def get_root_policy_nodes(tree: Any, focal_agent_idx: int) -> List[int]:
    """
    Identify indices of root-level policy nodes for the focal agent.

    - Start from the root node
    - Take its children_indices
    - Keep those that are used, belong to focal agent, and have a valid policy

    Parameters
    ----------
    tree : Any
        ToM tree structure with attributes:
        - children_indices: child node IDs
        - used: boolean mask of active nodes
        - agent_idx: which agent each node belongs to
        - policy: policy vector at each node
    focal_agent_idx : int
        Index of focal agent

    Returns
    -------
    valid_nodes : List[int]
        Indices of root-level policy nodes for focal agent
    """
    import jax.numpy as jnp

    root_index = root_idx(tree)

    child_indices = tree.children_indices[0, root_index]  # [max_branching]

    # Filter valid children
    valid_nodes = []
    for node_id in child_indices:
        if node_id < 0:
            continue
        if not bool(tree.used[0, node_id, 0]):
            continue
        if int(tree.agent_idx[0, node_id, 0]) != focal_agent_idx:
            continue
        # policy node (not observation)
        if (tree.policy[0, node_id] >= 0).any():
            valid_nodes.append(int(node_id))

    LOGGER.debug(f"Found {len(valid_nodes)} root policy nodes for agent {focal_agent_idx}")
    return valid_nodes


def predict_obs_dist(agent_model: Any, policy_id: int, t: int) -> np.ndarray:
    """
    Predict observation distribution p(o_t | π) at timestep t.

    TODO: Implement based on actual agent model structure.
    This should use the generative model (A, B matrices) to forward-simulate.
    """
    # Placeholder implementation
    num_obs = agent_model.A[0].shape[0] if hasattr(agent_model, 'A') else 10
    return np.ones(num_obs) / num_obs


def simulate_policy_and_compute_rollout_dists(
    policy_id: int,
    agent_model: Any,
    horizon: int,
) -> List[np.ndarray]:
    """
    Given a policy identifier (e.g. index into agent_model.policies),
    run the generative model forward to get p_i(o_t | π) for each t.

    Parameters
    ----------
    policy_id : int
        Index of policy in agent_model.policies
    agent_model : Any
        Agent's generative model with A, B matrices and policies
    horizon : int
        Number of timesteps to simulate

    Returns
    -------
    p_obs_over_time : List[np.ndarray]
        Predicted observation distributions [p(o_0|π), p(o_1|π), ...]
    """
    p_obs_over_time = []

    for t in range(horizon):
        # Predict observation distribution at timestep t
        p_o_t = predict_obs_dist(agent_model, policy_id, t)
        p_obs_over_time.append(p_o_t)

    return p_obs_over_time


def approximate_EFE_step(p_o_t: np.ndarray, preferences_C: np.ndarray) -> float:
    """
    Approximate EFE contribution at a single timestep.

    Simplified version: G_t ≈ -E[log C(o)]

    TODO: Full EFE includes epistemic and pragmatic terms.
    """
    # Risk term: expected negative log preference
    log_C = np.log(preferences_C + 1e-12)
    G_t = -np.sum(p_o_t * log_C)
    return float(G_t)


def compute_EFE_from_rollout(
    p_obs_over_time: List[np.ndarray],
    preferences_C: np.ndarray
) -> float:
    """
    Approximate EFE given predicted observation distributions and preferences.

    G(π) ≈ sum_t E_q(π)[- log C(o_t)]

    Parameters
    ----------
    p_obs_over_time : List[np.ndarray]
        Predicted observation distributions over time
    preferences_C : np.ndarray
        Preference vector over observations

    Returns
    -------
    G : float
        Expected free energy
    """
    G = 0.0
    for p_o_t in p_obs_over_time:
        G_t = approximate_EFE_step(p_o_t, preferences_C)
        G += G_t
    return float(G)


def get_p_o_given_a(agent_model: Any, policy_id: int, t: int) -> np.ndarray:
    """
    Get p(o | a) for empowerment computation at timestep t.

    TODO: Implement based on actual agent model structure.
    Should return [num_actions, num_obs] array.
    """
    # Placeholder: return uniform distribution
    num_actions = 4  # TODO: extract from model
    num_obs = agent_model.A[0].shape[0] if hasattr(agent_model, 'A') else 10
    return np.ones((num_actions, num_obs)) / num_obs


def compute_empowerment_along_rollout(
    agent_model: Any,
    policy_id: int,
    horizon: int
) -> float:
    """
    Compute average empowerment along a policy rollout.

    E(π) = (1/T) sum_t E_t
    where E_t = I(A_t; O_{t+1})

    Parameters
    ----------
    agent_model : Any
        Agent's generative model
    policy_id : int
        Index of policy
    horizon : int
        Planning horizon

    Returns
    -------
    E : float
        Average empowerment
    """
    from .empowerment import estimate_empowerment_one_step

    E_t = []
    for t in range(horizon):
        # Build p(o_{t+1} | a_t) given current belief and model
        p_o_given_a = get_p_o_given_a(agent_model, policy_id, t)
        E_t.append(estimate_empowerment_one_step(p_o_given_a))

    return float(np.mean(E_t))


def compute_returnability_from_rollout(
    p_obs_over_time: List[np.ndarray],
    shared_outcome_set: List[int],
) -> float:
    """
    R(π) = sum_t Pr(O_t in shared_outcomes | π)

    Parameters
    ----------
    p_obs_over_time : List[np.ndarray]
        Predicted observation distributions over time
    shared_outcome_set : List[int]
        Indices of "safe" observations

    Returns
    -------
    R : float
        Returnability score
    """
    R = 0.0
    for p_o_t in p_obs_over_time:
        R_t = sum(p_o_t[o] for o in shared_outcome_set if o < len(p_o_t))
        R += float(R_t)
    return R


def compute_overlap_from_two_rollouts(
    p_obs_i: List[np.ndarray],
    p_obs_j: List[np.ndarray]
) -> float:
    """
    O_ij(π) = sum_t sum_o min(p_i(o|t), p_j(o|t))

    Parameters
    ----------
    p_obs_i : List[np.ndarray]
        Focal agent's observation distributions over time
    p_obs_j : List[np.ndarray]
        Other agent's observation distributions over time

    Returns
    -------
    O : float
        Overlap score
    """
    O = 0.0
    for p_i_t, p_j_t in zip(p_obs_i, p_obs_j):
        for o in range(min(len(p_i_t), len(p_j_t))):
            O += float(min(p_i_t[o], p_j_t[o]))
    return O


def compute_path_flexibility_for_tree(
    focal_tree: Any,
    other_tree: Any,
    focal_agent_model: Any,
    other_agent_model: Any,
    focal_agent_idx: int,
    other_agent_idx: int,
    shared_outcome_set: List[int],
    horizon: int,
    lambdas: Tuple[float, float, float],
) -> List[Dict]:
    """
    Compute G_i, G_j, E, R, O, F for each root-level policy considered
    by the focal agent in the ToM tree.

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
    focal_agent_model : Any
        Focal agent's generative model (A, B matrices, policies, preferences)
    other_agent_model : Any
        Other agent's generative model
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
            "node_id": int,
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
    import jax.numpy as jnp

    LOGGER.info("Computing path flexibility for all policies in ToM tree")

    λE, λR, λO = lambdas

    # Get root policy nodes for focal agent
    root_policy_nodes = get_root_policy_nodes(focal_tree, focal_agent_idx)

    results = []

    for node_id in root_policy_nodes:
        LOGGER.debug(f"Processing policy node {node_id}")

        # 1) Identify which primitive policy this node corresponds to
        #    Each node stores a policy vector (e.g., one-hot over actions/policies)
        policy_vector = focal_tree.policy[0, node_id]  # [n_actions] or [n_policies]
        policy_id = int(jnp.argmax(policy_vector))

        # 2) Compute or read own EFE G_i(π)
        #    Option A: read from tree.G at this node
        G_i = float(focal_tree.G[0, node_id, 0])

        # 3) Compute rollout distributions for i and j
        p_obs_i = simulate_policy_and_compute_rollout_dists(
            policy_id, focal_agent_model, horizon
        )
        p_obs_j = simulate_policy_and_compute_rollout_dists(
            policy_id, other_agent_model, horizon
        )

        # 4) Compute other's EFE G_j(π)
        #    Assume j has its own preferences C_j
        preferences_j = (
            other_agent_model.C[0]
            if hasattr(other_agent_model, 'C')
            else np.ones(len(p_obs_j[0]))
        )
        G_j = compute_EFE_from_rollout(p_obs_j, preferences_j)

        # 5) Empowerment along rollout
        E_i = compute_empowerment_along_rollout(focal_agent_model, policy_id, horizon)
        E_j = compute_empowerment_along_rollout(other_agent_model, policy_id, horizon)

        # 6) Returnability
        R_i = compute_returnability_from_rollout(p_obs_i, shared_outcome_set)
        R_j = compute_returnability_from_rollout(p_obs_j, shared_outcome_set)

        # 7) Overlap (shared outcomes)
        O_ij = compute_overlap_from_two_rollouts(p_obs_i, p_obs_j)

        # 8) Path flexibility metrics
        F_i = λE * E_i + λR * R_i + λO * O_ij
        F_j = λE * E_j + λR * R_j + λO * O_ij

        LOGGER.debug(
            f"  Policy {policy_id}: "
            f"G_i={G_i:.3f}, G_j={G_j:.3f}, "
            f"E_i={E_i:.3f}, E_j={E_j:.3f}, "
            f"R_i={R_i:.3f}, R_j={R_j:.3f}, "
            f"O_ij={O_ij:.3f}, "
            f"F_i={F_i:.3f}, F_j={F_j:.3f}"
        )

        results.append({
            "node_id": node_id,
            "policy_id": policy_id,
            "G_i": G_i,
            "G_j": G_j,
            "G_joint": G_i + G_j,
            "E_i": E_i,
            "E_j": E_j,
            "R_i": R_i,
            "R_j": R_j,
            "O_ij": O_ij,
            "F_i": F_i,
            "F_j": F_j,
            "F_joint": F_i + F_j,
        })

    LOGGER.info(f"Computed metrics for {len(results)} policies")
    return results
