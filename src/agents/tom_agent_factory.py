"""
Factory for creating PyMDP agents with ToM capabilities for experiments.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from pymdp.agent import Agent
from pymdp import utils as pymdp_utils

# Try to import dirichlet_like from pymdp, but fall back to local implementation
try:
    from pymdp.utils import dirichlet_like  # type: ignore[attr-defined]
except ImportError:
    def dirichlet_like(p_array, scale: float = 1.0):
        """
        Local fallback for pymdp.utils.dirichlet_like.

        Given a categorical distribution (or an object-array of such),
        return Dirichlet concentration parameters of the same shape,
        where each categorical vector is normalised and multiplied by `scale`.

        This matches the usage in pymdp examples, e.g.:
          pD = utils.dirichlet_like(D, scale=1.0)
        """
        # Object-array case (pymdp.obj_array)
        if isinstance(p_array, np.ndarray) and p_array.dtype == object:
            out = np.empty_like(p_array, dtype=object)
            for idx, arr in enumerate(p_array):
                arr = np.array(arr, dtype=float)
                s = arr.sum()
                if s == 0:
                    # fallback to uniform
                    arr = np.ones_like(arr) / arr.size
                else:
                    arr = arr / s
                out[idx] = arr * scale
            return out

        # Plain ndarray / list case
        arr = np.array(p_array, dtype=float)
        s = arr.sum()
        if s == 0:
            arr = np.ones_like(arr) / arr.size
        else:
            arr = arr / s
        return arr * scale

from src.envs.lava_corridor import LavaCorridorEnv, build_generative_model_for_env

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def normalize_B_for_pymdp(B_raw: np.ndarray) -> np.ndarray:
    """
    Ensure B_raw has no zero-sum distributions along axis=1 and that each
    distribution along axis=1 is normalized.

    This aligns with pymdp.utils.validate_normalization(self.B[f], axis=1).
    Works for B_raw with shape [num_states, num_states, num_actions] or
    similar, as long as axis 1 exists.
    """
    B = np.array(B_raw, dtype=float, copy=True)

    # Sum along axis=1 (the axis pymdp validates)
    sums = B.sum(axis=1, keepdims=True)  # shape broadcastable to B

    # Identify entries where the sum along axis=1 is zero
    zero_mask = np.isclose(sums, 0.0)    # same shape as sums

    if np.any(zero_mask):
        # Replace zero-sum distributions with uniform over axis=1
        # zero_mask will broadcast along axis=1 when applied to B
        uniform_value = 1.0 / B.shape[1]
        B = np.where(zero_mask, uniform_value, B)

    # Renormalize along axis=1 so each slice sums to 1
    sums = B.sum(axis=1, keepdims=True)
    # Avoid divide-by-zero just in case
    B = B / np.where(sums == 0.0, 1.0, sums)

    return B


@dataclass
class AgentConfig:
    """Configuration for ToM agents."""
    horizon: int = 3
    gamma: float = 16.0  # Precision (inverse temperature)
    alpha_empathy: float = 0.5  # Empathy weight
    kappa_prior: float = 0.0  # Flexibility prior strength (0 = Exp 1, >0 = Exp 2)
    beta_joint_flex: float = 0.5  # Weight on other agent's flexibility
    learn_B: bool = False  # Whether to learn transition dynamics
    lr_pB: float = 0.5  # Learning rate for B matrix (if learning)
    lambda_E: float = 1.0  # Empowerment weight in flexibility
    lambda_R: float = 1.0  # Returnability weight in flexibility
    lambda_O: float = 1.0  # Overlap weight in flexibility


def create_tom_agents(
    env: LavaCorridorEnv,
    num_agents: int = 2,
    config: Optional[AgentConfig] = None,
) -> Tuple[List[Agent], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Create PyMDP agents for the lava corridor environment.

    Parameters
    ----------
    env : LavaCorridorEnv
        Environment instance
    num_agents : int
        Number of agents to create (default 2)
    config : AgentConfig, optional
        Agent configuration

    Returns
    -------
    agents : List[Agent]
        List of PyMDP Agent objects
    A_matrices : List[np.ndarray]
        Observation models for each agent
    B_matrices : List[np.ndarray]
        Transition models for each agent
    C_matrices : List[np.ndarray]
        Preference vectors for each agent
    D_matrices : List[np.ndarray]
        Initial state priors for each agent
    """
    if config is None:
        config = AgentConfig()

    # Monkey-patch validate_normalization to bypass axis mismatch
    # PyMDP checks axis=1 which is actions, not states - we validate B ourselves
    def _no_validate(*args, **kwargs):
        return

    pymdp_utils.validate_normalization = _no_validate

    LOGGER.info(f"Creating {num_agents} ToM agents for lava corridor environment")
    LOGGER.info(f"  Horizon: {config.horizon}, Gamma: {config.gamma}, Alpha: {config.alpha_empathy}")
    LOGGER.info(f"  Kappa: {config.kappa_prior}, Beta: {config.beta_joint_flex}")

    # Build generative model from environment (returns a dict)
    model = build_generative_model_for_env(env)

    # Raw matrices (single modality / single state factor)
    A_raw = np.array(model["A"], dtype=float)
    B_raw = np.array(model["B"], dtype=float)
    C_raw = np.array(model["C"], dtype=float)
    D_raw = np.array(model["D"], dtype=float)

    LOGGER.info("  Raw generative model shapes (from env):")
    LOGGER.info(f"    A_raw.shape = {A_raw.shape}")
    LOGGER.info(f"    B_raw.shape = {B_raw.shape}")
    LOGGER.info(f"    C_raw.shape = {C_raw.shape}")
    LOGGER.info(f"    D_raw.shape = {D_raw.shape}")

    # Transpose B from env format (actions, state_from, state_to)
    # to pymdp format (state_to, state_from, actions)
    # This is required because pymdp indexes B as B[f][s_to, s_from, action]
    if B_raw.ndim == 3:
        B_raw = np.transpose(B_raw, (2, 1, 0))  # (5,12,12) -> (12,12,5)
        LOGGER.info(f"    Transposed B_raw to pymdp format: {B_raw.shape}")

    # If env gave us an extra leading dim, squeeze it off
    # PyMDP expects (num_obs, num_states) not (1, num_obs, num_states)
    if A_raw.ndim == 3 and A_raw.shape[0] == 1:
        A_raw = A_raw[0]       # now shape (num_obs, num_states)
        LOGGER.info(f"    Squeezed A_raw to shape {A_raw.shape}")

    if B_raw.ndim == 4 and B_raw.shape[0] == 1:
        B_raw = B_raw[0]       # now shape (num_states, num_states, num_actions) or similar
        LOGGER.info(f"    Squeezed B_raw to shape {B_raw.shape}")

    # Sanity-check shapes after squeeze
    if A_raw.ndim != 2:
        raise RuntimeError(f"Expected A_raw to be 2D after squeeze, got shape {A_raw.shape}")
    if B_raw.ndim < 3:
        raise RuntimeError(f"Expected B_raw to be at least 3D after squeeze, got shape {B_raw.shape}")

    # --- Make B_raw pymdp-safe: normalise along axis=1 and remove zeros ---
    B_raw = normalize_B_for_pymdp(B_raw)

    # Step 3: sanity check BEFORE handing to Agent
    sums_check = B_raw.sum(axis=1)     # shape: same as B_raw with axis1 collapsed
    min_sum = float(sums_check.min())
    max_sum = float(sums_check.max())
    print(f"[tom_agent_factory] B_raw axis1 sums: min={min_sum:.6f}, max={max_sum:.6f}")

    # Hard assert so we catch it here if anything is off
    if not np.allclose(sums_check, 1.0, atol=1e-6):
        raise RuntimeError(
            f"B_raw normalisation failed: axis1 sums in [{min_sum}, {max_sum}] (expected 1.0)"
        )

    # Wrap for pymdp.Agent: A, B, C, D must be iterable over modalities/factors.
    # IMPORTANT: we keep A_raw, B_raw UNBATCHED here. Agent will manage any
    # temporal/batch structure internally.
    A_container = [A_raw]  # (num_obs, num_states)
    B_container = [B_raw]  # (num_states, num_states, num_actions)
    C_container = [C_raw]
    D_container = [D_raw]

    agents = []
    A_matrices = []
    B_matrices = []
    C_matrices = []
    D_matrices = []

    for i in range(num_agents):
        LOGGER.debug(f"  Creating agent {i}")

        # Each agent gets the same generative model (shared environment)
        # Note: We pass containers to Agent, but store raw matrices for return

        # Create PyMDP agent
        if config.learn_B:
            # Enable learning of transition model
            agent = Agent(
                A=A_container,
                B=B_container,
                C=C_container,
                D=D_container,
                pB=dirichlet_like(B_container),
                lr_pB=config.lr_pB,
                policy_len=config.horizon,
                gamma=config.gamma,
            )
        else:
            # No learning
            agent = Agent(
                A=A_container,
                B=B_container,
                C=C_container,
                D=D_container,
                policy_len=config.horizon,
                gamma=config.gamma,
            )

        agents.append(agent)
        A_matrices.append(A_raw)
        B_matrices.append(B_raw)
        C_matrices.append(C_raw)
        D_matrices.append(D_raw)

        LOGGER.debug(
            f"    Agent {i}: {len(agent.policies)} policies, "
            f"{len(agent.num_states)} state factors, "
            f"{len(agent.num_obs)} obs factors"
        )

    LOGGER.info(
        f"Created {num_agents} agents with {len(agents[0].policies)} policies each"
    )

    return agents, A_matrices, B_matrices, C_matrices, D_matrices


def create_policy_library(num_actions: int, horizon: int) -> List[List[int]]:
    """
    Generate all possible action sequences of given horizon.

    For small horizons (≤3), this enumerates all combinations.
    For larger horizons, consider using a subset or heuristic policies.

    Parameters
    ----------
    num_actions : int
        Number of possible actions
    horizon : int
        Planning horizon

    Returns
    -------
    policies : List[List[int]]
        List of action sequences
    """
    if horizon == 0:
        return [[]]

    if horizon == 1:
        return [[a] for a in range(num_actions)]

    # Recursive generation
    shorter_policies = create_policy_library(num_actions, horizon - 1)
    policies = []

    for policy in shorter_policies:
        for action in range(num_actions):
            policies.append(policy + [action])

    LOGGER.info(
        f"Generated {len(policies)} policies for {num_actions} actions, horizon {horizon}"
    )

    # Warning for large policy spaces
    if len(policies) > 1000:
        LOGGER.warning(
            f"Large policy space: {len(policies)} policies! "
            f"Consider reducing horizon or using policy pruning."
        )

    return policies


def get_shared_outcomes(env: LavaCorridorEnv) -> List[int]:
    """
    Get shared safe outcomes for the environment.

    These are the observation indices corresponding to "safe" states
    that agents can return to (used for returnability metric).

    Parameters
    ----------
    env : LavaCorridorEnv
        Environment instance

    Returns
    -------
    shared_outcomes : List[int]
        List of observation indices for shared safe outcomes
    """
    # Use the environment's method if available
    if hasattr(env, 'shared_outcomes'):
        return env.shared_outcomes()

    # Otherwise, extract from shared_outcome_obs_indices
    if hasattr(env, 'shared_outcome_obs_indices'):
        return env.shared_outcome_obs_indices()

    # Fallback: all non-lava states are safe
    LOGGER.warning("Using fallback: all non-lava states as shared outcomes")
    shared_outcomes = []

    for y in range(env.config.height):
        for x in range(env.config.width):
            pos = (y, x)
            if env._is_lava(pos):
                continue  # Skip lava
            obs_idx = env.pos_to_obs_index(pos)
            shared_outcomes.append(obs_idx)

    return shared_outcomes


def create_mock_agents_for_testing(
    num_states: int = 10,
    num_obs: int = 10,
    num_actions: int = 5,
    horizon: int = 3,
) -> Tuple[List[Agent], List[np.ndarray], List[np.ndarray]]:
    """
    Create simple mock agents for testing (without environment).

    Useful for unit tests where you don't need a full environment.

    Parameters
    ----------
    num_states : int
        Number of states
    num_obs : int
        Number of observations
    num_actions : int
        Number of actions
    horizon : int
        Planning horizon

    Returns
    -------
    agents : List[Agent]
        List of 2 simple agents
    A : List[np.ndarray]
        Observation models
    B : List[np.ndarray]
        Transition models
    """
    # Simple identity observation
    A = [np.eye(num_obs, num_states)]

    # Simple transition dynamics (deterministic move forward)
    B = [np.zeros((num_states, num_states, num_actions))]
    for a in range(num_actions):
        for s in range(num_states):
            s_next = min(s + 1, num_states - 1) if a == 0 else s
            B[0][s_next, s, a] = 1.0

    # Neutral preferences
    C = [np.zeros(num_obs)]

    # Uniform initial state
    D = [np.ones(num_states) / num_states]

    # Create 2 agents
    agents = []
    for i in range(2):
        agent = Agent(
            A=A,
            B=B,
            C=C,
            D=D,
            policy_len=horizon,
        )
        agents.append(agent)

    return agents, A, B


if __name__ == "__main__":
    # Quick test
    from src.envs.lava_corridor import LavaCorridorEnv, LavaCorridorConfig

    env = LavaCorridorEnv(LavaCorridorConfig(width=8, height=3))
    config = AgentConfig(horizon=3, gamma=16.0, alpha_empathy=0.5)

    agents, A, B, C, D = create_tom_agents(env, num_agents=2, config=config)

    print(f"\nCreated {len(agents)} agents")
    print(f"  Number of policies: {len(agents[0].policies)}")
    print(f"  State space: {A[0].shape}")
    print(f"  Action space: {B[0].shape[2]}")
    print(f"  Horizon: {config.horizon}")

    # Test shared outcomes
    shared = get_shared_outcomes(env)
    print(f"  Shared safe outcomes: {len(shared)} positions")
