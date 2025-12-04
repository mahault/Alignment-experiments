"""
Factory for creating PyMDP agents with ToM capabilities for experiments.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from pymdp.agent import Agent
from pymdp.utils import dirichlet_like

from src.envs.lava_corridor import LavaCorridorEnv, build_generative_model_for_env

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


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

    LOGGER.info(f"Creating {num_agents} ToM agents for lava corridor environment")
    LOGGER.info(f"  Horizon: {config.horizon}, Gamma: {config.gamma}, Alpha: {config.alpha_empathy}")
    LOGGER.info(f"  Kappa: {config.kappa_prior}, Beta: {config.beta_joint_flex}")

    # Build generative model from environment
    A, B, C, D = build_generative_model_for_env(env)

    agents = []
    A_matrices = []
    B_matrices = []
    C_matrices = []
    D_matrices = []

    for i in range(num_agents):
        LOGGER.debug(f"  Creating agent {i}")

        # Each agent gets the same generative model (shared environment)
        # But can have different initial beliefs or preferences if needed
        A_agent = A.copy()
        B_agent = B.copy()
        C_agent = C.copy()
        D_agent = D.copy()

        # Create PyMDP agent
        if config.learn_B:
            # Enable learning of transition model
            agent = Agent(
                A=A_agent,
                B=B_agent,
                C=C_agent,
                D=D_agent,
                pB=dirichlet_like(B_agent),
                lr_pB=config.lr_pB,
                policy_len=config.horizon,
                gamma=config.gamma,
            )
        else:
            # No learning
            agent = Agent(
                A=A_agent,
                B=B_agent,
                C=C_agent,
                D=D_agent,
                policy_len=config.horizon,
                gamma=config.gamma,
            )

        agents.append(agent)
        A_matrices.append(A_agent)
        B_matrices.append(B_agent)
        C_matrices.append(C_agent)
        D_matrices.append(D_agent)

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
    print(f"  State space: {A[0][0].shape}")
    print(f"  Action space: {B[0][0].shape[2]}")
    print(f"  Horizon: {config.horizon}")

    # Test shared outcomes
    shared = get_shared_outcomes(env)
    print(f"  Shared safe outcomes: {len(shared)} positions")
