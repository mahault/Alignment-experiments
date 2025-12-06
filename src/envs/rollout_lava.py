# src/envs/rollout_lava.py

"""
Multi-agent rollout for LavaCorridorEnv experiments.

This module provides rollout functions for Experiments 1 and 2:
- Exp 1: Standard ToM with empathy (α > 0)
- Exp 2: F-aware policy prior (κ > 0)

Features:
- Multi-agent coordination
- Collision detection
- Success tracking
- Comprehensive logging for debugging
- Compatible with LavaCorridorEnv API
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def rollout_multi_agent_lava(
    env,
    agents: List[Any],
    num_timesteps: int,
    rng_key: Any = None,
    use_F_prior: bool = False,
    tom_config: Optional[Any] = None,
) -> Tuple[Dict, Dict, Any]:
    """
    Rollout multiple ToM agents in LavaCorridorEnv.

    This function executes a multi-agent episode in the lava corridor,
    tracking all relevant information for path flexibility analysis.

    Parameters
    ----------
    env : LavaCorridorEnv
        The lava corridor environment
    agents : List[Any]
        List of ToM agents (typically 2)
    num_timesteps : int
        Maximum number of timesteps
    rng_key : Any, optional
        Random key for stochastic environments
    use_F_prior : bool, optional
        If True, use F-aware policy prior (Experiment 2)
        If False, use standard ToM (Experiment 1)
    tom_config : ToMPolicyConfig, optional
        Configuration for F-aware prior (required if use_F_prior=True)

    Returns
    -------
    last_carry : Dict
        Final state of the rollout with keys:
        - state: final environment state
        - obs: final observations
        - qs: final beliefs
        - t: final timestep
    info : Dict
        Information collected during rollout with keys:
        - states: list of environment states
        - observations: list of observations
        - actions: list of actions taken
        - beliefs: list of belief states
        - trees: list of ToM trees (if available)
        - collision: whether collision occurred
        - success_i: whether agent 0 reached goal
        - success_j: whether agent 1 reached goal
        - timesteps: number of steps taken
    env : LavaCorridorEnv
        Environment after rollout
    """
    LOGGER.info("=" * 80)
    LOGGER.info(f"Starting multi-agent rollout (T={num_timesteps}, F-prior={use_F_prior})")
    LOGGER.info("=" * 80)

    num_agents = len(agents)
    if num_agents != env.num_agents:
        raise ValueError(
            f"Number of agents ({num_agents}) does not match environment ({env.num_agents})"
        )

    # Import ToM functions
    if use_F_prior:
        if tom_config is None:
            raise ValueError("tom_config required when use_F_prior=True")
        from src.tom.si_tom_F_prior import run_tom_step_with_F_prior as tom_step_fn
        LOGGER.info(f"Using F-aware prior: κ={tom_config.kappa_prior}, β={tom_config.beta_joint_flex}")
    else:
        # Use standard ToM step
        try:
            from src.tom.si_tom import run_tom_step as tom_step_fn
            LOGGER.info("Using standard ToM (no F-prior)")
        except ImportError:
            LOGGER.warning("Could not import run_tom_step, using placeholder")
            tom_step_fn = None

    # Reset environment
    state, obs = env.reset(rng_key)
    LOGGER.info(f"Environment reset: t=0")
    for aid in range(num_agents):
        LOGGER.info(f"  Agent {aid}: initial position={obs[aid]}")

    # Initialize tracking
    states_history = [state]
    obs_history = [obs]
    actions_history = []
    beliefs_history = []
    trees_history = []

    # Initial beliefs (None = will use D from agents)
    qs_prev = None

    # Main rollout loop
    for t in range(num_timesteps):
        LOGGER.debug(f"\n--- Timestep t={t} ---")

        # Check if episode is done
        if state.get("done", False):
            LOGGER.info(f"Episode terminated at t={t}")
            LOGGER.info(f"  Reason: lava_hit={state['lava_hit']}, success={state['success']}")
            break

        # Convert observations to format expected by ToM
        # LavaCorridorEnv returns obs as {agent_id: (x, y)}
        # ToM agents expect observation indices
        o_array = np.array([
            env.pos_to_obs_index(obs[aid]) for aid in range(num_agents)
        ])

        LOGGER.debug(f"Observations: {o_array}")
        for aid in range(num_agents):
            LOGGER.debug(f"  Agent {aid}: pos={obs[aid]}, obs_idx={o_array[aid]}")

        # Run ToM step for each agent to get actions
        actions = {}

        if tom_step_fn is None:
            # Fallback: simple heuristic
            LOGGER.warning("Using heuristic actions (ToM not available)")
            from src.envs.lava_corridor import RIGHT, STAY
            for aid in range(num_agents):
                x, y = obs[aid]
                if x < env.goal_x:
                    actions[aid] = RIGHT
                else:
                    actions[aid] = STAY
        else:
            # Run ToM policy search
            LOGGER.debug("Running ToM policy search...")

            # Get B matrices for learning (use first agent's B for now)
            B_self = agents[0].B if hasattr(agents[0], 'B') else None

            if use_F_prior:
                # Experiment 2: F-aware prior
                tom_results, EFE_arr, Emp_arr = tom_step_fn(
                    agents=agents,
                    o=o_array,
                    qs_prev=qs_prev,
                    t=t,
                    learn=False,  # Can enable learning later
                    agent_num=0,  # Focal agent
                    B_self=B_self,
                    config=tom_config,
                )
            else:
                # Experiment 1: Standard ToM
                tom_results, EFE_arr, Emp_arr = tom_step_fn(
                    agents=agents,
                    o=o_array,
                    qs_prev=qs_prev,
                    t=t,
                    learn=False,
                    agent_num=0,
                    B_self=B_self,
                )

            # Extract actions from tom_results
            for aid in range(num_agents):
                # tom_results[aid]["action"] is the sampled action
                action = tom_results[aid]["action"]

                # Convert to integer if it's an array
                if isinstance(action, np.ndarray):
                    action = int(action[0]) if action.size == 1 else int(action)
                else:
                    action = int(action)

                actions[aid] = action

                LOGGER.debug(
                    f"  Agent {aid}: q_pi max={tom_results[aid]['q_pi'].max():.3f}, "
                    f"G mean={tom_results[aid]['G'].mean():.3f}, action={action}"
                )

            # Store beliefs for next timestep
            qs_prev = tom_results

            # Store beliefs in history
            beliefs_history.append({
                aid: tom_results[aid]["qs"] for aid in range(num_agents)
            })

            # Store tom_results for potential tree extraction
            trees_history.append(tom_results)

        LOGGER.debug(f"Actions: {actions}")
        for aid, action in actions.items():
            from src.envs.lava_corridor import ACTION_NAMES
            action_name = ACTION_NAMES.get(action, "UNKNOWN")
            LOGGER.debug(f"  Agent {aid}: action={action_name}")

        # Step environment
        state_next, obs_next, done, step_info = env.step(state, actions, rng_key)

        # Log step results
        LOGGER.debug(f"Step complete: t={t+1}, done={done}")
        if step_info.get("lava_hit", False):
            LOGGER.warning(f"  LAVA HIT at t={t+1}!")
        if step_info.get("collision", False):
            LOGGER.warning(f"  COLLISION at t={t+1}!")
        if step_info.get("success", False):
            LOGGER.info(f"  SUCCESS at t={t+1}!")

        # Store history
        actions_history.append(actions)
        states_history.append(state_next)
        obs_history.append(obs_next)

        # Update for next iteration
        state = state_next
        obs = obs_next

    # Compile final info
    final_t = state["t"]
    info = {
        "states": states_history,
        "observations": obs_history,
        "actions": actions_history,
        "beliefs": beliefs_history,
        "trees": trees_history,
        "collision": bool(state.get("collision", False)),
        "success": bool(state.get("success", False)),
        "success_i": bool(state.get("success_i", False) if num_agents > 0 else False),
        "success_j": bool(state.get("success_j", False) if num_agents > 1 else False),
        "lava_hit": bool(state.get("lava_hit", False)),
        "timesteps": final_t,
        "t": final_t,
    }

    last_carry = {
        "state": state,
        "obs": obs,
        "qs": qs_prev,
        "t": final_t,
    }

    LOGGER.info("=" * 80)
    LOGGER.info("Rollout complete")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Final timestep: t={final_t}/{num_timesteps}")
    LOGGER.info(f"Success: {info['success']}")
    LOGGER.info(f"  Agent 0 reached goal: {info['success_i']}")
    LOGGER.info(f"  Agent 1 reached goal: {info['success_j']}")
    LOGGER.info(f"Collision: {info['collision']}")
    LOGGER.info(f"Lava hit: {info['lava_hit']}")
    LOGGER.info("=" * 80)

    return last_carry, info, env


def rollout_exp1(
    env,
    agents: List[Any],
    num_timesteps: int,
    rng_key: Any = None,
    alpha_empathy: float = 1.0,
) -> Tuple[Dict, Dict, Any]:
    """
    Rollout for Experiment 1: Standard ToM with empathy.

    This is a convenience wrapper around rollout_multi_agent_lava
    with use_F_prior=False.

    Parameters
    ----------
    env : LavaCorridorEnv
        The lava corridor environment
    agents : List[Any]
        List of ToM agents
    num_timesteps : int
        Maximum number of timesteps
    rng_key : Any, optional
        Random key
    alpha_empathy : float, optional
        Empathy weight (for logging)

    Returns
    -------
    last_carry : Dict
        Final state
    info : Dict
        Rollout information
    env : LavaCorridorEnv
        Environment after rollout
    """
    LOGGER.info(f"Experiment 1: Standard ToM with α={alpha_empathy}")

    return rollout_multi_agent_lava(
        env=env,
        agents=agents,
        num_timesteps=num_timesteps,
        rng_key=rng_key,
        use_F_prior=False,
        tom_config=None,
    )


def rollout_exp2(
    env,
    agents: List[Any],
    num_timesteps: int,
    tom_config: Any,
    rng_key: Any = None,
) -> Tuple[Dict, Dict, Any]:
    """
    Rollout for Experiment 2: F-aware policy prior.

    This is a convenience wrapper around rollout_multi_agent_lava
    with use_F_prior=True.

    Parameters
    ----------
    env : LavaCorridorEnv
        The lava corridor environment
    agents : List[Any]
        List of ToM agents
    num_timesteps : int
        Maximum number of timesteps
    tom_config : ToMPolicyConfig
        Configuration with κ, β, shared outcomes
    rng_key : Any, optional
        Random key

    Returns
    -------
    last_carry : Dict
        Final state
    info : Dict
        Rollout information
    env : LavaCorridorEnv
        Environment after rollout
    """
    LOGGER.info(
        f"Experiment 2: F-aware prior with κ={tom_config.kappa_prior}, "
        f"β={tom_config.beta_joint_flex}"
    )

    return rollout_multi_agent_lava(
        env=env,
        agents=agents,
        num_timesteps=num_timesteps,
        rng_key=rng_key,
        use_F_prior=True,
        tom_config=tom_config,
    )
