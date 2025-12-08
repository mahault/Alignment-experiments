"""
Single-agent lava corridor demo using Phase 1 TOM planner.

This script demonstrates:
1. Creating TOM-style LavaModel and LavaAgent
2. Using LavaV1Env for environment interaction
3. Using LavaPlanner for EFE-based action selection (no empathy, no F)
4. Manual Bayesian belief updates
5. Simple rollout loop
"""

import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import jax.random as jr

from tom.models import LavaModel, LavaAgent
from tom.envs import LavaV1Env
from tom.planning.si_lava import LavaPlanner


def manual_belief_update(obs_dict, A, D):
    """
    Perform manual Bayesian belief update.

    Parameters
    ----------
    obs_dict : dict
        Observation dictionary with "location_obs" key
    A : np.ndarray
        Observation model (num_obs, num_states)
    D : np.ndarray
        Prior over states (num_states,)

    Returns
    -------
    qs : np.ndarray
        Posterior belief state (num_states,)
    """
    # Extract scalar observation index
    obs = int(np.asarray(obs_dict["location_obs"])[0])

    # Bayesian update: p(s|o) ∝ p(o|s) * p(s)
    likelihood = A[obs]  # p(o|s)
    unnorm = likelihood * D  # p(o,s)
    qs = unnorm / unnorm.sum()  # p(s|o)

    return qs


def render_state(env, state, agent_obs, qs):
    """Print current state in human-readable format."""
    # Get agent position from state
    pos = state["env_state"]["pos"][0]  # Agent 0's position
    x, y = pos

    print(f"\n  Position: ({x}, {y})")
    print(f"  Observation: {agent_obs}")
    print(f"  Belief (max): state {qs.argmax()} (p={qs.max():.3f})")

    # Simple ASCII visualization
    width = env.width
    height = env.height
    safe_y = env.safe_y

    grid = []
    for row_y in range(height):
        row = []
        for col_x in range(width):
            if (col_x, row_y) == pos:
                row.append("A")  # Agent
            elif row_y != safe_y:
                row.append("~")  # Lava
            elif col_x == env.goal_x and row_y == safe_y:
                row.append("G")  # Goal
            else:
                row.append(".")  # Safe corridor
        grid.append(" ".join(row))

    print("\n  Grid:")
    for row in grid:
        print(f"    {row}")


def main():
    print("=" * 60)
    print("Phase 1: Single-Agent TOM Planner Demo")
    print("=" * 60)

    # Configuration
    width = 5
    height = 3
    goal_x = 4
    num_timesteps = 10
    horizon = 5  # Planning horizon - must see ahead to find goal
    gamma = 8.0

    print(f"\nConfiguration:")
    print(f"  Grid: {width}x{height}")
    print(f"  Goal: ({goal_x}, 1)")  # Middle row
    print(f"  Max timesteps: {num_timesteps}")
    print(f"  Planning horizon: {horizon}")
    print(f"  Gamma (inverse temperature): {gamma}")

    # Create model and agent
    print("\n" + "-" * 60)
    print("Step 1: Creating TOM components")
    print("-" * 60)

    model = LavaModel(width=width, height=height, goal_x=goal_x)
    agent = LavaAgent(model, horizon=horizon, gamma=gamma)
    planner = LavaPlanner(agent)

    print(f"  Model states: {model.num_states}")
    print(f"  Agent policies: {len(agent.policies)}")
    print(f"  Actions: UP(0), DOWN(1), LEFT(2), RIGHT(3), STAY(4)")

    # Create environment
    env = LavaV1Env(width=width, height=height, num_agents=1, timesteps=num_timesteps)

    # Reset environment
    print("\n" + "-" * 60)
    print("Step 2: Resetting environment")
    print("-" * 60)

    key = jr.PRNGKey(42)
    state, obs = env.reset(key)

    # Initial belief
    obs_dict = obs[0]  # Agent 0's observation
    agent_obs = int(np.asarray(obs_dict["location_obs"])[0])

    A = np.asarray(model.A["location_obs"])
    D = np.asarray(model.D["location_state"])

    qs = manual_belief_update(obs_dict, A, D)

    print(f"  Initial state reset")
    render_state(env, state, agent_obs, qs)

    # Rollout loop
    print("\n" + "-" * 60)
    print("Step 3: Running rollout with LavaPlanner")
    print("-" * 60)

    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

    for t in range(num_timesteps):
        print(f"\n--- Timestep {t} ---")

        # Plan action using EFE-based planner
        G, q_pi, action = planner.plan(qs)

        print(f"  Planning:")
        print(f"    EFE values: {G}")
        print(f"    Policy posterior: {q_pi}")
        print(f"    Best policy: {np.argmax(q_pi)} (action: {action_names[action]})")

        # Take action in environment
        next_state, next_obs, reward, done, info = env.step(state, {0: action})

        # Update belief
        next_obs_dict = next_obs[0]
        next_agent_obs = int(np.asarray(next_obs_dict["location_obs"])[0])

        # Use B matrix for temporal prediction
        B = np.asarray(model.B["location_state"])
        qs_pred = B[:, :, action] @ qs  # Predicted belief after action

        # Bayesian update with new observation
        likelihood = A[next_agent_obs]
        unnorm = likelihood * qs_pred
        qs_next = unnorm / unnorm.sum()

        print(f"  Action taken: {action_names[action]}")
        render_state(env, next_state, next_agent_obs, qs_next)

        # Update for next iteration
        state = next_state
        obs = next_obs
        qs = qs_next

        # Check termination
        if done:
            print(f"\n  Episode terminated at timestep {t}")
            if "goal_reached" in info and info["goal_reached"]:
                print("  Reason: GOAL REACHED! ✓")
            elif "lava_hit" in info and info["lava_hit"]:
                print("  Reason: LAVA HIT! ✗")
            else:
                print("  Reason: Max timesteps reached")
            break

    # Final summary
    print("\n" + "=" * 60)
    print("Final Summary")
    print("=" * 60)

    final_pos = state["env_state"]["pos"][0]
    final_x, final_y = final_pos

    print(f"  Final position: ({final_x}, {final_y})")
    print(f"  Goal position: ({goal_x}, 1)")

    if final_x == goal_x and final_y == 1:
        print("\n  SUCCESS: Agent reached the goal! ✓")
    elif final_y != 1:
        print("\n  FAILURE: Agent hit lava ✗")
    else:
        print("\n  INCOMPLETE: Agent did not reach goal in time")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
