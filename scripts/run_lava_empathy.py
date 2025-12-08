"""
Two-agent empathic coordination demo using Phase 2 planner.

This script demonstrates:
1. Two agents navigating the lava corridor
2. Each agent considers the other's EFE (empathy)
3. Coordination to avoid collisions
4. Comparison between selfish (α=0) and empathic (α>0) behavior
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
from tom.planning.si_empathy_lava import EmpathicLavaPlanner


def manual_belief_update(obs_dict, A, D):
    """Perform manual Bayesian belief update."""
    obs = int(np.asarray(obs_dict["location_obs"])[0])
    likelihood = A[obs]
    unnorm = likelihood * D
    qs = unnorm / unnorm.sum()
    return qs


def render_state(env, state, qs_i, qs_j):
    """Print current state for both agents."""
    pos_i = state["env_state"]["pos"][0]
    pos_j = state["env_state"]["pos"][1]

    print(f"\n  Agent i position: {pos_i}, belief at state {qs_i.argmax()} (p={qs_i.max():.3f})")
    print(f"  Agent j position: {pos_j}, belief at state {qs_j.argmax()} (p={qs_j.max():.3f})")

    # ASCII visualization
    width = env.width
    height = env.height
    safe_y = env.safe_y

    grid = []
    for row_y in range(height):
        row = []
        for col_x in range(width):
            if (col_x, row_y) == pos_i and (col_x, row_y) == pos_j:
                row.append("X")  # Collision!
            elif (col_x, row_y) == pos_i:
                row.append("i")  # Agent i
            elif (col_x, row_y) == pos_j:
                row.append("j")  # Agent j
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


def run_episode(alpha, width=6, height=3, num_timesteps=15, seed=42):
    """Run one episode with given empathy parameter."""
    print("\n" + "=" * 60)
    print(f"Running episode with α={alpha}")
    print("=" * 60)

    # Configuration
    goal_x = width - 1
    horizon = 5
    gamma = 8.0

    # Create models and agents
    model_i = LavaModel(width=width, height=height, goal_x=goal_x)
    model_j = LavaModel(width=width, height=height, goal_x=goal_x)

    agent_i = LavaAgent(model_i, horizon=horizon, gamma=gamma)
    agent_j = LavaAgent(model_j, horizon=horizon, gamma=gamma)

    # Create empathic planners
    planner_i = EmpathicLavaPlanner(agent_i, agent_j, alpha=alpha)
    planner_j = EmpathicLavaPlanner(agent_j, agent_i, alpha=alpha)

    # Create environment
    env = LavaV1Env(width=width, height=height, num_agents=2, timesteps=num_timesteps)

    # Reset environment
    key = jr.PRNGKey(seed)
    state, obs = env.reset(key)

    # Initial beliefs
    A_i = np.asarray(model_i.A["location_obs"])
    D_i = np.asarray(model_i.D["location_state"])
    A_j = np.asarray(model_j.A["location_obs"])
    D_j = np.asarray(model_j.D["location_state"])

    qs_i = manual_belief_update(obs[0], A_i, D_i)
    qs_j = manual_belief_update(obs[1], A_j, D_j)

    print(f"\nConfiguration:")
    print(f"  Grid: {width}x{height}")
    print(f"  Goal: ({goal_x}, 1)")
    print(f"  Planning horizon: {horizon}")
    print(f"  Empathy α: {alpha}")

    render_state(env, state, qs_i, qs_j)

    # Rollout
    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    collision_occurred = False
    goal_i_reached = False
    goal_j_reached = False

    for t in range(num_timesteps):
        print(f"\n--- Timestep {t} ---")

        # Both agents plan simultaneously
        G_i, G_j_from_i, G_social_i, q_pi_i, action_i = planner_i.plan(qs_i, qs_j)
        G_j, G_i_from_j, G_social_j, q_pi_j, action_j = planner_j.plan(qs_j, qs_i)

        print(f"  Agent i planning:")
        print(f"    G_i: {G_i}")
        print(f"    G_j (simulated): {G_j_from_i}")
        print(f"    G_social: {G_social_i}")
        print(f"    Action: {action_names[action_i]}")

        print(f"  Agent j planning:")
        print(f"    G_j: {G_j}")
        print(f"    G_i (simulated): {G_i_from_j}")
        print(f"    G_social: {G_social_j}")
        print(f"    Action: {action_names[action_j]}")

        # Take actions in environment
        next_state, next_obs, reward, done, info = env.step(state, {0: action_i, 1: action_j})

        # Update beliefs
        B_i = np.asarray(model_i.B["location_state"])
        B_j = np.asarray(model_j.B["location_state"])

        qs_i_pred = B_i[:, :, action_i] @ qs_i
        qs_j_pred = B_j[:, :, action_j] @ qs_j

        next_obs_i = int(np.asarray(next_obs[0]["location_obs"])[0])
        next_obs_j = int(np.asarray(next_obs[1]["location_obs"])[0])

        likelihood_i = A_i[next_obs_i]
        likelihood_j = A_j[next_obs_j]

        qs_i = (likelihood_i * qs_i_pred)
        qs_i = qs_i / qs_i.sum()

        qs_j = (likelihood_j * qs_j_pred)
        qs_j = qs_j / qs_j.sum()

        render_state(env, next_state, qs_i, qs_j)

        # Check for collision
        pos_i = next_state["env_state"]["pos"][0]
        pos_j = next_state["env_state"]["pos"][1]

        if pos_i == pos_j:
            print("\n  ⚠️  COLLISION DETECTED!")
            collision_occurred = True

        # Check for goal
        if pos_i == (goal_x, 1):
            goal_i_reached = True
        if pos_j == (goal_x, 1):
            goal_j_reached = True

        # Update state
        state = next_state

        # Check termination
        if done:
            print(f"\n  Episode terminated at timestep {t}")
            break

    # Episode summary
    print("\n" + "-" * 60)
    print("Episode Summary")
    print("-" * 60)
    print(f"  Empathy α: {alpha}")
    print(f"  Collision: {collision_occurred}")
    print(f"  Agent i reached goal: {goal_i_reached}")
    print(f"  Agent j reached goal: {goal_j_reached}")
    print(f"  Joint success: {goal_i_reached and goal_j_reached and not collision_occurred}")

    return {
        "alpha": alpha,
        "collision": collision_occurred,
        "goal_i": goal_i_reached,
        "goal_j": goal_j_reached,
        "joint_success": goal_i_reached and goal_j_reached and not collision_occurred,
    }


def main():
    print("=" * 60)
    print("Phase 2: Empathic Multi-Agent Coordination Demo")
    print("=" * 60)

    # Test different empathy levels
    alphas = [0.0, 0.5, 1.0]
    results = []

    for alpha in alphas:
        result = run_episode(alpha, width=6, height=3, num_timesteps=15)
        results.append(result)

    # Final comparison
    print("\n" + "=" * 60)
    print("Final Comparison Across Empathy Levels")
    print("=" * 60)
    print(f"\n{'α':<8} {'Collision':<12} {'Goal i':<10} {'Goal j':<10} {'Joint Success':<15}")
    print("-" * 60)

    for r in results:
        print(f"{r['alpha']:<8.1f} {str(r['collision']):<12} {str(r['goal_i']):<10} "
              f"{str(r['goal_j']):<10} {str(r['joint_success']):<15}")

    print("\n" + "=" * 60)
    print("Expected behavior:")
    print("  α=0.0 (selfish): May collide, agents don't coordinate")
    print("  α=0.5 (balanced): Some coordination, reduced collisions")
    print("  α=1.0 (prosocial): Full coordination, both reach goal safely")
    print("=" * 60)


if __name__ == "__main__":
    main()
