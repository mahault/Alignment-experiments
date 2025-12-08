"""
Comprehensive empathy experiments across environment variants.

This script systematically tests:
1. All environment layouts (narrow, wide, bottleneck, risk_reward)
2. Symmetric empathy (both agents same α)
3. Asymmetric empathy (different α for each agent)
4. Clear analysis comparing coordination outcomes
"""

import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import jax.random as jr
from typing import Dict, List, Tuple

from tom.models import LavaModel, LavaAgent
from tom.envs import LavaV2Env
from tom.planning.si_empathy_lava import EmpathicLavaPlanner
from tom.planning import safe_belief_update


def get_other_agent_belief(obs_dict, num_states):
    """
    Get belief about other agent from direct observation.

    In LavaV2Env with extended observations, we directly observe
    the other agent's position, so belief is a delta function.

    Parameters
    ----------
    obs_dict : dict
        Observation dictionary with "other_obs" key
    num_states : int
        Number of states

    Returns
    -------
    qs_other : np.ndarray
        Belief about other agent (delta function at observed position)
    """
    if "other_obs" not in obs_dict:
        # If no other observation, uniform prior
        return np.ones(num_states) / num_states

    other_obs = int(np.asarray(obs_dict["other_obs"])[0])
    qs_other = np.zeros(num_states)
    qs_other[other_obs] = 1.0
    return qs_other


def manual_belief_update(obs_dict, A, D, num_states):
    """
    DEPRECATED: Use safe_belief_update instead.
    This function is kept for backwards compatibility but will be removed.
    """
    obs = int(np.asarray(obs_dict["location_obs"])[0])
    likelihood = A[obs]
    unnorm = likelihood * D
    denom = unnorm.sum()
    if denom > 1e-10:
        qs = unnorm / denom
    else:
        print(f"[WARN] manual_belief_update: denom={denom}, using uniform")
        qs = np.ones(num_states) / num_states
    return qs


def run_episode(
    env: LavaV2Env,
    planner_i: EmpathicLavaPlanner,
    planner_j: EmpathicLavaPlanner,
    max_timesteps: int = 20,
    verbose: bool = False
) -> Dict:
    """
    Run one episode with two empathic agents.

    Returns
    -------
    result : dict
        Episode outcome with metrics
    """
    # Reset environment
    key = jr.PRNGKey(42)
    state, obs = env.reset(key)

    # Initialize beliefs
    model_i = planner_i.agent_i.model
    model_j = planner_j.agent_i.model

    A_i = np.asarray(model_i.A["location_obs"])
    D_i = np.asarray(model_i.D["location_state"])
    A_j = np.asarray(model_j.A["location_obs"])
    D_j = np.asarray(model_j.D["location_state"])

    num_states_i = model_i.num_states
    num_states_j = model_j.num_states

    # Initial beliefs (own position + observed other position)
    qs_i, valid_i = safe_belief_update(obs[0], A_i, D_i, agent_name="agent_i", verbose=verbose)
    qs_j_observed = get_other_agent_belief(obs[0], num_states_j)

    qs_j, valid_j = safe_belief_update(obs[1], A_j, D_j, agent_name="agent_j", verbose=verbose)
    qs_i_observed = get_other_agent_belief(obs[1], num_states_i)

    if verbose:
        print(f"\nInitial state:")
        print(env.render_state(state))

    # Track metrics
    collisions = []
    timesteps_taken = 0
    trajectory_i = []
    trajectory_j = []
    goal_reached_i = False
    goal_reached_j = False

    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

    for t in range(max_timesteps):
        if verbose:
            print(f"\n--- Timestep {t} ---")

        # Both agents plan (using their belief about themselves and observation of other)
        # But skip planning if agent has already reached their goal (absorbing state)
        if not goal_reached_i:
            G_i, G_j_sim, G_social_i, q_pi_i, action_i = planner_i.plan(qs_i, qs_j_observed)
        else:
            action_i = 4  # STAY - agent reached goal

        if not goal_reached_j:
            G_j, G_i_sim, G_social_j, q_pi_j, action_j = planner_j.plan(qs_j, qs_i_observed)
        else:
            action_j = 4  # STAY - agent reached goal

        # DEBUG: At first timestep, verify empathy mechanism is working
        if t == 0 and verbose:
            print(f"\n  [DEBUG] Timestep 0 - Empathy Verification:")
            print(f"    G_i (agent i's self-interest EFE): {G_i}")
            print(f"    G_j_sim (agent i's model of j's EFE): {G_j_sim}")
            print(f"    G_i - G_j_sim difference: {G_i - G_j_sim}")
            print(f"    α_i={planner_i.alpha}, so G_social_i = G_i + {planner_i.alpha}*G_j_sim")
            print(f"    G_social_i: {G_social_i}")
            print(f"    → If G_i == G_j_sim, empathy has NO effect (just scales by 1+α)")
            print()

        if verbose:
            print(f"  Agent i: α={planner_i.alpha}, action={action_names[action_i]}")
            print(f"    G_i={G_i[action_i]:.2f}, G_social={G_social_i[action_i]:.2f}")
            print(f"  Agent j: α={planner_j.alpha}, action={action_names[action_j]}")
            print(f"    G_j={G_j[action_j]:.2f}, G_social={G_social_j[action_j]:.2f}")

        # Take actions
        next_state, next_obs, reward, done, info = env.step(state, {0: action_i, 1: action_j})

        # Record trajectory
        pos_i = next_state["env_state"]["pos"][0]
        pos_j = next_state["env_state"]["pos"][1]
        trajectory_i.append(pos_i)
        trajectory_j.append(pos_j)

        # Check collision
        if info["collision"]:
            collisions.append(t)

        # Update goal reached flags
        if info["goal_reached"].get(0, False):
            goal_reached_i = True
        if info["goal_reached"].get(1, False):
            goal_reached_j = True

        # Update beliefs using safe Bayesian update
        # Skip belief updates for agents that have reached their goals
        B_i = np.asarray(model_i.B["location_state"])
        B_j = np.asarray(model_j.B["location_state"])

        # Predict next state based on action
        qs_i_pred = B_i[:, :, action_i] @ qs_i
        qs_j_pred = B_j[:, :, action_j] @ qs_j

        # Update with observation (using safe update to avoid NaN)
        next_obs_i = int(np.asarray(next_obs[0]["location_obs"])[0])
        next_obs_j = int(np.asarray(next_obs[1]["location_obs"])[0])

        likelihood_i = A_i[next_obs_i]
        likelihood_j = A_j[next_obs_j]

        unnorm_i = likelihood_i * qs_i_pred
        denom_i = unnorm_i.sum()
        if denom_i > 1e-10:
            qs_i = unnorm_i / denom_i
        else:
            if verbose:
                print(f"  [WARN] Agent i: Bayes update failed (denom={denom_i:.6f}), using prior")
            qs_i = qs_i_pred.copy()

        unnorm_j = likelihood_j * qs_j_pred
        denom_j = unnorm_j.sum()
        if denom_j > 1e-10:
            qs_j = unnorm_j / denom_j
        else:
            if verbose:
                print(f"  [WARN] Agent j: Bayes update failed (denom={denom_j:.6f}), using prior")
            qs_j = qs_j_pred.copy()

        # Update observed other positions
        qs_j_observed = get_other_agent_belief(next_obs[0], num_states_j)
        qs_i_observed = get_other_agent_belief(next_obs[1], num_states_i)

        if verbose:
            print(env.render_state(next_state))
            if info["collision"]:
                print("  ⚠️  COLLISION!")
            if any(info["lava_hit"].values()):
                print(f"  ⚠️  LAVA HIT: {info['lava_hit']}")
            if any(info["goal_reached"].values()):
                print(f"  ✓ GOAL REACHED: {info['goal_reached']}")

        state = next_state
        timesteps_taken = t + 1

        # Early termination: both agents reached goal, or collision/lava hit
        if goal_reached_i and goal_reached_j:
            if verbose:
                print(f"\n  ✓ BOTH AGENTS REACHED THEIR GOALS!")
            break
        if done:
            break

    # Compile results using tracked flags (not just final info dict)
    result = {
        "layout": env.layout.name,
        "alpha_i": planner_i.alpha,
        "alpha_j": planner_j.alpha,
        "collision": len(collisions) > 0,
        "num_collisions": len(collisions),
        "collision_timesteps": collisions,
        "lava_hit_i": info.get("lava_hit", {}).get(0, False),
        "lava_hit_j": info.get("lava_hit", {}).get(1, False),
        "goal_reached_i": goal_reached_i,
        "goal_reached_j": goal_reached_j,
        "timesteps": timesteps_taken,
        "trajectory_i": trajectory_i,
        "trajectory_j": trajectory_j,
    }

    # Joint success: both reach goal without collision or lava
    result["joint_success"] = (
        goal_reached_i
        and goal_reached_j
        and not result["collision"]
        and not result["lava_hit_i"]
        and not result["lava_hit_j"]
    )

    return result


def print_results_table(results: List[Dict]):
    """Print formatted results table."""
    print("\n" + "=" * 100)
    print("EXPERIMENT RESULTS")
    print("=" * 100)

    # Group by layout
    layouts = sorted(set(r["layout"] for r in results))

    for layout in layouts:
        layout_results = [r for r in results if r["layout"] == layout]

        print(f"\n{'Layout: ' + layout.upper():-^100}")
        print(f"\n{'α_i':<6} {'α_j':<6} {'Collision':<12} {'Lava i':<10} {'Lava j':<10} "
              f"{'Goal i':<10} {'Goal j':<10} {'Joint Success':<15} {'Steps':<6}")
        print("-" * 100)

        for r in layout_results:
            print(f"{r['alpha_i']:<6.1f} {r['alpha_j']:<6.1f} "
                  f"{str(r['collision']):<12} "
                  f"{str(r['lava_hit_i']):<10} {str(r['lava_hit_j']):<10} "
                  f"{str(r['goal_reached_i']):<10} {str(r['goal_reached_j']):<10} "
                  f"{str(r['joint_success']):<15} {r['timesteps']:<6}")

    print("\n" + "=" * 100)


def print_analysis(results: List[Dict]):
    """Print detailed analysis of results."""
    print("\n" + "=" * 100)
    print("ANALYSIS")
    print("=" * 100)

    layouts = sorted(set(r["layout"] for r in results))

    for layout in layouts:
        layout_results = [r for r in results if r["layout"] == layout]

        print(f"\n## {layout.upper()} ##")

        # Success rate by empathy level
        empathy_configs = sorted(set((r["alpha_i"], r["alpha_j"]) for r in layout_results))

        for alpha_i, alpha_j in empathy_configs:
            condition_results = [
                r for r in layout_results
                if r["alpha_i"] == alpha_i and r["alpha_j"] == alpha_j
            ]

            if not condition_results:
                continue

            success_rate = sum(r["joint_success"] for r in condition_results) / len(condition_results)
            collision_rate = sum(r["collision"] for r in condition_results) / len(condition_results)
            avg_timesteps = np.mean([r["timesteps"] for r in condition_results])

            print(f"\n  α_i={alpha_i:.1f}, α_j={alpha_j:.1f}:")
            print(f"    Joint success rate: {success_rate:.1%}")
            print(f"    Collision rate: {collision_rate:.1%}")
            print(f"    Avg timesteps: {avg_timesteps:.1f}")

            # Qualitative interpretation
            if success_rate > 0.8:
                print(f"    → EXCELLENT coordination")
            elif success_rate > 0.5:
                print(f"    → MODERATE coordination")
            elif success_rate > 0.2:
                print(f"    → POOR coordination")
            else:
                print(f"    → FAILED to coordinate")

    print("\n" + "=" * 100)


def main():
    print("=" * 100)
    print("COMPREHENSIVE EMPATHY EXPERIMENTS")
    print("=" * 100)

    # Experiment configuration
    layouts_to_test = ["wide", "bottleneck"]  # Start with these, add narrow/risk_reward later
    empathy_configs = [
        (0.0, 0.0),  # Both selfish
        (0.5, 0.5),  # Both balanced
        (1.0, 1.0),  # Both prosocial
        (1.0, 0.0),  # Asymmetric: i altruist, j selfish
        (0.0, 1.0),  # Asymmetric: i selfish, j altruist
    ]

    horizon = 5
    gamma = 8.0
    max_timesteps = 25

    all_results = []

    for layout_name in layouts_to_test:
        print(f"\n{'Testing layout: ' + layout_name.upper():-^100}")

        # Create environment
        if layout_name == "wide":
            env = LavaV2Env(layout_name="wide", width=6, num_agents=2, timesteps=max_timesteps)
        elif layout_name == "bottleneck":
            env = LavaV2Env(layout_name="bottleneck", width=8, num_agents=2, timesteps=max_timesteps)
        else:
            env = LavaV2Env(layout_name=layout_name, num_agents=2, timesteps=max_timesteps)

        layout_info = env.get_layout_info()
        print(f"  Width: {layout_info['width']}, Height: {layout_info['height']}")
        print(f"  Goals: {layout_info['goal_positions']}")
        print(f"  Start positions: {layout_info['start_positions']}")

        # Create models - each agent gets model with their own starting position AND goal
        start_pos_i = layout_info['start_positions'][0]
        start_pos_j = layout_info['start_positions'][1]
        goal_pos_i = layout_info['goal_positions'][0]
        goal_pos_j = layout_info['goal_positions'][1]

        model_i = LavaModel(
            width=env.width,
            height=env.height,
            goal_x=goal_pos_i[0],
            goal_y=goal_pos_i[1],
            safe_cells=layout_info['safe_cells'],
            start_pos=start_pos_i
        )
        model_j = LavaModel(
            width=env.width,
            height=env.height,
            goal_x=goal_pos_j[0],
            goal_y=goal_pos_j[1],
            safe_cells=layout_info['safe_cells'],
            start_pos=start_pos_j
        )

        for alpha_i, alpha_j in empathy_configs:
            print(f"\n  Testing α_i={alpha_i:.1f}, α_j={alpha_j:.1f}...", end=" ")

            # Create agents and planners
            agent_i = LavaAgent(model_i, horizon=horizon, gamma=gamma)
            agent_j = LavaAgent(model_j, horizon=horizon, gamma=gamma)

            planner_i = EmpathicLavaPlanner(agent_i, agent_j, alpha=alpha_i)
            planner_j = EmpathicLavaPlanner(agent_j, agent_i, alpha=alpha_j)

            # Enable verbose for first wide corridor test with balanced empathy to debug
            verbose_debug = (layout_name == "wide" and alpha_i == 0.5 and alpha_j == 0.5)

            # Run episode
            result = run_episode(env, planner_i, planner_j, max_timesteps=max_timesteps, verbose=verbose_debug)
            all_results.append(result)

            # Print immediate feedback
            if result["joint_success"]:
                print(f"✓ JOINT SUCCESS (t={result['timesteps']})")
            elif result["goal_reached_i"] and not result["goal_reached_j"]:
                print("✓ PARTIAL (agent i only)")
            elif result["goal_reached_j"] and not result["goal_reached_i"]:
                print("✓ PARTIAL (agent j only)")
            elif result["collision"]:
                print("✗ COLLISION")
            elif result["lava_hit_i"] or result["lava_hit_j"]:
                print("✗ LAVA HIT")
            else:
                print("✗ INCOMPLETE")

    # Print comprehensive results
    print_results_table(all_results)
    print_analysis(all_results)

    print("\n" + "=" * 100)
    print("EXPERIMENTS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
