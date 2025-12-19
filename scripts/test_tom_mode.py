"""
Test script to compare deterministic vs probabilistic tom_mode.

Runs both modes across multiple layouts and empathy configurations
to see which produces better coordination outcomes.
"""

import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import jax.random as jr
from dataclasses import dataclass
from typing import Dict, List, Tuple

from tom.models import LavaModel, LavaAgent
from tom.envs import LavaV2Env
from tom.envs.env_lava_variants import get_layout
from tom.planning.si_empathy_lava import EmpathicLavaPlanner
from tom.planning import safe_belief_update


@dataclass
class TestConfig:
    layouts: List[str]
    empathy_configs: List[Tuple[float, float]]  # (alpha_i, alpha_j) pairs
    num_runs: int = 10  # Multiple runs for probabilistic mode
    max_timesteps: int = 15
    horizon: int = 3
    gamma: float = 16.0


def run_episode(
    layout_name: str,
    alpha_i: float,
    alpha_j: float,
    tom_mode: str,
    seed: int,
    config: TestConfig
) -> Dict:
    """Run single episode with given tom_mode."""
    layout = get_layout(layout_name, start_config="A")

    env = LavaV2Env(
        layout_name=layout_name,
        width=layout.width,
        num_agents=2,
        timesteps=config.max_timesteps,
        start_config="A"
    )

    layout_info = env.get_layout_info()

    # Create models
    start_pos_i = layout_info["start_positions"][0]
    start_pos_j = layout_info["start_positions"][1]
    goal_pos_i = layout_info["goal_positions"][0]
    goal_pos_j = layout_info["goal_positions"][1]

    model_i = LavaModel(
        width=env.width,
        height=env.height,
        goal_x=goal_pos_i[0],
        goal_y=goal_pos_i[1],
        safe_cells=layout_info["safe_cells"],
        start_pos=start_pos_i,
    )
    model_j = LavaModel(
        width=env.width,
        height=env.height,
        goal_x=goal_pos_j[0],
        goal_y=goal_pos_j[1],
        safe_cells=layout_info["safe_cells"],
        start_pos=start_pos_j,
    )

    # Create agents and planners with tom_mode
    agent_i = LavaAgent(model_i, horizon=config.horizon, gamma=config.gamma)
    agent_j = LavaAgent(model_j, horizon=config.horizon, gamma=config.gamma)

    planner_i = EmpathicLavaPlanner(
        agent_i, agent_j,
        alpha=alpha_i,
        alpha_other=alpha_j,
        tom_mode=tom_mode
    )
    planner_j = EmpathicLavaPlanner(
        agent_j, agent_i,
        alpha=alpha_j,
        alpha_other=alpha_i,
        tom_mode=tom_mode
    )

    # Run episode
    key = jr.PRNGKey(seed)
    state, obs = env.reset(key)

    # Initialize beliefs
    A_i = np.asarray(model_i.A["location_obs"])
    D_i = np.asarray(model_i.D["location_state"])
    A_j = np.asarray(model_j.A["location_obs"])
    D_j = np.asarray(model_j.D["location_state"])

    num_states_i = model_i.num_states
    num_states_j = model_j.num_states

    qs_i, _ = safe_belief_update(obs[0], A_i, D_i, agent_name="agent_i", verbose=False)
    qs_j, _ = safe_belief_update(obs[1], A_j, D_j, agent_name="agent_j", verbose=False)

    def get_other_belief(obs_dict, num_states):
        if "other_obs" not in obs_dict:
            return np.ones(num_states) / num_states
        other_obs = int(np.asarray(obs_dict["other_obs"])[0])
        qs_other = np.zeros(num_states)
        qs_other[other_obs] = 1.0
        return qs_other

    qs_j_observed = get_other_belief(obs[0], num_states_j)
    qs_i_observed = get_other_belief(obs[1], num_states_i)

    goal_reached_i = False
    goal_reached_j = False
    collision = False
    paralysis_count = 0
    last_pos = None

    for t in range(config.max_timesteps):
        pos_i = state["env_state"]["pos"][0]
        pos_j = state["env_state"]["pos"][1]
        current_pos = (tuple(pos_i), tuple(pos_j))

        # Paralysis detection
        if current_pos == last_pos:
            paralysis_count += 1
        else:
            paralysis_count = 0
        last_pos = current_pos

        if paralysis_count >= 3:
            break  # Stuck in paralysis

        # Plan actions
        if not goal_reached_i:
            _, _, _, _, action_i = planner_i.plan(qs_i, qs_j_observed)
        else:
            action_i = 4  # STAY

        if not goal_reached_j:
            _, _, _, _, action_j = planner_j.plan(qs_j, qs_i_observed)
        else:
            action_j = 4

        # Execute
        next_state, next_obs, reward, done, info = env.step(state, {0: action_i, 1: action_j})

        # Track outcomes
        if info.get("collision", False) or info.get("edge_collision", False):
            collision = True
        if info.get("goal_reached", {}).get(0, False):
            goal_reached_i = True
        if info.get("goal_reached", {}).get(1, False):
            goal_reached_j = True

        # Update beliefs
        B_i = np.asarray(model_i.B["location_state"])
        B_j = np.asarray(model_j.B["location_state"])

        if B_i.ndim == 3:
            qs_i_pred = B_i[:, :, action_i] @ qs_i
        else:
            qs_i_pred = np.zeros_like(qs_i)
            for s_other in range(len(qs_j_observed)):
                qs_i_pred += B_i[:, :, s_other, action_i] @ qs_i * qs_j_observed[s_other]

        if B_j.ndim == 3:
            qs_j_pred = B_j[:, :, action_j] @ qs_j
        else:
            qs_j_pred = np.zeros_like(qs_j)
            for s_other in range(len(qs_i_observed)):
                qs_j_pred += B_j[:, :, s_other, action_j] @ qs_j * qs_i_observed[s_other]

        next_obs_i = int(np.asarray(next_obs[0]["location_obs"])[0])
        next_obs_j = int(np.asarray(next_obs[1]["location_obs"])[0])

        likelihood_i = A_i[next_obs_i]
        likelihood_j = A_j[next_obs_j]

        unnorm_i = likelihood_i * qs_i_pred
        denom_i = unnorm_i.sum()
        qs_i = unnorm_i / denom_i if denom_i > 1e-10 else qs_i_pred

        unnorm_j = likelihood_j * qs_j_pred
        denom_j = unnorm_j.sum()
        qs_j = unnorm_j / denom_j if denom_j > 1e-10 else qs_j_pred

        qs_j_observed = get_other_belief(next_obs[0], num_states_j)
        qs_i_observed = get_other_belief(next_obs[1], num_states_i)

        state = next_state

        if goal_reached_i and goal_reached_j:
            break
        if done:
            break

    paralysis = paralysis_count >= 3
    success = goal_reached_i and goal_reached_j and not collision

    return {
        "success": success,
        "collision": collision,
        "paralysis": paralysis,
        "goal_i": goal_reached_i,
        "goal_j": goal_reached_j,
    }


def run_comparison(config: TestConfig):
    """Run comparison between deterministic and probabilistic tom_mode."""
    results = {
        "deterministic": {"success": 0, "collision": 0, "paralysis": 0, "total": 0},
        "probabilistic": {"success": 0, "collision": 0, "paralysis": 0, "total": 0},
    }

    print("=" * 70)
    print("TOM MODE COMPARISON: Deterministic vs Probabilistic")
    print("=" * 70)
    print(f"Layouts: {config.layouts}")
    print(f"Empathy configs: {config.empathy_configs}")
    print(f"Runs per config: {config.num_runs}")
    print()

    detailed_results = []

    for layout in config.layouts:
        print(f"\n--- {layout.upper()} ---")

        for alpha_i, alpha_j in config.empathy_configs:
            det_outcomes = {"success": 0, "collision": 0, "paralysis": 0}
            prob_outcomes = {"success": 0, "collision": 0, "paralysis": 0}

            # Run deterministic (only once since it's deterministic)
            result = run_episode(layout, alpha_i, alpha_j, "deterministic", 42, config)
            det_outcomes["success"] = 1 if result["success"] else 0
            det_outcomes["collision"] = 1 if result["collision"] else 0
            det_outcomes["paralysis"] = 1 if result["paralysis"] else 0

            results["deterministic"]["success"] += det_outcomes["success"]
            results["deterministic"]["collision"] += det_outcomes["collision"]
            results["deterministic"]["paralysis"] += det_outcomes["paralysis"]
            results["deterministic"]["total"] += 1

            # Run probabilistic multiple times
            for run in range(config.num_runs):
                np.random.seed(42 + run)  # Different seed for each run
                result = run_episode(layout, alpha_i, alpha_j, "probabilistic", 42, config)
                prob_outcomes["success"] += 1 if result["success"] else 0
                prob_outcomes["collision"] += 1 if result["collision"] else 0
                prob_outcomes["paralysis"] += 1 if result["paralysis"] else 0

            results["probabilistic"]["success"] += prob_outcomes["success"]
            results["probabilistic"]["collision"] += prob_outcomes["collision"]
            results["probabilistic"]["paralysis"] += prob_outcomes["paralysis"]
            results["probabilistic"]["total"] += config.num_runs

            # Print per-config results
            det_str = "SUCCESS" if det_outcomes["success"] else ("COLLISION" if det_outcomes["collision"] else "PARALYSIS")
            prob_success_rate = prob_outcomes["success"] / config.num_runs
            prob_collision_rate = prob_outcomes["collision"] / config.num_runs
            prob_paralysis_rate = prob_outcomes["paralysis"] / config.num_runs

            print(f"  alpha=({alpha_i}, {alpha_j}): Det={det_str}, Prob=S:{prob_success_rate:.0%}/C:{prob_collision_rate:.0%}/P:{prob_paralysis_rate:.0%}")

            detailed_results.append({
                "layout": layout,
                "alpha_i": alpha_i,
                "alpha_j": alpha_j,
                "det_success": det_outcomes["success"],
                "det_collision": det_outcomes["collision"],
                "det_paralysis": det_outcomes["paralysis"],
                "prob_success": prob_outcomes["success"],
                "prob_collision": prob_outcomes["collision"],
                "prob_paralysis": prob_outcomes["paralysis"],
            })

    # Print summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    for mode in ["deterministic", "probabilistic"]:
        total = results[mode]["total"]
        if total > 0:
            success_rate = results[mode]["success"] / total
            collision_rate = results[mode]["collision"] / total
            paralysis_rate = results[mode]["paralysis"] / total
            print(f"\n{mode.upper()}:")
            print(f"  Success rate:   {success_rate:.1%} ({results[mode]['success']}/{total})")
            print(f"  Collision rate: {collision_rate:.1%} ({results[mode]['collision']}/{total})")
            print(f"  Paralysis rate: {paralysis_rate:.1%} ({results[mode]['paralysis']}/{total})")

    # Verdict
    print("\n" + "=" * 70)
    det_success = results["deterministic"]["success"] / results["deterministic"]["total"]
    prob_success = results["probabilistic"]["success"] / results["probabilistic"]["total"]

    if prob_success > det_success + 0.05:
        print("VERDICT: Probabilistic ToM performs BETTER")
    elif det_success > prob_success + 0.05:
        print("VERDICT: Deterministic ToM performs BETTER")
    else:
        print("VERDICT: Performance is SIMILAR")

    return detailed_results


def main():
    config = TestConfig(
        layouts=["wide", "narrow", "bottleneck", "crossed_goals", "risk_reward"],
        empathy_configs=[
            (0.0, 0.0),   # Both selfish
            (0.5, 0.5),   # Both balanced
            (1.0, 1.0),   # Both prosocial
            (1.0, 0.0),   # Asymmetric: i empathic, j selfish
            (0.0, 1.0),   # Asymmetric: i selfish, j empathic
        ],
        num_runs=10,
        max_timesteps=15,
        horizon=3,
        gamma=16.0,
    )

    results = run_comparison(config)
    print("\nDone!")


if __name__ == "__main__":
    main()
