"""
Test VFE (Variational Free Energy) tracking during episodes.

In active inference:
- VFE = -log p(o) = surprise at observation (computed during belief updates)
- EFE = expected utility + epistemic value (computed during planning)

Emotional state mapping (from Pitliya et al. 2025):
- Arousal ~ EFE (expected difficulty/cost of achieving goals)
- Valence ~ -VFE (how well reality matches expectations)

This test runs episodes and tracks both VFE and EFE to understand
the relationship between surprise, planning, and coordination.
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


def run_episode_with_vfe(
    layout_name: str,
    alpha_i: float,
    alpha_j: float,
    seed: int = 42,
    max_timesteps: int = 15,
    horizon: int = 3,
    gamma: float = 16.0,
    verbose: bool = True,
) -> Dict:
    """Run episode tracking both VFE and EFE."""
    layout = get_layout(layout_name, start_config="A")

    env = LavaV2Env(
        layout_name=layout_name,
        width=layout.width,
        num_agents=2,
        timesteps=max_timesteps,
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

    # Create agents and planners
    agent_i = LavaAgent(model_i, horizon=horizon, gamma=gamma)
    agent_j = LavaAgent(model_j, horizon=horizon, gamma=gamma)

    planner_i = EmpathicLavaPlanner(
        agent_i, agent_j,
        alpha=alpha_i,
        alpha_other=alpha_j,
    )
    planner_j = EmpathicLavaPlanner(
        agent_j, agent_i,
        alpha=alpha_j,
        alpha_other=alpha_i,
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

    # Initial belief update with VFE
    qs_i, _, vfe_i = safe_belief_update(obs[0], A_i, D_i, agent_name="agent_i", return_vfe=True)
    qs_j, _, vfe_j = safe_belief_update(obs[1], A_j, D_j, agent_name="agent_j", return_vfe=True)

    def get_other_belief(obs_dict, num_states):
        if "other_obs" not in obs_dict:
            return np.ones(num_states) / num_states
        other_obs = int(np.asarray(obs_dict["other_obs"])[0])
        qs_other = np.zeros(num_states)
        qs_other[other_obs] = 1.0
        return qs_other

    qs_j_observed = get_other_belief(obs[0], num_states_j)
    qs_i_observed = get_other_belief(obs[1], num_states_i)

    # Track metrics over time
    vfe_history_i = [vfe_i]
    vfe_history_j = [vfe_j]
    efe_history_i = []  # G_social for agent i
    efe_history_j = []  # G_social for agent j

    goal_reached_i = False
    goal_reached_j = False
    collision = False

    if verbose:
        print(f"\n{'='*60}")
        print(f"Layout: {layout_name}, alpha=({alpha_i}, {alpha_j})")
        print(f"{'='*60}")
        print(f"t=0: VFE_i={vfe_i:.3f}, VFE_j={vfe_j:.3f}")

    for t in range(max_timesteps):
        pos_i = state["env_state"]["pos"][0]
        pos_j = state["env_state"]["pos"][1]

        # Plan actions (this computes EFE)
        if not goal_reached_i:
            G_i, G_j_from_i, G_social_i, q_pi_i, action_i = planner_i.plan(qs_i, qs_j_observed)
            efe_i = float(G_social_i.min())  # Best policy's G_social
        else:
            action_i = 4  # STAY
            efe_i = 0.0

        if not goal_reached_j:
            G_j, G_i_from_j, G_social_j, q_pi_j, action_j = planner_j.plan(qs_j, qs_i_observed)
            efe_j = float(G_social_j.min())  # Best policy's G_social
        else:
            action_j = 4
            efe_j = 0.0

        efe_history_i.append(efe_i)
        efe_history_j.append(efe_j)

        # Execute
        next_state, next_obs, reward, done, info = env.step(state, {0: action_i, 1: action_j})

        # Track outcomes
        if info.get("collision", False) or info.get("edge_collision", False):
            collision = True
        if info.get("goal_reached", {}).get(0, False):
            goal_reached_i = True
        if info.get("goal_reached", {}).get(1, False):
            goal_reached_j = True

        # Update beliefs with VFE tracking
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

        # Compute VFE: surprise = -log p(o)
        vfe_i = -np.log(max(denom_i, 1e-16))
        qs_i = unnorm_i / denom_i if denom_i > 1e-10 else qs_i_pred

        unnorm_j = likelihood_j * qs_j_pred
        denom_j = unnorm_j.sum()
        vfe_j = -np.log(max(denom_j, 1e-16))
        qs_j = unnorm_j / denom_j if denom_j > 1e-10 else qs_j_pred

        vfe_history_i.append(vfe_i)
        vfe_history_j.append(vfe_j)

        if verbose:
            action_names = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
            print(f"t={t+1}: a_i={action_names[action_i]}, a_j={action_names[action_j]}")
            print(f"       VFE_i={vfe_i:.3f}, VFE_j={vfe_j:.3f}")
            print(f"       EFE_i={efe_i:.3f}, EFE_j={efe_j:.3f}")
            if collision:
                print(f"       COLLISION!")
            if goal_reached_i:
                print(f"       Agent i reached goal")
            if goal_reached_j:
                print(f"       Agent j reached goal")

        qs_j_observed = get_other_belief(next_obs[0], num_states_j)
        qs_i_observed = get_other_belief(next_obs[1], num_states_i)

        state = next_state

        if goal_reached_i and goal_reached_j:
            break
        if done:
            break

    success = goal_reached_i and goal_reached_j and not collision

    # Compute emotional state metrics
    avg_vfe_i = np.mean(vfe_history_i)
    avg_vfe_j = np.mean(vfe_history_j)
    avg_efe_i = np.mean(efe_history_i) if efe_history_i else 0
    avg_efe_j = np.mean(efe_history_j) if efe_history_j else 0

    # Emotional mapping: arousal ~ EFE (difficulty), valence ~ -VFE (fit)
    arousal_i = avg_efe_i
    arousal_j = avg_efe_j
    valence_i = -avg_vfe_i
    valence_j = -avg_vfe_j

    if verbose:
        print(f"\n--- Summary ---")
        print(f"Success: {success}")
        print(f"Collision: {collision}")
        print(f"\nEmotional State (from Active Inference):")
        print(f"  Agent i: Arousal={arousal_i:.3f} (EFE), Valence={valence_i:.3f} (-VFE)")
        print(f"  Agent j: Arousal={arousal_j:.3f} (EFE), Valence={valence_j:.3f} (-VFE)")
        print(f"\nInterpretation:")
        print(f"  - High Arousal (EFE) = High expected cost/difficulty")
        print(f"  - High Valence (-VFE) = Reality matches expectations (low surprise)")

    return {
        "success": success,
        "collision": collision,
        "goal_i": goal_reached_i,
        "goal_j": goal_reached_j,
        "vfe_history_i": vfe_history_i,
        "vfe_history_j": vfe_history_j,
        "efe_history_i": efe_history_i,
        "efe_history_j": efe_history_j,
        "avg_vfe_i": avg_vfe_i,
        "avg_vfe_j": avg_vfe_j,
        "avg_efe_i": avg_efe_i,
        "avg_efe_j": avg_efe_j,
        "arousal_i": arousal_i,
        "arousal_j": arousal_j,
        "valence_i": valence_i,
        "valence_j": valence_j,
    }


def main():
    print("="*70)
    print("VFE TRACKING TEST: Active Inference Emotional State Analysis")
    print("="*70)
    print()
    print("In active inference:")
    print("  VFE = -log p(o) = surprise at observation (computed in belief update)")
    print("  EFE = expected utility + epistemic value (computed during planning)")
    print()
    print("Emotional state mapping (Pitliya et al. 2025):")
    print("  Arousal ~ EFE (expected difficulty)")
    print("  Valence ~ -VFE (how well reality matches expectations)")
    print()

    # Test different scenarios
    scenarios = [
        ("wide", 0.5, 0.5, "Cooperative, easy layout"),
        ("bottleneck", 0.5, 0.5, "Cooperative, hard layout"),
        ("bottleneck", 0.0, 0.0, "Selfish, hard layout"),
        ("crossed_goals", 1.0, 1.0, "Fully prosocial, crossing paths"),
    ]

    results = []
    for layout, alpha_i, alpha_j, description in scenarios:
        print(f"\n{'#'*70}")
        print(f"# {description}")
        print(f"{'#'*70}")
        result = run_episode_with_vfe(layout, alpha_i, alpha_j, verbose=True)
        results.append((description, result))

    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARATIVE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Scenario':<40} {'Success':<8} {'VFE_i':<8} {'EFE_i':<8} {'Valence_i':<10}")
    print("-"*70)
    for desc, r in results:
        print(f"{desc:<40} {str(r['success']):<8} {r['avg_vfe_i']:<8.3f} {r['avg_efe_i']:<8.3f} {r['valence_i']:<10.3f}")


if __name__ == "__main__":
    main()
