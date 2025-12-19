"""
Test emotional state tracking using the Circumplex Model.

Based on Pattisapu et al. (2024) "Free Energy in a Circumplex Model of Emotion":
- Arousal = H[Q(s|o)] = entropy of posterior beliefs (uncertainty)
- Valence = Utility - Expected Utility (reward prediction error)

The Circumplex maps emotions to a 2D space:
- 0deg = Happy, 45deg = Excited, 90deg = Alert, 135deg = Angry
- 180deg = Sad, 225deg = Depressed, 270deg = Calm, 315deg = Relaxed
"""

import os
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import jax.random as jr
from typing import Dict

from tom.models import LavaModel, LavaAgent
from tom.envs import LavaV2Env
from tom.envs.env_lava_variants import get_layout
from tom.planning.si_empathy_lava import EmpathicLavaPlanner
from tom.planning import safe_belief_update
from tom.planning.emotional_state import (
    EmotionalState,
    EmotionalStateTracker,
    compute_belief_entropy,
    compute_valence,
)


def run_episode_with_emotions(
    layout_name: str,
    alpha_i: float,
    alpha_j: float,
    seed: int = 42,
    max_timesteps: int = 15,
    horizon: int = 3,
    gamma: float = 16.0,
    verbose: bool = True,
) -> Dict:
    """Run episode tracking emotional states via Circumplex Model."""
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

    # Get model components for emotional computation
    A_i = np.asarray(model_i.A["location_obs"])
    D_i = np.asarray(model_i.D["location_state"])
    C_i = np.asarray(model_i.C["location_obs"])
    A_j = np.asarray(model_j.A["location_obs"])
    D_j = np.asarray(model_j.D["location_state"])
    C_j = np.asarray(model_j.C["location_obs"])

    num_states_i = model_i.num_states
    num_states_j = model_j.num_states

    # Emotional state trackers
    tracker_i = EmotionalStateTracker(
        arousal_scale=np.log(num_states_i),  # Max entropy for this state space
        valence_scale=5.0,
    )
    tracker_j = EmotionalStateTracker(
        arousal_scale=np.log(num_states_j),
        valence_scale=5.0,
    )

    # Initial beliefs
    qs_i_prior = D_i.copy()
    qs_j_prior = D_j.copy()

    qs_i, _ = safe_belief_update(obs[0], A_i, D_i, agent_name="agent_i")
    qs_j, _ = safe_belief_update(obs[1], A_j, D_j, agent_name="agent_j")

    # Record initial emotional states
    obs_i = int(np.asarray(obs[0]["location_obs"])[0])
    obs_j = int(np.asarray(obs[1]["location_obs"])[0])

    state_i = tracker_i.record_from_beliefs(qs_i, obs_i, qs_i_prior, A_i, C_i, timestep=0)
    state_j = tracker_j.record_from_beliefs(qs_j, obs_j, qs_j_prior, A_j, C_j, timestep=0)

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

    if verbose:
        print(f"\n{'='*70}")
        print(f"Layout: {layout_name}, alpha=({alpha_i}, {alpha_j})")
        print(f"{'='*70}")
        print(f"t=0: Agent i: {state_i.emotion_label()} (A={state_i.arousal:.2f}, V={state_i.valence:.2f})")
        print(f"     Agent j: {state_j.emotion_label()} (A={state_j.arousal:.2f}, V={state_j.valence:.2f})")

    for t in range(max_timesteps):
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

        # Prior for next step = predicted belief after action
        if B_i.ndim == 3:
            qs_i_prior = B_i[:, :, action_i] @ qs_i
        else:
            qs_i_prior = np.zeros_like(qs_i)
            for s_other in range(len(qs_j_observed)):
                qs_i_prior += B_i[:, :, s_other, action_i] @ qs_i * qs_j_observed[s_other]

        if B_j.ndim == 3:
            qs_j_prior = B_j[:, :, action_j] @ qs_j
        else:
            qs_j_prior = np.zeros_like(qs_j)
            for s_other in range(len(qs_i_observed)):
                qs_j_prior += B_j[:, :, s_other, action_j] @ qs_j * qs_i_observed[s_other]

        # Get observations
        next_obs_i = int(np.asarray(next_obs[0]["location_obs"])[0])
        next_obs_j = int(np.asarray(next_obs[1]["location_obs"])[0])

        # Belief update
        likelihood_i = A_i[next_obs_i]
        likelihood_j = A_j[next_obs_j]

        unnorm_i = likelihood_i * qs_i_prior
        denom_i = unnorm_i.sum()
        qs_i = unnorm_i / denom_i if denom_i > 1e-10 else qs_i_prior

        unnorm_j = likelihood_j * qs_j_prior
        denom_j = unnorm_j.sum()
        qs_j = unnorm_j / denom_j if denom_j > 1e-10 else qs_j_prior

        # Record emotional states
        state_i = tracker_i.record_from_beliefs(
            qs_i, next_obs_i, qs_i_prior, A_i, C_i, timestep=t+1
        )
        state_j = tracker_j.record_from_beliefs(
            qs_j, next_obs_j, qs_j_prior, A_j, C_j, timestep=t+1
        )

        if verbose:
            action_names = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
            print(f"t={t+1}: a_i={action_names[action_i]}, a_j={action_names[action_j]}")
            print(f"     Agent i: {state_i.emotion_label()} (A={state_i.arousal:.2f}, V={state_i.valence:.2f})")
            print(f"     Agent j: {state_j.emotion_label()} (A={state_j.arousal:.2f}, V={state_j.valence:.2f})")
            if collision:
                print(f"     COLLISION!")
            if goal_reached_i and t == len(tracker_i.history) - 2:
                print(f"     Agent i reached goal!")
            if goal_reached_j and t == len(tracker_j.history) - 2:
                print(f"     Agent j reached goal!")

        qs_j_observed = get_other_belief(next_obs[0], num_states_j)
        qs_i_observed = get_other_belief(next_obs[1], num_states_i)

        state = next_state

        if goal_reached_i and goal_reached_j:
            break
        if done:
            break

    success = goal_reached_i and goal_reached_j and not collision

    if verbose:
        print(f"\n--- Emotional Summary ---")
        print(f"Success: {success}, Collision: {collision}")
        print(f"\nAgent i:")
        print(tracker_i.summary())
        print(f"\nAgent j:")
        print(tracker_j.summary())

    return {
        "success": success,
        "collision": collision,
        "goal_i": goal_reached_i,
        "goal_j": goal_reached_j,
        "tracker_i": tracker_i,
        "tracker_j": tracker_j,
    }


def main():
    print("="*70)
    print("CIRCUMPLEX MODEL EMOTIONAL STATE TRACKING")
    print("="*70)
    print()
    print("Based on Pattisapu et al. (2024):")
    print("  Arousal = H[Q(s|o)] = entropy of posterior (uncertainty)")
    print("  Valence = U - EU = utility - expected utility (reward prediction error)")
    print()
    print("Circumplex emotions:")
    print("  0deg=Happy, 45deg=Excited, 90deg=Alert, 135deg=Angry")
    print("  180deg=Sad, 225deg=Depressed, 270deg=Calm, 315deg=Relaxed")
    print()

    # Test scenarios
    scenarios = [
        ("wide", 0.5, 0.5, "Easy cooperative"),
        ("bottleneck", 0.5, 0.5, "Hard cooperative"),
        ("bottleneck", 0.0, 0.0, "Hard selfish"),
        ("crossed_goals", 0.5, 0.5, "Crossing paths"),
    ]

    for layout, alpha_i, alpha_j, description in scenarios:
        print(f"\n{'#'*70}")
        print(f"# {description}: {layout}, alpha=({alpha_i}, {alpha_j})")
        print(f"{'#'*70}")
        result = run_episode_with_emotions(layout, alpha_i, alpha_j, verbose=True)


if __name__ == "__main__":
    main()
