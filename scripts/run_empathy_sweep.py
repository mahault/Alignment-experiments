"""
Comprehensive empathy experiment sweep.

This script systematically tests empathy effects across:
1. All environment layouts (including new ones)
2. Full symmetric and asymmetric empathy parameter sweeps
3. Both start configurations (A and B) for role asymmetry testing
4. Multiple seeds for statistical power

Emotional State Tracking (Circumplex Model - Pattisapu et al. 2024):
- Arousal = H[Q(s|o)] = entropy of posterior beliefs (uncertainty)
- Valence = Utility - Expected Utility (reward prediction error)
- Maps to 8 emotions: happy, excited, alert, angry, sad, depressed, calm, relaxed
- Social emotional state = weighted average of own + other's state (by empathy alpha)

Outputs:
- Detailed CSV with all metrics including emotional states
- Summary statistics
- Data ready for plotting in analysis/plot_empathy_sweeps.py

See ROADMAP.md for full experimental design.
"""

import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import jax.random as jr
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
import argparse
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        if desc:
            print(f"{desc}...")
        return iterable

from tom.models import LavaModel, LavaAgent
from tom.envs import LavaV2Env
from tom.envs.env_lava_variants import get_layout, get_all_layout_names, LAYOUT_COMPLEXITY
from tom.planning.si_empathy_lava import EmpathicLavaPlanner
from tom.planning.jax_hierarchical_planner import HierarchicalEmpathicPlannerJax, has_jax_zoned_layout
from tom.planning import safe_belief_update
from tom.planning.emotional_state import (
    EmotionalStateTracker,
    infer_other_emotional_state,
    compute_empathic_emotional_state,
)
from src.metrics.paralysis_detection import detect_paralysis


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the empathy sweep experiments."""
    # Layouts to test
    layouts: List[str] = None

    # Empathy parameters
    alphas_symmetric: List[float] = None  # For symmetric sweeps
    alphas_asymmetric: List[float] = None  # For asymmetric grid

    # Start configurations
    start_configs: List[str] = None

    # Seeds (environment is deterministic, so 1 seed is sufficient)
    num_seeds: int = 1
    base_seed: int = 42

    # Planning parameters
    horizon: int = 3
    gamma: float = 16.0
    max_timesteps: int = 15

    # Paralysis detection
    paralysis_cycle_threshold: int = 3  # Same state seen K times
    paralysis_stay_threshold: int = 3   # Both stay for M steps

    # Output
    output_dir: str = "results"
    verbose: bool = False

    # Planner type
    use_hierarchical: bool = False  # Use hierarchical planner for supported layouts

    # Empathy formulation mode
    empathy_mode: str = "additive"  # "additive" or "weighted" (Sanjeev's)

    def __post_init__(self):
        if self.layouts is None:
            self.layouts = get_all_layout_names()
        if self.alphas_symmetric is None:
            # Coarse: selfish, balanced, prosocial (3 values)
            self.alphas_symmetric = [0.0, 0.5, 1.0]
        if self.alphas_asymmetric is None:
            # Same coarse grid: 3x3 = 9 combinations
            self.alphas_asymmetric = [0.0, 0.5, 1.0]
        if self.start_configs is None:
            # A: Original config
            # B: Swapped agents (positions AND goals swapped)
            # C: Swapped goals only (same positions, each agent wants other's goal)
            self.start_configs = ["A", "B", "C"]


# =============================================================================
# Episode Runner
# =============================================================================

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


def describe_policy(policy_idx, num_policies):
    """Describe what a policy index means."""
    if policy_idx < len(ACTION_NAMES):
        return ACTION_NAMES[policy_idx]
    return f"Policy{policy_idx}"


def get_other_agent_belief(obs_dict, num_states):
    """Get belief about other agent from direct observation."""
    if "other_obs" not in obs_dict:
        return np.ones(num_states) / num_states
    other_obs = int(np.asarray(obs_dict["other_obs"])[0])
    qs_other = np.zeros(num_states)
    qs_other[other_obs] = 1.0
    return qs_other


def run_episode(
    env: LavaV2Env,
    planner_i: EmpathicLavaPlanner,
    planner_j: EmpathicLavaPlanner,
    seed: int,
    config: ExperimentConfig,
    verbose: bool = True,
) -> Dict:
    """
    Run one episode with two empathic agents.

    Returns detailed metrics for analysis.
    """
    # Reset environment with seed
    key = jr.PRNGKey(seed)
    state, obs = env.reset(key)

    if verbose:
        print(f"\n{'='*80}")
        print(f"Episode: layout={env.layout.name}, seed={seed}")
        print(f"  alpha_i={planner_i.alpha}, alpha_j={planner_j.alpha}")
        planner_type = "hierarchical" if isinstance(planner_i, HierarchicalEmpathicPlannerJax) else "flat"
        print(f"  planner={planner_type}")
        print(f"{'='*80}")
        print(f"\nInitial state:")
        print(env.render_state(state))

    # Initialize beliefs
    model_i = planner_i.agent_i.model
    model_j = planner_j.agent_i.model

    A_i = np.asarray(model_i.A["location_obs"])
    D_i = np.asarray(model_i.D["location_state"])
    C_i = np.asarray(model_i.C["location_obs"])
    A_j = np.asarray(model_j.A["location_obs"])
    D_j = np.asarray(model_j.D["location_state"])
    C_j = np.asarray(model_j.C["location_obs"])

    num_states_i = model_i.num_states
    num_states_j = model_j.num_states

    # Emotional state trackers (Circumplex Model: arousal=entropy, valence=reward prediction error)
    tracker_i = EmotionalStateTracker(
        arousal_scale=np.log(num_states_i),  # Max entropy for this state space
        valence_scale=5.0,
    )
    tracker_j = EmotionalStateTracker(
        arousal_scale=np.log(num_states_j),
        valence_scale=5.0,
    )

    # Initial beliefs
    qs_i, _ = safe_belief_update(obs[0], A_i, D_i, agent_name="agent_i", verbose=False)
    qs_j_observed = get_other_agent_belief(obs[0], num_states_j)
    qs_j, _ = safe_belief_update(obs[1], A_j, D_j, agent_name="agent_j", verbose=False)
    qs_i_observed = get_other_agent_belief(obs[1], num_states_i)

    # Record initial emotional states
    obs_i_init = int(np.asarray(obs[0]["location_obs"])[0])
    obs_j_init = int(np.asarray(obs[1]["location_obs"])[0])
    qs_i_prior = D_i.copy()
    qs_j_prior = D_j.copy()

    tracker_i.record_from_beliefs(qs_i, obs_i_init, qs_i_prior, A_i, C_i, timestep=0)
    tracker_j.record_from_beliefs(qs_j, obs_j_init, qs_j_prior, A_j, C_j, timestep=0)

    # Track metrics
    trajectory_i = []
    trajectory_j = []
    actions_i = []
    actions_j = []
    goal_reached_i = False
    goal_reached_j = False
    step_goal_i = None
    step_goal_j = None
    collisions = []
    G_i_total = 0.0
    G_j_total = 0.0

    lava_hit_i = False
    lava_hit_j = False
    cell_collision = False
    edge_collision = False

    for t in range(config.max_timesteps):
        # Record current positions
        pos_i = state["env_state"]["pos"][0]
        pos_j = state["env_state"]["pos"][1]
        trajectory_i.append(pos_i)
        trajectory_j.append(pos_j)

        if verbose:
            print(f"\n--- Timestep {t} ---")

        # Plan actions
        if not goal_reached_i:
            G_i, G_j_sim, G_social_i, q_pi_i, action_i = planner_i.plan(qs_i, qs_j_observed)
            G_i_total += G_social_i[action_i]
        else:
            action_i = 4  # STAY
            G_i = np.zeros(5)
            G_j_sim = np.zeros(5)
            G_social_i = np.zeros(5)

        if not goal_reached_j:
            G_j, G_i_sim, G_social_j, q_pi_j, action_j = planner_j.plan(qs_j, qs_i_observed)
            G_j_total += G_social_j[action_j]
        else:
            action_j = 4
            G_j = np.zeros(5)
            G_i_sim = np.zeros(5)
            G_social_j = np.zeros(5)

        # DEBUG: At first timestep, show empathy mechanism
        if t == 0 and verbose:
            print(f"\n  [DEBUG] Timestep 0 - Empathy & Policy Verification:")
            print(f"    Number of policies: {len(G_i)}")

            print(f"\n    Agent i (alpha={planner_i.alpha}):")
            print(f"      G_i (self-interest):  {G_i[:5]}")
            print(f"      G_j_sim (ToM of j):   {G_j_sim[:5]}")
            print(f"      G_social_i (mixed):   {G_social_i[:5]}")

            print(f"\n    Agent j (alpha={planner_j.alpha}):")
            print(f"      G_j (self-interest):  {G_j[:5]}")
            print(f"      G_i_sim (ToM of i):   {G_i_sim[:5]}")
            print(f"      G_social_j (mixed):   {G_social_j[:5]}")

            # Check if empathy changes policy choice
            selfish_choice_i = int(np.argmin(G_i))
            empathic_choice_i = int(np.argmin(G_social_i))
            selfish_choice_j = int(np.argmin(G_j))
            empathic_choice_j = int(np.argmin(G_social_j))

            print(f"\n    Policy choices:")
            print(f"      Selfish i: {selfish_choice_i} ({describe_policy(selfish_choice_i, len(G_i))})")
            print(f"      Empathic i: {empathic_choice_i} ({describe_policy(empathic_choice_i, len(G_i))})")
            print(f"      Selfish j: {selfish_choice_j} ({describe_policy(selfish_choice_j, len(G_j))})")
            print(f"      Empathic j: {empathic_choice_j} ({describe_policy(empathic_choice_j, len(G_j))})")

            if selfish_choice_i != empathic_choice_i:
                print(f"\n      EMPATHY CHANGES i's POLICY: "
                      f"{describe_policy(selfish_choice_i, len(G_i))} -> "
                      f"{describe_policy(empathic_choice_i, len(G_i))}")
            if selfish_choice_j != empathic_choice_j:
                print(f"      EMPATHY CHANGES j's POLICY: "
                      f"{describe_policy(selfish_choice_j, len(G_j))} -> "
                      f"{describe_policy(empathic_choice_j, len(G_j))}")

        if verbose:
            print(f"  Agent i: action={ACTION_NAMES[action_i]}, G_social={G_social_i[action_i]:.2f}")
            print(f"  Agent j: action={ACTION_NAMES[action_j]}, G_social={G_social_j[action_j]:.2f}")

        actions_i.append(action_i)
        actions_j.append(action_j)

        # Execute actions
        next_state, next_obs, reward, done, info = env.step(state, {0: action_i, 1: action_j})

        # Track collision types
        if info.get("collision", False):
            collisions.append(t)
            cell_collision = True
        if info.get("edge_collision", False):
            edge_collision = True

        # Track lava hits
        if info.get("lava_hit", {}).get(0, False):
            lava_hit_i = True
        if info.get("lava_hit", {}).get(1, False):
            lava_hit_j = True

        # Track goal reaching
        if info.get("goal_reached", {}).get(0, False) and not goal_reached_i:
            goal_reached_i = True
            step_goal_i = t + 1
        if info.get("goal_reached", {}).get(1, False) and not goal_reached_j:
            goal_reached_j = True
            step_goal_j = t + 1

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

        # Bayesian update
        next_obs_i = int(np.asarray(next_obs[0]["location_obs"])[0])
        next_obs_j = int(np.asarray(next_obs[1]["location_obs"])[0])

        likelihood_i = A_i[next_obs_i]
        likelihood_j = A_j[next_obs_j]

        # Store prior for emotional state computation
        qs_i_prior = qs_i_pred.copy()
        qs_j_prior = qs_j_pred.copy()

        unnorm_i = likelihood_i * qs_i_pred
        denom_i = unnorm_i.sum()
        if denom_i > 1e-10:
            qs_i = unnorm_i / denom_i
        else:
            qs_i = qs_i_pred.copy()

        unnorm_j = likelihood_j * qs_j_pred
        denom_j = unnorm_j.sum()
        if denom_j > 1e-10:
            qs_j = unnorm_j / denom_j
        else:
            qs_j = qs_j_pred.copy()

        # Record emotional states (Circumplex Model)
        tracker_i.record_from_beliefs(qs_i, next_obs_i, qs_i_prior, A_i, C_i, timestep=t+1)
        tracker_j.record_from_beliefs(qs_j, next_obs_j, qs_j_prior, A_j, C_j, timestep=t+1)

        # Update observed positions
        qs_j_observed = get_other_agent_belief(next_obs[0], num_states_j)
        qs_i_observed = get_other_agent_belief(next_obs[1], num_states_i)

        if verbose:
            print(env.render_state(next_state))
            if info.get("collision", False):
                print("  COLLISION!")
            if any(info.get("lava_hit", {}).values()):
                print(f"  LAVA HIT: {info['lava_hit']}")
            if any(info.get("goal_reached", {}).values()):
                print(f"  GOAL REACHED: {info['goal_reached']}")

        state = next_state

        # Check termination
        if goal_reached_i and goal_reached_j:
            if verbose:
                print(f"\n  SUCCESS: Both agents reached their goals!")
            break
        if done:
            if verbose:
                if goal_reached_i or goal_reached_j:
                    print(f"\n  PARTIAL: Only one agent reached goal")
                else:
                    print(f"\n  TIMEOUT/FAILURE: Episode ended without success")
            break

    # Paralysis detection
    paralysis_result = detect_paralysis(
        trajectory_i, trajectory_j,
        actions_i, actions_j,
        goal_reached_i, goal_reached_j,
        config.max_timesteps,
        config.paralysis_cycle_threshold,
        config.paralysis_stay_threshold
    )

    # Compute metrics
    timesteps = len(trajectory_i)

    # Arrival order and gap
    if step_goal_i is not None and step_goal_j is not None:
        arrival_gap = step_goal_j - step_goal_i
        if arrival_gap > 0:
            arrival_order = "i_first"
        elif arrival_gap < 0:
            arrival_order = "j_first"
        else:
            arrival_order = "tie"
    elif step_goal_i is not None:
        arrival_order = "i_only"
        arrival_gap = config.max_timesteps - step_goal_i
    elif step_goal_j is not None:
        arrival_order = "j_only"
        arrival_gap = step_goal_j - config.max_timesteps
    else:
        arrival_order = "neither"
        arrival_gap = 0

    # Success metrics
    any_collision = cell_collision or edge_collision
    any_lava = lava_hit_i or lava_hit_j
    both_success = goal_reached_i and goal_reached_j and not any_collision and not any_lava
    single_success = (goal_reached_i or goal_reached_j) and not both_success
    failure = not goal_reached_i and not goal_reached_j

    # Emotional state metrics (Circumplex Model)
    avg_state_i = tracker_i.get_average()
    avg_state_j = tracker_j.get_average()
    emotions_i = tracker_i.get_emotions()
    emotions_j = tracker_j.get_emotions()

    # Compute empathic emotional states (weighted by empathy parameter)
    if avg_state_i and avg_state_j:
        social_arousal_i, social_valence_i = compute_empathic_emotional_state(
            avg_state_i.arousal, avg_state_i.valence,
            avg_state_j.arousal, avg_state_j.valence,
            planner_i.alpha
        )
        social_arousal_j, social_valence_j = compute_empathic_emotional_state(
            avg_state_j.arousal, avg_state_j.valence,
            avg_state_i.arousal, avg_state_i.valence,
            planner_j.alpha
        )
    else:
        social_arousal_i = social_valence_i = 0.0
        social_arousal_j = social_valence_j = 0.0

    # Dominant emotion (most frequent)
    def get_dominant_emotion(emotions):
        if not emotions:
            return "none"
        from collections import Counter
        return Counter(emotions).most_common(1)[0][0]

    return {
        # Success metrics
        "both_success": both_success,
        "single_success": single_success,
        "failure": failure,
        "goal_reached_i": goal_reached_i,
        "goal_reached_j": goal_reached_j,

        # Collision metrics
        "lava_collision": any_lava,
        "lava_hit_i": lava_hit_i,
        "lava_hit_j": lava_hit_j,
        "cell_collision": cell_collision,
        "edge_collision": edge_collision,
        "num_collisions": len(collisions),

        # Paralysis metrics
        "paralysis": paralysis_result["paralysis"],
        "paralysis_type": paralysis_result["paralysis_type"],
        "cycle_length": paralysis_result["cycle_length"],
        "stay_streak": paralysis_result["stay_streak"],

        # Efficiency metrics
        "timesteps": timesteps,
        "steps_i": step_goal_i if step_goal_i else timesteps,
        "steps_j": step_goal_j if step_goal_j else timesteps,
        "arrival_order": arrival_order,
        "arrival_gap": arrival_gap,

        # Internal metrics
        "G_i": G_i_total,
        "G_j": G_j_total,

        # Emotional state metrics (Circumplex Model: Pattisapu et al. 2024)
        "arousal_i": avg_state_i.arousal if avg_state_i else 0.0,
        "valence_i": avg_state_i.valence if avg_state_i else 0.0,
        "emotion_i": avg_state_i.emotion_label() if avg_state_i else "none",
        "arousal_j": avg_state_j.arousal if avg_state_j else 0.0,
        "valence_j": avg_state_j.valence if avg_state_j else 0.0,
        "emotion_j": avg_state_j.emotion_label() if avg_state_j else "none",
        "dominant_emotion_i": get_dominant_emotion(emotions_i),
        "dominant_emotion_j": get_dominant_emotion(emotions_j),
        # Empathic emotional state (weighted by alpha)
        "social_arousal_i": social_arousal_i,
        "social_valence_i": social_valence_i,
        "social_arousal_j": social_arousal_j,
        "social_valence_j": social_valence_j,

        # Trajectories (as strings for CSV)
        "trajectory_i": str(trajectory_i),
        "trajectory_j": str(trajectory_j),
    }


# =============================================================================
# Experiment Runner
# =============================================================================

def setup_experiment(
    layout_name: str,
    start_config: str,
    alpha_i: float,
    alpha_j: float,
    config: ExperimentConfig
) -> Tuple[LavaV2Env, EmpathicLavaPlanner, EmpathicLavaPlanner]:
    """Set up environment and planners for an experiment."""
    # Get layout with start config
    layout = get_layout(layout_name, start_config=start_config)

    # Create environment
    env = LavaV2Env(
        layout_name=layout_name,
        width=layout.width,
        num_agents=2,
        timesteps=config.max_timesteps,
        start_config=start_config
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
    agent_i = LavaAgent(model_i, horizon=config.horizon, gamma=config.gamma)
    agent_j = LavaAgent(model_j, horizon=config.horizon, gamma=config.gamma)

    # Each planner knows its own alpha AND observes the other's alpha (for ToM)
    # planner_i: alpha=alpha_i (self), alpha_other=alpha_j (observed empathy of j)
    # planner_j: alpha=alpha_j (self), alpha_other=alpha_i (observed empathy of i)

    # Use hierarchical planner if requested and layout supports it
    if config.use_hierarchical and has_jax_zoned_layout(layout_name):
        planner_i = HierarchicalEmpathicPlannerJax(
            agent_i, agent_j,
            layout_name=layout_name,
            alpha=alpha_i,
            alpha_other=alpha_j,
            gamma=config.gamma,
            empathy_mode=config.empathy_mode,
        )
        planner_j = HierarchicalEmpathicPlannerJax(
            agent_j, agent_i,
            layout_name=layout_name,
            alpha=alpha_j,
            alpha_other=alpha_i,
            gamma=config.gamma,
            empathy_mode=config.empathy_mode,
        )
    else:
        planner_i = EmpathicLavaPlanner(agent_i, agent_j, alpha=alpha_i, alpha_other=alpha_j,
                                        empathy_mode=config.empathy_mode)
        planner_j = EmpathicLavaPlanner(agent_j, agent_i, alpha=alpha_j, alpha_other=alpha_i,
                                        empathy_mode=config.empathy_mode)

    return env, planner_i, planner_j


def run_sweep(config: ExperimentConfig, mode: str = "asymmetric") -> pd.DataFrame:
    """
    Run full experiment sweep.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    mode : str
        "symmetric" for alpha_i = alpha_j sweep
        "asymmetric" for full (alpha_i, alpha_j) grid
        "both" for both sweeps

    Returns
    -------
    df : pd.DataFrame
        Results dataframe
    """
    results = []

    # Build experiment list
    experiments = []

    for layout in config.layouts:
        for start_cfg in config.start_configs:
            if mode in ["symmetric", "both"]:
                for alpha in config.alphas_symmetric:
                    for seed in range(config.num_seeds):
                        experiments.append({
                            "layout": layout,
                            "start_config": start_cfg,
                            "alpha_i": alpha,
                            "alpha_j": alpha,
                            "seed": config.base_seed + seed,
                            "sweep_type": "symmetric"
                        })

            if mode in ["asymmetric", "both"]:
                for alpha_i in config.alphas_asymmetric:
                    for alpha_j in config.alphas_asymmetric:
                        # Skip symmetric cases if running both
                        if mode == "both" and alpha_i == alpha_j:
                            continue
                        for seed in range(config.num_seeds):
                            experiments.append({
                                "layout": layout,
                                "start_config": start_cfg,
                                "alpha_i": alpha_i,
                                "alpha_j": alpha_j,
                                "seed": config.base_seed + seed,
                                "sweep_type": "asymmetric"
                            })

    print(f"Running {len(experiments)} experiments...")
    print(f"Layouts: {config.layouts}")
    print(f"Start configs: {config.start_configs}")
    print(f"Seeds per config: {config.num_seeds}")
    print(f"Mode: {mode}")

    # Run experiments with progress bar
    for exp in tqdm(experiments, desc="Running experiments"):
        try:
            env, planner_i, planner_j = setup_experiment(
                exp["layout"],
                exp["start_config"],
                exp["alpha_i"],
                exp["alpha_j"],
                config
            )

            result = run_episode(env, planner_i, planner_j, exp["seed"], config,
                                 verbose=config.verbose)

            # Add experiment metadata
            result["layout"] = exp["layout"]
            result["start_config"] = exp["start_config"]
            result["alpha_i"] = exp["alpha_i"]
            result["alpha_j"] = exp["alpha_j"]
            result["seed"] = exp["seed"]
            result["sweep_type"] = exp["sweep_type"]
            result["complexity"] = LAYOUT_COMPLEXITY.get(exp["layout"], 0)

            results.append(result)

        except Exception as e:
            print(f"\nError in {exp}: {e}")
            continue

    return pd.DataFrame(results)


def save_results(df: pd.DataFrame, config: ExperimentConfig, prefix: str = "empathy_sweep"):
    """Save results to CSV with metadata."""
    os.makedirs(config.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(config.output_dir, f"{prefix}_{timestamp}.csv")

    # Reorder columns
    column_order = [
        "layout", "start_config", "alpha_i", "alpha_j", "seed", "sweep_type", "complexity",
        "both_success", "single_success", "failure",
        "goal_reached_i", "goal_reached_j",
        "lava_collision", "lava_hit_i", "lava_hit_j",
        "cell_collision", "edge_collision", "num_collisions",
        "paralysis", "paralysis_type", "cycle_length", "stay_streak",
        "timesteps", "steps_i", "steps_j", "arrival_order", "arrival_gap",
        "G_i", "G_j",
        # Emotional state metrics (Circumplex Model)
        "arousal_i", "valence_i", "emotion_i", "dominant_emotion_i",
        "arousal_j", "valence_j", "emotion_j", "dominant_emotion_j",
        "social_arousal_i", "social_valence_i",
        "social_arousal_j", "social_valence_j",
        "trajectory_i", "trajectory_j"
    ]
    column_order = [c for c in column_order if c in df.columns]
    df = df[column_order]

    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    return csv_path


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Overall stats
    print(f"\nTotal experiments: {len(df)}")
    print(f"Overall success rate: {df['both_success'].mean():.1%}")
    print(f"Overall collision rate: {(df['cell_collision'] | df['edge_collision']).mean():.1%}")
    print(f"Overall paralysis rate: {df['paralysis'].mean():.1%}")

    # By layout
    print("\n--- By Layout ---")
    for layout in df['layout'].unique():
        layout_df = df[df['layout'] == layout]
        print(f"\n{layout.upper()}:")
        print(f"  Success: {layout_df['both_success'].mean():.1%}")
        print(f"  Collision: {(layout_df['cell_collision'] | layout_df['edge_collision']).mean():.1%}")
        print(f"  Paralysis: {layout_df['paralysis'].mean():.1%}")

    # By symmetric empathy level
    print("\n--- Symmetric Empathy ---")
    sym_df = df[df['sweep_type'] == 'symmetric']
    if len(sym_df) > 0:
        for alpha in sorted(sym_df['alpha_i'].unique()):
            alpha_df = sym_df[sym_df['alpha_i'] == alpha]
            print(f"  alpha={alpha:.1f}: Success={alpha_df['both_success'].mean():.1%}, "
                  f"Paralysis={alpha_df['paralysis'].mean():.1%}")

    # Emotional state summary (if columns exist)
    if 'arousal_i' in df.columns and 'valence_i' in df.columns:
        print("\n--- Emotional States (Circumplex Model) ---")
        print(f"  Agent i: Arousal={df['arousal_i'].mean():.2f}, Valence={df['valence_i'].mean():.2f}")
        print(f"  Agent j: Arousal={df['arousal_j'].mean():.2f}, Valence={df['valence_j'].mean():.2f}")

        # Emotions by outcome
        if 'emotion_i' in df.columns:
            success_df = df[df['both_success'] == True]
            fail_df = df[df['both_success'] == False]
            if len(success_df) > 0:
                print(f"\n  Success cases:")
                print(f"    Arousal: i={success_df['arousal_i'].mean():.2f}, j={success_df['arousal_j'].mean():.2f}")
                print(f"    Valence: i={success_df['valence_i'].mean():.2f}, j={success_df['valence_j'].mean():.2f}")
            if len(fail_df) > 0:
                print(f"  Failure cases:")
                print(f"    Arousal: i={fail_df['arousal_i'].mean():.2f}, j={fail_df['arousal_j'].mean():.2f}")
                print(f"    Valence: i={fail_df['valence_i'].mean():.2f}, j={fail_df['valence_j'].mean():.2f}")

    print("\n" + "=" * 80)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run empathy sweep experiments")
    parser.add_argument("--layouts", nargs="+", default=None,
                       help="Layouts to test (default: all)")
    parser.add_argument("--mode", choices=["symmetric", "asymmetric", "both"],
                       default="both", help="Sweep mode")
    parser.add_argument("--seeds", type=int, default=1,
                       help="Number of seeds per configuration (env is deterministic, 1 is sufficient)")
    parser.add_argument("--horizon", type=int, default=4,
                       help="Planning horizon")
    parser.add_argument("--max-steps", type=int, default=15,
                       help="Maximum timesteps per episode")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with fewer configs")
    parser.add_argument("--hierarchical", action="store_true",
                       help="Use hierarchical planner (for vertical_bottleneck, symmetric_bottleneck, narrow)")
    parser.add_argument("--empathy-mode", choices=["additive", "weighted"],
                       default="additive",
                       help="Empathy formula: 'additive' (G_self + α*G_other) or 'weighted' ((1-α)*G_self + α*G_other)")

    args = parser.parse_args()

    # Create config
    config = ExperimentConfig(
        layouts=args.layouts,
        num_seeds=args.seeds,
        horizon=args.horizon,
        max_timesteps=args.max_steps,
        verbose=args.verbose,
        use_hierarchical=args.hierarchical,
        empathy_mode=args.empathy_mode,
    )

    # Quick mode for testing
    if args.quick:
        config.layouts = ["wide", "crossed_goals"]
        config.alphas_symmetric = [0.0, 0.5, 1.0]
        config.alphas_asymmetric = [0.0, 0.5, 1.0]
        config.num_seeds = 5
        config.start_configs = ["A"]

    print("=" * 80)
    print("EMPATHY SWEEP EXPERIMENTS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Layouts: {config.layouts}")
    print(f"  Mode: {args.mode}")
    print(f"  Seeds: {config.num_seeds}")
    print(f"  Horizon: {config.horizon}")
    print(f"  Max steps: {config.max_timesteps}")
    print(f"  Hierarchical: {config.use_hierarchical}")
    print(f"  Empathy mode: {config.empathy_mode}")

    # Run sweep
    df = run_sweep(config, mode=args.mode)

    # Save results
    csv_path = save_results(df, config)

    # Print summary
    print_summary(df)

    print(f"\nResults saved to: {csv_path}")
    print("Run analysis/plot_empathy_sweeps.py to generate plots.")


if __name__ == "__main__":
    main()
