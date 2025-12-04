"""
Experiment 2: Path-Flexibility-Aware Policy Prior

HYPOTHESIS:
Explicit F-prior should:
1. Push agents toward mutually safe, flexible policies
2. Reduce catastrophic failures (lava, collisions)
3. Trade off pragmatic efficiency for robustness

SETUP:
- Same environment as Exp 1 (lava corridor)
- Same ToM + empathy core
- NEW: Policy prior p(π) ∝ exp(κ [F_i(π) + β F_j(π)])
- Decision: J_i(π) = G_i(π) + α G_j(π) - (κ/γ)[F_i(π) + β F_j(π)]
           q(π) = softmax(-γ J_i(π))

MANIPULATIONS:
Sweep over κ ∈ [0, 0.5, 1.0, 2.0]:
- κ = 0: Reduces to Exp 1 (no F-prior)
- κ > 0: Increasing preference for flexible policies

ANALYSIS:
- Behavioral changes as κ increases
- F vs EFE trade-off
- Resilience to perturbations
"""

import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import pickle
from typing import List

import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np

# TODO: Import when available
# from src.envs.lava_corridor import LavaCorridorEnv
# from src.tom.si_tom import ToMAgent, si_policy_search_tom
# from src.tom.rollout_tom import rollout
# from src.metrics.path_flexibility import compute_path_flexibility_for_tree

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


@dataclass
class Exp2Config:
    """Configuration for Experiment 2."""

    # Episode parameters
    num_episodes: int = 100
    num_timesteps: int = 20

    # Planning parameters
    horizon: int = 5
    gamma: float = 16.0

    # Empathy + F-prior parameters
    alpha_empathy: float = 1.0      # Weight on other's EFE
    kappa_prior: float = 0.5        # Flexibility prior strength (SWEEP THIS)
    beta_joint_flex: float = 1.0    # Weight on other's flexibility

    # Path flexibility computation
    lambda_E: float = 1.0
    lambda_R: float = 1.0
    lambda_O: float = 1.0

    # Environment
    env_height: int = 7
    env_width: int = 11
    slip_prob: float = 0.05

    # Outcomes
    shared_outcome_set: list = None

    # I/O
    seed: int = 42
    output_dir: str = "results/exp2"


def init_env_and_agents(config: Exp2Config):
    """
    Initialize environment and ToM agents for Experiment 2.

    NOTE: Agents now have kappa > 0, enabling F-prior.

    Returns
    -------
    env : LavaCorridorEnv
    focal_agent : ToMAgent
    other_agents : List[ToMAgent]
    """
    LOGGER.info(f"Initializing Experiment 2 with κ={config.kappa_prior}")

    # TODO: Replace with actual imports
    # env = LavaCorridorEnv(
    #     height=config.env_height,
    #     width=config.env_width,
    #     slip_prob=config.slip_prob,
    # )

    # config.shared_outcome_set = env.shared_outcomes()

    # # Build ToM agents WITH F-prior
    # focal_agent = ToMAgent(
    #     num_agents=2,
    #     agent_idx=0,
    #     alpha=config.alpha_empathy,
    #     kappa=config.kappa_prior,        # F-prior strength
    #     beta=config.beta_joint_flex,     # Weight on other's F
    #     gamma=config.gamma,
    #     horizon=config.horizon,
    # )

    # other_agents = [
    #     ToMAgent(
    #         num_agents=2,
    #         agent_idx=1,
    #         alpha=config.alpha_empathy,
    #         kappa=config.kappa_prior,
    #         beta=config.beta_joint_flex,
    #         gamma=config.gamma,
    #         horizon=config.horizon,
    #     )
    # ]

    # STUB
    env = None
    focal_agent = None
    other_agents = None
    config.shared_outcome_set = [0, 1, 2]

    LOGGER.info("Environment and agents initialized with F-prior")
    return env, focal_agent, other_agents


def run_single_episode_exp2(
    key,
    env,
    focal_agent,
    other_agents,
    config: Exp2Config,
):
    """
    Run a single episode with F-aware decision rule.

    Returns same structure as Exp 1, but now J_i is also logged.
    """
    LOGGER.debug(f"Running episode with key={key}, κ={config.kappa_prior}")

    # TODO: Replace with actual rollout
    # The key difference: rollout will use si_policy_search_tom with κ, β
    # Inside si_tom.py, policy selection will be:
    #   J_i = G_i + α*G_j - (κ/γ)*(F_i + β*F_j)
    #   q_pi = softmax(-γ * J_i)

    # last, info, env_after = rollout(
    #     env=env,
    #     focal_agent=focal_agent,
    #     other_agents=other_agents,
    #     num_timesteps=config.num_timesteps,
    #     rng_key=key,
    # )

    # focal_tree_final = info["tree"][-1]
    # other_trees_final = info["other_tree"][-1]

    # focal_idx = 0
    # other_idx = 1

    # metrics_per_policy = compute_path_flexibility_for_tree(
    #     focal_tree=focal_tree_final,
    #     other_tree=other_trees_final,
    #     focal_agent_idx=focal_idx,
    #     other_agent_idx=other_idx,
    #     shared_outcome_set=config.shared_outcome_set,
    #     horizon=config.horizon,
    #     lambdas=(config.lambda_E, config.lambda_R, config.lambda_O),
    # )

    # episode_stats = {
    #     "collision": bool(info.get("collision", False)),
    #     "success_i": bool(info.get("success_i", False)),
    #     "success_j": bool(info.get("success_j", False)),
    #     "timesteps": int(info.get("timesteps", config.num_timesteps)),
    #     "kappa": config.kappa_prior,
    # }

    # STUB
    metrics_per_policy = [
        {
            "policy_id": i,
            "G_i": np.random.randn(),
            "G_j": np.random.randn(),
            "G_joint": np.random.randn(),
            "F_i": np.random.randn(),
            "F_j": np.random.randn(),
            "F_joint": np.random.randn(),
            "E_i": np.random.randn(),
            "E_j": np.random.randn(),
            "R_i": np.random.randn(),
            "R_j": np.random.randn(),
            "O_ij": np.random.randn(),
            "J_i": np.random.randn(),  # NEW: decision variable
        }
        for i in range(10)
    ]

    episode_stats = {
        "collision": False,
        "success_i": True,
        "success_j": True,
        "timesteps": config.num_timesteps,
        "kappa": config.kappa_prior,
    }

    return metrics_per_policy, episode_stats


def save_results(
    all_policy_metrics,
    all_episode_stats,
    config: Exp2Config,
):
    """Save experiment results to disk."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save with kappa in filename
    kappa_str = f"kappa_{config.kappa_prior:.2f}".replace(".", "p")

    policy_file = output_dir / f"policy_metrics_{kappa_str}.pkl"
    with open(policy_file, "wb") as f:
        pickle.dump(all_policy_metrics, f)
    LOGGER.info(f"Saved policy metrics to {policy_file}")

    episode_file = output_dir / f"episode_stats_{kappa_str}.json"
    with open(episode_file, "w") as f:
        json.dump(all_episode_stats, f, indent=2)
    LOGGER.info(f"Saved episode stats to {episode_file}")

    config_file = output_dir / f"config_{kappa_str}.json"
    with open(config_file, "w") as f:
        json.dump(asdict(config), f, indent=2)
    LOGGER.info(f"Saved config to {config_file}")

    summary = compute_summary_statistics(all_policy_metrics, all_episode_stats)
    summary_file = output_dir / f"summary_{kappa_str}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info(f"Saved summary to {summary_file}")


def compute_summary_statistics(all_policy_metrics, all_episode_stats):
    """Compute summary statistics for Experiment 2."""

    F_values = [m["F_joint"] for m in all_policy_metrics]
    G_values = [m["G_joint"] for m in all_policy_metrics]
    J_values = [m.get("J_i", 0) for m in all_policy_metrics]

    correlation_F_G = float(np.corrcoef(F_values, G_values)[0, 1])
    correlation_F_J = float(np.corrcoef(F_values, J_values)[0, 1])

    num_collisions = sum(e["collision"] for e in all_episode_stats)
    num_success_i = sum(e["success_i"] for e in all_episode_stats)
    num_success_j = sum(e["success_j"] for e in all_episode_stats)

    summary = {
        "kappa": all_episode_stats[0]["kappa"],
        "correlation_F_G": correlation_F_G,
        "correlation_F_J": correlation_F_J,
        "num_episodes": len(all_episode_stats),
        "num_policies_evaluated": len(all_policy_metrics),
        "collision_rate": num_collisions / len(all_episode_stats),
        "success_rate_i": num_success_i / len(all_episode_stats),
        "success_rate_j": num_success_j / len(all_episode_stats),
        "mean_F_joint": float(np.mean(F_values)),
        "std_F_joint": float(np.std(F_values)),
        "mean_G_joint": float(np.mean(G_values)),
        "std_G_joint": float(np.std(G_values)),
        "mean_J_i": float(np.mean(J_values)),
        "std_J_i": float(np.std(J_values)),
    }

    return summary


def run_kappa_sweep(kappa_values: List[float], base_config: Exp2Config):
    """
    Run Experiment 2 for multiple κ values and compare.

    Parameters
    ----------
    kappa_values : List[float]
        List of κ values to sweep over
    base_config : Exp2Config
        Base configuration (will be copied and modified for each κ)

    Returns
    -------
    all_summaries : List[Dict]
        Summary statistics for each κ value
    """
    LOGGER.info("=" * 80)
    LOGGER.info(f"EXPERIMENT 2: κ SWEEP over {kappa_values}")
    LOGGER.info("=" * 80)

    all_summaries = []

    for kappa in kappa_values:
        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Running with κ = {kappa}")
        LOGGER.info(f"{'='*80}")

        # Create config for this κ value
        config = Exp2Config(**asdict(base_config))
        config.kappa_prior = kappa

        # Initialize
        key = jr.PRNGKey(config.seed)
        env, focal_agent, other_agents = init_env_and_agents(config)

        # Run episodes
        all_policy_metrics = []
        all_episode_stats = []

        for ep in range(config.num_episodes):
            if ep % 10 == 0:
                LOGGER.info(f"Episode {ep}/{config.num_episodes}")

            key, subkey = jr.split(key)
            metrics, stats = run_single_episode_exp2(
                key=subkey,
                env=env,
                focal_agent=focal_agent,
                other_agents=other_agents,
                config=config,
            )

            all_policy_metrics.extend(metrics)
            all_episode_stats.append(stats)

        # Save and summarize
        save_results(all_policy_metrics, all_episode_stats, config)
        summary = compute_summary_statistics(all_policy_metrics, all_episode_stats)
        all_summaries.append(summary)

        LOGGER.info(f"κ={kappa}: Collision rate = {summary['collision_rate']:.2%}")
        LOGGER.info(f"κ={kappa}: Mean F_joint = {summary['mean_F_joint']:.4f}")

    # Save comparison summary
    output_dir = Path(base_config.output_dir)
    comparison_file = output_dir / "kappa_sweep_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    LOGGER.info(f"\nSaved κ sweep comparison to {comparison_file}")

    return all_summaries


def main():
    """Main entry point for Experiment 2."""

    # Option 1: Single κ value
    # config = Exp2Config(kappa_prior=1.0)
    # ... run single experiment ...

    # Option 2: κ sweep (recommended)
    base_config = Exp2Config()
    kappa_values = [0.0, 0.5, 1.0, 2.0]

    all_summaries = run_kappa_sweep(kappa_values, base_config)

    # Print comparison
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("EXPERIMENT 2 COMPLETE - κ SWEEP COMPARISON")
    LOGGER.info("=" * 80)
    LOGGER.info(f"{'κ':<10} {'Collision%':<15} {'Mean F_joint':<15} {'Mean G_joint':<15}")
    LOGGER.info("-" * 80)
    for summary in all_summaries:
        LOGGER.info(
            f"{summary['kappa']:<10.2f} "
            f"{summary['collision_rate']*100:<15.1f} "
            f"{summary['mean_F_joint']:<15.4f} "
            f"{summary['mean_G_joint']:<15.4f}"
        )
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()
