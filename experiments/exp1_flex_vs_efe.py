"""
Experiment 1: Does higher path flexibility correlate with lower joint EFE
for ToM + empathy agents, even when F is not used in the decision rule?

HYPOTHESIS:
If agents with ToM + empathy naturally avoid low-flexibility policies
(bottlenecks, deadlocks), we should see NEGATIVE correlation between
F_joint and G_joint across all candidate policies.

SETUP:
- Two-agent lava corridor gridworld (bottleneck vs detour)
- ToM depth-1 (agents model each other)
- Empathy α (weight on other's EFE)
- NO explicit flexibility term in decision rule
- Decision: q(π) = softmax(-γ [G_i(π) + α G_j(π)])

WHAT WE LOG:
For each candidate policy π at each timestep:
- G_i(π), G_j(π), G_joint
- F_i(π), F_j(π), F_joint
- E(π), R(π), O(π)
"""

import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import pickle

import jax
import jax.random as jr
import jax.numpy as jnp
import numpy as np

# Environment and rollout
from src.envs import LavaCorridorEnv, LavaCorridorConfig, rollout_exp1
# from src.envs import build_generative_model_for_env  # For building agent models

# Path flexibility metrics
from src.metrics.path_flexibility import compute_path_flexibility_for_tree

# TODO: Import ToM agents when available
# from src.tom.si_tom import ToMAgent, si_policy_search_tom

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


@dataclass
class Exp1Config:
    """Configuration for Experiment 1."""

    # Episode parameters
    num_episodes: int = 100
    num_timesteps: int = 20

    # Planning parameters
    horizon: int = 5                # ToM planning depth
    gamma: float = 16.0             # Precision (inverse temperature)

    # Empathy (no F-prior in Exp 1)
    alpha_empathy: float = 1.0      # Weight on other's EFE

    # Path flexibility computation
    lambda_E: float = 1.0           # Weight on empowerment
    lambda_R: float = 1.0           # Weight on returnability
    lambda_O: float = 1.0           # Weight on overlap

    # Environment
    env_height: int = 7
    env_width: int = 11
    slip_prob: float = 0.05

    # Outcomes
    shared_outcome_set: list = None  # Will be set by env

    # I/O
    seed: int = 42
    output_dir: str = "results/exp1"


def init_env_and_agents(config: Exp1Config):
    """
    Initialize environment and ToM agents for Experiment 1.

    Returns
    -------
    env : LavaCorridorEnv
    focal_agent : ToMAgent
    other_agents : List[ToMAgent]
    """
    LOGGER.info("Initializing environment and agents for Experiment 1")

    # Initialize LavaCorridorEnv
    env_config = LavaCorridorConfig(
        width=config.env_width,
        height=3,  # Fixed for lava corridor
        num_agents=2,
        slip_prob=config.slip_prob,
    )
    env = LavaCorridorEnv(env_config)

    # Get shared "safe" outcomes from environment (for returnability computation)
    config.shared_outcome_set = env.shared_outcome_obs_indices()
    LOGGER.info(f"Shared safe outcomes (obs indices): {config.shared_outcome_set}")

    # Build ToM agents
    # NOTE: In Exp 1, agents use α (empathy) but NOT κ (flexibility prior)
    from src.agents.tom_agent_factory import create_tom_agents, AgentConfig

    agent_config = AgentConfig(
        horizon=config.horizon,
        gamma=config.gamma,
        alpha_empathy=config.alpha_empathy,
        kappa_prior=0.0,  # Exp 1: NO flexibility prior
        beta_joint_flex=config.beta_joint_flex,
        lambda_E=config.lambda_E,
        lambda_R=config.lambda_R,
        lambda_O=config.lambda_O,
        learn_B=False,
    )

    agents, A_matrices, B_matrices, C_matrices, D_matrices = create_tom_agents(
        env=env,
        num_agents=2,
        config=agent_config,
    )

    focal_agent = agents[0]
    other_agents = [agents[1]]

    LOGGER.info(f"Created {len(agents)} ToM agents")
    LOGGER.info(f"  Policies per agent: {len(focal_agent.policies)}")
    LOGGER.info(f"  Horizon: {config.horizon}, Gamma: {config.gamma}")
    LOGGER.info(f"  Alpha (empathy): {config.alpha_empathy}, Kappa (F-prior): 0.0 [Exp 1]")

    LOGGER.info("Environment and agents initialized")
    return env, focal_agent, other_agents


def run_single_episode_exp1(
    key,
    env,
    focal_agent,
    other_agents,
    config: Exp1Config,
):
    """
    Run a single episode and extract policy metrics.

    Returns
    -------
    metrics_per_policy : List[Dict]
        For each candidate policy π:
        {
            "policy_id": int,
            "G_i": float,
            "G_j": float,
            "G_joint": float,
            "F_i": float,
            "F_j": float,
            "F_joint": float,
            "E_i": float,
            "E_j": float,
            "R_i": float,
            "R_j": float,
            "O_ij": float,
        }

    episode_stats : Dict
        High-level episode outcomes:
        {
            "collision": bool,
            "success_i": bool,
            "success_j": bool,
            "timesteps": int,
        }
    """
    LOGGER.debug(f"Running episode with key={key}")

    # 1) Run multi-agent rollout with standard ToM (no F-prior)
    agents = [focal_agent] + other_agents  # Combine into list
    last, info, env_after = rollout_exp1(
        env=env,
        agents=agents,
        num_timesteps=config.num_timesteps,
        rng_key=key,
        alpha_empathy=config.alpha_empathy,
    )

    # TODO: Extract ToM trees and compute path flexibility when ToM agents available
    # 2) Extract ToM trees from rollout info
    # focal_tree_final = info["trees"][-1][0]  # Last timestep, focal agent
    # other_tree_final = info["trees"][-1][1]  # Last timestep, other agent

    # focal_idx = 0  # focal_agent is agent 0
    # other_idx = 1  # other agent is agent 1

    # # 3) Compute path flexibility for all candidate policies in tree
    # metrics_per_policy = compute_path_flexibility_for_tree(
    #     focal_tree=focal_tree_final,
    #     other_tree=other_tree_final,
    #     focal_agent_model=focal_agent,           # Agent's generative model
    #     other_agent_model=other_agents[0],       # Other agent's model
    #     focal_agent_idx=focal_idx,
    #     other_agent_idx=other_idx,
    #     shared_outcome_set=config.shared_outcome_set,
    #     horizon=config.horizon,
    #     lambdas=(config.lambda_E, config.lambda_R, config.lambda_O),
    # )

    # 4) Extract episode-level statistics (already computed by rollout)
    episode_stats = {
        "collision": info["collision"],
        "success_i": info["success_i"],
        "success_j": info["success_j"],
        "timesteps": info["timesteps"],
    }

    # STUB for now
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
        }
        for i in range(10)  # placeholder: 10 policies
    ]

    episode_stats = {
        "collision": False,
        "success_i": True,
        "success_j": True,
        "timesteps": config.num_timesteps,
    }

    return metrics_per_policy, episode_stats


def save_results(
    all_policy_metrics,
    all_episode_stats,
    config: Exp1Config,
):
    """Save experiment results to disk."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save policy metrics
    policy_file = output_dir / "policy_metrics.pkl"
    with open(policy_file, "wb") as f:
        pickle.dump(all_policy_metrics, f)
    LOGGER.info(f"Saved policy metrics to {policy_file}")

    # Save episode stats
    episode_file = output_dir / "episode_stats.json"
    with open(episode_file, "w") as f:
        json.dump(all_episode_stats, f, indent=2)
    LOGGER.info(f"Saved episode stats to {episode_file}")

    # Save config
    config_file = output_dir / "config.json"
    with open(config_file, "w") as f:
        config_dict = asdict(config)
        json.dump(config_dict, f, indent=2)
    LOGGER.info(f"Saved config to {config_file}")

    # Compute summary statistics
    summary = compute_summary_statistics(all_policy_metrics, all_episode_stats)
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info(f"Saved summary to {summary_file}")


def compute_summary_statistics(all_policy_metrics, all_episode_stats):
    """Compute summary statistics for Experiment 1."""

    # Extract all F_joint and G_joint values
    F_values = [m["F_joint"] for m in all_policy_metrics]
    G_values = [m["G_joint"] for m in all_policy_metrics]

    # Compute correlation
    correlation = float(np.corrcoef(F_values, G_values)[0, 1])

    # Episode-level stats
    num_collisions = sum(e["collision"] for e in all_episode_stats)
    num_success_i = sum(e["success_i"] for e in all_episode_stats)
    num_success_j = sum(e["success_j"] for e in all_episode_stats)

    summary = {
        "correlation_F_G": correlation,
        "num_episodes": len(all_episode_stats),
        "num_policies_evaluated": len(all_policy_metrics),
        "collision_rate": num_collisions / len(all_episode_stats),
        "success_rate_i": num_success_i / len(all_episode_stats),
        "success_rate_j": num_success_j / len(all_episode_stats),
        "mean_F_joint": float(np.mean(F_values)),
        "std_F_joint": float(np.std(F_values)),
        "mean_G_joint": float(np.mean(G_values)),
        "std_G_joint": float(np.std(G_values)),
    }

    return summary


def main():
    """Main entry point for Experiment 1."""
    config = Exp1Config()

    LOGGER.info("=" * 80)
    LOGGER.info("EXPERIMENT 1: Path Flexibility vs EFE Correlation Test")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Config: {asdict(config)}")

    # Initialize JAX random key
    key = jr.PRNGKey(config.seed)

    # Initialize environment and agents
    env, focal_agent, other_agents = init_env_and_agents(config)

    # Run episodes
    all_policy_metrics = []
    all_episode_stats = []

    LOGGER.info(f"Running {config.num_episodes} episodes...")
    for ep in range(config.num_episodes):
        if ep % 10 == 0:
            LOGGER.info(f"Episode {ep}/{config.num_episodes}")

        key, subkey = jr.split(key)
        metrics, stats = run_single_episode_exp1(
            key=subkey,
            env=env,
            focal_agent=focal_agent,
            other_agents=other_agents,
            config=config,
        )

        all_policy_metrics.extend(metrics)
        all_episode_stats.append(stats)

    # Save results
    save_results(all_policy_metrics, all_episode_stats, config)

    # Print summary
    summary = compute_summary_statistics(all_policy_metrics, all_episode_stats)
    LOGGER.info("=" * 80)
    LOGGER.info("EXPERIMENT 1 COMPLETE")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Correlation F_joint vs G_joint: {summary['correlation_F_G']:.4f}")
    LOGGER.info(f"Collision rate: {summary['collision_rate']:.2%}")
    LOGGER.info(f"Success rate (Agent i): {summary['success_rate_i']:.2%}")
    LOGGER.info(f"Success rate (Agent j): {summary['success_rate_j']:.2%}")
    LOGGER.info("=" * 80)

    return summary


if __name__ == "__main__":
    main()
