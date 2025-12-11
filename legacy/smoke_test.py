"""
Smoke test to verify all components work together.

Run this script to quickly verify that:
1. Tests pass
2. Experiments can run
3. Basic functionality works end-to-end

Usage:
    python smoke_test.py
"""

import sys
import logging
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def test_imports():
    """Test that all critical imports work."""
    LOGGER.info("="*80)
    LOGGER.info("STEP 1: Testing imports...")
    LOGGER.info("="*80)

    try:
        # Environment
        from src.envs.lava_corridor import LavaCorridorEnv, LavaCorridorConfig
        from src.envs.rollout_lava import rollout_exp1, rollout_exp2
        LOGGER.info("  ✓ Environment imports")

        # Agents
        from src.agents.tom_agent_factory import create_tom_agents, AgentConfig
        LOGGER.info("  ✓ Agent factory imports")

        # Metrics
        from src.metrics.path_flexibility import compute_path_flexibility
        from src.metrics.empowerment import estimate_empowerment_one_step
        LOGGER.info("  ✓ Metrics imports")

        # ToM
        from src.tom.si_tom import run_tom_step
        from src.tom.si_tom_F_prior import run_tom_step_with_F_prior, ToMPolicyConfig
        LOGGER.info("  ✓ ToM imports")

        LOGGER.info("\n✅ All imports successful!\n")
        return True

    except Exception as e:
        LOGGER.error(f"\n❌ Import failed: {e}\n")
        return False


def test_env_and_agents():
    """Test creating environment and agents."""
    LOGGER.info("="*80)
    LOGGER.info("STEP 2: Testing environment and agent creation...")
    LOGGER.info("="*80)

    try:
        from src.envs.lava_corridor import LavaCorridorEnv, LavaCorridorConfig
        from src.agents.tom_agent_factory import create_tom_agents, AgentConfig

        # Create small environment (height must be 3 for lava-safe-lava design)
        env_config = LavaCorridorConfig(width=4, height=3, num_agents=2)
        env = LavaCorridorEnv(env_config)
        LOGGER.info(f"  ✓ Environment created: {env_config.width}x{env_config.height}")

        # Create agents with minimal horizon
        agent_config = AgentConfig(horizon=2, gamma=16.0)
        agents, A, B, C, D = create_tom_agents(env, num_agents=2, config=agent_config)
        LOGGER.info(f"  ✓ Agents created: {len(agents)} agents, {len(agents[0].policies)} policies")

        LOGGER.info("\n✅ Environment and agents created successfully!\n")
        return True

    except Exception as e:
        LOGGER.error(f"\n❌ Failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_exp1_rollout():
    """Test Experiment 1 rollout."""
    LOGGER.info("="*80)
    LOGGER.info("STEP 3: Testing Experiment 1 rollout...")
    LOGGER.info("="*80)

    try:
        from src.envs.lava_corridor import LavaCorridorEnv, LavaCorridorConfig
        from src.agents.tom_agent_factory import create_tom_agents, AgentConfig
        from src.envs.rollout_lava import rollout_exp1

        # Setup (height must be 3 for lava-safe-lava design)
        env = LavaCorridorEnv(LavaCorridorConfig(width=4, height=3, num_agents=2))
        agent_config = AgentConfig(horizon=2, gamma=16.0, kappa_prior=0.0)
        agents, _, _, _, _ = create_tom_agents(env, num_agents=2, config=agent_config)

        # Run short rollout
        LOGGER.info("  Running 3-step rollout...")
        last, info, env_after = rollout_exp1(
            env=env,
            agents=agents,
            num_timesteps=3,
            alpha_empathy=0.5,
        )

        LOGGER.info(f"  ✓ Rollout completed: {info['timesteps']} timesteps")
        LOGGER.info(f"  ✓ Tracked: states={len(info['states'])}, actions={len(info['actions'])}")

        LOGGER.info("\n✅ Experiment 1 rollout successful!\n")
        return True

    except Exception as e:
        LOGGER.error(f"\n❌ Exp1 failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_exp2_rollout():
    """Test Experiment 2 rollout with F-aware prior."""
    LOGGER.info("="*80)
    LOGGER.info("STEP 4: Testing Experiment 2 rollout (F-aware prior)...")
    LOGGER.info("="*80)

    try:
        from src.envs.lava_corridor import LavaCorridorEnv, LavaCorridorConfig
        from src.agents.tom_agent_factory import create_tom_agents, AgentConfig, get_shared_outcomes
        from src.envs.rollout_lava import rollout_exp2
        from src.tom.si_tom_F_prior import ToMPolicyConfig

        # Setup (height must be 3 for lava-safe-lava design)
        env = LavaCorridorEnv(LavaCorridorConfig(width=4, height=3, num_agents=2))
        shared_outcomes = get_shared_outcomes(env)

        agent_config = AgentConfig(horizon=2, gamma=16.0, kappa_prior=1.0, beta_joint_flex=0.5)
        agents, _, _, _, _ = create_tom_agents(env, num_agents=2, config=agent_config)

        tom_config = ToMPolicyConfig(
            horizon=2,
            gamma=16.0,
            kappa_prior=1.0,
            beta_joint_flex=0.5,
            shared_outcome_set=shared_outcomes,
        )

        # Run short rollout
        LOGGER.info("  Running 3-step rollout with κ=1.0...")
        last, info, env_after = rollout_exp2(
            env=env,
            agents=agents,
            num_timesteps=3,
            tom_config=tom_config,
        )

        LOGGER.info(f"  ✓ Rollout completed: {info['timesteps']} timesteps")
        LOGGER.info(f"  ✓ F-aware prior applied successfully")

        LOGGER.info("\n✅ Experiment 2 rollout successful!\n")
        return True

    except Exception as e:
        LOGGER.error(f"\n❌ Exp2 failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests."""
    print("\n" + "="*80)
    print(" "*25 + "SMOKE TEST SUITE")
    print("="*80 + "\n")

    results = []

    # Test 1: Imports
    results.append(("Imports", test_imports()))

    # Test 2: Environment and Agents
    results.append(("Environment & Agents", test_env_and_agents()))

    # Test 3: Exp1 Rollout
    results.append(("Experiment 1 Rollout", test_exp1_rollout()))

    # Test 4: Exp2 Rollout
    results.append(("Experiment 2 Rollout (F-prior)", test_exp2_rollout()))

    # Summary
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY")
    print("="*80)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:.<50} {status}")
        if not passed:
            all_passed = False

    print("="*80 + "\n")

    if all_passed:
        print("🎉 ALL TESTS PASSED! System is ready to run experiments.")
        print("\nNext steps:")
        print("  1. Run integration tests: pytest tests/test_integration_rollout.py -v -s")
        print("  2. Run Experiment 1: python experiments/exp1_flex_vs_efe.py")
        print("  3. Run Experiment 2: python experiments/exp2_flex_prior.py")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
