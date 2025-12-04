"""
Test script to verify LavaCorridorEnv and rollout integration.

This script tests:
1. Environment initialization
2. Basic rollout with stub agents
3. Collision detection
4. Success detection
5. Logging functionality
"""

import logging
import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.envs import (
    LavaCorridorEnv,
    LavaCorridorConfig,
    rollout_multi_agent_lava,
    build_generative_model_for_env,
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


def test_env_initialization():
    """Test that environment initializes correctly."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: Environment Initialization")
    LOGGER.info("=" * 80)

    config = LavaCorridorConfig(
        width=7,
        height=3,
        num_agents=2,
        slip_prob=0.0,
    )

    env = LavaCorridorEnv(config)

    # Test reset
    state, obs = env.reset()

    assert state["t"] == 0
    assert not state["done"]
    assert not state["lava_hit"]
    assert not state["success"]

    LOGGER.info("✓ Environment initialized successfully")
    LOGGER.info(f"  Grid size: {env.width}x{env.height}")
    LOGGER.info(f"  Num agents: {env.num_agents}")
    LOGGER.info(f"  Initial observations: {obs}")

    return env


def test_generative_model():
    """Test generative model construction."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: Generative Model Construction")
    LOGGER.info("=" * 80)

    config = LavaCorridorConfig(width=7, height=3, num_agents=2)
    env = LavaCorridorEnv(config)

    # Build model for agent 0
    model = build_generative_model_for_env(env, agent_id=0)

    assert "A" in model
    assert "B" in model
    assert "C" in model
    assert "D" in model

    num_states = env.get_num_states()
    num_obs = env.get_num_observations()
    num_actions = env.get_num_actions()

    assert model["A"].shape == (num_obs, num_states)
    assert model["B"].shape == (num_actions, num_states, num_states)
    assert model["C"].shape == (num_obs,)
    assert model["D"].shape == (num_states,)

    LOGGER.info("✓ Generative model constructed successfully")
    LOGGER.info(f"  States: {num_states}, Obs: {num_obs}, Actions: {num_actions}")
    LOGGER.info(f"  A shape: {model['A'].shape}")
    LOGGER.info(f"  B shape: {model['B'].shape}")

    return model


def test_shared_outcomes():
    """Test shared outcomes computation."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: Shared Outcomes")
    LOGGER.info("=" * 80)

    config = LavaCorridorConfig(width=7, height=3, num_agents=2)
    env = LavaCorridorEnv(config)

    outcomes = env.shared_outcomes()
    obs_indices = env.shared_outcome_obs_indices()

    assert len(outcomes) == env.width
    assert len(obs_indices) == env.width

    # Verify all outcomes are in safe row
    for x, y in outcomes:
        assert y == env.safe_y

    LOGGER.info("✓ Shared outcomes computed successfully")
    LOGGER.info(f"  Number of safe outcomes: {len(outcomes)}")
    LOGGER.info(f"  Positions: {outcomes}")
    LOGGER.info(f"  Obs indices: {obs_indices}")


def test_basic_rollout():
    """Test basic rollout with stub agents."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: Basic Rollout (Stub Agents)")
    LOGGER.info("=" * 80)

    config = LavaCorridorConfig(width=7, height=3, num_agents=2)
    env = LavaCorridorEnv(config)

    # Stub agents (None = rollout will use heuristic)
    agents = [None, None]

    last, info, env_after = rollout_multi_agent_lava(
        env=env,
        agents=agents,
        num_timesteps=10,
        use_F_prior=False,
    )

    assert "collision" in info
    assert "success" in info
    assert "timesteps" in info
    assert "states" in info
    assert "observations" in info
    assert "actions" in info

    LOGGER.info("✓ Basic rollout completed successfully")
    LOGGER.info(f"  Timesteps taken: {info['timesteps']}")
    LOGGER.info(f"  Collision: {info['collision']}")
    LOGGER.info(f"  Success: {info['success']}")
    LOGGER.info(f"  Success agent 0: {info['success_i']}")
    LOGGER.info(f"  Success agent 1: {info['success_j']}")
    LOGGER.info(f"  Lava hit: {info['lava_hit']}")


def test_rendering():
    """Test environment rendering."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Rendering")
    LOGGER.info("=" * 80)

    config = LavaCorridorConfig(width=7, height=3, num_agents=2)
    env = LavaCorridorEnv(config)

    state, obs = env.reset()

    render_str = env.render(state)

    assert len(render_str) > 0
    assert "t=0" in render_str

    LOGGER.info("✓ Rendering works")
    LOGGER.info("\n" + render_str)


def main():
    """Run all tests."""
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("LAVA CORRIDOR ENVIRONMENT & ROLLOUT TESTS")
    LOGGER.info("=" * 80 + "\n")

    try:
        test_env_initialization()
        test_generative_model()
        test_shared_outcomes()
        test_basic_rollout()
        test_rendering()

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("ALL TESTS PASSED ✓")
        LOGGER.info("=" * 80)

    except Exception as e:
        LOGGER.error(f"\n{'='*80}")
        LOGGER.error(f"TEST FAILED ✗")
        LOGGER.error(f"{'='*80}")
        LOGGER.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
