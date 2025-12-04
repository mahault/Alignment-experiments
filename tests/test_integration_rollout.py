"""
Integration test for full rollout with real ToM agents.

This test verifies that the entire pipeline works:
1. Create environment
2. Create agents
3. Run rollout
4. Compute path flexibility metrics
"""

import pytest
import numpy as np
import logging

from src.envs.lava_corridor import LavaCorridorEnv, LavaCorridorConfig
from src.agents.tom_agent_factory import create_tom_agents, AgentConfig, get_shared_outcomes
from src.envs.rollout_lava import rollout_exp1, rollout_exp2
from src.tom.si_tom_F_prior import ToMPolicyConfig

# Set logging to see what's happening
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class TestIntegrationBasic:
    """Basic integration tests."""

    def test_create_env_and_agents(self):
        """Test that we can create environment and agents."""
        env_config = LavaCorridorConfig(width=5, height=3, num_agents=2)
        env = LavaCorridorEnv(env_config)

        agent_config = AgentConfig(horizon=2, gamma=16.0)
        agents, A, B, C, D = create_tom_agents(env, num_agents=2, config=agent_config)

        print(f"\nEnvironment and agents created:")
        print(f"  Env size: {env_config.width}x{env_config.height}")
        print(f"  Num agents: {len(agents)}")
        print(f"  Policies per agent: {len(agents[0].policies)}")

        assert len(agents) == 2
        assert len(agents[0].policies) > 0


class TestExp1Rollout:
    """Test Experiment 1 rollout."""

    def test_exp1_basic_rollout(self):
        """Test a basic Exp1 rollout with small parameters."""
        # Create small environment
        env_config = LavaCorridorConfig(
            width=4,
            height=2,
            num_agents=2,
            lava_positions=[],  # No lava for simplicity
        )
        env = LavaCorridorEnv(env_config)

        # Create agents with small horizon
        agent_config = AgentConfig(
            horizon=2,  # Small horizon for speed
            gamma=16.0,
            alpha_empathy=0.5,
            kappa_prior=0.0,  # Exp 1: no F-prior
        )
        agents, A, B, C, D = create_tom_agents(env, num_agents=2, config=agent_config)

        print(f"\nExp1 rollout test:")
        print(f"  Agents: {len(agents)}")
        print(f"  Policies: {len(agents[0].policies)}")
        print(f"  Horizon: {agent_config.horizon}")

        # Run short rollout
        try:
            last, info, env_after = rollout_exp1(
                env=env,
                agents=agents,
                num_timesteps=5,
                alpha_empathy=0.5,
            )

            print(f"\nRollout completed:")
            print(f"  Timesteps: {info['timesteps']}")
            print(f"  Success: {info.get('success_i', False)} / {info.get('success_j', False)}")
            print(f"  Collision: {info.get('collision', False)}")

            # Basic checks
            assert info['timesteps'] > 0, "Should run at least 1 timestep"
            assert 'states' in info, "Should track states"
            assert 'actions' in info, "Should track actions"
            assert len(info['actions']) == info['timesteps'], "Actions should match timesteps"

            print("  ✓ Exp1 rollout successful!")

        except Exception as e:
            LOGGER.error(f"Exp1 rollout failed: {e}")
            raise


class TestExp2Rollout:
    """Test Experiment 2 rollout with F-aware prior."""

    def test_exp2_basic_rollout(self):
        """Test a basic Exp2 rollout with F-aware prior."""
        # Create environment
        env_config = LavaCorridorConfig(
            width=4,
            height=2,
            num_agents=2,
            lava_positions=[],
        )
        env = LavaCorridorEnv(env_config)

        # Get shared outcomes
        shared_outcomes = get_shared_outcomes(env)

        # Create agents
        agent_config = AgentConfig(
            horizon=2,
            gamma=16.0,
            alpha_empathy=0.5,
            kappa_prior=1.0,  # Exp 2: USE F-prior
            beta_joint_flex=0.5,
        )
        agents, A, B, C, D = create_tom_agents(env, num_agents=2, config=agent_config)

        # Create ToM config
        tom_config = ToMPolicyConfig(
            horizon=2,
            gamma=16.0,
            alpha_empathy=0.5,
            kappa_prior=1.0,
            beta_joint_flex=0.5,
            shared_outcome_set=shared_outcomes,
        )

        print(f"\nExp2 rollout test:")
        print(f"  Kappa: {tom_config.kappa_prior}")
        print(f"  Beta: {tom_config.beta_joint_flex}")
        print(f"  Shared outcomes: {len(shared_outcomes)}")

        # Run rollout
        try:
            last, info, env_after = rollout_exp2(
                env=env,
                agents=agents,
                num_timesteps=5,
                tom_config=tom_config,
            )

            print(f"\nRollout completed:")
            print(f"  Timesteps: {info['timesteps']}")
            print(f"  Success: {info.get('success_i', False)} / {info.get('success_j', False)}")
            print(f"  Collision: {info.get('collision', False)}")

            # Basic checks
            assert info['timesteps'] > 0
            assert 'states' in info
            assert 'actions' in info

            print("  ✓ Exp2 rollout with F-prior successful!")

        except Exception as e:
            LOGGER.error(f"Exp2 rollout failed: {e}")
            raise


class TestRolloutOutputs:
    """Test that rollout outputs have correct structure."""

    def test_rollout_info_structure(self):
        """Test that rollout info dict has all required keys."""
        env_config = LavaCorridorConfig(width=4, height=2, num_agents=2)
        env = LavaCorridorEnv(env_config)

        agent_config = AgentConfig(horizon=2)
        agents, _, _, _, _ = create_tom_agents(env, num_agents=2, config=agent_config)

        last, info, env_after = rollout_exp1(
            env=env,
            agents=agents,
            num_timesteps=3,
        )

        print(f"\nRollout info keys: {info.keys()}")

        # Check required keys
        required_keys = ['states', 'observations', 'actions', 'timesteps']
        for key in required_keys:
            assert key in info, f"Missing required key: {key}"

        # Check types
        assert isinstance(info['timesteps'], int)
        assert isinstance(info['states'], list)
        assert isinstance(info['observations'], list)
        assert isinstance(info['actions'], list)

        print("  ✓ Rollout output structure correct!")


@pytest.mark.skip(reason="Takes longer - run manually for full validation")
class TestLongerRollout:
    """Test with longer horizon and more timesteps."""

    def test_longer_exp1_rollout(self):
        """Test Exp1 with more realistic parameters."""
        env_config = LavaCorridorConfig(width=8, height=3, num_agents=2)
        env = LavaCorridorEnv(env_config)

        agent_config = AgentConfig(horizon=3, gamma=16.0, alpha_empathy=1.0)
        agents, _, _, _, _ = create_tom_agents(env, num_agents=2, config=agent_config)

        print(f"\nLonger rollout test:")
        print(f"  Horizon: 3")
        print(f"  Timesteps: 20")
        print(f"  Policies: {len(agents[0].policies)}")

        last, info, env_after = rollout_exp1(
            env=env,
            agents=agents,
            num_timesteps=20,
            alpha_empathy=1.0,
        )

        print(f"\nCompleted:")
        print(f"  Total timesteps: {info['timesteps']}")
        print(f"  Success: {info.get('success_i', False)}")

        assert info['timesteps'] <= 20
        print("  ✓ Longer rollout successful!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-k", "not skip"])
