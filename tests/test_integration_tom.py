"""
Integration test for TOM-style components working together.

This test verifies:
1. LavaModel + LavaAgent + LavaV1Env can be created together
2. Manual Bayesian inference works with env observations
3. Policy evaluation using B matrix
4. Multi-agent coordination (basic)
"""

import os
import sys

# Ensure repo root is on sys.path so `tom` can be imported
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import numpy as np
import jax.random as jr
import jax.numpy as jnp
import logging

from tom.models import LavaModel, LavaAgent
from tom.envs import LavaV1Env

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class TestIntegrationBasic:
    """Basic integration tests."""

    def test_create_all_components(self):
        """Test that we can create all TOM components together."""
        # Create model
        model = LavaModel(width=4, height=3, goal_x=3)

        # Create agent
        agent = LavaAgent(model, horizon=1, gamma=8.0)

        # Create environment
        env = LavaV1Env(width=4, height=3, num_agents=1, timesteps=10)

        print(f"\nAll components created:")
        print(f"  Model states: {model.num_states}")
        print(f"  Agent policies: {len(agent.policies)}")
        print(f"  Env size: {env.width}x{env.height}")

        assert model.num_states == 12  # 4*3
        assert len(agent.policies) == 5
        assert env.width == 4

    def test_model_env_compatibility(self):
        """Test that model and env have compatible state spaces."""
        model = LavaModel(width=5, height=3)
        env = LavaV1Env(width=5, height=3, num_agents=1, timesteps=10)

        # State spaces should match
        model_states = model.num_states
        env_states = env.width * env.height

        assert model_states == env_states, \
            f"Model ({model_states}) and env ({env_states}) state spaces should match"

        print(f"\nModel-Env compatibility:")
        print(f"  Model states: {model_states}")
        print(f"  Env states: {env_states}")


class TestManualInference:
    """Test manual Bayesian inference with environment."""

    def test_inference_from_env_observation(self):
        """Test that we can do Bayesian inference from env observation."""
        # Create components
        model = LavaModel(width=4, height=3)
        env = LavaV1Env(width=4, height=3, num_agents=1, timesteps=10)

        # Reset environment
        key = jr.PRNGKey(0)
        state, obs = env.reset(key)

        # Get observation for agent 0
        agent_obs = int(np.asarray(obs[0]["location_obs"])[0])

        # Manual Bayesian inference
        A0 = np.asarray(model.A["location_obs"])
        D0 = np.asarray(model.D["location_state"])

        likelihood = A0[agent_obs]
        unnorm = likelihood * D0
        qs = unnorm / unnorm.sum()

        print(f"\nManual inference:")
        print(f"  Observation: {agent_obs}")
        print(f"  Posterior shape: {qs.shape}")
        print(f"  Posterior sum: {qs.sum():.4f}")
        print(f"  Max belief: {qs.max():.4f} at state {qs.argmax()}")

        # Checks
        assert qs.shape == (12,), "Posterior should have 12 states"
        assert np.isclose(qs.sum(), 1.0), "Posterior should sum to 1"
        assert qs.argmax() == agent_obs, "Should believe in observed state"

    def test_belief_update_after_action(self):
        """Test belief update after taking an action."""
        model = LavaModel(width=4, height=3)
        env = LavaV1Env(width=4, height=3, num_agents=1, timesteps=10)

        key = jr.PRNGKey(0)
        state, obs = env.reset(key)

        # Initial observation and belief
        obs_0 = int(np.asarray(obs[0]["location_obs"])[0])
        A = np.asarray(model.A["location_obs"])
        B = np.asarray(model.B["location_state"])  # 4D: (s', s, s_other, a)
        D = np.asarray(model.D["location_state"])

        # Initial belief
        qs_0 = A[obs_0] * D
        qs_0 = qs_0 / qs_0.sum()

        # For single-agent, marginalize over s_other (uniform)
        qs_other = np.ones(model.num_states) / model.num_states

        # Take action RIGHT
        RIGHT = 3
        next_state, next_obs, _, _, _ = env.step(state, {0: RIGHT})

        # New observation
        obs_1 = int(np.asarray(next_obs[0]["location_obs"])[0])

        # Predicted belief (using 4D B matrix - marginalize over s_other)
        qs_pred = np.zeros(model.num_states)
        for s_other in range(model.num_states):
            qs_pred += B[:, :, s_other, RIGHT] @ qs_0 * qs_other[s_other]

        # Updated belief (using new observation)
        qs_1 = A[obs_1] * qs_pred
        qs_1 = qs_1 / qs_1.sum()

        print(f"\nBelief update:")
        print(f"  Initial obs: {obs_0}")
        print(f"  Action: RIGHT")
        print(f"  New obs: {obs_1}")
        print(f"  New belief max: {qs_1.max():.4f} at state {qs_1.argmax()}")

        assert np.isclose(qs_1.sum(), 1.0), "Updated belief should sum to 1"
        assert qs_1.argmax() == obs_1, "Should believe in new observed state"


class TestPolicyEvaluation:
    """Test policy evaluation using model."""

    def test_policy_forward_simulation(self):
        """Test forward simulation of a policy using B matrix."""
        model = LavaModel(width=4, height=3)

        # Start at state (1, 0) - middle row, leftmost
        s0 = 1 * model.width + 0  # y=1, x=0
        qs = np.zeros(model.num_states)
        qs[s0] = 1.0

        # For single-agent, marginalize over s_other (uniform)
        qs_other = np.ones(model.num_states) / model.num_states

        # Policy: [RIGHT, RIGHT] - move right twice
        policy = [3, 3]
        B = np.asarray(model.B["location_state"])  # 4D: (s', s, s_other, a)

        # Simulate forward
        qs_t = qs.copy()
        trajectory = [s0]

        for action in policy:
            # Predict next state (marginalize over s_other)
            qs_next = np.zeros(model.num_states)
            for s_other in range(model.num_states):
                qs_next += B[:, :, s_other, action] @ qs_t * qs_other[s_other]
            qs_t = qs_next
            s_t = qs_t.argmax()
            trajectory.append(s_t)

        print(f"\nPolicy simulation:")
        print(f"  Policy: RIGHT, RIGHT")
        print(f"  Trajectory: {trajectory}")

        # Should move right twice: 4 -> 5 -> 6
        expected = [4, 5, 6]
        assert trajectory == expected, \
            f"Trajectory {trajectory} should be {expected}"

    def test_expected_outcome_distribution(self):
        """Test computing expected outcome distribution over a policy."""
        model = LavaModel(width=4, height=3)
        A = np.asarray(model.A["location_obs"])
        B = np.asarray(model.B["location_state"])  # 4D: (s', s, s_other, a)
        D = np.asarray(model.D["location_state"])

        # Start belief
        qs = D.copy()

        # For single-agent, marginalize over s_other (uniform)
        qs_other = np.ones(model.num_states) / model.num_states

        # Policy: STAY for 3 steps
        policy = [4, 4, 4]

        # Collect observation distributions
        obs_dists = []

        qs_t = qs.copy()
        for action in policy:
            # Predict next belief (marginalize over s_other)
            qs_next = np.zeros(model.num_states)
            for s_other in range(model.num_states):
                qs_next += B[:, :, s_other, action] @ qs_t * qs_other[s_other]
            qs_t = qs_next

            # Compute observation distribution
            obs_dist = A @ qs_t
            obs_dists.append(obs_dist)

        print(f"\nExpected observations:")
        print(f"  Num timesteps: {len(obs_dists)}")
        print(f"  Each dist sums to: {[o.sum() for o in obs_dists]}")

        # All distributions should sum to 1
        for obs_dist in obs_dists:
            assert np.isclose(obs_dist.sum(), 1.0), \
                "Observation distribution should sum to 1"


class TestMultiAgentInteraction:
    """Test multi-agent scenarios."""

    def test_two_agent_env(self):
        """Test creating and stepping with two agents."""
        env = LavaV1Env(width=4, height=3, num_agents=2, timesteps=10)
        key = jr.PRNGKey(0)

        state, obs = env.reset(key)

        # Should have 2 observations
        assert len(obs) == 2, "Should have 2 agent observations"
        assert "location_obs" in obs[0]
        assert "location_obs" in obs[1]

        # Take step with both agents
        actions = {0: 3, 1: 3}  # Both move RIGHT
        next_state, next_obs, reward, done, info = env.step(state, actions)

        assert len(next_obs) == 2, "Should still have 2 observations"

        print(f"\nMulti-agent step:")
        # Positions are stored inside state["env_state"]["pos"]
        initial_pos = state["env_state"]["pos"]
        next_pos = next_state["env_state"]["pos"]

        print(f"  Initial positions: {initial_pos}")
        print(f"  Actions: {actions}")
        print(f"  New positions: {next_pos}")

        # Sanity checks
        assert isinstance(initial_pos, dict), "Positions should be dict"
        assert isinstance(next_pos, dict), "Positions should be dict"
        assert len(initial_pos) == 2, "Should have positions for 2 agents"
        assert len(next_pos) == 2, "Should have positions for 2 agents"

    def test_two_agent_separate_beliefs(self):
        """Test maintaining separate beliefs for two agents."""
        model_i = LavaModel(width=4, height=3)
        model_j = LavaModel(width=4, height=3)

        env = LavaV1Env(width=4, height=3, num_agents=2, timesteps=10)
        key = jr.PRNGKey(42)

        state, obs = env.reset(key)

        # Extract observations
        obs_i = int(np.asarray(obs[0]["location_obs"])[0])
        obs_j = int(np.asarray(obs[1]["location_obs"])[0])

        # Separate inference for each agent
        A_i = np.asarray(model_i.A["location_obs"])
        D_i = np.asarray(model_i.D["location_state"])

        A_j = np.asarray(model_j.A["location_obs"])
        D_j = np.asarray(model_j.D["location_state"])

        qs_i = A_i[obs_i] * D_i
        qs_i = qs_i / qs_i.sum()

        qs_j = A_j[obs_j] * D_j
        qs_j = qs_j / qs_j.sum()

        print(f"\nSeparate agent beliefs:")
        print(f"  Agent i obs: {obs_i}, belief at: {qs_i.argmax()}")
        print(f"  Agent j obs: {obs_j}, belief at: {qs_j.argmax()}")

        assert qs_i.argmax() == obs_i
        assert qs_j.argmax() == obs_j


class TestEndToEndScenario:
    """Test end-to-end scenario with all components."""

    def test_full_pipeline(self):
        """Test full pipeline: create, reset, observe, infer, predict, step."""
        # 1. Create components
        model = LavaModel(width=4, height=3)
        agent = LavaAgent(model, horizon=1, gamma=8.0)
        env = LavaV1Env(width=4, height=3, num_agents=1, timesteps=10)

        # 2. Reset environment
        key = jr.PRNGKey(0)
        state, obs = env.reset(key)

        # 3. Observe
        obs_t = int(np.asarray(obs[0]["location_obs"])[0])

        # 4. Infer state
        A = np.asarray(model.A["location_obs"])
        D = np.asarray(model.D["location_state"])
        qs = A[obs_t] * D
        qs = qs / qs.sum()

        # For single-agent, marginalize over s_other (uniform)
        qs_other = np.ones(model.num_states) / model.num_states

        # 5. Evaluate policies (simple: just count expected reward)
        B = np.asarray(model.B["location_state"])  # 4D: (s', s, s_other, a)
        C = np.asarray(model.C["location_obs"])

        policy_values = []
        for policy_idx in range(len(agent.policies)):
            action = int(agent.policies[policy_idx, 0, 0])

            # Predict next state (marginalize over s_other)
            qs_next = np.zeros(model.num_states)
            for s_other in range(model.num_states):
                qs_next += B[:, :, s_other, action] @ qs * qs_other[s_other]

            # Expected observation
            obs_dist = A @ qs_next

            # Expected preference
            expected_C = (obs_dist * C).sum()

            policy_values.append(expected_C)

        # 6. Select best policy (highest expected C)
        best_policy = np.argmax(policy_values)
        best_action = int(agent.policies[best_policy, 0, 0])

        # 7. Step
        next_state, next_obs, _, _, _ = env.step(state, {0: best_action})

        print(f"\nEnd-to-end pipeline:")
        print(f"  Initial obs: {obs_t}")
        print(f"  Policy values: {policy_values}")
        print(f"  Best policy: {best_policy}, action: {best_action}")
        print(f"  New obs: {int(np.asarray(next_obs[0]['location_obs'])[0])}")

        # Should have selected a valid action
        assert 0 <= best_action < 5, "Should select valid action"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
