"""
Unit tests for ToM agent factory.

Verifies that agents are created correctly with proper A, B, C, D matrices.
"""

import pytest
import numpy as np

from src.agents.tom_agent_factory import (
    create_tom_agents,
    create_policy_library,
    get_shared_outcomes,
    create_mock_agents_for_testing,
    AgentConfig,
)
from src.envs.lava_corridor import LavaCorridorEnv, LavaCorridorConfig


class TestAgentCreation:
    """Test agent creation from environment."""

    def test_create_agents_basic(self):
        """Test basic agent creation."""
        env = LavaCorridorEnv(LavaCorridorConfig(width=5, height=3))
        config = AgentConfig(horizon=2, gamma=16.0)

        agents, A, B, C, D = create_tom_agents(env, num_agents=2, config=config)

        print(f"\nBasic agent creation:")
        print(f"  Number of agents: {len(agents)}")
        print(f"  Policies per agent: {len(agents[0].policies)}")
        print(f"  State space size: {A[0][0].shape[1]}")
        print(f"  Obs space size: {A[0][0].shape[0]}")
        print(f"  Action space size: {B[0][0].shape[2]}")

        # Verify we got 2 agents
        assert len(agents) == 2, "Should create 2 agents"
        assert len(A) == 2, "Should have 2 A matrices"
        assert len(B) == 2, "Should have 2 B matrices"
        assert len(C) == 2, "Should have 2 C matrices"
        assert len(D) == 2, "Should have 2 D matrices"

        # Verify agents have policies
        assert len(agents[0].policies) > 0, "Agents should have policies"
        assert agents[0].policy_len == config.horizon, "Policy length should match horizon"

    def test_A_matrix_structure(self):
        """Test observation model (A matrix) is correctly structured."""
        env = LavaCorridorEnv(LavaCorridorConfig(width=4, height=2))
        agents, A, B, C, D = create_tom_agents(env, num_agents=1)

        A_matrix = A[0][0]  # First agent, first (and only) modality

        print(f"\nA matrix structure:")
        print(f"  Shape: {A_matrix.shape}")
        print(f"  Expected: ({env.config.height * env.config.width}, {env.config.height * env.config.width})")

        num_states = env.config.height * env.config.width

        # A should be [num_obs x num_states]
        assert A_matrix.shape[0] == num_states, "Rows should be number of observations"
        assert A_matrix.shape[1] == num_states, "Cols should be number of states"

        # Each column should sum to 1 (valid probability distribution)
        col_sums = A_matrix.sum(axis=0)
        assert np.allclose(col_sums, 1.0), "Each column should sum to 1"

        # For lava corridor, observations = states (identity), so A should be identity
        assert np.allclose(A_matrix, np.eye(num_states)), \
            "Lava corridor uses identity observation model"

    def test_B_matrix_structure(self):
        """Test transition model (B matrix) respects environment dynamics."""
        env = LavaCorridorEnv(LavaCorridorConfig(width=4, height=2))
        agents, A, B, C, D = create_tom_agents(env, num_agents=1)

        B_matrix = B[0][0]  # First agent, first state factor

        print(f"\nB matrix structure:")
        print(f"  Shape: {B_matrix.shape}")

        num_states = env.config.height * env.config.width
        num_actions = 5  # UP, DOWN, LEFT, RIGHT, STAY

        # B should be [num_states x num_states x num_actions]
        assert B_matrix.shape == (num_states, num_states, num_actions), \
            f"B should be [{num_states}, {num_states}, {num_actions}]"

        # Each transition matrix B[:, :, a] should have columns summing to 1
        for a in range(num_actions):
            col_sums = B_matrix[:, :, a].sum(axis=0)
            assert np.allclose(col_sums, 1.0), \
                f"B[:,:,{a}] columns should sum to 1 (valid transitions)"

    def test_B_matrix_dynamics(self):
        """Test that B matrix encodes correct movement dynamics."""
        env = LavaCorridorEnv(LavaCorridorConfig(width=4, height=2))
        agents, A, B, C, D = create_tom_agents(env, num_agents=1)

        B_matrix = B[0][0]

        # Test a specific transition: STAY action should keep agent in same state
        STAY = 4
        for s in range(B_matrix.shape[0]):
            # B[s, s, STAY] should be 1.0 (deterministic stay)
            assert B_matrix[s, s, STAY] == 1.0, \
                f"STAY action should keep agent in state {s}"

        # Test RIGHT action moves agent right (when not at wall)
        RIGHT = 3
        # State at (0, 0) should move to (0, 1) with RIGHT
        s_00 = env.pos_to_obs_index((0, 0))
        s_01 = env.pos_to_obs_index((0, 1))

        print(f"\nB dynamics test:")
        print(f"  State (0,0) -> (0,1) with RIGHT:")
        print(f"    B[{s_01}, {s_00}, RIGHT] = {B_matrix[s_01, s_00, RIGHT]}")

        assert B_matrix[s_01, s_00, RIGHT] > 0.9, \
            "RIGHT action should move agent right"

    def test_C_matrix_preferences(self):
        """Test preference vector (C) reflects goal and lava."""
        env = LavaCorridorEnv(LavaCorridorConfig(width=5, height=3))
        agents, A, B, C, D = create_tom_agents(env, num_agents=1)

        C_vec = C[0][0]  # First agent, first modality

        print(f"\nC vector (preferences):")
        print(f"  Shape: {C_vec.shape}")
        print(f"  Min: {C_vec.min():.2f}, Max: {C_vec.max():.2f}")

        num_states = env.config.height * env.config.width

        assert len(C_vec) == num_states, "C should have one entry per observation"

        # Goal should have highest preference
        goal_obs = env.pos_to_obs_index(env.goal)
        print(f"  Goal obs {goal_obs}: C = {C_vec[goal_obs]:.2f}")

        assert C_vec[goal_obs] > 0, "Goal should have positive preference"

        # Lava should have low/negative preference
        if env.config.lava_positions:
            lava_pos = env.config.lava_positions[0]
            lava_obs = env.pos_to_obs_index(lava_pos)
            print(f"  Lava obs {lava_obs}: C = {C_vec[lava_obs]:.2f}")

            assert C_vec[lava_obs] < C_vec[goal_obs], \
                "Lava should have lower preference than goal"

    def test_D_matrix_initial_state(self):
        """Test initial state prior (D) is valid probability distribution."""
        env = LavaCorridorEnv(LavaCorridorConfig(width=4, height=2))
        agents, A, B, C, D = create_tom_agents(env, num_agents=1)

        D_vec = D[0][0]  # First agent, first state factor

        print(f"\nD vector (initial state prior):")
        print(f"  Shape: {D_vec.shape}")
        print(f"  Sum: {D_vec.sum():.4f}")
        print(f"  Max: {D_vec.max():.4f}")

        # D should be valid probability distribution
        assert np.allclose(D_vec.sum(), 1.0), "D should sum to 1"
        assert np.all(D_vec >= 0), "D should be non-negative"

        # Check that initial positions have non-zero probability
        for start_pos in env.start_positions:
            start_obs = env.pos_to_obs_index(start_pos)
            print(f"  Start pos {start_pos} (obs {start_obs}): D = {D_vec[start_obs]:.4f}")
            assert D_vec[start_obs] > 0, \
                f"Start position {start_pos} should have non-zero prior"


class TestPolicyLibrary:
    """Test policy library generation."""

    def test_policy_enumeration(self):
        """Test that all policies are enumerated correctly."""
        num_actions = 3
        horizon = 2

        policies = create_policy_library(num_actions, horizon)

        print(f"\nPolicy enumeration:")
        print(f"  Actions: {num_actions}, Horizon: {horizon}")
        print(f"  Number of policies: {len(policies)}")
        print(f"  Expected: {num_actions ** horizon}")

        # Should have num_actions^horizon policies
        expected_count = num_actions ** horizon
        assert len(policies) == expected_count, \
            f"Should have {expected_count} policies"

        # Each policy should have correct length
        for p in policies:
            assert len(p) == horizon, "Each policy should have length equal to horizon"

        # All actions should be valid
        for p in policies:
            for a in p:
                assert 0 <= a < num_actions, "Action should be in valid range"

    def test_policy_uniqueness(self):
        """Test that all policies are unique."""
        policies = create_policy_library(num_actions=2, horizon=3)

        # Convert to tuples for set comparison
        policy_tuples = [tuple(p) for p in policies]
        unique_policies = set(policy_tuples)

        assert len(unique_policies) == len(policies), \
            "All policies should be unique"

    def test_zero_horizon(self):
        """Test edge case of zero horizon."""
        policies = create_policy_library(num_actions=5, horizon=0)

        assert len(policies) == 1, "Zero horizon should return single empty policy"
        assert policies[0] == [], "Zero horizon policy should be empty list"


class TestSharedOutcomes:
    """Test shared outcome extraction."""

    def test_shared_outcomes_excludes_lava(self):
        """Test that lava positions are not in shared outcomes."""
        config = LavaCorridorConfig(
            width=5,
            height=3,
            lava_positions=[(1, 2), (1, 3)]
        )
        env = LavaCorridorEnv(config)

        shared = get_shared_outcomes(env)

        print(f"\nShared outcomes test:")
        print(f"  Total outcomes: {len(shared)}")
        print(f"  Lava positions: {config.lava_positions}")

        # Lava observations should not be in shared outcomes
        for lava_pos in config.lava_positions:
            lava_obs = env.pos_to_obs_index(lava_pos)
            assert lava_obs not in shared, \
                f"Lava obs {lava_obs} should not be in shared outcomes"

    def test_shared_outcomes_includes_safe_states(self):
        """Test that safe (non-lava) states are in shared outcomes."""
        config = LavaCorridorConfig(width=4, height=2)
        env = LavaCorridorEnv(config)

        shared = get_shared_outcomes(env)

        print(f"\nShared outcomes includes safe states:")
        print(f"  Shared outcomes: {len(shared)}")

        # Goal should be in shared outcomes
        goal_obs = env.pos_to_obs_index(env.goal)
        assert goal_obs in shared, "Goal should be in shared outcomes"

        # Start positions should be in shared outcomes
        for start_pos in env.start_positions:
            start_obs = env.pos_to_obs_index(start_pos)
            assert start_obs in shared, \
                f"Start position {start_pos} should be in shared outcomes"


class TestMockAgents:
    """Test mock agent creation for unit tests."""

    def test_mock_agents_creation(self):
        """Test that mock agents are created correctly."""
        agents, A, B = create_mock_agents_for_testing(
            num_states=5,
            num_obs=5,
            num_actions=3,
            horizon=2
        )

        print(f"\nMock agents:")
        print(f"  Number of agents: {len(agents)}")
        print(f"  Policies: {len(agents[0].policies)}")

        assert len(agents) == 2, "Should create 2 mock agents"
        assert len(agents[0].policies) == 3**2, "Should have 9 policies (3 actions, horizon 2)"

    def test_mock_agents_have_valid_models(self):
        """Test that mock agents have valid generative models."""
        agents, A, B = create_mock_agents_for_testing(num_states=10, num_obs=10)

        # A should be [num_obs x num_states]
        assert A[0].shape == (10, 10), "A should be 10x10"

        # B should be [num_states x num_states x num_actions]
        assert B[0].shape[0] == 10, "B should have 10 states"
        assert B[0].shape[1] == 10, "B should have 10 states"

        # A and B should be valid probability distributions
        assert np.allclose(A[0].sum(axis=0), 1.0), "A columns should sum to 1"
        for a in range(B[0].shape[2]):
            assert np.allclose(B[0][:, :, a].sum(axis=0), 1.0), \
                f"B[:,:,{a}] columns should sum to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
