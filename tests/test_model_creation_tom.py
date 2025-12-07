"""
Unit tests for TOM-style model and agent creation.

Verifies that LavaModel and LavaAgent are created correctly with proper
A, B, C, D structure (dicts, not lists).
"""

import os
import sys

# Ensure repo root is on sys.path so `tom` can be imported
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import numpy as np
import jax.numpy as jnp

from tom.models import LavaModel, LavaAgent


class TestLavaModelCreation:
    """Test LavaModel creation and structure."""

    def test_model_basic_creation(self):
        """Test basic model creation with default parameters."""
        model = LavaModel(width=5, height=3)

        assert model.width == 5
        assert model.height == 3
        assert model.goal_x == 4  # Default: rightmost
        assert model.safe_y == 1  # Middle row

        print(f"\nBasic model creation:")
        print(f"  Width: {model.width}, Height: {model.height}")
        print(f"  Goal: ({model.goal_x}, {model.safe_y})")
        print(f"  Num states: {model.num_states}")

    def test_model_custom_goal(self):
        """Test model creation with custom goal position."""
        model = LavaModel(width=6, height=3, goal_x=2)

        assert model.goal_x == 2, "Should use custom goal_x"

        print(f"\nCustom goal model:")
        print(f"  Goal x: {model.goal_x}")

    def test_model_dict_structure(self):
        """Test that A, B, C, D are dicts (not lists)."""
        model = LavaModel(width=4, height=3)

        # All should be dicts
        assert isinstance(model.A, dict), "A should be dict"
        assert isinstance(model.B, dict), "B should be dict"
        assert isinstance(model.C, dict), "C should be dict"
        assert isinstance(model.D, dict), "D should be dict"

        # Check keys
        assert "location_obs" in model.A
        assert "location_state" in model.B
        assert "location_obs" in model.C
        assert "location_state" in model.D

        print(f"\nDict structure:")
        print(f"  A keys: {list(model.A.keys())}")
        print(f"  B keys: {list(model.B.keys())}")
        print(f"  C keys: {list(model.C.keys())}")
        print(f"  D keys: {list(model.D.keys())}")


class TestMatrixShapes:
    """Test that all matrices have correct shapes."""

    def test_A_matrix_shape(self):
        """Test observation matrix shape."""
        model = LavaModel(width=4, height=3)
        A = model.A["location_obs"]

        num_states = model.num_states
        expected_shape = (num_states, num_states)

        assert A.shape == expected_shape, f"A should be {expected_shape}"
        print(f"\nA matrix: {A.shape}")

    def test_B_matrix_shape(self):
        """Test transition matrix shape."""
        model = LavaModel(width=4, height=3)
        B = model.B["location_state"]

        num_states = model.num_states
        num_actions = 5  # UP, DOWN, LEFT, RIGHT, STAY
        expected_shape = (num_states, num_states, num_actions)

        assert B.shape == expected_shape, f"B should be {expected_shape}"
        print(f"\nB matrix: {B.shape}")

    def test_C_vector_shape(self):
        """Test preference vector shape."""
        model = LavaModel(width=4, height=3)
        C = model.C["location_obs"]

        num_obs = model.num_obs
        expected_shape = (num_obs,)

        assert C.shape == expected_shape, f"C should be {expected_shape}"
        print(f"\nC vector: {C.shape}")

    def test_D_vector_shape(self):
        """Test initial state prior shape."""
        model = LavaModel(width=4, height=3)
        D = model.D["location_state"]

        num_states = model.num_states
        expected_shape = (num_states,)

        assert D.shape == expected_shape, f"D should be {expected_shape}"
        print(f"\nD vector: {D.shape}")


class TestMatrixProperties:
    """Test mathematical properties of matrices."""

    def test_A_is_identity(self):
        """Test that A is identity (fully observable)."""
        model = LavaModel(width=4, height=3)
        A = np.asarray(model.A["location_obs"])

        # Should be identity matrix
        expected = np.eye(model.num_states)
        assert np.allclose(A, expected), "A should be identity (fully observable)"

        print(f"\nA is identity: {np.allclose(A, expected)}")

    def test_B_valid_transitions(self):
        """Test that B encodes valid probability transitions."""
        model = LavaModel(width=4, height=3)
        B = np.asarray(model.B["location_state"])

        num_actions = 5

        # Each B[:, :, a] should be a valid stochastic matrix
        # (columns sum to 1)
        for a in range(num_actions):
            col_sums = B[:, :, a].sum(axis=0)
            assert np.allclose(col_sums, 1.0), \
                f"B[:,:,{a}] columns should sum to 1"

        print(f"\nB transition validity: all columns sum to 1.0")

    def test_C_goal_positive_lava_negative(self):
        """Test that goal has positive preference, lava negative."""
        model = LavaModel(width=5, height=3, goal_x=4)
        C = np.asarray(model.C["location_obs"])

        # Goal state
        goal_idx = model.safe_y * model.width + model.goal_x
        goal_pref = C[goal_idx]

        # Lava state (y != safe_y)
        lava_idx = 0  # y=0, x=0 is lava
        lava_pref = C[lava_idx]

        assert goal_pref > 0, "Goal should have positive preference"
        assert lava_pref < 0, "Lava should have negative preference"
        assert goal_pref > lava_pref, "Goal > Lava"

        print(f"\nPreferences:")
        print(f"  Goal: {goal_pref:.2f}")
        print(f"  Lava: {lava_pref:.2f}")

    def test_D_is_probability_distribution(self):
        """Test that D is valid probability distribution."""
        model = LavaModel(width=4, height=3)
        D = np.asarray(model.D["location_state"])

        # Should sum to 1
        assert np.isclose(D.sum(), 1.0), "D should sum to 1"

        # Should be non-negative
        assert np.all(D >= 0), "D should be non-negative"

        # Start position should have prob 1
        start_idx = model.safe_y * model.width + 0
        assert np.isclose(D[start_idx], 1.0), "Should start at (0, safe_y)"

        print(f"\nD distribution:")
        print(f"  Sum: {D.sum():.4f}")
        print(f"  Start prob: {D[start_idx]:.4f}")


class TestTransitionDynamics:
    """Test specific transition dynamics."""

    def test_stay_action_deterministic(self):
        """Test that STAY action keeps agent in same state."""
        model = LavaModel(width=4, height=3)
        B = np.asarray(model.B["location_state"])

        STAY = 4

        for s in range(model.num_states):
            prob_stay = B[s, s, STAY]
            assert np.isclose(prob_stay, 1.0), \
                f"STAY at state {s} should be deterministic (p=1.0)"

        print(f"\nSTAY action: deterministic for all states")

    def test_right_action_moves_right(self):
        """Test that RIGHT action moves agent right (when not at wall)."""
        model = LavaModel(width=4, height=3)
        B = np.asarray(model.B["location_state"])

        RIGHT = 3

        # State (y=1, x=0) should move to (y=1, x=1) with RIGHT
        s_from = 1 * model.width + 0  # (1, 0)
        s_to = 1 * model.width + 1    # (1, 1)

        prob_right = B[s_to, s_from, RIGHT]
        assert np.isclose(prob_right, 1.0), \
            f"RIGHT from (1,0) should go to (1,1) with p=1.0"

        print(f"\nRIGHT action: moves agent right correctly")

    def test_up_action_moves_up(self):
        """Test that UP action moves agent up (decreases y)."""
        model = LavaModel(width=4, height=3)
        B = np.asarray(model.B["location_state"])

        UP = 0

        # State (y=2, x=1) should move to (y=1, x=1) with UP
        s_from = 2 * model.width + 1  # (2, 1)
        s_to = 1 * model.width + 1    # (1, 1)

        prob_up = B[s_to, s_from, UP]
        assert np.isclose(prob_up, 1.0), \
            f"UP from (2,1) should go to (1,1)"

        print(f"\nUP action: moves agent up correctly")

    def test_boundary_handling(self):
        """Test that actions at boundaries stay in place."""
        model = LavaModel(width=4, height=3)
        B = np.asarray(model.B["location_state"])

        RIGHT = 3

        # At right boundary (y=1, x=3), RIGHT should stay
        s_boundary = 1 * model.width + 3  # (1, 3)

        prob_stay = B[s_boundary, s_boundary, RIGHT]
        assert np.isclose(prob_stay, 1.0), \
            "RIGHT at right boundary should stay in place"

        print(f"\nBoundary handling: agent stays at walls")


class TestLavaAgentCreation:
    """Test LavaAgent creation."""

    def test_agent_basic_creation(self):
        """Test basic agent creation."""
        model = LavaModel(width=4, height=3)
        agent = LavaAgent(model, horizon=2, gamma=8.0)

        assert agent.model is model
        assert agent.horizon == 2
        assert agent.gamma == 8.0

        print(f"\nAgent creation:")
        print(f"  Horizon: {agent.horizon}")
        print(f"  Gamma: {agent.gamma}")

    def test_agent_exposes_model_dicts(self):
        """Test that agent exposes model's A, B, C, D as dicts."""
        model = LavaModel(width=4, height=3)
        agent = LavaAgent(model, horizon=1, gamma=8.0)

        # Should expose same dicts
        assert agent.A is model.A, "Agent.A should be model.A"
        assert agent.B is model.B, "Agent.B should be model.B"
        assert agent.C is model.C, "Agent.C should be model.C"
        assert agent.D is model.D, "Agent.D should be model.D"

        # Should still be dicts
        assert isinstance(agent.A, dict)
        assert isinstance(agent.B, dict)
        assert isinstance(agent.C, dict)
        assert isinstance(agent.D, dict)

        print(f"\nAgent exposes model dicts:")
        print(f"  A is dict: {isinstance(agent.A, dict)}")
        print(f"  B is dict: {isinstance(agent.B, dict)}")

    def test_agent_policy_shape(self):
        """Test that agent has correct policy shape."""
        model = LavaModel(width=4, height=3)
        agent = LavaAgent(model, horizon=1, gamma=8.0)

        # Policies should be (num_policies, horizon, num_state_factors)
        # For lava: (5, 1, 1) - 5 actions, horizon=1, 1 state factor
        expected_shape = (5, 1, 1)
        assert agent.policies.shape == expected_shape, \
            f"Policies should be {expected_shape}"

        print(f"\nAgent policies:")
        print(f"  Shape: {agent.policies.shape}")
        print(f"  Num policies: {len(agent.policies)}")

    def test_agent_policies_are_valid_actions(self):
        """Test that all policies contain valid actions."""
        model = LavaModel(width=4, height=3)
        agent = LavaAgent(model, horizon=1, gamma=8.0)

        # All policy actions should be in range [0, 4]
        policies = np.asarray(agent.policies)
        assert np.all(policies >= 0), "Actions should be >= 0"
        assert np.all(policies < 5), "Actions should be < 5"

        print(f"\nPolicy actions:")
        print(f"  Min: {policies.min()}")
        print(f"  Max: {policies.max()}")
        print(f"  Unique actions: {np.unique(policies[:, 0, 0])}")


class TestModelConsistency:
    """Test consistency across different model sizes."""

    def test_small_model(self):
        """Test 3x3 grid."""
        model = LavaModel(width=3, height=3)

        assert model.num_states == 9
        assert model.A["location_obs"].shape == (9, 9)
        assert model.B["location_state"].shape == (9, 9, 5)

        print(f"\nSmall model (3x3): {model.num_states} states")

    def test_large_model(self):
        """Test 10x3 grid."""
        model = LavaModel(width=10, height=3)

        assert model.num_states == 30
        assert model.A["location_obs"].shape == (30, 30)
        assert model.B["location_state"].shape == (30, 30, 5)

        print(f"\nLarge model (10x3): {model.num_states} states")

    def test_different_goals(self):
        """Test models with different goal positions."""
        model1 = LavaModel(width=5, height=3, goal_x=0)
        model2 = LavaModel(width=5, height=3, goal_x=4)

        C1 = np.asarray(model1.C["location_obs"])
        C2 = np.asarray(model2.C["location_obs"])

        # Goals should be at different positions
        goal1_idx = model1.safe_y * model1.width + 0
        goal2_idx = model2.safe_y * model2.width + 4

        assert C1[goal1_idx] > 0, "Goal at x=0 should be positive"
        assert C2[goal2_idx] > 0, "Goal at x=4 should be positive"

        print(f"\nDifferent goals:")
        print(f"  Model 1 goal at x=0: C={C1[goal1_idx]:.2f}")
        print(f"  Model 2 goal at x=4: C={C2[goal2_idx]:.2f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
