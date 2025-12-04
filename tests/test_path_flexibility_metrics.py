"""
Unit tests for path flexibility metrics (Empowerment, Returnability, Overlap).

These tests use toy generative models to verify that E, R, and O behave as expected
in known scenarios.
"""

import pytest
import numpy as np
from typing import Any, List
from dataclasses import dataclass

from src.metrics.path_flexibility import (
    compute_empowerment_along_path,
    compute_returnability,
    compute_overlap,
    compute_path_flexibility,
)


@dataclass
class ToyModel:
    """Simple generative model for testing."""
    A: List[np.ndarray]  # Observation model
    B: List[np.ndarray]  # Transition model
    policies: List[List[int]] = None  # Optional policy library


def create_simple_chain_model(num_states: int = 5, num_actions: int = 2):
    """
    Create a simple chain MDP for testing.

    States: 0 -> 1 -> 2 -> 3 -> 4
    Action 0: Move right (toward higher state)
    Action 1: Stay in place

    Observations = States (identity mapping)
    """
    # A matrix: Identity observation (obs = state)
    A = [np.eye(num_states)]

    # B matrix: Transition dynamics
    # B[state_to, state_from, action]
    B_right = np.zeros((num_states, num_states, num_actions))

    for action in range(num_actions):
        for s in range(num_states):
            if action == 0:  # Move right
                if s < num_states - 1:
                    B_right[s + 1, s, action] = 1.0  # Deterministic transition
                else:
                    B_right[s, s, action] = 1.0  # Stay at boundary
            else:  # Stay (action 1)
                B_right[s, s, action] = 1.0

    B = [B_right]

    return ToyModel(A=A, B=B)


def create_branching_model():
    """
    Create a model with branching paths (high empowerment).

    State 0 (start) can go to states 1, 2, or 3 depending on action.
    Action 0: go to state 1
    Action 1: go to state 2
    Action 2: go to state 3

    This should have HIGH empowerment at state 0.
    """
    num_states = 4
    num_actions = 3

    A = [np.eye(num_states)]

    B_branch = np.zeros((num_states, num_states, num_actions))

    # From state 0, can branch to 1, 2, or 3
    B_branch[1, 0, 0] = 1.0  # Action 0: 0 -> 1
    B_branch[2, 0, 1] = 1.0  # Action 1: 0 -> 2
    B_branch[3, 0, 2] = 1.0  # Action 2: 0 -> 3

    # From other states, stay in place
    for s in range(1, num_states):
        for a in range(num_actions):
            B_branch[s, s, a] = 1.0

    B = [B_branch]

    return ToyModel(A=A, B=B)


def create_dead_end_model():
    """
    Create a model with no empowerment (dead end).

    All actions lead to the same state (no choice).
    """
    num_states = 2
    num_actions = 3

    A = [np.eye(num_states)]

    B_dead = np.zeros((num_states, num_states, num_actions))

    # From state 0, all actions go to state 1
    for a in range(num_actions):
        B_dead[1, 0, a] = 1.0
        B_dead[1, 1, a] = 1.0  # State 1 stays at 1

    B = [B_dead]

    return ToyModel(A=A, B=B)


class TestEmpowerment:
    """Test empowerment computation."""

    def test_empowerment_branching_vs_dead_end(self):
        """Branching model should have higher empowerment than dead-end."""
        # Branching model (high empowerment)
        model_branch = create_branching_model()
        qs_branch = np.array([1.0, 0.0, 0.0, 0.0])  # Start at state 0
        policy_branch = [0]  # Doesn't matter, we care about initial state

        E_branch, _ = compute_empowerment_along_path(
            model_branch, policy_branch, horizon=1, current_qs=qs_branch
        )

        # Dead-end model (low empowerment)
        model_dead = create_dead_end_model()
        qs_dead = np.array([1.0, 0.0])
        policy_dead = [0]

        E_dead, _ = compute_empowerment_along_path(
            model_dead, policy_dead, horizon=1, current_qs=qs_dead
        )

        print(f"\nEmpowerment - Branching: {E_branch:.4f}, Dead-end: {E_dead:.4f}")

        # Branching should have higher empowerment
        assert E_branch > E_dead, f"Expected branching ({E_branch}) > dead-end ({E_dead})"
        assert E_branch > 0.5, "Branching model should have significant empowerment"
        assert E_dead < 0.1, "Dead-end model should have near-zero empowerment"

    def test_empowerment_decreases_along_chain(self):
        """Empowerment should decrease as we get closer to the end of a chain."""
        model = create_simple_chain_model(num_states=5)

        # Test from start (state 0) - has options ahead
        qs_start = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        policy = [0, 0, 0]  # Move right
        E_start, steps_start = compute_empowerment_along_path(
            model, policy, horizon=3, current_qs=qs_start
        )

        # Test from near end (state 3) - fewer options
        qs_end = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
        E_end, steps_end = compute_empowerment_along_path(
            model, policy, horizon=3, current_qs=qs_end
        )

        print(f"\nEmpowerment - Start: {E_start:.4f}, Near end: {E_end:.4f}")
        print(f"  Start steps: {steps_start}")
        print(f"  End steps: {steps_end}")

        # Later states should have less empowerment
        assert E_start >= E_end, "Empowerment should decrease toward dead-end"


class TestReturnability:
    """Test returnability computation."""

    def test_returnability_shared_outcomes(self):
        """Policy that stays in shared outcomes should have high returnability."""
        model = create_simple_chain_model(num_states=5)

        # Shared outcomes: states 0, 1, 2 (corridor)
        # Non-shared: states 3, 4 (lava)
        shared_outcomes = [0, 1, 2]

        # Start from state 2 (last safe state) so moving right enters "lava"
        qs_start = np.array([0.0, 0.0, 1.0, 0.0, 0.0])

        # Policy 1: Stay in place (safe)
        policy_safe = [1, 1, 1]  # Action 1 = Stay

        R_safe, steps_safe = compute_returnability(
            model, policy_safe, shared_outcomes, horizon=3, current_qs=qs_start
        )

        # Policy 2: Move into danger (states 3, 4)
        policy_danger = [0, 0, 0]  # Action 0 = Move right toward "lava"

        R_danger, steps_danger = compute_returnability(
            model, policy_danger, shared_outcomes, horizon=3, current_qs=qs_start
        )

        print(f"\nReturnability - Safe: {R_safe:.4f}, Danger: {R_danger:.4f}")
        print(f"  Safe steps: {steps_safe}")
        print(f"  Danger steps: {steps_danger}")

        # Staying in shared outcomes should have higher returnability
        assert R_safe > R_danger, f"Safe policy ({R_safe}) should have higher returnability than danger ({R_danger})"
        assert R_safe > 2.5, "Safe policy should stay in shared outcomes most of the time"

    def test_returnability_increases_with_horizon(self):
        """Longer horizons in shared outcomes should accumulate more returnability."""
        model = create_simple_chain_model(num_states=5)
        shared_outcomes = [0, 1, 2]
        qs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        policy_stay = [1, 1, 1, 1, 1]  # Stay in place

        R_short, _ = compute_returnability(
            model, policy_stay, shared_outcomes, horizon=2, current_qs=qs
        )

        R_long, _ = compute_returnability(
            model, policy_stay, shared_outcomes, horizon=5, current_qs=qs
        )

        print(f"\nReturnability - Short horizon: {R_short:.4f}, Long horizon: {R_long:.4f}")

        # Longer horizon should accumulate more returnability
        assert R_long > R_short, "Longer horizon should have higher total returnability"
        assert R_long > R_short * 2, "Should scale roughly with horizon"


class TestOverlap:
    """Test outcome overlap computation."""

    def test_overlap_identical_policies(self):
        """Two agents following identical policies should have high overlap."""
        model_i = create_simple_chain_model(num_states=5)
        model_j = create_simple_chain_model(num_states=5)

        qs_i = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        qs_j = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        # Same policy for both
        policy_same = [0, 0, 0]  # Both move right

        O_same, steps_same = compute_overlap(
            model_i, model_j, policy_same, policy_same,
            horizon=3, current_qs_i=qs_i, current_qs_j=qs_j
        )

        print(f"\nOverlap - Identical policies: {O_same:.4f}")
        print(f"  Steps: {steps_same}")

        # Identical policies should have high overlap (close to horizon)
        assert O_same > 2.5, "Identical policies should have high overlap"
        assert O_same <= 3.0, "Overlap should not exceed horizon"

    def test_overlap_diverging_policies(self):
        """Diverging policies should have lower overlap than identical ones."""
        model_i = create_simple_chain_model(num_states=5)
        model_j = create_simple_chain_model(num_states=5)

        qs_i = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        qs_j = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        # Same policy
        policy_same = [0, 0, 0]
        O_same, _ = compute_overlap(
            model_i, model_j, policy_same, policy_same,
            horizon=3, current_qs_i=qs_i, current_qs_j=qs_j
        )

        # Different policies
        policy_i = [0, 0, 0]  # Move right
        policy_j = [1, 1, 1]  # Stay in place
        O_diff, steps_diff = compute_overlap(
            model_i, model_j, policy_i, policy_j,
            horizon=3, current_qs_i=qs_i, current_qs_j=qs_j
        )

        print(f"\nOverlap - Same: {O_same:.4f}, Different: {O_diff:.4f}")
        print(f"  Different steps: {steps_diff}")

        # Diverging policies should have lower overlap
        assert O_same > O_diff, f"Same policies ({O_same}) should have higher overlap than different ({O_diff})"


class TestPathFlexibility:
    """Test combined path flexibility metric."""

    def test_flexibility_composition(self):
        """Test that F = λ_E·E + λ_R·R + λ_O·O."""
        model_i = create_simple_chain_model(num_states=5)
        model_j = create_simple_chain_model(num_states=5)

        qs_i = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        qs_j = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        policy_i = [0, 0]
        policy_j = [0, 0]

        shared_outcomes = [0, 1, 2]

        flex_i, flex_j = compute_path_flexibility(
            model_i, model_j, policy_i, policy_j,
            horizon=2, current_qs_i=qs_i, current_qs_j=qs_j,
            shared_outcome_set=shared_outcomes,
            lambda_E=1.0, lambda_R=1.0, lambda_O=1.0
        )

        # Check composition
        F_i_expected = flex_i.empowerment + flex_i.returnability + flex_i.overlap
        F_j_expected = flex_j.empowerment + flex_j.returnability + flex_j.overlap

        print(f"\nFlexibility composition check:")
        print(f"  Agent i: E={flex_i.empowerment:.4f}, R={flex_i.returnability:.4f}, O={flex_i.overlap:.4f}")
        print(f"  Expected F_i: {F_i_expected:.4f}, Actual: {flex_i.flexibility:.4f}")
        print(f"  Agent j: E={flex_j.empowerment:.4f}, R={flex_j.returnability:.4f}, O={flex_j.overlap:.4f}")
        print(f"  Expected F_j: {F_j_expected:.4f}, Actual: {flex_j.flexibility:.4f}")

        assert np.isclose(flex_i.flexibility, F_i_expected), "F_i should equal λ_E·E + λ_R·R + λ_O·O"
        assert np.isclose(flex_j.flexibility, F_j_expected), "F_j should equal λ_E·E + λ_R·R + λ_O·O"

    def test_flexibility_weights(self):
        """Test that flexibility weights work correctly."""
        model_i = create_simple_chain_model(num_states=5)
        model_j = create_simple_chain_model(num_states=5)

        qs_i = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        qs_j = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

        policy = [0, 0]
        shared_outcomes = [0, 1, 2]

        # Test with only empowerment
        flex_i_E, _ = compute_path_flexibility(
            model_i, model_j, policy, policy,
            horizon=2, current_qs_i=qs_i, current_qs_j=qs_j,
            shared_outcome_set=shared_outcomes,
            lambda_E=1.0, lambda_R=0.0, lambda_O=0.0
        )

        # Test with only returnability
        flex_i_R, _ = compute_path_flexibility(
            model_i, model_j, policy, policy,
            horizon=2, current_qs_i=qs_i, current_qs_j=qs_j,
            shared_outcome_set=shared_outcomes,
            lambda_E=0.0, lambda_R=1.0, lambda_O=0.0
        )

        print(f"\nFlexibility with different weights:")
        print(f"  Only E: F={flex_i_E.flexibility:.4f} (should equal E={flex_i_E.empowerment:.4f})")
        print(f"  Only R: F={flex_i_R.flexibility:.4f} (should equal R={flex_i_R.returnability:.4f})")

        assert np.isclose(flex_i_E.flexibility, flex_i_E.empowerment), "With λ_R=λ_O=0, F should equal E"
        assert np.isclose(flex_i_R.flexibility, flex_i_R.returnability), "With λ_E=λ_O=0, F should equal R"


class TestEdgeCases:
    """Test edge cases and numerical stability."""

    def test_single_state_model(self):
        """Model with single state should handle gracefully."""
        model = ToyModel(
            A=[np.array([[1.0]])],
            B=[np.array([[[1.0]]])]  # Single state, single action
        )

        qs = np.array([1.0])
        policy = [0]
        shared_outcomes = [0]

        # Should not crash
        E, _ = compute_empowerment_along_path(model, policy, horizon=1, current_qs=qs)
        R, _ = compute_returnability(model, policy, shared_outcomes, horizon=1, current_qs=qs)

        print(f"\nSingle state - E: {E:.4f}, R: {R:.4f}")

        # Single state has no empowerment
        assert E < 0.1, "Single state should have near-zero empowerment"
        # But should have full returnability if in shared outcome
        assert R > 0.9, "Should be in shared outcome"

    def test_zero_horizon(self):
        """Zero horizon should return empty metrics."""
        model = create_simple_chain_model()
        qs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        policy = []
        shared_outcomes = [0]

        E, steps_E = compute_empowerment_along_path(model, policy, horizon=0, current_qs=qs)
        R, steps_R = compute_returnability(model, policy, shared_outcomes, horizon=0, current_qs=qs)

        print(f"\nZero horizon - E: {E}, R: {R}")

        # Zero horizon means no steps, so empowerment is NaN (mean of empty array)
        # Returnability is sum, so it's 0
        assert np.isnan(E), "Zero horizon empowerment should be NaN (mean of empty array)"
        assert R == 0.0, "Zero horizon should have zero returnability (sum of empty)"
        assert len(steps_E) == 0, "Should have no steps"
        assert len(steps_R) == 0, "Should have no steps"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
