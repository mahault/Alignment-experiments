"""
Unit tests for F-aware policy prior.

Verifies that:
1. κ=0 recovers the baseline (standard softmax over EFE)
2. κ>0 biases toward high-F policies
3. β correctly weights joint vs individual flexibility
"""

import pytest
import numpy as np
from typing import List

from src.metrics.path_flexibility import (
    compute_F_arrays_for_policies,
    compute_q_pi_with_F_prior,
)


def create_mock_F_arrays(num_policies: int, high_F_idx: int = 0):
    """
    Create mock flexibility arrays for testing.

    Parameters
    ----------
    num_policies : int
        Number of policies
    high_F_idx : int
        Index of the policy with high flexibility

    Returns
    -------
    F_i, F_j, F_joint : np.ndarray
        Flexibility arrays
    """
    # Create baseline flexibility values
    F_i = np.ones(num_policies) * 2.0  # Base flexibility
    F_j = np.ones(num_policies) * 2.0

    # Make one policy have high flexibility
    F_i[high_F_idx] = 10.0
    F_j[high_F_idx] = 10.0

    # Joint flexibility
    F_joint = F_i + F_j

    return F_i, F_j, F_joint


class TestFAwarePriorBaseline:
    """Test that κ=0 recovers baseline behavior."""

    def test_kappa_zero_equals_baseline(self):
        """With κ=0, F-aware prior should match standard softmax(-γG)."""
        num_policies = 5
        gamma = 16.0

        # Create mock EFE (Expected Free Energy)
        # Lower EFE = better policy
        G_i = np.array([10.0, 8.0, 12.0, 9.0, 11.0])  # Policy 1 has lowest EFE
        G_j = np.array([10.0, 10.0, 10.0, 10.0, 10.0])  # Other agent has uniform EFE

        # Create mock flexibility (doesn't matter if κ=0)
        F_i, F_j, F_joint = create_mock_F_arrays(num_policies, high_F_idx=3)

        # Standard baseline: q(π) ∝ exp(-γ * G) with α=0 (no empathy)
        q_baseline = np.exp(-gamma * G_i)
        q_baseline = q_baseline / q_baseline.sum()

        # F-aware prior with κ=0, α=0
        q_F_aware = compute_q_pi_with_F_prior(
            G_i=G_i,
            G_j=G_j,
            F_i=F_i,
            F_j=F_j,
            gamma=gamma,
            alpha=0.0,  # No empathy
            kappa=0.0,  # κ=0 should recover baseline
            beta=0.5,   # Doesn't matter when κ=0
        )

        print(f"\nκ=0 test:")
        print(f"  Baseline q(π): {q_baseline}")
        print(f"  F-aware q(π):  {q_F_aware}")
        print(f"  Max diff: {np.max(np.abs(q_baseline - q_F_aware)):.6f}")

        # Should be identical (within numerical precision)
        assert np.allclose(q_baseline, q_F_aware, atol=1e-10), \
            "κ=0 should recover standard softmax(-γG)"

        # Policy 1 should be most probable (lowest EFE)
        assert np.argmax(q_F_aware) == 1, "Policy with lowest EFE should be most probable"


class TestFAwarePriorBiasing:
    """Test that κ>0 biases toward high-F policies."""

    def test_kappa_positive_biases_toward_high_F(self):
        """With κ>0, high-F policies should have higher probability."""
        num_policies = 5
        gamma = 16.0

        # All policies have equal EFE
        G_i = np.ones(num_policies) * 10.0
        G_j = np.ones(num_policies) * 10.0

        # Policy 3 has much higher flexibility
        F_i, F_j, F_joint = create_mock_F_arrays(num_policies, high_F_idx=3)

        # Baseline: equal probabilities (equal EFE)
        q_baseline = compute_q_pi_with_F_prior(
            G_i=G_i, G_j=G_j, F_i=F_i, F_j=F_j,
            gamma=gamma, alpha=0.0, kappa=0.0, beta=0.5
        )

        # F-aware: should bias toward policy 3
        q_F_aware = compute_q_pi_with_F_prior(
            G_i=G_i, G_j=G_j, F_i=F_i, F_j=F_j,
            gamma=gamma, alpha=0.0, kappa=1.0, beta=0.5
        )

        print(f"\nκ>0 biasing test:")
        print(f"  EFE: {G_i}")
        print(f"  F_i: {F_i}")
        print(f"  Baseline q(π): {q_baseline}")
        print(f"  F-aware q(π):  {q_F_aware}")
        print(f"  High-F policy (idx 3): baseline={q_baseline[3]:.4f}, F-aware={q_F_aware[3]:.4f}")

        # Baseline should be roughly uniform
        assert np.allclose(q_baseline, 1/num_policies, atol=0.01), \
            "With equal EFE and κ=0, should have uniform distribution"

        # F-aware should strongly prefer policy 3
        assert q_F_aware[3] > q_baseline[3], \
            "High-F policy should have higher probability with κ>0"
        assert q_F_aware[3] > 0.5, \
            "High-F policy should dominate when κ>0 and equal EFE"

    def test_kappa_strength(self):
        """Larger κ should more strongly bias toward high-F policies."""
        num_policies = 5
        gamma = 16.0

        G_i = np.ones(num_policies) * 10.0
        G_j = np.ones(num_policies) * 10.0
        F_i, F_j, F_joint = create_mock_F_arrays(num_policies, high_F_idx=2)

        # Test increasing κ
        kappas = [0.0, 0.5, 1.0, 2.0]
        probs_at_high_F = []

        for kappa in kappas:
            q = compute_q_pi_with_F_prior(
                G_i=G_i, G_j=G_j, F_i=F_i, F_j=F_j,
                gamma=gamma, alpha=0.0, kappa=kappa, beta=0.5
            )
            probs_at_high_F.append(q[2])  # Policy 2 has high F

        print(f"\nκ strength test:")
        print(f"  κ values: {kappas}")
        print(f"  P(high-F policy): {probs_at_high_F}")

        # Should be monotonically increasing
        for i in range(len(kappas) - 1):
            assert probs_at_high_F[i] < probs_at_high_F[i+1], \
                f"Larger κ should increase P(high-F policy): κ={kappas[i]} vs {kappas[i+1]}"


class TestJointFlexibilityWeighting:
    """Test that β correctly weights joint vs individual flexibility."""

    def test_beta_weighting(self):
        """β should control tradeoff between individual and joint flexibility."""
        num_policies = 3
        gamma = 16.0
        kappa = 1.0

        G_i = np.ones(num_policies) * 10.0
        G_j = np.ones(num_policies) * 10.0

        # Policy 0: high individual F_i, low F_j
        # Policy 1: medium both
        # Policy 2: low individual, high F_j
        F_i = np.array([10.0, 5.0, 2.0])
        F_j = np.array([2.0, 5.0, 10.0])
        # F_joint = F_i + beta * F_j will depend on beta

        # β=0: only care about individual flexibility
        q_individual = compute_q_pi_with_F_prior(
            G_i=G_i, G_j=G_j, F_i=F_i, F_j=F_j,
            gamma=gamma, alpha=0.0, kappa=kappa, beta=0.0
        )

        # β=1: care about both F_i and F_j equally
        q_joint = compute_q_pi_with_F_prior(
            G_i=G_i, G_j=G_j, F_i=F_i, F_j=F_j,
            gamma=gamma, alpha=0.0, kappa=kappa, beta=1.0
        )

        # β=0.5: balanced
        q_balanced = compute_q_pi_with_F_prior(
            G_i=G_i, G_j=G_j, F_i=F_i, F_j=F_j,
            gamma=gamma, alpha=0.0, kappa=kappa, beta=0.5
        )

        print(f"\nβ weighting test:")
        print(f"  F_i:      {F_i}")
        print(f"  F_j:      {F_j}")
        print(f"  β=0 (individual): {q_individual}")
        print(f"  β=1 (weighted):   {q_joint}")
        print(f"  β=0.5 (balanced): {q_balanced}")

        # β=0: should prefer policy 0 (high individual F_i)
        assert np.argmax(q_individual) == 0, \
            "β=0 should prefer policy with high individual F_i"

        # β=1: should prefer policy 2 (high F_j)
        # F_joint = F_i + 1.0 * F_j = [12, 10, 12] - policy 0 and 2 tied
        assert np.argmax(q_joint) in [0, 2], \
            "β=1 should prefer policy with high combined F"

        # β=0.5: should be more balanced
        # Check that it's not strongly dominated by any single policy
        assert q_balanced.max() < 0.8, \
            "β=0.5 should balance individual and other's flexibility"


class TestEFEAndFlexibilityTradeoff:
    """Test interaction between EFE and flexibility."""

    def test_EFE_dominates_when_kappa_small(self):
        """With small κ, low-EFE policies should dominate even if F is lower."""
        num_policies = 3
        gamma = 16.0

        # Policy 0: very low EFE, low F (optimal but inflexible)
        # Policy 1: medium EFE, medium F
        # Policy 2: high EFE, high F (flexible but not optimal)
        G_i = np.array([5.0, 10.0, 15.0])
        G_j = np.array([10.0, 10.0, 10.0])
        F_i, F_j, F_joint = create_mock_F_arrays(num_policies, high_F_idx=2)

        # Small κ: EFE should dominate
        q_small_kappa = compute_q_pi_with_F_prior(
            G_i=G_i, G_j=G_j, F_i=F_i, F_j=F_j,
            gamma=gamma, alpha=0.0, kappa=0.1, beta=0.5
        )

        print(f"\nSmall κ test:")
        print(f"  G_i: {G_i}")
        print(f"  F_i: {F_i}")
        print(f"  q(π) with κ=0.1: {q_small_kappa}")

        # Policy 0 (lowest EFE) should still dominate
        assert np.argmax(q_small_kappa) == 0, \
            "With small κ, low-EFE policy should dominate"

    def test_flexibility_can_overcome_EFE_when_kappa_large(self):
        """With large κ, high-F policy can be preferred even with higher EFE."""
        num_policies = 3
        gamma = 16.0

        # Small EFE difference, large F difference
        G_i = np.array([10.0, 11.0, 12.0])  # Small differences
        G_j = np.array([10.0, 10.0, 10.0])
        F_i = np.array([2.0, 2.0, 20.0])   # Large F for policy 2
        F_j = np.array([2.0, 2.0, 20.0])

        # Large κ: flexibility should become important
        q_large_kappa = compute_q_pi_with_F_prior(
            G_i=G_i, G_j=G_j, F_i=F_i, F_j=F_j,
            gamma=gamma, alpha=0.0, kappa=2.0, beta=0.5
        )

        print(f"\nLarge κ test:")
        print(f"  G_i: {G_i}")
        print(f"  F_i: {F_i}")
        print(f"  q(π) with κ=2.0: {q_large_kappa}")

        # Policy 2 (high F) should win despite higher EFE
        assert np.argmax(q_large_kappa) == 2, \
            "With large κ and small EFE differences, high-F policy should win"


class TestNumericalStability:
    """Test numerical stability of F-aware prior."""

    def test_large_gamma_values(self):
        """Should handle large γ values without overflow."""
        num_policies = 5
        gamma = 100.0  # Very large

        G_i = np.array([5.0, 10.0, 8.0, 12.0, 9.0])
        G_j = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        F_i, F_j, F_joint = create_mock_F_arrays(num_policies)

        q = compute_q_pi_with_F_prior(
            G_i=G_i, G_j=G_j, F_i=F_i, F_j=F_j,
            gamma=gamma, alpha=0.0, kappa=1.0, beta=0.5
        )

        print(f"\nLarge γ test (γ={gamma}):")
        print(f"  q(π): {q}")

        # Should be valid probability distribution
        assert np.all(np.isfinite(q)), "Should not have inf/nan"
        assert np.all(q >= 0), "Should be non-negative"
        assert np.isclose(q.sum(), 1.0), "Should sum to 1"

    def test_extreme_flexibility_differences(self):
        """Should handle extreme flexibility differences."""
        num_policies = 3
        gamma = 16.0

        G_i = np.ones(num_policies) * 10.0
        G_j = np.ones(num_policies) * 10.0
        F_i = np.array([1e-6, 5.0, 1000.0])  # Extreme range
        F_j = np.array([1e-6, 5.0, 1000.0])

        q = compute_q_pi_with_F_prior(
            G_i=G_i, G_j=G_j, F_i=F_i, F_j=F_j,
            gamma=gamma, alpha=0.0, kappa=1.0, beta=0.5
        )

        print(f"\nExtreme F test:")
        print(f"  F_i: {F_i}")
        print(f"  q(π): {q}")

        assert np.all(np.isfinite(q)), "Should handle extreme F values"
        assert np.isclose(q.sum(), 1.0), "Should still sum to 1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
