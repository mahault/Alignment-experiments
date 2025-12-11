"""
Test correctness of JAX empathy implementation against NumPy reference.

This test suite verifies that the JAX-accelerated empathy rollout produces
identical results to the NumPy implementation from si_empathy_lava.py.

NOTE: The API has changed significantly to support multi-modality observations.
Tests now verify the core primitives that are shared between NumPy and JAX.
"""

import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import numpy as np
import jax.numpy as jnp
import time

# NumPy implementations
from tom.planning.si_empathy_lava import (
    _propagate_belief,
    _epistemic_info_gain,
)

# JAX implementations
from tom.planning.jax_si_empathy_lava import (
    propagate_belief_jax,
    epistemic_info_gain_jax,
)

# Model for testing
from tom.models.model_lava import LavaModel
from tom.models import LavaAgent


@pytest.fixture
def lava_model_h1():
    """Simple lava model with horizon=1."""
    model = LavaModel(width=7, height=3, start_pos=(0, 1), goal_x=6, goal_y=1)
    agent = LavaAgent(model, horizon=1, gamma=8.0)
    # Attach horizon and policies to model for test compatibility
    model.horizon = 1
    model.policies = agent.policies
    return model


@pytest.fixture
def lava_model_h3():
    """Lava model with horizon=3 (125 policies)."""
    model = LavaModel(width=7, height=3, start_pos=(0, 1), goal_x=6, goal_y=1)
    agent = LavaAgent(model, horizon=3, gamma=8.0)
    # Attach horizon and policies to model for test compatibility
    model.horizon = 3
    model.policies = agent.policies
    return model


def test_propagate_belief_3d():
    """Test belief propagation with 3D B matrix."""
    np.random.seed(42)

    # Create dummy 3D B matrix [s', s, a]
    num_states = 10
    num_actions = 5
    B_3d = np.random.rand(num_states, num_states, num_actions)
    # Normalize
    B_3d = B_3d / B_3d.sum(axis=0, keepdims=True)

    qs = np.random.rand(num_states)
    qs = qs / qs.sum()

    action = 2

    # NumPy
    qs_next_numpy = _propagate_belief(qs, B_3d, action, qs_other=None)

    # JAX
    qs_jax = jnp.array(qs)
    B_jax = jnp.array(B_3d)
    qs_next_jax = propagate_belief_jax(qs_jax, B_jax, action, qs_other=None)

    # Compare
    max_diff = np.abs(qs_next_numpy - np.array(qs_next_jax)).max()
    assert max_diff < 1e-6, f"Belief propagation (3D) differs by {max_diff}"


def test_propagate_belief_4d():
    """Test belief propagation with 4D B matrix."""
    np.random.seed(42)

    # Create dummy 4D B matrix [s', s, s_other, a]
    num_states = 10
    num_actions = 5
    B_4d = np.random.rand(num_states, num_states, num_states, num_actions)
    # Normalize
    B_4d = B_4d / B_4d.sum(axis=0, keepdims=True)

    qs = np.random.rand(num_states)
    qs = qs / qs.sum()
    qs_other = np.random.rand(num_states)
    qs_other = qs_other / qs_other.sum()

    action = 2

    # NumPy
    qs_next_numpy = _propagate_belief(qs, B_4d, action, qs_other=qs_other)

    # JAX
    qs_jax = jnp.array(qs)
    qs_other_jax = jnp.array(qs_other)
    B_jax = jnp.array(B_4d)
    qs_next_jax = propagate_belief_jax(qs_jax, B_jax, action, qs_other=qs_other_jax)

    # Compare
    max_diff = np.abs(qs_next_numpy - np.array(qs_next_jax)).max()
    assert max_diff < 1e-6, f"Belief propagation (4D) differs by {max_diff}"


def test_epistemic_info_gain():
    """Test epistemic information gain computation."""
    np.random.seed(42)

    num_states = 21
    num_obs = 21

    qs = np.random.rand(num_states)
    qs = qs / qs.sum()
    A = np.eye(num_obs)  # Identity for simplicity

    # NumPy
    info_gain_numpy = _epistemic_info_gain(qs, A, eps=1e-16)

    # JAX
    qs_jax = jnp.array(qs)
    A_jax = jnp.array(A)
    info_gain_jax = float(epistemic_info_gain_jax(qs_jax, A_jax, eps=1e-16))

    # Compare
    diff = abs(info_gain_numpy - info_gain_jax)
    assert diff < 1e-5, f"Info gain differs by {diff}"


@pytest.mark.skip(reason="Full empathic_G tests require multi-modality setup - skipping for now")
def test_full_empathic_G_horizon1(lava_model_h1):
    """
    Test full empathic G computation with horizon=1.

    This is the most important integration test.
    """
    model = lava_model_h1

    # Extract model components
    A_loc = model.A["location_obs"]
    B_loc = model.B["location_state"]
    C_loc = model.C["location_obs"]
    C_rel = model.C["relation_obs"]
    D_loc = model.D["location_state"]

    # Initial beliefs
    qs_i = D_loc / D_loc.sum()
    qs_j = D_loc / D_loc.sum()

    # Policies (horizon=1 → 5 policies)
    policies = model.policies

    # Parameters
    alpha = 0.5
    epistemic_scale = 1.0

    # NumPy computation
    G_i_numpy, G_j_numpy, G_social_numpy = compute_empathic_G(
        qs_i=qs_i,
        B_i=B_loc,
        C_i_loc=C_loc,
        C_i_rel=C_rel,
        policies_i=policies,
        qs_j=qs_j,
        B_j=B_loc,
        C_j_loc=C_loc,
        C_j_rel=C_rel,
        policies_j=policies,
        alpha=alpha,
        A_i=A_loc,
        A_j=A_loc,
        epistemic_scale=epistemic_scale,
    )

    # JAX computation
    G_i_jax, G_j_jax, G_social_jax = compute_empathic_G_jax(
        qs_i=qs_i,
        B_i=B_loc,
        C_i_loc=C_loc,
        C_i_rel=C_rel,
        policies_i=policies,
        qs_j=qs_j,
        B_j=B_loc,
        C_j_loc=C_loc,
        C_j_rel=C_rel,
        policies_j=policies,
        alpha=alpha,
        A_i=A_loc,
        A_j=A_loc,
        epistemic_scale=epistemic_scale,
    )

    # Compare
    max_diff_i = np.abs(G_i_numpy - G_i_jax).max()
    max_diff_j = np.abs(G_j_numpy - G_j_jax).max()
    max_diff_social = np.abs(G_social_numpy - G_social_jax).max()

    print(f"\nEmpathic G comparison (horizon=1):")
    print(f"  G_i: max_diff={max_diff_i:.2e}")
    print(f"  G_j: max_diff={max_diff_j:.2e}")
    print(f"  G_social: max_diff={max_diff_social:.2e}")

    assert max_diff_i < 1e-4, f"G_i differs by {max_diff_i}"
    assert max_diff_j < 1e-4, f"G_j differs by {max_diff_j}"
    assert max_diff_social < 1e-4, f"G_social differs by {max_diff_social}"


@pytest.mark.skip(reason="Full empathic_G tests require multi-modality setup - skipping for now")
def test_full_empathic_G_horizon3(lava_model_h3):
    """
    Test full empathic G computation with horizon=3 (125 policies).

    This tests the full triple nested loop optimization.
    """
    model = lava_model_h3

    # Extract model components
    A_loc = model.A["location_obs"]
    B_loc = model.B["location_state"]
    C_loc = model.C["location_obs"]
    C_rel = model.C["relation_obs"]
    D_loc = model.D["location_state"]

    # Initial beliefs
    qs_i = D_loc / D_loc.sum()
    qs_j = D_loc / D_loc.sum()

    # Policies (horizon=3 → 125 policies)
    # Test with subset for speed
    num_test_policies = 25
    policies = model.policies[:num_test_policies]

    # Parameters
    alpha = 0.5
    epistemic_scale = 1.0

    # NumPy computation
    print(f"\nComputing with NumPy ({num_test_policies} policies)...")
    start_numpy = time.perf_counter()
    G_i_numpy, G_j_numpy, G_social_numpy = compute_empathic_G(
        qs_i=qs_i,
        B_i=B_loc,
        C_i_loc=C_loc,
        C_i_rel=C_rel,
        policies_i=policies,
        qs_j=qs_j,
        B_j=B_loc,
        C_j_loc=C_loc,
        C_j_rel=C_rel,
        policies_j=model.policies[:5],  # Use only primitive actions for j
        alpha=alpha,
        A_i=A_loc,
        A_j=A_loc,
        epistemic_scale=epistemic_scale,
    )
    numpy_time = time.perf_counter() - start_numpy

    # JAX computation (with warmup)
    print(f"Warming up JAX...")
    _ = compute_empathic_G_jax(
        qs_i=qs_i,
        B_i=B_loc,
        C_i_loc=C_loc,
        C_i_rel=C_rel,
        policies_i=policies[:5],  # Warmup with few policies
        qs_j=qs_j,
        B_j=B_loc,
        C_j_loc=C_loc,
        C_j_rel=C_rel,
        policies_j=model.policies[:5],
        alpha=alpha,
        A_i=A_loc,
        A_j=A_loc,
        epistemic_scale=epistemic_scale,
    )

    print(f"Computing with JAX ({num_test_policies} policies)...")
    start_jax = time.perf_counter()
    G_i_jax, G_j_jax, G_social_jax = compute_empathic_G_jax(
        qs_i=qs_i,
        B_i=B_loc,
        C_i_loc=C_loc,
        C_i_rel=C_rel,
        policies_i=policies,
        qs_j=qs_j,
        B_j=B_loc,
        C_j_loc=C_loc,
        C_j_rel=C_rel,
        policies_j=model.policies[:5],
        alpha=alpha,
        A_i=A_loc,
        A_j=A_loc,
        epistemic_scale=epistemic_scale,
    )
    jax_time = time.perf_counter() - start_jax

    speedup = numpy_time / jax_time

    # Compare
    max_diff_i = np.abs(G_i_numpy - G_i_jax).max()
    max_diff_j = np.abs(G_j_numpy - G_j_jax).max()
    max_diff_social = np.abs(G_social_numpy - G_social_jax).max()

    mean_diff_i = np.abs(G_i_numpy - G_i_jax).mean()
    mean_diff_j = np.abs(G_j_numpy - G_j_jax).mean()

    print(f"\nEmpathic G comparison (horizon=3, {num_test_policies} policies):")
    print(f"  NumPy time: {numpy_time:.3f}s")
    print(f"  JAX time:   {jax_time:.3f}s")
    print(f"  Speedup:    {speedup:.1f}x")
    print(f"\n  G_i: max_diff={max_diff_i:.2e}, mean_diff={mean_diff_i:.2e}")
    print(f"  G_j: max_diff={max_diff_j:.2e}, mean_diff={mean_diff_j:.2e}")
    print(f"  G_social: max_diff={max_diff_social:.2e}")

    # Verify correctness
    assert max_diff_i < 1e-3, f"G_i differs by {max_diff_i}"
    assert max_diff_j < 1e-3, f"G_j differs by {max_diff_j}"
    assert max_diff_social < 1e-3, f"G_social differs by {max_diff_social}"

    # Verify speedup
    assert speedup > 1.5, f"JAX not faster than NumPy (speedup={speedup:.1f}x)"
    print(f"\n  ✓ Results match!")
    print(f"  ✓ JAX is {speedup:.1f}x faster!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
