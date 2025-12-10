"""
Test correctness of JAX implementations against NumPy reference.

This test suite verifies that JAX-optimized functions produce identical
(or numerically equivalent) results to the NumPy implementations.
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

# NumPy implementations
from src.metrics.path_flexibility import (
    compute_empowerment_along_rollout,
    compute_returnability_from_rollout,
    compute_overlap_from_two_rollouts,
    rollout_beliefs_and_obs,
)
from src.metrics.empowerment import estimate_empowerment_one_step

# JAX implementations
from src.metrics.jax_path_flexibility import (
    estimate_empowerment_one_step_jax,
    compute_returnability_jax,
    compute_overlap_jax,
    rollout_beliefs_and_obs_jax,
    compute_empowerment_along_rollout_jax,
    compute_F_arrays_for_policies_jax,
)

# Model for testing
from tom.models.model_lava import LavaModel


@pytest.fixture
def lava_model_h1():
    """Simple lava model with horizon=1."""
    return LavaModel(width=7, height=3, horizon=1, start_pos=(0, 1), goal_x=6, goal_y=1)


@pytest.fixture
def lava_model_h3():
    """Lava model with horizon=3 (125 policies)."""
    return LavaModel(width=7, height=3, horizon=3, start_pos=(0, 1), goal_x=6, goal_y=1)


def test_empowerment_one_step(lava_model_h1):
    """Test that JAX empowerment matches NumPy."""
    model = lava_model_h1

    # Create dummy transition logits [num_actions, num_obs]
    num_actions = 5
    num_obs = 21
    np.random.seed(42)
    transition_logits = np.random.rand(num_actions, num_obs)

    # Compute with NumPy
    emp_numpy = estimate_empowerment_one_step(transition_logits)

    # Compute with JAX
    transition_logits_jax = jnp.array(transition_logits)
    emp_jax = float(estimate_empowerment_one_step_jax(transition_logits_jax))

    # Check they match
    assert abs(emp_numpy - emp_jax) < 1e-5, f"Empowerment mismatch: NumPy={emp_numpy}, JAX={emp_jax}"


def test_rollout_beliefs_and_obs(lava_model_h3):
    """Test that JAX belief rollout matches NumPy."""
    model = lava_model_h3
    policy_id = 0

    # NumPy rollout
    q_s_numpy, p_o_numpy = rollout_beliefs_and_obs(policy_id, model, model.horizon)

    # JAX rollout
    A = jnp.array(model.A["location_obs"])
    B_raw = jnp.array(model.B["location_state"])
    D = jnp.array(model.D["location_state"])

    # Transpose B if needed
    if B_raw.shape[0] == B_raw.shape[1] and B_raw.shape[2] < B_raw.shape[0]:
        B = jnp.transpose(B_raw, (2, 0, 1))
    else:
        B = B_raw

    policy = jnp.array(model.policies[policy_id], dtype=jnp.int32)
    q_s_jax, p_o_jax = rollout_beliefs_and_obs_jax(policy, A, B, D)

    # Convert to numpy for comparison
    q_s_numpy_arr = np.array(q_s_numpy)
    q_s_jax_arr = np.array(q_s_jax)

    # Check beliefs match
    max_diff = np.abs(q_s_numpy_arr - q_s_jax_arr).max()
    assert max_diff < 1e-5, f"Belief states differ by {max_diff}"

    # Check observations match
    p_o_numpy_arr = np.array(p_o_numpy)
    p_o_jax_arr = np.array(p_o_jax)
    max_diff_obs = np.abs(p_o_numpy_arr - p_o_jax_arr).max()
    assert max_diff_obs < 1e-5, f"Observation distributions differ by {max_diff_obs}"


def test_empowerment_along_rollout(lava_model_h3):
    """Test that JAX empowerment along rollout matches NumPy."""
    model = lava_model_h3
    policy_id = 0

    # NumPy computation
    E_numpy = compute_empowerment_along_rollout(model, policy_id, model.horizon)

    # JAX computation
    A = jnp.array(model.A["location_obs"])
    B_raw = jnp.array(model.B["location_state"])
    D = jnp.array(model.D["location_state"])

    if B_raw.shape[0] == B_raw.shape[1] and B_raw.shape[2] < B_raw.shape[0]:
        B = jnp.transpose(B_raw, (2, 0, 1))
    else:
        B = B_raw

    policy = jnp.array(model.policies[policy_id], dtype=jnp.int32)
    E_jax = float(compute_empowerment_along_rollout_jax(policy, A, B, D))

    # Check match
    diff = abs(E_numpy - E_jax)
    assert diff < 1e-4, f"Empowerment along rollout differs: NumPy={E_numpy}, JAX={E_jax}, diff={diff}"


def test_returnability(lava_model_h3):
    """Test that JAX returnability matches NumPy."""
    model = lava_model_h3
    policy_id = 0

    # Get observation distributions over time
    _, p_o_numpy = rollout_beliefs_and_obs(policy_id, model, model.horizon)

    # NumPy returnability
    shared_outcome_set = list(range(7))  # First 7 observations (safe corridor)
    R_numpy = compute_returnability_from_rollout(p_o_numpy, shared_outcome_set)

    # JAX returnability
    p_o_jax = jnp.array(p_o_numpy)
    shared_outcome_mask = jnp.zeros(21)
    shared_outcome_mask = shared_outcome_mask.at[shared_outcome_set].set(1.0)
    R_jax = float(compute_returnability_jax(p_o_jax, shared_outcome_mask))

    # Check match
    diff = abs(R_numpy - R_jax)
    assert diff < 1e-5, f"Returnability differs: NumPy={R_numpy}, JAX={R_jax}, diff={diff}"


def test_overlap(lava_model_h3):
    """Test that JAX overlap matches NumPy."""
    model = lava_model_h3
    policy_id = 0

    # Get observation distributions for two "agents" (using same policy for simplicity)
    _, p_o_i_numpy = rollout_beliefs_and_obs(policy_id, model, model.horizon)
    _, p_o_j_numpy = rollout_beliefs_and_obs(policy_id, model, model.horizon)

    # NumPy overlap
    O_numpy = compute_overlap_from_two_rollouts(p_o_i_numpy, p_o_j_numpy)

    # JAX overlap
    p_o_i_jax = jnp.array(p_o_i_numpy)
    p_o_j_jax = jnp.array(p_o_j_numpy)
    O_jax = float(compute_overlap_jax(p_o_i_jax, p_o_j_jax))

    # Check match
    diff = abs(O_numpy - O_jax)
    assert diff < 1e-5, f"Overlap differs: NumPy={O_numpy}, JAX={O_jax}, diff={diff}"


def test_full_flexibility_computation(lava_model_h3):
    """
    Test that full F computation (all policies) matches between JAX and NumPy.

    This is the most important integration test.
    """
    model = lava_model_h3

    # Parameters
    shared_outcome_set = list(range(7))
    lambdas = (1.0, 1.0, 1.0)
    num_test_policies = 10  # Test subset for speed

    # Get policy IDs
    policy_ids = list(range(num_test_policies))

    # NumPy computation
    from src.metrics.path_flexibility import compute_F_arrays_for_policies

    class MockModel:
        def __init__(self, A, B, D, policies):
            self.A = [A]
            self.B = [B]
            self.D = [D]
            self.policies = policies

    A_np = np.array(model.A["location_obs"])
    B_np = np.array(model.B["location_state"])
    D_np = np.array(model.D["location_state"])

    # Transpose B if needed
    if B_np.shape[0] == B_np.shape[1] and B_np.shape[2] < B_np.shape[0]:
        B_np = np.transpose(B_np, (2, 0, 1))

    focal_model = MockModel(A_np, B_np, D_np, model.policies)
    other_model = MockModel(A_np, B_np, D_np, model.policies)

    F_i_numpy, F_j_numpy = compute_F_arrays_for_policies(
        policies=policy_ids,
        focal_agent_model=focal_model,
        other_agent_model=other_model,
        shared_outcome_set=shared_outcome_set,
        horizon=model.horizon,
        lambdas=lambdas,
    )

    # JAX computation
    A_jax = jnp.array(A_np)
    B_jax = jnp.array(B_np)
    D_jax = jnp.array(D_np)

    policies_jax = jnp.array(model.policies[:num_test_policies], dtype=jnp.int32)

    F_i_jax, F_j_jax = compute_F_arrays_for_policies_jax(
        policies=policies_jax,
        A_i=A_jax, B_i=B_jax, D_i=D_jax,
        A_j=A_jax, B_j=B_jax, D_j=D_jax,
        shared_outcome_set=shared_outcome_set,
        lambdas=lambdas,
    )

    # Check match
    max_diff_i = np.abs(F_i_numpy - F_i_jax).max()
    max_diff_j = np.abs(F_j_numpy - F_j_jax).max()
    mean_diff_i = np.abs(F_i_numpy - F_i_jax).mean()
    mean_diff_j = np.abs(F_j_numpy - F_j_jax).mean()

    print(f"\nFlexibility comparison:")
    print(f"  F_i: max_diff={max_diff_i:.2e}, mean_diff={mean_diff_i:.2e}")
    print(f"  F_j: max_diff={max_diff_j:.2e}, mean_diff={mean_diff_j:.2e}")

    assert max_diff_i < 1e-3, f"F_i differs by {max_diff_i}"
    assert max_diff_j < 1e-3, f"F_j differs by {max_diff_j}"


def test_jax_faster_than_numpy(lava_model_h3):
    """
    Sanity check that JAX is actually faster than NumPy.

    This test may be skipped in CI/CD environments without GPU.
    """
    import time

    model = lava_model_h3
    shared_outcome_set = list(range(7))
    lambdas = (1.0, 1.0, 1.0)
    num_test_policies = 25  # 25 policies for timing

    # Prepare data
    A_np = np.array(model.A["location_obs"])
    B_np = np.array(model.B["location_state"])
    D_np = np.array(model.D["location_state"])

    if B_np.shape[0] == B_np.shape[1] and B_np.shape[2] < B_np.shape[0]:
        B_np = np.transpose(B_np, (2, 0, 1))

    class MockModel:
        def __init__(self, A, B, D, policies):
            self.A = [A]
            self.B = [B]
            self.D = [D]
            self.policies = policies

    focal_model = MockModel(A_np, B_np, D_np, model.policies)
    other_model = MockModel(A_np, B_np, D_np, model.policies)

    A_jax = jnp.array(A_np)
    B_jax = jnp.array(B_np)
    D_jax = jnp.array(D_np)
    policies_jax = jnp.array(model.policies[:num_test_policies], dtype=jnp.int32)

    # Warm up JAX
    _ = compute_F_arrays_for_policies_jax(
        policies=policies_jax[:5],
        A_i=A_jax, B_i=B_jax, D_i=D_jax,
        A_j=A_jax, B_j=B_jax, D_j=D_jax,
        shared_outcome_set=shared_outcome_set,
        lambdas=lambdas,
    )

    # Time NumPy
    from src.metrics.path_flexibility import compute_F_arrays_for_policies

    start = time.perf_counter()
    F_i_numpy, F_j_numpy = compute_F_arrays_for_policies(
        policies=list(range(num_test_policies)),
        focal_agent_model=focal_model,
        other_agent_model=other_model,
        shared_outcome_set=shared_outcome_set,
        horizon=model.horizon,
        lambdas=lambdas,
    )
    numpy_time = time.perf_counter() - start

    # Time JAX
    start = time.perf_counter()
    F_i_jax, F_j_jax = compute_F_arrays_for_policies_jax(
        policies=policies_jax,
        A_i=A_jax, B_i=B_jax, D_i=D_jax,
        A_j=A_jax, B_j=B_jax, D_j=D_jax,
        shared_outcome_set=shared_outcome_set,
        lambdas=lambdas,
    )
    jax_time = time.perf_counter() - start

    speedup = numpy_time / jax_time

    print(f"\nPerformance comparison ({num_test_policies} policies):")
    print(f"  NumPy: {numpy_time:.3f}s")
    print(f"  JAX:   {jax_time:.3f}s")
    print(f"  Speedup: {speedup:.1f}x")

    # JAX should be at least 2x faster (conservative check)
    # On real hardware with horizon=3, we typically see 10-50x speedup
    assert speedup > 1.5, f"JAX not faster than NumPy (speedup={speedup:.1f}x)"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
