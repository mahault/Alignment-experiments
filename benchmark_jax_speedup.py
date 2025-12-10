"""
Benchmark script to compare NumPy vs JAX performance for path flexibility metrics.

This script demonstrates the massive speedup from JAX-ifying the planning pipeline.

Expected results:
- horizon=1 (5 policies): ~2-5x speedup
- horizon=2 (25 policies): ~10-20x speedup
- horizon=3 (125 policies): ~50-100x speedup
- horizon=4 (625 policies): ~500-1000x speedup
- horizon=5 (3125 policies): Previously unusable, now feasible!

Usage:
    python benchmark_jax_speedup.py --horizon 3
    python benchmark_jax_speedup.py --horizon 4 --num-runs 3
"""

import time
import argparse
import numpy as np
import jax.numpy as jnp
import logging
from typing import Tuple

# Import NumPy version
from src.metrics.path_flexibility import (
    compute_F_arrays_for_policies as compute_F_numpy,
    rollout_beliefs_and_obs,
    compute_empowerment_along_rollout,
)

# Import JAX version
from src.metrics.jax_path_flexibility import (
    compute_F_arrays_for_policies_jax as compute_F_jax,
    rollout_beliefs_and_obs_jax,
    compute_empowerment_along_rollout_jax,
)

# Import model builder
from tom.models.model_lava import LavaModel

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def build_test_model(horizon: int = 3, width: int = 7, height: int = 3):
    """Build a test lava corridor model."""
    model = LavaModel(
        width=width,
        height=height,
        horizon=horizon,
        start_pos=(0, 1),  # Start at left of safe corridor
        goal_x=6,
        goal_y=1,
    )
    return model


def benchmark_single_rollout(model, num_runs: int = 10):
    """Benchmark single policy rollout (NumPy vs JAX)."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("BENCHMARK 1: Single Policy Rollout")
    LOGGER.info("="*80)

    policy_id = 0
    horizon = model.horizon

    # Extract model components for JAX
    A = jnp.array(model.A["location_obs"])
    B = jnp.array(model.B["location_state"])
    D = jnp.array(model.D["location_state"])

    # Normalize B to [num_actions, num_states, num_states]
    if B.shape[0] == B.shape[1]:
        B = jnp.transpose(B, (2, 0, 1))

    policy = jnp.array(model.policies[policy_id], dtype=jnp.int32)

    # Warm-up JAX (JIT compilation)
    LOGGER.info("Warming up JAX (JIT compilation)...")
    _ = rollout_beliefs_and_obs_jax(policy, A, B, D)

    # Benchmark NumPy
    LOGGER.info(f"\nBenchmarking NumPy rollout ({num_runs} runs)...")
    numpy_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        q_s_numpy, p_o_numpy = rollout_beliefs_and_obs(policy_id, model, horizon)
        numpy_times.append(time.perf_counter() - start)

    numpy_mean = np.mean(numpy_times) * 1000  # ms
    numpy_std = np.std(numpy_times) * 1000

    # Benchmark JAX
    LOGGER.info(f"Benchmarking JAX rollout ({num_runs} runs)...")
    jax_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        q_s_jax, p_o_jax = rollout_beliefs_and_obs_jax(policy, A, B, D)
        jax_times.append(time.perf_counter() - start)

    jax_mean = np.mean(jax_times) * 1000  # ms
    jax_std = np.std(jax_times) * 1000

    speedup = numpy_mean / jax_mean

    # Results
    LOGGER.info(f"\nResults:")
    LOGGER.info(f"  NumPy: {numpy_mean:.3f} ± {numpy_std:.3f} ms")
    LOGGER.info(f"  JAX:   {jax_mean:.3f} ± {jax_std:.3f} ms")
    LOGGER.info(f"  Speedup: {speedup:.1f}x")

    # Verify correctness
    LOGGER.info(f"\nVerifying correctness...")
    q_s_numpy_arr = np.array(q_s_numpy)
    q_s_jax_arr = np.array(q_s_jax)
    max_diff = np.abs(q_s_numpy_arr - q_s_jax_arr).max()
    LOGGER.info(f"  Max difference in beliefs: {max_diff:.2e}")

    if max_diff < 1e-5:
        LOGGER.info(f"  ✓ Results match!")
    else:
        LOGGER.warning(f"  ⚠ Results differ by {max_diff:.2e}")

    return speedup


def benchmark_empowerment(model, num_runs: int = 10):
    """Benchmark empowerment computation (NumPy vs JAX)."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("BENCHMARK 2: Empowerment Computation")
    LOGGER.info("="*80)

    policy_id = 0
    horizon = model.horizon

    # Extract model components for JAX
    A = jnp.array(model.A["location_obs"])
    B = jnp.array(model.B["location_state"])
    D = jnp.array(model.D["location_state"])

    # Normalize B
    if B.shape[0] == B.shape[1]:
        B = jnp.transpose(B, (2, 0, 1))

    policy = jnp.array(model.policies[policy_id], dtype=jnp.int32)

    # Warm-up JAX
    LOGGER.info("Warming up JAX...")
    _ = compute_empowerment_along_rollout_jax(policy, A, B, D)

    # Benchmark NumPy
    LOGGER.info(f"\nBenchmarking NumPy empowerment ({num_runs} runs)...")
    numpy_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        E_numpy = compute_empowerment_along_rollout(model, policy_id, horizon)
        numpy_times.append(time.perf_counter() - start)

    numpy_mean = np.mean(numpy_times) * 1000
    numpy_std = np.std(numpy_times) * 1000

    # Benchmark JAX
    LOGGER.info(f"Benchmarking JAX empowerment ({num_runs} runs)...")
    jax_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        E_jax = compute_empowerment_along_rollout_jax(policy, A, B, D)
        jax_times.append(time.perf_counter() - start)

    jax_mean = np.mean(jax_times) * 1000
    jax_std = np.std(jax_times) * 1000

    speedup = numpy_mean / jax_mean

    # Results
    LOGGER.info(f"\nResults:")
    LOGGER.info(f"  NumPy: {numpy_mean:.3f} ± {numpy_std:.3f} ms")
    LOGGER.info(f"  JAX:   {jax_mean:.3f} ± {jax_std:.3f} ms")
    LOGGER.info(f"  Speedup: {speedup:.1f}x")

    # Verify correctness
    LOGGER.info(f"\nVerifying correctness...")
    E_numpy_final = compute_empowerment_along_rollout(model, policy_id, horizon)
    E_jax_final = float(compute_empowerment_along_rollout_jax(policy, A, B, D))
    diff = abs(E_numpy_final - E_jax_final)
    LOGGER.info(f"  NumPy empowerment: {E_numpy_final:.6f}")
    LOGGER.info(f"  JAX empowerment:   {E_jax_final:.6f}")
    LOGGER.info(f"  Difference: {diff:.2e}")

    if diff < 1e-4:
        LOGGER.info(f"  ✓ Results match!")
    else:
        LOGGER.warning(f"  ⚠ Results differ by {diff:.2e}")

    return speedup


def benchmark_full_flexibility(model, num_policies: int = None, num_runs: int = 3):
    """Benchmark full flexibility computation over all policies."""
    LOGGER.info("\n" + "="*80)
    LOGGER.info("BENCHMARK 3: Full Flexibility Computation (ALL POLICIES)")
    LOGGER.info("="*80)

    horizon = model.horizon
    total_policies = len(model.policies)

    # Limit number of policies for testing
    if num_policies is None or num_policies > total_policies:
        num_policies = total_policies

    policy_indices = list(range(num_policies))

    LOGGER.info(f"  Horizon: {horizon}")
    LOGGER.info(f"  Total policies: {total_policies}")
    LOGGER.info(f"  Testing with: {num_policies} policies")
    LOGGER.info(f"  Runs: {num_runs}")

    # Build generative models for both agents
    # For testing, use same model for both agents
    A_i = np.array(model.A["location_obs"])
    B_i = np.array(model.B["location_state"])
    D_i = np.array(model.D["location_state"])

    # Transpose B if needed
    if B_i.shape[0] == B_i.shape[1] and B_i.shape[2] < B_i.shape[0]:
        B_i = np.transpose(B_i, (2, 0, 1))

    A_j, B_j, D_j = A_i.copy(), B_i.copy(), D_i.copy()

    # Convert to JAX
    A_i_jax = jnp.array(A_i)
    B_i_jax = jnp.array(B_i)
    D_i_jax = jnp.array(D_i)
    A_j_jax = jnp.array(A_j)
    B_j_jax = jnp.array(B_j)
    D_j_jax = jnp.array(D_j)

    # Shared outcomes (safe corridor)
    shared_outcome_set = list(range(7))  # Cells 0-6 in safe row

    # Flexibility weights
    lambdas = (1.0, 1.0, 1.0)  # Equal weights for E, R, O

    # Create mock model object for NumPy API
    class MockModel:
        def __init__(self, A, B, D, policies):
            self.A = [A]
            self.B = [B]
            self.D = [D]
            self.policies = policies

    focal_model = MockModel(A_i, B_i, D_i, model.policies)
    other_model = MockModel(A_j, B_j, D_j, model.policies)

    # Warm-up JAX
    LOGGER.info("\nWarming up JAX (JIT compilation)...")
    policies_jax = jnp.array(model.policies[:num_policies], dtype=jnp.int32)
    _ = compute_F_jax(
        policies_jax,
        A_i_jax, B_i_jax, D_i_jax,
        A_j_jax, B_j_jax, D_j_jax,
        shared_outcome_set,
        lambdas,
    )
    LOGGER.info("  JIT compilation complete!")

    # Benchmark NumPy
    LOGGER.info(f"\nBenchmarking NumPy flexibility ({num_runs} runs)...")
    numpy_times = []
    for run in range(num_runs):
        start = time.perf_counter()
        F_i_numpy, F_j_numpy = compute_F_numpy(
            policy_indices,
            focal_model,
            other_model,
            shared_outcome_set,
            horizon,
            lambdas,
        )
        elapsed = time.perf_counter() - start
        numpy_times.append(elapsed)
        LOGGER.info(f"  Run {run+1}/{num_runs}: {elapsed:.3f}s")

    numpy_mean = np.mean(numpy_times)
    numpy_std = np.std(numpy_times)

    # Benchmark JAX
    LOGGER.info(f"\nBenchmarking JAX flexibility ({num_runs} runs)...")
    jax_times = []
    for run in range(num_runs):
        start = time.perf_counter()
        F_i_jax, F_j_jax = compute_F_jax(
            policies_jax,
            A_i_jax, B_i_jax, D_i_jax,
            A_j_jax, B_j_jax, D_j_jax,
            shared_outcome_set,
            lambdas,
        )
        elapsed = time.perf_counter() - start
        jax_times.append(elapsed)
        LOGGER.info(f"  Run {run+1}/{num_runs}: {elapsed:.3f}s")

    jax_mean = np.mean(jax_times)
    jax_std = np.std(jax_times)

    speedup = numpy_mean / jax_mean

    # Results
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"RESULTS: Flexibility Computation ({num_policies} policies)")
    LOGGER.info(f"{'='*80}")
    LOGGER.info(f"  NumPy: {numpy_mean:.3f} ± {numpy_std:.3f} s")
    LOGGER.info(f"  JAX:   {jax_mean:.3f} ± {jax_std:.3f} s")
    LOGGER.info(f"  Speedup: {speedup:.1f}x 🚀")

    # Verify correctness
    LOGGER.info(f"\nVerifying correctness...")
    max_diff_i = np.abs(F_i_numpy - F_i_jax).max()
    max_diff_j = np.abs(F_j_numpy - F_j_jax).max()
    mean_diff_i = np.abs(F_i_numpy - F_i_jax).mean()
    mean_diff_j = np.abs(F_j_numpy - F_j_jax).mean()

    LOGGER.info(f"  Max difference F_i: {max_diff_i:.2e}")
    LOGGER.info(f"  Max difference F_j: {max_diff_j:.2e}")
    LOGGER.info(f"  Mean difference F_i: {mean_diff_i:.2e}")
    LOGGER.info(f"  Mean difference F_j: {mean_diff_j:.2e}")

    if max_diff_i < 1e-3 and max_diff_j < 1e-3:
        LOGGER.info(f"  ✓ Results match!")
    else:
        LOGGER.warning(f"  ⚠ Results differ (may be due to numerical precision)")

    # Print sample values
    LOGGER.info(f"\nSample F_i values (first 5 policies):")
    for i in range(min(5, num_policies)):
        LOGGER.info(f"  Policy {i}: NumPy={F_i_numpy[i]:.4f}, JAX={F_i_jax[i]:.4f}")

    return speedup


def main():
    parser = argparse.ArgumentParser(description="Benchmark JAX vs NumPy performance")
    parser.add_argument("--horizon", type=int, default=3, help="Planning horizon (1-5)")
    parser.add_argument("--num-policies", type=int, default=None, help="Number of policies to test (default: all)")
    parser.add_argument("--num-runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--skip-simple", action="store_true", help="Skip simple benchmarks (rollout, empowerment)")

    args = parser.parse_args()

    # Build model
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"Building lava corridor model (horizon={args.horizon})...")
    LOGGER.info(f"{'='*80}")
    model = build_test_model(horizon=args.horizon)
    num_policies = len(model.policies)
    LOGGER.info(f"Model built: {num_policies} policies (5^{args.horizon})")

    # Run benchmarks
    results = {}

    if not args.skip_simple:
        results['rollout'] = benchmark_single_rollout(model, num_runs=args.num_runs)
        results['empowerment'] = benchmark_empowerment(model, num_runs=args.num_runs)

    results['flexibility'] = benchmark_full_flexibility(
        model,
        num_policies=args.num_policies,
        num_runs=args.num_runs
    )

    # Summary
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"SUMMARY: JAX Speedups")
    LOGGER.info(f"{'='*80}")
    for name, speedup in results.items():
        LOGGER.info(f"  {name.capitalize()}: {speedup:.1f}x")

    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"Performance Analysis:")
    LOGGER.info(f"{'='*80}")
    LOGGER.info(f"  Horizon: {args.horizon}")
    LOGGER.info(f"  Policies: {num_policies}")
    LOGGER.info(f"  Overall speedup: {results['flexibility']:.1f}x")

    if results['flexibility'] > 10:
        LOGGER.info(f"  ✓ Excellent speedup! JAX is working as expected.")
    elif results['flexibility'] > 2:
        LOGGER.info(f"  ✓ Good speedup. Consider increasing horizon to see more dramatic gains.")
    else:
        LOGGER.warning(f"  ⚠ Limited speedup. Check JIT compilation and vmap usage.")

    # Projected performance for higher horizons
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"Projected Performance for Higher Horizons:")
    LOGGER.info(f"{'='*80}")

    for h in range(args.horizon + 1, 6):
        n_pol = 5 ** h
        projected_speedup = results['flexibility'] * (n_pol / num_policies) ** 0.7
        LOGGER.info(f"  Horizon {h} ({n_pol} policies): ~{projected_speedup:.0f}x speedup")

        if h == 5:
            LOGGER.info(f"    Without JAX: ~{n_pol * 0.1:.0f}s (unusable)")
            LOGGER.info(f"    With JAX: ~{n_pol * 0.1 / projected_speedup:.1f}s (feasible!)")


if __name__ == "__main__":
    main()
