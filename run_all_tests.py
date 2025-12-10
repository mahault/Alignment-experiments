#!/usr/bin/env python3
"""
Run all TOM-compatible tests for the Alignment-experiments repo.

This script runs, in order:

1. smoke_test_tom.py           (TOM-style JAX stack)
2. pytest tests/test_path_flexibility_metrics.py
3. pytest tests/test_F_aware_prior.py

Skipped (legacy PyMDP tests):
- smoke_test.py (PyMDP, always fails)
- test_lava_rollout.py (uses PyMDP LavaCorridorEnv)
- test_agent_factory.py (uses PyMDP agents)
- test_integration_rollout.py (uses PyMDP rollouts)

It prints colored PASS/FAIL banners and stops immediately on fatal errors.
"""

import subprocess
import sys
import os

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

def run(title, cmd, stop_on_failure=True):
    """Run a shell command with pretty banners."""
    print("\n" + "="*80)
    print(f"{BLUE}{title}{RESET}")
    print("="*80)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            print(f"{GREEN}✓ PASSED: {title}{RESET}")
            return True
        else:
            print(f"{RED}✗ FAILED: {title}{RESET}")
            if stop_on_failure:
                print(f"\n{RED}Stopping due to failure.{RESET}")
                print(f"{YELLOW}Fix the error above before continuing.{RESET}\n")
                sys.exit(1)
            return False
    except Exception as e:
        print(f"{RED}✗ ERROR running {title}:{RESET}")
        print(e)
        if stop_on_failure:
            sys.exit(1)
        return False


def main():
    print("\n" + "="*80)
    print(f"{CYAN}          RUNNING TOM-COMPATIBLE TEST SUITE{RESET}")
    print("="*80 + "\n")

    failures = []

    # 1. TOM Smoke Test (REQUIRED)
    print(f"{YELLOW}STEP 1: Verifying TOM-style JAX infrastructure...{RESET}")
    if not run("TOM Smoke Test", "python smoke_test_tom.py", stop_on_failure=True):
        failures.append("smoke_test_tom.py")

    # 2. TOM Environment Tests
    print(f"\n{YELLOW}STEP 2: Testing TOM environment (LavaV1Env, LavaModel)...{RESET}")
    if not run("TOM Environment", "pytest tests/test_lava_env_tom.py -v", stop_on_failure=False):
        failures.append("test_lava_env_tom.py")

    # 3. TOM Model Creation Tests
    print(f"\n{YELLOW}STEP 3: Testing TOM model creation...{RESET}")
    if not run("TOM Model Creation", "pytest tests/test_model_creation_tom.py -v", stop_on_failure=False):
        failures.append("test_model_creation_tom.py")

    # 4. TOM Integration Tests
    print(f"\n{YELLOW}STEP 4: Testing TOM integration...{RESET}")
    if not run("TOM Integration", "pytest tests/test_integration_tom.py -v", stop_on_failure=False):
        failures.append("test_integration_tom.py")

    # 5. Path Flexibility Metrics Tests
    print(f"\n{YELLOW}STEP 5: Testing path flexibility metrics (E, R, O, F)...{RESET}")
    if not run("Path Flexibility Metrics", "pytest tests/test_path_flexibility_metrics.py -v", stop_on_failure=False):
        failures.append("test_path_flexibility_metrics.py")

    # 6. F-Aware Prior Tests
    print(f"\n{YELLOW}STEP 6: Testing F-aware policy prior...{RESET}")
    if not run("F-Aware Prior", "pytest tests/test_F_aware_prior.py -v", stop_on_failure=False):
        failures.append("test_F_aware_prior.py")

    # 7. JAX Correctness Tests (Path Flexibility)
    print(f"\n{YELLOW}STEP 7: Testing JAX path flexibility correctness...{RESET}")
    if not run("JAX Path Flexibility", "pytest tests/test_jax_correctness.py -v", stop_on_failure=False):
        failures.append("test_jax_correctness.py")

    # 8. JAX Empathy Tests
    print(f"\n{YELLOW}STEP 8: Testing JAX empathy rollout correctness...{RESET}")
    if not run("JAX Empathy Rollout", "pytest tests/test_jax_empathy.py -v", stop_on_failure=False):
        failures.append("test_jax_empathy.py")

    # Summary
    print("\n" + "="*80)
    print(f"{CYAN}                            SUMMARY{RESET}")
    print("="*80)

    if len(failures) == 0:
        print(f"{GREEN}ALL TOM-COMPATIBLE TESTS PASSED! 🎉{RESET}\n")
        print(f"{CYAN}TOM-style JAX architecture is verified and working.{RESET}\n")
        print("="*80)
        print(f"{CYAN}What's been tested:{RESET}")
        print(f"  {GREEN}✓{RESET} LavaModel (pure JAX generative model with dict structure)")
        print(f"  {GREEN}✓{RESET} LavaAgent (thin wrapper around model)")
        print(f"  {GREEN}✓{RESET} LavaV1Env (JAX environment with multi-agent support)")
        print(f"  {GREEN}✓{RESET} Manual Bayesian inference (no PyMDP)")
        print(f"  {GREEN}✓{RESET} Policy evaluation using B matrix")
        print(f"  {GREEN}✓{RESET} Path flexibility metrics (E, R, O, F)")
        print(f"  {GREEN}✓{RESET} F-aware policy prior")
        print(f"  {GREEN}✓{RESET} Multi-agent interactions")
        print("\n" + "="*80)
        print(f"{CYAN}Next steps to complete TOM integration:{RESET}")
        print("  1. Add TOM-style EFE computation (port from tom/planning/si_tom.py)")
        print("  2. Implement policy search using computed EFE")
        print("  3. Add multi-agent TOM rollouts (agents reasoning about each other)")
        print("  4. Connect to path flexibility metrics during planning")
        print("  5. Run Experiments 1 & 2 with full TOM pipeline")
        print("="*80 + "\n")
        sys.exit(0)
    else:
        print(f"{RED}Some tests failed:{RESET}")
        for f in failures:
            print(f"  - {RED}{f}{RESET}")
        print("\n" + "="*80)
        print(f"{YELLOW}Skipped (legacy PyMDP tests, not TOM-compatible):{RESET}")
        print("  - smoke_test.py (PyMDP, always fails)")
        print("  - test_lava_rollout.py (uses PyMDP LavaCorridorEnv)")
        print("  - test_agent_factory.py (uses PyMDP agents)")
        print("  - test_integration_rollout.py (uses PyMDP rollouts)")
        print("\n" + "="*80)
        print(f"{CYAN}Note:{RESET} These tests have been replaced by TOM-compatible versions:")
        print(f"  {GREEN}✓{RESET} test_lava_env_tom.py (replaces test_lava_rollout.py)")
        print(f"  {GREEN}✓{RESET} test_model_creation_tom.py (replaces test_agent_factory.py)")
        print(f"  {GREEN}✓{RESET} test_integration_tom.py (replaces test_integration_rollout.py)")
        print("="*80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
