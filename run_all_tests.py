#!/usr/bin/env python3
"""
Run all tests for the Alignment-experiments repo.

This script runs all TOM-compatible JAX-based tests:

1. smoke_test_tom.py              - TOM infrastructure smoke test
2. test_lava_env_tom.py           - LavaV1Env, LavaModel, LavaAgent
3. test_model_creation_tom.py     - TOM model creation (dict structure)
4. test_integration_tom.py        - TOM component integration
5. test_lava_v2_env.py            - LavaV2Env with layout variants
6. test_path_flexibility_metrics.py - E, R, O, F metrics
7. test_F_aware_prior.py          - F-aware policy prior
8. test_jax_correctness.py        - JAX vs NumPy correctness
9. test_jax_empathy.py            - JAX empathy implementation
10. test_4d_b_matrix.py           - 4D B matrix and multi-step ToM
11. test_jax_planner.py           - JAX planner speedup verification

Outputs are saved to: results/test_logs/
The entire suite always runs regardless of failures.
"""

import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

# Colors (for terminal output)
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

# Setup log directory
LOG_DIR = Path(__file__).parent / "results" / "test_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Create timestamped log file
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"test_run_{TIMESTAMP}.log"


def log(message, also_print=True):
    """Write message to log file and optionally print to console."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        # Strip ANSI color codes for log file
        clean_msg = message
        for code in [GREEN, RED, YELLOW, BLUE, CYAN, RESET]:
            clean_msg = clean_msg.replace(code, "")
        f.write(clean_msg + "\n")
    if also_print:
        print(message)


def run(title, cmd):
    """Run a shell command with pretty banners and logging."""
    log("\n" + "="*80)
    log(f"{BLUE}{title}{RESET}")
    log("="*80)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            check=False,
            capture_output=True,
        )

        # Log stdout and stderr
        if result.stdout:
            log(result.stdout, also_print=True)
        if result.stderr:
            log(result.stderr, also_print=True)

        if result.returncode == 0:
            log(f"{GREEN}PASSED: {title}{RESET}")
            return True
        else:
            log(f"{RED}FAILED: {title}{RESET}")
            return False
    except Exception as e:
        log(f"{RED}ERROR running {title}:{RESET}")
        log(str(e))
        return False


def main():
    log(f"\nTest run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Log file: {LOG_FILE}\n")

    log("\n" + "="*80)
    log(f"{CYAN}          RUNNING FULL TEST SUITE{RESET}")
    log("="*80 + "\n")

    failures = []

    # 1. TOM Smoke Test
    log(f"{YELLOW}STEP 1: TOM Infrastructure Smoke Test...{RESET}")
    if not run("TOM Smoke Test", "python smoke_test_tom.py"):
        failures.append("smoke_test_tom.py")

    # 2. TOM Environment Tests
    log(f"\n{YELLOW}STEP 2: TOM Environment (LavaV1Env, LavaModel)...{RESET}")
    if not run("TOM Environment", "pytest tests/test_lava_env_tom.py -v"):
        failures.append("test_lava_env_tom.py")

    # 3. TOM Model Creation Tests
    log(f"\n{YELLOW}STEP 3: TOM Model Creation...{RESET}")
    if not run("TOM Model Creation", "pytest tests/test_model_creation_tom.py -v"):
        failures.append("test_model_creation_tom.py")

    # 4. TOM Integration Tests
    log(f"\n{YELLOW}STEP 4: TOM Integration...{RESET}")
    if not run("TOM Integration", "pytest tests/test_integration_tom.py -v"):
        failures.append("test_integration_tom.py")

    # 5. LavaV2Env Layout Variants
    log(f"\n{YELLOW}STEP 5: LavaV2Env Layout Variants...{RESET}")
    if not run("LavaV2Env Variants", "pytest tests/test_lava_v2_env.py -v"):
        failures.append("test_lava_v2_env.py")

    # 6. Path Flexibility Metrics Tests
    log(f"\n{YELLOW}STEP 6: Path Flexibility Metrics (E, R, O, F)...{RESET}")
    if not run("Path Flexibility Metrics", "pytest tests/test_path_flexibility_metrics.py -v"):
        failures.append("test_path_flexibility_metrics.py")

    # 7. F-Aware Prior Tests
    log(f"\n{YELLOW}STEP 7: F-Aware Policy Prior...{RESET}")
    if not run("F-Aware Prior", "pytest tests/test_F_aware_prior.py -v"):
        failures.append("test_F_aware_prior.py")

    # 8. JAX Correctness Tests (Path Flexibility)
    log(f"\n{YELLOW}STEP 8: JAX Path Flexibility Correctness...{RESET}")
    if not run("JAX Path Flexibility", "pytest tests/test_jax_correctness.py -v"):
        failures.append("test_jax_correctness.py")

    # 9. JAX Empathy Tests
    log(f"\n{YELLOW}STEP 9: JAX Empathy Rollout Correctness...{RESET}")
    if not run("JAX Empathy Rollout", "pytest tests/test_jax_empathy.py -v"):
        failures.append("test_jax_empathy.py")

    # 10. 4D B Matrix Tests (root level)
    log(f"\n{YELLOW}STEP 10: 4D B Matrix and Multi-Step ToM...{RESET}")
    if not run("4D B Matrix", "python test_4d_b_matrix.py"):
        failures.append("test_4d_b_matrix.py")

    # 11. JAX Planner Speedup Test (root level)
    log(f"\n{YELLOW}STEP 11: JAX Planner Speedup Verification...{RESET}")
    if not run("JAX Planner Speedup", "python test_jax_planner.py"):
        failures.append("test_jax_planner.py")

    # Summary
    log("\n" + "="*80)
    log(f"{CYAN}                            SUMMARY{RESET}")
    log("="*80)

    total_tests = 11
    passed_tests = total_tests - len(failures)

    if len(failures) == 0:
        log(f"{GREEN}ALL {total_tests} TESTS PASSED!{RESET}\n")
        log("="*80)
        log(f"{CYAN}Verified components:{RESET}")
        log(f"  {GREEN}+{RESET} LavaModel (JAX generative model with dict structure)")
        log(f"  {GREEN}+{RESET} LavaAgent (thin wrapper around model)")
        log(f"  {GREEN}+{RESET} LavaV1Env (JAX environment with multi-agent support)")
        log(f"  {GREEN}+{RESET} LavaV2Env (layout variants: narrow, wide, bottleneck, risk-reward)")
        log(f"  {GREEN}+{RESET} Manual Bayesian inference (no PyMDP dependency)")
        log(f"  {GREEN}+{RESET} 4D B matrix for collision-aware transitions")
        log(f"  {GREEN}+{RESET} Path flexibility metrics (E, R, O, F)")
        log(f"  {GREEN}+{RESET} F-aware policy prior")
        log(f"  {GREEN}+{RESET} JAX-accelerated empathy planner")
        log(f"  {GREEN}+{RESET} Multi-agent ToM interactions")
        log("="*80)
    else:
        log(f"{RED}FAILED: {len(failures)}/{total_tests} tests{RESET}")
        log(f"{GREEN}PASSED: {passed_tests}/{total_tests} tests{RESET}\n")
        log(f"{RED}Failed tests:{RESET}")
        for f in failures:
            log(f"  - {RED}{f}{RESET}")
        log("="*80)

    log(f"\nTest run completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Log saved to: {LOG_FILE}\n")

    if len(failures) == 0:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
