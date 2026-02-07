"""
Run Webots experiments with empathic multi-agent coordination.

This script launches Webots simulations with different empathy parameters
and records results, similar to the alignment-experiments grid sweep.

Usage:
    python run_webots_experiment.py --world narrow_corridor --alpha 0.5
    python run_webots_experiment.py --sweep  # Run full parameter sweep
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Paths
WEBOTS_SIM_DIR = Path(__file__).parent
ALIGNMENT_DIR = WEBOTS_SIM_DIR.parent
WEBOTS_INSTALL = ALIGNMENT_DIR / "webots"

# Find Webots executable
WEBOTS_EXE = None
for candidate in [
    WEBOTS_INSTALL / "msys64" / "mingw64" / "bin" / "webots.exe",
    WEBOTS_INSTALL / "webots.exe",
    Path("C:/Program Files/Webots/msys64/mingw64/bin/webots.exe"),
]:
    if candidate.exists():
        WEBOTS_EXE = candidate
        break


def run_simulation(world_file: str, alpha_1: float = 0.5, alpha_2: float = 0.5,
                   timeout: int = 120, headless: bool = False):
    """
    Run a single Webots simulation.

    Parameters
    ----------
    world_file : str
        Name of the world file (without path)
    alpha_1 : float
        Empathy parameter for Robot 1
    alpha_2 : float
        Empathy parameter for Robot 2
    timeout : int
        Maximum simulation time in seconds
    headless : bool
        Run without GUI (faster)

    Returns
    -------
    dict
        Results including success, collision, time
    """
    world_path = WEBOTS_SIM_DIR / "worlds" / world_file

    if not world_path.exists():
        print(f"Error: World file not found: {world_path}")
        return None

    if WEBOTS_EXE is None:
        print("Error: Webots executable not found")
        return None

    # Set environment variables for the controller
    env = os.environ.copy()
    env['WEBOTS_ALPHA_1'] = str(alpha_1)
    env['WEBOTS_ALPHA_2'] = str(alpha_2)
    env['PYTHONPATH'] = str(ALIGNMENT_DIR) + os.pathsep + env.get('PYTHONPATH', '')

    # Build command
    cmd = [str(WEBOTS_EXE)]

    if headless:
        cmd.extend(['--mode=fast', '--no-rendering', '--minimize'])

    cmd.append(str(world_path))

    print(f"Running: {world_file} with alpha=({alpha_1}, {alpha_2})")

    try:
        proc = subprocess.Popen(cmd, env=env)
        proc.wait(timeout=timeout)
        return {'status': 'completed', 'world': world_file, 'alpha_1': alpha_1, 'alpha_2': alpha_2}
    except subprocess.TimeoutExpired:
        proc.kill()
        return {'status': 'timeout', 'world': world_file, 'alpha_1': alpha_1, 'alpha_2': alpha_2}
    except Exception as e:
        return {'status': 'error', 'error': str(e), 'world': world_file}


def run_parameter_sweep(world_file: str, alphas: list = None):
    """
    Run a sweep over empathy parameters.

    Parameters
    ----------
    world_file : str
        World file to use
    alphas : list
        List of alpha values to test
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = []

    for alpha_1 in alphas:
        for alpha_2 in alphas:
            result = run_simulation(world_file, alpha_1, alpha_2)
            if result:
                results.append(result)
                print(f"  Result: {result['status']}")

    return results


def list_worlds():
    """List available world files."""
    worlds_dir = WEBOTS_SIM_DIR / "worlds"
    if worlds_dir.exists():
        return [f.name for f in worlds_dir.glob("*.wbt")]
    return []


def main():
    parser = argparse.ArgumentParser(description="Run Webots empathy experiments")
    parser.add_argument('--world', type=str, default='narrow_corridor.wbt',
                        help='World file to run')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Symmetric empathy parameter for both agents')
    parser.add_argument('--alpha1', type=float, default=None,
                        help='Empathy parameter for agent 1')
    parser.add_argument('--alpha2', type=float, default=None,
                        help='Empathy parameter for agent 2')
    parser.add_argument('--sweep', action='store_true',
                        help='Run full parameter sweep')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Simulation timeout in seconds')
    parser.add_argument('--list', action='store_true',
                        help='List available worlds')

    args = parser.parse_args()

    if args.list:
        print("Available worlds:")
        for world in list_worlds():
            print(f"  {world}")
        return

    if WEBOTS_EXE is None:
        print("Error: Webots not found. Please install Webots or set WEBOTS_HOME.")
        print(f"Looked in: {WEBOTS_INSTALL}")
        return

    print(f"Using Webots: {WEBOTS_EXE}")

    alpha_1 = args.alpha1 if args.alpha1 is not None else args.alpha
    alpha_2 = args.alpha2 if args.alpha2 is not None else args.alpha

    if args.sweep:
        results = run_parameter_sweep(args.world)
        print(f"\nCompleted {len(results)} simulations")
    else:
        result = run_simulation(
            args.world,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            timeout=args.timeout,
            headless=args.headless
        )
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
