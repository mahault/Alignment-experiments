# Legacy Code Archive

This folder contains deprecated implementations that have been superseded by the modern JAX-based Active Inference framework.

## Why These Are Legacy

### 1. PyMDP-Based Implementation
The original codebase used **PyMDP** (Python package for Markov Decision Processes and Active Inference). While functional, this approach had several limitations:

- **Performance**: Pure Python/NumPy was slow for multi-agent simulations
- **Scalability**: Difficult to run large parameter sweeps or batch experiments
- **Vectorization**: Limited support for parallel trajectory computation

### 2. Migration to JAX
The current implementation uses **JAX** for:
- GPU acceleration and JIT compilation
- Efficient batched operations with `vmap`
- Automatic differentiation (useful for future gradient-based methods)
- Much faster experiment iteration (~10-100x speedup)

## What's In Here

### `/agents/`
- `empathetic_agent.py` - Original PyMDP empathetic agent (196 lines)
- `empowerment.py` - Early empowerment metric implementation

### `/env/`
- `gridworld.py` - Legacy gridworld environment

### `/models/`
- `model_cleanup.py` - CleanUp game model (multi-agent coordination task)
- `model_foraging.py` - Foraging environment model
- `model_ocv1.py` - Overcooked V1 environment model

### `/tom/`
- Old Theory of Mind package structure before refactoring

### `/notebooks/`
- Experimental notebooks (IWAI 2025, MeltingPot experiments)

### `sim.py`, `sweep.py`
- Legacy simulation and parameter sweep scripts

## Current Active Code

The active codebase is in:
- `tom/` - Modern JAX-based environments, models, and planning
- `src/` - Metrics, configuration, and utilities
- `experiments/` - Current experiment scripts
- `scripts/` - Runnable experiment launchers

## Historical Notes

These environments were explored during development but not used in final experiments:
- **CleanUp**: Multi-agent social dilemma (complex state space)
- **Foraging**: Resource collection (simpler than needed)
- **Overcooked**: Cooking coordination (integration complexity with MeltingPot)

The **Lava Corridor** environment was chosen for final experiments because:
1. Clear coordination problem with measurable outcomes
2. Simple enough for interpretable results
3. Rich enough to demonstrate Theory of Mind benefits
4. Multiple layout variants for generalization testing
