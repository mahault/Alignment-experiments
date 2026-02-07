# Multi-Agent Theory of Mind with Empathy in Active Inference

This repository implements a framework for studying **coordination, alignment, and robustness** in multi-agent systems through:

- **Active Inference & Expected Free Energy (EFE)**
- **Recursive Theory of Mind (ToM) planning**
- **Empathy-weighted decision-making**
- **JAX-accelerated computation** (30-86x speedup)
- **Hierarchical zone-based planning** for complex layouts

The central research goal is to test whether **alignment emerges naturally** when agents attempt to preserve each other's future option sets — and whether **asymmetric empathy** enables coordination in constrained environments.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Running Experiments](#2-running-experiments)
3. [Architecture Overview](#3-architecture-overview)
4. [Code Structure](#4-code-structure)
5. [Key Concepts](#5-key-concepts)
6. [Hierarchical Planning](#6-hierarchical-planning)
7. [Understanding the Results](#7-understanding-the-results)
8. [JAX Acceleration](#8-jax-acceleration)
9. [Webots Robot Simulation](#9-webots-robot-simulation)
10. [Future Roadmap](#10-future-roadmap)
11. [Citation](#11-citation)

---

## 1. Quick Start

### Installation

```bash
# Create environment
conda create -n alignment python=3.10
conda activate alignment

# Install dependencies
pip install -r requirements.txt

# Install JAX (recommended for 20-100x speedup)
pip install jax  # CPU version
# OR for GPU: pip install jax[cuda12]
```

### Smoke Test

```bash
python tests/smoke_test_tom.py
```

Expected output:
- ✅ TOM imports
- ✅ LavaModel / LavaAgent creation
- ✅ LavaV2Env reset + step
- ✅ Collision detection (cell + edge)

### Run a Quick Experiment

```bash
# Quick test on narrow corridor (18 experiments, ~3 minutes)
python scripts/run_empathy_sweep.py --layouts narrow --max-steps 10 --seeds 1
```

---

## 2. Running Experiments

### Main Experiment: Empathy Sweep

The primary script is `scripts/run_empathy_sweep.py`. It tests how different empathy configurations affect coordination.

#### Basic Usage

```bash
# Run on a single layout
python scripts/run_empathy_sweep.py --layouts narrow

# Run on multiple layouts
python scripts/run_empathy_sweep.py --layouts narrow bottleneck wide

# Run all layouts (takes longer)
python scripts/run_empathy_sweep.py
```

#### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--layouts` | Layouts to test (space-separated) | All layouts |
| `--mode` | `symmetric`, `asymmetric`, or `both` | `both` |
| `--max-steps` | Max timesteps per episode | 15 |
| `--horizon` | Planning horizon | 4 |
| `--seeds` | Number of random seeds | 1 |
| `--hierarchical` | Use hierarchical planner (faster for bottlenecks) | False |
| `--verbose` | Print every timestep | False |

#### Recommended Experiments

```bash
# 1. Quick test - narrow corridor, asymmetric empathy
python scripts/run_empathy_sweep.py --layouts narrow --mode asymmetric --max-steps 10 --seeds 1

# 2. Full sweep on bottleneck layouts (uses hierarchical planner)
python scripts/run_empathy_sweep.py --layouts vertical_bottleneck symmetric_bottleneck --hierarchical

# 3. Compare symmetric vs asymmetric empathy
python scripts/run_empathy_sweep.py --layouts bottleneck --mode both

# 4. Debug a specific case
python scripts/run_empathy_sweep.py --layouts narrow --mode asymmetric --verbose
```

#### Available Layouts

| Layout | Description | Difficulty |
|--------|-------------|------------|
| `wide` | 6x3 open corridor | Easy |
| `narrow` | 6x3 single-file corridor | Hard |
| `bottleneck` | Wide with central chokepoint | Medium |
| `vertical_bottleneck` | Vertical with central chokepoint | Medium |
| `symmetric_bottleneck` | Equal-sized zones around chokepoint | Medium |
| `crossed_goals` | Goals require path crossing | Hard |
| `double_bottleneck` | Two sequential chokepoints | Hard |
| `passing_bay` | Narrow with one passing spot | Medium |
| `risk_reward` | Safe long path vs risky short path | Medium |
| `t_junction` | T-shaped intersection | Hard |
| `asymmetric_detour` | One agent must detour | Medium |

### Other Scripts

```bash
# Test asymmetric empathy scenarios
python scripts/test_asymmetric_empathy.py

# Single-agent demo
python scripts/run_lava_si.py

# Two-agent empathy demo
python scripts/run_lava_empathy.py

# Diagnose ToM behavior
python scripts/diagnose_tom.py
```

---

## 3. Architecture Overview

### How Planning Works

```
┌─────────────────────────────────────────────────────────────┐
│                    EmpathicLavaPlanner                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. RECURSIVE ToM: Predict other agent's action            │
│     ┌──────────────────────────────────────────────────┐   │
│     │  predict_other_action_recursive_jax()            │   │
│     │  - depth=2: I think that you think that I...     │   │
│     │  - horizon=3: Multi-step lookahead               │   │
│     │  - Uses JAX JIT for 20x speedup                  │   │
│     └──────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  2. EMPATHIC EFE: Compute G_social for all policies        │
│     ┌──────────────────────────────────────────────────┐   │
│     │  compute_empathic_G_jax()                        │   │
│     │  - G_social = G_self + α * G_other               │   │
│     │  - Collision detection (cell + edge)             │   │
│     │  - vmap over 125-625 policies                    │   │
│     └──────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  3. ACTION SELECTION: Softmax over G_social                │
│     ┌──────────────────────────────────────────────────┐   │
│     │  action = argmin(G_social)                       │   │
│     │  OR sample from q(π) ∝ exp(-γ * G_social)        │   │
│     └──────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Collision Detection

The system detects two types of collisions:

1. **Cell collision**: Both agents end up in the same cell
   - Detected via `A_cell_collision` observation matrix
   - Penalty in `C_cell_collision`

2. **Edge collision (swap)**: Agents try to pass through each other
   - Agent i moves A→B while agent j moves B→A
   - Detected via swap probability computation
   - Same penalty as cell collision

---

## 4. Code Structure

```
Alignment-experiments/
│
├── tom/                          # Core library
│   ├── models/
│   │   └── model_lava.py         # LavaModel: A, B, C, D matrices
│   │
│   ├── envs/
│   │   ├── env_lava_v2.py        # Multi-agent environment
│   │   └── env_lava_variants.py  # Layout definitions (11 layouts)
│   │
│   └── planning/
│       ├── si_empathy_lava.py    # EmpathicLavaPlanner (main class)
│       ├── jax_si_empathy_lava.py # JAX-accelerated functions
│       └── jax_hierarchical_planner.py # Hierarchical zone planner
│
├── webots_sim/                   # Physical robot simulation
│   ├── controllers/tiago_empathic/
│   │   ├── tiago_empathic.py     # Robot controller (motor, sensors)
│   │   └── tom_planner.py        # Discrete POMDP with EFE + ToM
│   ├── worlds/                   # 12 world files (11 generated + 1 hand-tuned)
│   ├── protos/                   # HazardObstacle, Target protos
│   ├── generate_worlds.py        # Generate all worlds from grid layouts
│   └── README.md                 # Webots-specific documentation
│
├── scripts/                      # Runnable experiments
│   ├── run_empathy_sweep.py      # Main experiment sweep
│   ├── test_asymmetric_empathy.py # ToM verification tests
│   ├── run_lava_empathy.py       # Two-agent demo
│   └── diagnose_tom.py           # Debug ToM predictions
│
├── tests/                        # Test suite
│   ├── smoke_test_tom.py         # Quick sanity check
│   ├── test_jax_planner.py       # JAX correctness tests
│   └── run_all_tests.py          # Full test suite
│
├── results/                      # Experiment outputs
│   ├── empathy_sweep_*.csv       # Sweep results
│   └── figs/                     # Generated plots
│
└── legacy/webots/                # Previous controller iterations
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `si_empathy_lava.py` | Main `EmpathicLavaPlanner` class. Orchestrates ToM + empathy |
| `jax_si_empathy_lava.py` | JAX-accelerated ToM functions (`predict_other_action_recursive_jax`) |
| `run_empathy_sweep.py` | Runs experiments across layouts and empathy configurations |
| `test_asymmetric_empathy.py` | Validates ToM produces correct predictions |

---

## 5. Key Concepts

### Theory of Mind (ToM)

Agents recursively model each other's beliefs and actions:

```
Depth 0: "What will j do?" → Assume j stays in place
Depth 1: "What will j do, given j thinks I'll stay?" → Better prediction
Depth 2: "What will j do, given j thinks I think j stays?" → Even better
```

The `TOM_DEPTH = 2` and `TOM_HORIZON = 3` parameters control recursion depth and lookahead.

### Empathy Parameter (α)

The empathy parameter α ∈ [0, 1] determines how much an agent weighs the other's utility:

```
G_social(π) = G_self(π) + α * G_other(π)
```

| α value | Behavior |
|---------|----------|
| α = 0 | Purely selfish - only cares about own goals |
| α = 0.5 | Balanced - weighs both equally |
| α = 1.0 | Fully empathic - other's utility as important as own |

### Asymmetric Empathy

The key insight: when agents have **different** empathy levels, coordination emerges:

| Agent i (α_i) | Agent j (α_j) | Outcome |
|---------------|---------------|---------|
| 0.0 (selfish) | 0.0 (selfish) | Both rush → **Collision** |
| 0.0 (selfish) | 1.0 (empathic) | i rushes, j yields → **Success** |
| 1.0 (empathic) | 0.0 (selfish) | i yields, j rushes → **Success** |
| 1.0 (empathic) | 1.0 (empathic) | Both yield → **Paralysis** (deadlock) |

### Expected Free Energy (EFE)

Each action is evaluated by its expected free energy:

```
G(a) = -pragmatic - epistemic
     = -E[utility(observations)] - info_gain(about_world)
```

Components:
- **Pragmatic**: Goal-seeking, collision avoidance
- **Epistemic**: Information gain (exploration)

---

## 6. Hierarchical Planning

For complex layouts with bottlenecks, the **hierarchical planner** decomposes planning into two levels:

### Two-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              HierarchicalEmpathicPlannerJax                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. HIGH-LEVEL: Zone transition planning                   │
│     ┌──────────────────────────────────────────────────┐   │
│     │  high_level_plan_jax()                           │   │
│     │  - State: (my_zone, other_zone)                  │   │
│     │  - Actions: STAY, FORWARD, BACK                  │   │
│     │  - Empathy at zone level (yielding bottleneck)   │   │
│     └──────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  2. LOW-LEVEL: Within-zone navigation                      │
│     ┌──────────────────────────────────────────────────┐   │
│     │  low_level_plan_multistep_jax()                  │   │
│     │  - Subgoal: exit point or final goal             │   │
│     │  - Multi-step ToM (depth=2, horizon=3)           │   │
│     │  - Smart subgoal switching at boundaries         │   │
│     └──────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Complexity Reduction

| Approach | Policies | Memory |
|----------|----------|--------|
| Flat H=7 | 5^7 = 78,125 | OOM |
| Hierarchical | 3^3 × 5^3 = 3,375 | OK |

### Usage

```bash
# Enable hierarchical planning
python scripts/run_empathy_sweep.py --layouts risk_reward --hierarchical

# Test asymmetric empathy with hierarchical planner
python scripts/test_asymmetric_empathy.py --layout risk_reward
```

### Supported Layouts

The hierarchical planner has zone definitions for:
- `vertical_bottleneck` - Vertical corridor with central chokepoint
- `symmetric_bottleneck` - Equal-sized zones around chokepoint
- `narrow` - Single-file corridor (3 zones)
- `risk_reward` - Safe long path vs risky short path (3 zones)

### Key Result: Asymmetric Empathy Enables Coordination

On `risk_reward` layout with asymmetric empathy (α_i=1.0, α_j=0.0):

```
Step 4:  i@(3,1) -> STAY    (empathic yields at bottleneck)
Step 5:  i@(3,1) -> STAY    (continues yielding)
...
Step 9:  j@(0,0) -> DOWN    (selfish passes through)
Step 10: i@(3,1) -> UP      (empathic resumes after j clears)
...
Step 14: Both reach goals   -> SUCCESS!
```

---

## 7. Understanding the Results

### Output Files

Results are saved to `results/empathy_sweep_YYYYMMDD_HHMMSS.csv`:

| Column | Description |
|--------|-------------|
| `layout` | Environment layout name |
| `start_config` | Starting configuration (A, B, C, D) |
| `alpha_i`, `alpha_j` | Empathy parameters |
| `both_success` | Both agents reached goals without collision |
| `cell_collision` | Agents ended up in same cell |
| `edge_collision` | Agents tried to swap positions |
| `paralysis` | Agents got stuck (both yielding) |
| `steps` | Number of timesteps |
| `trajectory_i`, `trajectory_j` | Position sequences |

### Key Metrics

1. **Success rate**: Both agents reach goals without collision
2. **Collision rate**: Agents crashed into each other
3. **Paralysis rate**: Both agents got stuck yielding to each other

### Analyzing Results

```python
import pandas as pd

# Load results
df = pd.read_csv("results/empathy_sweep_*.csv")

# Success rate by empathy configuration
success = df.groupby(['alpha_i', 'alpha_j'])['both_success'].mean()
print(success.unstack())

# Which layouts have highest collision rate?
collision_by_layout = df.groupby('layout')['cell_collision'].mean()
print(collision_by_layout.sort_values(ascending=False))
```

### Interpreting Trajectories

Look for yielding behavior in trajectories:
```
# Agent yields if they stay in place while other passes
trajectory_j: (4,1) → (4,1) → (4,1) → (3,1) → (2,1) → goal
                 ↑ stayed    ↑ stayed    ↑ started moving
```

---

## 8. JAX Acceleration

JAX provides **30-86x speedup** for planning computations through JIT compilation.

### Performance Comparison

| Function | NumPy | JAX (cached) | Speedup |
|----------|-------|--------------|---------|
| `predict_other_action_recursive` | ~0.5s | ~0.025s | **20x** |
| `compute_empathic_G` (125 policies) | ~45s | ~0.5s | **90x** |
| Hierarchical planner (multi-step ToM) | ~1.0s | ~0.013s | **86x** |
| JAX vs NumPy (ToM prediction) | ~0.12s | ~0.004s | **30x** |

### Usage

JAX is enabled by default when available:

```python
from tom.planning.si_empathy_lava import EmpathicLavaPlanner

# Automatically uses JAX (20-100x faster)
planner = EmpathicLavaPlanner(agent_i, agent_j, alpha=0.5)

# Disable JAX for debugging
planner = EmpathicLavaPlanner(agent_i, agent_j, alpha=0.5, use_jax=False)
```

### First-Call Compilation

JAX compiles functions on first call (JIT). Expect:
- First call: ~1s (compilation)
- Subsequent calls: ~0.025s (cached)

---

## 9. Webots Robot Simulation

The grid-based experiments are complemented by a **physical robot simulation** in [Webots](https://cyberbotics.com/) using two TIAGo robots. The same Active Inference + ToM framework runs on continuous coordinates with a discrete POMDP generative model.

### How It Works

Two TIAGo robots navigate toward each other's starting positions. The **empathic** robot (high alpha) discovers open areas in the arena and yields laterally to let the **selfish** robot (low alpha) pass. This yielding behavior emerges from Expected Free Energy computation, not hard-coded rules.

### Quick Start

```bash
# Generate all 11 world files from grid layouts
python webots_sim/generate_worlds.py

# Open a world in Webots
..\webots\msys64\mingw64\bin\webots.exe webots_sim/worlds/tiago_passing_bay.wbt

# Run the planner standalone (no Webots needed)
cd webots_sim/controllers/tiago_empathic && python tom_planner.py
```

### Available Worlds

| World | Challenge | Description |
|-------|-----------|-------------|
| `tiago_empathic_test` | Reference | Hand-tuned 5x2m corridor |
| `tiago_narrow` | Collision unavoidable | Single-file, no room to pass |
| `tiago_wide` | Easy passing | Two-lane corridor |
| `tiago_bottleneck` | Choke point | Wide areas + narrow center |
| `tiago_passing_bay` | Altruistic yielding | Narrow corridor with one bay |
| `tiago_symmetric_bottleneck` | Pure coordination | Opposite sides, same bottleneck |
| `tiago_t_junction` | Intersection | Agents approach from different directions |

See [`webots_sim/README.md`](webots_sim/README.md) for the full list and technical details.

---

## 10. Future Roadmap

See `HIERARCHICAL_PLANNER_ROADMAP.md` for detailed plans. Key upcoming features:

### Path Flexibility Metrics

Measure how robust a trajectory is:

- **Empowerment**: How many future options remain available
- **Returnability**: Probability of reaching safe states
- **Outcome overlap**: Similarity of predicted futures between agents

```
F(π) = λ_E * Empowerment(π) + λ_R * Returnability(π) + λ_O * Overlap(π)
```

### Flexibility-Aware Policy Prior

Bias agents toward flexible (robust) trajectories:

```
p(π) ∝ exp(κ * [F_i(π) + β * F_j(π)])
```

Combined objective:
```
J_i(π) = G_i + α*G_j - (κ/γ) * [F_i + β*F_j]
```

### Observation-Based Collision Inference

Replace hard-coded collision penalties with learned beliefs:
1. Track P(collision | zone_i, zone_j)
2. Update beliefs based on observed collisions
3. High-level planner uses inferred probabilities

---

## 11. Citation

If you use this codebase, please cite:

```bibtex
@software{albarracin2025_empathic_tom,
  title={Multi-Agent Theory of Mind with Empathy in Active Inference},
  author={Mahault Albarracin},
  year={2025},
  url={https://github.com/mahault/Alignment-experiments}
}
```

---

## Contact

Issues & discussions: https://github.com/mahault/Alignment-experiments/issues
