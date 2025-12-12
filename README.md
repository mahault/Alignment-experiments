# Path Flexibility, Empathy, and Theory of Mind in Active Inference

This repository implements a framework for studying **coordination, alignment, and robustness** in multi-agent systems through:

- **Active Inference & Expected Free Energy (EFE)**
- **Theory of Mind (ToM) planning**
- **Empathy-weighted decision-making**
- **Path flexibility metrics** (empowerment, returnability, overlap)

The central research goal is to test whether **alignment emerges naturally** when agents attempt to preserve each other's future option sets — and whether a **flexibility-aware prior** improves cooperative behavior in challenging environments.

---

## 1. Conceptual Overview

### Active Inference

Agents select policies by minimizing **expected free energy**:

\[
q(\pi) \propto \exp(-\gamma G(\pi))
\]

where \( G(\pi) \) combines:
- **Pragmatic value**: Preference satisfaction (goal-seeking, lava avoidance)
- **Epistemic value**: Information gain (exploration)
- **Collision avoidance**: Multi-agent coordination penalties

### Theory of Mind (ToM)

Agents maintain generative models of other agents' beliefs and policies. During planning, each agent simulates the other's EFE landscape and best-responds to predicted actions.

### Empathy

Empathy parameter α ∈ [0,1] weights the other agent's EFE:

\[
G_{\text{social}}^i(\pi) = G_i(\pi) + \alpha\, G_j(\pi)
\]

- α = 0 → purely selfish
- α = 1 → fully prosocial

### Path Flexibility

Path flexibility measures how robust a future trajectory is using:

- **Empowerment** — how many future observations remain under agent control
- **Returnability** — probability of reaching common safe outcomes
- **Outcome overlap** — similarity of predicted future outcomes between agents

\[
F(\pi) = \lambda_E E(\pi) + \lambda_R R(\pi) + \lambda_O O(\pi)
\]

High flexibility ⇒ agents preserve each other's future option sets.

### Flexibility-Aware Policy Prior (Experiment 2)

A policy prior biases agents toward flexible (robust) trajectories:

\[
p(\pi) \propto \exp\big(\kappa \left[F_i(\pi) + \beta F_j(\pi)\right]\big)
\]

Combined objective:

\[
J_i(\pi)=G_i + \alpha G_j - \frac{\kappa}{\gamma}[F_i + \beta F_j]
\]

---

## 2. Repository Structure

```text
Alignment-experiments/
│
├── tom/
│   ├── models/
│   │   └── model_lava.py              # LavaModel & LavaAgent (JAX)
│   ├── envs/
│   │   ├── env_lava_v2.py             # Multi-layout environment
│   │   └── env_lava_variants.py       # Layout definitions
│   ├── planning/
│   │   ├── si_lava.py                 # Single-agent EFE planner
│   │   ├── si_empathy_lava.py         # Empathy planner (NumPy)
│   │   ├── jax_si_empathy_lava.py     # Empathy planner (JAX - 50-100x faster)
│   │   ├── jax_hierarchical_planner.py # Hierarchical zone-based planner (JAX)
│   │   └── si_tom_F_prior.py          # Flexibility-aware ToM planner
│   │
│   └── metrics/
│       ├── path_flexibility.py        # F, E, R, O metrics (NumPy)
│       └── jax_path_flexibility.py    # F, E, R, O metrics (JAX - 60-130x faster)
│
├── scripts/
│   ├── run_lava_si.py                 # Single-agent demo
│   ├── run_lava_empathy.py            # Two-agent empathy demo
│   └── run_empathy_sweep.py           # Full empathy sweep experiments
│
├── experiments/
│   ├── exp1_flex_vs_efe.py            # Measure F–EFE correlation
│   └── exp2_flex_prior.py             # Flexibility-aware prior experiments
│
└── tests/                             # Comprehensive test suite
```

---

## 3. Core Components

### LavaModel (JAX Generative Model)

Implements proper multi-agent Active Inference with:

**Joint B matrix**: `B[s', s, s_other, a]` conditions transitions on other agent's position
- Enforces single-occupancy: can't move into occupied cells
- Prevents edge collisions: both agents blocked if trying to swap positions

**Multi-modal observations**:
- `location_obs`: Agent's own position
- `edge_obs`: Edge being traversed (for path tracking)
- `cell_collision_obs`: Same-cell collision detection
- `edge_collision_obs`: Edge-swap collision detection (crossing same edge from opposite sides)

**Collision penalties in C matrix**:
- Cell collision: C = -100 (agents in same cell)
- Edge collision: C = -100 (agents swapping through same edge)
- Lava: C = -100
- Goal: C = +10

**Multi-step policies** (horizon H > 1):
- Enables planning of multi-step detours and turn-taking sequences

### EmpathicLavaPlanner

Implements Theory of Mind with recursive planning:

**Single-step (H=1)**: Conditions G_j on i's predicted next position

**Multi-step (H>1)**: Full recursive rollout over horizon
- For each timestep t:
  1. i takes action a_i[t] (committed)
  2. j observes i's new position
  3. j computes G_j for ALL actions and selects best response
  4. Both beliefs updated for next timestep
- Accumulated EFE over full horizon

**JAX acceleration**: 50-100x faster than NumPy (enabled by default)

**Empathy weight α**: `G_social = G_i + α * G_j`

### LavaV2Env

Multi-layout environment with:
- Full observability: agents see both positions
- Collision detection: cell collisions AND edge collisions
- Layouts: Wide, Bottleneck, Narrow, Crossed Goals, Risk-Reward

---

## 4. JAX Acceleration 🚀

JAX provides **50-130x speedup** for planning, making horizon 4-5 experiments feasible.

### Performance

| Component | Horizon | Policies | NumPy | JAX | Speedup |
|-----------|---------|----------|-------|-----|---------|
| **Empathy Planning** | H=3 | 125 | ~45s | ~0.5s | **90x** |
| | H=4 | 625 | ~5 min | ~3s | **100x** |
| **Path Flexibility** | H=3 | 125 | ~45s | ~0.7s | **60x** |
| | H=5 | 3125 | ~30 min | ~15s | **130x** |

**Full episode (20 timesteps, H=3)**: 15 min → 10s (90x speedup)

### Usage

**Default behavior** (JAX enabled automatically):
```python
from tom.planning.si_empathy_lava import EmpathicLavaPlanner

# Automatically uses JAX (50-100x faster)
planner = EmpathicLavaPlanner(agent_i, agent_j, alpha=0.5)

# Falls back to NumPy if JAX not installed (with warning)
```

**Explicit control**:
```python
# Use JAX (default)
planner = EmpathicLavaPlanner(agent_i, agent_j, alpha=0.5, use_jax=True)

# Disable JAX (for debugging)
planner = EmpathicLavaPlanner(agent_i, agent_j, alpha=0.5, use_jax=False)
```

**Environment variables**:
```bash
# Disable JAX
export USE_JAX=0

# Force CPU (no GPU)
export JAX_FORCE_CPU=1

# Limit GPU memory to 50%
export JAX_MEMORY_FRACTION=0.5
```

### Installation

**Basic (NumPy only)**:
```bash
pip install -r requirements.txt
```

**With JAX (recommended for 50-130x speedup)**:
```bash
pip install -r requirements.txt
pip install jax  # CPU version

# OR for GPU support:
pip install jax[cuda12]  # CUDA 12
```

### Testing

```bash
# Quick smoke test
python test_jax_planner.py

# Full test suite
python run_all_tests.py
```

---

## 5. Running Experiments

### Main Experiment: Empathy Sweep

The primary experiment script is `scripts/run_empathy_sweep.py`, which tests empathy effects across layouts and agent configurations.

**Command-line options**:
```bash
python scripts/run_empathy_sweep.py [OPTIONS]

Options:
  --layouts LAYOUT [LAYOUT ...]   # Specific layouts (default: all)
  --mode {symmetric,asymmetric,both}
      symmetric   : Only α_i == α_j (e.g., 0.0/0.0, 0.5/0.5, 1.0/1.0)
      asymmetric  : Full grid of (α_i, α_j) combinations
      both        : Both sweeps (skips duplicates)
  --hierarchical  # Use hierarchical zone planner (faster, for bottleneck layouts)
  --verbose       # Print every timestep (for debugging)
  --seeds N       # Number of seeds (default: 1, env is deterministic)
  --horizon N     # Planning horizon (default: 4)
  --max-steps N   # Max timesteps per episode (default: 15)
```

**Recommended runs**:

```bash
# Quick test on hierarchical layouts (vertical_bottleneck, symmetric_bottleneck, narrow)
python scripts/run_empathy_sweep.py --layouts vertical_bottleneck --mode asymmetric --hierarchical

# Full hierarchical experiment (fastest)
python scripts/run_empathy_sweep.py --layouts vertical_bottleneck symmetric_bottleneck narrow --mode both --hierarchical

# All layouts (flat planner fallback for non-hierarchical layouts - slower)
python scripts/run_empathy_sweep.py --mode both --hierarchical

# Debug a specific case with verbose output
python scripts/run_empathy_sweep.py --layouts vertical_bottleneck --mode asymmetric --hierarchical --verbose
```

### Hierarchical vs Flat Planner

| Planner | Supported Layouts | Speed | Use Case |
|---------|-------------------|-------|----------|
| **Hierarchical** | `vertical_bottleneck`, `symmetric_bottleneck`, `narrow` | Fast (~0.1s/episode) | Bottleneck coordination |
| **Flat** | All layouts | Slow (~30-60s/episode) | General purpose |

The hierarchical planner decomposes the grid into **zones** and plans at two levels:
1. **High-level**: Zone transitions (STAY, FORWARD, BACK)
2. **Low-level**: Within-zone navigation to subgoals

This reduces complexity from O(5^H) to O(3^H × 5^h) where H=high-level horizon, h=low-level horizon.

### Output Files

Results are saved to `results/empathy_sweep_YYYYMMDD_HHMMSS.csv` with columns:

| Column | Description |
|--------|-------------|
| `layout`, `start_config` | Environment and starting positions |
| `alpha_i`, `alpha_j` | Empathy parameters for each agent |
| `both_success` | Both agents reached their goals without collision |
| `cell_collision`, `edge_collision` | Collision types |
| `paralysis` | Agents got stuck (cyclic behavior or mutual yielding) |
| `trajectory_i`, `trajectory_j` | Full position sequences |
| `G_i`, `G_j` | Accumulated expected free energy |

### Interpreting Results

**Key metrics**:

1. **Success rate** (`both_success`): Both agents reach goals without collision
2. **Collision rate** (`cell_collision | edge_collision`): Agents crashed
3. **Paralysis rate** (`paralysis`): Agents got stuck in loops or mutual yielding

**Expected patterns by empathy configuration**:

| α_i | α_j | Expected Outcome |
|-----|-----|------------------|
| 0.0 | 0.0 | **Collision** - Both selfish, rush forward, crash |
| 0.0 | 0.5+ | **Success** - Selfish i rushes, empathetic j yields |
| 0.5+ | 0.0 | **Success** - Empathetic i yields, selfish j rushes |
| 0.5+ | 0.5+ | **Paralysis** - "After you" deadlock, both too polite |

**Analyzing trajectories**:
- Look at `trajectory_i` and `trajectory_j` columns
- Yielding behavior: agent stays in place while other passes
- Example success: `i: (1,2)→(2,2)→(3,2)→...→goal`, `j: (4,5)→(4,5)→(4,5)→(3,5)→...→goal`
  - j waited (repeated position) while i passed through bottleneck

### Legacy Experiments

For flexibility-EFE correlation and flexibility-aware priors:

```bash
python experiments/exp1_flex_vs_efe.py   # Flexibility vs EFE correlation
python experiments/exp2_flex_prior.py    # Flexibility-aware prior experiments
```

---

## 6. Quick Start

### Installation

```bash
conda create -n alignment python=3.10
conda activate alignment
pip install -r requirements.txt
pip install jax  # Optional but recommended for 50-130x speedup
```

### Smoke Test

```bash
python smoke_test_tom.py
```

Expected output:
- ✅ TOM imports
- ✅ LavaModel / LavaAgent creation
- ✅ LavaV2Env reset + step
- ✅ Collision detection (cell + edge)

### Run Demos

**Single agent**:
```bash
python scripts/run_lava_si.py
```

**Empathy demo**:
```bash
python scripts/run_lava_empathy.py
```

**Empathy sweep (hierarchical planner)**:
```bash
python scripts/run_empathy_sweep.py --layouts vertical_bottleneck --mode asymmetric --hierarchical
```

**Full experiment sweep**:
```bash
python scripts/run_empathy_sweep.py --mode both --hierarchical
```

---

## 7. Key Features

### Edge Collision Detection

The system now properly detects **edge collisions** (swap collisions):
- When agent i moves A→B and agent j moves B→A simultaneously, both are blocked
- Edge collision is a separate observation modality with its own A matrix and C preferences
- Critical for crossed-goals scenarios where agents need to coordinate path timing

### Theory of Mind with Proper Conditioning

Agents model how their actions affect the other agent:
```python
# For each candidate action k:
    qs_i_next = propagate_belief(qs_i, action=k)  # i's predicted position
    G_j_best[k] = min_over_j_actions(G_j | qs_i_next)  # j's best response
    G_social[k] = G_i[k] + α * G_j_best[k]  # Combined EFE
```

Now empathy actually affects policy selection by predicting how i's actions impact j.

### Multi-Step Planning

With horizon H > 1, agents can plan sequences like:
- **Bottleneck detour**: "UP → RIGHT × 3 → DOWN" (requires H ≥ 4)
- **Crossed goals**: "WAIT × 2 → RIGHT × 5" (turn-taking, H ≥ 3)
- **Risk-reward trade-offs**: "Risky short path vs safe detour"

---

## 8. Key Findings

### Empirical Results

- **Empathy enables coordination** in spatially open environments (wide corridor)
- **Asymmetric empathy** (one empathic, one selfish) → successful coordination
- **Symmetric high empathy** (both α=1.0) → coordination deadlock (over-cooperation)
- **Edge collision detection** prevents ghosting artifacts in crossed-goals scenarios
- **Flexibility-aware priors** trade efficiency for robustness

### Theoretical Insights

- Collision avoidance in active inference requires proper observation modalities (cell + edge)
- Theory of Mind requires conditioning on predicted future states, not just current states
- Pure empathy (α=1.0) can lead to pathological coordination patterns (mirroring/deadlock)

(See `results/` and analysis notebooks for quantitative details)

---

## 9. Citation

If you use this codebase or its ideas, please cite:

```bibtex
@software{albarracin2025_pathflexibility,
  title={Path Flexibility, Empathy, and Theory of Mind in Active Inference},
  author={Mahault Albarracin},
  year={2025},
  url={https://github.com/mahault/Alignment-experiments}
}
```

---

## 10. Contact

Issues & discussions: https://github.com/mahault/Alignment-experiments/issues
