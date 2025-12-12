# Empathy & Coordination Experiment Suite - Roadmap

**Created:** 2025-12-11
**Purpose:** Systematic investigation of empathy's effects on multi-agent coordination

---

## 1. Research Questions

### 1.1 Core Hypotheses

1. **H1 - Empathy Improves Coordination**: Cooperation (both agents reach goals) increases with empathy parameter alpha, especially in layouts allowing spatial separation.

2. **H2 - Asymmetry Effects**: In (alpha_i, alpha_j) space, there exist regions where:
   - The altruist is exploited (one-sided sacrifice)
   - Coordination works despite asymmetry
   - System gets stuck in "paralysis" (mutual deference / oscillations)

3. **H3 - Complexity Interaction**: More constrained layouts require higher empathy levels to achieve coordination, showing sharper phase transitions.

4. **H4 - Phase Diagrams**: We can identify distinct behavioral regimes (collision, cooperation, paralysis) as functions of empathy parameters.

---

## 2. Experimental Factors

### 2.1 Layout Types

| Tag | Description | Conflict Type |
|-----|-------------|---------------|
| `narrow` | Single-file corridor, 1 cell wide | Forced collision |
| `wide` | 2+ rows, agents can pass | Optional coordination |
| `bottleneck` | Wide areas connected by 1-cell choke | Sequential coordination |
| `crossed_goals` | Agents must cross paths | Timing coordination |
| `risk_reward` | Short risky vs long safe path | Risk/empathy tradeoff |
| `double_bottleneck` | Two bottlenecks with passing bay between | Multi-stage coordination |
| `passing_bay` | Mostly 1-cell with one 2x2 bay | Pull-aside coordination |
| `asymmetric_detour` | One agent has shorter path to bottleneck | Fairness vs efficiency |

### 2.2 Empathy Parameters

**Symmetric sweeps:**
```python
alphas_symmetric = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# alpha_i = alpha_j = alpha
```

**Asymmetric grid:**
```python
alphas_asymmetric = [0.0, 0.25, 0.5, 0.75, 1.0]
# All combinations of (alpha_i, alpha_j)
```

### 2.3 Start Configuration (Role Asymmetry)

For each layout, define two start configurations:
- **Config A**: Agent 0 on "advantaged" side (shorter path, first to bottleneck)
- **Config B**: Swapped positions

This tests whether empathic behavior depends on spatial advantage.

### 2.4 Seeds

- Minimum 30 seeds per configuration for statistical power
- Recommended: 50 seeds for publication-quality results

---

## 3. Metrics

### 3.1 Success Metrics

| Metric | Description |
|--------|-------------|
| `both_success` | Both agents reach goals without collision/lava |
| `single_success` | Exactly one agent reaches goal |
| `failure` | Neither agent reaches goal |
| `goal_reached_i/j` | Individual goal achievement |

### 3.2 Collision Metrics

| Metric | Description |
|--------|-------------|
| `lava_collision` | Any agent hits lava |
| `cell_collision` | Both occupy same cell |
| `edge_collision` | Agents swap positions (cross paths) |
| `collision_timestep` | When first collision occurred |

### 3.3 Paralysis Detection

**Definition:** Episode ends at T_max without success and without fatal collision.

**Detection criteria (any of):**
1. Episode reaches T_max with `both_success == False` and no lava/collision
2. State repetition: same (pos_i, pos_j) seen > K times (K=3)
3. Oscillation: alternating between 2-3 states without goal progress
4. Mutual stay: both agents choose STAY action for > M consecutive steps (M=3)

| Metric | Description |
|--------|-------------|
| `paralysis` | Boolean flag for paralysis detection |
| `paralysis_type` | "timeout" / "cycle" / "oscillation" / "mutual_stay" |
| `cycle_length` | Length of detected cycle if applicable |

### 3.4 Efficiency Metrics

| Metric | Description |
|--------|-------------|
| `steps_to_goal_i/j` | Timesteps for each agent to reach goal (or max) |
| `arrival_order` | Which agent arrived first (0, 1, or "tie") |
| `arrival_gap` | steps_to_goal_j - steps_to_goal_i (signed) |
| `efficiency` | (optimal_steps) / (actual_steps) |

### 3.5 Fairness Metrics

| Metric | Description |
|--------|-------------|
| `sacrifice_i` | Extra steps agent i took compared to selfish baseline |
| `sacrifice_j` | Extra steps agent j took compared to selfish baseline |
| `exploitation` | `both_success` AND high `arrival_gap` AND altruist arrives later |

### 3.6 Internal Metrics (Optional)

| Metric | Description |
|--------|-------------|
| `G_i_cumulative` | Total EFE for agent i over episode |
| `G_j_cumulative` | Total EFE for agent j over episode |
| `G_social_cumulative` | Total social EFE over episode |

---

## 4. Output Format

### 4.1 CSV Structure

File: `results/empathy_sweeps_{timestamp}.csv`

Columns:
```
layout, start_config, alpha_i, alpha_j, seed,
both_success, single_success, failure,
goal_reached_i, goal_reached_j,
lava_collision, cell_collision, edge_collision,
paralysis, paralysis_type,
steps_i, steps_j, timesteps,
arrival_order, arrival_gap,
sacrifice_i, sacrifice_j,
trajectory_i, trajectory_j,
G_i, G_j
```

### 4.2 Summary Statistics

Per (layout, start_config, alpha_i, alpha_j):
- P(both_success): Mean of `both_success`
- P(collision): Mean of any collision
- P(paralysis): Mean of `paralysis`
- Mean efficiency
- Mean arrival_gap

---

## 5. Visualizations

### 5.1 Cooperation vs Empathy (1D - Symmetric)

For each layout:
- X-axis: alpha (symmetric: alpha_i = alpha_j)
- Y-axis: P(both_success)
- Optional overlays: P(paralysis), P(collision)

Shows whether empathy smoothly improves coordination or shows phase transition.

### 5.2 Asymmetry Heatmaps (2D)

For each layout and start_config:
- X-axis: alpha_i
- Y-axis: alpha_j
- Color: P(both_success) or P(paralysis) or P(exploitation)

Reveals:
- Cooperation region
- Exploitation region (one agent takes advantage)
- Paralysis region (too deferential)

### 5.3 Layout Complexity Plot

- X-axis: Layout complexity index (num bottlenecks, constraint level)
- Y-axis: Critical alpha* where P(both_success) > 0.8 (symmetric case)

Shows how much empathy is needed as environment gets harder.

### 5.4 Paralysis Curves

For constrained layouts:
- X-axis: alpha (symmetric)
- Y1: P(both_success)
- Y2: P(paralysis)

Watch for "too much empathy" regime.

### 5.5 Role Effect Comparison

For asymmetric layouts:
- Compare Config A vs Config B at same (alpha_i, alpha_j)
- Shows whether spatial advantage affects empathy dynamics

---

## 6. Implementation Plan

### Phase 1: Infrastructure (COMPLETE)

- [x] Document roadmap (this file)
- [x] Add new layouts: `double_bottleneck`, `passing_bay`, `asymmetric_detour`, `t_junction`
- [x] Add `start_config` support to layouts (A/B variants via `swap_agents()`)
- [x] Add paralysis detection function (`src/metrics/paralysis_detection.py`)
- [x] Add layout complexity indices

### Phase 2: Experiment Runner (COMPLETE)

- [x] Create `scripts/run_empathy_sweep.py` for full sweep
- [x] Add all new metrics collection (paralysis, efficiency, fairness)
- [x] Add progress logging with tqdm (falls back gracefully)
- [x] Add verbose debug output showing empathy effects
- [x] Verify JAX acceleration (`use_jax=True` default in planner)
- [x] CSV output with all metrics

### Phase 3: Analysis & Plotting (COMPLETE)

- [x] Create `analysis/plot_empathy_sweeps.py`
- [x] Implement 1D cooperation curves
- [x] Implement 2D heatmaps (success, paralysis, collision)
- [x] Implement layout complexity plot
- [x] Implement paralysis phase diagrams
- [x] Implement exploitation analysis
- [x] Implement role effect comparisons (Config A vs B)

### Phase 4: Execution & Results (IN PROGRESS)

- [ ] Run full experiment suite
- [ ] Generate all plots
- [ ] Document findings

---

## 7. Code Locations

| Component | File |
|-----------|------|
| Layout definitions | `tom/envs/env_lava_variants.py` |
| Environment | `tom/envs/env_lava_v2.py` |
| Empathic planner | `tom/planning/si_empathy_lava.py` |
| JAX planner | `tom/planning/jax_si_empathy_lava.py` |
| Paralysis detection | `src/metrics/paralysis_detection.py` |
| Experiment runner (sweep) | `scripts/run_empathy_sweep.py` |
| Experiment runner (original) | `scripts/run_empathy_experiments.py` |
| Analysis plots | `analysis/plot_empathy_sweeps.py` |
| Tests for new layouts | `tests/test_new_layouts.py` |
| Results output | `results/empathy_sweep_*.csv` |

---

## 8. Configuration Reference

### Default Parameters

```python
# Planning
HORIZON = 3
GAMMA = 16.0  # Inverse temperature
MAX_TIMESTEPS = 25

# Sweep parameters
LAYOUTS = [
    "narrow", "wide", "bottleneck", "crossed_goals",
    "risk_reward", "double_bottleneck", "passing_bay", "asymmetric_detour"
]
ALPHAS_SYMMETRIC = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ALPHAS_ASYMMETRIC = [0.0, 0.25, 0.5, 0.75, 1.0]
START_CONFIGS = ["A", "B"]
NUM_SEEDS = 50

# Paralysis detection
PARALYSIS_CYCLE_THRESHOLD = 3  # Same state seen K times
PARALYSIS_STAY_THRESHOLD = 3  # Both stay for M steps
```

### Estimated Runtime

- Per episode: ~0.1-0.5s (with JAX)
- Total configurations: 8 layouts × 2 configs × 5 × 5 alphas × 50 seeds = 20,000 episodes
- Estimated total: ~30-60 minutes with JAX

---

## 9. Recovery Instructions

If Claude crashes mid-session, resume by:

1. Check `results/` for any partial output files
2. Review this ROADMAP.md for current phase
3. Check git status for uncommitted changes
4. Continue from the appropriate phase

Key files to check:
- `tom/envs/env_lava_variants.py` - for new layout definitions
- `scripts/run_empathy_experiments.py` - for experiment runner updates
- `analysis/plot_empathy_sweeps.py` - for plotting code

---

## 10. Success Criteria

The experiment suite is complete when:

1. All 8 layouts implemented and tested
2. Full parameter sweep executed (20,000+ episodes)
3. CSV results saved with all metrics
4. All 5 visualization types generated
5. Clear phase diagrams showing cooperation/paralysis regions
6. Documented findings on empathy-coordination relationship
