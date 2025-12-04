# Path Flexibility, Empathy, and Theory of Mind in Active Inference

## Motivation & High-Level Idea

**Alignment need not be explicitly encoded in preference vectors.** Instead, we propose that alignment emerges naturally when agents maintain **mutual path flexibility**—the capacity to preserve each other's future option sets while pursuing their own goals.

In multi-agent active inference systems, agents minimize expected free energy (EFE) by balancing epistemic value (information gain) with pragmatic value (preference satisfaction). When combined with **Theory of Mind (ToM)**, agents form predictions about each other's beliefs, policies, and expected futures. When augmented with **empathy** (weighting others' EFE in their own decision-making), agents can coordinate without explicit reward sharing.

**The central question:** Does high joint path flexibility naturally correlate with low joint EFE? Or do we need an explicit flexibility-aware prior to guide agents toward mutually robust policies? This repository implements two experiments to test these hypotheses in a transparent, controlled multi-agent gridworld.

---

## Conceptual Building Blocks

### 1. Active Inference & Expected Free Energy (EFE)

Active inference agents don't maximize rewards—they minimize **surprise** about preferred observations. The core quantity is EFE:

```
G(π) = E_q(π)[KL[q(o_τ|π) || p(o_τ)] - log p(o_τ)]
```

- **Epistemic term**: Information gain (exploring uncertainty)
- **Pragmatic term**: Achieving preferred observations (encoded in prior C)

Policies are selected via: `q(π) = softmax(-γ G(π))`

### 2. Empowerment

**Empowerment** E is the channel capacity between actions and future observations:

```
E = max_p(a) I(A; O_future)
```

It measures **how many distinct, controllable futures** an agent can induce from a given state.

- **High empowerment** → wide corridor of possible futures; robust, multi-path behavior
- **Low empowerment** → bottlenecks, traps; small perturbations cause failure

### 3. Path Flexibility

We define **path flexibility** F(π) for a policy π as a weighted combination of:

```
F(π) = λ_E · E(π) + λ_R · R(π) + λ_O · O(π)
```

Where:

- **E(π)**: Expected empowerment along the trajectory
- **R(π)**: Returnability—probability of reaching shared "safe" outcome sets
- **O(π)**: Outcome overlap—how much agents' predicted outcome distributions overlap

**In multi-agent settings**, path flexibility becomes **relational**:
- If Agent A's actions constrain Agent B's movements, B's empowerment drops
- If A preserves B's option space, both maintain high flexibility
- **Relational collapse of empowerment** = interaction-induced brittleness

### 4. Theory of Mind (ToM)

Agents maintain nested generative models of each other. Using ToM tree search:

- Agent i simulates its own future rollout under each candidate policy
- Agent i also simulates Agent j's future rollout (including j's beliefs and policies)
- This allows Agent i to evaluate:
  - Its own EFE: G_i(π)
  - Agent j's EFE: G_j(π)
  - Its own empowerment: E_i(π)
  - Agent j's empowerment: E_j(π)

### 5. Empathy

Agent i weights Agent j's EFE using empathy parameter α ∈ [0,1]:

```
G_social^i(π) = G_i(π) + α · G_j(π)
```

- α = 0: Purely selfish
- α > 0: Partially or fully prosocial

---

## Experiment 1: Hypothesis Test (Flexibility ↔ Joint EFE?)

### Goal
Test whether **high joint path flexibility naturally correlates with low joint EFE**, even when agents don't explicitly optimize for flexibility.

### Setup

**Environment**: Two-agent lava corridor gridworld
- Narrow, risky bottleneck in the center
- Wider, safe detour path
- Stochastic slips (5% chance)
- Collision = catastrophe
- Individual goals (no shared rewards)

**Agents**:
- Full ToM (depth-1 nested belief tracking)
- Empathy parameter α (weight on other's EFE)
- **No explicit flexibility term in decision rule**

**Decision rule**:
```
q(π) = softmax(-γ [G_i(π) + α G_j(π)])
```

### What We Log

For each candidate policy π during planning:
- **EFE**: G_i(π), G_j(π), G_joint = G_i + G_j
- **Path flexibility**: F_i(π), F_j(π), F_joint = F_i + F_j
  - Empowerment E(π)
  - Returnability R(π)
  - Overlap O(π)

### Analysis
- **Correlation plots**: F_joint vs G_joint across all considered policies
- **Selected policies**: Do agents naturally choose high-F policies when minimizing G?
- **Behavioral outcomes**: Collision rates, bottleneck usage, goal achievement

**Hypothesis**: If agents with ToM + empathy naturally avoid low-flexibility policies (bottlenecks, deadlocks), we should see **negative correlation** between F and G.

---

## Experiment 2: Path-Flexibility-Aware Policy Prior

### Goal
Explicitly add a **flexibility-aware prior** over policies and compare behavior to Experiment 1.

### Setup

Same environment, same ToM + empathy core, but now:

**Policy prior**:
```
p(π) ∝ exp(κ [F_i(π) + β F_j(π)])
```

Where:
- κ: Strength of flexibility preference
- β: How much agent i cares about agent j's flexibility

**Decision rule** (combined objective):
```
J_i(π) = G_i(π) + α G_j(π) - (κ/γ)[F_i(π) + β F_j(π)]
q(π) = softmax(-γ J_i(π))
```

### Manipulations

Sweep over κ ∈ [0, 0.5, 1.0, 2.0]:
- κ = 0: Reduces to Experiment 1 (no F-prior)
- κ > 0: Increasing preference for flexible policies

### Analysis
- **Behavioral changes**: As κ increases:
  - Do agents prefer detours over bottlenecks?
  - Do collision rates drop?
  - Do agents sacrifice pragmatic value (slower goal achievement) for safety?
- **F vs EFE trade-off**: Plot F(π_chosen) vs G(π_chosen) as function of κ
- **Resilience**: Introduce layout perturbations mid-episode—do high-κ agents recover better?

**Hypothesis**: Explicit F-prior should:
1. Push agents toward mutually safe, flexible policies
2. Reduce catastrophic failures (lava, collisions)
3. Trade off pragmatic efficiency for robustness

---

## Code Architecture

```
Alignment-experiments/
├── README.md                    # This file
├── requirements.txt
├── pyproject.toml              # Optional packaging config
│
├── src/
│   ├── agents/
│   │   ├── empathetic_agent.py    # Empathy-weighted EFE agent (legacy/baseline)
│   │   └── __init__.py
│   │
│   ├── tom/
│   │   ├── si_tom.py              # ToM tree search & policy evaluation
│   │   ├── rollout_tom.py         # Multi-agent rollout with ToM
│   │   └── __init__.py
│   │
│   ├── envs/
│   │   ├── lava_corridor.py       # Two-agent gridworld environment
│   │   └── __init__.py
│   │
│   ├── metrics/
│   │   ├── empowerment.py         # Empowerment computation (MI estimation)
│   │   ├── path_flexibility.py    # F(π) = λE·E + λR·R + λO·O
│   │   └── __init__.py
│   │
│   └── common/
│       ├── types.py               # Shared dataclasses/types
│       └── __init__.py
│
├── experiments/
│   ├── exp1_flex_vs_efe.py        # Experiment 1: Measure F-EFE correlation
│   ├── exp2_flex_prior.py         # Experiment 2: F-aware policy prior
│   └── utils_plotting.py          # Shared plotting utilities
│
├── notebooks/
│   ├── sanity_check_single_agent.ipynb
│   ├── visualize_lava_corridor.ipynb
│   └── analysis_flex_vs_efe.ipynb
│
└── results/                       # Saved metrics, plots, logs
    ├── exp1/
    └── exp2/
```

### Key Modules

**`tom/si_tom.py`**
- `run_tom_step()`: Theory of Mind inference for all K agents
- Returns: `tom_results`, `EFE_arr [K, num_policies]`, `Emp_arr [K, num_policies]`
- Handles state inference, policy inference, belief updates, learning

**`tom/rollout_tom.py`**
- `rollout()`: Multi-agent environment interaction loop
- Integrates ToM tree search at each timestep
- Logs trees, beliefs, policies, metrics

**`metrics/path_flexibility.py`**
- `compute_empowerment_along_path()`: Average empowerment over policy trajectory
- `compute_returnability()`: Probability of reaching shared safe outcomes
- `compute_overlap()`: Outcome distribution overlap between agents
- `compute_path_flexibility()`: Combined F(π) metric

**`envs/lava_corridor.py`**
- Multi-agent gridworld with:
  - Narrow bottleneck (high risk, low flexibility)
  - Wide detour (safe, high flexibility)
  - Stochastic slips
  - Collision detection
  - Individual goal states

---

## How to Run

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Alignment-experiments.git
cd Alignment-experiments

# Create conda environment
conda create -n alignment python=3.10
conda activate alignment

# Install dependencies
pip install -r requirements.txt
```

### Run Experiment 1

```bash
python experiments/exp1_flex_vs_efe.py \
  --num_episodes 100 \
  --alpha 0.5 \
  --gamma 16.0 \
  --output_dir results/exp1
```

**Output**:
- `results/exp1/metrics.pkl` - Raw policy metrics (G, F per policy per episode)
- `results/exp1/correlation_plot.png` - F vs EFE scatter plot
- `results/exp1/behavioral_stats.json` - Collision rates, goal achievement, etc.

### Run Experiment 2

```bash
python experiments/exp2_flex_prior.py \
  --num_episodes 100 \
  --alpha 0.5 \
  --kappa_values 0.0 0.5 1.0 2.0 \
  --beta 1.0 \
  --gamma 16.0 \
  --output_dir results/exp2
```

**Output**:
- `results/exp2/metrics_kappa_{k}.pkl` - Metrics per κ value
- `results/exp2/comparison_plots.png` - Behavior across κ values
- `results/exp2/resilience_test.json` - Performance under perturbations

### Analysis Notebooks

```bash
jupyter notebook notebooks/analysis_flex_vs_efe.ipynb
```

Generates:
- Correlation plots (F vs EFE)
- Policy selection heatmaps
- Trajectory visualizations
- Statistical tests (Pearson correlation, t-tests)

---

## Integration & Implementation Notes

### Key Integration Points

The experiment scaffolds are complete, but several integration points need implementation:

#### 1. `compute_path_flexibility_for_tree()` (CRITICAL PATH)

**Location**: `src/metrics/path_flexibility.py:230`

Currently a STUB. This function bridges the ToM tree output with the flexibility metrics:

```python
def compute_path_flexibility_for_tree(focal_tree, other_tree, ...):
    # TODO: Extract policies from ToM tree structure
    # TODO: Extract G_i(π), G_j(π) from tree
    # TODO: For each policy, compute E, R, O components
    # TODO: Return List[PolicyMetrics]
```

**Next steps**:
- Inspect actual ToM tree structure from `tom/si_tom.py`
- Extract EFE values (currently returned as `EFE_arr [K, num_policies]`)
- Map tree policies to agent models for E, R, O computation

#### 2. ToM Rollout Function

**Location**: `src/tom/rollout_tom.py` (needs creation)

Multi-agent interaction loop that:
- Calls `run_tom_step()` each timestep
- Steps environment
- Logs trees, beliefs, metrics
- Returns episode info with collision/success flags

#### 3. F-Prior Integration (Experiment 2)

**Location**: `tom/si_tom.py` (modify policy search)

For Exp 2, policy selection must use:
```
J_i(π) = G_i(π) + α·G_j(π) - (κ/γ)[F_i(π) + β·F_j(π)]
q(π) = softmax(-γ J_i(π))
```

Requires:
- Accepting κ, β parameters in `run_tom_step()`
- Computing F for each candidate policy during tree search
- Updating policy sampling to use J_i instead of G_i

#### 4. Environment Completion

**Location**: `src/envs/lava_corridor.py`

Needs `shared_outcomes()` method:
```python
def shared_outcomes(self) -> List[int]:
    """Return indices of 'safe' outcomes (not lava, not walls)."""
```

### Data Flow

```
Experiment → rollout() → run_tom_step() → policy search (with optional F-prior)
                  ↓
         Extract trees from final timestep
                  ↓
    compute_path_flexibility_for_tree()
                  ↓
         Aggregate metrics & analyze
```

---

## Roadmap / TODO

### Near-term (Current Focus)
- [x] Refactor ToM into standalone `tom/si_tom.py` module
- [x] Add comprehensive logging to all modules
- [x] Reorganize repository structure (src/, experiments/, notebooks/)
- [x] Implement `metrics/path_flexibility.py` (E, R, O computations)
- [x] Create shared type system (src/common/types.py)
- [x] Write `experiments/exp1_flex_vs_efe.py` script
- [x] Write `experiments/exp2_flex_prior.py` script
- [ ] Implement `compute_path_flexibility_for_tree()` (currently STUB in path_flexibility.py)
- [ ] Complete `envs/lava_corridor.py` (add shared_outcomes() method)
- [ ] Wire α, κ, β parameters into si_policy_search_tom for Exp 2
- [ ] Connect actual ToM rollout to experiments (replace STUBs)
- [ ] Test full pipeline end-to-end

### Mid-term (Extensions)
- [ ] Multi-agent (K > 2) experiments
- [ ] More complex environments (partially observable, larger state spaces)
- [ ] Connect to geodesic metrics from "Belief Geodesics" framework
- [ ] Preference plasticity (agents learn C over time)
- [ ] Power/precision asymmetry experiments

### Long-term (Research Directions)
- [ ] Hierarchical ToM (depth > 1 nested beliefs)
- [ ] Continuous action/state spaces
- [ ] Language-grounded communication protocols
- [ ] Connection to game-theoretic equilibria
- [ ] Formal proofs of alignment under flexibility constraints

---

## Citation

If you use this code or build on these ideas, please cite:

```bibtex
@software{path_flexibility_tom_2025,
  title={Path Flexibility, Empathy, and Theory of Mind in Active Inference},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Alignment-experiments}
}
```

---

## License

MIT License - See LICENSE file for details.

---

## Contact

For questions, suggestions, or collaborations:
- GitHub Issues: [https://github.com/yourusername/Alignment-experiments/issues](https://github.com/yourusername/Alignment-experiments/issues)
