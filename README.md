# **Path Flexibility, Empathy, and Theory of Mind in Active Inference**

This repository implements a framework for studying **coordination, alignment, and robustness** in multi-agent systems through:

- **Active Inference & Expected Free Energy (EFE)**
- **Theory of Mind (ToM) planning**
- **Empathy-weighted decision-making**
- **Path flexibility metrics** (empowerment, returnability, overlap)

The central research goal is to test whether **alignment emerges naturally** when agents attempt to preserve each other’s future option sets — and whether a **flexibility-aware prior** improves cooperative behavior in challenging environments.

The project includes:  
1. A JAX-based generative model + environment (Lava Corridor)  
2. Empathy-aware and flexibility-aware planners  
3. Full experimental pipelines (Exp. 1 & 2)  
4. A complete automated test suite  

---

## **1. Conceptual Overview**

### **Active Inference**

Agents select policies by minimizing **expected free energy**:

\[
q(\pi) \propto \exp(-\gamma G(\pi))
\]

where \( G(\pi) \) combines information gain and preference satisfaction.

---

### **Path Flexibility**

Path flexibility measures how robust a future trajectory is using:

- **Empowerment** — how many future observations remain under agent control  
- **Returnability** — probability of reaching common safe outcomes  
- **Outcome overlap** — similarity of predicted future outcomes between agents  

\[
F(\pi) = \lambda_E E(\pi) + \lambda_R R(\pi) + \lambda_O O(\pi)
\]

High flexibility ⇒ agents preserve each other’s future option sets.

---

### **Theory of Mind (ToM)**

Agents maintain generative models of other agents’ beliefs and policies.  
During planning, each agent simulates the other’s EFE landscape.

---

### **Empathy**

Empathy parameter α ∈ [0,1] weights the other agent’s EFE:

\[
G_{\text{social}}^i(\pi) = G_i(\pi) + \alpha\, G_j(\pi)
\]

- α = 0 → purely selfish  
- α = 1 → fully prosocial  

---

### **Flexibility-Aware Policy Prior (Experiment 2)**

A policy prior biases agents toward flexible (robust) trajectories:

\[
p(\pi) \propto \exp\big(\kappa \left[F_i(\pi) + \beta F_j(\pi)\right]\big)
\]

In the combined objective:

\[
J_i(\pi)=G_i + \alpha G_j - \frac{\kappa}{\gamma}[F_i + \beta F_j]
\]

---

## **2. Repository Structure**

```text
Alignment-experiments/
│
├── src/
│   ├── tom/
│   │   ├── models/
│   │   │   └── model_lava.py           # LavaModel & LavaAgent (pure JAX)
│   │   ├── envs/
│   │   │   ├── env_lava.py             # Basic JAX Lava environment
│   │   │   ├── env_lava_v2.py          # Multi-layout, extended observations
│   │   │   └── env_lava_variants.py    # Layout definitions
│   │   ├── planning/
│   │   │   ├── si_lava.py              # Single-agent EFE planner
│   │   │   ├── si_empathy_lava.py      # Empathy-enabled multi-agent planner
│   │   │   ├── si_tom.py               # ToM inference functions
│   │   │   └── si_tom_F_prior.py       # Flexibility-aware ToM planner
│   │   └── __init__.py
│   │
│   ├── metrics/
│   │   ├── empowerment.py
│   │   └── path_flexibility.py         # Computes E, R, O, F metrics
│   │
│   ├── envs/
│   │   ├── lava_corridor.py            # PyMDP-compatible legacy environment
│   │   └── rollout_lava.py             # Multi-agent rollout logic
│   │
│   ├── agents/
│   │   └── tom_agent_factory.py        # Builds PyMDP ToM-ready agents
│   │
│   └── common/types.py                 # Shared type definitions
│
├── scripts/
│   ├── run_lava_si.py                  # Single-agent demo
│   ├── run_lava_empathy.py             # Basic two-agent empathy demo
│   └── run_empathy_experiments.py      # Full Experiment 1/2 sweeps
│
├── experiments/
│   ├── exp1_flex_vs_efe.py             # Measure F–EFE correlation
│   └── exp2_flex_prior.py              # Flexibility-aware prior experiments
│
├── tests/
│   ├── smoke_test.py                   # Legacy PyMDP smoke test
│   ├── smoke_test_tom.py               # TOM-style JAX smoke test
│   ├── test_lava_env_tom.py
│   ├── test_model_creation_tom.py
│   ├── test_integration_tom.py
│   ├── test_path_flexibility_metrics.py
│   └── test_F_aware_prior.py
│
├── notebooks/                          # Visualization & analysis
└── results/                            # Automatically generated experiment logs

## **3. Core Components**

### **LavaModel (Pure JAX Generative Model)**

- Transparent A, B, C, D matrices  
- Hard-coded lava corridor transitions  
- No PyMDP dependency  

### **LavaAgent**

- Multi-horizon policies  
- Exposes model dicts  
- Works with TOM planners  

### **LavaV2Env**

- Multi-layout environment  
- Agents observe both their own and the other agent's positions  
- Supports Wide, Bottleneck, Narrow, and Risk-Reward layouts  

### **EmpathicLavaPlanner**

Computes:

- Self EFE  
- Other-agent EFE via ToM  
- Social EFE using α  

### **Flexibility-Aware ToM Planner**

Adds:

- Empowerment along policy rollout  
- Returnability, overlap metrics  
- κ, β hyperparameters  

---

## **4. Experiments**

### **Experiment 1: Does flexibility emerge naturally?**

Conditions:
- Empathy α ∈ {0, 0.5, 1.0}
- No flexibility prior (κ = 0)

Outputs:
- F_joint vs G_joint correlations
- Collision rates
- Coordination behaviors
- Policy selection heatmaps

Run:

```bash
python experiments/exp1_flex_vs_efe.py

```

###  Experiment 2: Does a flexibility-aware prior improve coordination?
We fix the empathy parameter:

- **α = 0.5** (held constant)

We vary the **flexibility-prior strength**:

- **κ ∈ {0, 0.5, 1.0, 2.0}**

We vary the **flexibility-weighting parameter**:

- **β ∈ [0, 1]**  
  (controls how much each agent weights the *other agent’s* path flexibility in its Expected Free Energy)

---

### **Run the Experiment**

```bash
python experiments/exp2_flex_prior.py

```

### **5. How to Run the System**

#### **Install Dependencies**
```bash
conda create -n alignment python=3.10
conda activate alignment
pip install -r requirements.txt
```

Quick Verification

```bash
python smoke_test_tom.py
```

You should see:

✅ TOM imports

✅ LavaModel / LavaAgent creation

✅ LavaV1Env reset + step

✅ Manual Bayesian inference


### **Run Demos**

#### **Single agent:**
```bash
python scripts/run_lava_si.py
```

#### Empathy demo:
```bash
python scripts/run_lava_empathy.py

```
#### Full experiment sweep:
```bash
python scripts/run_empathy_experiments.py


```

### **6. Key Findings (High-Level)**

- **Empathy enables coordination** when the environment allows spatial separation.  
- **Bottlenecks require stronger empathy and/or flexibility priors** to support sequential coordination.  
- **Asymmetric empathy can lead to exploitation**, where altruistic agents defer and selfish agents exploit.  
- **Flexibility-aware priors reduce catastrophic failures**, trading efficiency for robustness and resilience.  

(These are qualitative; see `results/` and analysis notebooks for quantitative outcomes.)

---

### **7. Citation**

If you use this codebase or its ideas, please cite:

```bibtex
@software{albarracin2025_pathflexibility,
  title={Path Flexibility, Empathy, and Theory of Mind in Active Inference},
  author={Mahault Albarracin},
  year={2025},
  url={https://github.com/mahault/Alignment-experiments}
}
```

### Contact

Issues & discussions:
https://github.com/mahault/Alignment-experiments/issues
