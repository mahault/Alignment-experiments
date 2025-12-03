# Refactoring Notes

## Theory of Mind (ToM) Extraction

### What We Did

We extracted the Theory of Mind logic from `EmpatheticAgent` into a standalone module to:
1. Make the code more modular and testable
2. Prepare for real empowerment computation
3. Improve code reusability

### Changes Made

#### 1. Created `tom/si_tom.py`
- **`run_tom_step()`**: Main function that runs ToM for all K agents
  - Returns: `tom_results`, `EFE_arr`, `Emp_arr`
  - Handles state inference, policy inference, and action sampling for all agents
  - Includes comprehensive logging at DEBUG level

- **`_update_B_with_learning()`**: Handles B-matrix learning
  - Updates self agent's B-matrix with previous observations
  - Infers other agents' actions and updates their B-matrices

- **`_infer_others_action()`**: Heuristic for inferring others' actions
  - Currently implements PD-style (Prisoner's Dilemma) heuristic
  - Can be replaced with more sophisticated inference later

#### 2. Updated `agents/empathetic_agent.py`
- **Imports**: Added `from tom.si_tom import run_tom_step`
- **`step()` method**: Simplified to use `run_tom_step()`
  ```python
  # Old (multiple calls):
  tom_results = self._theory_of_mind(...)
  EFE_arr = self._extract_tom_EFE(...)
  Emp_arr = self._compute_empowerment_matrix(...)

  # New (single call):
  tom_results, EFE_arr, Emp_arr = run_tom_step(
      agents=self.agents,
      o=o,
      qs_prev=self.qs_prev,
      t=t,
      learn=self.learn,
      agent_num=self.agent_num,
      B_self=self.B,
  )
  ```

- **Deprecated methods**: Marked old methods as DEPRECATED
  - `_theory_of_mind()` → replaced by `run_tom_step()`
  - `_learn()` → logic in `_update_B_with_learning()`
  - `infer_others_action()` → logic in `_infer_others_action()`
  - `_extract_tom_EFE()` → EFE_arr now returned directly
  - `_compute_empowerment_matrix()` → Emp_arr now returned directly

### Benefits

1. **Modularity**: ToM logic is now self-contained
2. **Testability**: Can unit test ToM independently
3. **Clarity**: Clear separation of concerns
4. **Extensibility**: Easy to plug in real empowerment computation
5. **Logging**: Comprehensive debug logging for troubleshooting

### Next Steps

#### Option 1: Wire in Real Empowerment (Recommended)
Update `run_tom_step()` in `tom/si_tom.py` to compute actual empowerment:
```python
# Inside run_tom_step(), replace:
Emp_arr = np.zeros((K, num_policies))  # placeholder

# With real empowerment computation using agents[k].A, agents[k].B, etc.
from agents.empowerment import estimate_empowerment_one_step

for k in range(K):
    for policy_idx in range(num_policies):
        # Build transition_logits from A/B given this policy
        # Then compute empowerment
        Emp_arr[k, policy_idx] = estimate_empowerment_one_step(transition_logits)
```

#### Option 2: Test Current Implementation
Run existing experiments to verify the refactoring didn't break anything:
```bash
cd experiments
python sim.py  # or whatever your entry point is
```

#### Option 3: Add Unit Tests
Create tests for `tom/si_tom.py`:
```python
# tests/test_tom.py
def test_run_tom_step():
    # Setup mock agents, observations
    # Call run_tom_step()
    # Assert correct shapes and values
    pass
```

### Architecture Overview

```
EmpatheticAgent.step()
    │
    ├─> run_tom_step()  [tom/si_tom.py]
    │   ├─> For each agent k:
    │   │   ├─> infer_states()
    │   │   ├─> _update_B_with_learning()  [if learning enabled]
    │   │   ├─> infer_policies()
    │   │   └─> sample_action()
    │   │
    │   └─> Returns: tom_results, EFE_arr, Emp_arr
    │
    ├─> _expected_value_EFE(EFE_arr, Emp_arr)
    │   └─> Combines EFE and Empowerment with empathy weighting
    │
    └─> softmax(-exp_EFE) → sample action
```

### Files Modified

1. **Created**:
   - `tom/si_tom.py` - New ToM module with logging

2. **Updated**:
   - `agents/empathetic_agent.py` - Uses new `run_tom_step()`
   - All files now have comprehensive logging

3. **Deprecated** (but kept for reference):
   - `EmpatheticAgent._theory_of_mind()`
   - `EmpatheticAgent._learn()`
   - `EmpatheticAgent.infer_others_action()`
   - `EmpatheticAgent._extract_tom_EFE()`
   - `EmpatheticAgent._compute_empowerment_matrix()`
