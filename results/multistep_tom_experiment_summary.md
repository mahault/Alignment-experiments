# Multi-Step Theory of Mind Experiment Summary

## Date: 2025-12-17

## Problem Statement
In simultaneous play with single-step planning, both agents choose STAY when adjacent and facing each other (paralysis). This happens because:
1. Moving toward each other causes collision (penalty)
2. Neither agent has incentive to yield because the other is also STAYing
3. ToM correctly predicts the other will STAY, so there's no reason to yield

## Solution: Multi-Step Planning (horizon=3)

### What We Changed
- Extended planning horizon from 1 step to 3 steps
- For each first action, compute cumulative EFE over the full horizon
- Use greedy rollout for steps 2 and 3

### Key Functions Added
1. `compute_G_multistep()`: Computes EFE for each first action over multi-step horizon
2. `compute_G_empathic_multistep()`: Simulates joint trajectory for empathic planning
3. Updated `predict_other_action()` to use multi-step planning

### Why Multi-Step Planning Works
With single-step planning:
- G_self(STAY) ≈ G_self(MOVE) when other is STAYing (no immediate collision)
- No reason to move or yield

With multi-step planning:
- Agent sees that moving NOW enables reaching goal in future steps
- Total 3-step G for moving toward goal >> Total G for staying
- Distance shaping accumulates over multiple steps

---

## Experiment 1: Default collision penalty (-30)

### Test Output (horizon=3, collision=-30)
```
Both selfish (alpha=0):        i=RIGHT, j=LEFT
i empathic, j selfish:         i=RIGHT, j=LEFT
i selfish, j empathic:         i=RIGHT, j=LEFT

ToM ACCURACY: All predictions CORRECT
```

### Analysis
1. **Paralysis BROKEN**: Agents now choose to move (RIGHT/LEFT) instead of STAY
2. **ToM Predictions CORRECT**: Multi-step prediction matches actual behavior
3. **New Issue - COLLISION**: Both agents move toward each other

---

## Experiment 2: Increased collision penalty (-50)

### Test Output (horizon=3, collision=-50)
```
Both selfish (alpha=0):        i=RIGHT, j=LEFT
i empathic, j selfish:         i=RIGHT, j=LEFT
i selfish, j empathic:         i=RIGHT, j=LEFT
```

### Analysis
Still same behavior - collision penalty not high enough. G_self(RIGHT) = -23.8 still beats G_self(STAY) = 18.3.

---

## Experiment 3: High collision penalty (-100)

### Test Output (horizon=3, collision=-100)
```
Both selfish (alpha=0):        i=STAY, j=STAY  (PARALYSIS!)
i empathic, j selfish:         i=LEFT (yield), j=STAY
i selfish, j empathic:         i=STAY, j=RIGHT (yield)

ToM ACCURACY: All predictions CORRECT
```

### Key Observations
1. **Selfish agents now paralyzed**: G_self(move) = 26.2 > G_self(STAY) = 18.3
2. **Empathic agents yield**: Move away from conflict (LEFT/RIGHT away from goal)
3. **But selfish agents don't capitalize**: They just STAY instead of moving forward

### Critical Insight: ToM Prediction Not Used in Planning!

This reveals a **BUG** in the implementation:

**Case 2 (j selfish, i empathic):**
- j correctly predicts: "i will move LEFT"
- If i moves LEFT to (1,1), j could safely move LEFT toward goal (no collision)
- BUT j's G_self computation still uses **i's CURRENT position (2,1)**, not predicted future position
- So j sees phantom collision risk and chooses STAY

**The Problem:**
The ToM prediction is computed correctly, but then it's **not used** when computing the agent's own EFE!
- `predict_other_action()` returns the predicted action
- But `compute_G_multistep()` uses `qs_other` (current position), not the predicted future position

**This is why the selfish agent doesn't move forward even when it correctly predicts the empathic agent will yield!**

---

## Fix Plan: Use ToM Prediction in EFE Computation

### Current Flow (Buggy)
```
1. predict_other_action() → returns predicted_action, G_other
2. compute_G_multistep(qs_self, qs_other)  # qs_other = CURRENT position
3. Collision checked against other's CURRENT position (wrong!)
```

### Fixed Flow
```
1. predict_other_action() → returns predicted_action, G_other
2. Compute qs_other_predicted = propagate(qs_other, predicted_action)
3. compute_G_multistep(qs_self, qs_other_predicted)  # Use PREDICTED position
4. Collision checked against other's PREDICTED position (correct!)
```

### Implementation Changes

**In `plan_with_empathy()`:**
1. After getting `action_other_predicted`, compute other's predicted next position
2. Pass this predicted position to `compute_G_multistep()` for step 0 collision checking
3. For subsequent steps, continue using greedy rollout

**Key Insight:**
- Step 0: Both agents act simultaneously without knowing other's action
- But agent CAN predict other's action using ToM
- So collision check at step 0 should use: (my_next_pos, other_predicted_next_pos)
- NOT: (my_next_pos, other_current_pos)

### Expected Outcome
With collision=-100:
- Selfish j predicts empathic i will yield (LEFT)
- j computes collision against i's predicted position (1,1), not current (2,1)
- j sees path is clear → j moves LEFT toward goal
- Result: i=LEFT (yield), j=LEFT (progress) → Coordination achieved!

---

## Experiment 4: With ToM Prediction Fix (collision=-100)

### Test Output
```
Both selfish (alpha=0):        i=STAY, j=STAY  (paralysis)
i empathic, j selfish:         i=LEFT, j=LEFT  (coordination!)
i selfish, j empathic:         i=RIGHT, j=RIGHT  (coordination!)
```

### Analysis: Coordination Achieved!

**Case 2 (i empathic, j selfish):**
- i at (2,1) moves LEFT to (1,1) = YIELDS (away from goal)
- j at (3,1) moves LEFT to (2,1) = ADVANCES (toward goal)
- **Successful coordination!** Empathic agent yielded, selfish agent passed.

**Case 3 (i selfish, j empathic):**
- i at (2,1) moves RIGHT to (3,1) = ADVANCES (toward goal)
- j at (3,1) moves RIGHT to (4,1) = YIELDS (away from goal)
- **Successful coordination!** Empathic agent yielded, selfish agent passed.

### ToM Prediction Accuracy Issue

Some predictions are "wrong" but coordination still works:
- i (empathic) predicted j=STAY, but j=LEFT
- j (empathic) predicted i=STAY, but i=RIGHT

**Why predictions are wrong:**
There's a circular dependency / level-of-reasoning mismatch:
1. i predicts j assuming j doesn't know i will yield
2. But j NOW uses ToM to predict i WILL yield
3. So j moves (not stays), making i's prediction "wrong"

**Why coordination still works:**
Both agents independently:
1. Use ToM to predict the other's action
2. Compute their OWN action using predicted other position
3. The selfish agent sees "other will yield → path clear → I advance"
4. The empathic agent sees "other won't yield → I should yield"

The asymmetry in alpha creates the symmetry breaking:
- Empathic agent: considers joint harm → yields
- Selfish agent: only considers own benefit → advances

### Key Insight
The ToM prediction being "wrong" at level-1 doesn't prevent coordination!
What matters is that both agents use the same reasoning framework,
and the empathy asymmetry creates different incentives that break symmetry.

---

## Experiment 5: Symmetric Agent Architecture (FINAL)

### Key Principle
**Both agents are IDENTICAL except for alpha.** The planning mechanism is:
1. Predict other's action using recursive ToM (depth=2)
2. Use other's predicted position for collision checking
3. Compute G_social = G_self + alpha * G_other
4. Choose action minimizing G_social

The ONLY difference between agents is their alpha value.

### Code Architecture

```python
def predict_other_action_recursive(model_other, model_self, qs_other, qs_self,
                                    alpha_other, alpha_self, depth=2):
    """
    Recursive ToM prediction. Both agents use this SAME function.

    depth=0: Base case, assume opponent stays in place
    depth=1: Predict opponent assuming they use depth=0
    depth=2: Predict opponent assuming they use depth=1
    """
    if depth == 0:
        # Base case: assume opponent stays, compute G_social with their alpha
        G_other_self, G_other_social = compute_G_empathic_multistep(
            model_other, model_self, qs_other, qs_self, alpha_other,
            qs_other_predicted=None  # Opponent stays
        )
        return argmin(G_other_social)

    # Recursive: predict what other predicts we'll do
    our_predicted_action = predict_other_action_recursive(
        model_self, model_other, qs_self, qs_other,
        alpha_self, alpha_other, depth=depth-1
    )

    # Use our predicted position
    qs_self_predicted = propagate(our_predicted_action)

    # Other computes G_social using our predicted position
    G_other_self, G_other_social = compute_G_empathic_multistep(
        model_other, model_self, qs_other, qs_self, alpha_other,
        qs_other_predicted=qs_self_predicted
    )

    return argmin(G_other_social)


def plan_with_empathy(model_self, model_other, qs_self, qs_other,
                      alpha_self, alpha_other):
    """
    BOTH agents use this SAME planning function.
    Only difference is alpha_self.
    """
    # Step 1: Predict other's action
    action_other = predict_other_action(alpha_other, alpha_self)

    # Step 2: Use predicted position
    qs_other_predicted = propagate(action_other)

    # Step 3: Compute G_social (same for all agents, alpha differs)
    G_self, G_social = compute_G_empathic_multistep(
        model_self, model_other, qs_self, qs_other, alpha_self,
        qs_other_predicted=qs_other_predicted
    )

    # Step 4: Choose action
    return argmin(G_social)
```

### Results (collision=-100, depth=2)

```
Both selfish (alpha=0):        i=STAY, j=STAY  (Paralysis)
i empathic, j selfish:         i=RIGHT, j=LEFT (Both advance, swap)
i selfish, j empathic:         i=RIGHT, j=LEFT (Both advance, swap)

ToM PREDICTIONS: ALL CORRECT!
  Case 2: i predicts j=LEFT [OK], j predicts i=RIGHT [OK]
  Case 3: i predicts j=LEFT [OK], j predicts i=RIGHT [OK]
```

### Analysis

With symmetric architecture and accurate predictions:
- Both selfish agents STAY (paralysis) - predictions correct
- In asymmetric cases, BOTH agents advance (swap positions)
- Neither agent yields because swapping has better G than staying

**Problem:** Swapping in a 1-wide corridor is an EDGE COLLISION.
Both agents try to move through each other simultaneously.

---

## Experiment 6: Edge Collision Through C (FINAL SOLUTION)

### Problem
In Experiment 5, both agents in asymmetric cases chose to swap positions (i=RIGHT, j=LEFT).
In a 1-wide corridor, this is an **edge collision** - both agents try to pass through each other.

### Solution: Edge Collision Uses Same C as Cell Collision

Edge collision is handled identically to cell collision, using the same preference vector C:

```python
# Check swap: self ends up at other's start, other ends up at self's start
prob_self_at_other_start = np.sum(qs_self_1 * qs_other)
prob_other_at_self_start = np.sum(qs_other_step0 * qs_self)
swap_prob = prob_self_at_other_start * prob_other_at_self_start

# Edge collision observation distribution: [P(no_swap), P(swap)]
edge_obs_dist = np.array([1 - swap_prob, swap_prob])

# Use same C as cell collision - both agents experience the collision
edge_coll_utility_self = float((edge_obs_dist * C_self_cell_collision).sum())
edge_coll_utility_other = float((edge_obs_dist * C_other_cell_collision).sum())

# Subtract utility (same pattern as cell collision)
total_G_self = G_self_0 - edge_coll_utility_self
total_G_other = best_G_other_0 - edge_coll_utility_other
```

**Key insight:** Both agents feel the collision penalty C when computing their G values.
This is physically correct - a collision hurts both parties.

### Final Results (collision=-100, depth=2)

```
Both selfish (alpha=0):        i=STAY, j=STAY  (Paralysis - correct!)
i empathic, j selfish:         i=STAY, j=LEFT  (Coordination!)
i selfish, j empathic:         i=RIGHT, j=STAY (Coordination!)

ToM PREDICTIONS: ALL CORRECT!
```

### Analysis: Full Coordination Achieved!

**Case 1 (Both selfish):**
- Both predict other will STAY → both STAY
- Paralysis is the Nash equilibrium for symmetric selfish agents
- Predictions correct ✓

**Case 2 (i empathic, j selfish):**
- i (empathic) weighs G_social = G_self + 1.0 * G_other
- i sees that advancing causes edge collision → hurts both
- i chooses to STAY (yield)
- j predicts i will STAY → j advances LEFT toward goal
- Coordination achieved! ✓

**Case 3 (i selfish, j empathic):**
- j (empathic) weighs G_social = G_self + 1.0 * G_other
- j sees that advancing causes edge collision → hurts both
- j chooses to STAY (yield)
- i predicts j will STAY → i advances RIGHT toward goal
- Coordination achieved! ✓

---

## Summary: Complete Solution

### Architecture
1. **Recursive ToM (depth=2)**: Both agents accurately predict each other's actions
2. **Unified planning**: Both agents use identical `compute_G_empathic_multistep()`, only alpha differs
3. **G_social = G_self + alpha * G_other**: Empathy is just a weighting parameter
4. **Collision through C**: Both cell and edge collisions use same preference vector

### Key Parameters
- `POLICY_LENGTH = 3`: Multi-step horizon breaks single-step paralysis
- `COLLISION_PENALTY = -100.0`: High enough to deter collision but not paralyze goal-seeking
- `TOM_DEPTH = 2`: Sufficient recursion for accurate mutual prediction

### Theoretical Insights

1. **ToM vs Empathy are INDEPENDENT:**
   - ToM = prediction mechanism (ALL agents use it)
   - Empathy = decision weighting (alpha affects G_social)
   - The ONLY difference between agents is alpha

2. **Symmetry Breaking via Empathy:**
   - Selfish agents (alpha=0): Only consider own benefit → paralysis in symmetric case
   - Empathic agents (alpha>0): Consider joint welfare → yield to avoid mutual harm
   - Asymmetric empathy creates asymmetric behavior → coordination

3. **Edge Collision = Cell Collision:**
   - Both are physical collisions
   - Both use the same penalty C
   - Both affect both agents' G values
