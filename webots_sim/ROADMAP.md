# Webots Empathic Robot Simulation - Roadmap

## Overview

Two TIAGo robots navigate toward opposite goals. Empathic coordination emerges naturally from Theory of Mind (ToM) based Expected Free Energy (EFE) planning.

**Key Insight**: Yielding is NOT hardcoded. It emerges from:
```
G_social = G_self + alpha * G_other
```
- Higher alpha = more empathic = considers other's path more = yields
- Lower alpha = more selfish = prioritizes own goal = pushes forward

---

## Folder Structure

```
webots_sim/
├── worlds/
│   └── tiago_empathic_test.wbt    # Main simulation world
├── controllers/
│   └── tiago_empathic/
│       ├── tiago_empathic.py      # Robot controller (runs in Webots)
│       └── tom_planner.py         # ToM-based continuous planner
├── protos/
│   ├── Target.proto               # Goal markers
│   └── HazardObstacle.proto       # Hazard/lava zones
├── planning_server.py             # [DEPRECATED] Grid-based JAX planner
└── ROADMAP.md                     # This file
```

---

## Planners

### 1. ToM Planner (CURRENT - Recommended)
**File**: `controllers/tiago_empathic/tom_planner.py`

Continuous-space Theory of Mind planner using EFE.

**Features**:
- Recursive ToM: predicts other robot's action by simulating THEIR decision
- Continuous coordinates (no grid discretization)
- Emergent yielding from alpha differences
- Self-contained (no external server needed)

**Key Classes**:
- `ToMPlanner`: Main planner class
  - `plan()`: Returns target (x, y) position
  - `predict_other_action()`: Recursive ToM prediction
  - `compute_g_self()`: Cost to reach own goal
  - `compute_g_other()`: How position affects other agent

### 2. Position Planner (Simple/Legacy)
**File**: `controllers/tiago_empathic/position_planner.py`

Simpler rule-based planner (less interesting for research).

### 3. JAX Planning Server (DEPRECATED)
**File**: `planning_server.py`

Grid-based Active Inference planner using JAX. Requires separate server process.
Not recommended - discrete grid doesn't map well to continuous robot motion.

---

## How to Run

### Quick Start (Recommended)

1. **Open Webots** and load the world:
   ```
   webots_sim/worlds/tiago_empathic_test.wbt
   ```

2. **Press Play** - robots will start navigating automatically

3. **Watch the behavior**:
   - RED marker (TIAGo_1): Selfish (alpha=0.0) - pushes forward, ignores other
   - GREEN marker (TIAGo_2): Empathic (alpha=6.0) - yields (backs up) to let other pass

### Console Output

The Webots console shows real-time planning decisions:
```
TIAGo_1: pos=(-0.65,-0.35) other=(1.00,0.00) target=(0.00,-0.35) | alpha=0.3 G=3.84 [TOWARD_GOAL]
TIAGo_2: pos=(1.35,0.35) other=(-0.65,-0.35) target=(1.35,0.35) | alpha=0.7 G=5.25 [HOLDING]
```

---

## Configuration

### Robot Parameters (in .wbt file)

Each TIAGo robot has `customData` with format:
```
goal_x, goal_y, alpha, agent_id
```

Example:
- TIAGo_1: `"1.25,0.0,0.0,0"` → goal=(1.25,0), alpha=0.0 (purely selfish)
- TIAGo_2: `"-1.25,0.0,6.0,1"` → goal=(-1.25,0), alpha=6.0 (highly empathic)

### Planner Parameters (in tom_planner.py)

```python
self.num_samples = 8       # Candidate positions to evaluate
self.sample_radius = 0.5   # How far to look for candidates
self.robot_radius = 0.3    # Collision radius
self.goal_tolerance = 0.3  # When to consider goal reached
```

---

## Expected Behavior

With default settings (alpha=0.0 vs alpha=6.0):

1. **Initial**: Both robots face each other in narrow corridor
2. **Approach**: Both initially move toward each other
3. **Yielding**: Empathic robot (alpha=6.0) backs up repeatedly to create space
4. **Passing**: Selfish robot advances while empathic robot continues backing up
5. **Completion**: Selfish robot reaches goal; empathic robot stays backed up

**Key insight**: Yielding emerges from pure EFE - no hardcoded rules. High alpha makes
`G_social = G_self + alpha * G_other` favor actions that help the other robot reach goal.

---

## Testing the Planner Standalone

Run the planner tests without Webots:

```bash
cd webots_sim/controllers/tiago_empathic
python tom_planner.py
```

This runs `test_tom_planner()` and `simulate_interaction()` to verify the logic.

---

## Development History & Lessons Learned

### What We Tried (and Why It Was Wrong)

#### 1. Commitment Bias (REMOVED)
**Idea**: Add 10% penalty for switching lateral directions (UP→DOWN or DOWN→UP) to prevent oscillation.
**Why Removed**: This is hardcoding behavior. If UP and DOWN have identical G values, that's a symptom that the EFE formulation doesn't capture something important about the geometry - not something to patch with biases.

#### 2. BACK Preference (REMOVED)
**Idea**: When UP and DOWN have nearly equal G values, give BACK a 15% bonus to break the tie.
**Why Removed**: This directly hardcodes "prefer backing up" which defeats the entire purpose. Yielding should emerge from `G_social = G_self + alpha * G_other`, not from rules.

#### 3. Continue-Backing Bonus (REMOVED)
**Idea**: If the robot was backing up last step, give 5% bonus to continue backing up.
**Why Removed**: Same problem - hardcoding yielding behavior.

### Current State (Pure EFE)

The planner now uses pure Expected Free Energy with NO hardcoded preferences:
```python
best_idx = int(jnp.argmin(all_g))  # Just pick lowest G, no adjustments
```

**What works**:
- Empathic robot (alpha=6.0) correctly identifies BACK as beneficial early on
- Decision logging shows clear G values for each action option
- JAX vectorization efficiently evaluates all 625 policies (5^4)

**Current issues to investigate**:
- When empathic robot reaches wall (BACK clamped), UP and DOWN have identical G values
- This causes random selection between them, leading to no clear lateral commitment
- The selfish robot then gets stuck because neither robot commits to a passing lane

### Root Cause Analysis

The issue is that UP and DOWN are **geometrically symmetric** when:
1. Both robots are on Y=0 (centerline)
2. The corridor is symmetric around Y=0

The EFE formula `G = distance_to_goal² / (2 * temp²)` doesn't break this symmetry because:
- Going UP then DOWN gets you back to Y=0 (same as going DOWN then UP)
- The other robot is predicted to stay on Y=0, so both directions give equal clearance

**Potential solutions (to explore, NOT hardcode)**:
1. Longer planning horizon to see that committing to one direction is better than oscillating
2. Model the other robot's lateral response (currently assumes they stay on Y=0)
3. Add noise/stochasticity to break ties naturally
4. Investigate if the corridor geometry truly allows passing (Y separation needed vs available)

---

## Future Work

- [ ] Investigate why EFE doesn't differentiate UP vs DOWN (symmetry breaking)
- [ ] Model other robot's lateral movement in ToM prediction
- [ ] Test if corridor is actually wide enough for passing (1.0m corridor, 0.7m collision threshold)
- [ ] Add more robots (n-agent coordination)
- [ ] Dynamic alpha adjustment based on urgency
- [ ] Integrate with original JAX Active Inference models
- [ ] Add obstacle avoidance for dynamic obstacles
- [ ] Visualize EFE landscape in real-time
