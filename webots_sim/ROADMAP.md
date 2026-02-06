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
│       ├── tom_planner.py         # Proper Active Inference EFE planner (CURRENT)
│       └── tom_planner_legacy.py  # Old continuous-distance heuristic (DEPRECATED)
├── protos/
│   ├── Target.proto               # Goal markers
│   └── HazardObstacle.proto       # Hazard/lava zones
├── planning_server.py             # [DEPRECATED] Grid-based JAX planner
└── ROADMAP.md                     # This file
```

---

## Planners

### 1. Proper Active Inference EFE Planner (CURRENT - Recommended)
**File**: `controllers/tiago_empathic/tom_planner.py`

Discrete generative model (A/B/C/D matrices) with proper Expected Free Energy.
Based on Sophisticated Inference (Friston et al. 2020) and ToM extension (2508.00401v2).

**Features**:
- Proper EFE = pragmatic (utility) + epistemic (information gain about other's role)
- Collision avoidance via preferences C, NOT hard clamps
- ToM: `Q(a_other) ~ softmax(-gamma * G_other)`, not greedy-x heuristic
- Bayesian belief update over other agent's hidden role/intent
- Blocked-motion mixing: congested transitions predict "stuck" futures (not ghosting)
- JAX vmap over 5^5=3125 policies, lax.scan over horizon steps, JIT-compiled
- Self-contained (no external server needed)

**Key Classes**:
- `CorridorModel`: Discrete POMDP generative model (all A/B/C/D matrices)
- `ToMPlanner`: Main planner class
  - `plan()`: Returns target (x, y) position + debug string
  - Internally: discretize -> compute EFE for all policies -> social EFE -> convert back

See **Architecture: Proper EFE Planner** section below for full technical details.

### 2. Legacy Continuous-Distance Planner (DEPRECATED)
**File**: `controllers/tiago_empathic/tom_planner_legacy.py`

Old continuous-space planner using fake EFE (distance²/2σ²). Replaced because:
- No epistemic value → lateral moves never explored
- Hard-clamp collision → no soft preference gradient
- Greedy-x other prediction → can't predict lateral yielding
- Caused deadlock when robots face each other

### 3. Position Planner (Simple/Legacy)
**File**: `controllers/tiago_empathic/position_planner.py`

Simpler rule-based planner (less interesting for research).

### 4. JAX Planning Server (DEPRECATED)
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
self.depth = 5              # Sophisticated inference depth (5^5=3125 policies)
self.gamma = 8.0            # Inverse temperature for softmax action selection
self.epistemic_scale = 1.0  # Weight on epistemic EFE term
self.discount = 0.9         # Future EFE discount factor
self.goal_tolerance = 0.3   # When to consider goal reached
```

**Discretization**:
```python
N_X = 11    # X bins at 0.4m over [-2.2, 2.2]
N_Y = 5     # Y bins for finer lateral resolution over [-0.5, 0.5]
N_POSE = 55 # Total pose states per agent (11 * 5)
N_ACTIONS = 5  # {STAY, FORWARD, BACK, LEFT, RIGHT}
N_ROLES = 4    # Hidden role states {PUSH, YIELD_LEFT, YIELD_RIGHT, WAIT}
ROBOT_RADIUS = 0.25  # meters
COLLISION_DIST = 0.50 # danger threshold (physical contact)
CAUTION_DIST = 0.80   # caution threshold (uncomfortably close)
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

## Architecture: Proper EFE Planner

This section documents the full technical design of the current `tom_planner.py`, based on
Sophisticated Inference (Friston et al. 2020) and empathic Active Inference (2508.00401v2).

### Why the Rewrite Was Needed

The old planner (`tom_planner_legacy.py`) used a **fake EFE**: `G = distance²/(2σ²)`.
This caused deadlock because:
1. **No epistemic value** → lateral "probe" moves never win over staying put
2. **Hard-clamp collision** → binary avoid/don't-avoid, no gradient to learn from
3. **Greedy-x other prediction** → assumed other always moves straight toward goal on X axis
4. **Symmetric lateral G** → UP and DOWN always have identical costs, causing oscillation

### Discrete Generative Model (A/B/C/D)

The planner uses a POMDP generative model with three state factors and four observation modalities.

#### State Factors

| Factor | Size | Description |
|--------|------|-------------|
| `s_self` | 55 | Own pose (11 X-bins x 5 Y-bins) |
| `s_other` | 55 | Other's pose (11 X-bins x 5 Y-bins) |
| `s_role` | 4 | Other's hidden intent: {PUSH, YIELD_LEFT, YIELD_RIGHT, WAIT} |

#### A Matrices (Observation Likelihoods)

| Matrix | Shape | Type | Purpose |
|--------|-------|------|---------|
| `A_self` | (55,55) | Identity | Self pose fully observable |
| `A_other` | (55,55) | Identity | Other pose fully observable |
| `A_risk` | (3,55,55) | Deterministic | {safe, caution, danger} from Euclidean distance between pose bins |
| `A_motion` | (4,4) | **Probabilistic** | {forward, backward, lateral, still} given role |

`A_motion` is the key non-identity likelihood — it connects the hidden role variable
to observed motion patterns with noise, making role inference meaningful:

```
              PUSH  YIELD_L  YIELD_R  WAIT
forward:     [0.80,  0.05,    0.05,   0.05]
backward:    [0.05,  0.10,    0.10,   0.05]
lateral:     [0.05,  0.75,    0.75,   0.10]
still:       [0.10,  0.10,    0.10,   0.80]
```

`A_risk` thresholds: danger < 0.5m (collision distance), caution < 0.8m, else safe.

#### B Matrices (Transitions)

| Matrix | Shape | Description |
|--------|-------|-------------|
| `B_self` | (55,55,5) | Self transition given action {STAY, FORWARD, BACK, LEFT, RIGHT} |
| `B_other_pose` | (55,55,4) | Other transition given role |
| `B_role` | (4,4) | Sticky role transitions (0.70 diagonal) |

**No hard occupancy constraint in B**. Collision avoidance is entirely in `C_risk` preferences.
FORWARD/BACK are relative to goal direction (`+x_bin` or `-x_bin` depending on which goal).

#### C Vectors (Preferences)

| Vector | Shape | Values |
|--------|-------|--------|
| `C_self` | (55,) | +80 at goal bin, -3*manhattan elsewhere, -0.1 offset |
| `C_risk` | (3,) | [0, -2, -25] for safe/caution/danger |
| `C_motion` | (4,) | [0, 0, 0, 0] neutral (purely epistemic modality) |

`C_risk` is the key innovation: collision avoidance as preference, not hard clamp.
`-25` for danger must compete with goal utility (+80) so ToM correctly values clearing space.
`-2` for caution gives a mild gradient to maintain distance.

#### D Vectors (Priors)

| Vector | Shape | Values |
|--------|-------|--------|
| `D_self` | (55,) | Delta at current bin (set each call) |
| `D_other` | (55,) | Delta at observed bin (set each call) |
| `D_role` | (4,) | [0.4, 0.2, 0.2, 0.2] — persists across calls, updated via Bayes |

### EFE Computation

For each candidate action `a` at each step in the horizon:

```
G(a) = G_pragmatic(a) + G_epistemic(a)

G_pragmatic = -(C_self . q(s_self'))                          [location utility]
              -(C_risk . P(o_risk | s_self', s_other'))       [clearance utility]

G_epistemic = -scale * [H(q(role')) - E_o[H(q(role'|o))]]    [info gain about role]
```

**Sophisticated inference** (depth=5): Enumerate all 5^5 = 3125 policies.
For each policy `[a1, a2, a3, a4, a5]`:
- Roll out 5 steps, accumulating discounted G at each step
- At each step: transition beliefs via B matrices, compute pragmatic + epistemic
- Use `lax.scan` for the horizon loop, `vmap` over all 3125 policies

### Blocked-Motion Mixing (Fix A)

After computing raw transitions, the model checks `p(danger)` from the predicted
joint next-state. If high, both agents' transitions mix with "stuck at current state":

```
q(s_agent') = (1 - p_block) * q(s_agent'_raw) + p_block * q(s_agent)
where p_block = clip(p_danger * 0.9, 0, 0.95)
```

This is the key mechanism for ToM empathy: if the other tries to push through you,
its predicted future becomes "no progress + danger" → high G_other. When the empathic
agent considers moving aside, it predicts lower G_other, and empathy emerges correctly.

**Not a hack**: this encodes physics ("pushing into an occupied space leads to stuck/bad
outcomes") into the generative model. ToM does the rest.

### Theory of Mind (ToM) Other Prediction

**Replaces greedy-x** with:
```
Q(a_other) ∝ softmax(-gamma * G_other(a_other))
```

Where `G_other` is computed from the other's perspective using their goal and preferences,
**including blocked-motion mixing** (so the other can't "ghost" through you in the rollout).

**Social EFE**: For each of our first actions `a`:
```
G_social(policy) = G_self(policy) + alpha * G_other_best_response(first_action)
```
Where `G_other_best_response = min over 5^3 policies of G_other`, with blocked-motion,
given our predicted next position. All JIT-compiled via JAX vmap.

### Bayesian Role Inference (Fix D)

Each `plan()` call:
1. Observe other's actual motion `Δ(x,y)` → classify as {forward, backward, lateral, still}
2. Bayesian update: `q(role | o) ∝ A_motion[o, :]^confidence * q(role)`
3. Propagate: `q(role) = B_role @ q(role)`

**confidence** is attenuated during near-contact (0.2 in danger zone, 0.5 in caution,
1.0 when far). This prevents confounded evidence: during physical contact, displacement
is caused by collision physics, not the other's intended policy.

### JAX Parallelization

All computation is JIT-compiled:
```
Self EFE:   vmap(3125 policies) → lax.scan(5 horizon steps)
Social EFE: vmap(5 our actions) → vmap(125 other policies) → lax.scan(3 steps)
```
GPU-ready: all matrices are `jnp.array`, functions decorated with `@jax.jit`.

Reference patterns from `tom/planning/jax_si_empathy_lava.py`.

### Interface Contract (unchanged from legacy)

```python
class ToMPlanner:
    def __init__(self, agent_id: int, goal_x: float, goal_y: float, alpha: float)
    def plan(self, my_x, my_y, other_x, other_y, other_goal_x, other_goal_y, other_alpha)
        -> (target_x: float, target_y: float, debug_str: str)
```

No changes needed to `tiago_empathic.py` controller — same interface preserved.

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

### Current State (Proper Active Inference EFE — v3, blocked-motion)

The planner was **fully rewritten** and then refined with 4 targeted fixes to address
why the empathic agent only "half-yields" instead of fully clearing the path.

**Standalone test result** (both robots reach goals in 18 steps):
```
Steps 1-2:  Both approach (FORWARD), distance narrows 3.6m → 2.0m
Steps 3-4:  R2 (alpha=6.0) yields (BACK) to wall while R1 advances
Steps 5-8:  R1 pushes through (caution zone, no blocking)
Step  9:    R1 reaches goal
Steps 10-18: R2 heads to its goal unobstructed
```

#### The 4 Fixes (all consistent with pure ToM+EFE)

**Fix A: Blocked-motion in transitions** (the critical one)
- Problem: self and other positions evolved independently ("ghosting") — the other's
  simulated dynamics did not reflect being blocked, so G_other was not harmed by
  you being "in the way", so empathy couldn't value clearing space.
- Fix: after computing raw transitions, mix with "stuck" proportional to p(danger).
  Now pushing through congestion predicts no progress + danger → high G_other.
- This encodes physics into the generative model, not a yield rule.

**Fix B: Danger penalty scaled to compete with goal utility**
- Problem: goal utility was +80 and danger was only -6. ToM would rationally accept
  danger to achieve the goal, so empathy said "fine, let them push through".
- Fix: `C_risk = [0, -2, -25]`. Danger at -25 competes with goal gradient (-3/step).

**Fix C: Horizon increased to 5 steps (3125 policies)**
- Problem: with horizon 3, the planner couldn't see a full pass-by maneuver (needs
  5-8 steps: pull aside, wait, re-enter). It picked the best local move: back a bit.
- Fix: depth=5 (3125 policies), still fast with JAX vmap + lax.scan.

**Fix D: Attenuated belief update during contact**
- Problem: during physical contact, displacement is confounded by collision physics,
  not the other's intended policy. This caused incorrect role inference (e.g., inferring
  "PUSH" when the other was actually trying to yield but got bumped forward).
- Fix: `confidence` parameter attenuated to 0.2 in danger zone, 0.5 in caution.
  Uses tempered likelihood: `P(o|role)^confidence`.

#### What Remains: "Get Out of the Way" Problem

The empathic agent currently yields by backing straight up along the centerline. This
creates space but doesn't create a **passing lane**. The agent should identify a
**clearance position** — a location where it won't block the other's path — and move
there. This position depends on the geometry and should be reidentified each step.

**What the planner needs to solve this**:
The agent must predict: "where should I be such that the other agent's predicted EFE
is minimized?" This is already encoded in the social EFE term, but currently the
empathic agent picks BACK because:
1. BACK creates distance (reduces danger/caution for the other)
2. Lateral moves cost manhattan distance to own goal without enough social payoff
3. The social EFE only evaluates the other's response to our FIRST action, not our
   full trajectory

**Possible approaches (still pure ToM+EFE)**:
1. Evaluate social EFE over full policy trajectory, not just first action
2. Add macro-actions: PULL_ASIDE_LEFT = [LEFT, LEFT, STAY], etc.
3. Precompute "clearance map": for each of our positions, what is the other's best G?
   Then add preference for clearance positions to C_self when alpha > 0

---

## Future Work

### Done
- [x] ~~Investigate why EFE doesn't differentiate UP vs DOWN (symmetry breaking)~~ → Solved by epistemic value + role inference
- [x] ~~Model other robot's lateral movement in ToM prediction~~ → Solved by softmax(-γG) ToM with role-based predictions
- [x] ~~Blocked-motion in transitions~~ → Fix A: congested transitions predict stuck futures
- [x] ~~Scale danger to compete with goal utility~~ → Fix B: C_risk = [0, -2, -25]
- [x] ~~Increase planning horizon~~ → Fix C: depth=5 (3125 policies)
- [x] ~~Fix ToM evidence during contact~~ → Fix D: attenuated belief update

### In Progress
- [ ] **"Get out of the way" yielding** — empathic agent should identify a clearance
      position (back + lateral) rather than just backing up on centerline. The social
      EFE should evaluate over the full policy trajectory so lateral maneuvers are valued.
- [ ] **Validate in Webots** (tiago_empathic_test.wbt) — verify robots coordinate

### To Do
- [ ] Install `jax[cuda]` for GPU acceleration (currently CPU-only)
- [ ] Profile JIT warmup vs steady-state timing; ensure < 200ms per plan() call
- [ ] Tune hyperparameters: epistemic_scale, gamma, C_risk values, depth
- [ ] Add more robots (n-agent coordination)
- [ ] Dynamic alpha adjustment based on urgency
- [ ] Integrate with original JAX Active Inference models (tom/planning/jax_si_empathy_lava.py)
- [ ] Add obstacle avoidance for dynamic obstacles
- [ ] Visualize EFE landscape in real-time
- [ ] Investigate stochastic policy selection (softmax over G_social) vs argmin
