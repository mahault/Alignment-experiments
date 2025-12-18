# Hierarchical Spatial Planner Roadmap

## Problem Statement

The current flat planner enumerates all 5^H policies, causing:
- Horizon 3: 125 policies (fast)
- Horizon 5: 3125 policies (OOM on JAX)
- Horizon 7: 78125 policies (impossible)

For layouts like `vertical_bottleneck`, agents need ~7 steps to reach goals,
but the distance shaping creates local minima that trap agents with short horizons.

## Solution: Two-Level Hierarchical Planning

### Core Idea

Decompose the grid into **spatial zones** and plan at two levels:

1. **High-level**: Plan zone transitions (small state space)
2. **Low-level**: Plan within-zone navigation (small horizon needed)

### Zone Decomposition Example (vertical_bottleneck)

```
~ ~ ~ . ~ ~
~ ~ ~ . ~ ~
. 0 . . B .   ZONE 0: top_wide (y=2, all x)    ~6 cells
~ ~ ~ . ~ ~
~ ~ ~ . ~ ~   ZONE 1: bottleneck (x=3, y!=2,5) ~6 cells
. A . . 1 .   ZONE 2: bottom_wide (y=5, all x) ~6 cells
~ ~ ~ . ~ ~
~ ~ ~ . ~ ~
```

### Complexity Reduction

| Approach | Policies | Memory |
|----------|----------|--------|
| Flat H=7 | 5^7 = 78,125 | OOM |
| Hierarchical | 3^3 × 5^3 = 27 × 125 = 3,375 | OK |

---

## Implementation Plan

### Phase 1: Zone Infrastructure

**File: `tom/planning/hierarchical_planner.py`**

#### 1.1 SpatialZone Dataclass
```python
@dataclass
class SpatialZone:
    zone_id: int
    name: str
    cells: Set[Tuple[int, int]]
    entry_points: Dict[int, List[Tuple[int, int]]]  # from_zone -> cells
    exit_points: Dict[int, List[Tuple[int, int]]]   # to_zone -> cells
    adjacent_zones: List[int]
```

#### 1.2 ZonedLayout Dataclass
```python
@dataclass
class ZonedLayout:
    width: int
    height: int
    zones: List[SpatialZone]
    zone_graph: Dict[int, List[int]]      # adjacency
    cell_to_zone: Dict[Tuple[int, int], int]

    def get_zone_path(from_zone, to_zone) -> List[int]  # BFS
```

#### 1.3 Zone Factory Functions
- `create_vertical_bottleneck_zones(width, height) -> ZonedLayout`
- `create_symmetric_bottleneck_zones(width) -> ZonedLayout`
- `create_narrow_zones(width) -> ZonedLayout`  # Even narrow can benefit

---

### Phase 2: High-Level Zone Planner

**Purpose**: Decide sequence of zone transitions

#### 2.1 Zone State Space
```python
# State: (my_zone, other_zone)
# 3 zones × 3 zones = 9 joint states

# Actions: STAY, MOVE_TOWARD_GOAL, MOVE_AWAY (yield)
# 3 actions
```

#### 2.2 Zone Transition Model
```python
def zone_B(current_zone, action, goal_zone) -> next_zone:
    if action == STAY:
        return current_zone
    elif action == MOVE_TOWARD_GOAL:
        path = get_zone_path(current_zone, goal_zone)
        return path[1] if len(path) > 1 else current_zone
    elif action == MOVE_AWAY:
        # Move to adjacent zone away from goal
        ...
```

#### 2.3 Zone Preferences
```python
def zone_C(zone, goal_zone) -> float:
    # Reward for being in goal zone
    # Penalty for being far from goal zone
    # Collision penalty if both agents in same zone (especially bottleneck)
```

#### 2.4 Zone Collision Model
```python
def zone_collision_prob(my_zone, other_zone) -> float:
    # High if both in bottleneck
    # Low if in different wide zones
```

---

### Phase 3: Low-Level Within-Zone Planner

**Purpose**: Navigate within current zone toward subgoal

#### 3.1 Subgoal Selection
```python
def get_subgoal(current_zone, next_zone, goal_pos) -> Tuple[int, int]:
    if current_zone == next_zone:
        # Stay in zone - subgoal is final goal or zone center
        return goal_pos if goal_pos in zone.cells else zone_center
    else:
        # Moving to next zone - subgoal is exit point
        return current_zone.exit_points[next_zone][0]
```

#### 3.2 Local Model (Zone-Restricted)
```python
class LocalLavaModel:
    """LavaModel restricted to a single zone's cells."""

    def __init__(self, zone: SpatialZone, subgoal: Tuple[int, int]):
        # Only include cells from this zone
        self.cells = zone.cells
        self.goal = subgoal
        # Build A, B, C, D for zone cells only
```

#### 3.3 Local Planner
```python
def plan_within_zone(zone, subgoal, current_pos, other_pos, horizon=3):
    """
    Short-horizon EFE planning within zone.

    Uses existing EmpathicLavaPlanner but with:
    - Restricted state space (zone cells only)
    - Subgoal as target
    - Short horizon (2-3 steps)
    """
```

---

### Phase 4: Hierarchical Empathic Planner

**File: `tom/planning/hierarchical_empathy.py`**

#### 4.1 Main Planner Class
```python
@dataclass
class HierarchicalEmpathicPlanner:
    zoned_layout: ZonedLayout
    goal_pos: Tuple[int, int]
    alpha: float  # Empathy
    high_level_horizon: int = 3
    low_level_horizon: int = 3

    def plan(self, my_pos, other_pos) -> int:
        # 1. Determine current zones
        my_zone = self.zoned_layout.get_zone_for_cell(my_pos)
        other_zone = self.zoned_layout.get_zone_for_cell(other_pos)
        goal_zone = self.zoned_layout.get_zone_for_cell(self.goal_pos)

        # 2. High-level: pick zone transition
        zone_action = self.high_level_plan(my_zone, other_zone, goal_zone)

        # 3. Determine subgoal from zone action
        next_zone = self.apply_zone_action(my_zone, zone_action, goal_zone)
        subgoal = self.get_subgoal(my_zone, next_zone)

        # 4. Low-level: plan within zone toward subgoal
        action = self.low_level_plan(my_pos, other_pos, subgoal)

        return action
```

#### 4.2 Empathy at Both Levels

**High-level empathy**:
- Consider other agent's zone preferences
- Yield (move away) if other needs bottleneck more

**Low-level empathy**:
- Existing collision avoidance within zone
- Edge/cell collision preferences

---

### Phase 5: Testing & Validation ✅ COMPLETE

#### 5.1 Unit Tests ✅
- Zone creation and cell membership
- Zone path finding (BFS)
- Subgoal selection
- JAX compilation and correctness
- 40 tests passing

#### 5.2 Integration Tests ✅
- JAX hierarchical planner integrated with experiment sweep
- `--hierarchical` flag added to `run_empathy_sweep.py`

#### 5.3 JAX Implementation ✅
- `jax_hierarchical_planner.py` with full JIT compilation
- `HierarchicalEmpathicPlannerJax` compatible with experiment infrastructure

---

### Phase 5.4: Exit Point Bug Fix ✅ COMPLETE

**Problem**: Exit points were configured to be in the SOURCE zone instead of the
DESTINATION zone. Agents would navigate to exit points, find they were already
at the subgoal, and STAY forever.

**Fix**: Updated all three layouts to set exit points IN the destination zone:
- `vertical_bottleneck`: exit_points[0,1] = (3,3) not (3,2), exit_points[2,1] = (3,4) not (3,5)
- `symmetric_bottleneck`: exit_points[0,1] = (4,1) not (3,1), etc.
- `narrow`: Similar corrections

**Result**: Agents now move through bottleneck (paralysis = 0%), but both rush
forward and collide (collision rate = 100%). Empathy parameter has no effect.

---

### Phase 5.5: High-Level Empathy via Theory of Mind

**Problem**: The current empathy calculation is broken:

```python
# BROKEN: other_dist is CONSTANT for all actions!
other_dist = compute_zone_distance_jax(other_zone, other_goal_zone, zone_adjacency)
empathy_utility = alpha * zone_distance_cost * other_dist
```

This adds the same constant to all zone actions (STAY, FORWARD, BACK), so empathy
has zero effect on action selection. All alpha values produce identical behavior.

**Solution**: Match the flat planner's empathy structure (si_empathy_lava.py):

```python
# FLAT PLANNER FORMULA (line 496):
G_social = G_i + alpha * G_j_best_response
```

The key insight: **different actions from me lead to different outcomes for j**.
We must compute j's EFE *given my action*, not a constant.

**Implementation**:

For each of my zone actions, simulate j's best response and compute j's EFE:

```python
# 1. If I take this action, what zone will I be in?
next_zone = apply_zone_action(my_zone, action, my_goal_zone)

# 2. Given my next_zone, what's j's best response?
#    j wants to move toward other_goal_zone while avoiding collision with me
j_next_zone_if_forward = get_next_zone_toward(other_zone, other_goal_zone)

# 3. Am I blocking j's path?
#    I block j if I move into the bottleneck that j needs
j_needs_bottleneck = zone_is_bottleneck[j_next_zone_if_forward]
i_blocks_j = (next_zone == j_next_zone_if_forward) & j_needs_bottleneck

# 4. Compute j's EFE given my action
#    - If I block: j stays, doesn't progress, bad EFE
#    - If I don't block: j moves forward, good EFE
j_dist_if_blocked = distance(other_zone, other_goal_zone)
j_dist_if_clear = distance(j_next_zone_if_forward, other_goal_zone)

j_utility = where(i_blocks_j, j_dist_if_blocked, j_dist_if_clear) * zone_distance_cost
G_j_best = -j_utility  # Convert to EFE (lower = better for j)

# 5. Social utility: my utility + alpha * j's outcome
#    Since G = -utility, and G_social = G_i + alpha*G_j:
#    utility_social = utility_i - alpha * G_j
my_utility = distance_utility + goal_utility + collision_utility
total_utility = my_utility - alpha * G_j_best
```

**Expected Behavior After Fix**:

| alpha_i | alpha_j | Prediction |
|---------|---------|------------|
| 0.0 | 0.0 | Both rush forward, collide |
| 0.0 | 1.0 | i rushes, j yields (G_j bad if blocked, j cares) |
| 1.0 | 0.0 | i yields (cares about blocking j), j rushes through |
| 1.0 | 1.0 | Both consider other's EFE, may alternate or one yields |

**Status**: ✅ COMPLETE

**Tuning**: `bottleneck_collision_cost` tuned from -5 to -15:
- At -5: FORWARD always wins (empathy too weak)
- At -15: α=0 → FORWARD, α≥0.5 → STAY (correct behavior)
- At -30: STAY always wins (collision dominates even for selfish)

---

### Phase 6: Collision Inference from Observations

**Problem**: The high-level planner currently uses a hard-coded collision penalty
that is too aggressive. Agents refuse to enter bottleneck even when other agent
is on the opposite side (no real collision risk).

**Principle**: Collision probability should be INFERRED from observations at
the low level, not hard-coded at the high level. This follows active inference
principles where beliefs are updated by observations.

#### 6.1 Simple Fix: Reduce High-Level Collision Penalty

**Status**: To implement

Reduce `bottleneck_collision_cost` from -50 to -5 so distance cost dominates:
- High-level primarily plans zone sequence based on distance to goal
- Low-level handles actual collision avoidance via C matrices
- If agents get stuck (paralysis), that's observable feedback

```python
# Before: collision penalty dominates
bottleneck_collision_cost: float = -50.0  # Blocks forward movement

# After: distance cost dominates
bottleneck_collision_cost: float = -5.0   # Small nudge, doesn't override
```

#### 6.2 Observation-Based Collision Inference

**Status**: To implement after 6.1

Proper active inference approach:

1. **Collision belief state**: Track P(collision | zone_i, zone_j)
2. **Prior**: Start with low collision probability
3. **Observation**: Low-level reports collision events/near-misses
4. **Update**: Bayesian update of collision belief
5. **Planning**: High-level uses inferred P(collision), not hard-coded penalty

```python
@dataclass
class HierarchicalEmpathicPlannerJax:
    # Collision belief: P(collision | my_zone, other_zone)
    collision_belief: jnp.ndarray  # Shape: [num_zones, num_zones]

    def observe_collision(self, my_zone: int, other_zone: int, collision: bool):
        """Update collision belief based on low-level observation."""
        # Bayesian update
        prior = self.collision_belief[my_zone, other_zone]
        likelihood = 0.9 if collision else 0.1
        posterior = (likelihood * prior) / normalizer
        self.collision_belief = self.collision_belief.at[my_zone, other_zone].set(posterior)

    def high_level_plan(self, ...):
        # Use inferred collision probability instead of hard-coded penalty
        collision_prob = self.collision_belief[next_zone, other_zone]
        collision_utility = collision_cost * collision_prob
```

**Benefits**:
- Agent learns collision patterns from experience
- No hard-coded "same side" vs "opposite side" logic
- True active inference: beliefs updated by observations
- Empathy can modulate willingness to risk collision

#### 6.3 Retreat Behavior

With observation-based inference, retreat emerges naturally:
1. Agent moves forward (low prior on collision)
2. Low-level detects collision risk, agent gets stuck
3. Collision belief increases for this zone configuration
4. High-level re-plans with higher collision cost
5. ZONE_BACK becomes attractive (retreat to let other pass)

---

### Phase 7: Flat Planner Issues (Diagnosed 2024-12-12)

**Status**: In Progress

#### 7.1 Results Summary (horizon=3, flat planner)

Latest experiment results (`empathy_sweep_20251212_164010.csv`):

| Layout | Success | Collision | Paralysis | Issue |
|--------|---------|-----------|-----------|-------|
| **wide** | 100% | 0% | 0% | ✅ Works |
| **bottleneck** | 100% | 0% | 0% | ✅ Works |
| **double_bottleneck** | 67% | 0% | 33% | Partial |
| **vertical_bottleneck** | 44% | 11% | 44% | Partial |
| **passing_bay** | 44% | 0% | 56% | Partial |
| **narrow** | 0% | 0% | 100% | ❌ All paralyze |
| **crossed_goals** | 0% | 100% | 0% | ❌ All collide |
| **risk_reward** | 0% | 0% | 100% | ❌ All paralyze |
| **symmetric_bottleneck** | 0% | 0% | 100% | ❌ All paralyze |
| **asymmetric_detour** | 0% | 89% | 11% | ❌ Mostly collide |
| **t_junction** | 0% | 100% | 0% | ❌ All collide |

#### 7.2 Root Cause Analysis: Collision Avoidance Too Strong

**Key Finding**: Even with α=0 (selfish), agents avoid collisions because the collision
penalty (-30) is in their **OWN** C matrix. Empathy only affects G_j weighting, not
the agent's self-interest in avoiding collision damage.

Example from narrow corridor at distance=1:
```
t=2: i@(2, 1), j@(3, 1) | dist=1
     G_social by first action:
       STAY: 6.1   (BEST - no collision)
       LEFT: 8.1   (move away from goal)
       RIGHT: 14.1 (collision penalty -30!)
```

Both selfish agents choose STAY because RIGHT means they personally take -30 damage.
Neither will sacrifice themselves.

**Problem**: Current design assumes collision_cost in C affects both agents equally.
In reality, both agents avoid collision for self-preservation, leading to deadlock.

#### 7.3 Theory of Mind (alpha_other) IS Working

The `alpha_other` parameter correctly scales j's collision preferences when computing
j's best response:
```python
collision_scale = 1.0 + alpha_other  # alpha_other=1 → 2x collision cost
C_j_cell_collision_scaled = C_j_cell_collision * collision_scale
```

So if i knows j is empathic (alpha_other=1), i predicts j will more strongly avoid
collisions. But this doesn't help in narrow corridor because BOTH agents avoid
collisions for self-interest.

#### 7.4 Why Empathic Agents Don't Yield

In narrow corridor, yielding means:
- STAY: G=6.1 (neutral, no progress, no collision)
- LEFT (retreat): G=8.1 (move away from goal)
- RIGHT (forward): G=14.1 (collision penalty dominates)

An empathic agent (α=1) computes G_social = G_i + α*G_j_best:
- If i STAYs, j also STAYs → G_j_best ≈ 6 → G_social ≈ 12
- If i moves LEFT (retreats), j moves RIGHT → G_j_best ≈ 6 → G_social ≈ 14

Retreating doesn't help because:
1. Retreating costs i distance to goal (+2)
2. j's G_best doesn't improve enough to compensate
3. The narrow corridor has no passing space - retreat just delays the deadlock

**Resolution**: Empathic agents DO yield at the single-step level, but:

1. **Narrow corridor is impossible**: No passing space. Empathic yielding only delays
   collision - the yielder eventually hits the wall and collision becomes unavoidable.

2. **ToM is working**: When adjacent, empathic j retreats (RIGHT) to let selfish i pass.
   But j backs up to (5,1), then i catches up and collision at t=5.

3. **Why experiments show paralysis not collision**: The experiment runner detects paralysis
   (both STAYing) before collision because:
   - At dist=5: both rush (RIGHT/LEFT)
   - At dist=1: when both are moderately empathic, BOTH try to yield
   - Both yield = both STAY = paralysis detected

**Test Results (horizon=3, adjacent at dist=1)**:
- α=0/0: Both STAY → PARALYSIS
- α=0/1.0: i STAY, j yields → j eventually hits wall → COLLISION
- α=1.0/0: i yields, j STAY → i eventually hits wall → COLLISION
- α=0.5/0.5: BOTH yield (both retreat) → weird dynamics
- α=1.0/1.0: BOTH yield → weird dynamics

#### 7.6 ROOT CAUSE: Goal/Collision Ratio Changed

**The coordination parameters were changed, breaking the incentive structure:**

```python
# BEFORE (coordination worked):
goal_reward = 50.0
collision_penalty = -100.0
# Net if collision + goal: 50 - 100 = -50 (AVOID collision!)

# AFTER (coordination broken):
goal_reward = 80.0   # Increased "to make goal more appealing"
collision_penalty = -30.0   # Reduced from -100
# Net if collision + goal: 80 - 30 = +50 (ACCEPT collision!)
```

**Why this breaks coordination:**
- In crossed_goals, both agents reach (5,1)/(5,2) and try to swap rows
- The swap causes edge collision (-30) but reaches the goal (+80)
- Net utility: +50, which is BETTER than avoiding collision
- So agents always rush forward and accept collision

**Fix options:**
1. **Increase collision penalty**: Back to -100, or at least > goal_reward
2. **Decrease goal reward**: Back to 50
3. **Make collision penalty proportional**: collision = -2 * goal_reward

**Recommended**: collision_penalty should be at least 2x goal_reward for coordination:
```python
goal_reward = 50.0
collision_penalty = -100.0  # -2 * goal
```

#### 7.5 Layouts Added: Goal-Swapped Configurations

Added `swap_goals()` and `swap_agents_and_goals()` methods to LavaLayout:
- **Config A**: Original (agent i has goal 0, agent j has goal 1)
- **Config B**: Swapped agents (positions AND goals swapped)
- **Config C**: Swapped goals only (same positions, each wants other's goal)
- **Config D**: Swapped agents + goals (agents swap positions, original goals)

This allows testing crossed-path scenarios systematically.

#### 7.7 CRITICAL: Simultaneous-Move Coordination Failure

**Status**: Discovered 2025-12-12

**Problem**: Even with correct edge collision detection (P(collision)=1.0) and strong
collision penalty (-100), agents STILL collide in crossed_goals layout.

**Root Cause**: The ToM assumes SEQUENTIAL moves (i commits, then j responds), but
agents actually plan SIMULTANEOUSLY.

**Detailed Trace** (crossed_goals at positions (5,1) and (5,2)):

```python
# AGENT 0's PLANNING (at (5,1), goal at (5,2)):
# Consider action UP:
#   ToM: "If I go UP, what does j at (5,2) do?"
#   - j going DOWN would cause edge collision (swap through same edge)
#   - j evaluates: DOWN = goal(+50) + collision(-100) = -50
#   - j evaluates: STAY = 0 (no goal, no collision)
#   - ToM predicts: j will STAY (0 > -50)
#   -> Agent 0 chooses UP expecting NO collision

# AGENT 1's PLANNING (at (5,2), goal at (5,1)) SIMULTANEOUSLY:
# Consider action DOWN:
#   ToM: "If I go DOWN, what does i at (5,1) do?"
#   - i going UP would cause edge collision
#   - i evaluates: UP = goal(+50) + collision(-100) = -50
#   - i evaluates: STAY = 0
#   - ToM predicts: i will STAY (0 > -50)
#   -> Agent 1 chooses DOWN expecting NO collision

# REALITY:
# Both agents' ToM models predict the OTHER will yield!
# Agent 0 commits to UP, Agent 1 commits to DOWN
# => EDGE COLLISION!
```

**This is the "After you" problem in reverse**: Each agent assumes they have
right-of-way because their ToM model treats them as the "first mover."

**Code Evidence** (run_empathy_sweep.py, lines 201 & 210):
```python
# Line 201: Agent i plans based on CURRENT state
G_i, G_j_sim, G_social_i, q_pi_i, action_i = planner_i.plan(qs_i, qs_j_observed)

# Line 210: Agent j plans based on SAME CURRENT state (not i's result!)
G_j, G_i_sim, G_social_j, q_pi_j, action_j = planner_j.plan(qs_j, qs_i_observed)

# Line 262: Both actions executed simultaneously
next_state, next_obs, reward, done, info = env.step(state, {0: action_i, 1: action_j})
```

**Potential Solutions**:

1. **Uncertainty over other's action**: Instead of assuming j best-responds,
   maintain a distribution over j's possible actions:
   ```python
   # For each action a_i, compute EXPECTED collision over j's actions
   for a_j in actions:
       p_a_j = softmax(-G_j[a_j])  # j's likely action distribution
       expected_collision += p_a_j * P(collision | a_i, a_j)
   ```

2. **Mixed-strategy Nash equilibrium**: Use Lemke-Howson or fictitious play
   to find the equilibrium strategy profile.

3. **Correlated equilibrium via signaling**: Introduce a "signal" mechanism
   that breaks symmetry (e.g., agent with lower ID goes first).

4. **Multi-agent MCTS**: Tree search that explicitly models simultaneous moves.

5. **Communication primitive**: Allow agents to "announce" intended action,
   then plan based on announced actions.

**Recommended Fix**: Option 1 (uncertainty over j's actions) is most compatible
with active inference principles. Instead of `argmin G_j`, use `softmax(-G_j)`
to get a probability distribution, then compute EXPECTED collision.

---

## Key Design Decisions

### Q1: How to handle zone boundaries?

**Decision**: Entry/exit points are shared between zones.
When agent is at exit point of zone A (which is entry point of zone B),
they are considered "in zone A" until they take an action that moves them
into zone B's interior.

### Q2: What if goal is in different zone?

**Decision**: High-level planner creates zone path to goal zone.
Low-level planner navigates to exit points until reaching goal zone,
then navigates to actual goal position.

### Q3: Empathy at which level?

**Decision**: Both levels.
- High-level: "Should I yield the bottleneck to the other agent?"
- Low-level: "Should I avoid this specific cell collision?"

### Q4: What if zones have different sizes?

**Decision**: Each zone gets its own local model with its own state space.
Larger zones have more states but still bounded.
Bottleneck zones are intentionally small.

---

## File Structure

```
tom/planning/
├── hierarchical_planner.py      # Zone infrastructure (Phase 1)
├── hierarchical_empathy.py      # Main planner (Phase 4)
├── zone_level_planner.py        # High-level zone planning (Phase 2)
├── local_zone_planner.py        # Low-level within-zone (Phase 3)
└── __init__.py                  # Export new classes

tests/
└── test_hierarchical.py         # All hierarchical tests
```

---

## Success Criteria

1. **Memory**: Can run horizon-equivalent-7 without OOM
2. **Correctness**: Agents reach goals on vertical_bottleneck
3. **Empathy**: Prosocial agents yield at bottleneck
4. **Speed**: < 5 seconds per episode planning time
5. **Compatibility**: Works with existing experiment sweep infrastructure
