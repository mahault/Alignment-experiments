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
