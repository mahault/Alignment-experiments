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

### Phase 5: Testing & Validation

#### 5.1 Unit Tests
- Zone creation and cell membership
- Zone path finding (BFS)
- Subgoal selection
- Local model restriction

#### 5.2 Integration Tests
- Run on `vertical_bottleneck` with hierarchical planner
- Compare with flat planner (where flat works)
- Verify empathy effects at zone level

#### 5.3 Sweep Tests
- All empathy combinations (0, 0.5, 1.0)
- Both start configs (A, B)
- Success rate, collision rate, steps to goal

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
