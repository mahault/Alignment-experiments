"""
Hierarchical spatial planner for multi-agent coordination.

This module implements a two-level hierarchical planning approach:

HIGH LEVEL (Zone Planning):
- State space: Which zone each agent is in
- Actions: Stay in zone, move to adjacent zone
- Small policy space: 3^H_high (e.g., 27 for H=3)

LOW LEVEL (Within-Zone Navigation):
- State space: Grid cells within current zone only
- Actions: Primitive moves (UP, DOWN, LEFT, RIGHT, STAY)
- Reduced policy space: 5^H_low per zone (e.g., 125 for H=3)

Key insight: By decomposing spatially, we can achieve effective horizon ~7
with computation cost of horizon ~3-4.

See HIERARCHICAL_PLANNER_ROADMAP.md for full design details.
"""

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
from enum import IntEnum


# =============================================================================
# Phase 1: Zone Infrastructure
# =============================================================================

class ZoneAction(IntEnum):
    """High-level zone transition actions."""
    STAY = 0           # Remain in current zone
    MOVE_FORWARD = 1   # Move toward goal zone
    MOVE_BACK = 2      # Move away from goal zone (yield)


@dataclass
class SpatialZone:
    """
    Defines a spatial zone within the grid.

    Zones partition the grid into regions for hierarchical planning.
    Each zone has entry/exit points that connect to adjacent zones.

    Attributes
    ----------
    zone_id : int
        Unique identifier for this zone
    name : str
        Human-readable name (e.g., "top_wide", "bottleneck", "bottom_wide")
    cells : Set[Tuple[int, int]]
        Set of (x, y) grid cells belonging to this zone
    entry_points : Dict[int, List[Tuple[int, int]]]
        Maps adjacent zone_id -> list of cells where you can enter FROM that zone
    exit_points : Dict[int, List[Tuple[int, int]]]
        Maps adjacent zone_id -> list of cells where you can exit TO that zone
    adjacent_zones : List[int]
        List of zone_ids that are adjacent to this zone
    is_bottleneck : bool
        True if this zone is a constrained bottleneck (higher collision risk)
    """
    zone_id: int
    name: str
    cells: Set[Tuple[int, int]]
    entry_points: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    exit_points: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    adjacent_zones: List[int] = field(default_factory=list)
    is_bottleneck: bool = False

    def contains(self, pos: Tuple[int, int]) -> bool:
        """Check if position is in this zone."""
        return pos in self.cells

    def get_entry_from(self, from_zone_id: int) -> List[Tuple[int, int]]:
        """Get entry points when coming from specified zone."""
        return self.entry_points.get(from_zone_id, [])

    def get_exit_to(self, to_zone_id: int) -> List[Tuple[int, int]]:
        """Get exit points when going to specified zone."""
        return self.exit_points.get(to_zone_id, [])

    @property
    def num_cells(self) -> int:
        return len(self.cells)

    def get_center(self) -> Tuple[float, float]:
        """Get center of zone (average of cell positions)."""
        if not self.cells:
            return (0.0, 0.0)
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass
class ZonedLayout:
    """
    A grid layout with spatial zone decomposition.

    Extends the basic layout with zone information for hierarchical planning.

    Attributes
    ----------
    width : int
        Grid width
    height : int
        Grid height
    zones : List[SpatialZone]
        List of spatial zones partitioning the grid
    zone_graph : Dict[int, List[int]]
        Adjacency graph: zone_id -> list of adjacent zone_ids
    cell_to_zone : Dict[Tuple[int, int], int]
        Maps each cell to its zone_id
    """
    width: int
    height: int
    zones: List[SpatialZone]
    zone_graph: Dict[int, List[int]] = field(default_factory=dict)
    cell_to_zone: Dict[Tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self):
        # Build cell_to_zone mapping
        for zone in self.zones:
            for cell in zone.cells:
                self.cell_to_zone[cell] = zone.zone_id

        # Build zone adjacency graph
        for zone in self.zones:
            self.zone_graph[zone.zone_id] = zone.adjacent_zones.copy()

    def get_zone(self, zone_id: int) -> Optional[SpatialZone]:
        """Get zone by ID."""
        for zone in self.zones:
            if zone.zone_id == zone_id:
                return zone
        return None

    def get_zone_for_cell(self, pos: Tuple[int, int]) -> Optional[int]:
        """Get zone_id for a grid cell."""
        return self.cell_to_zone.get(pos)

    def get_zone_path(self, from_zone: int, to_zone: int) -> List[int]:
        """
        Find shortest path between zones using BFS.

        Returns list of zone_ids from from_zone to to_zone (inclusive).
        """
        if from_zone == to_zone:
            return [from_zone]

        # BFS
        visited = {from_zone}
        queue = [(from_zone, [from_zone])]

        while queue:
            current, path = queue.pop(0)

            for neighbor in self.zone_graph.get(current, []):
                if neighbor == to_zone:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return []  # No path found

    def get_next_zone_toward(self, from_zone: int, to_zone: int) -> int:
        """Get the next zone to visit when moving from from_zone toward to_zone."""
        path = self.get_zone_path(from_zone, to_zone)
        if len(path) >= 2:
            return path[1]
        return from_zone  # Already at destination or no path

    def get_next_zone_away(self, from_zone: int, to_zone: int) -> int:
        """Get a zone to move to when yielding (moving away from to_zone)."""
        # Find an adjacent zone that's not toward the goal
        path_to_goal = self.get_zone_path(from_zone, to_zone)
        next_toward = path_to_goal[1] if len(path_to_goal) >= 2 else None

        # Pick any adjacent zone that's not the next step toward goal
        for adj in self.zone_graph.get(from_zone, []):
            if adj != next_toward:
                return adj

        # If no other option, stay in place
        return from_zone

    @property
    def num_zones(self) -> int:
        return len(self.zones)

    def get_bottleneck_zones(self) -> List[SpatialZone]:
        """Get all zones marked as bottlenecks."""
        return [z for z in self.zones if z.is_bottleneck]


# =============================================================================
# Zone Factory Functions
# =============================================================================

def create_vertical_bottleneck_zones(width: int = 6, height: int = 8) -> ZonedLayout:
    """
    Create zone decomposition for vertical_bottleneck layout.

    Layout:
    ~ ~ ~ . ~ ~
    ~ ~ ~ . ~ ~
    . 0 . . B .   ZONE 0: top_wide (y=2, all x)
    ~ ~ ~ . ~ ~
    ~ ~ ~ . ~ ~   ZONE 1: bottleneck (x=3, y=3,4 and edges)
    . A . . 1 .   ZONE 2: bottom_wide (y=5, all x)
    ~ ~ ~ . ~ ~
    ~ ~ ~ . ~ ~

    Returns
    -------
    ZonedLayout
        Layout with 3 zones defined
    """
    mid_x = width // 2  # 3
    wide_top_y = 2
    wide_bottom_y = height - 3  # 5

    # Zone 0: Top wide area (y=2, all safe x)
    top_cells = {(x, wide_top_y) for x in range(width)}

    # Zone 1: Bottleneck corridor (x=mid_x for all y except wide rows)
    # Include the cells at (mid_x, y) for y in rows that aren't the wide areas
    bottleneck_cells = set()
    for y in range(height):
        if y != wide_top_y and y != wide_bottom_y:
            bottleneck_cells.add((mid_x, y))

    # Zone 2: Bottom wide area (y=5, all safe x)
    bottom_cells = {(x, wide_bottom_y) for x in range(width)}

    # Define zones with entry/exit points
    zone_top = SpatialZone(
        zone_id=0,
        name="top_wide",
        cells=top_cells,
        adjacent_zones=[1],
        # Entry/exit at the bottleneck connection point
        entry_points={1: [(mid_x, wide_top_y)]},
        exit_points={1: [(mid_x, wide_top_y)]},
        is_bottleneck=False,
    )

    # Bottleneck entry/exit points
    # From top: enter at the cell just below top wide (if exists)
    # From bottom: enter at the cell just above bottom wide (if exists)
    bottleneck_entry_from_top = [(mid_x, wide_top_y + 1)] if (mid_x, wide_top_y + 1) in bottleneck_cells else [(mid_x, wide_top_y)]
    bottleneck_entry_from_bottom = [(mid_x, wide_bottom_y - 1)] if (mid_x, wide_bottom_y - 1) in bottleneck_cells else [(mid_x, wide_bottom_y)]

    zone_bottleneck = SpatialZone(
        zone_id=1,
        name="bottleneck",
        cells=bottleneck_cells,
        adjacent_zones=[0, 2],
        entry_points={
            0: bottleneck_entry_from_top,
            2: bottleneck_entry_from_bottom,
        },
        exit_points={
            0: bottleneck_entry_from_top,  # Same as entry (bidirectional)
            2: bottleneck_entry_from_bottom,
        },
        is_bottleneck=True,
    )

    zone_bottom = SpatialZone(
        zone_id=2,
        name="bottom_wide",
        cells=bottom_cells,
        adjacent_zones=[1],
        entry_points={1: [(mid_x, wide_bottom_y)]},
        exit_points={1: [(mid_x, wide_bottom_y)]},
        is_bottleneck=False,
    )

    return ZonedLayout(
        width=width,
        height=height,
        zones=[zone_top, zone_bottleneck, zone_bottom],
    )


def create_symmetric_bottleneck_zones(width: int = 10) -> ZonedLayout:
    """
    Create zone decomposition for symmetric_bottleneck layout.

    Layout (width=10):
    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    0 . . . B B . . . 1   (row 1)
    . . . . B B . . . .   (row 2)
    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    Zones:
    - Zone 0: Left wide (x < 4)
    - Zone 1: Bottleneck (4 <= x < 6, only row 1 safe)
    - Zone 2: Right wide (x >= 6)
    """
    height = 4
    bottleneck_start = width // 2 - 1  # 4
    bottleneck_end = width // 2 + 1    # 6

    # Zone 0: Left wide (x < bottleneck_start)
    left_cells = set()
    for x in range(bottleneck_start):
        left_cells.add((x, 1))
        left_cells.add((x, 2))

    # Zone 1: Bottleneck (only row 1 is safe in this region)
    bottleneck_cells = {(x, 1) for x in range(bottleneck_start, bottleneck_end)}

    # Zone 2: Right wide (x >= bottleneck_end)
    right_cells = set()
    for x in range(bottleneck_end, width):
        right_cells.add((x, 1))
        right_cells.add((x, 2))

    zone_left = SpatialZone(
        zone_id=0,
        name="left_wide",
        cells=left_cells,
        adjacent_zones=[1],
        entry_points={1: [(bottleneck_start - 1, 1)]},
        exit_points={1: [(bottleneck_start - 1, 1)]},
        is_bottleneck=False,
    )

    zone_bottleneck = SpatialZone(
        zone_id=1,
        name="bottleneck",
        cells=bottleneck_cells,
        adjacent_zones=[0, 2],
        entry_points={
            0: [(bottleneck_start, 1)],
            2: [(bottleneck_end - 1, 1)],
        },
        exit_points={
            0: [(bottleneck_start, 1)],
            2: [(bottleneck_end - 1, 1)],
        },
        is_bottleneck=True,
    )

    zone_right = SpatialZone(
        zone_id=2,
        name="right_wide",
        cells=right_cells,
        adjacent_zones=[1],
        entry_points={1: [(bottleneck_end, 1)]},
        exit_points={1: [(bottleneck_end, 1)]},
        is_bottleneck=False,
    )

    return ZonedLayout(
        width=width,
        height=height,
        zones=[zone_left, zone_bottleneck, zone_right],
    )


def create_narrow_zones(width: int = 6) -> ZonedLayout:
    """
    Create zone decomposition for narrow corridor layout.

    Even though narrow is a single-file corridor, we can still
    decompose it into zones to reason about "who has the right of way".

    Layout:
    ~ ~ ~ ~ ~ ~
    0 . . . . 1   (single row, y=1)
    ~ ~ ~ ~ ~ ~

    Zones (thirds of corridor):
    - Zone 0: Left third (agent 0's start area)
    - Zone 1: Middle (contested area)
    - Zone 2: Right third (agent 1's start area)
    """
    height = 3
    safe_y = 1

    third = width // 3

    # Zone 0: Left third
    left_cells = {(x, safe_y) for x in range(third)}

    # Zone 1: Middle third
    middle_cells = {(x, safe_y) for x in range(third, 2 * third)}

    # Zone 2: Right third
    right_cells = {(x, safe_y) for x in range(2 * third, width)}

    zone_left = SpatialZone(
        zone_id=0,
        name="left_start",
        cells=left_cells,
        adjacent_zones=[1],
        entry_points={1: [(third - 1, safe_y)]},
        exit_points={1: [(third - 1, safe_y)]},
        is_bottleneck=False,
    )

    zone_middle = SpatialZone(
        zone_id=1,
        name="middle_contested",
        cells=middle_cells,
        adjacent_zones=[0, 2],
        entry_points={
            0: [(third, safe_y)],
            2: [(2 * third - 1, safe_y)],
        },
        exit_points={
            0: [(third, safe_y)],
            2: [(2 * third - 1, safe_y)],
        },
        is_bottleneck=True,  # High collision risk in narrow corridor
    )

    zone_right = SpatialZone(
        zone_id=2,
        name="right_start",
        cells=right_cells,
        adjacent_zones=[1],
        entry_points={1: [(2 * third, safe_y)]},
        exit_points={1: [(2 * third, safe_y)]},
        is_bottleneck=False,
    )

    return ZonedLayout(
        width=width,
        height=height,
        zones=[zone_left, zone_middle, zone_right],
    )


# =============================================================================
# Registry
# =============================================================================

ZONED_LAYOUTS = {
    "vertical_bottleneck": create_vertical_bottleneck_zones,
    "symmetric_bottleneck": create_symmetric_bottleneck_zones,
    "narrow": create_narrow_zones,
}


def get_zoned_layout(layout_name: str, **kwargs) -> ZonedLayout:
    """
    Get a zoned layout by name.

    Parameters
    ----------
    layout_name : str
        Name of the layout (must have zone definition)
    **kwargs : dict
        Arguments passed to zone factory function (e.g., width, height)

    Returns
    -------
    ZonedLayout
        Layout with zone decomposition
    """
    if layout_name not in ZONED_LAYOUTS:
        available = list(ZONED_LAYOUTS.keys())
        raise ValueError(f"No zone definition for layout: {layout_name}. Available: {available}")
    return ZONED_LAYOUTS[layout_name](**kwargs)


def has_zoned_layout(layout_name: str) -> bool:
    """Check if a layout has zone definitions."""
    return layout_name in ZONED_LAYOUTS


# =============================================================================
# Phase 2: High-Level Zone Planner
# =============================================================================

@dataclass
class ZoneState:
    """State representation at zone level."""
    my_zone: int
    other_zone: int
    my_goal_zone: int
    other_goal_zone: int


def compute_zone_distance(zoned_layout: ZonedLayout, from_zone: int, to_zone: int) -> int:
    """Compute number of zone transitions needed."""
    path = zoned_layout.get_zone_path(from_zone, to_zone)
    return max(0, len(path) - 1)


def apply_zone_action(
    zoned_layout: ZonedLayout,
    current_zone: int,
    action: ZoneAction,
    goal_zone: int
) -> int:
    """
    Apply a zone-level action to get next zone.

    Parameters
    ----------
    zoned_layout : ZonedLayout
        The zoned layout
    current_zone : int
        Current zone ID
    action : ZoneAction
        Zone action to take
    goal_zone : int
        Target goal zone

    Returns
    -------
    int
        Next zone ID after taking action
    """
    if action == ZoneAction.STAY:
        return current_zone
    elif action == ZoneAction.MOVE_FORWARD:
        return zoned_layout.get_next_zone_toward(current_zone, goal_zone)
    elif action == ZoneAction.MOVE_BACK:
        return zoned_layout.get_next_zone_away(current_zone, goal_zone)
    else:
        return current_zone


def compute_zone_G(
    zoned_layout: ZonedLayout,
    my_zone: int,
    other_zone: int,
    my_goal_zone: int,
    action: ZoneAction,
    alpha: float,
    other_goal_zone: int,
    bottleneck_collision_cost: float = -50.0,
    zone_distance_cost: float = -10.0,
    goal_zone_reward: float = 100.0,
) -> float:
    """
    Compute EFE for a zone-level action.

    Components:
    1. Distance to goal zone (closer = better)
    2. Goal zone reward (being in goal zone)
    3. Bottleneck collision risk (both agents in bottleneck = bad)
    4. Empathy: consider other agent's progress too

    Parameters
    ----------
    zoned_layout : ZonedLayout
        Zone layout
    my_zone : int
        Current zone
    other_zone : int
        Other agent's zone
    my_goal_zone : int
        My goal zone
    action : ZoneAction
        Action to evaluate
    alpha : float
        Empathy weight
    other_goal_zone : int
        Other agent's goal zone
    bottleneck_collision_cost : float
        Penalty when both agents in bottleneck
    zone_distance_cost : float
        Cost per zone away from goal
    goal_zone_reward : float
        Reward for being in goal zone

    Returns
    -------
    float
        Expected free energy (lower = better)
    """
    # Apply action to get next zone
    next_zone = apply_zone_action(zoned_layout, my_zone, action, my_goal_zone)
    next_zone_obj = zoned_layout.get_zone(next_zone)

    # 1. Distance to goal
    dist_to_goal = compute_zone_distance(zoned_layout, next_zone, my_goal_zone)
    distance_utility = zone_distance_cost * dist_to_goal

    # 2. Goal zone reward
    goal_utility = goal_zone_reward if next_zone == my_goal_zone else 0.0

    # 3. Bottleneck collision risk
    collision_utility = 0.0
    if next_zone_obj and next_zone_obj.is_bottleneck:
        # Predict other might also be in bottleneck
        # Simple heuristic: if other is adjacent to bottleneck or in it
        other_zone_obj = zoned_layout.get_zone(other_zone)
        if other_zone_obj:
            other_adjacent_to_bottleneck = next_zone in other_zone_obj.adjacent_zones
            other_in_bottleneck = other_zone_obj.is_bottleneck

            if other_in_bottleneck or other_adjacent_to_bottleneck:
                # Higher collision risk
                collision_utility = bottleneck_collision_cost * 0.5
            if other_zone == next_zone:
                # Both definitely in same zone
                collision_utility = bottleneck_collision_cost

    # 4. Empathy: consider other's progress
    other_dist = compute_zone_distance(zoned_layout, other_zone, other_goal_zone)
    empathy_utility = alpha * zone_distance_cost * other_dist

    # Total utility (higher = better)
    total_utility = distance_utility + goal_utility + collision_utility + empathy_utility

    # Convert to EFE (lower = better)
    G = -total_utility
    return G


def high_level_plan(
    zoned_layout: ZonedLayout,
    my_zone: int,
    other_zone: int,
    my_goal_zone: int,
    other_goal_zone: int,
    alpha: float,
    horizon: int = 3,
    gamma: float = 8.0,
) -> Tuple[ZoneAction, np.ndarray, np.ndarray]:
    """
    High-level zone planning using simplified EFE.

    Enumerates zone-action sequences and picks best based on EFE.

    Parameters
    ----------
    zoned_layout : ZonedLayout
        Zone layout
    my_zone : int
        Current zone
    other_zone : int
        Other agent's current zone
    my_goal_zone : int
        My goal zone
    other_goal_zone : int
        Other's goal zone
    alpha : float
        Empathy weight
    horizon : int
        Planning horizon (zone transitions)
    gamma : float
        Inverse temperature

    Returns
    -------
    best_action : ZoneAction
        First action of best policy
    G_values : np.ndarray
        G values for each first action
    q_pi : np.ndarray
        Policy posterior for each first action
    """
    import itertools

    actions = [ZoneAction.STAY, ZoneAction.MOVE_FORWARD, ZoneAction.MOVE_BACK]
    num_actions = len(actions)

    # For simplicity, evaluate single-step actions (greedy at zone level)
    # Full multi-step would be 3^H but zone transitions are coarse anyway
    G_values = np.zeros(num_actions)

    for i, action in enumerate(actions):
        G_values[i] = compute_zone_G(
            zoned_layout,
            my_zone,
            other_zone,
            my_goal_zone,
            action,
            alpha,
            other_goal_zone,
        )

    # Softmax policy selection
    log_q_pi = -gamma * G_values
    log_q_pi = log_q_pi - log_q_pi.max()
    q_pi = np.exp(log_q_pi)
    q_pi = q_pi / q_pi.sum()

    # Select best action
    best_idx = np.argmin(G_values)
    best_action = actions[best_idx]

    return best_action, G_values, q_pi


# =============================================================================
# Phase 3: Low-Level Within-Zone Planner
# =============================================================================

def get_subgoal(
    zoned_layout: ZonedLayout,
    current_zone: int,
    zone_action: ZoneAction,
    goal_zone: int,
    final_goal: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Determine subgoal based on zone action.

    Parameters
    ----------
    zoned_layout : ZonedLayout
        Zone layout
    current_zone : int
        Current zone
    zone_action : ZoneAction
        High-level zone action
    goal_zone : int
        Ultimate goal zone
    final_goal : Tuple[int, int]
        Final goal position

    Returns
    -------
    subgoal : Tuple[int, int]
        Grid cell to navigate toward
    """
    current_zone_obj = zoned_layout.get_zone(current_zone)
    if current_zone_obj is None:
        return final_goal

    if zone_action == ZoneAction.STAY:
        # Stay in zone - subgoal is center or final goal if in zone
        if final_goal in current_zone_obj.cells:
            return final_goal
        else:
            # Return a central cell
            center = current_zone_obj.get_center()
            # Find closest cell to center
            best_cell = min(current_zone_obj.cells,
                          key=lambda c: abs(c[0]-center[0]) + abs(c[1]-center[1]))
            return best_cell

    elif zone_action == ZoneAction.MOVE_FORWARD:
        # Moving toward goal - subgoal is exit point toward next zone
        next_zone = zoned_layout.get_next_zone_toward(current_zone, goal_zone)
        exit_points = current_zone_obj.get_exit_to(next_zone)
        if exit_points:
            return exit_points[0]
        else:
            # Fallback: final goal or center
            return final_goal if final_goal in current_zone_obj.cells else list(current_zone_obj.cells)[0]

    elif zone_action == ZoneAction.MOVE_BACK:
        # Yielding - move to exit point away from goal
        next_zone = zoned_layout.get_next_zone_away(current_zone, goal_zone)
        exit_points = current_zone_obj.get_exit_to(next_zone)
        if exit_points:
            return exit_points[0]
        else:
            # Just stay in place
            return list(current_zone_obj.cells)[0]

    return final_goal


def manhattan_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Compute Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def low_level_greedy_action(
    current_pos: Tuple[int, int],
    subgoal: Tuple[int, int],
    other_pos: Tuple[int, int],
    safe_cells: Set[Tuple[int, int]],
    width: int,
    height: int,
    collision_penalty: float = -30.0,
    alpha: float = 0.0,
) -> int:
    """
    Simple greedy action selection toward subgoal.

    For full EFE-based low-level planning, use the existing EmpathicLavaPlanner
    with a restricted state space. This greedy version is faster for testing.

    Parameters
    ----------
    current_pos : Tuple[int, int]
        Current position
    subgoal : Tuple[int, int]
        Target position
    other_pos : Tuple[int, int]
        Other agent's position
    safe_cells : Set[Tuple[int, int]]
        Set of safe cells
    width, height : int
        Grid dimensions
    collision_penalty : float
        Penalty for potential collision
    alpha : float
        Empathy weight

    Returns
    -------
    action : int
        Primitive action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY)
    """
    # Action effects
    # 0=UP (y-1), 1=DOWN (y+1), 2=LEFT (x-1), 3=RIGHT (x+1), 4=STAY
    dx = [0, 0, -1, 1, 0]
    dy = [-1, 1, 0, 0, 0]

    best_action = 4  # Default: STAY
    best_score = float('-inf')

    for action in range(5):
        nx = current_pos[0] + dx[action]
        ny = current_pos[1] + dy[action]

        # Clamp to grid
        nx = max(0, min(width - 1, nx))
        ny = max(0, min(height - 1, ny))
        next_pos = (nx, ny)

        # Check if safe
        if next_pos not in safe_cells:
            continue  # Skip lava

        # Score: negative distance to subgoal
        dist = manhattan_distance(next_pos, subgoal)
        score = -dist * 10  # Scale for readability

        # Collision avoidance
        if next_pos == other_pos:
            score += collision_penalty * (1 + alpha)  # Empathy increases collision aversion

        # Prefer progress
        current_dist = manhattan_distance(current_pos, subgoal)
        if dist < current_dist:
            score += 5  # Bonus for making progress

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


# =============================================================================
# Phase 4: Hierarchical Empathic Planner
# =============================================================================

@dataclass
class HierarchicalEmpathicPlanner:
    """
    Two-level hierarchical planner with empathy.

    High-level: Zone transition planning (coarse, long horizon)
    Low-level: Within-zone navigation (fine, short horizon)

    Attributes
    ----------
    zoned_layout : ZonedLayout
        Layout with zone decomposition
    goal_pos : Tuple[int, int]
        Final goal position
    safe_cells : Set[Tuple[int, int]]
        Set of safe (non-lava) cells
    width : int
        Grid width
    height : int
        Grid height
    alpha : float
        Empathy weight (0 = selfish, 1 = prosocial)
    alpha_other : float
        Believed empathy of other agent
    high_level_horizon : int
        Horizon for zone planning
    use_greedy_low_level : bool
        If True, use fast greedy action selection
        If False, use full EFE-based low-level planning (slower but better)
    """
    zoned_layout: ZonedLayout
    goal_pos: Tuple[int, int]
    safe_cells: Set[Tuple[int, int]]
    width: int
    height: int
    alpha: float = 0.5
    alpha_other: float = 0.0
    high_level_horizon: int = 3
    use_greedy_low_level: bool = True

    def get_goal_zone(self) -> int:
        """Get zone containing the goal."""
        zone_id = self.zoned_layout.get_zone_for_cell(self.goal_pos)
        return zone_id if zone_id is not None else 0

    def plan(
        self,
        my_pos: Tuple[int, int],
        other_pos: Tuple[int, int],
        other_goal_pos: Tuple[int, int],
    ) -> int:
        """
        Plan next action using two-level hierarchy.

        Parameters
        ----------
        my_pos : Tuple[int, int]
            My current position
        other_pos : Tuple[int, int]
            Other agent's position
        other_goal_pos : Tuple[int, int]
            Other agent's goal position

        Returns
        -------
        action : int
            Primitive action (0-4)
        """
        # 1. Determine current zones
        my_zone = self.zoned_layout.get_zone_for_cell(my_pos)
        other_zone = self.zoned_layout.get_zone_for_cell(other_pos)
        my_goal_zone = self.get_goal_zone()
        other_goal_zone = self.zoned_layout.get_zone_for_cell(other_goal_pos)

        # Handle edge cases
        if my_zone is None:
            my_zone = 0
        if other_zone is None:
            other_zone = 0
        if other_goal_zone is None:
            other_goal_zone = 0

        # 2. High-level: decide zone action
        zone_action, G_zone, q_zone = high_level_plan(
            self.zoned_layout,
            my_zone,
            other_zone,
            my_goal_zone,
            other_goal_zone,
            self.alpha,
            horizon=self.high_level_horizon,
        )

        # 3. Determine subgoal from zone action
        subgoal = get_subgoal(
            self.zoned_layout,
            my_zone,
            zone_action,
            my_goal_zone,
            self.goal_pos,
        )

        # 4. Low-level: navigate toward subgoal
        if self.use_greedy_low_level:
            action = low_level_greedy_action(
                my_pos,
                subgoal,
                other_pos,
                self.safe_cells,
                self.width,
                self.height,
                alpha=self.alpha,
            )
        else:
            # TODO: Use full EmpathicLavaPlanner with restricted state space
            # For now, fall back to greedy
            action = low_level_greedy_action(
                my_pos,
                subgoal,
                other_pos,
                self.safe_cells,
                self.width,
                self.height,
                alpha=self.alpha,
            )

        return action

    def plan_with_debug(
        self,
        my_pos: Tuple[int, int],
        other_pos: Tuple[int, int],
        other_goal_pos: Tuple[int, int],
    ) -> Dict:
        """
        Plan with debug information.

        Returns dict with action and intermediate values.
        """
        my_zone = self.zoned_layout.get_zone_for_cell(my_pos)
        other_zone = self.zoned_layout.get_zone_for_cell(other_pos)
        my_goal_zone = self.get_goal_zone()
        other_goal_zone = self.zoned_layout.get_zone_for_cell(other_goal_pos)

        if my_zone is None: my_zone = 0
        if other_zone is None: other_zone = 0
        if other_goal_zone is None: other_goal_zone = 0

        zone_action, G_zone, q_zone = high_level_plan(
            self.zoned_layout, my_zone, other_zone, my_goal_zone,
            other_goal_zone, self.alpha, self.high_level_horizon,
        )

        subgoal = get_subgoal(
            self.zoned_layout, my_zone, zone_action, my_goal_zone, self.goal_pos
        )

        action = low_level_greedy_action(
            my_pos, subgoal, other_pos, self.safe_cells,
            self.width, self.height, alpha=self.alpha,
        )

        return {
            "action": action,
            "my_zone": my_zone,
            "other_zone": other_zone,
            "my_goal_zone": my_goal_zone,
            "zone_action": zone_action,
            "subgoal": subgoal,
            "G_zone": G_zone,
            "q_zone": q_zone,
        }
