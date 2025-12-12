"""
JAX-accelerated hierarchical spatial planner for multi-agent coordination.

This module provides a JAX reimplementation of the hierarchical planner,
enabling JIT compilation for significant speedups.

Architecture:
- HIGH LEVEL (Zone Planning): JAX-compiled EFE over zone transitions
- LOW LEVEL (Within-Zone): Uses existing JAX EFE primitives with restricted state space

Key optimizations:
- Zone-level planning is fully JIT-compiled
- Low-level uses vmap over action evaluations
- State-to-zone mappings use static arrays for traceability
"""

import jax
import jax.numpy as jnp
from jax import lax, vmap
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Set, List
from functools import partial

from tom.planning.jax_si_empathy_lava import (
    propagate_belief_jax,
    expected_pragmatic_utility_jax,
    epistemic_info_gain_jax,
)


# =============================================================================
# Subgoal-Oriented Preferences
# =============================================================================

@jax.jit
def create_subgoal_C_loc_jax(
    subgoal_state: int,
    original_C_loc: jnp.ndarray,
    width: int,
    lava_penalty: float = -100.0,
    distance_cost: float = -2.0,
    subgoal_reward: float = 20.0,
) -> jnp.ndarray:
    """
    Create C_loc with preferences toward subgoal (JAX-compiled).

    Keeps lava penalties from original C_loc but replaces distance
    shaping to point toward subgoal instead of final goal.

    Parameters
    ----------
    subgoal_state : int
        Target state index
    original_C_loc : jnp.ndarray
        Original C_loc (used to identify lava cells)
    width : int
        Grid width
    lava_penalty : float
        Penalty for lava cells (should match original)
    distance_cost : float
        Cost per Manhattan distance unit from subgoal
    subgoal_reward : float
        Reward for being at subgoal

    Returns
    -------
    C_loc : jnp.ndarray
        Subgoal-oriented preference vector
    """
    num_states = original_C_loc.shape[0]
    states = jnp.arange(num_states)

    # Subgoal coordinates
    subgoal_x = subgoal_state % width
    subgoal_y = subgoal_state // width

    # State coordinates
    state_x = states % width
    state_y = states // width

    # Manhattan distance to subgoal
    manhattan_dist = jnp.abs(state_x - subgoal_x) + jnp.abs(state_y - subgoal_y)

    # Distance-based preferences (closer = better)
    distance_prefs = distance_cost * manhattan_dist

    # Subgoal reward (bonus for being at subgoal)
    at_subgoal = (states == subgoal_state).astype(jnp.float32)
    subgoal_prefs = at_subgoal * subgoal_reward

    # Identify lava cells from original C_loc (they have very negative values)
    is_lava = original_C_loc < -50.0  # Lava has -100

    # Combine: use lava penalty where lava, else use distance shaping
    C_loc = jnp.where(is_lava, lava_penalty, distance_prefs + subgoal_prefs)

    return C_loc


# =============================================================================
# Zone Infrastructure (JAX-compatible)
# =============================================================================

@dataclass
class JaxZonedLayout:
    """
    JAX-compatible zoned layout representation.

    All mappings are converted to JAX arrays for JIT compilation.

    Attributes
    ----------
    width : int
        Grid width
    height : int
        Grid height
    num_zones : int
        Number of spatial zones
    num_states : int
        Total number of grid cells
    cell_to_zone : jnp.ndarray
        Array mapping state index to zone ID [num_states]
    zone_adjacency : jnp.ndarray
        Adjacency matrix [num_zones, num_zones], 1 if zones are adjacent
    zone_is_bottleneck : jnp.ndarray
        Boolean array [num_zones] indicating bottleneck zones
    zone_centers : jnp.ndarray
        Center positions for each zone [num_zones, 2] (x, y)
    exit_points : jnp.ndarray
        Exit point states for zone transitions [num_zones, num_zones]
        exit_points[from_zone, to_zone] = state index of exit point
        -1 if no direct transition
    goal_zone : int
        Zone containing the goal (for this agent)
    """
    width: int
    height: int
    num_zones: int
    num_states: int
    cell_to_zone: jnp.ndarray
    zone_adjacency: jnp.ndarray
    zone_is_bottleneck: jnp.ndarray
    zone_centers: jnp.ndarray
    exit_points: jnp.ndarray
    goal_zone: int


def create_jax_vertical_bottleneck_layout(
    width: int = 6,
    height: int = 8,
    goal_pos: Tuple[int, int] = None,
) -> JaxZonedLayout:
    """
    Create JAX-compatible zone layout for vertical_bottleneck.

    Layout:
    ~ ~ ~ . ~ ~
    ~ ~ ~ . ~ ~
    . . . . . .   ZONE 0: top_wide (y=2, all x)
    ~ ~ ~ . ~ ~
    ~ ~ ~ . ~ ~   ZONE 1: bottleneck (x=3, y!=2,5)
    . . . . . .   ZONE 2: bottom_wide (y=5, all x)
    ~ ~ ~ . ~ ~
    ~ ~ ~ . ~ ~

    Parameters
    ----------
    width : int
        Grid width (default 6)
    height : int
        Grid height (default 8)
    goal_pos : Tuple[int, int], optional
        Goal position to determine goal zone

    Returns
    -------
    JaxZonedLayout
        JAX-compatible zoned layout
    """
    num_states = width * height
    mid_x = width // 2
    wide_top_y = 2
    wide_bottom_y = height - 3

    # Build cell-to-zone mapping
    cell_to_zone = np.full(num_states, -1, dtype=np.int32)

    def pos_to_idx(x, y):
        return y * width + x

    def idx_to_pos(idx):
        return idx % width, idx // width

    # Zone 0: Top wide area (y=wide_top_y, all x)
    for x in range(width):
        cell_to_zone[pos_to_idx(x, wide_top_y)] = 0

    # Zone 1: Bottleneck corridor (x=mid_x for y != wide rows)
    for y in range(height):
        if y != wide_top_y and y != wide_bottom_y:
            cell_to_zone[pos_to_idx(mid_x, y)] = 1

    # Zone 2: Bottom wide area (y=wide_bottom_y, all x)
    for x in range(width):
        cell_to_zone[pos_to_idx(x, wide_bottom_y)] = 2

    # Zone adjacency: 0 <-> 1 <-> 2
    zone_adjacency = np.array([
        [0, 1, 0],  # Zone 0: adjacent to zone 1
        [1, 0, 1],  # Zone 1: adjacent to zones 0 and 2
        [0, 1, 0],  # Zone 2: adjacent to zone 1
    ], dtype=np.float32)

    # Bottleneck flags
    zone_is_bottleneck = np.array([False, True, False])

    # Zone centers (approximate)
    zone_centers = np.array([
        [width / 2, wide_top_y],      # Zone 0: center of top wide
        [mid_x, height / 2],          # Zone 1: center of bottleneck
        [width / 2, wide_bottom_y],   # Zone 2: center of bottom wide
    ], dtype=np.float32)

    # Exit points for zone transitions
    # exit_points[from, to] = state index IN THE DESTINATION ZONE to navigate toward
    # CRITICAL: Exit points must be in the destination zone, not the source zone!
    exit_points = np.full((3, 3), -1, dtype=np.int32)

    # From zone 0 to zone 1: exit INTO zone 1 (one step below top wide row)
    exit_points[0, 1] = pos_to_idx(mid_x, wide_top_y + 1)  # (3, 3) - in bottleneck
    # From zone 1 to zone 0: exit INTO zone 0 (into top wide row)
    exit_points[1, 0] = pos_to_idx(mid_x, wide_top_y)      # (3, 2) - in top_wide
    # From zone 1 to zone 2: exit INTO zone 2 (into bottom wide row)
    exit_points[1, 2] = pos_to_idx(mid_x, wide_bottom_y)   # (3, 5) - in bottom_wide
    # From zone 2 to zone 1: exit INTO zone 1 (one step above bottom wide row)
    exit_points[2, 1] = pos_to_idx(mid_x, wide_bottom_y - 1)  # (3, 4) - in bottleneck

    # Determine goal zone
    goal_zone = 0
    if goal_pos is not None:
        goal_idx = pos_to_idx(goal_pos[0], goal_pos[1])
        if 0 <= goal_idx < num_states:
            goal_zone = int(cell_to_zone[goal_idx])
            if goal_zone < 0:
                goal_zone = 0  # Default if goal not in any zone

    return JaxZonedLayout(
        width=width,
        height=height,
        num_zones=3,
        num_states=num_states,
        cell_to_zone=jnp.array(cell_to_zone),
        zone_adjacency=jnp.array(zone_adjacency),
        zone_is_bottleneck=jnp.array(zone_is_bottleneck),
        zone_centers=jnp.array(zone_centers),
        exit_points=jnp.array(exit_points),
        goal_zone=goal_zone,
    )


def create_jax_symmetric_bottleneck_layout(
    width: int = 10,
    height: int = None,  # Ignored - symmetric_bottleneck is always height=4
    goal_pos: Tuple[int, int] = None,
) -> JaxZonedLayout:
    """
    Create JAX-compatible zone layout for symmetric_bottleneck.

    Layout (width=10):
    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    0 . . . B B . . . 1   (row 1 - agents start opposite ends)
    . . . . B B . . . .   (row 2)
    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    Zones:
    - Zone 0: Left wide (x < 4)
    - Zone 1: Bottleneck (4 <= x < 6)
    - Zone 2: Right wide (x >= 6)
    """
    height = 4
    num_states = width * height
    bottleneck_start = width // 2 - 1
    bottleneck_end = width // 2 + 1

    cell_to_zone = np.full(num_states, -1, dtype=np.int32)

    def pos_to_idx(x, y):
        return y * width + x

    # Zone 0: Left wide
    for x in range(bottleneck_start):
        cell_to_zone[pos_to_idx(x, 1)] = 0
        cell_to_zone[pos_to_idx(x, 2)] = 0

    # Zone 1: Bottleneck (only row 1 safe typically, but we include all safe)
    for x in range(bottleneck_start, bottleneck_end):
        cell_to_zone[pos_to_idx(x, 1)] = 1

    # Zone 2: Right wide
    for x in range(bottleneck_end, width):
        cell_to_zone[pos_to_idx(x, 1)] = 2
        cell_to_zone[pos_to_idx(x, 2)] = 2

    zone_adjacency = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32)

    zone_is_bottleneck = np.array([False, True, False])

    zone_centers = np.array([
        [bottleneck_start / 2, 1.5],
        [(bottleneck_start + bottleneck_end) / 2, 1],
        [(bottleneck_end + width) / 2, 1.5],
    ], dtype=np.float32)

    # Exit points: must be IN THE DESTINATION ZONE
    exit_points = np.full((3, 3), -1, dtype=np.int32)
    # From zone 0 to zone 1: exit INTO zone 1 (first bottleneck cell)
    exit_points[0, 1] = pos_to_idx(bottleneck_start, 1)      # x=4, in bottleneck
    # From zone 1 to zone 0: exit INTO zone 0 (last left-wide cell)
    exit_points[1, 0] = pos_to_idx(bottleneck_start - 1, 1)  # x=3, in left_wide
    # From zone 1 to zone 2: exit INTO zone 2 (first right-wide cell)
    exit_points[1, 2] = pos_to_idx(bottleneck_end, 1)        # x=6, in right_wide
    # From zone 2 to zone 1: exit INTO zone 1 (last bottleneck cell)
    exit_points[2, 1] = pos_to_idx(bottleneck_end - 1, 1)    # x=5, in bottleneck

    goal_zone = 0
    if goal_pos is not None:
        goal_idx = pos_to_idx(goal_pos[0], goal_pos[1])
        if 0 <= goal_idx < num_states:
            goal_zone = int(cell_to_zone[goal_idx])
            if goal_zone < 0:
                goal_zone = 0

    return JaxZonedLayout(
        width=width,
        height=height,
        num_zones=3,
        num_states=num_states,
        cell_to_zone=jnp.array(cell_to_zone),
        zone_adjacency=jnp.array(zone_adjacency),
        zone_is_bottleneck=jnp.array(zone_is_bottleneck),
        zone_centers=jnp.array(zone_centers),
        exit_points=jnp.array(exit_points),
        goal_zone=goal_zone,
    )


def create_jax_narrow_layout(
    width: int = 6,
    height: int = None,  # Ignored - narrow is always height=3
    goal_pos: Tuple[int, int] = None,
) -> JaxZonedLayout:
    """
    Create JAX-compatible zone layout for narrow corridor.

    Layout:
    ~ ~ ~ ~ ~ ~
    0 . . . . 1   (single row, y=1)
    ~ ~ ~ ~ ~ ~

    Zones (thirds):
    - Zone 0: Left third
    - Zone 1: Middle (contested)
    - Zone 2: Right third
    """
    height = 3
    num_states = width * height
    safe_y = 1
    third = width // 3

    cell_to_zone = np.full(num_states, -1, dtype=np.int32)

    def pos_to_idx(x, y):
        return y * width + x

    for x in range(third):
        cell_to_zone[pos_to_idx(x, safe_y)] = 0
    for x in range(third, 2 * third):
        cell_to_zone[pos_to_idx(x, safe_y)] = 1
    for x in range(2 * third, width):
        cell_to_zone[pos_to_idx(x, safe_y)] = 2

    zone_adjacency = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.float32)

    zone_is_bottleneck = np.array([False, True, False])

    zone_centers = np.array([
        [third / 2, safe_y],
        [third * 1.5, safe_y],
        [third * 2.5, safe_y],
    ], dtype=np.float32)

    # Exit points: must be IN THE DESTINATION ZONE
    exit_points = np.full((3, 3), -1, dtype=np.int32)
    # From zone 0 to zone 1: exit INTO zone 1 (first middle cell)
    exit_points[0, 1] = pos_to_idx(third, safe_y)           # x=2, in middle zone
    # From zone 1 to zone 0: exit INTO zone 0 (last left cell)
    exit_points[1, 0] = pos_to_idx(third - 1, safe_y)       # x=1, in left zone
    # From zone 1 to zone 2: exit INTO zone 2 (first right cell)
    exit_points[1, 2] = pos_to_idx(2 * third, safe_y)       # x=4, in right zone
    # From zone 2 to zone 1: exit INTO zone 1 (last middle cell)
    exit_points[2, 1] = pos_to_idx(2 * third - 1, safe_y)   # x=3, in middle zone

    goal_zone = 0
    if goal_pos is not None:
        goal_idx = pos_to_idx(goal_pos[0], goal_pos[1])
        if 0 <= goal_idx < num_states:
            goal_zone = int(cell_to_zone[goal_idx])
            if goal_zone < 0:
                goal_zone = 0

    return JaxZonedLayout(
        width=width,
        height=height,
        num_zones=3,
        num_states=num_states,
        cell_to_zone=jnp.array(cell_to_zone),
        zone_adjacency=jnp.array(zone_adjacency),
        zone_is_bottleneck=jnp.array(zone_is_bottleneck),
        zone_centers=jnp.array(zone_centers),
        exit_points=jnp.array(exit_points),
        goal_zone=goal_zone,
    )


# =============================================================================
# Zone Layout Registry
# =============================================================================

JAX_ZONED_LAYOUTS = {
    "vertical_bottleneck": create_jax_vertical_bottleneck_layout,
    "symmetric_bottleneck": create_jax_symmetric_bottleneck_layout,
    "narrow": create_jax_narrow_layout,
}


def get_jax_zoned_layout(layout_name: str, goal_pos: Tuple[int, int] = None, **kwargs) -> JaxZonedLayout:
    """Get a JAX-compatible zoned layout by name."""
    if layout_name not in JAX_ZONED_LAYOUTS:
        available = list(JAX_ZONED_LAYOUTS.keys())
        raise ValueError(f"No JAX zone definition for layout: {layout_name}. Available: {available}")
    return JAX_ZONED_LAYOUTS[layout_name](goal_pos=goal_pos, **kwargs)


def has_jax_zoned_layout(layout_name: str) -> bool:
    """Check if a layout has JAX zone definitions."""
    return layout_name in JAX_ZONED_LAYOUTS


# =============================================================================
# High-Level Zone Planning (JAX)
# =============================================================================

# Zone actions
ZONE_STAY = 0
ZONE_FORWARD = 1
ZONE_BACK = 2


@jax.jit
def get_zone_from_belief(
    qs: jnp.ndarray,
    cell_to_zone: jnp.ndarray,
) -> int:
    """
    Get most likely zone given belief distribution.

    Parameters
    ----------
    qs : jnp.ndarray
        Belief over states [num_states]
    cell_to_zone : jnp.ndarray
        Mapping from state to zone [num_states]

    Returns
    -------
    zone : int
        Most likely zone
    """
    # Get most likely state
    most_likely_state = jnp.argmax(qs)
    return cell_to_zone[most_likely_state]


@jax.jit
def compute_zone_distance_jax(
    from_zone: int,
    to_zone: int,
    zone_adjacency: jnp.ndarray,
) -> int:
    """
    Compute zone distance using adjacency (simplified BFS).

    For 3 zones in a line (0-1-2), distance is |from - to|.
    """
    # Simple implementation for linear zone topology
    return jnp.abs(from_zone - to_zone)


@jax.jit
def get_next_zone_toward_jax(
    from_zone: int,
    to_zone: int,
    zone_adjacency: jnp.ndarray,
) -> int:
    """Get next zone when moving toward target zone."""
    # For linear topology: move toward target
    diff = to_zone - from_zone
    next_zone = jnp.where(
        diff > 0,
        from_zone + 1,
        jnp.where(diff < 0, from_zone - 1, from_zone)
    )
    return jnp.clip(next_zone, 0, zone_adjacency.shape[0] - 1)


@jax.jit
def get_next_zone_away_jax(
    from_zone: int,
    to_zone: int,
    zone_adjacency: jnp.ndarray,
) -> int:
    """Get next zone when moving away from target zone (yielding)."""
    diff = to_zone - from_zone
    # Move in opposite direction
    next_zone = jnp.where(
        diff > 0,
        from_zone - 1,
        jnp.where(diff < 0, from_zone + 1, from_zone)
    )
    return jnp.clip(next_zone, 0, zone_adjacency.shape[0] - 1)


@jax.jit
def apply_zone_action_jax(
    current_zone: int,
    action: int,
    goal_zone: int,
    zone_adjacency: jnp.ndarray,
) -> int:
    """
    Apply zone-level action to get next zone.

    Parameters
    ----------
    current_zone : int
        Current zone ID
    action : int
        Zone action (0=STAY, 1=FORWARD, 2=BACK)
    goal_zone : int
        Target goal zone
    zone_adjacency : jnp.ndarray
        Zone adjacency matrix

    Returns
    -------
    next_zone : int
        Zone after taking action
    """
    stay_result = current_zone
    forward_result = get_next_zone_toward_jax(current_zone, goal_zone, zone_adjacency)
    back_result = get_next_zone_away_jax(current_zone, goal_zone, zone_adjacency)

    return jnp.where(
        action == ZONE_STAY,
        stay_result,
        jnp.where(action == ZONE_FORWARD, forward_result, back_result)
    )


@jax.jit
def compute_zone_G_jax(
    my_zone: int,
    other_zone: int,
    my_goal_zone: int,
    other_goal_zone: int,
    action: int,
    alpha: float,
    zone_adjacency: jnp.ndarray,
    zone_is_bottleneck: jnp.ndarray,
    bottleneck_collision_cost: float = -15.0,  # Tuned so empathetic agents yield, selfish push through
    zone_distance_cost: float = -10.0,
    goal_zone_reward: float = 100.0,
) -> float:
    """
    Compute EFE for a zone-level action (JAX-compiled).

    Components:
    1. Distance to goal zone (closer = better)
    2. Goal zone reward (being in goal zone)
    3. Bottleneck collision risk (both agents in bottleneck = bad)
    4. Empathy: consider other agent's progress

    Parameters
    ----------
    my_zone : int
        Current zone
    other_zone : int
        Other agent's zone
    my_goal_zone : int
        My goal zone
    other_goal_zone : int
        Other's goal zone
    action : int
        Zone action (0=STAY, 1=FORWARD, 2=BACK)
    alpha : float
        Empathy weight
    zone_adjacency : jnp.ndarray
        Zone adjacency matrix
    zone_is_bottleneck : jnp.ndarray
        Boolean array of bottleneck flags
    bottleneck_collision_cost : float
        Penalty for both in bottleneck
    zone_distance_cost : float
        Cost per zone from goal
    goal_zone_reward : float
        Reward for being in goal zone

    Returns
    -------
    G : float
        Expected free energy (lower = better)
    """
    # Apply action to get next zone
    next_zone = apply_zone_action_jax(my_zone, action, my_goal_zone, zone_adjacency)

    # 1. Distance to goal
    dist_to_goal = compute_zone_distance_jax(next_zone, my_goal_zone, zone_adjacency)
    distance_utility = zone_distance_cost * dist_to_goal

    # 2. Goal zone reward
    goal_utility = jnp.where(next_zone == my_goal_zone, goal_zone_reward, 0.0)

    # 3. Bottleneck collision risk (for my own EFE)
    next_is_bottleneck = zone_is_bottleneck[next_zone]

    # Check if other is adjacent to our next zone
    other_adjacent = zone_adjacency[next_zone, other_zone] > 0

    collision_utility = jnp.where(
        next_is_bottleneck & (other_zone == next_zone),
        bottleneck_collision_cost,  # Both in same bottleneck
        jnp.where(
            next_is_bottleneck & other_adjacent,
            bottleneck_collision_cost * 0.5,  # High risk
            0.0
        )
    )

    # My utility (G_i component)
    my_utility = distance_utility + goal_utility + collision_utility

    # 4. Empathy via Theory of Mind: compute j's best response EFE given my action
    # Following the flat planner structure (jax_si_empathy_lava.py):
    #   G_social = G_i + alpha * G_j_best
    #
    # For each j zone action (STAY, FORWARD, BACK), compute j's EFE given my next_zone,
    # then pick j's best response.

    # j's next zone for each of j's possible actions
    j_next_if_stay = other_zone
    j_next_if_forward = get_next_zone_toward_jax(other_zone, other_goal_zone, zone_adjacency)
    j_next_if_back = get_next_zone_away_jax(other_zone, other_goal_zone, zone_adjacency)

    # j's distance utility for each action
    j_dist_stay = compute_zone_distance_jax(j_next_if_stay, other_goal_zone, zone_adjacency)
    j_dist_forward = compute_zone_distance_jax(j_next_if_forward, other_goal_zone, zone_adjacency)
    j_dist_back = compute_zone_distance_jax(j_next_if_back, other_goal_zone, zone_adjacency)

    j_util_stay = zone_distance_cost * j_dist_stay
    j_util_forward = zone_distance_cost * j_dist_forward
    j_util_back = zone_distance_cost * j_dist_back

    # j's goal zone reward
    j_goal_stay = jnp.where(j_next_if_stay == other_goal_zone, goal_zone_reward, 0.0)
    j_goal_forward = jnp.where(j_next_if_forward == other_goal_zone, goal_zone_reward, 0.0)
    j_goal_back = jnp.where(j_next_if_back == other_goal_zone, goal_zone_reward, 0.0)

    # j's collision penalty (j collides with me if j enters same zone as my next_zone)
    j_coll_stay = jnp.where(
        zone_is_bottleneck[j_next_if_stay] & (j_next_if_stay == next_zone),
        bottleneck_collision_cost, 0.0
    )
    j_coll_forward = jnp.where(
        zone_is_bottleneck[j_next_if_forward] & (j_next_if_forward == next_zone),
        bottleneck_collision_cost, 0.0
    )
    j_coll_back = jnp.where(
        zone_is_bottleneck[j_next_if_back] & (j_next_if_back == next_zone),
        bottleneck_collision_cost, 0.0
    )

    # j's total utility for each action
    j_total_stay = j_util_stay + j_goal_stay + j_coll_stay
    j_total_forward = j_util_forward + j_goal_forward + j_coll_forward
    j_total_back = j_util_back + j_goal_back + j_coll_back

    # j's EFE for each action (G = -utility, lower is better)
    G_j_stay = -j_total_stay
    G_j_forward = -j_total_forward
    G_j_back = -j_total_back

    # j's best response: pick action with lowest G
    G_j_best = jnp.minimum(G_j_stay, jnp.minimum(G_j_forward, G_j_back))

    # Social EFE: G_social = G_i + alpha * G_j_best
    # Since we're computing utility (higher = better), and G = -utility:
    # total_utility = my_utility, and we add alpha * (-G_j_best) = alpha * j's best utility
    # But G_social = G_i + alpha * G_j_best, so:
    # -total_utility_social = -my_utility + alpha * G_j_best
    # total_utility_social = my_utility - alpha * G_j_best
    total_utility = my_utility - alpha * G_j_best

    # Convert to EFE (lower = better)
    G = -total_utility
    return G


@jax.jit
def high_level_plan_jax(
    my_zone: int,
    other_zone: int,
    my_goal_zone: int,
    other_goal_zone: int,
    alpha: float,
    zone_adjacency: jnp.ndarray,
    zone_is_bottleneck: jnp.ndarray,
    gamma: float = 8.0,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    High-level zone planning using JAX-compiled EFE.

    Parameters
    ----------
    my_zone : int
        Current zone
    other_zone : int
        Other's zone
    my_goal_zone : int
        My goal zone
    other_goal_zone : int
        Other's goal zone
    alpha : float
        Empathy weight
    zone_adjacency : jnp.ndarray
        Zone adjacency matrix
    zone_is_bottleneck : jnp.ndarray
        Bottleneck flags
    gamma : float
        Inverse temperature

    Returns
    -------
    best_action : int
        Best zone action (0=STAY, 1=FORWARD, 2=BACK)
    G_values : jnp.ndarray
        G values for each action [3]
    q_pi : jnp.ndarray
        Policy posterior [3]
    """
    actions = jnp.array([ZONE_STAY, ZONE_FORWARD, ZONE_BACK])

    # Compute G for each action
    def compute_G_for_action(action):
        return compute_zone_G_jax(
            my_zone, other_zone, my_goal_zone, other_goal_zone,
            action, alpha, zone_adjacency, zone_is_bottleneck,
        )

    G_values = vmap(compute_G_for_action)(actions)

    # Softmax policy selection
    log_q_pi = -gamma * G_values
    log_q_pi = log_q_pi - log_q_pi.max()
    q_pi = jnp.exp(log_q_pi)
    q_pi = q_pi / q_pi.sum()

    # Best action
    best_idx = jnp.argmin(G_values)
    best_action = actions[best_idx]

    return best_action, G_values, q_pi


# =============================================================================
# Low-Level Within-Zone Planning (JAX)
# =============================================================================

@jax.jit
def get_subgoal_state_jax(
    current_zone: int,
    zone_action: int,
    goal_zone: int,
    goal_state: int,
    exit_points: jnp.ndarray,
    zone_adjacency: jnp.ndarray,
    cell_to_zone: jnp.ndarray,
) -> int:
    """
    Determine subgoal state based on zone action (JAX-compiled).

    Parameters
    ----------
    current_zone : int
        Current zone
    zone_action : int
        High-level zone action
    goal_zone : int
        Ultimate goal zone
    goal_state : int
        Final goal state index
    exit_points : jnp.ndarray
        Exit point states [num_zones, num_zones]
    zone_adjacency : jnp.ndarray
        Zone adjacency matrix
    cell_to_zone : jnp.ndarray
        Cell to zone mapping

    Returns
    -------
    subgoal_state : int
        State to navigate toward
    """
    # If staying, subgoal is goal if in same zone, else stay put
    goal_in_current_zone = cell_to_zone[goal_state] == current_zone
    stay_subgoal = jnp.where(goal_in_current_zone, goal_state, goal_state)

    # If moving forward, subgoal is exit toward next zone
    next_zone_forward = get_next_zone_toward_jax(current_zone, goal_zone, zone_adjacency)
    forward_subgoal = exit_points[current_zone, next_zone_forward]
    # Handle -1 (no exit): use goal_state as fallback
    forward_subgoal = jnp.where(forward_subgoal >= 0, forward_subgoal, goal_state)

    # If moving back (yielding), subgoal is exit away from goal
    next_zone_back = get_next_zone_away_jax(current_zone, goal_zone, zone_adjacency)
    back_subgoal = exit_points[current_zone, next_zone_back]
    back_subgoal = jnp.where(back_subgoal >= 0, back_subgoal, goal_state)

    return jnp.where(
        zone_action == ZONE_STAY,
        stay_subgoal,
        jnp.where(zone_action == ZONE_FORWARD, forward_subgoal, back_subgoal)
    )


@jax.jit
def compute_low_level_G_jax(
    qs_self: jnp.ndarray,
    qs_other: jnp.ndarray,
    action: int,
    B: jnp.ndarray,
    A_loc: jnp.ndarray,
    C_loc: jnp.ndarray,
    A_cell_collision: jnp.ndarray,
    C_cell_collision: jnp.ndarray,
    A_edge: jnp.ndarray,
    C_edge: jnp.ndarray,
    A_edge_collision: jnp.ndarray,
    C_edge_collision: jnp.ndarray,
    action_other: int,
    alpha: float,
    eps: float = 1e-16,
) -> float:
    """
    Compute EFE for low-level action (JAX-compiled).

    Uses full active inference EFE with:
    - Pragmatic value (location, collision preferences)
    - Epistemic value (information gain)

    The C_loc parameter should already encode subgoal preferences
    (created via create_subgoal_C_loc_jax).

    Parameters
    ----------
    qs_self : jnp.ndarray
        Belief over own state [num_states]
    qs_other : jnp.ndarray
        Belief over other's state [num_states]
    action : int
        Action to evaluate
    B : jnp.ndarray
        Transition model [s', s, s_other, a]
    A_loc : jnp.ndarray
        Location observation model
    C_loc : jnp.ndarray
        Location preferences (should include subgoal preferences)
    A_cell_collision : jnp.ndarray
        Cell collision observation model
    C_cell_collision : jnp.ndarray
        Cell collision preferences
    A_edge : jnp.ndarray
        Edge observation model
    C_edge : jnp.ndarray
        Edge preferences
    A_edge_collision : jnp.ndarray
        Edge collision model
    C_edge_collision : jnp.ndarray
        Edge collision preferences
    action_other : int
        Predicted other agent's action (for edge collision)
    alpha : float
        Empathy weight
    eps : float
        Numerical stability

    Returns
    -------
    G : float
        Expected free energy
    """
    # Propagate belief
    qs_self_next = propagate_belief_jax(qs_self, B, action, qs_other, eps)

    # Pragmatic utility using existing function
    # C_loc now contains subgoal-oriented preferences (distance shaping + lava penalties)
    pragmatic = expected_pragmatic_utility_jax(
        qs_self_current=qs_self,
        qs_other_current=qs_other,
        qs_self_next=qs_self_next,
        qs_other_next=qs_other,  # Assume other doesn't move in our prediction
        action_self=action,
        action_other=action_other,
        A_loc=A_loc,
        C_loc=C_loc,
        A_edge=A_edge,
        C_edge=C_edge,
        A_cell_collision=A_cell_collision,
        C_cell_collision=C_cell_collision * (1.0 + alpha),  # Empathy scaling
        A_edge_collision=A_edge_collision,
        C_edge_collision=C_edge_collision * (1.0 + alpha),
    )

    # Epistemic value
    epistemic = epistemic_info_gain_jax(qs_self, A_loc, eps)

    # Total EFE (lower = better)
    # Subgoal preferences are now encoded in C_loc, so pragmatic value
    # naturally steers agent toward subgoal
    G = -pragmatic - epistemic
    return G


@jax.jit
def low_level_plan_jax(
    qs_self: jnp.ndarray,
    qs_other: jnp.ndarray,
    subgoal_state: int,
    B: jnp.ndarray,
    A_loc: jnp.ndarray,
    C_loc_original: jnp.ndarray,
    A_cell_collision: jnp.ndarray,
    C_cell_collision: jnp.ndarray,
    A_edge: jnp.ndarray,
    C_edge: jnp.ndarray,
    A_edge_collision: jnp.ndarray,
    C_edge_collision: jnp.ndarray,
    alpha: float,
    width: int,
    gamma: float = 8.0,
    eps: float = 1e-16,
) -> Tuple[int, jnp.ndarray, jnp.ndarray]:
    """
    Low-level action planning using EFE (JAX-compiled).

    Creates a subgoal-oriented C_loc from the original C_loc, then uses
    standard EFE computation to select actions.

    Parameters
    ----------
    qs_self : jnp.ndarray
        Belief over own state
    qs_other : jnp.ndarray
        Belief over other's state
    subgoal_state : int
        Target state from zone planning
    B : jnp.ndarray
        Transition model
    A_loc : jnp.ndarray
        Location observation model
    C_loc_original : jnp.ndarray
        Original location preferences (used to identify lava cells)
    A_cell_collision, C_cell_collision : jnp.ndarray
        Cell collision model and preferences
    A_edge, C_edge : jnp.ndarray
        Edge model and preferences
    A_edge_collision, C_edge_collision : jnp.ndarray
        Edge collision model and preferences
    alpha : float
        Empathy weight
    width : int
        Grid width for subgoal C_loc creation
    gamma : float
        Inverse temperature
    eps : float
        Numerical stability

    Returns
    -------
    best_action : int
        Best primitive action (0-4)
    G_values : jnp.ndarray
        EFE for each action [5]
    q_pi : jnp.ndarray
        Policy posterior [5]
    """
    # Create subgoal-oriented C_loc
    # This encodes preferences toward the subgoal while preserving lava penalties
    C_loc_subgoal = create_subgoal_C_loc_jax(
        subgoal_state=subgoal_state,
        original_C_loc=C_loc_original,
        width=width,
    )

    actions = jnp.arange(5)  # UP, DOWN, LEFT, RIGHT, STAY

    # Compute G for each action (assume other stays for edge collision calc)
    def compute_G_for_action(action):
        return compute_low_level_G_jax(
            qs_self, qs_other, action,
            B, A_loc, C_loc_subgoal, A_cell_collision, C_cell_collision,
            A_edge, C_edge, A_edge_collision, C_edge_collision,
            action_other=4,  # Assume other stays
            alpha=alpha,
            eps=eps,
        )

    G_values = vmap(compute_G_for_action)(actions)

    # Softmax
    log_q_pi = -gamma * G_values
    log_q_pi = log_q_pi - log_q_pi.max()
    q_pi = jnp.exp(log_q_pi)
    q_pi = q_pi / q_pi.sum()

    best_action = jnp.argmin(G_values)

    return best_action, G_values, q_pi


# =============================================================================
# Full Hierarchical Planner (JAX)
# =============================================================================

@dataclass
class JaxHierarchicalPlanner:
    """
    JAX-accelerated hierarchical empathic planner.

    Two-level planning:
    - High-level: Zone transition decisions
    - Low-level: Within-zone navigation using full EFE

    Attributes
    ----------
    zoned_layout : JaxZonedLayout
        JAX-compatible zone layout
    goal_state : int
        Goal state index
    alpha : float
        Empathy weight (0=selfish, 1=prosocial)
    alpha_other : float
        Believed empathy of other agent
    gamma : float
        Inverse temperature
    B : jnp.ndarray
        Transition model
    A_loc, C_loc : jnp.ndarray
        Location observation model and preferences
    A_cell_collision, C_cell_collision : jnp.ndarray
        Cell collision model and preferences
    A_edge, C_edge : jnp.ndarray
        Edge model and preferences
    A_edge_collision, C_edge_collision : jnp.ndarray
        Edge collision model and preferences
    """
    zoned_layout: JaxZonedLayout
    goal_state: int
    alpha: float
    alpha_other: float
    gamma: float
    B: jnp.ndarray
    A_loc: jnp.ndarray
    C_loc: jnp.ndarray
    A_cell_collision: jnp.ndarray
    C_cell_collision: jnp.ndarray
    A_edge: jnp.ndarray
    C_edge: jnp.ndarray
    A_edge_collision: jnp.ndarray
    C_edge_collision: jnp.ndarray

    @classmethod
    def from_model(
        cls,
        model,  # LavaModel
        layout_name: str,
        alpha: float = 0.5,
        alpha_other: float = 0.0,
        gamma: float = 8.0,
    ) -> "JaxHierarchicalPlanner":
        """
        Create planner from LavaModel.

        Parameters
        ----------
        model : LavaModel
            Agent's generative model
        layout_name : str
            Name of layout for zone decomposition
        alpha : float
            Empathy weight
        alpha_other : float
            Believed empathy of other agent
        gamma : float
            Inverse temperature

        Returns
        -------
        JaxHierarchicalPlanner
            Configured planner
        """
        # Get goal position
        goal_pos = (model.goal_x, model.goal_y)
        goal_state = model.goal_y * model.width + model.goal_x

        # Create zoned layout
        zoned_layout = get_jax_zoned_layout(
            layout_name,
            goal_pos=goal_pos,
            width=model.width,
            height=model.height,
        )

        return cls(
            zoned_layout=zoned_layout,
            goal_state=goal_state,
            alpha=alpha,
            alpha_other=alpha_other,
            gamma=gamma,
            B=jnp.array(model.B["location_state"]),
            A_loc=jnp.array(model.A["location_obs"]),
            C_loc=jnp.array(model.C["location_obs"]),
            A_cell_collision=jnp.array(model.A["cell_collision_obs"]),
            C_cell_collision=jnp.array(model.C["cell_collision_obs"]),
            A_edge=jnp.array(model.A["edge_obs"]),
            C_edge=jnp.array(model.C["edge_obs"]),
            A_edge_collision=jnp.array(model.A["edge_collision_obs"]),
            C_edge_collision=jnp.array(model.C["edge_collision_obs"]),
        )

    def plan(
        self,
        qs_self: jnp.ndarray,
        qs_other: jnp.ndarray,
        other_goal_state: int,
    ) -> int:
        """
        Plan next action using hierarchical approach.

        Parameters
        ----------
        qs_self : jnp.ndarray
            Belief over own state
        qs_other : jnp.ndarray
            Belief over other's state
        other_goal_state : int
            Other agent's goal state

        Returns
        -------
        action : int
            Primitive action (0-4)
        """
        result = self.plan_with_debug(qs_self, qs_other, other_goal_state)
        return int(result["action"])

    def plan_with_debug(
        self,
        qs_self: jnp.ndarray,
        qs_other: jnp.ndarray,
        other_goal_state: int,
    ) -> Dict:
        """
        Plan with debug information.

        Returns dict with action and intermediate values.
        """
        layout = self.zoned_layout

        # 1. Determine current zones from beliefs
        my_zone = get_zone_from_belief(qs_self, layout.cell_to_zone)
        other_zone = get_zone_from_belief(qs_other, layout.cell_to_zone)
        my_goal_zone = layout.goal_zone
        other_goal_zone = int(layout.cell_to_zone[other_goal_state])

        # 2. High-level zone planning
        zone_action, G_zone, q_zone = high_level_plan_jax(
            my_zone, other_zone, my_goal_zone, other_goal_zone,
            self.alpha,
            layout.zone_adjacency,
            layout.zone_is_bottleneck,
            self.gamma,
        )

        # 3. Get subgoal from zone action
        subgoal_state = get_subgoal_state_jax(
            my_zone, zone_action, my_goal_zone, self.goal_state,
            layout.exit_points, layout.zone_adjacency, layout.cell_to_zone,
        )

        # 4. Low-level planning toward subgoal
        action, G_low, q_low = low_level_plan_jax(
            qs_self, qs_other, subgoal_state,
            self.B, self.A_loc, self.C_loc,
            self.A_cell_collision, self.C_cell_collision,
            self.A_edge, self.C_edge,
            self.A_edge_collision, self.C_edge_collision,
            self.alpha, layout.width, self.gamma,
        )

        return {
            "action": int(action),
            "my_zone": int(my_zone),
            "other_zone": int(other_zone),
            "my_goal_zone": int(my_goal_zone),
            "other_goal_zone": int(other_goal_zone),
            "zone_action": int(zone_action),
            "subgoal_state": int(subgoal_state),
            "G_zone": np.array(G_zone),
            "q_zone": np.array(q_zone),
            "G_low": np.array(G_low),
            "q_low": np.array(q_low),
        }


# =============================================================================
# Planner Interface (Compatible with EmpathicLavaPlanner)
# =============================================================================

class HierarchicalEmpathicPlannerJax:
    """
    JAX hierarchical planner with interface compatible with EmpathicLavaPlanner.

    This class wraps JaxHierarchicalPlanner to provide a similar interface
    to the existing EmpathicLavaPlanner for easy integration.

    Parameters
    ----------
    agent_i : LavaAgent
        Agent i (self)
    agent_j : LavaAgent
        Agent j (other)
    layout_name : str
        Name of layout for zone decomposition
    alpha : float
        Empathy weight
    alpha_other : float
        Believed empathy of other agent
    gamma : float
        Inverse temperature
    """

    def __init__(
        self,
        agent_i,
        agent_j,
        layout_name: str,
        alpha: float = 0.5,
        alpha_other: float = 0.0,
        gamma: float = 8.0,
    ):
        self.agent_i = agent_i
        self.agent_j = agent_j
        self.layout_name = layout_name
        self.alpha = alpha
        self.alpha_other = alpha_other
        self.gamma = gamma

        # Create JAX planners for both agents
        self.planner_i = JaxHierarchicalPlanner.from_model(
            agent_i.model,
            layout_name,
            alpha=alpha,
            alpha_other=alpha_other,
            gamma=gamma,
        )

        # For j, we need their goal
        self.planner_j = JaxHierarchicalPlanner.from_model(
            agent_j.model,
            layout_name,
            alpha=alpha_other,  # j uses their own empathy
            alpha_other=alpha,  # j's belief about i's empathy
            gamma=gamma,
        )

        # Store goal states for ToM
        self.goal_state_i = agent_i.model.goal_y * agent_i.model.width + agent_i.model.goal_x
        self.goal_state_j = agent_j.model.goal_y * agent_j.model.width + agent_j.model.goal_x

    def plan(
        self,
        qs_i: np.ndarray,
        qs_j: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Plan using hierarchical approach.

        Parameters
        ----------
        qs_i : np.ndarray
            Agent i's belief over own state
        qs_j : np.ndarray
            Agent i's belief over j's state

        Returns
        -------
        G_i : np.ndarray
            Placeholder G values (for compatibility)
        G_j : np.ndarray
            Placeholder G values (for compatibility)
        G_social : np.ndarray
            Placeholder G values (for compatibility)
        q_pi : np.ndarray
            Action probabilities
        action : int
            Selected action
        """
        # Convert to JAX arrays
        qs_i_jax = jnp.array(qs_i)
        qs_j_jax = jnp.array(qs_j)

        # Plan for agent i
        result = self.planner_i.plan_with_debug(
            qs_i_jax, qs_j_jax, self.goal_state_j
        )

        action = result["action"]
        q_low = result["q_low"]

        # Return compatibility format
        # G values are just for compatibility - return low-level G
        G_i = result["G_low"]
        G_j = np.zeros_like(G_i)  # Placeholder
        G_social = G_i  # Placeholder

        return G_i, G_j, G_social, q_low, action

    def plan_with_debug(
        self,
        qs_i: np.ndarray,
        qs_j: np.ndarray,
    ) -> Dict:
        """
        Plan with full debug information.
        """
        qs_i_jax = jnp.array(qs_i)
        qs_j_jax = jnp.array(qs_j)

        return self.planner_i.plan_with_debug(
            qs_i_jax, qs_j_jax, self.goal_state_j
        )
