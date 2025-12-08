"""
LavaCorridor environment variants for testing multi-agent coordination.

This module provides different corridor layouts to test various coordination scenarios:
- Variant 1 (Narrow): Single-file corridor - collision unavoidable
- Variant 2 (Wide): Multi-row safe corridor - agents can pass each other
- Variant 3 (Bottleneck): Wide areas with narrow bottleneck - coordination test
- Variant 4 (Risk-Reward): Fast risky path vs slow safe detour
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class LavaLayout:
    """
    Defines a lava corridor layout.

    Attributes
    ----------
    width : int
        Grid width
    height : int
        Grid height
    safe_cells : List[Tuple[int, int]]
        List of (x, y) coordinates that are safe (not lava)
    goal_positions : List[Tuple[int, int]]
        Goal positions for each agent [(goal_0), (goal_1), ...]
    start_positions : List[Tuple[int, int]]
        Starting positions for agents
    name : str
        Layout name for identification
    """
    width: int
    height: int
    safe_cells: List[Tuple[int, int]]
    goal_positions: List[Tuple[int, int]]
    start_positions: List[Tuple[int, int]]
    name: str


def create_narrow_corridor(width: int = 6) -> LavaLayout:
    """
    Variant 1: Narrow single-file corridor.

    Layout:
    ~ ~ ~ ~ ~ G
    . . . . . .  (1 cell wide - collision unavoidable)
    ~ ~ ~ ~ ~ ~

    Both agents start at opposite ends of the corridor.
    """
    height = 3
    safe_y = 1

    # All cells in middle row are safe
    safe_cells = [(x, safe_y) for x in range(width)]

    # Each agent has goal at opposite end from their start
    # Agent 0: starts left (0, 1), goal right (5, 1)
    # Agent 1: starts right (5, 1), goal left (0, 1)
    goal_positions = [(width - 1, safe_y), (0, safe_y)]

    # Agents start at opposite ends
    start_positions = [(0, safe_y), (width - 1, safe_y)]

    return LavaLayout(
        width=width,
        height=height,
        safe_cells=safe_cells,
        goal_positions=goal_positions,
        start_positions=start_positions,
        name="narrow_corridor"
    )


def create_wide_corridor(width: int = 6) -> LavaLayout:
    """
    Variant 2: Wide corridor with space to pass.

    Layout:
    ~ ~ ~ ~ ~ G
    . . . . . .  (2 cells wide - can coordinate)
    . . . . . .
    ~ ~ ~ ~ ~ ~

    Agents can move to different rows to avoid collision.
    """
    height = 4
    safe_rows = [1, 2]

    # All cells in safe rows
    safe_cells = [(x, y) for x in range(width) for y in safe_rows]

    # Each agent has goal in their own row at rightmost column
    # Agent 0: starts (0, 1), goal (5, 1)
    # Agent 1: starts (0, 2), goal (5, 2)
    goal_positions = [(width - 1, 1), (width - 1, 2)]

    # Agents start at same x but different rows
    start_positions = [(0, 1), (0, 2)]

    return LavaLayout(
        width=width,
        height=height,
        safe_cells=safe_cells,
        goal_positions=goal_positions,
        start_positions=start_positions,
        name="wide_corridor"
    )


def create_bottleneck(width: int = 8) -> LavaLayout:
    """
    Variant 3: Wide areas connected by narrow bottleneck.

    Layout:
    ~ ~ ~ B B ~ ~ G
    . . . B B . . .  (bottleneck in middle)
    . . . . . . . .
    ~ ~ ~ ~ ~ ~ ~ ~

    Tests coordination: who goes through bottleneck first?
    """
    height = 4

    # Define bottleneck positions (middle third of corridor)
    bottleneck_start = width // 3
    bottleneck_end = 2 * width // 3

    safe_cells = []
    for x in range(width):
        if bottleneck_start <= x < bottleneck_end:
            # Bottleneck: only row 1 is safe
            safe_cells.append((x, 1))
        else:
            # Wide areas: rows 1 and 2 are safe
            safe_cells.append((x, 1))
            safe_cells.append((x, 2))

    # Each agent has goal in their own row at rightmost column
    # Agent 0: starts (0, 1), goal (7, 1)
    # Agent 1: starts (0, 2), goal (7, 2)
    goal_positions = [(width - 1, 1), (width - 1, 2)]

    # Agents start at left, different rows
    start_positions = [(0, 1), (0, 2)]

    return LavaLayout(
        width=width,
        height=height,
        safe_cells=safe_cells,
        goal_positions=goal_positions,
        start_positions=start_positions,
        name="bottleneck"
    )


def create_crossed_goals(width: int = 6) -> LavaLayout:
    """
    Variant 4: Crossed goals - agents must swap lanes.

    Layout:
    ~ ~ ~ ~ ~ ~
    0 . . . . G1  (agent 0 starts row 1, needs goal at row 2)
    1 . . . . G0  (agent 1 starts row 2, needs goal at row 1)
    ~ ~ ~ ~ ~ ~

    Forces true coordination: agents must cross paths or take turns.
    This makes empathy critical - selfish agents will collide.
    """
    height = 4
    safe_rows = [1, 2]

    # All cells in safe rows
    safe_cells = [(x, y) for x in range(width) for y in safe_rows]

    # SWAPPED goals: agent 0 needs to reach opposite lane
    # Agent 0: starts (0, 1), goal (5, 2) - must cross UP
    # Agent 1: starts (0, 2), goal (5, 1) - must cross DOWN
    goal_positions = [(width - 1, 2), (width - 1, 1)]  # Swapped!

    # Agents start at same x but different rows
    start_positions = [(0, 1), (0, 2)]

    return LavaLayout(
        width=width,
        height=height,
        safe_cells=safe_cells,
        goal_positions=goal_positions,
        start_positions=start_positions,
        name="crossed_goals"
    )


def create_risk_reward(width: int = 8) -> LavaLayout:
    """
    Variant 5: Risky fast path vs safe slow detour.

    Layout:
    G . . . . . . .  (risky path - narrow, goal nearby)
    ~ ~ ~ . . . . .
    ~ ~ ~ . . . . .  (safe detour - wide, goal far)
    ~ ~ ~ ~ ~ ~ ~ ~

    Tests risk preferences and coordination on different paths.
    """
    height = 4

    # Risky path: row 0, shorter to goal
    # Safe detour: rows 1-2, longer to goal
    safe_cells = []

    # Risky path: all of row 0
    for x in range(width):
        safe_cells.append((x, 0))

    # Safe detour: rows 1-2, but blocked at start
    for x in range(3, width):
        safe_cells.append((x, 1))
        safe_cells.append((x, 2))

    # Each agent has goal in their own row at leftmost column
    # Agent 0: starts (7, 1), goal (0, 1) - can take risky row 0 or safe detour
    # Agent 1: starts (7, 2), goal (0, 2) - same trade-off
    goal_positions = [(0, 1), (0, 2)]

    # Agents start at right side
    start_positions = [(width - 1, 1), (width - 1, 2)]

    return LavaLayout(
        width=width,
        height=height,
        safe_cells=safe_cells,
        goal_positions=goal_positions,
        start_positions=start_positions,
        name="risk_reward"
    )


# Layout registry
LAYOUTS = {
    "narrow": create_narrow_corridor,
    "wide": create_wide_corridor,
    "bottleneck": create_bottleneck,
    "crossed_goals": create_crossed_goals,
    "risk_reward": create_risk_reward,
}


def get_layout(layout_name: str, **kwargs) -> LavaLayout:
    """
    Get a predefined layout by name.

    Parameters
    ----------
    layout_name : str
        One of: "narrow", "wide", "bottleneck", "risk_reward"
    **kwargs : dict
        Additional arguments for layout creation (e.g., width=8)

    Returns
    -------
    layout : LavaLayout
        The requested layout configuration
    """
    if layout_name not in LAYOUTS:
        raise ValueError(f"Unknown layout: {layout_name}. Choose from {list(LAYOUTS.keys())}")

    return LAYOUTS[layout_name](**kwargs)
