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
    start_config : str
        Configuration variant ("A" or "B") for role asymmetry testing
    """
    width: int
    height: int
    safe_cells: List[Tuple[int, int]]
    goal_positions: List[Tuple[int, int]]
    start_positions: List[Tuple[int, int]]
    name: str
    start_config: str = "A"

    def swap_agents(self) -> "LavaLayout":
        """Return a new layout with agents swapped (config B)."""
        return LavaLayout(
            width=self.width,
            height=self.height,
            safe_cells=self.safe_cells.copy(),
            goal_positions=[self.goal_positions[1], self.goal_positions[0]],
            start_positions=[self.start_positions[1], self.start_positions[0]],
            name=self.name,
            start_config="B"
        )


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

    # Goal cells - must be safe (not lava)
    safe_cells.append((0, 1))
    safe_cells.append((0, 2))

    # Each agent has goal in their own row at leftmost column
    # Agent 0: starts (7, 1), goal (0, 1) - must go via risky row 0
    # Agent 1: starts (7, 2), goal (0, 2) - must go via risky row 0
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


def create_double_bottleneck(width: int = 10) -> LavaLayout:
    """
    Variant 6: Two bottlenecks in series with a passing bay between.

    Layout (width=10):
    ~ ~ B ~ ~ ~ ~ B ~ G
    . . B . . . . B . .  (two bottlenecks at cols 2 and 7)
    . . . . . . . . . .
    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    The middle section (cols 3-6) is wide, allowing agents to pass.
    Tests multi-stage coordination: must coordinate through two choke points.
    """
    height = 4

    # Bottleneck positions
    bottleneck1 = 2
    bottleneck2 = width - 3  # 7 for width=10

    safe_cells = []
    for x in range(width):
        if x == bottleneck1 or x == bottleneck2:
            # Bottleneck: only row 1 is safe
            safe_cells.append((x, 1))
        else:
            # Wide areas: rows 1 and 2 are safe
            safe_cells.append((x, 1))
            safe_cells.append((x, 2))

    # Agent 0: starts (0, 1), goal (width-1, 1)
    # Agent 1: starts (0, 2), goal (width-1, 2)
    goal_positions = [(width - 1, 1), (width - 1, 2)]
    start_positions = [(0, 1), (0, 2)]

    return LavaLayout(
        width=width,
        height=height,
        safe_cells=safe_cells,
        goal_positions=goal_positions,
        start_positions=start_positions,
        name="double_bottleneck"
    )


def create_passing_bay(width: int = 8) -> LavaLayout:
    """
    Variant 7: Mostly single-file corridor with one 2x2 passing bay.

    Layout (width=8):
    ~ ~ ~ ~ ~ ~ ~ G
    . . . X X . . .  (X marks the 2x2 passing bay at cols 3-4)
    ~ ~ ~ X X ~ ~ ~
    ~ ~ ~ ~ ~ ~ ~ ~

    To coordinate, one agent must pull into the bay to let the other pass.
    Critical test for altruistic yielding behavior.
    """
    height = 4

    # Bay position (middle of corridor)
    bay_start = width // 2 - 1  # 3 for width=8
    bay_end = width // 2 + 1    # 5 for width=8

    safe_cells = []
    # Main corridor: row 1, all columns
    for x in range(width):
        safe_cells.append((x, 1))

    # Passing bay: row 2, only at bay columns
    for x in range(bay_start, bay_end):
        safe_cells.append((x, 2))

    # Agents must cross - one starts left, one starts right
    # Agent 0: starts (0, 1), goal (width-1, 1)
    # Agent 1: starts (width-1, 1), goal (0, 1)
    goal_positions = [(width - 1, 1), (0, 1)]
    start_positions = [(0, 1), (width - 1, 1)]

    return LavaLayout(
        width=width,
        height=height,
        safe_cells=safe_cells,
        goal_positions=goal_positions,
        start_positions=start_positions,
        name="passing_bay"
    )


def create_asymmetric_detour(width: int = 8) -> LavaLayout:
    """
    Variant 8: One agent has shorter direct path, other must detour.

    Layout (width=8):
    ~ ~ ~ ~ ~ ~ ~ G0
    0 . . . B . . .   (agent 0 has direct path, blocked at col 4)
    1 . . . . . . G1  (agent 1 must go around)
    ~ ~ ~ ~ ~ ~ ~ ~

    Agent 0 can reach goal directly through bottleneck.
    Agent 1 must detour but has no bottleneck.
    Tests fairness: will agent 0 wait at bottleneck for agent 1?
    """
    height = 4

    bottleneck_col = width // 2  # 4 for width=8

    safe_cells = []

    # Row 1: full corridor but bottleneck blocks row 2 access at that column
    for x in range(width):
        safe_cells.append((x, 1))

    # Row 2: full corridor (agent 1's path)
    for x in range(width):
        safe_cells.append((x, 2))

    # Agent 0: starts (0, 1), goal at far end row 0 area
    # Actually, let's make it simpler: both can reach goals but one path is shorter
    # Agent 0: starts (0, 1), goal (width-1, 1) - direct 7 steps
    # Agent 1: starts (0, 2), goal (width-1, 1) - must cross to row 1, potential conflict

    # Better design: Agent 0 has short path, Agent 1 needs to cross
    goal_positions = [(width - 1, 1), (width - 1, 1)]  # Same goal cell!
    start_positions = [(0, 1), (0, 2)]

    return LavaLayout(
        width=width,
        height=height,
        safe_cells=safe_cells,
        goal_positions=goal_positions,
        start_positions=start_positions,
        name="asymmetric_detour"
    )


def create_t_junction(width: int = 7) -> LavaLayout:
    """
    Variant 9: T-junction where agents approach from different directions.

    Layout (width=7, height=5):
    ~ ~ ~ 0 ~ ~ ~   (agent 0 comes from top)
    ~ ~ ~ . ~ ~ ~
    G1. . . . . G0  (horizontal corridor with goals at ends)
    ~ ~ ~ . ~ ~ ~
    ~ ~ ~ 1 ~ ~ ~   (agent 1 comes from bottom)

    Agents must coordinate at the junction cell (3, 2).
    """
    height = 5
    mid_x = width // 2  # 3
    mid_y = height // 2  # 2

    safe_cells = []

    # Horizontal corridor (row mid_y)
    for x in range(width):
        safe_cells.append((x, mid_y))

    # Vertical corridor (column mid_x)
    for y in range(height):
        if (mid_x, y) not in safe_cells:
            safe_cells.append((mid_x, y))

    # Agent 0: starts top, goal right
    # Agent 1: starts bottom, goal left
    goal_positions = [(width - 1, mid_y), (0, mid_y)]
    start_positions = [(mid_x, 0), (mid_x, height - 1)]

    return LavaLayout(
        width=width,
        height=height,
        safe_cells=safe_cells,
        goal_positions=goal_positions,
        start_positions=start_positions,
        name="t_junction"
    )


def create_symmetric_bottleneck(width: int = 10) -> LavaLayout:
    """
    Variant 10: Symmetric bottleneck - agents start on opposite sides.

    Layout (width=10):
    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    0 . . . B B . . . 1   (agents start on opposite ends, row 1)
    . . . . B B . . . .   (wide on both sides, bottleneck in middle)
    ~ ~ ~ ~ ~ ~ ~ ~ ~ ~

    Both agents face identical constraints - must pass through same bottleneck.
    Agent 0: starts left (0,1), goal right (9,1)
    Agent 1: starts right (9,1), goal left (0,1)

    Tests pure coordination without asymmetric advantage.
    """
    height = 4

    # Bottleneck in exact middle
    bottleneck_start = width // 2 - 1  # 4 for width=10
    bottleneck_end = width // 2 + 1    # 6 for width=10

    safe_cells = []
    for x in range(width):
        if bottleneck_start <= x < bottleneck_end:
            # Bottleneck: only row 1 is safe (single file)
            safe_cells.append((x, 1))
        else:
            # Wide areas: rows 1 and 2 are safe
            safe_cells.append((x, 1))
            safe_cells.append((x, 2))

    # Agents start on opposite ends, must swap positions
    # Agent 0: starts (0, 1), goal (width-1, 1)
    # Agent 1: starts (width-1, 1), goal (0, 1)
    goal_positions = [(width - 1, 1), (0, 1)]
    start_positions = [(0, 1), (width - 1, 1)]

    return LavaLayout(
        width=width,
        height=height,
        safe_cells=safe_cells,
        goal_positions=goal_positions,
        start_positions=start_positions,
        name="symmetric_bottleneck"
    )


def create_vertical_bottleneck(width: int = 6, height: int = 8) -> LavaLayout:
    """
    Variant 11: Vertical bottleneck - agents start in opposite wide areas.

    Layout (width=6, height=8):
    ~ ~ ~ ~ ~ ~
    ~ 0 . . G1~   (agent 0 starts top-left, agent 1's goal top-right)
    . . . . . .   (wide area top - row 2)
    ~ ~ ~ . ~ ~   (bottleneck - only column 3 safe)
    ~ ~ ~ . ~ ~   (bottleneck continues)
    . . . . . .   (wide area bottom - row 5)
    ~ G0. . 1 ~   (agent 0's goal bottom-left, agent 1 starts bottom-right)
    ~ ~ ~ ~ ~ ~

    Agents must pass through bottleneck to reach goals on opposite side.
    Tests coordination with passing opportunities in wide areas.
    """
    mid_x = width // 2  # 3 for width=6

    # Wide rows at top and bottom
    wide_top = 2
    wide_bottom = height - 3  # 5 for height=8

    safe_cells = []

    # Vertical corridor through middle (bottleneck)
    for y in range(height):
        safe_cells.append((mid_x, y))

    # Wide area at top (full row)
    for x in range(width):
        if (x, wide_top) not in safe_cells:
            safe_cells.append((x, wide_top))

    # Wide area at bottom (full row)
    for x in range(width):
        if (x, wide_bottom) not in safe_cells:
            safe_cells.append((x, wide_bottom))

    # Start/goal positions in the wide areas
    # Agent 0: starts top-left wide area, goal bottom-left wide area
    # Agent 1: starts bottom-right wide area, goal top-right wide area
    start_positions = [(1, wide_top), (width - 2, wide_bottom)]
    goal_positions = [(1, wide_bottom), (width - 2, wide_top)]

    return LavaLayout(
        width=width,
        height=height,
        safe_cells=safe_cells,
        goal_positions=goal_positions,
        start_positions=start_positions,
        name="vertical_bottleneck"
    )


# Layout registry
LAYOUTS = {
    "narrow": create_narrow_corridor,
    "wide": create_wide_corridor,
    "bottleneck": create_bottleneck,
    "crossed_goals": create_crossed_goals,
    "risk_reward": create_risk_reward,
    "double_bottleneck": create_double_bottleneck,
    "passing_bay": create_passing_bay,
    "asymmetric_detour": create_asymmetric_detour,
    "t_junction": create_t_junction,
    "symmetric_bottleneck": create_symmetric_bottleneck,
    "vertical_bottleneck": create_vertical_bottleneck,
}

# Layout complexity index (for analysis)
LAYOUT_COMPLEXITY = {
    "wide": 1,              # Easy - can always pass
    "narrow": 5,            # Impossible - forced collision, no passing space
    "bottleneck": 3,        # Medium - one choke point
    "crossed_goals": 3,     # Medium - must cross paths
    "risk_reward": 3,       # Medium - path choice
    "double_bottleneck": 4, # Hard - two choke points
    "passing_bay": 4,       # Hard - must use bay
    "asymmetric_detour": 3, # Medium - asymmetric paths
    "t_junction": 4,        # Hard - single junction cell
    "symmetric_bottleneck": 4,  # Hard - opposite sides through bottleneck
    "vertical_bottleneck": 4,   # Hard - vertical with wide passing areas
}


def get_layout(layout_name: str, start_config: str = "A", **kwargs) -> LavaLayout:
    """
    Get a predefined layout by name.

    Parameters
    ----------
    layout_name : str
        One of: "narrow", "wide", "bottleneck", "crossed_goals", "risk_reward",
        "double_bottleneck", "passing_bay", "asymmetric_detour", "t_junction"
    start_config : str
        "A" for default configuration, "B" for swapped agent positions
    **kwargs : dict
        Additional arguments for layout creation (e.g., width=8)

    Returns
    -------
    layout : LavaLayout
        The requested layout configuration
    """
    if layout_name not in LAYOUTS:
        raise ValueError(f"Unknown layout: {layout_name}. Choose from {list(LAYOUTS.keys())}")

    layout = LAYOUTS[layout_name](**kwargs)

    if start_config == "B":
        layout = layout.swap_agents()

    return layout


def get_all_layout_names() -> List[str]:
    """Return list of all available layout names."""
    return list(LAYOUTS.keys())
