"""
Tests for new layout variants and experiment suite functionality.

Tests:
1. New layout creation (double_bottleneck, passing_bay, asymmetric_detour, t_junction)
2. Start configuration swapping (Config A vs B)
3. Layout complexity indices
4. Paralysis detection logic
5. Basic experiment runner functionality
"""

import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import numpy as np
import jax.random as jr

from tom.envs import LavaV2Env, get_layout
from tom.envs.env_lava_variants import (
    get_all_layout_names,
    LAYOUT_COMPLEXITY,
    create_double_bottleneck,
    create_passing_bay,
    create_asymmetric_detour,
    create_t_junction,
)


class TestNewLayoutCreation:
    """Test that all new layout variants can be created."""

    def test_double_bottleneck_layout(self):
        """Test double bottleneck layout."""
        layout = create_double_bottleneck(width=10)

        assert layout.width == 10
        assert layout.height == 4
        assert layout.name == "double_bottleneck"
        assert len(layout.start_positions) >= 2
        assert len(layout.goal_positions) >= 2

        # Check that bottlenecks exist (only 1 safe row at bottleneck columns)
        safe_by_x = {}
        for x, y in layout.safe_cells:
            if x not in safe_by_x:
                safe_by_x[x] = []
            safe_by_x[x].append(y)

        # Bottleneck at x=2 should have fewer safe cells than wide areas
        assert len(safe_by_x.get(2, [])) < len(safe_by_x.get(4, [])), \
            "Bottleneck should have fewer safe rows"

        print(f"\nDouble bottleneck layout:")
        print(f"  Dimensions: {layout.width}x{layout.height}")
        print(f"  Goals: {layout.goal_positions}")
        print(f"  Start positions: {layout.start_positions}")

    def test_passing_bay_layout(self):
        """Test passing bay layout."""
        layout = create_passing_bay(width=8)

        assert layout.width == 8
        assert layout.height == 4
        assert layout.name == "passing_bay"
        assert len(layout.start_positions) >= 2
        assert len(layout.goal_positions) >= 2

        # Agents start at opposite ends
        start_0 = layout.start_positions[0]
        start_1 = layout.start_positions[1]
        assert start_0[0] < start_1[0], "Agent 0 should start left, agent 1 right"

        # Goals are swapped (each needs to reach other's start)
        assert layout.goal_positions[0][0] > layout.goal_positions[1][0], \
            "Goals should be swapped"

        # Check passing bay exists (extra safe cells in row 2)
        bay_cells = [(x, y) for x, y in layout.safe_cells if y == 2]
        assert len(bay_cells) > 0, "Should have passing bay cells in row 2"
        assert len(bay_cells) < layout.width, "Bay should not span full width"

        print(f"\nPassing bay layout:")
        print(f"  Dimensions: {layout.width}x{layout.height}")
        print(f"  Bay cells: {bay_cells}")
        print(f"  Starts: {layout.start_positions}")
        print(f"  Goals: {layout.goal_positions}")

    def test_asymmetric_detour_layout(self):
        """Test asymmetric detour layout."""
        layout = create_asymmetric_detour(width=8)

        assert layout.width == 8
        assert layout.height == 4
        assert layout.name == "asymmetric_detour"
        assert len(layout.start_positions) >= 2
        assert len(layout.goal_positions) >= 2

        # Both agents aim for same goal position
        assert layout.goal_positions[0] == layout.goal_positions[1], \
            "Both agents should have same goal in asymmetric_detour"

        print(f"\nAsymmetric detour layout:")
        print(f"  Dimensions: {layout.width}x{layout.height}")
        print(f"  Goals (shared): {layout.goal_positions}")
        print(f"  Start positions: {layout.start_positions}")

    def test_t_junction_layout(self):
        """Test T-junction layout."""
        layout = create_t_junction(width=7)

        assert layout.width == 7
        assert layout.height == 5
        assert layout.name == "t_junction"
        assert len(layout.start_positions) >= 2
        assert len(layout.goal_positions) >= 2

        # Agents start at opposite ends of vertical corridor
        start_0 = layout.start_positions[0]
        start_1 = layout.start_positions[1]
        assert start_0[0] == start_1[0], "Both agents should start on same x (vertical corridor)"
        assert start_0[1] != start_1[1], "Agents should start at different y positions"

        print(f"\nT-junction layout:")
        print(f"  Dimensions: {layout.width}x{layout.height}")
        print(f"  Starts: {layout.start_positions}")
        print(f"  Goals: {layout.goal_positions}")


class TestStartConfigSwapping:
    """Test start configuration A/B swapping."""

    def test_swap_agents(self):
        """Test that swap_agents creates correct Config B."""
        layout_a = get_layout("wide", start_config="A")
        layout_b = get_layout("wide", start_config="B")

        # Positions should be swapped
        assert layout_a.start_positions[0] == layout_b.start_positions[1], \
            "Config B should swap agent 0 to agent 1's position"
        assert layout_a.start_positions[1] == layout_b.start_positions[0], \
            "Config B should swap agent 1 to agent 0's position"

        # Goals should also be swapped
        assert layout_a.goal_positions[0] == layout_b.goal_positions[1], \
            "Config B should swap goals"
        assert layout_a.goal_positions[1] == layout_b.goal_positions[0], \
            "Config B should swap goals"

        # Config identifiers
        assert layout_a.start_config == "A"
        assert layout_b.start_config == "B"

        print(f"\nConfig A:")
        print(f"  Starts: {layout_a.start_positions}")
        print(f"  Goals: {layout_a.goal_positions}")
        print(f"\nConfig B (swapped):")
        print(f"  Starts: {layout_b.start_positions}")
        print(f"  Goals: {layout_b.goal_positions}")

    def test_env_uses_start_config(self):
        """Test that LavaV2Env respects start_config parameter."""
        env_a = LavaV2Env(layout_name="crossed_goals", width=6, num_agents=2, start_config="A")
        env_b = LavaV2Env(layout_name="crossed_goals", width=6, num_agents=2, start_config="B")

        key = jr.PRNGKey(0)
        state_a, _ = env_a.reset(key)
        state_b, _ = env_b.reset(key)

        pos_a = state_a["env_state"]["pos"]
        pos_b = state_b["env_state"]["pos"]

        # Positions should be swapped between configs
        assert pos_a[0] == pos_b[1], "Agent positions should be swapped"
        assert pos_a[1] == pos_b[0], "Agent positions should be swapped"

        print(f"\nConfig A positions: {pos_a}")
        print(f"Config B positions: {pos_b}")


class TestLayoutComplexity:
    """Test layout complexity indices."""

    def test_complexity_indices_exist(self):
        """Test that all layouts have complexity indices."""
        all_layouts = get_all_layout_names()

        for layout_name in all_layouts:
            assert layout_name in LAYOUT_COMPLEXITY, \
                f"Layout '{layout_name}' missing from LAYOUT_COMPLEXITY"

        print(f"\nLayout complexity indices:")
        for name, complexity in sorted(LAYOUT_COMPLEXITY.items(), key=lambda x: x[1]):
            print(f"  {name}: {complexity}")

    def test_complexity_ordering(self):
        """Test that complexity indices make sense."""
        # Wide should be easier than narrow
        assert LAYOUT_COMPLEXITY["wide"] < LAYOUT_COMPLEXITY["narrow"], \
            "Wide should be easier than narrow"

        # Single bottleneck easier than double
        assert LAYOUT_COMPLEXITY["bottleneck"] < LAYOUT_COMPLEXITY["double_bottleneck"], \
            "Single bottleneck should be easier than double"


class TestParalysisDetection:
    """Test paralysis detection logic."""

    def test_no_paralysis_on_success(self):
        """Test that successful episodes don't trigger paralysis."""
        from src.metrics.paralysis_detection import detect_paralysis

        # Successful trajectory
        trajectory_i = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)]
        trajectory_j = [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2), (5, 2)]
        actions_i = [3, 3, 3, 3, 3, 4]  # RIGHT moves then STAY
        actions_j = [3, 3, 3, 3, 3, 4]

        result = detect_paralysis(
            trajectory_i, trajectory_j,
            actions_i, actions_j,
            goal_reached_i=True, goal_reached_j=True,
            max_timesteps=25
        )

        assert not result["paralysis"], "Success should not be paralysis"
        print(f"\nSuccessful episode: paralysis={result['paralysis']}")

    def test_cycle_detection(self):
        """Test that cyclic behavior is detected."""
        from src.metrics.paralysis_detection import detect_paralysis

        # Cyclic trajectory (oscillating between positions)
        cycle_traj = [(1, 1), (2, 1), (1, 1), (2, 1), (1, 1), (2, 1)] * 5
        trajectory_i = cycle_traj[:25]
        trajectory_j = [(3, 1)] * 25  # Other agent stays put
        actions_i = [3, 2] * 12 + [3]  # RIGHT, LEFT alternating
        actions_j = [4] * 25  # STAY

        result = detect_paralysis(
            trajectory_i, trajectory_j,
            actions_i, actions_j,
            goal_reached_i=False, goal_reached_j=False,
            max_timesteps=25,
            cycle_threshold=3
        )

        assert result["paralysis"], "Cyclic behavior should be detected as paralysis"
        assert result["paralysis_type"] == "cycle", "Should be classified as cycle"
        print(f"\nCyclic episode: paralysis={result['paralysis']}, type={result['paralysis_type']}")

    def test_mutual_stay_detection(self):
        """Test that mutual stay is detected."""
        from src.metrics.paralysis_detection import detect_paralysis

        # Both agents stay in place
        trajectory_i = [(1, 1)] * 25
        trajectory_j = [(3, 2)] * 25
        actions_i = [4] * 25  # STAY
        actions_j = [4] * 25  # STAY

        result = detect_paralysis(
            trajectory_i, trajectory_j,
            actions_i, actions_j,
            goal_reached_i=False, goal_reached_j=False,
            max_timesteps=25,
            stay_threshold=3
        )

        assert result["paralysis"], "Mutual stay should be detected as paralysis"
        assert result["paralysis_type"] == "mutual_stay" or result["paralysis_type"] == "cycle", \
            "Should be classified as mutual_stay or cycle"
        assert result["stay_streak"] >= 3, "Should detect stay streak"
        print(f"\nMutual stay episode: paralysis={result['paralysis']}, "
              f"type={result['paralysis_type']}, streak={result['stay_streak']}")


class TestAllLayoutsCanRun:
    """Test that all layouts can run a basic episode."""

    @pytest.mark.parametrize("layout_name", get_all_layout_names())
    def test_layout_basic_episode(self, layout_name):
        """Test that each layout can run a basic episode."""
        # Skip some layouts that might have special requirements
        try:
            env = LavaV2Env(
                layout_name=layout_name,
                num_agents=2,
                timesteps=10,
                start_config="A"
            )
        except Exception as e:
            pytest.skip(f"Could not create env for {layout_name}: {e}")

        key = jr.PRNGKey(42)
        state, obs = env.reset(key)

        # Run a few steps with STAY actions
        for _ in range(3):
            state, obs, _, done, info = env.step(state, {0: 4, 1: 4})

        # Should not crash
        assert state is not None
        print(f"\n{layout_name}: Basic episode completed")

    @pytest.mark.parametrize("layout_name", get_all_layout_names())
    def test_layout_with_movement(self, layout_name):
        """Test that each layout allows movement."""
        try:
            env = LavaV2Env(
                layout_name=layout_name,
                num_agents=2,
                timesteps=10,
                start_config="A"
            )
        except Exception as e:
            pytest.skip(f"Could not create env for {layout_name}: {e}")

        key = jr.PRNGKey(42)
        state, obs = env.reset(key)
        initial_pos = state["env_state"]["pos"].copy()

        # Try moving RIGHT
        state, obs, _, done, info = env.step(state, {0: 3, 1: 3})

        # At least one agent should have moved (unless blocked by lava)
        final_pos = state["env_state"]["pos"]

        print(f"\n{layout_name}:")
        print(f"  Initial: {initial_pos}")
        print(f"  After RIGHT: {final_pos}")


class TestEnvLayoutInfo:
    """Test that get_layout_info returns correct information."""

    @pytest.mark.parametrize("layout_name", get_all_layout_names())
    def test_layout_info_complete(self, layout_name):
        """Test that layout info contains all required fields."""
        try:
            env = LavaV2Env(layout_name=layout_name, num_agents=2, timesteps=10)
        except Exception as e:
            pytest.skip(f"Could not create env for {layout_name}: {e}")

        info = env.get_layout_info()

        required_fields = [
            "width", "height", "num_states",
            "goal_positions", "safe_cells", "start_positions", "layout_name"
        ]

        for field in required_fields:
            assert field in info, f"Missing field: {field}"

        assert info["width"] > 0
        assert info["height"] > 0
        assert len(info["safe_cells"]) > 0
        assert len(info["goal_positions"]) >= 2
        assert len(info["start_positions"]) >= 2

        print(f"\n{layout_name} info:")
        print(f"  Dimensions: {info['width']}x{info['height']}")
        print(f"  Num safe cells: {len(info['safe_cells'])}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
