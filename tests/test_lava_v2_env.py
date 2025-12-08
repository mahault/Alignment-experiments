"""
Unit tests for LavaV2Env with environment variants.

Verifies that:
1. All layout variants can be created
2. Extended observations include other agent's position
3. Different starting positions work correctly
4. Collision detection works properly
5. Goal reaching is detected correctly
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


class TestLayoutCreation:
    """Test that all layout variants can be created."""

    def test_narrow_layout(self):
        """Test narrow corridor layout."""
        layout = get_layout("narrow", width=6)

        assert layout.width == 6
        assert layout.height == 3
        assert layout.name == "narrow_corridor"
        assert len(layout.start_positions) >= 2

        print(f"\nNarrow layout:")
        print(f"  Dimensions: {layout.width}x{layout.height}")
        print(f"  Goal: {layout.goal_pos}")
        print(f"  Start positions: {layout.start_positions}")

    def test_wide_layout(self):
        """Test wide corridor layout."""
        layout = get_layout("wide", width=6)

        assert layout.width == 6
        assert layout.height == 4  # Wider
        assert layout.name == "wide_corridor"

        # Should have safe cells in multiple rows
        safe_y_coords = set(y for x, y in layout.safe_cells)
        assert len(safe_y_coords) >= 2, "Wide layout should have multiple safe rows"

        print(f"\nWide layout:")
        print(f"  Dimensions: {layout.width}x{layout.height}")
        print(f"  Safe rows: {sorted(safe_y_coords)}")

    def test_bottleneck_layout(self):
        """Test bottleneck layout."""
        layout = get_layout("bottleneck", width=8)

        assert layout.width == 8
        assert layout.name == "bottleneck"

        print(f"\nBottleneck layout:")
        print(f"  Dimensions: {layout.width}x{layout.height}")
        print(f"  Num safe cells: {len(layout.safe_cells)}")

    def test_risk_reward_layout(self):
        """Test risk-reward layout."""
        layout = get_layout("risk_reward", width=8)

        assert layout.width == 8
        assert layout.name == "risk_reward"

        print(f"\nRisk-reward layout:")
        print(f"  Dimensions: {layout.width}x{layout.height}")


class TestLavaV2EnvCreation:
    """Test LavaV2Env creation with different layouts."""

    def test_create_wide_env(self):
        """Test creating environment with wide layout."""
        env = LavaV2Env(layout_name="wide", width=6, num_agents=2, timesteps=20)

        assert env.width == 6
        assert env.height == 4
        assert env.num_agents == 2

        print(f"\nWide environment created:")
        print(f"  Dimensions: {env.width}x{env.height}")

    def test_create_bottleneck_env(self):
        """Test creating environment with bottleneck layout."""
        env = LavaV2Env(layout_name="bottleneck", width=8, num_agents=2, timesteps=20)

        assert env.width == 8
        assert env.num_agents == 2

        print(f"\nBottleneck environment created:")
        print(f"  Dimensions: {env.width}x{env.height}")


class TestExtendedObservations:
    """Test that observations include other agent's position."""

    def test_observation_structure(self):
        """Test that observations include both own and other position."""
        env = LavaV2Env(layout_name="wide", width=6, num_agents=2, timesteps=20)

        key = jr.PRNGKey(0)
        state, obs = env.reset(key)

        # Check observation structure for agent 0
        assert 0 in obs, "Should have observation for agent 0"
        assert "location_obs" in obs[0], "Should have location_obs"
        assert "other_obs" in obs[0], "Should have other_obs (extended observation)"

        # Check values are different (agents start at different positions)
        my_obs = int(obs[0]["location_obs"][0])
        other_obs = int(obs[0]["other_obs"][0])

        print(f"\nExtended observations:")
        print(f"  Agent 0 sees itself at: {my_obs}")
        print(f"  Agent 0 sees other at: {other_obs}")

        # In wide layout, agents start at different positions
        assert my_obs != other_obs, "Agents should start at different positions in wide layout"

    def test_symmetric_observations(self):
        """Test that both agents see each other correctly."""
        env = LavaV2Env(layout_name="wide", width=6, num_agents=2, timesteps=20)

        key = jr.PRNGKey(0)
        state, obs = env.reset(key)

        # Agent 0's view
        i_my_pos = int(obs[0]["location_obs"][0])
        i_other_pos = int(obs[0]["other_obs"][0])

        # Agent 1's view
        j_my_pos = int(obs[1]["location_obs"][0])
        j_other_pos = int(obs[1]["other_obs"][0])

        # Symmetric observation: what i sees as "other" should be what j sees as "self"
        assert i_other_pos == j_my_pos, "Agent i's 'other' should match agent j's 'self'"
        assert j_other_pos == i_my_pos, "Agent j's 'other' should match agent i's 'self'"

        print(f"\nSymmetric observations:")
        print(f"  Agent i: my_pos={i_my_pos}, other_pos={i_other_pos}")
        print(f"  Agent j: my_pos={j_my_pos}, other_pos={j_other_pos}")


class TestDifferentStartPositions:
    """Test that agents can start at different positions."""

    def test_wide_different_starts(self):
        """Test that wide layout starts agents at different positions."""
        env = LavaV2Env(layout_name="wide", width=6, num_agents=2, timesteps=20)

        key = jr.PRNGKey(0)
        state, obs = env.reset(key)

        pos_0 = state["env_state"]["pos"][0]
        pos_1 = state["env_state"]["pos"][1]

        assert pos_0 != pos_1, "Agents should start at different positions"

        print(f"\nDifferent start positions:")
        print(f"  Agent 0: {pos_0}")
        print(f"  Agent 1: {pos_1}")

    def test_narrow_same_x_different_y(self):
        """Test narrow layout start positions."""
        env = LavaV2Env(layout_name="narrow", width=6, num_agents=2, timesteps=20)

        key = jr.PRNGKey(0)
        state, obs = env.reset(key)

        pos_0 = state["env_state"]["pos"][0]
        pos_1 = state["env_state"]["pos"][1]

        # In narrow layout, agents start at opposite ends
        assert pos_0[0] != pos_1[0], "Agents should start at different x positions"

        print(f"\nNarrow layout start positions:")
        print(f"  Agent 0: {pos_0}")
        print(f"  Agent 1: {pos_1}")


class TestCollisionDetection:
    """Test collision detection."""

    def test_collision_when_same_position(self):
        """Test that collision is detected when agents at same position."""
        env = LavaV2Env(layout_name="wide", width=6, num_agents=2, timesteps=20)

        key = jr.PRNGKey(0)
        state, obs = env.reset(key)

        # Force both agents to same position by moving them there
        # Get current positions
        pos_0 = state["env_state"]["pos"][0]
        pos_1 = state["env_state"]["pos"][1]

        # If agents can reach same position by both moving RIGHT
        # This depends on layout, so let's just test the collision detection logic directly
        test_positions = {0: (2, 1), 1: (2, 1)}  # Same position
        collision = env._check_collision(test_positions)

        assert collision, "Should detect collision when agents at same position"

        print(f"\nCollision detection:")
        print(f"  Positions: {test_positions}")
        print(f"  Collision detected: {collision}")

    def test_no_collision_different_positions(self):
        """Test that no collision when agents at different positions."""
        env = LavaV2Env(layout_name="wide", width=6, num_agents=2, timesteps=20)

        test_positions = {0: (2, 1), 1: (3, 1)}  # Different positions
        collision = env._check_collision(test_positions)

        assert not collision, "Should not detect collision when agents at different positions"

        print(f"\nNo collision:")
        print(f"  Positions: {test_positions}")
        print(f"  Collision detected: {collision}")


class TestGoalReaching:
    """Test goal reaching detection."""

    def test_goal_detection(self):
        """Test that goal reaching is detected correctly."""
        env = LavaV2Env(layout_name="wide", width=6, num_agents=2, timesteps=20)

        goal_pos = env.layout.goal_pos

        # Create state with agent at goal
        state = {
            "env_state": {
                "pos": {0: goal_pos, 1: (0, 1)},
                "timestep": 0,
            },
            "timestep": 0,
            "done": False,
        }

        # Step with STAY action
        next_state, obs, reward, done, info = env.step(state, {0: 4, 1: 4})

        assert info["goal_reached"][0], "Agent 0 should have reached goal"
        assert not info["goal_reached"][1], "Agent 1 should not have reached goal"

        print(f"\nGoal detection:")
        print(f"  Goal position: {goal_pos}")
        print(f"  Agent 0 reached goal: {info['goal_reached'][0]}")
        print(f"  Agent 1 reached goal: {info['goal_reached'][1]}")


class TestRendering:
    """Test environment rendering."""

    def test_render_initial_state(self):
        """Test that initial state can be rendered."""
        env = LavaV2Env(layout_name="wide", width=6, num_agents=2, timesteps=20)

        key = jr.PRNGKey(0)
        state, obs = env.reset(key)

        ascii_art = env.render_state(state)

        assert isinstance(ascii_art, str)
        assert len(ascii_art) > 0
        assert "0" in ascii_art or "1" in ascii_art, "Should show agent positions"
        assert "G" in ascii_art, "Should show goal"

        print(f"\nRendered initial state:")
        print(ascii_art)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
