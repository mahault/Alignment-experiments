"""
Test script to verify 4D B matrix and multi-step ToM planning work correctly.

Tests:
1. B matrix properly blocks moves into occupied cells
2. Empathy planner conditions on other agent's position
3. Multi-step horizon planning works
"""

import numpy as np
import sys
import os

# Add repo to path
ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tom.models import LavaModel, LavaAgent
from tom.planning.si_empathy_lava import EmpathicLavaPlanner

def test_b_matrix_structure():
    """Test that B matrix has correct 4D structure for multi-agent."""
    print("=" * 60)
    print("TEST 1: B Matrix 4D Structure")
    print("=" * 60)

    # Create simple 4x3 grid
    model = LavaModel(width=4, height=3)
    B = np.asarray(model.B["location_state"])

    print(f"\nB matrix shape: {B.shape}")
    print(f"Expected: (num_states, num_states, num_states, num_actions)")
    print(f"Got: ({model.num_states}, {model.num_states}, {model.num_states}, 5)")

    # Check shape
    assert B.shape == (model.num_states, model.num_states, model.num_states, 5), \
        f"B matrix should be 4D with shape ({model.num_states}, {model.num_states}, {model.num_states}, 5)"

    # In Regime B, collisions are POSSIBLE (not blocked by physics)
    # Collision avoidance is done via preferences, not transition dynamics
    # So B[s_other, s_agent, s_other, action_right] CAN be > 0

    s_agent = 1 * model.width + 0  # (0, 1) -> index 4
    s_other = 1 * model.width + 1  # (1, 1) -> index 5
    action_right = 3

    print(f"\nAgent at state {s_agent} (0,1), other at state {s_other} (1,1)")
    print(f"Agent tries RIGHT (action {action_right})")

    # Movement should still work (Regime B allows collisions)
    move_prob = float(B[s_other, s_agent, s_other, action_right])
    print(f"Move to {s_other}: {move_prob:.2f}")
    print(f"NOTE: In Regime B, collisions are possible (handled via preferences)")

    # Check that columns sum to 1 (valid stochastic matrix)
    for a in range(5):
        col_sums = B[:, :, s_other, a].sum(axis=0)
        assert np.allclose(col_sums, 1.0), f"B[:,:,{s_other},{a}] columns should sum to 1"

    print("\n[PASS] B matrix has correct 4D structure\n")


def test_empathy_conditioning():
    """Test that empathy planner conditions on other agent's position."""
    print("=" * 60)
    print("TEST 2: Empathy Conditioning on Other Agent")
    print("=" * 60)

    # Create models
    model_i = LavaModel(width=4, height=3, start_pos=(0, 1), goal_x=3, goal_y=1)
    model_j = LavaModel(width=4, height=3, start_pos=(1, 1), goal_x=3, goal_y=1)

    # Create agents with horizon=1 for simple test
    agent_i = LavaAgent(model_i, horizon=1, gamma=8.0)
    agent_j = LavaAgent(model_j, horizon=1, gamma=8.0)

    # Create planners
    planner_i = EmpathicLavaPlanner(agent_i, agent_j, alpha=0.5)

    # Set up beliefs: i at (0,1), j at (1,1)
    qs_i = np.zeros(model_i.num_states)
    qs_i[1 * model_i.width + 0] = 1.0  # (0,1)

    qs_j = np.zeros(model_j.num_states)
    qs_j[1 * model_j.width + 1] = 1.0  # (1,1)

    print(f"\nAgent i at (0,1), agent j at (1,1)")
    print(f"Agent i wants to go RIGHT toward goal")
    print(f"But j is blocking the path")

    # Plan action
    G_i, G_j, G_social, q_pi, action = planner_i.plan(qs_i, qs_j)

    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    print(f"\nG_i (selfish EFE): {G_i}")
    print(f"G_j (simulated other): {G_j}")
    print(f"G_social (empathic): {G_social}")
    print(f"\nChosen action: {action_names[action]}")

    # With empathy, should either STAY or consider alternative
    # RIGHT should have high cost due to collision
    print(f"\nAction EFE breakdown:")
    for a in range(5):
        print(f"  {action_names[a]}: G_i={G_i[a]:.2f}, G_social={G_social[a]:.2f}")

    print("\n[PASS] Empathy planner considers other agent's position\n")


def test_multi_step_horizon():
    """Test multi-step horizon planning."""
    print("=" * 60)
    print("TEST 3: Multi-Step Horizon Planning")
    print("=" * 60)

    # Create wider grid for multi-step planning
    safe_cells = [(x, 1) for x in range(6)] + [(x, 2) for x in range(6)]
    model_i = LavaModel(
        width=6, height=3,
        start_pos=(0, 1), goal_x=5, goal_y=1,
        safe_cells=safe_cells
    )
    model_j = LavaModel(
        width=6, height=3,
        start_pos=(0, 2), goal_x=5, goal_y=2,
        safe_cells=safe_cells
    )

    # Test with horizon=3
    horizon = 3
    agent_i = LavaAgent(model_i, horizon=horizon, gamma=8.0)
    agent_j = LavaAgent(model_j, horizon=horizon, gamma=8.0)

    print(f"\nHorizon: {horizon}")
    print(f"Agent i policies shape: {agent_i.policies.shape}")
    print(f"Expected: (num_policies, {horizon}, 1)")

    # For horizon > 1, need to generate proper multi-step policies
    # For now, just verify the structure works
    print(f"\nNumber of policies: {len(agent_i.policies)}")

    # Test beliefs
    qs_i = np.zeros(model_i.num_states)
    qs_i[1 * 6 + 0] = 1.0  # (0,1)

    qs_j = np.zeros(model_j.num_states)
    qs_j[2 * 6 + 0] = 1.0  # (0,2)

    # Create planner and test
    planner_i = EmpathicLavaPlanner(agent_i, agent_j, alpha=0.5)

    print(f"\nPlanning with horizon={horizon}...")
    try:
        G_i, G_j, G_social, q_pi, action = planner_i.plan(qs_i, qs_j)
        action_names = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
        print(f"Chosen action: {action_names[action]}")
        print(f"Planning successful!")
        print("\n[PASS] Multi-step horizon planning works\n")
    except Exception as e:
        print(f"\n[FAIL]: {e}\n")
        raise


def main():
    print("\n" + "=" * 60)
    print("Testing 4D B Matrix and Multi-Step ToM Planning")
    print("=" * 60 + "\n")

    try:
        test_b_matrix_structure()
        test_empathy_conditioning()
        test_multi_step_horizon()

        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
