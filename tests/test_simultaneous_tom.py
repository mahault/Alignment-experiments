"""
Test simultaneous Theory of Mind (ToM) predictions.

This script tests whether agents can accurately predict each other's actions
in simultaneous play scenarios.

Key insight: In simultaneous play, agents don't know each other's actions.
Each agent should predict the other's INDEPENDENT action (what they do
without knowing the other's choice).

Tests:
1. Asymmetric: One agent has ToM, one doesn't - should work
2. Symmetric: Both agents have ToM - may fail due to recursion
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from tom.envs.env_lava_variants import get_layout
from tom.models.model_lava import LavaModel, LavaAgent
from tom.planning.si_empathy_lava import EmpathicLavaPlanner

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


def predict_independent_action(model, qs_self, qs_other):
    """
    Predict agent's independent action using corrected ToM.

    This computes what the agent would do without knowing the other's action,
    but checking collision with the other's CURRENT position.

    Returns
    -------
    action : int
        Predicted action index
    G : np.ndarray
        EFE values for each action
    """
    B = np.array(model.B["location_state"])
    A_loc = np.array(model.A["location_obs"])
    C_loc = np.array(model.C["location_obs"])
    A_edge = np.array(model.A["edge_obs"])
    C_edge = np.array(model.C["edge_obs"])
    A_cell_collision = np.array(model.A["cell_collision_obs"])
    C_cell_collision = np.array(model.C["cell_collision_obs"])

    G = []
    for a in range(5):
        # Propagate belief
        if B.ndim == 4:
            qs_next = np.zeros_like(qs_self)
            for s_other in range(len(qs_other)):
                qs_next += B[:, :, s_other, a] @ qs_self * qs_other[s_other]
        else:
            qs_next = B[:, :, a] @ qs_self

        # Location utility
        obs_dist = A_loc @ qs_next
        loc_utility = float((obs_dist * C_loc).sum())

        # Edge utility
        edge_dist = A_edge[:, :, a] @ qs_next
        edge_utility = float((edge_dist * C_edge).sum())

        # Cell collision with other's CURRENT position
        cell_obs_dist = np.einsum("oij,i,j->o", A_cell_collision, qs_next, qs_other)
        cell_coll_utility = float((cell_obs_dist * C_cell_collision).sum())

        G.append(-loc_utility - edge_utility - cell_coll_utility)

    return int(np.argmin(G)), np.array(G)


def test_asymmetric_tom(layout_name="crossed_goals"):
    """
    Test ToM when one agent has ToM and one doesn't.

    Setup:
    - Agent j: alpha=0 (non-ToM), just minimizes own EFE
    - Agent i: has ToM, predicts j's independent action

    This should work because j's behavior is predictable.
    """
    print("=" * 70)
    print(f"ASYMMETRIC ToM TEST: {layout_name}")
    print("  j: alpha=0 (non-ToM)")
    print("  i: has ToM, predicts j")
    print("=" * 70)

    layout = get_layout(layout_name)
    goal_i, goal_j = layout.goal_positions[0], layout.goal_positions[1]
    start_i, start_j = layout.start_positions[0], layout.start_positions[1]

    print(f"\ni at {start_i} -> goal {goal_i}")
    print(f"j at {start_j} -> goal {goal_j}")

    model_i = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells,
        goal_x=goal_i[0], goal_y=goal_i[1],
        start_pos=start_i, num_empathy_levels=3
    )
    model_j = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells,
        goal_x=goal_j[0], goal_y=goal_j[1],
        start_pos=start_j, num_empathy_levels=3
    )

    agent_i = LavaAgent(model=model_i)
    agent_j = LavaAgent(model=model_j)

    qs_i = np.array(model_i.D["location_state"])
    qs_j = np.array(model_j.D["location_state"])

    # j's actual action (non-ToM, alpha=0)
    planner_j = EmpathicLavaPlanner(agent_j, agent_i, alpha=0.0, alpha_other=0.0)
    G_j_actual, _, _, _, action_j_actual = planner_j.plan(qs_j, qs_i)

    print(f"\nj (non-ToM) actual action: {ACTION_NAMES[action_j_actual]}")
    print(f"j's G: {[f'{ACTION_NAMES[a]}:{G_j_actual[a]:.1f}' for a in range(5)]}")

    # i's ToM prediction of j
    action_j_predicted, G_j_pred = predict_independent_action(model_j, qs_j, qs_i)

    print(f"\ni's ToM predicts j will do: {ACTION_NAMES[action_j_predicted]}")

    if action_j_predicted == action_j_actual:
        print("[SUCCESS] ToM prediction is CORRECT!")
        return True
    else:
        print("[FAIL] ToM prediction is WRONG!")
        print(f"  Predicted G_j: {[f'{G_j_pred[a]:.1f}' for a in range(5)]}")
        print(f"  Actual G_j:    {[f'{G_j_actual[a]:.1f}' for a in range(5)]}")
        return False


def test_symmetric_tom(layout_name="crossed_goals"):
    """
    Test ToM when BOTH agents have ToM (alpha > 0).

    This tests whether predictions are still accurate when both agents
    use ToM-based planning.
    """
    print("\n" + "=" * 70)
    print(f"SYMMETRIC ToM TEST: {layout_name}")
    print("  Both agents have alpha=1 (ToM)")
    print("=" * 70)

    layout = get_layout(layout_name)
    goal_i, goal_j = layout.goal_positions[0], layout.goal_positions[1]
    start_i, start_j = layout.start_positions[0], layout.start_positions[1]

    print(f"\ni at {start_i} -> goal {goal_i}")
    print(f"j at {start_j} -> goal {goal_j}")

    model_i = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells,
        goal_x=goal_i[0], goal_y=goal_i[1],
        start_pos=start_i, num_empathy_levels=3
    )
    model_j = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells,
        goal_x=goal_j[0], goal_y=goal_j[1],
        start_pos=start_j, num_empathy_levels=3
    )

    agent_i = LavaAgent(model=model_i)
    agent_j = LavaAgent(model=model_j)

    qs_i = np.array(model_i.D["location_state"])
    qs_j = np.array(model_j.D["location_state"])

    # i's ToM prediction of j (assuming j is independent/selfish)
    action_j_pred_by_i, _ = predict_independent_action(model_j, qs_j, qs_i)
    print(f"\ni predicts j will do: {ACTION_NAMES[action_j_pred_by_i]}")

    # j's ToM prediction of i (assuming i is independent/selfish)
    action_i_pred_by_j, _ = predict_independent_action(model_i, qs_i, qs_j)
    print(f"j predicts i will do: {ACTION_NAMES[action_i_pred_by_j]}")

    # i with ToM (alpha=1) plans
    planner_i = EmpathicLavaPlanner(agent_i, agent_j, alpha=1.0, alpha_other=0.0)
    G_i, G_j_sim, G_social_i, _, action_i_tom = planner_i.plan(qs_i, qs_j)

    # j with ToM (alpha=1) plans
    planner_j = EmpathicLavaPlanner(agent_j, agent_i, alpha=1.0, alpha_other=0.0)
    G_j, G_i_sim, G_social_j, _, action_j_tom = planner_j.plan(qs_j, qs_i)

    print(f"\ni with ToM (alpha=1) chooses: {ACTION_NAMES[action_i_tom]}")
    print(f"  G_social: {[f'{ACTION_NAMES[a]}:{G_social_i[a]:.1f}' for a in range(5)]}")

    print(f"\nj with ToM (alpha=1) chooses: {ACTION_NAMES[action_j_tom]}")
    print(f"  G_social: {[f'{ACTION_NAMES[a]}:{G_social_j[a]:.1f}' for a in range(5)]}")

    # Check predictions
    print("\n--- Prediction Accuracy ---")

    i_pred_correct = action_j_pred_by_i == action_j_tom
    j_pred_correct = action_i_pred_by_j == action_i_tom

    print(f"i predicted j={ACTION_NAMES[action_j_pred_by_i]}, j actually did {ACTION_NAMES[action_j_tom]}: {'CORRECT' if i_pred_correct else 'WRONG'}")
    print(f"j predicted i={ACTION_NAMES[action_i_pred_by_j]}, i actually did {ACTION_NAMES[action_i_tom]}: {'CORRECT' if j_pred_correct else 'WRONG'}")

    if i_pred_correct and j_pred_correct:
        print("\n[SUCCESS] Both predictions correct!")
    else:
        print("\n[FAIL] At least one prediction wrong!")
        print("\nWhy? The ToM assumes the other agent acts independently (alpha=0),")
        print("but both agents actually have alpha=1, which changes their actions.")

    return i_pred_correct and j_pred_correct


def test_adjacent_conflict():
    """
    Test the critical adjacent conflict scenario.

    Agents are facing each other in a narrow corridor.
    This is where coordination is most challenging.
    """
    print("\n" + "=" * 70)
    print("ADJACENT CONFLICT TEST")
    print("  Agents facing each other in narrow corridor")
    print("=" * 70)

    layout = get_layout("narrow")
    pos_i, pos_j = (2, 1), (3, 1)
    goal_i, goal_j = (5, 1), (0, 1)

    print(f"\ni at {pos_i} -> goal {goal_i} (wants RIGHT)")
    print(f"j at {pos_j} -> goal {goal_j} (wants LEFT)")
    print("They are ADJACENT and FACING each other!")

    model_i = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells,
        goal_x=goal_i[0], goal_y=goal_i[1],
        start_pos=pos_i, num_empathy_levels=3
    )
    model_j = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells,
        goal_x=goal_j[0], goal_y=goal_j[1],
        start_pos=pos_j, num_empathy_levels=3
    )

    agent_i = LavaAgent(model=model_i)
    agent_j = LavaAgent(model=model_j)

    # Point beliefs at current positions
    qs_i = np.zeros(layout.width * layout.height)
    qs_i[pos_i[1] * layout.width + pos_i[0]] = 1.0
    qs_j = np.zeros(layout.width * layout.height)
    qs_j[pos_j[1] * layout.width + pos_j[0]] = 1.0

    # Test 1: Both alpha=0 (selfish)
    print("\n--- Both alpha=0 (selfish) ---")
    planner_i_0 = EmpathicLavaPlanner(agent_i, agent_j, alpha=0.0, alpha_other=0.0)
    planner_j_0 = EmpathicLavaPlanner(agent_j, agent_i, alpha=0.0, alpha_other=0.0)

    G_i_0, _, _, _, action_i_0 = planner_i_0.plan(qs_i, qs_j)
    G_j_0, _, _, _, action_j_0 = planner_j_0.plan(qs_j, qs_i)

    print(f"i G: {[f'{ACTION_NAMES[a]}:{G_i_0[a]:.1f}' for a in range(5)]}")
    print(f"i chooses: {ACTION_NAMES[action_i_0]}")
    print(f"j G: {[f'{ACTION_NAMES[a]}:{G_j_0[a]:.1f}' for a in range(5)]}")
    print(f"j chooses: {ACTION_NAMES[action_j_0]}")

    # Test 2: Both alpha=1 (ToM/empathic)
    print("\n--- Both alpha=1 (ToM/empathic) ---")
    planner_i_1 = EmpathicLavaPlanner(agent_i, agent_j, alpha=1.0, alpha_other=0.0)
    planner_j_1 = EmpathicLavaPlanner(agent_j, agent_i, alpha=1.0, alpha_other=0.0)

    G_i_1, G_j_sim_1, G_social_i_1, _, action_i_1 = planner_i_1.plan(qs_i, qs_j)
    G_j_1, G_i_sim_1, G_social_j_1, _, action_j_1 = planner_j_1.plan(qs_j, qs_i)

    print(f"i G_social: {[f'{ACTION_NAMES[a]}:{G_social_i_1[a]:.1f}' for a in range(5)]}")
    print(f"i chooses: {ACTION_NAMES[action_i_1]}")
    print(f"j G_social: {[f'{ACTION_NAMES[a]}:{G_social_j_1[a]:.1f}' for a in range(5)]}")
    print(f"j chooses: {ACTION_NAMES[action_j_1]}")

    # ToM predictions
    print("\n--- ToM Predictions ---")
    action_j_pred, _ = predict_independent_action(model_j, qs_j, qs_i)
    action_i_pred, _ = predict_independent_action(model_i, qs_i, qs_j)

    print(f"i predicts j will do: {ACTION_NAMES[action_j_pred]}")
    print(f"j predicts i will do: {ACTION_NAMES[action_i_pred]}")
    print(f"i actually does: {ACTION_NAMES[action_i_1]}")
    print(f"j actually does: {ACTION_NAMES[action_j_1]}")

    i_pred_correct = action_j_pred == action_j_1
    j_pred_correct = action_i_pred == action_i_1

    print(f"\ni's prediction: {'CORRECT' if i_pred_correct else 'WRONG'}")
    print(f"j's prediction: {'CORRECT' if j_pred_correct else 'WRONG'}")

    # Analysis
    print("\n--- ANALYSIS ---")
    if action_i_0 == action_i_1 and action_j_0 == action_j_1:
        print("Alpha=0 and alpha=1 produce SAME actions.")
        print("ToM predictions are accurate because empathy doesn't change behavior here.")
    else:
        print("Alpha changes the actions!")
        print("ToM predictions may be wrong if they assume alpha=0 but agent has alpha>0.")

    if action_i_1 == 4 and action_j_1 == 4:  # STAY
        print("\nBoth choose STAY - PARALYSIS!")
        print("Neither agent can make progress because moving forward = collision.")

    return i_pred_correct and j_pred_correct


def run_all_tests():
    """Run all ToM tests and summarize results."""
    print("\n" + "=" * 80)
    print("SIMULTANEOUS ToM TEST SUITE")
    print("=" * 80)

    results = {}

    # Asymmetric tests
    for layout in ["crossed_goals", "narrow", "t_junction", "symmetric_bottleneck"]:
        results[f"asym_{layout}"] = test_asymmetric_tom(layout)

    # Symmetric tests
    for layout in ["crossed_goals", "narrow"]:
        results[f"sym_{layout}"] = test_symmetric_tom(layout)

    # Adjacent conflict
    results["adjacent_conflict"] = test_adjacent_conflict()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    total_pass = sum(results.values())
    total = len(results)
    print(f"\nTotal: {total_pass}/{total} tests passed")

    return results


if __name__ == "__main__":
    run_all_tests()
