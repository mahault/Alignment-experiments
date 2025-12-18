"""
Diagnostic script to verify ToM predictions match actual decisions.

Tests whether agent i's prediction of j's action matches j's actual choice.
"""

import sys
sys.path.insert(0, ".")

import jax.numpy as jnp
import numpy as np
from tom.envs.env_lava_variants import get_layout
from tom.models.model_lava import LavaModel, LavaAgent
from tom.planning.si_empathy_lava import EmpathicLavaPlanner
from tom.planning.jax_si_empathy_lava import (
    compute_j_best_response_jax,
    propagate_belief_jax,
)

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


def compute_independent_optimal(model, qs_self, qs_other, alpha):
    """
    Compute agent's independent optimal action.

    This is what the agent would do using their own EFE,
    WITHOUT conditioning on the other agent's action.
    """
    import numpy as np

    B = np.asarray(model.B["location_state"])
    A_loc = np.asarray(model.A["location_obs"])
    C_loc = np.asarray(model.C["location_obs"])

    G = np.zeros(5)

    for action in range(5):
        # Propagate belief
        if B.ndim == 4:
            qs_next = np.zeros_like(qs_self)
            for s_other in range(len(qs_other)):
                qs_next += B[:, :, s_other, action] @ qs_self * qs_other[s_other]
        else:
            qs_next = B[:, :, action] @ qs_self

        # Location utility (main driver)
        obs_dist = A_loc @ qs_next
        location_utility = float((obs_dist * C_loc).sum())

        # Simple EFE (negative utility)
        G[action] = -location_utility

    best_action = int(np.argmin(G))
    return best_action, G


def test_independent_tom(layout_name, alpha_i, alpha_j):
    """
    Test if predicting j's INDEPENDENT optimal matches j's actual choice.

    The idea: instead of "what does j do in response to my action?",
    we ask "what does j do on their own, given j's beliefs and preferences?"
    """
    print(f"\n{'='*60}")
    print(f"INDEPENDENT ToM TEST: {layout_name}")
    print(f"  alpha_i={alpha_i}, alpha_j={alpha_j}")
    print(f"{'='*60}\n")

    layout = get_layout(layout_name)

    goal_i = layout.goal_positions[0]
    goal_j = layout.goal_positions[1]
    start_i = layout.start_positions[0]
    start_j = layout.start_positions[1]

    model_i = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_i[0], goal_y=goal_i[1],
        start_pos=start_i, num_empathy_levels=3
    )
    model_j = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_j[0], goal_y=goal_j[1],
        start_pos=start_j, num_empathy_levels=3
    )

    agent_i = LavaAgent(model=model_i)
    agent_j = LavaAgent(model=model_j)

    qs_i = np.array(model_i.D["location_state"])
    qs_j = np.array(model_j.D["location_state"])

    print(f"Positions: i@{start_i}, j@{start_j}")
    print(f"Goals: i->{goal_i}, j->{goal_j}")

    # 1. Compute j's INDEPENDENT optimal (what i predicts j will do)
    j_independent, G_j_independent = compute_independent_optimal(
        model_j, qs_j, qs_i, alpha_j
    )
    print(f"\ni predicts j's independent optimal: {ACTION_NAMES[j_independent]}")
    print(f"  G_j by action: {[f'{ACTION_NAMES[a]}:{G_j_independent[a]:.1f}' for a in range(5)]}")

    # 2. Compute i's INDEPENDENT optimal (what j predicts i will do)
    i_independent, G_i_independent = compute_independent_optimal(
        model_i, qs_i, qs_j, alpha_i
    )
    print(f"\nj predicts i's independent optimal: {ACTION_NAMES[i_independent]}")
    print(f"  G_i by action: {[f'{ACTION_NAMES[a]}:{G_i_independent[a]:.1f}' for a in range(5)]}")

    # 3. What do the actual planners choose?
    planner_i = EmpathicLavaPlanner(agent_i, agent_j, alpha=alpha_i, alpha_other=alpha_j)
    planner_j = EmpathicLavaPlanner(agent_j, agent_i, alpha=alpha_j, alpha_other=alpha_i)

    _, _, _, _, actual_i = planner_i.plan(qs_i, qs_j)
    _, _, _, _, actual_j = planner_j.plan(qs_j, qs_i)

    print(f"\nActual choices (sequential ToM):")
    print(f"  i chose: {ACTION_NAMES[actual_i]}")
    print(f"  j chose: {ACTION_NAMES[actual_j]}")

    print(f"\nPrediction accuracy:")
    if j_independent == actual_j:
        print(f"  [OK] i correctly predicted j would do {ACTION_NAMES[actual_j]}")
    else:
        print(f"  [FAIL] i predicted j={ACTION_NAMES[j_independent]}, but j did {ACTION_NAMES[actual_j]}")

    if i_independent == actual_i:
        print(f"  [OK] j correctly predicted i would do {ACTION_NAMES[actual_i]}")
    else:
        print(f"  [FAIL] j predicted i={ACTION_NAMES[i_independent]}, but i did {ACTION_NAMES[actual_i]}")

    return {
        "i_predicted_j": j_independent,
        "j_predicted_i": i_independent,
        "actual_i": actual_i,
        "actual_j": actual_j,
    }


def test_simultaneous_tom_full_planning(layout_name, alpha_i, alpha_j):
    """
    Test simultaneous ToM using FULL planning for predictions.

    Legacy ToM approach:
    1. Run full planning for j independently -> j's policy
    2. Run full planning for i independently -> i's policy
    3. Each agent predicts the other by simulating their full planning

    In simultaneous games, the prediction should be:
    "What will j do given j's own beliefs/goals?" (not "response to my action")
    """
    print(f"\n{'='*60}")
    print(f"SIMULTANEOUS ToM (Full Planning): {layout_name}")
    print(f"  alpha_i={alpha_i}, alpha_j={alpha_j}")
    print(f"{'='*60}\n")

    layout = get_layout(layout_name)

    goal_i = layout.goal_positions[0]
    goal_j = layout.goal_positions[1]
    start_i = layout.start_positions[0]
    start_j = layout.start_positions[1]

    # Create models for both agents
    model_i = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_i[0], goal_y=goal_i[1],
        start_pos=start_i, num_empathy_levels=3
    )
    model_j = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_j[0], goal_y=goal_j[1],
        start_pos=start_j, num_empathy_levels=3
    )

    agent_i = LavaAgent(model=model_i)
    agent_j = LavaAgent(model=model_j)

    qs_i = jnp.array(model_i.D["location_state"])
    qs_j = jnp.array(model_j.D["location_state"])

    print(f"Positions: i@{start_i}, j@{start_j}")
    print(f"Goals: i->{goal_i}, j->{goal_j}")

    # STEP 1: Run full planning for j independently (as if j is deciding alone)
    # This is what i SHOULD predict j will do in simultaneous setting
    # Use a "selfish" planner for j (no empathy term) to get j's independent action
    # Then j with empathy would modify based on predicted i

    # For now: use the planner but with alpha_other=0 to get pure independent action
    # Actually, we want j's TRUE action given j's actual alpha

    # j plans independently
    planner_j_independent = EmpathicLavaPlanner(
        agent_j, agent_i, alpha=alpha_j, alpha_other=alpha_i
    )
    G_j, G_i_sim_j, G_social_j, _, action_j = planner_j_independent.plan(qs_j, qs_i)

    # i plans independently
    planner_i_independent = EmpathicLavaPlanner(
        agent_i, agent_j, alpha=alpha_i, alpha_other=alpha_j
    )
    G_i, G_j_sim_i, G_social_i, _, action_i = planner_i_independent.plan(qs_i, qs_j)

    print(f"\n--- AGENT J's INDEPENDENT PLANNING ---")
    print(f"j's G values by action:")
    for a in range(5):
        print(f"  {ACTION_NAMES[a]:5s}: G_j={G_j[a]:7.2f}, G_i_sim={G_i_sim_j[a]:7.2f}, G_social={G_social_j[a]:7.2f}")
    print(f"j chooses: {ACTION_NAMES[action_j]}")

    print(f"\n--- AGENT I's INDEPENDENT PLANNING ---")
    print(f"i's G values by action:")
    for a in range(5):
        print(f"  {ACTION_NAMES[a]:5s}: G_i={G_i[a]:7.2f}, G_j_sim={G_j_sim_i[a]:7.2f}, G_social={G_social_i[a]:7.2f}")
    print(f"i chooses: {ACTION_NAMES[action_i]}")

    # STEP 2: What SHOULD the ToM predictions be for simultaneous play?
    # Answer: Each agent should predict the other's INDEPENDENT action
    # i's prediction of j = what j does when j plans independently = action_j
    # j's prediction of i = what i does when i plans independently = action_i

    # STEP 3: What does the CURRENT ToM predict?
    # The current ToM is baked into G_j_sim and G_i_sim
    # G_j_sim[a] = expected G for j GIVEN i takes action a (response-based)
    # We need to extract what action j is predicted to take for each i action

    print(f"\n--- ToM ANALYSIS ---")
    print(f"For SIMULTANEOUS play, correct predictions should be:")
    print(f"  i should predict j does: {ACTION_NAMES[action_j]} (j's independent optimal)")
    print(f"  j should predict i does: {ACTION_NAMES[action_i]} (i's independent optimal)")

    # The current ToM computes j's best response to each of i's actions
    # Let's see what it predicts for the action i actually takes
    B_j = jnp.array(model_j.B["location_state"])
    B_i = jnp.array(model_i.B["location_state"])
    actions = jnp.arange(5)

    print(f"\nCurrent ToM (response-based) predictions:")

    # i's prediction of j's response to i's chosen action
    qs_i_next = propagate_belief_jax(qs_i, B_i, action_i, qs_j, eps=1e-16)
    _, j_predicted_response = compute_j_best_response_jax(
        qs_j=qs_j, qs_i=qs_i, qs_i_next=qs_i_next, action_i=action_i,
        B_j=B_j,
        A_j_loc=jnp.array(model_j.A["location_obs"]),
        C_j_loc=jnp.array(model_j.C["location_obs"]),
        A_j_edge=jnp.array(model_j.A["edge_obs"]),
        C_j_edge=jnp.array(model_j.C["edge_obs"]),
        A_j_cell_collision=jnp.array(model_j.A["cell_collision_obs"]),
        C_j_cell_collision=jnp.array(model_j.C["cell_collision_obs"]),
        A_j_edge_collision=jnp.array(model_j.A["edge_collision_obs"]),
        C_j_edge_collision=jnp.array(model_j.C["edge_collision_obs"]),
        actions_j=actions, epistemic_scale=0.1, alpha_other=alpha_j, eps=1e-16,
    )
    print(f"  i predicts j's RESPONSE to i={ACTION_NAMES[action_i]}: {ACTION_NAMES[int(j_predicted_response)]}")

    # j's prediction of i's response to j's chosen action
    qs_j_next = propagate_belief_jax(qs_j, B_j, action_j, qs_i, eps=1e-16)
    _, i_predicted_response = compute_j_best_response_jax(
        qs_j=qs_i, qs_i=qs_j, qs_i_next=qs_j_next, action_i=action_j,
        B_j=B_i,
        A_j_loc=jnp.array(model_i.A["location_obs"]),
        C_j_loc=jnp.array(model_i.C["location_obs"]),
        A_j_edge=jnp.array(model_i.A["edge_obs"]),
        C_j_edge=jnp.array(model_i.C["edge_obs"]),
        A_j_cell_collision=jnp.array(model_i.A["cell_collision_obs"]),
        C_j_cell_collision=jnp.array(model_i.C["cell_collision_obs"]),
        A_j_edge_collision=jnp.array(model_i.A["edge_collision_obs"]),
        C_j_edge_collision=jnp.array(model_i.C["edge_collision_obs"]),
        actions_j=actions, epistemic_scale=0.1, alpha_other=alpha_i, eps=1e-16,
    )
    print(f"  j predicts i's RESPONSE to j={ACTION_NAMES[action_j]}: {ACTION_NAMES[int(i_predicted_response)]}")

    print(f"\n--- VERDICT ---")

    # For simultaneous play, correct ToM should predict INDEPENDENT actions, not responses
    i_correct = (int(j_predicted_response) == action_j)
    j_correct = (int(i_predicted_response) == action_i)

    if i_correct:
        print(f"  [OK] i's prediction matches j's actual independent choice")
    else:
        print(f"  [FAIL] i predicted j={ACTION_NAMES[int(j_predicted_response)]}, but j independently chose {ACTION_NAMES[action_j]}")
        print(f"         (Response-based ToM gives wrong prediction for simultaneous play)")

    if j_correct:
        print(f"  [OK] j's prediction matches i's actual independent choice")
    else:
        print(f"  [FAIL] j predicted i={ACTION_NAMES[int(i_predicted_response)]}, but i independently chose {ACTION_NAMES[action_i]}")
        print(f"         (Response-based ToM gives wrong prediction for simultaneous play)")

    # What would happen with CORRECT simultaneous ToM?
    print(f"\n--- IF SIMULTANEOUS ToM WERE USED ---")
    print(f"i would predict j does {ACTION_NAMES[action_j]} -> i can plan accordingly")
    print(f"j would predict i does {ACTION_NAMES[action_i]} -> j can plan accordingly")

    # Check if collision would occur
    # Get next positions
    def get_next_pos(pos, action):
        dx = [0, 0, -1, 1, 0]  # UP, DOWN, LEFT, RIGHT, STAY
        dy = [-1, 1, 0, 0, 0]
        new_x = pos[0] + dx[action]
        new_y = pos[1] + dy[action]
        # Bounds check
        if 0 <= new_x < layout.width and 0 <= new_y < layout.height:
            new_pos = (new_x, new_y)
            if new_pos in layout.safe_cells:
                return new_pos
        return pos  # Stay in place if invalid

    next_i = get_next_pos(start_i, action_i)
    next_j = get_next_pos(start_j, action_j)

    collision = (next_i == next_j)
    swap = (next_i == start_j and next_j == start_i)

    if collision or swap:
        print(f"\n  [WARNING] With current choices: COLLISION!")
        print(f"    i: {start_i} -> {next_i}")
        print(f"    j: {start_j} -> {next_j}")
        if collision:
            print(f"    Both end up at same cell!")
        if swap:
            print(f"    Agents try to swap positions!")
    else:
        print(f"\n  With current choices: No collision")
        print(f"    i: {start_i} -> {next_i}")
        print(f"    j: {start_j} -> {next_j}")

    return {
        "action_i": action_i,
        "action_j": action_j,
        "i_predicted_j": int(j_predicted_response),
        "j_predicted_i": int(i_predicted_response),
        "correct_i_prediction": action_j,  # What i SHOULD predict for simultaneous
        "correct_j_prediction": action_i,  # What j SHOULD predict for simultaneous
        "collision": collision or swap,
    }


def diagnose_tom(layout_name="narrow", alpha_i=0.0, alpha_j=0.0, alpha_other_i=None, alpha_other_j=None):
    """
    Diagnose ToM predictions for a given scenario.

    Parameters
    ----------
    layout_name : str
        Layout to test
    alpha_i : float
        Agent i's empathy
    alpha_j : float
        Agent j's empathy
    alpha_other_i : float, optional
        i's belief about j's empathy (defaults to alpha_j)
    alpha_other_j : float, optional
        j's belief about i's empathy (defaults to alpha_i)
    """
    if alpha_other_i is None:
        alpha_other_i = alpha_j
    if alpha_other_j is None:
        alpha_other_j = alpha_i

    print(f"\n{'='*60}")
    print(f"DIAGNOSING ToM: {layout_name}")
    print(f"  Agent i: alpha={alpha_i}, believes j has alpha={alpha_other_i}")
    print(f"  Agent j: alpha={alpha_j}, believes i has alpha={alpha_other_j}")
    print(f"{'='*60}\n")

    # Setup
    layout = get_layout(layout_name)

    # Create agents
    goal_i = layout.goal_positions[0]
    goal_j = layout.goal_positions[1]
    start_i = layout.start_positions[0]
    start_j = layout.start_positions[1]

    model_i = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_i[0], goal_y=goal_i[1],
        start_pos=start_i, num_empathy_levels=3
    )
    model_j = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_j[0], goal_y=goal_j[1],
        start_pos=start_j, num_empathy_levels=3
    )

    agent_i = LavaAgent(model=model_i)
    agent_j = LavaAgent(model=model_j)

    # Create planners
    planner_i = EmpathicLavaPlanner(
        agent_i, agent_j, alpha=alpha_i, alpha_other=alpha_other_i
    )
    planner_j = EmpathicLavaPlanner(
        agent_j, agent_i, alpha=alpha_j, alpha_other=alpha_other_j
    )

    # Initial beliefs (point mass at start positions)
    qs_i = jnp.array(model_i.D["location_state"])
    qs_j = jnp.array(model_j.D["location_state"])

    print(f"Positions: i@{start_i}, j@{start_j}")
    print(f"Goals: i->{goal_i}, j->{goal_j}")
    print()

    # Agent i plans
    print("--- AGENT I's PLANNING ---")
    G_i, G_j_sim, G_social_i, q_pi_i, action_i = planner_i.plan(qs_i, qs_j)

    print(f"i's G values by action:")
    for a in range(5):
        print(f"  {ACTION_NAMES[a]:5s}: G_i={G_i[a]:7.2f}, G_j_sim={G_j_sim[a]:7.2f}, G_social={G_social_i[a]:7.2f}")
    print(f"i chooses: {ACTION_NAMES[action_i]}")
    print()

    # What does i predict j will do?
    # For each of i's actions, compute j's predicted best response
    print("i's ToM predictions (what i thinks j will do given i's action):")
    B_j = jnp.array(model_j.B["location_state"])
    actions = jnp.arange(5)

    for action_i_test in range(5):
        # i's next state given this action
        B_i = jnp.array(model_i.B["location_state"])
        qs_i_next = propagate_belief_jax(qs_i, B_i, action_i_test, qs_j, eps=1e-16)

        # What does i predict j will do?
        G_j_best, j_predicted = compute_j_best_response_jax(
            qs_j=qs_j,
            qs_i=qs_i,
            qs_i_next=qs_i_next,
            action_i=action_i_test,
            B_j=B_j,
            A_j_loc=jnp.array(model_j.A["location_obs"]),
            C_j_loc=jnp.array(model_j.C["location_obs"]),
            A_j_edge=jnp.array(model_j.A["edge_obs"]),
            C_j_edge=jnp.array(model_j.C["edge_obs"]),
            A_j_cell_collision=jnp.array(model_j.A["cell_collision_obs"]),
            C_j_cell_collision=jnp.array(model_j.C["cell_collision_obs"]),
            A_j_edge_collision=jnp.array(model_j.A["edge_collision_obs"]),
            C_j_edge_collision=jnp.array(model_j.C["edge_collision_obs"]),
            actions_j=actions,
            epistemic_scale=0.1,
            alpha_other=alpha_other_i,  # i's belief about j's empathy
            eps=1e-16,
        )
        print(f"  If i does {ACTION_NAMES[action_i_test]:5s}: i predicts j does {ACTION_NAMES[int(j_predicted)]:5s} (G_j={G_j_best:.2f})")

    print()

    # Agent j plans (actual decision)
    print("--- AGENT J's PLANNING ---")
    G_j, G_i_sim, G_social_j, q_pi_j, action_j = planner_j.plan(qs_j, qs_i)

    print(f"j's G values by action:")
    for a in range(5):
        print(f"  {ACTION_NAMES[a]:5s}: G_j={G_j[a]:7.2f}, G_i_sim={G_i_sim[a]:7.2f}, G_social={G_social_j[a]:7.2f}")
    print(f"j chooses: {ACTION_NAMES[action_j]}")
    print()

    # Compare prediction vs reality
    print("--- COMPARISON ---")
    # i predicted j's response to i's chosen action
    B_i = jnp.array(model_i.B["location_state"])
    qs_i_next = propagate_belief_jax(qs_i, B_i, action_i, qs_j, eps=1e-16)
    _, j_predicted_for_actual_i = compute_j_best_response_jax(
        qs_j=qs_j, qs_i=qs_i, qs_i_next=qs_i_next, action_i=action_i,
        B_j=B_j,
        A_j_loc=jnp.array(model_j.A["location_obs"]),
        C_j_loc=jnp.array(model_j.C["location_obs"]),
        A_j_edge=jnp.array(model_j.A["edge_obs"]),
        C_j_edge=jnp.array(model_j.C["edge_obs"]),
        A_j_cell_collision=jnp.array(model_j.A["cell_collision_obs"]),
        C_j_cell_collision=jnp.array(model_j.C["cell_collision_obs"]),
        A_j_edge_collision=jnp.array(model_j.A["edge_collision_obs"]),
        C_j_edge_collision=jnp.array(model_j.C["edge_collision_obs"]),
        actions_j=actions, epistemic_scale=0.1, alpha_other=alpha_other_i, eps=1e-16,
    )

    print(f"i chose {ACTION_NAMES[action_i]}, predicted j would do {ACTION_NAMES[int(j_predicted_for_actual_i)]}")
    print(f"j actually chose: {ACTION_NAMES[action_j]}")

    if int(j_predicted_for_actual_i) == action_j:
        print("[OK] ToM CORRECT: i's prediction matches j's actual choice")
    else:
        print("[FAIL] ToM MISMATCH: i predicted wrong!")

    # Check the reverse too
    B_j = jnp.array(model_j.B["location_state"])
    qs_j_next = propagate_belief_jax(qs_j, B_j, action_j, qs_i, eps=1e-16)
    _, i_predicted_for_actual_j = compute_j_best_response_jax(
        qs_j=qs_i, qs_i=qs_j, qs_i_next=qs_j_next, action_i=action_j,
        B_j=jnp.array(model_i.B["location_state"]),
        A_j_loc=jnp.array(model_i.A["location_obs"]),
        C_j_loc=jnp.array(model_i.C["location_obs"]),
        A_j_edge=jnp.array(model_i.A["edge_obs"]),
        C_j_edge=jnp.array(model_i.C["edge_obs"]),
        A_j_cell_collision=jnp.array(model_i.A["cell_collision_obs"]),
        C_j_cell_collision=jnp.array(model_i.C["cell_collision_obs"]),
        A_j_edge_collision=jnp.array(model_i.A["edge_collision_obs"]),
        C_j_edge_collision=jnp.array(model_i.C["edge_collision_obs"]),
        actions_j=actions, epistemic_scale=0.1, alpha_other=alpha_other_j, eps=1e-16,
    )

    print(f"j chose {ACTION_NAMES[action_j]}, predicted i would do {ACTION_NAMES[int(i_predicted_for_actual_j)]}")
    print(f"i actually chose: {ACTION_NAMES[action_i]}")

    if int(i_predicted_for_actual_j) == action_i:
        print("[OK] ToM CORRECT: j's prediction matches i's actual choice")
    else:
        print("[FAIL] ToM MISMATCH: j predicted wrong!")

    return {
        "action_i": action_i,
        "action_j": action_j,
        "i_predicted_j": int(j_predicted_for_actual_i),
        "j_predicted_i": int(i_predicted_for_actual_j),
    }


def test_alpha_knowledge_effect(layout_name, alpha_i, alpha_j):
    """
    Test whether knowing the other's alpha helps prediction accuracy.

    Key insight: If i knows j is prosocial (alpha_j=1), i can predict
    j will consider i's welfare and potentially yield.
    """
    print(f"\n{'='*60}")
    print(f"ALPHA KNOWLEDGE TEST: {layout_name}")
    print(f"  TRUE alphas: i={alpha_i}, j={alpha_j}")
    print(f"{'='*60}\n")

    layout = get_layout(layout_name)

    goal_i = layout.goal_positions[0]
    goal_j = layout.goal_positions[1]
    start_i = layout.start_positions[0]
    start_j = layout.start_positions[1]

    model_i = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_i[0], goal_y=goal_i[1],
        start_pos=start_i, num_empathy_levels=3
    )
    model_j = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_j[0], goal_y=goal_j[1],
        start_pos=start_j, num_empathy_levels=3
    )

    agent_i = LavaAgent(model=model_i)
    agent_j = LavaAgent(model=model_j)

    qs_i = jnp.array(model_i.D["location_state"])
    qs_j = jnp.array(model_j.D["location_state"])

    print(f"Positions: i@{start_i}, j@{start_j}")
    print(f"Goals: i->{goal_i}, j->{goal_j}")

    # Test 1: Correct alpha beliefs
    print(f"\n--- CORRECT ALPHA BELIEFS ---")
    print(f"i believes j has alpha={alpha_j} (CORRECT)")
    print(f"j believes i has alpha={alpha_i} (CORRECT)")

    planner_i_correct = EmpathicLavaPlanner(
        agent_i, agent_j, alpha=alpha_i, alpha_other=alpha_j
    )
    planner_j_correct = EmpathicLavaPlanner(
        agent_j, agent_i, alpha=alpha_j, alpha_other=alpha_i
    )

    G_i, G_j_sim_i, G_social_i, _, action_i_correct = planner_i_correct.plan(qs_i, qs_j)
    G_j, G_i_sim_j, G_social_j, _, action_j_correct = planner_j_correct.plan(qs_j, qs_i)

    print(f"i chooses: {ACTION_NAMES[action_i_correct]}")
    print(f"j chooses: {ACTION_NAMES[action_j_correct]}")

    # Check collision
    def get_next_pos(pos, action, layout):
        dx = [0, 0, -1, 1, 0]
        dy = [-1, 1, 0, 0, 0]
        new_x = pos[0] + dx[action]
        new_y = pos[1] + dy[action]
        if 0 <= new_x < layout.width and 0 <= new_y < layout.height:
            new_pos = (new_x, new_y)
            if new_pos in layout.safe_cells:
                return new_pos
        return pos

    next_i = get_next_pos(start_i, action_i_correct, layout)
    next_j = get_next_pos(start_j, action_j_correct, layout)

    collision_correct = (next_i == next_j) or (next_i == start_j and next_j == start_i)
    if collision_correct:
        print(f"  -> COLLISION! i:{start_i}->{next_i}, j:{start_j}->{next_j}")
    else:
        print(f"  -> No collision. i:{start_i}->{next_i}, j:{start_j}->{next_j}")

    # Test 2: Wrong alpha beliefs (both think other is selfish)
    print(f"\n--- WRONG ALPHA BELIEFS (both think other is selfish) ---")
    print(f"i believes j has alpha=0.0 (actual={alpha_j})")
    print(f"j believes i has alpha=0.0 (actual={alpha_i})")

    planner_i_wrong = EmpathicLavaPlanner(
        agent_i, agent_j, alpha=alpha_i, alpha_other=0.0
    )
    planner_j_wrong = EmpathicLavaPlanner(
        agent_j, agent_i, alpha=alpha_j, alpha_other=0.0
    )

    _, _, _, _, action_i_wrong = planner_i_wrong.plan(qs_i, qs_j)
    _, _, _, _, action_j_wrong = planner_j_wrong.plan(qs_j, qs_i)

    print(f"i chooses: {ACTION_NAMES[action_i_wrong]}")
    print(f"j chooses: {ACTION_NAMES[action_j_wrong]}")

    next_i_wrong = get_next_pos(start_i, action_i_wrong, layout)
    next_j_wrong = get_next_pos(start_j, action_j_wrong, layout)

    collision_wrong = (next_i_wrong == next_j_wrong) or (next_i_wrong == start_j and next_j_wrong == start_i)
    if collision_wrong:
        print(f"  -> COLLISION! i:{start_i}->{next_i_wrong}, j:{start_j}->{next_j_wrong}")
    else:
        print(f"  -> No collision. i:{start_i}->{next_i_wrong}, j:{start_j}->{next_j_wrong}")

    # Test 3: Asymmetric - what if only prosocial agent knows the truth?
    if alpha_i != alpha_j:
        print(f"\n--- ASYMMETRIC KNOWLEDGE ---")
        # Prosocial agent knows truth, selfish doesn't
        if alpha_i > alpha_j:
            # i is prosocial, j is selfish
            print(f"i (prosocial) believes j has alpha={alpha_j} (CORRECT)")
            print(f"j (selfish) believes i has alpha=0.0 (WRONG, actual={alpha_i})")
            planner_i_asym = EmpathicLavaPlanner(agent_i, agent_j, alpha=alpha_i, alpha_other=alpha_j)
            planner_j_asym = EmpathicLavaPlanner(agent_j, agent_i, alpha=alpha_j, alpha_other=0.0)
        else:
            # j is prosocial, i is selfish
            print(f"i (selfish) believes j has alpha=0.0 (WRONG, actual={alpha_j})")
            print(f"j (prosocial) believes i has alpha={alpha_i} (CORRECT)")
            planner_i_asym = EmpathicLavaPlanner(agent_i, agent_j, alpha=alpha_i, alpha_other=0.0)
            planner_j_asym = EmpathicLavaPlanner(agent_j, agent_i, alpha=alpha_j, alpha_other=alpha_i)

        _, _, _, _, action_i_asym = planner_i_asym.plan(qs_i, qs_j)
        _, _, _, _, action_j_asym = planner_j_asym.plan(qs_j, qs_i)

        print(f"i chooses: {ACTION_NAMES[action_i_asym]}")
        print(f"j chooses: {ACTION_NAMES[action_j_asym]}")

        next_i_asym = get_next_pos(start_i, action_i_asym, layout)
        next_j_asym = get_next_pos(start_j, action_j_asym, layout)

        collision_asym = (next_i_asym == next_j_asym) or (next_i_asym == start_j and next_j_asym == start_i)
        if collision_asym:
            print(f"  -> COLLISION! i:{start_i}->{next_i_asym}, j:{start_j}->{next_j_asym}")
        else:
            print(f"  -> No collision. i:{start_i}->{next_i_asym}, j:{start_j}->{next_j_asym}")

    return {
        "correct_beliefs": {"i": action_i_correct, "j": action_j_correct, "collision": collision_correct},
        "wrong_beliefs": {"i": action_i_wrong, "j": action_j_wrong, "collision": collision_wrong},
    }


def test_asymmetric_tom(layout_name, alpha_tom=0.5):
    """
    Test if ToM works in ASYMMETRIC setup: ToM agent vs non-ToM agent.

    Hypothesis: When one agent has ToM and the other doesn't, the ToM agent
    can accurately predict the non-ToM agent's behavior (since non-ToM agent
    just minimizes own EFE without recursive reasoning).

    Setup:
    - Agent i (ToM): alpha > 0, correctly believes j is selfish (alpha_other=0)
    - Agent j (non-ToM): alpha = 0, just minimizes own EFE

    This avoids the recursivity problem where both agents predict "the other will yield".
    """
    print(f"\n{'='*60}")
    print(f"ASYMMETRIC ToM TEST: {layout_name}")
    print(f"  Agent i (ToM): alpha={alpha_tom}, alpha_other=0 (knows j is selfish)")
    print(f"  Agent j (non-ToM): alpha=0 (selfish, no ToM reasoning)")
    print(f"{'='*60}\n")

    layout = get_layout(layout_name)

    goal_i = layout.goal_positions[0]
    goal_j = layout.goal_positions[1]
    start_i = layout.start_positions[0]
    start_j = layout.start_positions[1]

    model_i = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_i[0], goal_y=goal_i[1],
        start_pos=start_i, num_empathy_levels=3
    )
    model_j = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_j[0], goal_y=goal_j[1],
        start_pos=start_j, num_empathy_levels=3
    )

    agent_i = LavaAgent(model=model_i)
    agent_j = LavaAgent(model=model_j)

    qs_i = jnp.array(model_i.D["location_state"])
    qs_j = jnp.array(model_j.D["location_state"])

    print(f"Positions: i@{start_i}, j@{start_j}")
    print(f"Goals: i->{goal_i}, j->{goal_j}")

    # --- ASYMMETRIC SETUP ---
    # Agent i (ToM): Has empathy, but correctly believes j is selfish
    planner_i_tom = EmpathicLavaPlanner(
        agent_i, agent_j, alpha=alpha_tom, alpha_other=0.0  # Knows j is selfish
    )

    # Agent j (non-ToM): Just minimizes own EFE, no empathy
    planner_j_selfish = EmpathicLavaPlanner(
        agent_j, agent_i, alpha=0.0, alpha_other=0.0  # Selfish, doesn't model i
    )

    # Plan independently
    G_i, G_j_sim, G_social_i, _, action_i = planner_i_tom.plan(qs_i, qs_j)
    G_j, G_i_sim, G_social_j, _, action_j = planner_j_selfish.plan(qs_j, qs_i)

    print(f"\n--- ASYMMETRIC PLANNING ---")
    print(f"Agent i (ToM, alpha={alpha_tom}):")
    for a in range(5):
        print(f"  {ACTION_NAMES[a]:5s}: G_i={G_i[a]:7.2f}, G_j_sim={G_j_sim[a]:7.2f}, G_social={G_social_i[a]:7.2f}")
    print(f"  -> i chooses: {ACTION_NAMES[action_i]}")

    print(f"\nAgent j (non-ToM, alpha=0):")
    for a in range(5):
        print(f"  {ACTION_NAMES[a]:5s}: G_j={G_j[a]:7.2f}")
    print(f"  -> j chooses: {ACTION_NAMES[action_j]}")

    # --- ToM PREDICTION CHECK ---
    # What does i's ToM predict j will do in response to i's chosen action?
    B_j = jnp.array(model_j.B["location_state"])
    B_i = jnp.array(model_i.B["location_state"])
    actions = jnp.arange(5)

    qs_i_next = propagate_belief_jax(qs_i, B_i, action_i, qs_j, eps=1e-16)
    _, j_predicted_response = compute_j_best_response_jax(
        qs_j=qs_j, qs_i=qs_i, qs_i_next=qs_i_next, action_i=action_i,
        B_j=B_j,
        A_j_loc=jnp.array(model_j.A["location_obs"]),
        C_j_loc=jnp.array(model_j.C["location_obs"]),
        A_j_edge=jnp.array(model_j.A["edge_obs"]),
        C_j_edge=jnp.array(model_j.C["edge_obs"]),
        A_j_cell_collision=jnp.array(model_j.A["cell_collision_obs"]),
        C_j_cell_collision=jnp.array(model_j.C["cell_collision_obs"]),
        A_j_edge_collision=jnp.array(model_j.A["edge_collision_obs"]),
        C_j_edge_collision=jnp.array(model_j.C["edge_collision_obs"]),
        actions_j=actions, epistemic_scale=0.1, alpha_other=0.0,  # Model j as selfish
        eps=1e-16,
    )

    print(f"\n--- ToM PREDICTION CHECK ---")
    print(f"i predicts: given i does {ACTION_NAMES[action_i]}, j will do {ACTION_NAMES[int(j_predicted_response)]}")
    print(f"j actually chose: {ACTION_NAMES[action_j]}")

    # The key insight: In asymmetric setup, i's ToM should correctly predict j
    # because j is just doing argmin(G_j) without any recursive reasoning about i
    tom_accurate = (int(j_predicted_response) == action_j)

    if tom_accurate:
        print(f"[OK] ToM ACCURATE: i correctly predicted j's action!")
    else:
        print(f"[FAIL] ToM INACCURATE: i predicted {ACTION_NAMES[int(j_predicted_response)]}, j did {ACTION_NAMES[action_j]}")

    # --- COLLISION CHECK ---
    def get_next_pos(pos, action):
        dx = [0, 0, -1, 1, 0]  # UP, DOWN, LEFT, RIGHT, STAY
        dy = [-1, 1, 0, 0, 0]
        new_x = pos[0] + dx[action]
        new_y = pos[1] + dy[action]
        if 0 <= new_x < layout.width and 0 <= new_y < layout.height:
            new_pos = (new_x, new_y)
            if new_pos in layout.safe_cells:
                return new_pos
        return pos

    next_i = get_next_pos(start_i, action_i)
    next_j = get_next_pos(start_j, action_j)

    cell_collision = (next_i == next_j)
    edge_collision = (next_i == start_j and next_j == start_i)
    collision = cell_collision or edge_collision

    print(f"\n--- COLLISION CHECK ---")
    print(f"i: {start_i} -> {next_i}")
    print(f"j: {start_j} -> {next_j}")

    if collision:
        if cell_collision:
            print(f"[COLLISION] Both agents end up at {next_i}!")
        if edge_collision:
            print(f"[COLLISION] Agents try to swap positions!")
    else:
        print(f"[OK] No collision")

    # --- COMPARE WITH SYMMETRIC (BOTH ToM) ---
    print(f"\n--- COMPARISON WITH SYMMETRIC SETUP ---")

    planner_i_sym = EmpathicLavaPlanner(
        agent_i, agent_j, alpha=alpha_tom, alpha_other=alpha_tom
    )
    planner_j_sym = EmpathicLavaPlanner(
        agent_j, agent_i, alpha=alpha_tom, alpha_other=alpha_tom
    )

    _, _, _, _, action_i_sym = planner_i_sym.plan(qs_i, qs_j)
    _, _, _, _, action_j_sym = planner_j_sym.plan(qs_j, qs_i)

    next_i_sym = get_next_pos(start_i, action_i_sym)
    next_j_sym = get_next_pos(start_j, action_j_sym)
    collision_sym = (next_i_sym == next_j_sym) or (next_i_sym == start_j and next_j_sym == start_i)

    print(f"Symmetric (both alpha={alpha_tom}):")
    print(f"  i chooses: {ACTION_NAMES[action_i_sym]}, j chooses: {ACTION_NAMES[action_j_sym]}")
    if collision_sym:
        print(f"  [COLLISION] Symmetric ToM leads to conflict!")
    else:
        print(f"  [OK] No collision in symmetric case")

    print(f"\nAsymmetric (i has ToM, j is selfish):")
    print(f"  i chooses: {ACTION_NAMES[action_i]}, j chooses: {ACTION_NAMES[action_j]}")
    if collision:
        print(f"  [COLLISION] Asymmetric still leads to conflict")
    else:
        print(f"  [OK] No collision - asymmetry resolves coordination!")

    return {
        "layout": layout_name,
        "asymmetric": {
            "action_i": action_i,
            "action_j": action_j,
            "tom_accurate": tom_accurate,
            "collision": collision,
        },
        "symmetric": {
            "action_i": action_i_sym,
            "action_j": action_j_sym,
            "collision": collision_sym,
        }
    }


def test_adjacent_conflict_alpha_asymmetry():
    """
    Test alpha asymmetry in ADJACENT CONFLICT scenario.

    Key question: Does having asymmetric alphas (one empathic, one selfish)
    produce better outcomes than symmetric alphas?

    Setup:
    - Agent i at (2,1), goal (5,1) -> wants RIGHT
    - Agent j at (3,1), goal (0,1) -> wants LEFT
    - They are adjacent and facing each other!

    Test configurations:
    1. Both selfish (alpha_i=0, alpha_j=0)
    2. Both empathic (alpha_i=1, alpha_j=1) - symmetric
    3. Asymmetric: i empathic, j selfish (alpha_i=1, alpha_j=0)
    4. Asymmetric: i selfish, j empathic (alpha_i=0, alpha_j=1)
    """
    print(f"\n{'='*70}")
    print(f"ADJACENT CONFLICT: ALPHA ASYMMETRY TEST")
    print(f"  Agent i at (2,1), goal (5,1) -> wants RIGHT")
    print(f"  Agent j at (3,1), goal (0,1) -> wants LEFT")
    print(f"  CONFLICT: Both agents want to pass through each other!")
    print(f"{'='*70}\n")

    layout = get_layout("narrow")  # 6x3 corridor

    pos_i = (2, 1)
    pos_j = (3, 1)
    goal_i = (5, 1)
    goal_j = (0, 1)

    model_i = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_i[0], goal_y=goal_i[1],
        start_pos=pos_i, num_empathy_levels=3
    )
    model_j = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells, goal_x=goal_j[0], goal_y=goal_j[1],
        start_pos=pos_j, num_empathy_levels=3
    )

    agent_i = LavaAgent(model=model_i)
    agent_j = LavaAgent(model=model_j)

    # Point beliefs at current positions
    qs_i = np.zeros(layout.width * layout.height)
    qs_i[pos_i[1] * layout.width + pos_i[0]] = 1.0
    qs_j = np.zeros(layout.width * layout.height)
    qs_j[pos_j[1] * layout.width + pos_j[0]] = 1.0

    def get_next_pos(pos, action):
        dx = [0, 0, -1, 1, 0]
        dy = [-1, 1, 0, 0, 0]
        new_x = pos[0] + dx[action]
        new_y = pos[1] + dy[action]
        if 0 <= new_x < layout.width and 0 <= new_y < layout.height:
            new_pos = (new_x, new_y)
            if new_pos in layout.safe_cells:
                return new_pos
        return pos

    def check_collision(a_i, a_j):
        next_i = get_next_pos(pos_i, a_i)
        next_j = get_next_pos(pos_j, a_j)
        cell_coll = (next_i == next_j)
        edge_coll = (next_i == pos_j and next_j == pos_i)
        return cell_coll or edge_coll, next_i, next_j

    results = []

    # Test configurations: (alpha_i, alpha_j, alpha_other_i, alpha_other_j, label)
    # alpha_other = what each agent BELIEVES the other's alpha is
    configs = [
        (0.0, 0.0, 0.0, 0.0, "Both selfish"),
        (1.0, 1.0, 1.0, 1.0, "Both empathic (symmetric)"),
        (1.0, 0.0, 0.0, 1.0, "i=empathic, j=selfish (CORRECT beliefs)"),
        (1.0, 0.0, 0.0, 0.0, "i=empathic, j=selfish (j doesn't know i is empathic)"),
        (0.0, 1.0, 1.0, 0.0, "i=selfish, j=empathic (CORRECT beliefs)"),
    ]

    print(f"{'Configuration':<50} {'i':<8} {'j':<8} {'Collision':<12}")
    print("-"*80)

    for alpha_i, alpha_j, alpha_other_i, alpha_other_j, label in configs:
        planner_i = EmpathicLavaPlanner(
            agent_i, agent_j, alpha=alpha_i, alpha_other=alpha_other_i
        )
        planner_j = EmpathicLavaPlanner(
            agent_j, agent_i, alpha=alpha_j, alpha_other=alpha_other_j
        )

        _, _, _, _, action_i = planner_i.plan(qs_i, qs_j)
        _, _, _, _, action_j = planner_j.plan(qs_j, qs_i)

        collision, next_i, next_j = check_collision(action_i, action_j)

        coll_str = "COLLISION" if collision else "OK"
        print(f"{label:<50} {ACTION_NAMES[action_i]:<8} {ACTION_NAMES[action_j]:<8} {coll_str:<12}")

        results.append({
            "label": label,
            "alpha_i": alpha_i,
            "alpha_j": alpha_j,
            "action_i": action_i,
            "action_j": action_j,
            "collision": collision,
        })

    # Analysis
    print(f"\n--- ANALYSIS ---")
    both_selfish = results[0]
    both_empathic = results[1]
    asym_i_emp = results[2]

    if both_selfish["collision"] and not asym_i_emp["collision"]:
        print("[SUCCESS] Alpha asymmetry resolves collision!")
    elif both_empathic["collision"] and not asym_i_emp["collision"]:
        print("[SUCCESS] Asymmetric alpha better than symmetric empathy!")
    elif asym_i_emp["collision"]:
        print("[FAIL] Asymmetric alpha still causes collision")
    else:
        print("[OK] No collision in asymmetric case")

    return results


def run_asymmetric_hypothesis_test():
    """
    Run comprehensive asymmetric ToM hypothesis test.

    Tests the hypothesis that asymmetric alpha values produce better coordination:
    - One agent empathic (alpha > 0) can anticipate and yield
    - One agent selfish (alpha = 0) acts predictably

    This avoids the recursivity problem of symmetric empathy.
    """
    print("\n" + "="*80)
    print("ASYMMETRIC ALPHA HYPOTHESIS TEST")
    print("="*80)
    print("\nHypothesis: Asymmetric alpha (one empathic, one selfish) should")
    print("produce better coordination than symmetric alpha (both same).")
    print("\nRationale: The empathic agent predicts the selfish agent will NOT yield,")
    print("so the empathic agent yields -> avoids collision.")

    # Critical test: adjacent conflict
    test_adjacent_conflict_alpha_asymmetry()

    # Standard layouts
    print("\n" + "="*80)
    print("STANDARD LAYOUT TESTS (default starting positions)")
    print("="*80)

    layouts_to_test = ["crossed_goals", "narrow", "t_junction", "symmetric_bottleneck"]
    alpha_values = [0.5, 1.0]

    results = []

    for layout in layouts_to_test:
        for alpha in alpha_values:
            print(f"\n{'='*80}")
            result = test_asymmetric_tom(layout, alpha_tom=alpha)
            results.append(result)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n{'Layout':<25} {'Alpha':<8} {'Asym ToM':<12} {'Asym Coll':<12} {'Sym Coll':<12} {'Better?':<10}")
    print("-"*80)

    for idx, r in enumerate(results):
        layout = r["layout"]
        alpha = "0.5" if idx % 2 == 0 else "1.0"

        tom_ok = "OK" if r["asymmetric"]["tom_accurate"] else "FAIL"
        asym_coll = "COLLISION" if r["asymmetric"]["collision"] else "OK"
        sym_coll = "COLLISION" if r["symmetric"]["collision"] else "OK"

        if not r["asymmetric"]["collision"] and r["symmetric"]["collision"]:
            better = "YES!"
        elif r["asymmetric"]["collision"] == r["symmetric"]["collision"] and r["asymmetric"]["tom_accurate"]:
            better = "SAME"
        elif r["asymmetric"]["collision"] and not r["symmetric"]["collision"]:
            better = "NO"
        else:
            better = "SAME"

        print(f"{layout:<25} {alpha:<8} {tom_ok:<12} {asym_coll:<12} {sym_coll:<12} {better:<10}")

    return results


if __name__ == "__main__":
    import sys

    # Check for command line args to run specific tests
    run_all = len(sys.argv) == 1 or "all" in sys.argv
    run_simultaneous = "sim" in sys.argv or "simultaneous" in sys.argv
    run_alpha = "alpha" in sys.argv
    run_asymmetric = "asym" in sys.argv or "asymmetric" in sys.argv

    # If asymmetric flag is set, just run that test
    if run_asymmetric:
        run_asymmetric_hypothesis_test()
        sys.exit(0)

    if run_all or "basic" in sys.argv:
        # Test symmetric cases
        print("\n" + "="*80)
        print("SYMMETRIC EMPATHY TESTS")
        print("="*80)

        diagnose_tom("narrow", alpha_i=0.0, alpha_j=0.0)  # Both selfish
        diagnose_tom("narrow", alpha_i=0.5, alpha_j=0.5)  # Both moderate
        diagnose_tom("narrow", alpha_i=1.0, alpha_j=1.0)  # Both prosocial

        print("\n" + "="*80)
        print("ASYMMETRIC EMPATHY TESTS")
        print("="*80)

        diagnose_tom("narrow", alpha_i=0.0, alpha_j=1.0)  # i selfish, j prosocial
        diagnose_tom("narrow", alpha_i=1.0, alpha_j=0.0)  # i prosocial, j selfish

        print("\n" + "="*80)
        print("CROSSED GOALS TEST")
        print("="*80)

        diagnose_tom("crossed_goals", alpha_i=0.0, alpha_j=0.0)
        diagnose_tom("crossed_goals", alpha_i=1.0, alpha_j=1.0)

    if run_all or "independent" in sys.argv:
        print("\n" + "="*80)
        print("INDEPENDENT ToM TESTS (Simultaneous Move)")
        print("="*80)

        test_independent_tom("narrow", alpha_i=0.0, alpha_j=0.0)
        test_independent_tom("narrow", alpha_i=1.0, alpha_j=1.0)
        test_independent_tom("crossed_goals", alpha_i=0.0, alpha_j=0.0)
        test_independent_tom("crossed_goals", alpha_i=1.0, alpha_j=1.0)

    if run_simultaneous:
        print("\n" + "="*80)
        print("SIMULTANEOUS ToM FULL PLANNING TESTS")
        print("="*80)
        print("Testing if response-based ToM matches independent actions")
        print("(For simultaneous play, predictions should match INDEPENDENT optimal)")

        print("\n" + "-"*80)
        print("NARROW CORRIDOR TESTS")
        print("-"*80)
        test_simultaneous_tom_full_planning("narrow", alpha_i=0.0, alpha_j=0.0)
        test_simultaneous_tom_full_planning("narrow", alpha_i=1.0, alpha_j=1.0)

        print("\n" + "-"*80)
        print("CROSSED GOALS TESTS")
        print("-"*80)
        test_simultaneous_tom_full_planning("crossed_goals", alpha_i=0.0, alpha_j=0.0)
        test_simultaneous_tom_full_planning("crossed_goals", alpha_i=1.0, alpha_j=1.0)

        print("\n" + "-"*80)
        print("T-JUNCTION TESTS")
        print("-"*80)
        test_simultaneous_tom_full_planning("t_junction", alpha_i=0.0, alpha_j=0.0)
        test_simultaneous_tom_full_planning("t_junction", alpha_i=1.0, alpha_j=1.0)

        print("\n" + "-"*80)
        print("SYMMETRIC BOTTLENECK TESTS")
        print("-"*80)
        test_simultaneous_tom_full_planning("symmetric_bottleneck", alpha_i=0.0, alpha_j=0.0)
        test_simultaneous_tom_full_planning("symmetric_bottleneck", alpha_i=1.0, alpha_j=1.0)

    if run_alpha:
        print("\n" + "="*80)
        print("ALPHA KNOWLEDGE TESTS - ADJACENT CONFLICT")
        print("="*80)
        print("Testing asymmetric alpha when agents are ADJACENT in narrow corridor")
        print("Agent i at (2,1), Agent j at (3,1) - must coordinate to pass")

        layout = get_layout("narrow")

        # Place agents adjacent in middle of corridor
        pos_i = (2, 1)
        pos_j = (3, 1)
        goal_i = (5, 1)  # i wants to go right
        goal_j = (0, 1)  # j wants to go left

        model_i = LavaModel(
            width=layout.width, height=layout.height,
            safe_cells=layout.safe_cells, goal_x=goal_i[0], goal_y=goal_i[1],
            start_pos=pos_i, num_empathy_levels=3
        )
        model_j = LavaModel(
            width=layout.width, height=layout.height,
            safe_cells=layout.safe_cells, goal_x=goal_j[0], goal_y=goal_j[1],
            start_pos=pos_j, num_empathy_levels=3
        )

        agent_i = LavaAgent(model=model_i)
        agent_j = LavaAgent(model=model_j)

        # Beliefs at current positions
        qs_i = np.zeros(layout.width * layout.height)
        qs_i[pos_i[1] * layout.width + pos_i[0]] = 1.0
        qs_j = np.zeros(layout.width * layout.height)
        qs_j[pos_j[1] * layout.width + pos_j[0]] = 1.0

        def get_next_pos(pos, action):
            dx = [0, 0, -1, 1, 0]
            dy = [-1, 1, 0, 0, 0]
            new_x = pos[0] + dx[action]
            new_y = pos[1] + dy[action]
            if 0 <= new_x < layout.width and 0 <= new_y < layout.height:
                new_pos = (new_x, new_y)
                if new_pos in layout.safe_cells:
                    return new_pos
            return pos

        # Test different alpha combinations
        test_cases = [
            (0.0, 0.0, "Both SELFISH"),
            (1.0, 1.0, "Both PROSOCIAL"),
            (0.0, 1.0, "i=SELFISH, j=PROSOCIAL"),
            (1.0, 0.0, "i=PROSOCIAL, j=SELFISH"),
        ]

        for alpha_i, alpha_j, label in test_cases:
            print(f"\n--- {label} ---")
            print(f"alpha_i={alpha_i}, alpha_j={alpha_j}")

            # Each agent knows the other's true alpha
            planner_i = EmpathicLavaPlanner(agent_i, agent_j, alpha=alpha_i, alpha_other=alpha_j)
            planner_j = EmpathicLavaPlanner(agent_j, agent_i, alpha=alpha_j, alpha_other=alpha_i)

            G_i, G_j_sim, G_social_i, _, action_i = planner_i.plan(qs_i, qs_j)
            G_j, G_i_sim, G_social_j, _, action_j = planner_j.plan(qs_j, qs_i)

            print(f"i's social G: {[f'{ACTION_NAMES[a]}:{G_social_i[a]:.1f}' for a in range(5)]}")
            print(f"j's social G: {[f'{ACTION_NAMES[a]}:{G_social_j[a]:.1f}' for a in range(5)]}")
            print(f"i chooses: {ACTION_NAMES[action_i]}, j chooses: {ACTION_NAMES[action_j]}")

            next_i = get_next_pos(pos_i, action_i)
            next_j = get_next_pos(pos_j, action_j)

            # Check for collision types
            cell_collision = (next_i == next_j)
            edge_collision = (next_i == pos_j and next_j == pos_i)

            if cell_collision:
                print(f"  -> CELL COLLISION! Both end at {next_i}")
            elif edge_collision:
                print(f"  -> EDGE COLLISION! i:{pos_i}->{next_i}, j:{pos_j}->{next_j} (swap)")
            else:
                print(f"  -> OK: i:{pos_i}->{next_i}, j:{pos_j}->{next_j}")

            # Show what ToM predicts for key actions
            B_j = jnp.array(model_j.B["location_state"])
            B_i = jnp.array(model_i.B["location_state"])
            actions = jnp.arange(5)

            print(f"  ToM predictions:")
            # What does i predict j will do if i goes RIGHT?
            qs_i_right = propagate_belief_jax(jnp.array(qs_i), jnp.array(B_i), 3, jnp.array(qs_j), eps=1e-16)
            _, j_response_to_i_right = compute_j_best_response_jax(
                qs_j=jnp.array(qs_j), qs_i=jnp.array(qs_i), qs_i_next=qs_i_right, action_i=3,
                B_j=B_j,
                A_j_loc=jnp.array(model_j.A["location_obs"]),
                C_j_loc=jnp.array(model_j.C["location_obs"]),
                A_j_edge=jnp.array(model_j.A["edge_obs"]),
                C_j_edge=jnp.array(model_j.C["edge_obs"]),
                A_j_cell_collision=jnp.array(model_j.A["cell_collision_obs"]),
                C_j_cell_collision=jnp.array(model_j.C["cell_collision_obs"]),
                A_j_edge_collision=jnp.array(model_j.A["edge_collision_obs"]),
                C_j_edge_collision=jnp.array(model_j.C["edge_collision_obs"]),
                actions_j=actions, epistemic_scale=0.1, alpha_other=alpha_j, eps=1e-16,
            )
            print(f"    If i->RIGHT: i predicts j will {ACTION_NAMES[int(j_response_to_i_right)]}")

            # What does j predict i will do if j goes LEFT?
            qs_j_left = propagate_belief_jax(jnp.array(qs_j), B_j, 2, jnp.array(qs_i), eps=1e-16)
            _, i_response_to_j_left = compute_j_best_response_jax(
                qs_j=jnp.array(qs_i), qs_i=jnp.array(qs_j), qs_i_next=qs_j_left, action_i=2,
                B_j=jnp.array(B_i),
                A_j_loc=jnp.array(model_i.A["location_obs"]),
                C_j_loc=jnp.array(model_i.C["location_obs"]),
                A_j_edge=jnp.array(model_i.A["edge_obs"]),
                C_j_edge=jnp.array(model_i.C["edge_obs"]),
                A_j_cell_collision=jnp.array(model_i.A["cell_collision_obs"]),
                C_j_cell_collision=jnp.array(model_i.C["cell_collision_obs"]),
                A_j_edge_collision=jnp.array(model_i.A["edge_collision_obs"]),
                C_j_edge_collision=jnp.array(model_i.C["edge_collision_obs"]),
                actions_j=actions, epistemic_scale=0.1, alpha_other=alpha_i, eps=1e-16,
            )
            print(f"    If j->LEFT: j predicts i will {ACTION_NAMES[int(i_response_to_j_left)]}")
