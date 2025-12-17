"""
Test asymmetric empathy with corrected ToM for simultaneous play.

Setup:
- Agent i: alpha=1 (empathic), knows j has alpha=0
- Agent j: alpha=0 (selfish), knows i has alpha=1

Hypothesis: The empathic agent i will yield because i predicts j won't.

Now with MULTI-STEP planning (policy_length=3) to see if yielding now
helps the other agent in future steps.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from tom.envs.env_lava_variants import get_layout
from tom.models.model_lava import LavaModel, LavaAgent

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
POLICY_LENGTH = 3  # Multi-step horizon

# Tunable parameters for finding the sweet spot
# We want: selfish agent moves (collision < goal benefit for self)
#          empathic agent yields (2*collision > goal benefit for self)
COLLISION_PENALTY = -100.0  # Override model's default (-30)


def propagate_belief(model, qs, qs_other, action):
    """
    Propagate belief state given an action.
    Returns the next belief state after taking action.
    """
    B = np.array(model.B["location_state"])
    if B.ndim == 4:
        qs_next = np.zeros_like(qs)
        for s_other in range(len(qs_other)):
            qs_next += B[:, :, s_other, action] @ qs * qs_other[s_other]
    else:
        qs_next = B[:, :, action] @ qs
    return qs_next


def compute_G_independent(model, qs_self, qs_other, collision_penalty=None):
    """
    Compute EFE for each action (independent, checking collision with other's current position).
    """
    B = np.array(model.B["location_state"])
    A_loc = np.array(model.A["location_obs"])
    C_loc = np.array(model.C["location_obs"])
    A_edge = np.array(model.A["edge_obs"])
    C_edge = np.array(model.C["edge_obs"])
    A_cell_collision = np.array(model.A["cell_collision_obs"])

    # Allow override of collision penalty
    if collision_penalty is not None:
        C_cell_collision = np.array([0.0, collision_penalty])
    else:
        C_cell_collision = np.array(model.C["cell_collision_obs"])

    G = []
    for a in range(5):
        if B.ndim == 4:
            qs_next = np.zeros_like(qs_self)
            for s_other in range(len(qs_other)):
                qs_next += B[:, :, s_other, a] @ qs_self * qs_other[s_other]
        else:
            qs_next = B[:, :, a] @ qs_self

        obs_dist = A_loc @ qs_next
        loc_utility = float((obs_dist * C_loc).sum())

        edge_dist = A_edge[:, :, a] @ qs_next
        edge_utility = float((edge_dist * C_edge).sum())

        cell_obs_dist = np.einsum("oij,i,j->o", A_cell_collision, qs_next, qs_other)
        cell_coll_utility = float((cell_obs_dist * C_cell_collision).sum())

        G.append(-loc_utility - edge_utility - cell_coll_utility)

    return np.array(G)


def compute_G_multistep(model, qs_self, qs_other, horizon=POLICY_LENGTH, collision_penalty=None,
                        qs_other_predicted=None):
    """
    Compute EFE for each FIRST action over a multi-step horizon.

    Assumes the other agent takes their greedy action at each step.
    Returns G for first action only (averaged over future optimal actions).

    Parameters
    ----------
    qs_other_predicted : array, optional
        If provided, use this as the other's position for step 0 collision checking.
        This allows using ToM prediction: if we predict other will move, check collision
        against their PREDICTED position, not current position.
    """
    B = np.array(model.B["location_state"])
    A_loc = np.array(model.A["location_obs"])
    C_loc = np.array(model.C["location_obs"])
    A_edge = np.array(model.A["edge_obs"])
    C_edge = np.array(model.C["edge_obs"])
    A_cell_collision = np.array(model.A["cell_collision_obs"])

    # Allow override of collision penalty
    if collision_penalty is not None:
        C_cell_collision = np.array([0.0, collision_penalty])
    else:
        C_cell_collision = np.array(model.C["cell_collision_obs"])

    # For step 0, use predicted other position if provided
    qs_other_step0 = qs_other_predicted if qs_other_predicted is not None else qs_other

    def propagate(qs, qs_oth, action):
        if B.ndim == 4:
            qs_next = np.zeros_like(qs)
            for s_other in range(len(qs_oth)):
                qs_next += B[:, :, s_other, action] @ qs * qs_oth[s_other]
        else:
            qs_next = B[:, :, action] @ qs
        return qs_next

    def compute_step_G(qs, qs_oth, action):
        qs_next = propagate(qs, qs_oth, action)

        obs_dist = A_loc @ qs_next
        loc_utility = float((obs_dist * C_loc).sum())

        edge_dist = A_edge[:, :, action] @ qs_next
        edge_utility = float((edge_dist * C_edge).sum())

        cell_obs_dist = np.einsum("oij,i,j->o", A_cell_collision, qs_next, qs_oth)
        cell_coll_utility = float((cell_obs_dist * C_cell_collision).sum())

        return -loc_utility - edge_utility - cell_coll_utility, qs_next

    # For each first action, compute cumulative G over horizon
    G_first_actions = []
    for a0 in range(5):
        # Step 0: Use predicted other position for collision checking
        # This is the key fix: check collision against where other WILL BE,
        # not where they currently are
        G_step, qs_1 = compute_step_G(qs_self, qs_other_step0, a0)
        total_G = G_step

        # For steps 1..horizon-1, take greedy action
        # After step 0, other is at their predicted position
        qs_t = qs_1
        qs_other_t = qs_other_step0  # Other is now at predicted position

        for t in range(1, horizon):
            # Find best action at this step
            best_G = float('inf')
            best_qs = qs_t
            for a in range(5):
                G_a, qs_next = compute_step_G(qs_t, qs_other_t, a)
                if G_a < best_G:
                    best_G = G_a
                    best_qs = qs_next
            total_G += best_G
            qs_t = best_qs

        G_first_actions.append(total_G)

    return np.array(G_first_actions)


def predict_other_action(model_other, model_self, qs_other, qs_self, alpha_other,
                         use_multistep=True, collision_penalty=None):
    """
    Predict what the other agent will do.
    If alpha_other=0: other just minimizes their own G (selfish)
    If alpha_other>0: other uses empathy, so we simulate their empathic planning
    """
    if alpha_other == 0:
        # Other is selfish, just minimizes own G
        if use_multistep:
            G_other = compute_G_multistep(model_other, qs_other, qs_self,
                                          collision_penalty=collision_penalty)
        else:
            G_other = compute_G_independent(model_other, qs_other, qs_self,
                                            collision_penalty=collision_penalty)
        return int(np.argmin(G_other)), G_other
    else:
        # Other is empathic! They will predict WE are selfish (level-1 reasoning)
        # and plan empathically
        if use_multistep:
            G_self_independent = compute_G_multistep(model_self, qs_self, qs_other,
                                                     collision_penalty=collision_penalty)
        else:
            G_self_independent = compute_G_independent(model_self, qs_self, qs_other,
                                                       collision_penalty=collision_penalty)
        my_predicted_action = int(np.argmin(G_self_independent))

        # Now compute other's empathic G_social
        if use_multistep:
            G_other_self, G_other_social = compute_G_empathic_multistep(
                model_other, model_self, qs_other, qs_self, alpha_other,
                collision_penalty=collision_penalty
            )
        else:
            G_other_self, G_other_social = compute_G_empathic_given_action(
                model_other, model_self, qs_other, qs_self,
                my_predicted_action, alpha_other
            )
        return int(np.argmin(G_other_social)), G_other_social


def compute_G_empathic_given_action(model_self, model_other, qs_self, qs_other,
                                    action_other_predicted, alpha_self):
    """
    Same as compute_G_empathic but used for recursive prediction.
    """
    return compute_G_empathic(model_self, model_other, qs_self, qs_other,
                              action_other_predicted, alpha_self)


def compute_G_empathic(model_self, model_other, qs_self, qs_other,
                       action_other_predicted, alpha_self, enable_path_bonus=False):
    """
    Compute empathic EFE: G_social(a_i) = G_i(a_i) + alpha * G_j_outcome(a_i)

    If enable_path_bonus=True, also consider whether yielding enables the other's
    BEST action (not just their predicted action).
    """
    G_self = compute_G_independent(model_self, qs_self, qs_other)

    if alpha_self == 0:
        return G_self, G_self

    B_self = np.array(model_self.B["location_state"])
    B_other = np.array(model_other.B["location_state"])
    A_other_loc = np.array(model_other.A["location_obs"])
    C_other_loc = np.array(model_other.C["location_obs"])
    A_other_cell_collision = np.array(model_other.A["cell_collision_obs"])
    C_other_cell_collision = np.array(model_other.C["cell_collision_obs"])

    G_social = []
    for a_self in range(5):
        if B_self.ndim == 4:
            qs_self_next = np.zeros_like(qs_self)
            for s_other in range(len(qs_other)):
                qs_self_next += B_self[:, :, s_other, a_self] @ qs_self * qs_other[s_other]
        else:
            qs_self_next = B_self[:, :, a_self] @ qs_self

        if enable_path_bonus:
            # Compute what's the BEST other can do if we take action a_self
            # This is forward-looking: "if I move here, does other get a better path?"
            best_other_outcome = float('inf')
            for a_other_test in range(5):
                if B_other.ndim == 4:
                    qs_other_test = np.zeros_like(qs_other)
                    for s_self in range(len(qs_self_next)):
                        qs_other_test += B_other[:, :, s_self, a_other_test] @ qs_other * qs_self_next[s_self]
                else:
                    qs_other_test = B_other[:, :, a_other_test] @ qs_other

                obs_dist = A_other_loc @ qs_other_test
                loc_utility = float((obs_dist * C_other_loc).sum())
                cell_obs_dist = np.einsum("oij,i,j->o", A_other_cell_collision, qs_other_test, qs_self_next)
                cell_coll_utility = float((cell_obs_dist * C_other_cell_collision).sum())
                G_other_test = -loc_utility - cell_coll_utility
                if G_other_test < best_other_outcome:
                    best_other_outcome = G_other_test

            G_social_a = G_self[a_self] + alpha_self * best_other_outcome
        else:
            # Original: use predicted action
            if B_other.ndim == 4:
                qs_other_next = np.zeros_like(qs_other)
                for s_self in range(len(qs_self)):
                    qs_other_next += B_other[:, :, s_self, action_other_predicted] @ qs_other * qs_self[s_self]
            else:
                qs_other_next = B_other[:, :, action_other_predicted] @ qs_other

            obs_dist = A_other_loc @ qs_other_next
            loc_utility = float((obs_dist * C_other_loc).sum())

            cell_obs_dist = np.einsum("oij,i,j->o", A_other_cell_collision, qs_other_next, qs_self_next)
            cell_coll_utility = float((cell_obs_dist * C_other_cell_collision).sum())

            G_other_outcome = -loc_utility - cell_coll_utility
            G_social_a = G_self[a_self] + alpha_self * G_other_outcome

        G_social.append(G_social_a)

    return G_self, np.array(G_social)


def compute_G_empathic_multistep(model_self, model_other, qs_self, qs_other,
                                  alpha_self, horizon=POLICY_LENGTH, collision_penalty=None,
                                  qs_other_predicted=None):
    """
    Compute empathic EFE over multiple steps.

    For each first action a_self:
    - Compute self's cumulative G over horizon (taking greedy actions after step 1)
    - Compute other's cumulative G over horizon (taking greedy actions)
    - G_social = G_self + alpha * G_other

    Key insight: If self yields at t=0, other can move at t=1, improving other's multi-step G.

    qs_other_predicted: If provided, use this as the other's position after step 0.
    """
    # Use predicted other position for step 0 if provided
    qs_other_step0 = qs_other_predicted if qs_other_predicted is not None else qs_other
    B_self = np.array(model_self.B["location_state"])
    B_other = np.array(model_other.B["location_state"])
    A_self_loc = np.array(model_self.A["location_obs"])
    C_self_loc = np.array(model_self.C["location_obs"])
    A_other_loc = np.array(model_other.A["location_obs"])
    C_other_loc = np.array(model_other.C["location_obs"])
    A_self_edge = np.array(model_self.A["edge_obs"])
    C_self_edge = np.array(model_self.C["edge_obs"])
    A_self_cell_collision = np.array(model_self.A["cell_collision_obs"])
    A_other_cell_collision = np.array(model_other.A["cell_collision_obs"])

    # Allow override of collision penalty
    if collision_penalty is not None:
        C_self_cell_collision = np.array([0.0, collision_penalty])
        C_other_cell_collision = np.array([0.0, collision_penalty])
    else:
        C_self_cell_collision = np.array(model_self.C["cell_collision_obs"])
        C_other_cell_collision = np.array(model_other.C["cell_collision_obs"])

    def propagate_self(qs, qs_oth, action):
        if B_self.ndim == 4:
            qs_next = np.zeros_like(qs)
            for s_other in range(len(qs_oth)):
                qs_next += B_self[:, :, s_other, action] @ qs * qs_oth[s_other]
        else:
            qs_next = B_self[:, :, action] @ qs
        return qs_next

    def propagate_other(qs, qs_self_pos, action):
        if B_other.ndim == 4:
            qs_next = np.zeros_like(qs)
            for s_self in range(len(qs_self_pos)):
                qs_next += B_other[:, :, s_self, action] @ qs * qs_self_pos[s_self]
        else:
            qs_next = B_other[:, :, action] @ qs
        return qs_next

    def compute_self_step_G(qs, qs_oth, action):
        qs_next = propagate_self(qs, qs_oth, action)
        obs_dist = A_self_loc @ qs_next
        loc_utility = float((obs_dist * C_self_loc).sum())
        edge_dist = A_self_edge[:, :, action] @ qs_next
        edge_utility = float((edge_dist * C_self_edge).sum())
        cell_obs_dist = np.einsum("oij,i,j->o", A_self_cell_collision, qs_next, qs_oth)
        cell_coll_utility = float((cell_obs_dist * C_self_cell_collision).sum())
        return -loc_utility - edge_utility - cell_coll_utility, qs_next

    def compute_other_step_G(qs, qs_self_pos, action):
        qs_next = propagate_other(qs, qs_self_pos, action)
        obs_dist = A_other_loc @ qs_next
        loc_utility = float((obs_dist * C_other_loc).sum())
        cell_obs_dist = np.einsum("oij,i,j->o", A_other_cell_collision, qs_next, qs_self_pos)
        cell_coll_utility = float((cell_obs_dist * C_other_cell_collision).sum())
        return -loc_utility - cell_coll_utility, qs_next

    def best_action_G(compute_fn, qs, qs_other_pos):
        best_G = float('inf')
        best_qs = qs
        for a in range(5):
            G_a, qs_next = compute_fn(qs, qs_other_pos, a)
            if G_a < best_G:
                best_G = G_a
                best_qs = qs_next
        return best_G, best_qs

    G_self_all = []
    G_social_all = []

    for a0_self in range(5):
        # === Simulate joint trajectory for this first action ===

        # Step 0: Self takes action a0_self, other takes their PREDICTED action
        # KEY FIX: Use predicted other position for collision checking

        # Self's step 0: check collision against where other WILL BE
        G_self_0, qs_self_1 = compute_self_step_G(qs_self, qs_other_step0, a0_self)

        # Other's step 0: they take their predicted action (already computed)
        # We use qs_other_step0 as their next position
        # For other's G, check collision against self's CURRENT position (simultaneous)
        best_G_other_0, _ = best_action_G(
            lambda qs, qs_s, a: compute_other_step_G(qs, qs_s, a),
            qs_other, qs_self  # Other checks collision against self's current pos
        )
        qs_other_1 = qs_other_step0  # Other ends up at predicted position

        total_G_self = G_self_0
        total_G_other = best_G_other_0

        qs_self_t = qs_self_1
        qs_other_t = qs_other_1

        # Steps 1..horizon-1: Both take greedy actions with knowledge of new positions
        for t in range(1, horizon):
            # Self's greedy action given other's new position
            G_self_t, qs_self_next = best_action_G(
                compute_self_step_G, qs_self_t, qs_other_t
            )
            # Other's greedy action given self's new position
            G_other_t, qs_other_next = best_action_G(
                lambda qs, qs_s, a: compute_other_step_G(qs, qs_s, a),
                qs_other_t, qs_self_t
            )

            total_G_self += G_self_t
            total_G_other += G_other_t
            qs_self_t = qs_self_next
            qs_other_t = qs_other_next

        G_self_all.append(total_G_self)
        G_social_all.append(total_G_self + alpha_self * total_G_other)

    return np.array(G_self_all), np.array(G_social_all)


def plan_with_empathy(model_self, model_other, qs_self, qs_other,
                      alpha_self, alpha_other, use_multistep=True, collision_penalty=None):
    """Plan using corrected ToM with empathy.

    If use_multistep=True, uses multi-step planning (horizon=POLICY_LENGTH).
    collision_penalty: Override the model's default collision penalty.

    KEY FIX: After predicting other's action, we compute their PREDICTED position
    and use that for collision checking. This allows the agent to see that if
    the other will yield, the path is clear.
    """
    action_other_predicted, G_other = predict_other_action(
        model_other, model_self, qs_other, qs_self, alpha_other,
        collision_penalty=collision_penalty
    )

    # KEY FIX: Compute other's predicted position after taking their predicted action
    # This is where they WILL BE, not where they currently ARE
    qs_other_predicted = propagate_belief(model_other, qs_other, qs_self, action_other_predicted)

    if use_multistep:
        # For selfish agents (alpha=0), we still need to use ToM prediction for collision
        if alpha_self == 0:
            # Selfish: just minimize own G, but use predicted other position
            G_self = compute_G_multistep(
                model_self, qs_self, qs_other,
                collision_penalty=collision_penalty,
                qs_other_predicted=qs_other_predicted
            )
            G_social = G_self
        else:
            # Empathic: use full empathic computation with predicted other position
            G_self, G_social = compute_G_empathic_multistep(
                model_self, model_other, qs_self, qs_other, alpha_self,
                collision_penalty=collision_penalty,
                qs_other_predicted=qs_other_predicted
            )
    else:
        G_self, G_social = compute_G_empathic(
            model_self, model_other, qs_self, qs_other,
            action_other_predicted, alpha_self
        )
    action_self = int(np.argmin(G_social))

    return {
        "G_self": G_self,
        "G_social": G_social,
        "action": action_self,
        "predicted_other": action_other_predicted,
        "G_other": G_other,
    }


def test_adjacent_conflict():
    """Test asymmetric empathy in adjacent conflict scenario."""

    print("=" * 70)
    print(f"ASYMMETRIC EMPATHY WITH MULTI-STEP ToM (horizon={POLICY_LENGTH})")
    print(f"Collision penalty: {COLLISION_PENALTY}")
    print("=" * 70)

    layout = get_layout("narrow")
    pos_i, pos_j = (2, 1), (3, 1)
    goal_i, goal_j = (5, 1), (0, 1)

    print(f"\ni at {pos_i} -> goal {goal_i} (wants RIGHT)")
    print(f"j at {pos_j} -> goal {goal_j} (wants LEFT)")
    print("ADJACENT and FACING each other!")

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

    qs_i = np.zeros(layout.width * layout.height)
    qs_i[pos_i[1] * layout.width + pos_i[0]] = 1.0
    qs_j = np.zeros(layout.width * layout.height)
    qs_j[pos_j[1] * layout.width + pos_j[0]] = 1.0

    # CASE 1: Both selfish
    print("\n" + "-" * 70)
    print("CASE 1: Both selfish (alpha=0)")
    print("-" * 70)

    result_i_0 = plan_with_empathy(model_i, model_j, qs_i, qs_j, alpha_self=0.0, alpha_other=0.0,
                                    collision_penalty=COLLISION_PENALTY)
    result_j_0 = plan_with_empathy(model_j, model_i, qs_j, qs_i, alpha_self=0.0, alpha_other=0.0,
                                    collision_penalty=COLLISION_PENALTY)

    print(f"\ni predicts j: {ACTION_NAMES[result_i_0['predicted_other']]}")
    print("i G_self: " + ", ".join([f"{ACTION_NAMES[a]}:{result_i_0['G_self'][a]:.1f}" for a in range(5)]))
    print(f"i chooses: {ACTION_NAMES[result_i_0['action']]}")

    print(f"\nj predicts i: {ACTION_NAMES[result_j_0['predicted_other']]}")
    print("j G_self: " + ", ".join([f"{ACTION_NAMES[a]}:{result_j_0['G_self'][a]:.1f}" for a in range(5)]))
    print(f"j chooses: {ACTION_NAMES[result_j_0['action']]}")

    # CASE 2: Asymmetric - i empathic, j selfish
    print("\n" + "-" * 70)
    print("CASE 2: ASYMMETRIC EMPATHY")
    print("  i: alpha=1 (empathic), knows j has alpha=0 (selfish)")
    print("  j: alpha=0 (selfish), knows i has alpha=1 (empathic)")
    print("-" * 70)

    result_i_asym = plan_with_empathy(model_i, model_j, qs_i, qs_j, alpha_self=1.0, alpha_other=0.0,
                                       collision_penalty=COLLISION_PENALTY)
    result_j_asym = plan_with_empathy(model_j, model_i, qs_j, qs_i, alpha_self=0.0, alpha_other=1.0,
                                       collision_penalty=COLLISION_PENALTY)

    print(f"\ni (empathic, knows j is selfish):")
    print(f"  Predicts j will do: {ACTION_NAMES[result_i_asym['predicted_other']]}")
    print("  G_self:   " + ", ".join([f"{ACTION_NAMES[a]}:{result_i_asym['G_self'][a]:.1f}" for a in range(5)]))
    print("  G_social: " + ", ".join([f"{ACTION_NAMES[a]}:{result_i_asym['G_social'][a]:.1f}" for a in range(5)]))
    print(f"  i chooses: {ACTION_NAMES[result_i_asym['action']]}")

    print(f"\nj (selfish, knows i is empathic):")
    print(f"  Predicts i will do: {ACTION_NAMES[result_j_asym['predicted_other']]}")
    print("  G_self: " + ", ".join([f"{ACTION_NAMES[a]}:{result_j_asym['G_self'][a]:.1f}" for a in range(5)]))
    print(f"  j chooses: {ACTION_NAMES[result_j_asym['action']]}")

    # CASE 3: Reversed - i selfish, j empathic
    print("\n" + "-" * 70)
    print("CASE 3: REVERSED ASYMMETRIC")
    print("  i: alpha=0 (selfish), knows j has alpha=1 (empathic)")
    print("  j: alpha=1 (empathic), knows i has alpha=0 (selfish)")
    print("-" * 70)

    result_i_rev = plan_with_empathy(model_i, model_j, qs_i, qs_j, alpha_self=0.0, alpha_other=1.0,
                                      collision_penalty=COLLISION_PENALTY)
    result_j_rev = plan_with_empathy(model_j, model_i, qs_j, qs_i, alpha_self=1.0, alpha_other=0.0,
                                      collision_penalty=COLLISION_PENALTY)

    print(f"\ni (selfish, knows j is empathic):")
    print(f"  Predicts j will do: {ACTION_NAMES[result_i_rev['predicted_other']]}")
    print(f"  i chooses: {ACTION_NAMES[result_i_rev['action']]}")

    print(f"\nj (empathic, knows i is selfish):")
    print(f"  Predicts i will do: {ACTION_NAMES[result_j_rev['predicted_other']]}")
    print("  G_self:   " + ", ".join([f"{ACTION_NAMES[a]}:{result_j_rev['G_self'][a]:.1f}" for a in range(5)]))
    print("  G_social: " + ", ".join([f"{ACTION_NAMES[a]}:{result_j_rev['G_social'][a]:.1f}" for a in range(5)]))
    print(f"  j chooses: {ACTION_NAMES[result_j_rev['action']]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBoth selfish (alpha=0):        i={ACTION_NAMES[result_i_0['action']]}, j={ACTION_NAMES[result_j_0['action']]}")
    print(f"i empathic, j selfish:         i={ACTION_NAMES[result_i_asym['action']]}, j={ACTION_NAMES[result_j_asym['action']]}")
    print(f"i selfish, j empathic:         i={ACTION_NAMES[result_i_rev['action']]}, j={ACTION_NAMES[result_j_rev['action']]}")

    # Check ToM accuracy
    print("\n" + "=" * 70)
    print("ToM ACCURACY CHECK")
    print("=" * 70)

    print("\nCase 2 (i empathic, j selfish):")
    print(f"  i predicted j would do: {ACTION_NAMES[result_i_asym['predicted_other']]}")
    print(f"  j actually did: {ACTION_NAMES[result_j_asym['action']]}")
    i_pred_correct = result_i_asym['predicted_other'] == result_j_asym['action']
    print(f"  i's prediction: {'CORRECT' if i_pred_correct else 'WRONG'}")

    print(f"\n  j predicted i would do: {ACTION_NAMES[result_j_asym['predicted_other']]}")
    print(f"  i actually did: {ACTION_NAMES[result_i_asym['action']]}")
    j_pred_correct = result_j_asym['predicted_other'] == result_i_asym['action']
    print(f"  j's prediction: {'CORRECT' if j_pred_correct else 'WRONG'}")

    print("\nCase 3 (i selfish, j empathic):")
    print(f"  i predicted j would do: {ACTION_NAMES[result_i_rev['predicted_other']]}")
    print(f"  j actually did: {ACTION_NAMES[result_j_rev['action']]}")
    i_pred_correct_rev = result_i_rev['predicted_other'] == result_j_rev['action']
    print(f"  i's prediction: {'CORRECT' if i_pred_correct_rev else 'WRONG'}")

    print(f"\n  j predicted i would do: {ACTION_NAMES[result_j_rev['predicted_other']]}")
    print(f"  i actually did: {ACTION_NAMES[result_i_rev['action']]}")
    j_pred_correct_rev = result_j_rev['predicted_other'] == result_i_rev['action']
    print(f"  j's prediction: {'CORRECT' if j_pred_correct_rev else 'WRONG'}")

    # Check if symmetry is broken
    print("\n" + "=" * 70)
    print("SYMMETRY BREAKING CHECK")
    print("=" * 70)

    if result_i_asym['action'] != result_j_asym['action']:
        print("\nCase 2: SYMMETRY BROKEN! Agents choose different actions.")
        if result_i_asym['action'] == 2 and result_j_asym['action'] == 4:
            print("  i (empathic) yields LEFT, j (selfish) STAYs -> j can pass next turn!")
        elif result_i_asym['action'] == 4 and result_j_asym['action'] == 2:
            print("  i STAYs, j goes LEFT toward goal!")
    else:
        print("\nCase 2: Symmetry NOT broken - both choose same action.")

    if result_i_rev['action'] != result_j_rev['action']:
        print("\nCase 3: SYMMETRY BROKEN! Agents choose different actions.")
    else:
        print("\nCase 3: Symmetry NOT broken - both choose same action.")


if __name__ == "__main__":
    test_adjacent_conflict()
