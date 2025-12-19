"""
Test smart subgoal switching: use original C_loc when at subgoal.
"""
import numpy as np
import jax.numpy as jnp
import sys
sys.path.insert(0, ".")

from tom.envs.env_lava_variants import get_layout
from tom.models.model_lava import LavaModel, LavaAgent
from tom.envs.env_lava_v2 import LavaV2Env
from tom.planning.jax_hierarchical_planner import (
    HierarchicalEmpathicPlannerJax,
    JaxZonedLayout,
    create_subgoal_C_loc_jax,
    get_zone_from_belief,
    get_subgoal_state_jax,
    high_level_plan_jax,
    _compute_G_empathic_multistep_hierarchical_jit,
    _propagate_belief_tom_hierarchical,
    predict_other_action_recursive_hierarchical_jax,
    TOM_DEPTH, TOM_HORIZON,
)

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]

def create_modified_risk_reward_layout(width=8, height=None, goal_pos=None):
    """Create risk_reward layout with exit_points[1,2] = (0,0)"""
    height = 4
    num_states = width * height
    cell_to_zone = np.full(num_states, -1, dtype=np.int32)

    def pos_to_idx(x, y):
        return y * width + x

    for x in range(3, width):
        cell_to_zone[pos_to_idx(x, 1)] = 0
        cell_to_zone[pos_to_idx(x, 2)] = 0
    for x in range(width):
        cell_to_zone[pos_to_idx(x, 0)] = 1
    cell_to_zone[pos_to_idx(0, 1)] = 2
    cell_to_zone[pos_to_idx(0, 2)] = 2

    zone_adjacency = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
    zone_is_bottleneck = np.array([False, True, False])
    zone_centers = np.array([[(3 + width) / 2, 1.5], [width / 2, 0], [0, 1.5]], dtype=np.float32)

    exit_points = np.full((3, 3), -1, dtype=np.int32)
    exit_points[0, 1] = pos_to_idx(3, 0)
    exit_points[1, 0] = pos_to_idx(3, 1)
    exit_points[1, 2] = pos_to_idx(0, 0)  # Changed to (0,0)
    exit_points[2, 1] = pos_to_idx(0, 0)

    goal_zone = 0
    if goal_pos is not None:
        goal_idx = pos_to_idx(goal_pos[0], goal_pos[1])
        if 0 <= goal_idx < num_states:
            goal_zone = int(cell_to_zone[goal_idx])
            if goal_zone < 0:
                goal_zone = 0

    return JaxZonedLayout(
        width=width, height=height, num_zones=3, num_states=num_states,
        cell_to_zone=jnp.array(cell_to_zone),
        zone_adjacency=jnp.array(zone_adjacency),
        zone_is_bottleneck=jnp.array(zone_is_bottleneck),
        zone_centers=jnp.array(zone_centers),
        exit_points=jnp.array(exit_points),
        goal_zone=goal_zone,
    )


def smart_subgoal_C_loc(current_state, subgoal_state, original_C_loc, width):
    """Use subgoal C_loc normally, but switch to original when at subgoal."""
    if current_state == subgoal_state:
        return original_C_loc
    else:
        return create_subgoal_C_loc_jax(subgoal_state, original_C_loc, width)


class SmartHierarchicalPlanner(HierarchicalEmpathicPlannerJax):
    """Modified planner that switches to original C_loc when at subgoal."""

    def plan_with_debug(self, qs_i, qs_j):
        qs_i_jax = jnp.array(qs_i)
        qs_j_jax = jnp.array(qs_j)

        layout = self.planner_i.zoned_layout

        current_state_i = int(jnp.argmax(qs_i_jax))
        current_state_j = int(jnp.argmax(qs_j_jax))

        my_zone = get_zone_from_belief(qs_i_jax, layout.cell_to_zone)
        other_zone = get_zone_from_belief(qs_j_jax, layout.cell_to_zone)
        my_goal_zone = self.planner_i.zoned_layout.goal_zone
        other_goal_zone = self.planner_j.zoned_layout.goal_zone

        zone_action_i, G_zone_i, q_zone_i = high_level_plan_jax(
            my_zone, other_zone, my_goal_zone, other_goal_zone,
            self.alpha, layout.zone_adjacency, layout.zone_is_bottleneck, self.gamma
        )

        subgoal_i = int(get_subgoal_state_jax(
            my_zone, zone_action_i, my_goal_zone, self.goal_state_i,
            layout.exit_points, layout.zone_adjacency, layout.cell_to_zone
        ))

        zone_action_j, _, _ = high_level_plan_jax(
            other_zone, my_zone, other_goal_zone, my_goal_zone,
            self.alpha_other, layout.zone_adjacency, layout.zone_is_bottleneck, self.gamma
        )

        subgoal_j = int(get_subgoal_state_jax(
            other_zone, zone_action_j, other_goal_zone, self.goal_state_j,
            layout.exit_points, layout.zone_adjacency, layout.cell_to_zone
        ))

        # SMART C_loc: switch to original when at subgoal
        C_loc_i = smart_subgoal_C_loc(current_state_i, subgoal_i, self.C_loc_i, layout.width)
        C_loc_j = smart_subgoal_C_loc(current_state_j, subgoal_j, self.C_loc_j, layout.width)

        predicted_other, _ = predict_other_action_recursive_hierarchical_jax(
            qs_j, qs_i, self.alpha_other, self.alpha,
            self.B_j, self.B_i,
            self.A_loc_j, C_loc_j,
            self.A_edge, self.C_edge,
            self.A_cell_collision, self.C_cell_collision,
            self.A_loc_i, C_loc_i,
            self.A_edge, self.C_edge,
            self.A_cell_collision, self.C_cell_collision,
            depth=TOM_DEPTH, horizon=self.tom_horizon,
        )

        qs_other_predicted = _propagate_belief_tom_hierarchical(
            qs_j_jax, qs_i_jax, predicted_other, self.B_j
        )

        G_self_all, G_social_all = _compute_G_empathic_multistep_hierarchical_jit(
            qs_i_jax, qs_j_jax, self.alpha,
            self.B_i, self.B_j,
            self.A_loc_i, C_loc_i,
            self.A_edge, self.C_edge,
            self.A_cell_collision, self.C_cell_collision,
            self.A_loc_j, self.C_loc_j,
            self.A_cell_collision, self.C_cell_collision,
            qs_other_predicted, self.tom_horizon,
        )

        log_q_pi = -self.gamma * G_social_all
        log_q_pi = log_q_pi - log_q_pi.max()
        q_pi = jnp.exp(log_q_pi)
        q_pi = q_pi / q_pi.sum()

        best_action = int(jnp.argmin(G_social_all))

        return {
            "action": best_action,
            "my_zone": int(my_zone),
            "other_zone": int(other_zone),
            "zone_action": int(zone_action_i),
            "subgoal_state": subgoal_i,
            "G_low": np.array(G_social_all),
            "q_low": np.array(q_pi),
            "current_state": current_state_i,
            "at_subgoal": current_state_i == subgoal_i,
        }


def run_smart_episode(layout_name, config, alpha_i, alpha_j, max_steps=15):
    layout = get_layout(layout_name, start_config=config)

    model_i = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells,
        goal_x=layout.goal_positions[0][0], goal_y=layout.goal_positions[0][1],
        start_pos=layout.start_positions[0], num_empathy_levels=3
    )
    model_j = LavaModel(
        width=layout.width, height=layout.height,
        safe_cells=layout.safe_cells,
        goal_x=layout.goal_positions[1][0], goal_y=layout.goal_positions[1][1],
        start_pos=layout.start_positions[1], num_empathy_levels=3
    )

    agent_i = LavaAgent(model=model_i)
    agent_j = LavaAgent(model=model_j)

    # Patch layout
    import tom.planning.jax_hierarchical_planner as hp
    original_layouts = hp.JAX_ZONED_LAYOUTS.copy()
    hp.JAX_ZONED_LAYOUTS["risk_reward"] = create_modified_risk_reward_layout

    planner_i = SmartHierarchicalPlanner(
        agent_i, agent_j, layout_name,
        alpha=alpha_i, alpha_other=alpha_j,
        use_multistep_tom=True,
    )
    planner_j = SmartHierarchicalPlanner(
        agent_j, agent_i, layout_name,
        alpha=alpha_j, alpha_other=alpha_i,
        use_multistep_tom=True,
    )

    hp.JAX_ZONED_LAYOUTS = original_layouts

    env = LavaV2Env(layout_name, num_agents=2, start_config=config)
    state, _ = env.reset()
    pos_i = state["env_state"]["pos"][0]
    pos_j = state["env_state"]["pos"][1]
    trajectory_i = [pos_i]
    trajectory_j = [pos_j]

    info = {}
    for step in range(max_steps):
        qs_i = np.zeros(layout.width * layout.height)
        qs_i[pos_i[1] * layout.width + pos_i[0]] = 1.0
        qs_j = np.zeros(layout.width * layout.height)
        qs_j[pos_j[1] * layout.width + pos_j[0]] = 1.0

        result_i = planner_i.plan_with_debug(qs_i, qs_j)
        result_j = planner_j.plan_with_debug(qs_j, qs_i)
        action_i = result_i["action"]
        action_j = result_j["action"]

        at_subgoal_i = "*" if result_i["at_subgoal"] else ""
        at_subgoal_j = "*" if result_j["at_subgoal"] else ""
        print(f"  Step {step}: i@{pos_i}{at_subgoal_i} -> {ACTION_NAMES[action_i]}, j@{pos_j}{at_subgoal_j} -> {ACTION_NAMES[action_j]}")

        actions = {0: action_i, 1: action_j}
        state, _, _, done, info = env.step(state, actions)
        pos_i = state["env_state"]["pos"][0]
        pos_j = state["env_state"]["pos"][1]
        trajectory_i.append(pos_i)
        trajectory_j.append(pos_j)

        if done:
            break

    goal_i = layout.goal_positions[0]
    goal_j = layout.goal_positions[1]

    return {
        "both_success": pos_i == goal_i and pos_j == goal_j,
        "collision": info.get("collision", False),
        "steps": step + 1,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TEST: Smart subgoal switching (original C_loc when at subgoal)")
    print("=" * 70)
    print()

    print("Both selfish (alpha=0.0, 0.0):")
    result = run_smart_episode("risk_reward", "A", 0.0, 0.0, max_steps=20)
    status = "SUCCESS" if result["both_success"] else ("COLLISION" if result["collision"] else "PARALYSIS")
    print(f"  Result: {status}")
    print()

    print("Asymmetric (alpha=1.0, 0.0):")
    result = run_smart_episode("risk_reward", "A", 1.0, 0.0, max_steps=20)
    status = "SUCCESS" if result["both_success"] else ("COLLISION" if result["collision"] else "PARALYSIS")
    print(f"  Result: {status}")
