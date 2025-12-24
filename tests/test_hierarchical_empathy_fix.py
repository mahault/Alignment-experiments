"""
Test script to verify empathy effects in hierarchical vs flat planner.

This script tests:
1. Current hierarchical planner (where empathy doesn't affect paths)
2. Fixed hierarchical planner (with proper G_social computation)
3. Comparison with flat planner behavior

Run on risk_reward layout to show the difference.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import jax.numpy as jnp
from tom.envs.env_lava_variants import get_layout
from tom.models.model_lava import LavaModel, LavaAgent
from tom.envs.env_lava_v2 import LavaV2Env
from tom.planning.si_empathy_lava import EmpathicLavaPlanner
from tom.planning.jax_hierarchical_planner import (
    HierarchicalEmpathicPlannerJax,
    JaxHierarchicalPlanner,
    get_jax_zoned_layout,
    has_jax_zoned_layout,
    create_subgoal_C_loc_jax,
    propagate_belief_jax,
    expected_pragmatic_utility_jax,
    epistemic_info_gain_jax,
    get_zone_from_belief,
    high_level_plan_jax,
    get_subgoal_state_jax,
    ZONE_STAY,
    ZONE_FORWARD,
    ZONE_BACK,
)
from jax import vmap
import jax

ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]


def create_agents_for_layout(layout_name, config="A"):
    """Create agents for a given layout and configuration."""
    layout = get_layout(layout_name, start_config=config)

    # Extract positions from layout
    start_i = layout.start_positions[0]
    start_j = layout.start_positions[1]
    goal_i = layout.goal_positions[0]
    goal_j = layout.goal_positions[1]

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

    # Add convenience attributes to layout for easier access
    layout.start_i = start_i
    layout.start_j = start_j
    layout.goal_i = goal_i
    layout.goal_j = goal_j

    return layout, agent_i, agent_j


def test_single_step_decision(layout_name, config, alpha_i, alpha_j, use_hierarchical=True):
    """
    Test a single planning step and return the decisions.

    Returns dict with:
    - action_i, action_j: chosen actions
    - G values for both agents
    - zone information (for hierarchical)
    """
    layout, agent_i, agent_j = create_agents_for_layout(layout_name, config)

    # Initial beliefs (point mass at start positions)
    qs_i = np.zeros(layout.width * layout.height)
    qs_i[layout.start_i[1] * layout.width + layout.start_i[0]] = 1.0
    qs_j = np.zeros(layout.width * layout.height)
    qs_j[layout.start_j[1] * layout.width + layout.start_j[0]] = 1.0

    if use_hierarchical and has_jax_zoned_layout(layout_name):
        # Use hierarchical planner
        planner_i = HierarchicalEmpathicPlannerJax(
            agent_i, agent_j, layout_name,
            alpha=alpha_i, alpha_other=alpha_j
        )
        planner_j = HierarchicalEmpathicPlannerJax(
            agent_j, agent_i, layout_name,
            alpha=alpha_j, alpha_other=alpha_i
        )

        # Plan for i
        result_i = planner_i.plan_with_debug(qs_i, qs_j)
        # Plan for j
        result_j = planner_j.plan_with_debug(qs_j, qs_i)

        return {
            "action_i": result_i["action"],
            "action_j": result_j["action"],
            "zone_action_i": result_i["zone_action"],
            "zone_action_j": result_j["zone_action"],
            "G_low_i": result_i["G_low"],
            "G_low_j": result_j["G_low"],
            "my_zone_i": result_i["my_zone"],
            "my_zone_j": result_j["my_zone"],
            "subgoal_i": result_i["subgoal_state"],
            "subgoal_j": result_j["subgoal_state"],
        }
    else:
        # Use flat planner
        planner_i = EmpathicLavaPlanner(
            agent_i, agent_j,
            alpha=alpha_i, alpha_other=alpha_j
        )
        planner_j = EmpathicLavaPlanner(
            agent_j, agent_i,
            alpha=alpha_j, alpha_other=alpha_i
        )

        G_i, G_j_from_i, G_social_i, q_pi_i, action_i = planner_i.plan(qs_i, qs_j)
        G_j, G_i_from_j, G_social_j, q_pi_j, action_j = planner_j.plan(qs_j, qs_i)

        return {
            "action_i": action_i,
            "action_j": action_j,
            "G_i": G_i,
            "G_j": G_j,
            "G_social_i": G_social_i,
            "G_social_j": G_social_j,
        }


def run_episode(layout_name, config, alpha_i, alpha_j, max_steps=15, use_hierarchical=True,
                verbose=False, use_multistep_tom=False):
    """
    Run a full episode and return trajectory.
    """
    layout, agent_i, agent_j = create_agents_for_layout(layout_name, config)
    env = LavaV2Env(layout_name, num_agents=2, start_config=config)

    # Create planners
    if use_hierarchical and has_jax_zoned_layout(layout_name):
        planner_i = HierarchicalEmpathicPlannerJax(
            agent_i, agent_j, layout_name,
            alpha=alpha_i, alpha_other=alpha_j,
            use_multistep_tom=use_multistep_tom,
        )
        planner_j = HierarchicalEmpathicPlannerJax(
            agent_j, agent_i, layout_name,
            alpha=alpha_j, alpha_other=alpha_i,
            use_multistep_tom=use_multistep_tom,
        )
    else:
        planner_i = EmpathicLavaPlanner(
            agent_i, agent_j,
            alpha=alpha_i, alpha_other=alpha_j
        )
        planner_j = EmpathicLavaPlanner(
            agent_j, agent_i,
            alpha=alpha_j, alpha_other=alpha_i
        )

    state, _ = env.reset()
    pos_i = state["env_state"]["pos"][0]
    pos_j = state["env_state"]["pos"][1]
    trajectory_i = [pos_i]
    trajectory_j = [pos_j]

    info = {}
    for step in range(max_steps):
        # Get beliefs
        qs_i = np.zeros(layout.width * layout.height)
        qs_i[pos_i[1] * layout.width + pos_i[0]] = 1.0
        qs_j = np.zeros(layout.width * layout.height)
        qs_j[pos_j[1] * layout.width + pos_j[0]] = 1.0

        # Plan
        if hasattr(planner_i, 'plan_with_debug'):
            result_i = planner_i.plan_with_debug(qs_i, qs_j)
            result_j = planner_j.plan_with_debug(qs_j, qs_i)
            action_i = result_i["action"]
            action_j = result_j["action"]
        else:
            _, _, _, _, action_i = planner_i.plan(qs_i, qs_j)
            _, _, _, _, action_j = planner_j.plan(qs_j, qs_i)

        if verbose:
            print(f"  Step {step}: i@{pos_i} -> {ACTION_NAMES[action_i]}, j@{pos_j} -> {ACTION_NAMES[action_j]}")

        # Step
        actions = {0: action_i, 1: action_j}
        state, _, _, done, info = env.step(state, actions)
        pos_i = state["env_state"]["pos"][0]
        pos_j = state["env_state"]["pos"][1]
        trajectory_i.append(pos_i)
        trajectory_j.append(pos_j)

        if done:
            break

    return {
        "trajectory_i": trajectory_i,
        "trajectory_j": trajectory_j,
        "success_i": pos_i == layout.goal_i,
        "success_j": pos_j == layout.goal_j,
        "both_success": pos_i == layout.goal_i and pos_j == layout.goal_j,
        "collision": info.get("collision", False),
        "steps": step + 1,
    }


def test_empathy_effect_on_decisions():
    """
    Test whether empathy affects decisions in hierarchical planner.

    This is the key diagnostic: if alpha has no effect, trajectories will be identical.
    """
    print("=" * 70)
    print("TEST: Does Empathy Affect Hierarchical Planner Decisions?")
    print("=" * 70)

    layout_name = "narrow"  # Simple layout for debugging
    config = "A"

    empathy_configs = [
        (0.0, 0.0, "Both selfish"),
        (1.0, 0.0, "i empathic, j selfish"),
        (0.0, 1.0, "i selfish, j empathic"),
        (1.0, 1.0, "Both empathic"),
    ]

    print(f"\nLayout: {layout_name}, Config: {config}")
    print("-" * 70)

    results = []
    for alpha_i, alpha_j, desc in empathy_configs:
        result = test_single_step_decision(layout_name, config, alpha_i, alpha_j, use_hierarchical=True)
        results.append((alpha_i, alpha_j, desc, result))
        print(f"\n{desc} (alpha_i={alpha_i}, alpha_j={alpha_j}):")
        print(f"  i: action={ACTION_NAMES[result['action_i']]}, zone_action={result['zone_action_i']}")
        print(f"  j: action={ACTION_NAMES[result['action_j']]}, zone_action={result['zone_action_j']}")
        print(f"  i G_low: {result['G_low_i']}")

    # Check if actions differ with different empathy
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)

    actions_i = [r[3]['action_i'] for r in results]
    actions_j = [r[3]['action_j'] for r in results]

    if len(set(actions_i)) == 1:
        print(f"\nWARNING: Agent i takes SAME action ({ACTION_NAMES[actions_i[0]]}) regardless of empathy!")
        print("This suggests empathy is NOT affecting low-level planning.")
    else:
        print(f"\nAgent i's actions vary with empathy: {[ACTION_NAMES[a] for a in actions_i]}")
        print("Empathy IS affecting decisions.")

    if len(set(actions_j)) == 1:
        print(f"\nWARNING: Agent j takes SAME action ({ACTION_NAMES[actions_j[0]]}) regardless of empathy!")
    else:
        print(f"\nAgent j's actions vary with empathy: {[ACTION_NAMES[a] for a in actions_j]}")


def test_risk_reward_hierarchical():
    """
    Test hierarchical planner on risk_reward layout.

    This is the layout where empathy should matter most - one path is risky but short.
    """
    print("\n" + "=" * 70)
    print("TEST: Risk/Reward Layout - Hierarchical Planner")
    print("=" * 70)

    layout_name = "risk_reward"

    configs = ["A", "B", "C"]
    empathy_configs = [
        (0.0, 0.0, "Both selfish"),
        (1.0, 0.0, "i empathic, j selfish"),
        (0.0, 1.0, "i selfish, j empathic"),
        (1.0, 1.0, "Both empathic"),
    ]

    print(f"\nLayout: {layout_name}")

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Config {config}:")
        print("=" * 70)

        layout, _, _ = create_agents_for_layout(layout_name, config)
        print(f"  Start i: {layout.start_i} -> Goal: {layout.goal_i}")
        print(f"  Start j: {layout.start_j} -> Goal: {layout.goal_j}")

        for alpha_i, alpha_j, desc in empathy_configs:
            result = run_episode(layout_name, config, alpha_i, alpha_j, max_steps=15, use_hierarchical=True)
            status = "SUCCESS" if result["both_success"] else ("COLLISION" if result["collision"] else "PARALYSIS")
            print(f"\n  {desc}:")
            print(f"    Status: {status} in {result['steps']} steps")
            print(f"    Traj i: {' -> '.join(str(p) for p in result['trajectory_i'][:6])}{'...' if len(result['trajectory_i']) > 6 else ''}")
            print(f"    Traj j: {' -> '.join(str(p) for p in result['trajectory_j'][:6])}{'...' if len(result['trajectory_j']) > 6 else ''}")


def compare_flat_vs_hierarchical():
    """
    Compare flat planner (where empathy works) vs hierarchical planner.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: Flat vs Hierarchical Planner")
    print("=" * 70)

    layout_name = "narrow"
    config = "A"

    print(f"\nLayout: {layout_name}, Config: {config}")
    print("Testing asymmetric empathy (i=1.0, j=0.0)")
    print("-" * 70)

    # Flat planner
    print("\nFLAT PLANNER (empathy should work):")
    result_flat = run_episode(layout_name, config, 1.0, 0.0, max_steps=15, use_hierarchical=False, verbose=True)
    print(f"  Result: {'SUCCESS' if result_flat['both_success'] else ('COLLISION' if result_flat['collision'] else 'PARALYSIS')}")

    # Hierarchical planner
    print("\nHIERARCHICAL PLANNER (empathy currently broken):")
    result_hier = run_episode(layout_name, config, 1.0, 0.0, max_steps=15, use_hierarchical=True, verbose=True)
    print(f"  Result: {'SUCCESS' if result_hier['both_success'] else ('COLLISION' if result_hier['collision'] else 'PARALYSIS')}")

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON:")
    print("=" * 70)
    if result_flat["trajectory_i"] == result_hier["trajectory_i"] and result_flat["trajectory_j"] == result_hier["trajectory_j"]:
        print("Trajectories IDENTICAL - hierarchical planner matching flat planner")
    else:
        print("Trajectories DIFFER:")
        print(f"  Flat i:  {result_flat['trajectory_i']}")
        print(f"  Hier i:  {result_hier['trajectory_i']}")
        print(f"  Flat j:  {result_flat['trajectory_j']}")
        print(f"  Hier j:  {result_hier['trajectory_j']}")


def diagnose_low_level_empathy():
    """
    Diagnose exactly where empathy fails in low-level planning.
    """
    print("\n" + "=" * 70)
    print("DIAGNOSIS: Low-Level Empathy Computation")
    print("=" * 70)

    layout_name = "narrow"
    config = "A"
    layout, agent_i, agent_j = create_agents_for_layout(layout_name, config)

    # Initial beliefs
    qs_i = np.zeros(layout.width * layout.height)
    qs_i[layout.start_i[1] * layout.width + layout.start_i[0]] = 1.0
    qs_j = np.zeros(layout.width * layout.height)
    qs_j[layout.start_j[1] * layout.width + layout.start_j[0]] = 1.0

    print(f"\nAgent i at {layout.start_i}, goal at {layout.goal_i}")
    print(f"Agent j at {layout.start_j}, goal at {layout.goal_j}")

    # Get zone layout
    zoned_layout = get_jax_zoned_layout(layout_name, goal_pos=layout.goal_i, width=layout.width)

    # Create hierarchical planner for i
    planner_i_selfish = JaxHierarchicalPlanner.from_model(
        agent_i.model, layout_name, alpha=0.0, alpha_other=0.0
    )
    planner_i_empathic = JaxHierarchicalPlanner.from_model(
        agent_i.model, layout_name, alpha=1.0, alpha_other=0.0
    )

    goal_state_j = layout.goal_j[1] * layout.width + layout.goal_j[0]

    # Plan with selfish alpha
    result_selfish = planner_i_selfish.plan_with_debug(jnp.array(qs_i), jnp.array(qs_j), goal_state_j)
    # Plan with empathic alpha
    result_empathic = planner_i_empathic.plan_with_debug(jnp.array(qs_i), jnp.array(qs_j), goal_state_j)

    print("\n--- High-Level (Zone) Planning ---")
    print(f"Selfish (alpha=0): zone_action={result_selfish['zone_action']}, G_zone={result_selfish['G_zone']}")
    print(f"Empathic (alpha=1): zone_action={result_empathic['zone_action']}, G_zone={result_empathic['G_zone']}")

    print("\n--- Low-Level Planning ---")
    print(f"Selfish:  action={ACTION_NAMES[result_selfish['action']]}, subgoal={result_selfish['subgoal_state']}, G_low={result_selfish['G_low']}")
    print(f"Empathic: action={ACTION_NAMES[result_empathic['action']]}, subgoal={result_empathic['subgoal_state']}, G_low={result_empathic['G_low']}")

    print("\n--- Analysis ---")
    if result_selfish['zone_action'] == result_empathic['zone_action']:
        print("Zone actions SAME - high-level empathy not differentiating (may be correct if not at bottleneck)")
    else:
        print("Zone actions DIFFER - high-level empathy IS working")

    if result_selfish['action'] == result_empathic['action']:
        print("Low-level actions SAME - empathy NOT affecting primitive actions (BUG!)")
    else:
        print("Low-level actions DIFFER - empathy IS affecting primitive actions")

    if np.allclose(result_selfish['G_low'], result_empathic['G_low']):
        print("G_low values IDENTICAL - empathy not incorporated in EFE computation (BUG!)")
    else:
        print("G_low values DIFFER - empathy IS changing EFE values")
        print(f"  Difference: {result_empathic['G_low'] - result_selfish['G_low']}")


def test_multistep_tom_hierarchical():
    """
    Test hierarchical planner with MULTI-STEP ToM (matching test_asymmetric_empathy.py).

    Multi-step ToM allows agents to see long-term consequences of yielding:
    - Depth 2: "I think you think I..."
    - Horizon 3: See multi-step consequences

    This should prevent oscillation seen in single-step empathy.
    """
    print("\n" + "=" * 70)
    print("TEST: Multi-Step ToM Hierarchical Planner")
    print("=" * 70)

    layouts = ["narrow", "risk_reward"]
    configs = ["A", "B"]

    # Key test: asymmetric empathy where one agent should yield
    empathy_configs = [
        (0.0, 0.0, "Both selfish"),
        (1.0, 0.0, "i empathic, j selfish"),
        (0.0, 1.0, "i selfish, j empathic"),
    ]

    for layout_name in layouts:
        if not has_jax_zoned_layout(layout_name):
            print(f"\nSkipping {layout_name} - no zone layout defined")
            continue

        print(f"\n{'=' * 70}")
        print(f"LAYOUT: {layout_name}")
        print("=" * 70)

        for config in configs:
            print(f"\n--- Config {config} ---")
            layout, _, _ = create_agents_for_layout(layout_name, config)
            print(f"  Start i: {layout.start_i} -> Goal: {layout.goal_i}")
            print(f"  Start j: {layout.start_j} -> Goal: {layout.goal_j}")

            for alpha_i, alpha_j, desc in empathy_configs:
                print(f"\n  {desc} (alpha_i={alpha_i}, alpha_j={alpha_j}):")

                # Run with multi-step ToM
                result = run_episode(
                    layout_name, config, alpha_i, alpha_j,
                    max_steps=15, use_hierarchical=True,
                    use_multistep_tom=True, verbose=False
                )

                status = "SUCCESS" if result["both_success"] else ("COLLISION" if result["collision"] else "PARALYSIS")
                print(f"    Status: {status} in {result['steps']} steps")
                print(f"    Traj i: {' -> '.join(str(p) for p in result['trajectory_i'][:8])}{'...' if len(result['trajectory_i']) > 8 else ''}")
                print(f"    Traj j: {' -> '.join(str(p) for p in result['trajectory_j'][:8])}{'...' if len(result['trajectory_j']) > 8 else ''}")


def compare_singlestep_vs_multistep_tom():
    """
    Compare single-step vs multi-step ToM in hierarchical planner.

    Key question: Does multi-step ToM prevent oscillation?
    """
    print("\n" + "=" * 70)
    print("COMPARISON: Single-Step vs Multi-Step ToM")
    print("=" * 70)

    layout_name = "narrow"
    config = "A"

    # Asymmetric empathy where one agent should yield
    alpha_i, alpha_j = 1.0, 0.0

    print(f"\nLayout: {layout_name}, Config: {config}")
    print(f"Empathy: i={alpha_i} (empathic), j={alpha_j} (selfish)")
    print("-" * 70)

    # Single-step empathy
    print("\nSINGLE-STEP EMPATHY (may oscillate):")
    result_single = run_episode(
        layout_name, config, alpha_i, alpha_j,
        max_steps=15, use_hierarchical=True,
        use_multistep_tom=False, verbose=True
    )
    status_single = "SUCCESS" if result_single["both_success"] else ("COLLISION" if result_single["collision"] else "PARALYSIS")
    print(f"  Result: {status_single}")

    # Multi-step ToM
    print("\nMULTI-STEP ToM (should be more stable):")
    result_multi = run_episode(
        layout_name, config, alpha_i, alpha_j,
        max_steps=15, use_hierarchical=True,
        use_multistep_tom=True, verbose=True
    )
    status_multi = "SUCCESS" if result_multi["both_success"] else ("COLLISION" if result_multi["collision"] else "PARALYSIS")
    print(f"  Result: {status_multi}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)

    # Check for oscillation in single-step
    traj_i_single = result_single["trajectory_i"]
    oscillating = False
    for i in range(2, len(traj_i_single)):
        if traj_i_single[i] == traj_i_single[i-2] and traj_i_single[i] != traj_i_single[i-1]:
            oscillating = True
            break

    if oscillating:
        print("Single-step: OSCILLATION detected in trajectory")
    else:
        print("Single-step: No obvious oscillation")

    print(f"\nSingle-step result: {status_single}")
    print(f"Multi-step result:  {status_multi}")

    if status_multi == "SUCCESS" and status_single != "SUCCESS":
        print("\n>>> Multi-step ToM IMPROVED coordination!")
    elif status_multi == "SUCCESS" and status_single == "SUCCESS":
        print("\n>>> Both methods succeeded")
    else:
        print("\n>>> Neither method achieved full success")


if __name__ == "__main__":
    print("=" * 70)
    print("HIERARCHICAL PLANNER EMPATHY DIAGNOSTIC")
    print("=" * 70)

    import sys
    if "--multistep-only" in sys.argv:
        # Quick test of multi-step ToM
        compare_singlestep_vs_multistep_tom()
        test_multistep_tom_hierarchical()
    else:
        # Full diagnostics
        diagnose_low_level_empathy()
        test_empathy_effect_on_decisions()
        compare_flat_vs_hierarchical()
        test_risk_reward_hierarchical()

        # Multi-step ToM tests
        print("\n" + "=" * 70)
        print("MULTI-STEP ToM TESTS")
        print("=" * 70)
        compare_singlestep_vs_multistep_tom()
        test_multistep_tom_hierarchical()
