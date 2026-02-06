"""
Theory of Mind Planner using proper Expected Free Energy (EFE).

Based on Active Inference principles from AIMAPP paper and Sophisticated Inference (Friston 2020):
- EFE = Epistemic (info gain) + Pragmatic (preference satisfaction)
- Pragmatic value = E[log P(o|C)] where C encodes preferences
- Sophisticated inference: recursive EFE over belief trajectories

G_social = G_self + alpha * G_other

Uses JAX for parallel evaluation of all action sequences (no pruning needed).
"""

import math
import numpy as np
from typing import Tuple, List

# JAX for parallel computation
import jax
import jax.numpy as jnp
from functools import partial


class ToMPlanner:
    """
    Theory of Mind planner using Expected Free Energy.

    Key concepts:
    - Beliefs Q(s): Probability distribution over states (positions)
    - Preferences C: Log-probability of preferred observations
    - EFE G(π): Expected surprise under policy π
    - Pragmatic value: -E[log P(o|C)] = expected distance from preferred state
    """

    def __init__(self, agent_id: int, goal_x: float, goal_y: float, alpha: float):
        self.agent_id = agent_id
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.alpha = alpha  # Empathy weight

        # Planning parameters
        self.num_actions = 8  # Discretized action directions
        self.step_size = 0.5  # Step size in meters

        # TIAGo base is ~0.54m wide, so radius ~0.27m
        # Use 0.35m for safety margin (collision threshold = 0.7m)
        self.robot_radius = 0.35

        self.goal_tolerance = 0.3
        self.horizon = 3  # Planning horizon (longer for better lookahead)

        # Discount factor for future EFE (1.0 = no discounting, value all steps equally)
        # With no discounting, the planner can see long-term benefit of backing up
        self.discount = 1.0

        # Preference temperature (lower = sharper preferences)
        self.preference_temp = 0.5

        # Bounds (safe corridor - must match Webots hazard positions!)
        # Hazards at Y=±0.75 with Y-size 0.5 block Y from ±0.5 to ±1.0
        # Clear corridor Y: -0.5 to 0.5 (1.0m wide)
        # Robot center can reach Y=±0.5 (Webots physics handles hazard collision)
        # X bounds account for robot radius touching arena wall
        self.x_min, self.x_max = -2.2, 2.2  # Accounts for robot body at wall
        self.y_min, self.y_max = -0.5, 0.5  # Full corridor width

        # Corridor geometry awareness
        self.corridor_width = self.y_max - self.y_min  # 0.3m effective width for centers
        self.min_passing_clearance = 2 * self.robot_radius  # 0.7m needed to pass
        # With 0.3m Y separation and 0.7m threshold: need sqrt(0.7^2 - 0.3^2) = 0.63m X sep
        self.can_pass_laterally = False  # Too narrow, must back up

        # Corridor geometry for passing decisions
        self.passing_clearance = self.robot_radius * 2  # Need this much lateral space to pass

        # Debug/logging control
        self.verbose = True  # Enable detailed decision logging
        self.debug_counter = 0  # Track planning calls for debug output frequency
        self.verbose_interval = 10  # Output every N calls (5 = ~1sec at 0.2s intervals)

        # Turning cost: penalize actions that require turning before moving
        # This prevents oscillation between lateral moves that require 90° turns
        self.turning_cost_weight = 2.0  # Weight for heading change penalty
        self.last_action = None  # Track last action for commitment bias

        # Precompute action directions (8 directions + stay)
        self.actions = self._generate_actions()

    def _generate_actions(self) -> List[Tuple[float, float]]:
        """Generate discrete action set: 8 directions + stay."""
        actions = [(0.0, 0.0)]  # Stay
        for i in range(self.num_actions):
            angle = 2 * math.pi * i / self.num_actions
            dx = self.step_size * math.cos(angle)
            dy = self.step_size * math.sin(angle)
            actions.append((dx, dy))
        return actions

    def is_backing_up(self, action: Tuple[float, float]) -> bool:
        """Check if action moves away from goal (backing up)."""
        dx, dy = action
        # Backing up means moving in opposite X direction from goal
        goal_direction_x = 1 if self.goal_x > 0 else -1
        action_direction_x = 1 if dx > 0.1 else (-1 if dx < -0.1 else 0)
        return action_direction_x != 0 and action_direction_x != goal_direction_x

    def compute_passing_feasibility(self, my_x: float, my_y: float,
                                     other_x: float, other_y: float) -> bool:
        """
        Check if lateral passing is geometrically feasible.

        Returns True if robots can pass by moving to opposite Y boundaries.
        Returns False if backing up is required.
        """
        # Max possible Y separation (both at opposite boundaries)
        max_y_separation = self.y_max - self.y_min  # 1.0m

        # Current X separation
        x_separation = abs(my_x - other_x)

        # At the point of closest approach (same X), what's the distance?
        # If they try to pass side-by-side at same X with max Y separation
        min_distance_during_pass = max_y_separation

        # Can they pass? Need distance > collision threshold
        can_pass_laterally = min_distance_during_pass > self.min_passing_clearance

        # Also check: with current X separation, can they pass?
        # At current X separation, with max Y, distance would be:
        distance_with_max_y = math.sqrt(x_separation**2 + max_y_separation**2)
        can_pass_at_current_x = distance_with_max_y > self.min_passing_clearance

        return can_pass_laterally and can_pass_at_current_x

    def distance(self, x1: float, y1: float, x2: float, y2: float) -> float:
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def clamp_position(self, x: float, y: float) -> Tuple[float, float]:
        """Clamp position to valid bounds."""
        return (
            max(self.x_min, min(self.x_max, x)),
            max(self.y_min, min(self.y_max, y))
        )

    def transition(self, x: float, y: float, action: Tuple[float, float],
                   other_x: float, other_y: float) -> Tuple[float, float]:
        """
        Predict next position given action.

        Physical and geometric constraints:
        1. Clamp to corridor bounds
        2. If move would cause collision with other robot, transition fails
        3. If corridor is too narrow to pass AND move doesn't increase X separation,
           the move is infeasible (lateral movement won't help)
        """
        dx, dy = action
        next_x, next_y = x + dx, y + dy

        # Clamp to bounds (physical corridor walls)
        next_x, next_y = self.clamp_position(next_x, next_y)

        # Check collision with other robot
        dist_to_other = self.distance(next_x, next_y, other_x, other_y)
        if dist_to_other < self.robot_radius * 2:
            # Collision! Transition fails, stay in place
            return x, y

        # No special blocking logic - pure physics only (collision + bounds)
        # Yielding emerges naturally from EFE when alpha is high enough
        return next_x, next_y

    def preference(self, x: float, y: float, goal_x: float, goal_y: float) -> float:
        """
        Compute log-preference for being at position (x, y).

        log P(o|C) = -distance^2 / (2 * temp^2)

        Higher (less negative) = more preferred.
        At goal: preference = 0 (log(1) = 0)
        Far from goal: preference << 0
        """
        dist = self.distance(x, y, goal_x, goal_y)
        return -(dist ** 2) / (2 * self.preference_temp ** 2)

    def compute_efe(self, start_x: float, start_y: float,
                    goal_x: float, goal_y: float,
                    other_x: float, other_y: float,
                    policy: List[Tuple[float, float]]) -> float:
        """
        Compute Expected Free Energy for a policy.

        EFE G(π) = Σ_τ -E[log P(o_τ|C)]

        = Sum of expected surprises over horizon
        = How far are predicted observations from preferred observations?

        Lower G = better (more preferred observations expected).
        """
        x, y = start_x, start_y
        total_efe = 0.0

        for action in policy:
            # Predict next state given action
            next_x, next_y = self.transition(x, y, action, other_x, other_y)

            # Pragmatic value: how much do I prefer this observation?
            # EFE contribution = -log P(o|C) = negative preference
            pref = self.preference(next_x, next_y, goal_x, goal_y)
            total_efe += -pref  # Negative because lower EFE = better

            x, y = next_x, next_y

        return total_efe / len(policy)  # Average over horizon

    def generate_policies(self) -> List[List[Tuple[float, float]]]:
        """
        Generate candidate policies (action sequences).

        For horizon=2, each policy is [action1, action2].
        """
        policies = []

        # Single-step policies (for quick response)
        for a in self.actions:
            policies.append([a])

        # Two-step policies (if horizon > 1)
        if self.horizon >= 2:
            for a1 in self.actions:
                for a2 in self.actions:
                    policies.append([a1, a2])

        return policies

    def sophisticated_efe_jax(self, my_x: float, my_y: float,
                               other_x: float, other_y: float,
                               other_goal_x: float, other_goal_y: float,
                               other_alpha: float,
                               depth: int,
                               debug: bool = False,
                               verbose: bool = False) -> Tuple[Tuple[float, float], float]:
        """
        Sophisticated inference using JAX for parallel policy evaluation.

        Evaluates first 4 actions using JAX vectorization for speed.
        Uses reduced action set (5 key actions) for efficiency: STAY, FORWARD, BACK, UP, DOWN
        This gives 5^4 = 625 policies instead of 9^4 = 6561.
        """
        # Key actions: STAY, toward goal (±X), lateral (±Y)
        # Determine which direction is "toward goal" and "back"
        toward_x = self.step_size if self.goal_x > my_x else -self.step_size
        back_x = -toward_x

        # Lateral step - use same as forward step since corridor is now wider
        lateral_step = self.step_size

        key_actions = [
            (0.0, 0.0),          # STAY
            (toward_x, 0.0),     # TOWARD goal
            (back_x, 0.0),       # BACK (yield)
            (0.0, lateral_step),   # UP (limited by corridor)
            (0.0, -lateral_step),  # DOWN (limited by corridor)
        ]

        action_names = ["STAY", "TOWARD", "BACK", "UP", "DOWN"]

        # Convert to JAX arrays
        actions_jax = jnp.array(key_actions)
        n_actions = len(key_actions)

        # Generate all policy indices (5^4 = 625)
        policy_indices = jnp.array([
            [a1, a2, a3, a4]
            for a1 in range(n_actions)
            for a2 in range(n_actions)
            for a3 in range(n_actions)
            for a4 in range(n_actions)
        ])

        # Extract constants for JAX (can't use self inside JIT)
        x_min, x_max = self.x_min, self.x_max
        y_min, y_max = self.y_min, self.y_max
        robot_radius = self.robot_radius
        preference_temp = self.preference_temp
        step_size = self.step_size
        discount_factor = self.discount
        alpha = self.alpha

        # Vectorized rollout function - all params passed explicitly
        def rollout_policy(policy_idx, my_pos, other_pos, my_goal, other_goal, actions):
            mx, my_p = my_pos[0], my_pos[1]
            ox, oy = other_pos[0], other_pos[1]

            total_g = 0.0
            discount = 1.0

            # Unroll the loop for JAX compatibility
            for step_idx in range(depth):
                # Get action for this step
                if step_idx < 4:
                    action = actions[policy_idx[step_idx]]
                else:
                    action = actions[0]  # STAY for steps >= 4

                # My move
                next_mx = jnp.clip(mx + action[0], x_min, x_max)
                next_my = jnp.clip(my_p + action[1], y_min, y_max)

                # Check collision
                dist = jnp.sqrt((next_mx - ox)**2 + (next_my - oy)**2)
                collision = dist < robot_radius * 2
                next_mx = jnp.where(collision, mx, next_mx)
                next_my = jnp.where(collision, my_p, next_my)

                # My EFE
                my_dist = jnp.sqrt((next_mx - my_goal[0])**2 + (next_my - my_goal[1])**2)
                g_self = my_dist**2 / (2 * preference_temp**2)

                # Other's simple response (greedy toward their goal)
                # Use clip instead of sign*step to avoid oscillation around goal
                other_toward_x = jnp.clip(other_goal[0] - ox, -step_size, step_size)
                next_ox = jnp.clip(ox + other_toward_x, x_min, x_max)
                next_oy = oy

                # Check other's collision with my new position
                other_dist_to_me = jnp.sqrt((next_ox - next_mx)**2 + (next_oy - next_my)**2)
                other_collision = other_dist_to_me < robot_radius * 2
                next_ox = jnp.where(other_collision, ox, next_ox)
                next_oy = jnp.where(other_collision, oy, next_oy)

                # Other's EFE
                other_dist = jnp.sqrt((next_ox - other_goal[0])**2 + (next_oy - other_goal[1])**2)
                g_other = other_dist**2 / (2 * preference_temp**2)

                # Social EFE
                g_step = g_self + alpha * g_other
                total_g = total_g + discount * g_step
                discount = discount * discount_factor

                # Update positions
                mx, my_p = next_mx, next_my
                ox, oy = next_ox, next_oy

            return total_g

        # JIT compile the rollout
        rollout_jit = jax.jit(jax.vmap(
            lambda p: rollout_policy(p,
                                     jnp.array([my_x, my_y]),
                                     jnp.array([other_x, other_y]),
                                     jnp.array([self.goal_x, self.goal_y]),
                                     jnp.array([other_goal_x, other_goal_y]),
                                     actions_jax)
        ))

        # Evaluate all policies in parallel
        all_g = rollout_jit(policy_indices)

        # Pure EFE - no hardcoded preferences or biases
        # Yielding emerges from G_social = G_self + alpha * G_other

        best_idx = int(jnp.argmin(all_g))
        best_policy = policy_indices[best_idx]
        best_g = float(all_g[best_idx])  # Report original G, not adjusted
        best_action = tuple(float(x) for x in key_actions[int(best_policy[0])])

        if debug:
            print(f"    Best policy: {[self._action_name(key_actions[int(i)]) for i in best_policy]}")
            print(f"    Best G: {best_g:.2f}")

        # VERBOSE DEBUG: Show all first-action options and why the choice was made
        if verbose:
            print(f"\n{'='*60}")
            print(f"DECISION LOG for agent {self.agent_id} (alpha={self.alpha})")
            print(f"{'='*60}")
            print(f"  My position: ({my_x:.2f}, {my_y:.2f})")
            print(f"  My goal: ({self.goal_x:.2f}, {self.goal_y:.2f})")
            print(f"  Other position: ({other_x:.2f}, {other_y:.2f})")
            print(f"  Other goal: ({other_goal_x:.2f}, {other_goal_y:.2f})")
            print(f"  Distance to goal: {self.distance(my_x, my_y, self.goal_x, self.goal_y):.2f}")
            print(f"  Distance to other: {self.distance(my_x, my_y, other_x, other_y):.2f}")
            print(f"  Collision threshold: {2*self.robot_radius:.2f}")
            print(f"\n  Key actions: toward_x={toward_x:.1f}, back_x={back_x:.1f}")

            # Show what each first action results in (clamped position)
            print(f"\n  First action feasibility:")
            for i, (action, name) in enumerate(zip(key_actions, action_names)):
                next_x = max(self.x_min, min(self.x_max, my_x + action[0]))
                next_y = max(self.y_min, min(self.y_max, my_y + action[1]))
                clamped = (next_x != my_x + action[0]) or (next_y != my_y + action[1])
                dist_other = self.distance(next_x, next_y, other_x, other_y)
                collision = dist_other < 2*self.robot_radius
                actual_x, actual_y = (my_x, my_y) if collision else (next_x, next_y)
                clamp_str = " [CLAMPED]" if clamped else ""
                coll_str = " [COLLISION]" if collision else ""
                print(f"    {name:8s}: ({my_x:.2f},{my_y:.2f}) -> ({actual_x:.2f},{actual_y:.2f}){clamp_str}{coll_str}")

            # Best G for each first action (aggregate over all 125 policies starting with that action)
            print(f"\n  Best G by first action:")
            for first_action_idx in range(n_actions):
                # Find policies that start with this action
                mask = policy_indices[:, 0] == first_action_idx
                matching_g = all_g[mask]
                if len(matching_g) > 0:
                    best_for_action = float(jnp.min(matching_g))
                    best_policy_idx_for_action = int(jnp.argmin(matching_g))
                    # Find the actual policy
                    matching_policies = policy_indices[mask]
                    best_4step = matching_policies[best_policy_idx_for_action]
                    policy_str = "->".join([action_names[int(j)] for j in best_4step])
                    marker = " <-- CHOSEN" if first_action_idx == int(best_policy[0]) else ""
                    print(f"    {action_names[first_action_idx]:8s}: G={best_for_action:8.2f}  best_policy=[{policy_str}]{marker}")

            # Show the full best policy rollout
            print(f"\n  Best policy rollout ({[action_names[int(i)] for i in best_policy]}):")
            mx, my_p = my_x, my_y
            ox, oy = other_x, other_y
            cumulative_g = 0.0
            for step_idx in range(min(depth, 4)):
                action = key_actions[int(best_policy[step_idx])]
                # My move
                next_mx = max(self.x_min, min(self.x_max, mx + action[0]))
                next_my = max(self.y_min, min(self.y_max, my_p + action[1]))
                dist = math.sqrt((next_mx - ox)**2 + (next_my - oy)**2)
                if dist < self.robot_radius * 2:
                    next_mx, next_my = mx, my_p
                # My EFE
                my_dist = math.sqrt((next_mx - self.goal_x)**2 + (next_my - self.goal_y)**2)
                g_self = my_dist**2 / (2 * self.preference_temp**2)
                # Other moves (greedy toward their goal)
                other_toward_x = max(-self.step_size, min(self.step_size, other_goal_x - ox))
                next_ox = max(self.x_min, min(self.x_max, ox + other_toward_x))
                next_oy = oy
                other_dist_to_me = math.sqrt((next_ox - next_mx)**2 + (next_oy - next_my)**2)
                if other_dist_to_me < self.robot_radius * 2:
                    next_ox, next_oy = ox, oy
                # Other's EFE
                other_dist = math.sqrt((next_ox - other_goal_x)**2 + (next_oy - other_goal_y)**2)
                g_other = other_dist**2 / (2 * self.preference_temp**2)
                g_step = g_self + self.alpha * g_other
                cumulative_g += g_step
                print(f"    Step {step_idx+1}: {action_names[int(best_policy[step_idx])]:8s} me=({next_mx:.2f},{next_my:.2f}) other=({next_ox:.2f},{next_oy:.2f}) G_self={g_self:.2f} G_other={g_other:.2f} G_step={g_step:.2f}")
                mx, my_p = next_mx, next_my
                ox, oy = next_ox, next_oy
            print(f"    Cumulative G (4 steps): {cumulative_g:.2f}")
            print(f"{'='*60}\n")

        return best_action, best_g

    def _rollout_efe_four_step(self, my_x: float, my_y: float,
                                action1: Tuple[float, float], action2: Tuple[float, float],
                                action3: Tuple[float, float], action4: Tuple[float, float],
                                other_x: float, other_y: float,
                                other_goal_x: float, other_goal_y: float,
                                other_alpha: float,
                                depth: int) -> float:
        """
        Rollout with first FOUR actions specified explicitly.

        This allows the planner to evaluate multi-step backing up strategies
        like "back up → back up → back up → back up" which may be necessary
        when one robot must yield significantly to let the other pass.
        """
        mx, my = my_x, my_y
        ox, oy = other_x, other_y
        explicit_actions = [action1, action2, action3, action4]

        total_g_self = 0.0
        total_g_other = 0.0
        discount = 1.0

        for step in range(depth):
            if step < 4:
                my_action = explicit_actions[step]
            else:
                # Use simple greedy for remaining steps
                my_action = self._greedy_action_1step(
                    mx, my, self.goal_x, self.goal_y, ox, oy
                )

            # I move
            mx_new, my_new = self.transition(mx, my, my_action, ox, oy)

            # My EFE contribution
            g_self_step = -self.preference(mx_new, my_new, self.goal_x, self.goal_y)
            total_g_self += discount * g_self_step

            # Other responds (simple prediction to keep computation tractable)
            other_action = self._predict_other_action_simple(
                ox, oy, other_goal_x, other_goal_y, mx_new, my_new
            )
            ox_new, oy_new = self.transition(ox, oy, other_action, mx_new, my_new)

            # Other's EFE contribution
            g_other_step = -self.preference(ox_new, oy_new, other_goal_x, other_goal_y)
            total_g_other += discount * g_other_step

            # Update positions
            mx, my = mx_new, my_new
            ox, oy = ox_new, oy_new

            # Discount future
            discount *= self.discount

        # Social EFE
        return total_g_self + self.alpha * total_g_other

    def _rollout_efe_two_step(self, my_x: float, my_y: float,
                               action1: Tuple[float, float], action2: Tuple[float, float],
                               other_x: float, other_y: float,
                               other_goal_x: float, other_goal_y: float,
                               other_alpha: float,
                               depth: int) -> float:
        """
        Rollout with first TWO actions specified explicitly.

        Pure EFE - no penalties. The accumulated EFE over the trajectory
        should naturally show that strategies leading to being stuck have
        higher total EFE than strategies that enable reaching the goal.
        """
        mx, my = my_x, my_y
        ox, oy = other_x, other_y

        total_g_self = 0.0
        total_g_other = 0.0
        discount = 1.0

        for step in range(depth):
            if step == 0:
                my_action = action1
            elif step == 1:
                my_action = action2
            else:
                # Use remaining horizon for lookahead, capped for efficiency
                remaining_horizon = min(depth - step, 8)  # Cap at 8 to avoid slow computation
                my_action = self._greedy_action(
                    mx, my, self.goal_x, self.goal_y, ox, oy,
                    other_goal_x, other_goal_y, other_alpha, lookahead=remaining_horizon
                )

            # I move
            mx_new, my_new = self.transition(mx, my, my_action, ox, oy)

            # My EFE contribution
            g_self_step = -self.preference(mx_new, my_new, self.goal_x, self.goal_y)
            total_g_self += discount * g_self_step

            # Other responds (using their alpha!)
            other_action = self.predict_other_action(
                ox, oy, other_goal_x, other_goal_y, other_alpha, mx_new, my_new
            )
            ox_new, oy_new = self.transition(ox, oy, other_action, mx_new, my_new)

            # Other's EFE contribution
            g_other_step = -self.preference(ox_new, oy_new, other_goal_x, other_goal_y)
            total_g_other += discount * g_other_step

            # Update positions
            mx, my = mx_new, my_new
            ox, oy = ox_new, oy_new

            # Discount future
            discount *= self.discount

        # Social EFE
        return total_g_self + self.alpha * total_g_other

    def _action_name(self, action: Tuple[float, float]) -> str:
        """Get human-readable action name."""
        if action == (0.0, 0.0):
            return "STAY"
        angle = math.atan2(action[1], action[0])
        if abs(angle) < 0.4:
            return "RIGHT"
        elif abs(angle - math.pi) < 0.4 or abs(angle + math.pi) < 0.4:
            return "LEFT"
        elif abs(angle - math.pi/2) < 0.4:
            return "UP"
        elif abs(angle + math.pi/2) < 0.4:
            return "DOWN"
        elif angle > 0 and angle < math.pi/2:
            return "UP-RIGHT"
        elif angle > math.pi/2:
            return "UP-LEFT"
        elif angle < 0 and angle > -math.pi/2:
            return "DOWN-RIGHT"
        else:
            return "DOWN-LEFT"

    def _rollout_efe_jax(self, my_x: float, my_y: float, first_action: Tuple[float, float],
                         other_x: float, other_y: float,
                         other_goal_x: float, other_goal_y: float,
                         other_alpha: float,
                         depth: int) -> float:
        """
        Rollout a trajectory starting with first_action.

        Pure EFE-based: no hardcoded intent classification.
        At each step after the first, pick greedy action based on immediate EFE.

        The key insight: if first_action leads to a blocked state, subsequent
        greedy actions will also be blocked (no progress), accumulating high EFE.
        Lateral/yielding first_actions open up space for greedy progress later,
        resulting in lower cumulative EFE.

        Other agent responds using their alpha-aware policy.
        """
        mx, my = my_x, my_y
        ox, oy = other_x, other_y

        total_g_self = 0.0
        total_g_other = 0.0
        discount = 1.0

        for step in range(depth):
            if step == 0:
                my_action = first_action
            else:
                # Use remaining horizon for lookahead, capped for efficiency
                remaining_horizon = min(depth - step, 8)  # Cap at 8 to avoid slow computation
                my_action = self._greedy_action(
                    mx, my, self.goal_x, self.goal_y, ox, oy,
                    other_goal_x, other_goal_y, other_alpha, lookahead=remaining_horizon
                )

            # I move
            mx_new, my_new = self.transition(mx, my, my_action, ox, oy)

            # My EFE contribution
            g_self_step = -self.preference(mx_new, my_new, self.goal_x, self.goal_y)
            total_g_self += discount * g_self_step

            # Other responds (using their alpha!)
            other_action = self.predict_other_action(
                ox, oy, other_goal_x, other_goal_y, other_alpha, mx_new, my_new
            )
            ox_new, oy_new = self.transition(ox, oy, other_action, mx_new, my_new)

            # Other's EFE contribution
            g_other_step = -self.preference(ox_new, oy_new, other_goal_x, other_goal_y)
            total_g_other += discount * g_other_step

            # Update positions
            mx, my = mx_new, my_new
            ox, oy = ox_new, oy_new

            # Discount future
            discount *= self.discount

        # Social EFE
        return total_g_self + self.alpha * total_g_other

    def _greedy_action(self, x: float, y: float, goal_x: float, goal_y: float,
                       other_x: float, other_y: float,
                       other_goal_x: float = None, other_goal_y: float = None,
                       other_alpha: float = 0.5,
                       lookahead: int = 2) -> Tuple[float, float]:
        """
        Select action with lowest EFE using multi-step lookahead (pure EFE).

        Instead of 1-step greedy, we look ahead `lookahead` steps to see
        which action leads to best cumulative EFE. This allows the planner
        to see that "move lateral now → can move toward goal later" is
        better than "try toward goal now → get blocked".

        Uses JAX-compatible vectorized computation.
        """
        if other_goal_x is None:
            other_goal_x = -goal_x  # Assume opposite goal
        if other_goal_y is None:
            other_goal_y = goal_y

        best_action = (0.0, 0.0)
        best_efe = float('inf')

        for action in self.actions:
            # Simulate lookahead steps
            mx, my = x, y
            ox, oy = other_x, other_y
            total_efe = 0.0
            discount = 1.0

            for step in range(lookahead):
                if step == 0:
                    my_action = action
                else:
                    # 1-step greedy for subsequent steps (to avoid infinite recursion)
                    my_action = self._greedy_action_1step(mx, my, goal_x, goal_y, ox, oy)

                # I move
                mx_new, my_new = self.transition(mx, my, my_action, ox, oy)
                total_efe += discount * (-self.preference(mx_new, my_new, goal_x, goal_y))

                # Other responds
                other_action = self._predict_other_action_simple(
                    ox, oy, other_goal_x, other_goal_y, mx_new, my_new
                )
                ox_new, oy_new = self.transition(ox, oy, other_action, mx_new, my_new)

                mx, my = mx_new, my_new
                ox, oy = ox_new, oy_new
                discount *= self.discount

            if total_efe < best_efe:
                best_efe = total_efe
                best_action = action

        return best_action

    def _greedy_action_1step(self, x: float, y: float, goal_x: float, goal_y: float,
                              other_x: float, other_y: float) -> Tuple[float, float]:
        """Simple 1-step greedy action selection (used inside lookahead to avoid recursion)."""
        best_action = (0.0, 0.0)
        best_efe = float('inf')

        for action in self.actions:
            nx, ny = self.transition(x, y, action, other_x, other_y)
            efe = -self.preference(nx, ny, goal_x, goal_y)
            if efe < best_efe:
                best_efe = efe
                best_action = action

        return best_action

    def _predict_other_action_simple(self, other_x: float, other_y: float,
                                      other_goal_x: float, other_goal_y: float,
                                      my_x: float, my_y: float) -> Tuple[float, float]:
        """Simple prediction of other's action (1-step greedy toward their goal)."""
        best_action = (0.0, 0.0)
        best_efe = float('inf')

        for action in self.actions:
            nx, ny = self.transition(other_x, other_y, action, my_x, my_y)
            efe = -self.preference(nx, ny, other_goal_x, other_goal_y)
            if efe < best_efe:
                best_efe = efe
                best_action = action

        return best_action

    def predict_other_action(self, other_x: float, other_y: float,
                              other_goal_x: float, other_goal_y: float,
                              other_alpha: float,
                              my_x: float, my_y: float) -> Tuple[float, float]:
        """
        Theory of Mind: Predict other's best action using 2-step lookahead.

        KEY: Simulate their FULL decision process including:
        1. Their 2-step lookahead (they also plan ahead)
        2. Their empathy (other_alpha weights how much they care about me)

        other's G_social = other's G_self + other_alpha * G_for_me
        """
        best_action = (0.0, 0.0)
        best_g_social = float('inf')

        for action in self.actions:
            # Simulate 2-step lookahead for OTHER's action evaluation
            ox, oy = other_x, other_y
            mx, my_pos = my_x, my_y

            total_other_g_self = 0.0
            total_my_g = 0.0
            discount = 1.0

            for step in range(2):  # 2-step lookahead
                if step == 0:
                    other_action = action
                else:
                    # Other's greedy action (1-step to avoid deep recursion)
                    other_action = self._greedy_action_1step(ox, oy, other_goal_x, other_goal_y, mx, my_pos)

                # Other moves
                ox_new, oy_new = self.transition(ox, oy, other_action, mx, my_pos)
                total_other_g_self += discount * (-self.preference(ox_new, oy_new, other_goal_x, other_goal_y))

                # I respond (from other's perspective, predicting what I'll do)
                my_action = self._greedy_action_1step(mx, my_pos, self.goal_x, self.goal_y, ox_new, oy_new)
                mx_new, my_new = self.transition(mx, my_pos, my_action, ox_new, oy_new)
                total_my_g += discount * (-self.preference(mx_new, my_new, self.goal_x, self.goal_y))

                ox, oy = ox_new, oy_new
                mx, my_pos = mx_new, my_new
                discount *= self.discount

            # Other's social EFE with their empathy
            other_g_social = total_other_g_self + other_alpha * total_my_g

            if other_g_social < best_g_social:
                best_g_social = other_g_social
                best_action = action

        return best_action

    def plan(self, my_x: float, my_y: float,
             other_x: float, other_y: float,
             other_goal_x: float, other_goal_y: float,
             other_alpha: float) -> Tuple[float, float, str]:
        """
        Plan next position using SOPHISTICATED INFERENCE (Friston 2020).

        Uses recursive EFE computation that rolls out:
        - My action → Other's response (using their alpha) → My next action → ...

        This allows the empathic agent to see that backing up NOW
        enables both agents to reach their goals LATER.
        """
        # At goal?
        if self.distance(my_x, my_y, self.goal_x, self.goal_y) < self.goal_tolerance:
            return self.goal_x, self.goal_y, "AT_GOAL"

        # Control verbose output frequency (every N calls to avoid flooding)
        self.debug_counter += 1
        show_verbose = self.verbose and (self.debug_counter % self.verbose_interval == 1)

        # Use JAX-based sophisticated inference (no pruning, all paths evaluated)
        # Longer horizon (16 steps) to see full benefit of backing up strategies
        best_action, best_g = self.sophisticated_efe_jax(
            my_x, my_y,
            other_x, other_y,
            other_goal_x, other_goal_y,
            other_alpha,
            depth=12,  # Balanced horizon for 4-step policy evaluation
            verbose=show_verbose
        )

        # Execute the best action
        # Check collision against other's CURRENT position (consistent with rollout)
        # The rollout already accounts for other's response in the EFE calculation
        next_x, next_y = self.transition(
            my_x, my_y, best_action, other_x, other_y
        )

        # Debug label - distinguish between selfish maneuvering and empathic yielding
        dist_now = self.distance(my_x, my_y, self.goal_x, self.goal_y)
        dist_after = self.distance(next_x, next_y, self.goal_x, self.goal_y)

        if abs(next_x - my_x) < 0.01 and abs(next_y - my_y) < 0.01:
            label = "STAY"
        elif dist_after < dist_now - 0.05:
            label = "TOWARD"
        elif dist_after > dist_now + 0.05:
            # Moving away from goal - but WHY?
            if self.alpha < 0.1:
                label = "MANEUVER"  # Selfish robot navigating around obstacle
            else:
                label = "YIELD"     # Empathic robot yielding for other
        else:
            label = "LATERAL"

        debug = f"a={self.alpha:.1f} G={best_g:.2f} [{label}]"

        # Track last action for commitment bias
        self.last_action = best_action

        return next_x, next_y, debug


def debug_plan(planner, my_x, my_y, other_x, other_y, other_goal_x, other_goal_y, other_alpha):
    """Debug version showing sophisticated inference results."""
    print(f"\n  DEBUG for agent {planner.agent_id} at ({my_x:.2f}, {my_y:.2f}):")
    print(f"  Goal: ({planner.goal_x}, {planner.goal_y}), Other at: ({other_x:.2f}, {other_y:.2f})")
    print(f"  Using JAX SOPHISTICATED INFERENCE (depth=12, 4-step policies)")

    # Show what sophisticated inference chooses with debug=True
    best_action, best_g = planner.sophisticated_efe_jax(
        my_x, my_y, other_x, other_y, other_goal_x, other_goal_y, other_alpha, depth=12, debug=True
    )

    # Direction name
    if best_action == (0, 0):
        name = "STAY"
    else:
        angle = math.atan2(best_action[1], best_action[0])
        if abs(angle) < 0.4: name = "RIGHT"
        elif abs(angle - math.pi) < 0.4 or abs(angle + math.pi) < 0.4: name = "LEFT"
        elif abs(angle - math.pi/2) < 0.4: name = "UP"
        elif abs(angle + math.pi/2) < 0.4: name = "DOWN"
        else: name = "DIAG"

    print(f"  Best action: {name} with G={best_g:.2f}")


def debug_wall_situation():
    """Debug the stuck-at-wall scenario to understand why lateral isn't chosen."""
    print("\n" + "=" * 70)
    print("WALL SITUATION DEBUG - Why doesn't empathic robot move laterally?")
    print("=" * 70)

    # Recreate the stuck situation
    p2 = ToMPlanner(agent_id=1, goal_x=-1.8, goal_y=0.0, alpha=6.0)

    # TIAGo_2 at wall, TIAGo_1 nearby
    my_x, my_y = 2.24, 0.0  # TIAGo_2 (empathic) at wall
    other_x, other_y = 1.16, 0.0  # TIAGo_1 (selfish)
    other_goal_x, other_goal_y = 1.8, 0.0

    print(f"\nTIAGo_2 (empathic, alpha={p2.alpha}):")
    print(f"  Position: ({my_x}, {my_y})")
    print(f"  Goal: ({p2.goal_x}, {p2.goal_y})")
    print(f"  X bounds: [{p2.x_min}, {p2.x_max}]")
    print(f"  Y bounds: [{p2.y_min}, {p2.y_max}]")

    print(f"\nTIAGo_1 (selfish):")
    print(f"  Position: ({other_x}, {other_y})")
    print(f"  Goal: ({other_goal_x}, {other_goal_y})")

    print(f"\nDistance between: {p2.distance(my_x, my_y, other_x, other_y):.2f}m")
    print(f"Collision threshold: {2 * p2.robot_radius}m")

    # Test each first action manually
    print("\n" + "-" * 70)
    print("Evaluating first action options:")
    print("-" * 70)

    toward_x = p2.step_size if p2.goal_x > my_x else -p2.step_size
    actions = [
        ((0.0, 0.0), "STAY"),
        ((toward_x, 0.0), "TOWARD"),
        ((-toward_x, 0.0), "BACK"),
        ((0.0, p2.step_size), "UP"),
        ((0.0, -p2.step_size), "DOWN"),
    ]

    for action, name in actions:
        # Where would I end up?
        next_x = max(p2.x_min, min(p2.x_max, my_x + action[0]))
        next_y = max(p2.y_min, min(p2.y_max, my_y + action[1]))

        # Check collision with other
        dist_to_other = p2.distance(next_x, next_y, other_x, other_y)
        blocked = dist_to_other < 2 * p2.robot_radius

        # If other advances toward their goal, what's the distance?
        other_next_x = other_x + 0.5  # Assume other moves toward goal
        dist_if_other_advances = p2.distance(next_x, next_y, other_next_x, other_y)
        other_can_advance = dist_if_other_advances >= 2 * p2.robot_radius

        print(f"\n  {name}: ({my_x:.2f},{my_y:.2f}) -> ({next_x:.2f},{next_y:.2f})")
        print(f"    Clipped: {next_x != my_x + action[0] or next_y != my_y + action[1]}")
        print(f"    Dist to other: {dist_to_other:.2f}m {'(BLOCKED)' if blocked else '(OK)'}")
        print(f"    If other advances to {other_next_x:.2f}: dist={dist_if_other_advances:.2f}m {'-> CAN PASS' if other_can_advance else '-> STILL BLOCKED'}")


def test_planner():
    """Test with two robots approaching each other."""
    p1 = ToMPlanner(agent_id=0, goal_x=1.25, goal_y=0.0, alpha=0.0)  # Purely selfish (ignores other completely)
    p2 = ToMPlanner(agent_id=1, goal_x=-1.25, goal_y=0.0, alpha=6.0)  # Highly empathic (altruistic)

    print("EFE-based ToM Planner Test")
    print("=" * 60)
    print(f"Robot 1: alpha={p1.alpha} (selfish), goal=({p1.goal_x}, {p1.goal_y})")
    print(f"Robot 2: alpha={p2.alpha} (empathic), goal=({p2.goal_x}, {p2.goal_y})")
    print()

    # Simulate
    r1_x, r1_y = -1.0, 0.0
    r2_x, r2_y = 1.0, 0.0

    for step in range(15):
        # Debug at step 4 when they get stuck
        if step == 3:
            debug_plan(p1, r1_x, r1_y, r2_x, r2_y, p2.goal_x, p2.goal_y, p2.alpha)
            debug_plan(p2, r2_x, r2_y, r1_x, r1_y, p1.goal_x, p1.goal_y, p1.alpha)

        t1_x, t1_y, info1 = p1.plan(r1_x, r1_y, r2_x, r2_y, p2.goal_x, p2.goal_y, p2.alpha)
        t2_x, t2_y, info2 = p2.plan(r2_x, r2_y, r1_x, r1_y, p1.goal_x, p1.goal_y, p1.alpha)

        print(f"{step+1:2d}. R1({r1_x:+.2f},{r1_y:+.2f})->({t1_x:+.2f},{t1_y:+.2f}) {info1}")
        print(f"    R2({r2_x:+.2f},{r2_y:+.2f})->({t2_x:+.2f},{t2_y:+.2f}) {info2}")

        r1_x, r1_y = t1_x, t1_y
        r2_x, r2_y = t2_x, t2_y

        # Check goals
        if p1.distance(r1_x, r1_y, p1.goal_x, p1.goal_y) < 0.3 and \
           p2.distance(r2_x, r2_y, p2.goal_x, p2.goal_y) < 0.3:
            print("\nBoth reached goals!")
            break
        print()


def test_narrow_corridor():
    """Test with a narrow corridor where robots CAN'T pass side-by-side.

    This should force one robot to back up to let the other through.
    """
    print("\n" + "=" * 60)
    print("NARROW CORRIDOR TEST - Backing up should be required")
    print("=" * 60)

    # Create planners with NARROW corridor bounds
    p1 = ToMPlanner(agent_id=0, goal_x=1.25, goal_y=0.0, alpha=0.0)  # Purely selfish
    p2 = ToMPlanner(agent_id=1, goal_x=-1.25, goal_y=0.0, alpha=6.0)  # Highly empathic

    # Override corridor bounds to be IMPOSSIBLE to pass side-by-side
    # Robot radius is 0.5, so need 1.0m clearance to pass
    # With Y bounds of ±0.4, corridor is 0.8m - NOT enough to pass!
    # One robot MUST back up to let the other through.
    narrow_y = 0.4
    p1.y_min, p1.y_max = -narrow_y, narrow_y
    p2.y_min, p2.y_max = -narrow_y, narrow_y

    print(f"Corridor Y bounds: [{-narrow_y}, {narrow_y}] (width={2*narrow_y:.1f}m)")
    print(f"Robot radius: {p1.robot_radius}m, need {2*p1.robot_radius}m clearance to pass")
    print(f"Result: CANNOT pass side-by-side - one must BACK UP completely!")
    print()

    # Start robots closer together
    r1_x, r1_y = -0.6, 0.0
    r2_x, r2_y = 0.6, 0.0

    print(f"Initial distance: {p1.distance(r1_x, r1_y, r2_x, r2_y):.2f}m")
    print(f"Collision threshold: {2*p1.robot_radius}m")
    print()

    for step in range(15):
        # Debug every step to see what's happening
        if step < 5:
            print(f"\n--- Step {step+1} Debug ---")
            debug_plan(p1, r1_x, r1_y, r2_x, r2_y, p2.goal_x, p2.goal_y, p2.alpha)

        t1_x, t1_y, info1 = p1.plan(r1_x, r1_y, r2_x, r2_y, p2.goal_x, p2.goal_y, p2.alpha)
        t2_x, t2_y, info2 = p2.plan(r2_x, r2_y, r1_x, r1_y, p1.goal_x, p1.goal_y, p1.alpha)

        # Check if either robot moved backwards (away from goal)
        r1_moved_back = (p1.goal_x > r1_x and t1_x < r1_x) or (p1.goal_x < r1_x and t1_x > r1_x)
        r2_moved_back = (p2.goal_x > r2_x and t2_x < r2_x) or (p2.goal_x < r2_x and t2_x > r2_x)

        back_indicator1 = " <-- BACKED UP!" if r1_moved_back else ""
        back_indicator2 = " <-- BACKED UP!" if r2_moved_back else ""

        print(f"{step+1:2d}. R1({r1_x:+.2f},{r1_y:+.2f})->({t1_x:+.2f},{t1_y:+.2f}) {info1}{back_indicator1}")
        print(f"    R2({r2_x:+.2f},{r2_y:+.2f})->({t2_x:+.2f},{t2_y:+.2f}) {info2}{back_indicator2}")

        # Check for deadlock (neither moved)
        if abs(t1_x - r1_x) < 0.01 and abs(t1_y - r1_y) < 0.01 and \
           abs(t2_x - r2_x) < 0.01 and abs(t2_y - r2_y) < 0.01:
            print("\n*** DEADLOCK - Neither robot moved! ***")
            print("This means backward movement isn't being considered properly.")
            break

        r1_x, r1_y = t1_x, t1_y
        r2_x, r2_y = t2_x, t2_y

        # Check goals
        if p1.distance(r1_x, r1_y, p1.goal_x, p1.goal_y) < 0.3 and \
           p2.distance(r2_x, r2_y, p2.goal_x, p2.goal_y) < 0.3:
            print("\nBoth reached goals!")
            break
        print()


def test_empathy_yielding():
    """
    Test that empathy (high alpha) makes yielding preferable to staying.

    The empathic robot's social EFE: G_social = G_self + alpha * G_other

    When yielding:
    - G_self increases (I'm further from my goal)
    - G_other decreases (other robot can now make progress)

    If alpha * decrease_in_G_other > increase_in_G_self, yielding wins.
    """
    print("\n" + "=" * 70)
    print("EMPATHY YIELDING TEST - Why should high alpha prefer yielding?")
    print("=" * 70)

    # Create empathic robot (alpha=6.0 = highly altruistic)
    p_empathic = ToMPlanner(agent_id=1, goal_x=-1.25, goal_y=0.0, alpha=6.0)

    # Scenario: robots facing each other, close enough that forward is blocked
    # but far enough that lateral movement is possible
    # Empathic robot at positive X, goal at negative X (needs to go LEFT)
    # Other robot at negative X, goal at positive X (needs to go RIGHT)
    my_x, my_y = 0.6, 0.0  # Empathic robot position
    other_x, other_y = -0.6, 0.0  # Selfish robot position (1.2m apart - just over collision)
    other_goal_x, other_goal_y = 1.25, 0.0  # Selfish robot's goal
    other_alpha = 0.0  # Selfish robot ignores other completely

    print(f"\nEmpathic robot (alpha={p_empathic.alpha}):")
    print(f"  Position: ({my_x}, {my_y})")
    print(f"  Goal: ({p_empathic.goal_x}, {p_empathic.goal_y}) <-- needs to go LEFT")
    print(f"\nSelfish robot (alpha={other_alpha}):")
    print(f"  Position: ({other_x}, {other_y})")
    print(f"  Goal: ({other_goal_x}, {other_goal_y}) <-- needs to go RIGHT")

    dist_between = p_empathic.distance(my_x, my_y, other_x, other_y)
    print(f"\nDistance between robots: {dist_between:.2f}m")
    print(f"Collision threshold: {2 * p_empathic.robot_radius}m")

    # Check which actions are blocked
    print("\n" + "-" * 70)
    print("Action feasibility check (first move):")
    print("-" * 70)

    for action, name in [((-0.4, 0.0), "LEFT (toward goal)"),
                         ((0.4, 0.0), "RIGHT (yield/back up)"),
                         ((0.0, 0.4), "UP (lateral)"),
                         ((0.0, -0.4), "DOWN (lateral)"),
                         ((0.0, 0.0), "STAY")]:
        nx, ny = p_empathic.transition(my_x, my_y, action, other_x, other_y)
        blocked = "(BLOCKED)" if abs(nx - my_x) < 0.01 and abs(ny - my_y) < 0.01 and action != (0.0, 0.0) else ""
        print(f"  {name:<25} -> ({nx:.2f}, {ny:.2f}) {blocked}")

    print("\n" + "-" * 70)
    print(f"Comparing actions for EMPATHIC robot (G_social = G_self + {p_empathic.alpha}*G_other):")
    print("-" * 70)

    # Manually compute rollout for key actions showing G_self and G_other
    actions_to_test = [
        ((0.0, 0.0), "STAY"),
        ((0.4, 0.0), "YIELD (back up RIGHT)"),
        ((0.0, 0.4), "LATERAL (up)"),
        ((0.0, -0.4), "LATERAL (down)"),
        ((-0.4, 0.0), "FORWARD (toward goal LEFT)"),
    ]

    depth = 8
    results = []

    for action, name in actions_to_test:
        # Simulate rollout manually to show G_self and G_other separately
        mx, my_pos = my_x, my_y
        ox, oy = other_x, other_y

        total_g_self = 0.0
        total_g_other = 0.0
        discount = 1.0

        trajectory = []

        for step in range(depth):
            if step == 0:
                my_action = action
            else:
                my_action = p_empathic._greedy_action(mx, my_pos, p_empathic.goal_x, p_empathic.goal_y, ox, oy)

            # I move
            mx_new, my_new = p_empathic.transition(mx, my_pos, my_action, ox, oy)

            # My EFE contribution
            g_self_step = -p_empathic.preference(mx_new, my_new, p_empathic.goal_x, p_empathic.goal_y)
            total_g_self += discount * g_self_step

            # Other responds
            other_action = p_empathic.predict_other_action(
                ox, oy, other_goal_x, other_goal_y, other_alpha, mx_new, my_new
            )
            ox_new, oy_new = p_empathic.transition(ox, oy, other_action, mx_new, my_new)

            # Other's EFE contribution
            g_other_step = -p_empathic.preference(ox_new, oy_new, other_goal_x, other_goal_y)
            total_g_other += discount * g_other_step

            trajectory.append((mx_new, my_new, ox_new, oy_new, p_empathic._action_name(my_action)))

            mx, my_pos = mx_new, my_new
            ox, oy = ox_new, oy_new
            discount *= p_empathic.discount

        g_social = total_g_self + p_empathic.alpha * total_g_other
        results.append((name, action, total_g_self, total_g_other, g_social, trajectory))

    # Sort by social EFE (best first)
    results.sort(key=lambda x: x[4])

    print(f"\n{'Action':<25} {'G_self':>10} {'G_other':>10} {'alpha*G_other':>14} {'G_social':>12}")
    print("-" * 75)

    for name, action, g_self, g_other, g_social, traj in results:
        alpha_g_other = p_empathic.alpha * g_other
        best_marker = " <-- BEST" if g_social == results[0][4] else ""
        print(f"{name:<25} {g_self:>10.2f} {g_other:>10.2f} {alpha_g_other:>14.2f} {g_social:>12.2f}{best_marker}")

    # Show trajectory for best and worst
    print("\n" + "-" * 70)
    print("Trajectory comparison (first 5 steps):")
    print("-" * 70)

    best_result = results[0]
    stay_result = next((r for r in results if r[0] == "STAY"), results[-1])

    print(f"\nBEST ({best_result[0]}):")
    print(f"  Step | My position    | Other position | My->goal | Other->goal | My action")
    for i, (mx, my_pos, ox, oy, act) in enumerate(best_result[5][:5]):
        my_dist = p_empathic.distance(mx, my_pos, p_empathic.goal_x, p_empathic.goal_y)
        other_dist = p_empathic.distance(ox, oy, other_goal_x, other_goal_y)
        print(f"  {i+1:4d} | ({mx:+.2f}, {my_pos:+.2f}) | ({ox:+.2f}, {oy:+.2f})  | {my_dist:8.2f} | {other_dist:11.2f} | {act}")

    print(f"\nSTAY:")
    print(f"  Step | My position    | Other position | My->goal | Other->goal | My action")
    for i, (mx, my_pos, ox, oy, act) in enumerate(stay_result[5][:5]):
        my_dist = p_empathic.distance(mx, my_pos, p_empathic.goal_x, p_empathic.goal_y)
        other_dist = p_empathic.distance(ox, oy, other_goal_x, other_goal_y)
        print(f"  {i+1:4d} | ({mx:+.2f}, {my_pos:+.2f}) | ({ox:+.2f}, {oy:+.2f})  | {my_dist:8.2f} | {other_dist:11.2f} | {act}")

    # Analysis
    print("\n" + "-" * 70)
    print("ANALYSIS:")
    print("-" * 70)

    best_name, _, best_g_self, best_g_other, best_g_social, _ = best_result
    stay_name, _, stay_g_self, stay_g_other, stay_g_social, _ = stay_result

    print(f"\nComparing {best_name} vs STAY:")
    print(f"  G_self difference:  {best_g_self - stay_g_self:+.2f} (positive = worse for me)")
    print(f"  G_other difference: {best_g_other - stay_g_other:+.2f} (negative = better for other)")
    print(f"  alpha * G_other diff: {p_empathic.alpha * (best_g_other - stay_g_other):+.2f}")
    print(f"  Net G_social diff: {best_g_social - stay_g_social:+.2f}")

    if best_g_social < stay_g_social:
        if best_g_self > stay_g_self:
            print(f"\n  ==> {best_name} WINS!")
            print(f"      Self-sacrifice (+{best_g_self - stay_g_self:.2f}) is outweighed by")
            print(f"      empathy benefit ({p_empathic.alpha:.1f} * {stay_g_other - best_g_other:.2f} = {p_empathic.alpha * (stay_g_other - best_g_other):.2f})")
        else:
            print(f"\n  ==> {best_name} WINS (better for both self and other!)")
    else:
        print(f"\n  ==> STAY wins - empathy benefit doesn't outweigh self-cost")


def test_webots_scenario():
    """Test with actual Webots starting positions to debug oscillation."""
    print("\n" + "=" * 70)
    print("WEBOTS SCENARIO TEST - Starting positions (-1.8, 0) and (1.8, 0)")
    print("=" * 70)

    # Match Webots world file
    p1 = ToMPlanner(agent_id=0, goal_x=1.8, goal_y=0.0, alpha=0.0)   # Selfish
    p2 = ToMPlanner(agent_id=1, goal_x=-1.8, goal_y=0.0, alpha=6.0)  # Empathic

    print(f"Robot 1: alpha={p1.alpha} (selfish), goal=({p1.goal_x}, {p1.goal_y})")
    print(f"Robot 2: alpha={p2.alpha} (empathic), goal=({p2.goal_x}, {p2.goal_y})")
    print(f"Bounds: X=[{p1.x_min}, {p1.x_max}], Y=[{p1.y_min}, {p1.y_max}]")
    print()

    r1_x, r1_y = -1.8, 0.0
    r2_x, r2_y = 1.8, 0.0

    for step in range(20):
        t1_x, t1_y, info1 = p1.plan(r1_x, r1_y, r2_x, r2_y, p2.goal_x, p2.goal_y, p2.alpha)
        t2_x, t2_y, info2 = p2.plan(r2_x, r2_y, r1_x, r1_y, p1.goal_x, p1.goal_y, p1.alpha)

        print(f"{step+1:2d}. R1({r1_x:+.2f},{r1_y:+.2f})->({t1_x:+.2f},{t1_y:+.2f}) {info1}")
        print(f"    R2({r2_x:+.2f},{r2_y:+.2f})->({t2_x:+.2f},{t2_y:+.2f}) {info2}")

        # Check for oscillation (robot not making X progress toward goal)
        r2_moved_x = abs(t2_x - r2_x) > 0.1
        r2_moved_y = abs(t2_y - r2_y) > 0.1
        if r2_moved_y and not r2_moved_x:
            print(f"    ^^^ R2 LATERAL ONLY (possible oscillation)")

        r1_x, r1_y = t1_x, t1_y
        r2_x, r2_y = t2_x, t2_y

        # Check goals
        if p1.distance(r1_x, r1_y, p1.goal_x, p1.goal_y) < 0.3:
            print(f"\n*** Robot 1 reached goal! ***")
        if p2.distance(r2_x, r2_y, p2.goal_x, p2.goal_y) < 0.3:
            print(f"\n*** Robot 2 reached goal! ***")
        if p1.distance(r1_x, r1_y, p1.goal_x, p1.goal_y) < 0.3 and \
           p2.distance(r2_x, r2_y, p2.goal_x, p2.goal_y) < 0.3:
            print("\nBoth reached goals!")
            break
        print()


if __name__ == "__main__":
    # Run the Webots scenario test first to debug oscillation
    test_webots_scenario()

    # Uncomment for other tests:
    # debug_wall_situation()
    # test_empathy_yielding()
    # test_planner()
