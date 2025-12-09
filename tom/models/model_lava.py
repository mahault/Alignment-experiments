"""
LavaCorridor TOM model - pure JAX, TOM-style.

This defines the generative model for a lava corridor environment where:
- Agents navigate a grid with lava cells
- Goal is to reach the rightmost safe cell
- Agents must avoid lava and coordinate to prevent collisions

Key difference from PyMDP:
- LavaModel: dataclass with dict-structured A, B, C, D (pure JAX)
- LavaAgent: thin wrapper that exposes model dicts + policies
- No PyMDP Agent.infer_states - use explicit Bayesian updates
"""

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass


@dataclass
class LavaModel:
    """
    Pure JAX generative model for lava corridor.

    The corridor has:
    - Width x Height grid
    - Safe row at y=1 (middle row)
    - Lava at y=0 and y=2
    - Goal at rightmost position of safe row

    Attributes
    ----------
    width : int
        Grid width
    height : int
        Grid height (must be 3 for lava-safe-lava design)
    goal_x : int
        X-coordinate of goal
    start_pos : tuple
        Starting position (x, y) for this agent's prior
    A : dict
        Observation model {"location_obs": array}
    B : dict
        Transition model {"location_state": array}
    C : dict
        Preference model {"location_obs": array}
    D : dict
        Prior over initial state {"location_state": array}
    """
    width: int = 4
    height: int = 3
    goal_x: int = None
    goal_y: int = None
    safe_cells: list = None  # Optional: List of (x,y) tuples that are safe
    start_pos: tuple = None  # Optional: Starting position for this agent

    def __post_init__(self):
        if self.goal_x is None:
            self.goal_x = self.width - 1

        if self.goal_y is None:
            self.goal_y = 1  # Default: middle row for height=3

        # If safe_cells not provided, use default lava-safe-lava for height=3
        if self.safe_cells is None:
            if self.height == 3:
                # Classic design: row 1 is safe, rows 0 and 2 are lava
                self.safe_y = 1
                self.safe_cells = [(x, self.safe_y) for x in range(self.width)]
            else:
                # For other heights, make all cells safe (can override with safe_cells parameter)
                self.safe_cells = [(x, y) for x in range(self.width) for y in range(self.height)]
                self.safe_y = 1  # Set to 1 for backwards compatibility
        else:
            # If safe_cells provided, set safe_y to first safe row
            safe_y_values = sorted(set(y for x, y in self.safe_cells))
            self.safe_y = safe_y_values[0] if safe_y_values else 1

        self.safe_cells_set = set(self.safe_cells)
        self.num_states = self.width * self.height
        self.num_obs = self.num_states

        # Build generative model components
        self.A = self._build_A()
        self.B = self._build_B()
        self.C = self._build_C()
        self.D = self._build_D()

    def _build_A(self):
        """
        Build observation model - fully observable.

        Three observation modalities:
        1. location_obs: Agent's own position (num_states,)
        2. other_location_obs: Other agent's position (num_states,)
        3. relation_obs: Relational state (3 categories):
           - 0: Different rows
           - 1: Same row, different cells
           - 2: Same cell (collision)

        For relation_obs, we need to map joint state (s_i, s_j) → relation category.
        A_relation shape: (3, num_states_i, num_states_j) or simplified to (3, num_states)

        Since this is a generative model for a single agent and we observe other's
        position separately, we'll build relation_obs dynamically during inference.
        For now, keep it as identity mapping.
        """
        A_self = jnp.eye(self.num_obs, self.num_states)
        A_other = jnp.eye(self.num_obs, self.num_states)  # Also fully observable

        # For now, relation_obs uses simplified 1D mapping
        # Will be properly computed during inference based on both positions
        A_relation = jnp.zeros((3, self.num_states))

        return {
            "location_obs": A_self,
            "other_location_obs": A_other,
            "relation_obs": A_relation  # Placeholder, computed during inference
        }

    def _build_B(self):
        """
        Build transition model - grid navigation with 5 actions.

        Multi-agent version: B conditions on other agent's position.
        Shape: B[s_next, s_current, s_other, action]

        Implements single-occupancy constraint:
        - If action would move agent to s_other's current position, stay in place
        - This prevents collisions and edge swaps naturally

        Returns
        -------
        B : dict
            {"location_state": array of shape (num_states, num_states, num_states, num_actions)}
        """
        num_states = self.num_states
        num_actions = 5  # UP, DOWN, LEFT, RIGHT, STAY

        # 4D tensor: [next_state, current_state, other_agent_state, action]
        B = np.zeros((num_states, num_states, num_states, num_actions))

        def pos_to_idx(x, y):
            """Convert (x,y) coordinates to flat state index."""
            return y * self.width + x

        def idx_to_pos(idx):
            """Convert flat state index to (x,y) coordinates."""
            y = idx // self.width
            x = idx % self.width
            return x, y

        # Define transitions for each state, other agent position, and action
        for s_from in range(num_states):
            x, y = idx_to_pos(s_from)

            for s_other in range(num_states):
                # Get other agent's position
                x_other, y_other = idx_to_pos(s_other)

                for action in range(num_actions):
                    # Compute intended next position
                    x_next, y_next = x, y

                    if action == 0:  # UP
                        y_next = max(0, y - 1)
                    elif action == 1:  # DOWN
                        y_next = min(self.height - 1, y + 1)
                    elif action == 2:  # LEFT
                        x_next = max(0, x - 1)
                    elif action == 3:  # RIGHT
                        x_next = min(self.width - 1, x + 1)
                    elif action == 4:  # STAY
                        pass  # stay at (x, y)

                    s_next_intended = pos_to_idx(x_next, y_next)

                    # Check single-occupancy constraint: can't move into other agent's cell
                    if (x_next, y_next) == (x_other, y_other):
                        # Move blocked: stay in current position
                        B[s_from, s_from, s_other, action] = 1.0
                    else:
                        # Move allowed: transition to intended next state
                        B[s_next_intended, s_from, s_other, action] = 1.0

        return {"location_state": jnp.array(B)}

    def _build_C(self):
        """
        Build preference model - goal positive, lava CATASTROPHIC, collision avoided.

        Two preference modalities:
        1. location_obs (own position):
           - Goal: High reward (+10)
           - Lava: CATASTROPHIC penalty (-100)
           - Safe cells: Distance shaping + small time cost

        2. other_location_obs (other agent's position):
           - Same cell: CATASTROPHIC penalty (-100, collision)
           - Adjacent cell: Mild penalty (-2, encourage spacing)
           - Same row in narrow corridors: Mild penalty (-1, encourage turn-taking)
        """
        # Preferences over own location
        C_self = np.zeros(self.num_obs)

        # Distance-based shaping weight (0 = no shaping, >0 = reward proximity)
        lambda_dist = 0.5  # Small shaping to encourage progress toward goal

        # Time cost (small penalty per timestep for not being at goal)
        time_cost = 0.1  # Encourages reaching goal faster

        # Lava penalty (CATASTROPHIC - makes lava absolutely unacceptable)
        lava_penalty = -100.0

        for s in range(self.num_states):
            y = s // self.width
            x = s % self.width
            pos = (x, y)

            if pos not in self.safe_cells_set:
                # Lava: CATASTROPHIC penalty - never worth risking
                C_self[s] = lava_penalty
            elif x == self.goal_x and y == self.goal_y:
                # Goal: high preference (no time cost here)
                C_self[s] = 10.0
            else:
                # Safe corridor: small shaping based on distance to goal
                manhattan_dist = abs(x - self.goal_x) + abs(y - self.goal_y)

                # Reward getting closer to goal + small time penalty for stalling
                C_self[s] = -lambda_dist * manhattan_dist - time_cost

        # Preferences over other agent's location (collision/proximity avoidance)
        C_other = np.zeros(self.num_obs)

        for s_other in range(self.num_states):
            y_other = s_other // self.width
            x_other = s_other % self.width

            # For each possible position of other agent, compute preference
            # This will be combined with own position during inference
            # to detect collision/proximity

            # Default: neutral (other agent far away is fine)
            C_other[s_other] = 0.0

            # Note: Actual collision detection requires comparing both positions
            # during action selection. These are baseline preferences.
            # Specific collision penalty is computed in a separate observable
            # or handled via the B matrix blocking invalid transitions.

        # For now, keep C_other neutral since collision is prevented by B matrix
        # Can be extended to add proximity penalties if needed

        # Preferences over relational state
        C_relation = np.zeros(3)
        C_relation[0] = 0.0    # Different rows: neutral
        C_relation[1] = -1.0   # Same row: mild penalty (encourages turn-taking in narrow spaces)
        C_relation[2] = -100.0  # Same cell (collision): CATASTROPHIC penalty

        return {
            "location_obs": jnp.array(C_self),
            "other_location_obs": jnp.array(C_other),
            "relation_obs": jnp.array(C_relation)
        }

    def _build_D(self):
        """Build prior over initial state - concentrated at agent's starting position."""
        D = np.zeros(self.num_states)

        # Use specified start_pos if provided, otherwise use first safe cell
        if self.start_pos is not None:
            start_x, start_y = self.start_pos
            start_idx = start_y * self.width + start_x
            D[start_idx] = 1.0
        elif self.safe_cells:
            start_x, start_y = self.safe_cells[0]  # First safe cell
            start_idx = start_y * self.width + start_x
            D[start_idx] = 1.0
        else:
            # Fallback: uniform over all states (shouldn't happen)
            D = np.ones(self.num_states) / self.num_states

        return {"location_state": jnp.array(D)}


@dataclass
class LavaAgent:
    """
    TOM-style agent for lava corridor - thin wrapper around model.

    This agent:
    - Exposes model's A, B, C, D as dicts (not PyMDP lists)
    - Defines a simple policy set (5 primitive actions)
    - Does NOT use PyMDP's infer_states/infer_policies

    Attributes
    ----------
    model : LavaModel
        Generative model
    horizon : int
        Planning horizon
    gamma : float
        Inverse temperature for action selection
    policies : jnp.ndarray
        Policy set, shape (num_policies, horizon, num_state_factors)
    """
    model: LavaModel
    horizon: int = 1
    gamma: float = 8.0

    def __post_init__(self):
        # Expose model dicts directly
        self.A = self.model.A
        self.B = self.model.B
        self.C = self.model.C
        self.D = self.model.D

        # Build simple policy set: 5 primitive actions
        self.policies = self._build_policies()

    def _build_policies(self):
        """
        Build policy set for lava corridor.

        For horizon=1: Creates one policy per primitive action (5 policies)
        For horizon>1: Creates straight-line policies (repeat same action)
                       plus some mixed policies for flexibility

        Returns
        -------
        policies : jnp.ndarray
            Shape (num_policies, horizon, num_state_factors)
        """
        num_actions = 5  # UP, DOWN, LEFT, RIGHT, STAY
        num_state_factors = 1  # Only location_state

        if self.horizon == 1:
            # Simple case: one policy per primitive action
            policies = jnp.zeros((num_actions, 1, num_state_factors), dtype=jnp.int32)
            for i in range(num_actions):
                policies = policies.at[i, 0, 0].set(i)
        else:
            # Multi-step: create straight-line policies (repeat same action)
            # This gives agents ability to plan multi-step movements
            policies_list = []

            # Straight-line policies: repeat each action for full horizon
            for action in range(num_actions):
                policy = np.full((self.horizon, num_state_factors), action, dtype=np.int32)
                policies_list.append(policy)

            # Convert to array
            policies = jnp.array(policies_list, dtype=jnp.int32)

        return policies
