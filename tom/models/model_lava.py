"""
LavaCorridor TOM model - pure JAX, TOM-style.

This defines the generative model for a lava corridor environment where:
- Agents navigate a grid with lava cells
- Goal is to reach the rightmost safe cell
- Agents must avoid lava and (softly) avoid collisions

Key differences from PyMDP:
- LavaModel: dataclass with dict-structured A, B, C, D (pure JAX)
- LavaAgent: thin wrapper that exposes model dicts + policies
- No PyMDP Agent.infer_states - use explicit Bayesian updates

Regime B (used here):
- Transition model B encodes simple grid physics (no hard occupancy constraint)
- Collisions are physically possible
- Collision / proximity aversion is encoded in preferences C and can be
  incorporated into G via relational preferences.
"""

import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
import itertools 


@dataclass
class LavaModel:
    """
    Pure JAX generative model for lava corridor.

    The corridor has:
    - Width x Height grid
    - Safe row at y=1 (middle row)
    - Lava at y=0 and y=2 (for height=3)
    - Goal at rightmost position of safe row

    Attributes
    ----------
    width : int
        Grid width
    height : int
        Grid height (must be 3 for classic lava-safe-lava design)
    goal_x : int
        X-coordinate of goal
    goal_y : int
        Y-coordinate of goal
    safe_cells : list
        List of (x,y) tuples that are safe (no lava)
    start_pos : tuple
        Starting position (x, y) for this agent's prior
    A : dict
        Observation model with modalities:
        - "location_obs": Own position (num_states observations)
        - "edge_obs": Which edge traversing (num_edges + 1 observations)
        - "cell_collision_obs": Binary {no_collision, collision}
        - "edge_collision_obs": Binary {no_collision, collision}
    B : dict
        Transition model {"location_state": array}
    C : dict
        Preference model with same keys as A
    D : dict
        Prior over initial state {"location_state": array}
    num_edges : int
        Total number of edges in the grid (computed automatically)
    """
    width: int = 4
    height: int = 3
    goal_x: int | None = None
    goal_y: int | None = None
    safe_cells: list | None = None  # Optional: List of (x,y) tuples that are safe
    start_pos: tuple | None = None  # Optional: Starting position for this agent
    num_empathy_levels: int = 3  # Number of discrete empathy states (for inferring other's empathy)

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

        # Discrete empathy levels for modeling beliefs about other agent's empathy
        # Maps from index to actual empathy value: e.g., [0.0, 0.5, 1.0] for 3 levels
        self.empathy_levels = np.linspace(0.0, 1.0, self.num_empathy_levels)

        # Build edge mappings for edge collision detection
        self._build_edge_mappings()

        # Build generative model components
        self.A = self._build_A()
        self.B = self._build_B()
        self.C = self._build_C()
        self.D = self._build_D()

    def _build_edge_mappings(self):
        """
        Build edge index mappings for edge collision detection.

        Each edge connects two adjacent cells. Edges are undirected (bidirectional).
        We assign each edge a unique index.

        Edge categories:
        - Horizontal edges: connect cells (x, y) and (x+1, y)
        - Vertical edges: connect cells (x, y) and (x, y+1)

        Total edges = (width-1)*height + width*(height-1)

        Creates:
        - self.num_edges: Total number of edges
        - self.edge_to_idx: Dict mapping (pos1, pos2) -> edge_idx (undirected)
        - self.idx_to_edge: Dict mapping edge_idx -> (pos1, pos2)
        - self.state_action_to_edge: Dict mapping (state, action) -> edge_idx or None
        """
        self.edge_to_idx = {}
        self.idx_to_edge = {}
        edge_idx = 0

        # Horizontal edges (left-right)
        for y in range(self.height):
            for x in range(self.width - 1):
                pos1 = (x, y)
                pos2 = (x + 1, y)
                # Store both directions mapping to same edge index
                self.edge_to_idx[(pos1, pos2)] = edge_idx
                self.edge_to_idx[(pos2, pos1)] = edge_idx
                self.idx_to_edge[edge_idx] = (pos1, pos2)
                edge_idx += 1

        # Vertical edges (up-down)
        for y in range(self.height - 1):
            for x in range(self.width):
                pos1 = (x, y)
                pos2 = (x, y + 1)
                # Store both directions mapping to same edge index
                self.edge_to_idx[(pos1, pos2)] = edge_idx
                self.edge_to_idx[(pos2, pos1)] = edge_idx
                self.idx_to_edge[edge_idx] = (pos1, pos2)
                edge_idx += 1

        self.num_edges = edge_idx

        # Build mapping from (state_idx, action) -> edge_idx
        # This tells us which edge is traversed when taking an action from a state
        self.state_action_to_edge = {}

        def pos_to_idx(x, y):
            return y * self.width + x

        def idx_to_pos(idx):
            y = idx // self.width
            x = idx % self.width
            return x, y

        for state in range(self.num_states):
            x, y = idx_to_pos(state)
            pos_from = (x, y)

            # For each action, determine which edge is traversed
            for action in range(5):  # UP, DOWN, LEFT, RIGHT, STAY
                x_to, y_to = x, y

                if action == 0:  # UP
                    y_to = max(0, y - 1)
                elif action == 1:  # DOWN
                    y_to = min(self.height - 1, y + 1)
                elif action == 2:  # LEFT
                    x_to = max(0, x - 1)
                elif action == 3:  # RIGHT
                    x_to = min(self.width - 1, x + 1)
                elif action == 4:  # STAY
                    # STAY action doesn't traverse any edge
                    self.state_action_to_edge[(state, action)] = None
                    continue

                pos_to = (x_to, y_to)

                # If action doesn't move (e.g., UP at top edge), no edge traversed
                if pos_from == pos_to:
                    self.state_action_to_edge[(state, action)] = None
                else:
                    # Get edge index for this transition
                    edge_idx = self.edge_to_idx.get((pos_from, pos_to))
                    self.state_action_to_edge[(state, action)] = edge_idx

    def _build_A(self):
        """
        Build observation model.

        Observation modalities:
        1. location_obs: Agent's own position (fully observable)
           - Shape: (num_states, num_states) - identity matrix

        2. edge_obs: Which edge is being traversed by this agent
           - Shape: (num_edges + 1, num_states, num_actions)
           - Index num_edges means "no edge" (STAY or boundary bounce)
           - Maps (state, action) -> edge being traversed

        3. cell_collision_obs: Whether agents occupy same cell (binary)
           - Shape: (2, num_states, num_states) - maps from joint state
           - A[0, s_i, s_j] = 1 if s_i != s_j (no collision)
           - A[1, s_i, s_j] = 1 if s_i == s_j (collision)

        4. edge_collision_obs: Whether agents traverse same edge (binary)
           - Shape: (2, num_states, num_states, num_actions, num_actions)
           - A[0, s_i, s_j, a_i, a_j] = 1 if no edge collision
           - A[1, s_i, s_j, a_i, a_j] = 1 if both traverse same edge

        Note: Collision A matrices map from joint states/actions to observations.
        The planner marginalizes using beliefs over both agents' states.
        """
        # Own location: fully observable
        A_loc = jnp.eye(self.num_obs, self.num_states)

        # Edge observation: deterministic mapping from (state, action) -> edge
        num_actions = 5
        A_edge = np.zeros((self.num_edges + 1, self.num_states, num_actions))

        for state in range(self.num_states):
            for action in range(num_actions):
                edge_idx = self.state_action_to_edge.get((state, action))
                if edge_idx is None:
                    # No edge traversed (STAY or boundary bounce) -> maps to last index
                    A_edge[self.num_edges, state, action] = 1.0
                else:
                    A_edge[edge_idx, state, action] = 1.0

        # Cell collision observation model: (2, num_states, num_states)
        # A[o, s_self, s_other] where o ∈ {no_collision, collision}
        A_cell_collision = np.zeros((2, self.num_states, self.num_states))
        for s_i in range(self.num_states):
            for s_j in range(self.num_states):
                if s_i == s_j:
                    A_cell_collision[1, s_i, s_j] = 1.0  # Collision
                else:
                    A_cell_collision[0, s_i, s_j] = 1.0  # No collision

        # Edge collision observation model: (2, num_states, num_states, num_actions, num_actions)
        # A[o, s_self, s_other, a_self, a_other] where o ∈ {no_edge_collision, edge_collision}
        A_edge_collision = np.zeros((2, self.num_states, self.num_states, num_actions, num_actions))

        for s_i in range(self.num_states):
            for s_j in range(self.num_states):
                for a_i in range(num_actions):
                    for a_j in range(num_actions):
                        # Get edges traversed by each agent
                        edge_i = self.state_action_to_edge.get((s_i, a_i))
                        edge_j = self.state_action_to_edge.get((s_j, a_j))

                        # Edge collision if both traverse same edge (and not None)
                        if edge_i is not None and edge_j is not None and edge_i == edge_j:
                            A_edge_collision[1, s_i, s_j, a_i, a_j] = 1.0  # Edge collision
                        else:
                            A_edge_collision[0, s_i, s_j, a_i, a_j] = 1.0  # No edge collision

        # Empathy observation model: identity matrix (perfect observation for now)
        # Maps from hidden state (other agent's true empathy) to observation
        # Shape: (num_empathy_levels, num_empathy_levels)
        # Later this could be made noisy to model imperfect empathy inference
        A_empathy = jnp.eye(self.num_empathy_levels)

        return {
            "location_obs": A_loc,
            "edge_obs": jnp.array(A_edge),
            "cell_collision_obs": jnp.array(A_cell_collision),
            "edge_collision_obs": jnp.array(A_edge_collision),
            "empathy_obs": A_empathy,
        }

    def _build_B(self):
        """
        Build transition model - grid navigation with 5 actions.

        Multi-agent version: B conditions on other agent's position only in shape,
        but NOT in physics. Under Regime B:

        - B encodes single-agent grid dynamics:
            UP, DOWN, LEFT, RIGHT, STAY
        - Collisions are physically possible (no hard occupancy constraint).
        - Collision avoidance is handled via preferences C and planning,
          not by blocking transitions.

        Shape: B[s_next, s_current, s_other, action], but the s_other dimension
        simply replicates the same single-agent dynamics.

        Returns
        -------
        B : dict
            {"location_state": array of shape (num_states, num_states, num_states, num_actions)}
        """
        num_states = self.num_states
        num_actions = 5  # UP, DOWN, LEFT, RIGHT, STAY

        def pos_to_idx(x, y):
            """Convert (x,y) coordinates to flat state index."""
            return y * self.width + x

        def idx_to_pos(idx):
            """Convert flat state index to (x,y) coordinates."""
            y = idx // self.width
            x = idx % self.width
            return x, y

        # First build a 3D single-agent transition tensor:
        # B_single[s_next, s_current, action]
        B_single = np.zeros((num_states, num_states, num_actions))

        for s_from in range(num_states):
            x, y = idx_to_pos(s_from)

            for action in range(num_actions):
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

                s_next = pos_to_idx(x_next, y_next)
                B_single[s_next, s_from, action] = 1.0

        # Now expand to 4D: [next_state, current_state, other_agent_state, action]
        # The other_agent_state dimension does not change the dynamics; it simply
        # replicates the same physics for each possible position of the other agent.
        B = np.zeros((num_states, num_states, num_states, num_actions))
        for s_other in range(num_states):
            B[:, :, s_other, :] = B_single

        return {"location_state": jnp.array(B)}

    def _build_C(self):
        """
        Build preference model - goal positive, lava catastrophic, collisions avoided.

        Preference modalities:
        1. location_obs (own position):
           - Goal: High reward (+50)
           - Lava: CATASTROPHIC penalty (-100)
           - Safe cells: distance shaping + small time cost

        2. edge_obs (which edge traversing):
           - All edges: neutral (no inherent preference for specific edges)

        3. cell_collision_obs (binary):
           - Index 0 (no collision): neutral (0)
           - Index 1 (collision): CATASTROPHIC penalty (-100)

        4. edge_collision_obs (binary):
           - Index 0 (no edge collision): neutral (0)
           - Index 1 (edge collision/swap): CATASTROPHIC penalty (-100)
        """
        # Preferences over own location
        C_self = np.zeros(self.num_obs)

        # GOAL REWARD: Very strong preference for reaching exact goal
        # Increased from 50 to make goal more appealing relative to collision costs (-30)
        goal_reward = 80.0

        # Distance-based shaping: guides agent towards goal when unreachable within horizon
        # This is CRITICAL - without it, agents can't find the goal if it's >horizon steps away
        lambda_dist = 2.0  # Restored (was 0.5, increased to 2.0 for stronger gradient)

        # Time cost (small penalty per timestep for not being at goal)
        time_cost = 0.1

        # Lava penalty
        lava_penalty = -100.0

        for s in range(self.num_states):
            y = s // self.width
            x = s % self.width
            pos = (x, y)

            if pos not in self.safe_cells_set:
                # Lava: catastrophic
                C_self[s] = lava_penalty
            elif x == self.goal_x and y == self.goal_y:
                # Goal: STRONG preference (no time cost, no distance shaping)
                # This is the ONLY cell with large positive reward
                C_self[s] = goal_reward
            else:
                # Safe corridor: distance shaping + time cost
                # Distance shaping creates gradient towards goal (essential when goal > horizon steps away)
                # The gradient is lane-neutral - only Manhattan distance to goal matters
                manhattan_dist = abs(x - self.goal_x) + abs(y - self.goal_y)
                C_self[s] = -lambda_dist * manhattan_dist - time_cost

        # Preferences over edge traversal
        # Neutral for all edges (no preference for which edge to use)
        C_edge = np.zeros(self.num_edges + 1)

        # Preferences over cell collision (binary)
        # Penalty reduced from -100 to -30 so that collision is costly but not
        # catastrophic - a selfish agent may still collide if goal reward (+50)
        # outweighs the collision cost (-30). Empathic agents will weight the
        # other agent's collision cost too, making them more cautious.
        C_cell_collision = np.array([
            0.0,    # Index 0: no cell collision - neutral
            -30.0   # Index 1: cell collision - significant but not catastrophic
        ])

        # Preferences over edge collision (binary)
        C_edge_collision = np.array([
            0.0,    # Index 0: no edge collision - neutral
            -30.0   # Index 1: edge collision (swap) - significant but not catastrophic
        ])

        # Preferences over other agent's empathy (neutral - no preference for particular level)
        C_empathy = np.zeros(self.num_empathy_levels)

        return {
            "location_obs": jnp.array(C_self),
            "edge_obs": jnp.array(C_edge),
            "cell_collision_obs": jnp.array(C_cell_collision),
            "edge_collision_obs": jnp.array(C_edge_collision),
            "empathy_obs": jnp.array(C_empathy),
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

        For horizon=1:
            - Creates one policy per primitive action (5 policies).

        For horizon>1:
            - Enumerates ALL possible sequences of primitive actions of length `horizon`
            (pymdp-style), i.e. all combinations in {0..4}^horizon.

        This ensures that agents can represent:
            - STAY-then-GO
            - GO-then-WAIT
            - detours and probes
            - simple turn-taking and coordination patterns.

        Returns
        -------
        policies : jnp.ndarray
            Shape (num_policies, horizon, num_state_factors)
        """
        num_actions = 5  # UP, DOWN, LEFT, RIGHT, STAY
        num_state_factors = 1  # Only location_state

        # Horizon 1: simple case, one policy per action
        if self.horizon == 1:
            policies = jnp.zeros((num_actions, 1, num_state_factors), dtype=jnp.int32)
            for a in range(num_actions):
                policies = policies.at[a, 0, 0].set(a)
            return policies

        # Horizon > 1: enumerate all action sequences of length horizon
        # This matches pymdp's "all policies up to horizon" idea:
        #   number of policies = num_actions ** horizon
        policies_list = []

        for action_seq in itertools.product(range(num_actions), repeat=self.horizon):
            # action_seq is a tuple like (a0, a1, ..., a_{H-1})
            policy = np.array(action_seq, dtype=np.int32).reshape(self.horizon, num_state_factors)
            policies_list.append(policy)

        policies = jnp.array(policies_list, dtype=jnp.int32)
        return policies

