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

    def __post_init__(self):
        if self.height != 3:
            raise ValueError("LavaModel currently only supports height=3 (lava-safe-lava design)")

        if self.goal_x is None:
            self.goal_x = self.width - 1

        self.safe_y = 1  # Middle row is safe
        self.num_states = self.width * self.height
        self.num_obs = self.num_states

        # Build generative model components
        self.A = self._build_A()
        self.B = self._build_B()
        self.C = self._build_C()
        self.D = self._build_D()

    def _build_A(self):
        """Build observation model - fully observable."""
        A_pos = jnp.eye(self.num_obs, self.num_states)
        return {"location_obs": A_pos}

    def _build_B(self):
        """Build transition model - grid navigation with 5 actions."""
        num_states = self.num_states
        num_actions = 5  # UP, DOWN, LEFT, RIGHT, STAY
        B = np.zeros((num_states, num_states, num_actions))

        def pos_to_idx(x, y):
            """Convert (x,y) coordinates to flat state index."""
            return y * self.width + x

        def idx_to_pos(idx):
            """Convert flat state index to (x,y) coordinates."""
            y = idx // self.width
            x = idx % self.width
            return x, y

        # Define transitions for each state and action
        for s_from in range(num_states):
            x, y = idx_to_pos(s_from)

            for action in range(num_actions):
                # Default: stay in place
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
                B[s_next, s_from, action] = 1.0

        return {"location_state": jnp.array(B)}

    def _build_C(self):
        """Build preference model - goal positive, lava negative."""
        C = np.zeros(self.num_obs)

        for s in range(self.num_states):
            y = s // self.width
            x = s % self.width

            if y != self.safe_y:
                # Lava: very low preference
                C[s] = -10.0
            elif x == self.goal_x and y == self.safe_y:
                # Goal: high preference
                C[s] = 10.0
            else:
                # Safe corridor: neutral
                C[s] = 0.0

        return {"location_obs": jnp.array(C)}

    def _build_D(self):
        """Build prior over initial state - start at (0, safe_y)."""
        D = np.zeros(self.num_states)
        start_idx = self.safe_y * self.width + 0  # pos_to_idx(0, safe_y)
        D[start_idx] = 1.0
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

        For Phase 1, we use repeated primitive actions rather than
        enumerating all sequences (which explodes for long horizons).

        Returns
        -------
        policies : jnp.ndarray
            Shape (num_policies, horizon, num_state_factors)
            Each policy repeats a single action for the full horizon
        """
        num_actions = 5  # UP, DOWN, LEFT, RIGHT, STAY
        num_state_factors = 1  # Only location_state

        if self.horizon == 1:
            # Single timestep: 5 policies
            policies = jnp.arange(num_actions)[:, None, None]
        else:
            # Multi-timestep: repeat each primitive action for full horizon
            # This gives 5 policies total (not 5^H)
            policies_list = []
            for action in range(num_actions):
                # Repeat this action for all timesteps
                policy = jnp.full((self.horizon, num_state_factors), action, dtype=jnp.int32)
                policies_list.append(policy)

            policies = jnp.stack(policies_list, axis=0)  # (5, horizon, 1)

        return policies
