# src/envs/lava_corridor.py

"""
Lava Corridor Environment for Path Flexibility Experiments.

A simple 3-row gridworld:
- Row 0: Lava (instant death)
- Row 1: Safe corridor (agents must stay here)
- Row 2: Lava (instant death)

Two agents start at the left and must reach the goal on the right
while staying in the safe corridor and avoiding each other.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Any, List
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


# =============================================================================
# Action Constants
# =============================================================================

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

ACTIONS = [UP, DOWN, LEFT, RIGHT, STAY]
ACTION_NAMES = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
    STAY: "STAY",
}


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LavaCorridorConfig:
    """Configuration for Lava Corridor environment."""

    width: int = 7
    height: int = 3  # Fixed: 3 rows (lava-safe-lava)
    num_agents: int = 2

    # Optional: stochastic transitions (slip probability)
    slip_prob: float = 0.0

    # Initial positions (can be overridden)
    start_positions: Dict[int, Tuple[int, int]] = None

    def __post_init__(self):
        assert self.height == 3, "LavaCorridorEnv requires exactly 3 rows: lava-safe-lava"

        if self.start_positions is None:
            # Default: both agents start at left, safe row
            self.start_positions = {
                0: (0, 1),  # Agent 0: (x=0, y=1)
                1: (0, 1),  # Agent 1: (x=0, y=1)
            }

        LOGGER.info(f"LavaCorridorConfig: width={self.width}, num_agents={self.num_agents}, slip_prob={self.slip_prob}")
        LOGGER.info(f"  Start positions: {self.start_positions}")


# =============================================================================
# Lava Corridor Environment
# =============================================================================

class LavaCorridorEnv:
    """
    Lava Corridor Environment.

    Grid Layout:
        Row 0: LAVA 🔥🔥🔥🔥🔥🔥🔥
        Row 1: ⬜⬜⬜⬜⬜⬜🎯  (safe corridor + goal)
        Row 2: LAVA 🔥🔥🔥🔥🔥🔥🔥

    State:
        - pos[agent_id]: (x, y) position for each agent
        - t: timestep
        - done: episode terminated
        - lava_hit: True if any agent hit lava
        - success: True if all agents reached goal

    Observations:
        - Each agent observes its own (x, y) position
        - Mapped to discrete observation index via pos_to_obs_index()
    """

    def __init__(self, config: LavaCorridorConfig):
        """
        Initialize Lava Corridor environment.

        Parameters
        ----------
        config : LavaCorridorConfig
            Environment configuration
        """
        self.config = config

        # Environment dimensions
        self.width = config.width
        self.height = config.height
        self.num_agents = config.num_agents

        # Start and goal
        self.start_positions = config.start_positions
        self.goal_x = config.width - 1
        self.safe_y = 1  # Middle row is safe

        # State
        self.state = None

        LOGGER.info("=" * 80)
        LOGGER.info("Lava Corridor Environment Initialized")
        LOGGER.info("=" * 80)
        LOGGER.info(f"Grid: {self.width}x{self.height} (width x height)")
        LOGGER.info(f"Safe row: y={self.safe_y}")
        LOGGER.info(f"Goal: x={self.goal_x}, y={self.safe_y}")
        LOGGER.info(f"Num agents: {self.num_agents}")
        LOGGER.info(f"Start positions: {self.start_positions}")
        LOGGER.info(f"Slip probability: {self.config.slip_prob}")
        LOGGER.info("=" * 80)

    # =========================================================================
    # Core API
    # =========================================================================

    def reset(self, rng_key: Any = None) -> Tuple[Dict, Dict[int, Tuple[int, int]]]:
        """
        Reset environment to initial state.

        Parameters
        ----------
        rng_key : Any, optional
            Random key (unused for now, for JAX compatibility)

        Returns
        -------
        state : Dict
            Internal state dict with keys:
            - pos: {agent_id: (x, y)}
            - t: timestep
            - done: episode terminated
            - lava_hit: any agent hit lava
            - success: all agents reached goal
        obs : Dict[int, Tuple[int, int]]
            Per-agent observations (positions)
        """
        LOGGER.info("Resetting environment")

        self.state = {
            "pos": {aid: self.start_positions[aid] for aid in range(self.num_agents)},
            "t": 0,
            "done": False,
            "lava_hit": False,
            "success": False,
            "collision": False,
        }

        obs = {aid: self.state["pos"][aid] for aid in range(self.num_agents)}

        LOGGER.info(f"Reset complete: t=0")
        for aid in range(self.num_agents):
            pos = self.state["pos"][aid]
            LOGGER.info(f"  Agent {aid}: position={pos}, obs_idx={self.pos_to_obs_index(pos)}")

        return self.state, obs

    def step(
        self,
        state: Dict,
        actions: Dict[int, int],
        rng_key: Any = None,
    ) -> Tuple[Dict, Dict[int, Tuple[int, int]], bool, Dict]:
        """
        Take one environment step.

        Parameters
        ----------
        state : Dict
            Current state
        actions : Dict[int, int]
            Actions per agent {agent_id: action_index}
        rng_key : Any, optional
            Random key for stochastic transitions

        Returns
        -------
        new_state : Dict
            Updated state
        obs : Dict[int, Tuple[int, int]]
            Per-agent observations
        done : bool
            Episode terminated
        info : Dict
            Additional info (lava_hit, success, collision, etc.)
        """
        if state.get("done", False):
            LOGGER.debug("Episode already done, returning current state")
            obs = {aid: state["pos"][aid] for aid in range(self.num_agents)}
            return state, obs, True, state

        t = state["t"]
        LOGGER.debug(f"Step t={t}: actions={actions}")

        new_state = dict(state)
        new_pos = {}

        # 1. Move each agent independently
        for aid in range(self.num_agents):
            x, y = state["pos"][aid]
            a = actions.get(aid, STAY)

            old_pos = (x, y)

            # Apply action with deterministic physics
            if a == UP:
                y = max(0, y - 1)
            elif a == DOWN:
                y = min(self.height - 1, y + 1)
            elif a == LEFT:
                x = max(0, x - 1)
            elif a == RIGHT:
                x = min(self.width - 1, x + 1)
            elif a == STAY:
                pass
            else:
                LOGGER.warning(f"Agent {aid}: unknown action {a}, treating as STAY")

            # Apply slip (if slip_prob > 0 and rng_key provided)
            # TODO: Implement stochastic transitions if needed

            new_pos[aid] = (x, y)

            action_name = ACTION_NAMES.get(a, "UNKNOWN")
            LOGGER.debug(f"  Agent {aid}: {old_pos} --[{action_name}]--> {new_pos[aid]}")

        # 2. Check lava / collision / goal / done conditions
        lava_hit = False
        collision = False
        success = True

        # Check lava
        for aid, (x, y) in new_pos.items():
            if y != self.safe_y:  # Row 0 or 2 -> lava
                lava_hit = True
                LOGGER.warning(f"  Agent {aid} hit LAVA at {(x, y)}!")

        # Check collision (both agents in same cell)
        if self.num_agents == 2:
            pos_0 = new_pos[0]
            pos_1 = new_pos[1]
            if pos_0 == pos_1:
                collision = True
                LOGGER.warning(f"  COLLISION: Both agents at {pos_0}!")

        # Check goal (all agents must reach goal)
        for aid, (x, y) in new_pos.items():
            if not (x == self.goal_x and y == self.safe_y):
                success = False

        if success:
            LOGGER.info(f"  SUCCESS: All agents reached goal at t={t+1}!")

        # Update state
        new_state["pos"] = new_pos
        new_state["t"] = t + 1
        new_state["lava_hit"] = lava_hit
        new_state["collision"] = collision
        new_state["success"] = success
        new_state["done"] = bool(lava_hit or success)

        obs = {aid: new_pos[aid] for aid in range(self.num_agents)}

        info = {
            "lava_hit": lava_hit,
            "collision": collision,
            "success": success,
            "success_i": (new_pos[0][0] == self.goal_x and new_pos[0][1] == self.safe_y) if self.num_agents > 0 else False,
            "success_j": (new_pos[1][0] == self.goal_x and new_pos[1][1] == self.safe_y) if self.num_agents > 1 else False,
            "t": new_state["t"],
            "timesteps": new_state["t"],
        }

        LOGGER.debug(f"Step complete: t={new_state['t']}, done={new_state['done']}, info={info}")

        return new_state, obs, new_state["done"], info

    # =========================================================================
    # Shared Outcomes for Path Flexibility
    # =========================================================================

    def shared_outcomes(self) -> List[Tuple[int, int]]:
        """
        Return list of 'shared outcome' positions for returnability computation.

        Shared outcomes = safe corridor cells (y=1) that both agents can reach.

        Returns
        -------
        outcomes : List[Tuple[int, int]]
            List of (x, y) positions in safe corridor
        """
        outcomes = []
        for x in range(self.width):
            outcomes.append((x, self.safe_y))

        LOGGER.debug(f"Shared outcomes (positions): {outcomes}")
        return outcomes

    def pos_to_obs_index(self, pos: Tuple[int, int]) -> int:
        """
        Map (x, y) position to discrete observation index.

        Flattening: obs_idx = y * width + x

        Parameters
        ----------
        pos : Tuple[int, int]
            (x, y) position

        Returns
        -------
        obs_idx : int
            Observation index
        """
        x, y = pos
        obs_idx = y * self.width + x
        return obs_idx

    def obs_index_to_pos(self, obs_idx: int) -> Tuple[int, int]:
        """
        Map observation index to (x, y) position.

        Inverse of pos_to_obs_index.

        Parameters
        ----------
        obs_idx : int
            Observation index

        Returns
        -------
        pos : Tuple[int, int]
            (x, y) position
        """
        y = obs_idx // self.width
        x = obs_idx % self.width
        return (x, y)

    def shared_outcome_obs_indices(self) -> List[int]:
        """
        Return shared outcomes as observation indices (for generative model).

        Returns
        -------
        obs_indices : List[int]
            List of observation indices corresponding to safe corridor
        """
        positions = self.shared_outcomes()
        obs_indices = [self.pos_to_obs_index(p) for p in positions]

        LOGGER.debug(f"Shared outcomes (obs indices): {obs_indices}")
        return obs_indices

    # =========================================================================
    # Utilities
    # =========================================================================

    def render(self, state: Dict = None) -> str:
        """
        Render environment state as ASCII art.

        Parameters
        ----------
        state : Dict, optional
            State to render (default: self.state)

        Returns
        -------
        render_str : str
            ASCII representation
        """
        if state is None:
            state = self.state

        if state is None:
            return "Environment not initialized (call reset first)"

        grid = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                cell = "."

                # Mark lava
                if y != self.safe_y:
                    cell = "🔥"

                # Mark goal
                if x == self.goal_x and y == self.safe_y:
                    cell = "🎯"

                # Mark agents
                for aid in range(self.num_agents):
                    if state["pos"][aid] == (x, y):
                        cell = str(aid)

                row.append(cell)
            grid.append(" ".join(row))

        header = f"t={state['t']} | Lava: {state['lava_hit']} | Success: {state['success']}"
        return header + "\n" + "\n".join(grid)

    def get_num_states(self) -> int:
        """Total number of states (width * height)."""
        return self.width * self.height

    def get_num_observations(self) -> int:
        """Total number of observations (same as states for fully observable)."""
        return self.get_num_states()

    def get_num_actions(self) -> int:
        """Number of actions per agent."""
        return len(ACTIONS)


# =============================================================================
# Generative Model Builder
# =============================================================================

def sanitize_B(B_raw: np.ndarray) -> np.ndarray:
    """
    Ensure B_raw has no zero-sum distributions along axis=1,
    and that everything along axis=1 is normalized.

    This matches pymdp.utils.validate_normalization(self.B[f], axis=1).

    Parameters
    ----------
    B_raw : np.ndarray
        Transition tensor, typically shape [num_states, num_states, num_actions]
        or [num_actions, num_states, num_states]

    Returns
    -------
    B : np.ndarray
        Sanitized and normalized transition tensor
    """
    B = B_raw.astype(float).copy()

    # Sum along axis=1 (the axis pymdp validates)
    sums = B.sum(axis=1, keepdims=True)  # shape broadcastable to B

    # Identify slices where the sum along axis=1 is zero
    zero_mask = np.isclose(sums, 0.0)

    if np.any(zero_mask):
        # Replace zero-sum slices with uniform along axis=1
        # B has shape (..., axis=1, ...). We want a uniform over axis=1.
        uniform_value = 1.0 / B.shape[1]
        B = np.where(zero_mask, uniform_value, B)
        LOGGER.warning(
            f"Found {zero_mask.sum()} zero-sum slices in B matrix along axis=1. "
            f"Replacing with uniform distribution."
        )

    # Renormalize along axis=1 so each distribution sums to 1
    sums = B.sum(axis=1, keepdims=True)
    B /= np.where(sums == 0.0, 1.0, sums)

    return B


def build_generative_model_for_env(
    env: LavaCorridorEnv,
    agent_id: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Build a generative model (A, B, C, D) consistent with LavaCorridorEnv.

    This creates a simple single-agent model where:
    - States: (x, y) positions flattened to s = y * width + x
    - Observations: same as states (fully observable)
    - Actions: UP, DOWN, LEFT, RIGHT, STAY
    - Preferences: high for goal, neutral for safe corridor, low for lava

    Parameters
    ----------
    env : LavaCorridorEnv
        Environment instance
    agent_id : int
        Agent ID (for logging purposes)

    Returns
    -------
    model : Dict[str, np.ndarray]
        Generative model with keys:
        - A: observation likelihood [num_obs, num_states]
        - B: transition dynamics [num_actions, num_states, num_states]
        - C: log preferences [num_obs]
        - D: initial state prior [num_states]
        - policies: list of action sequences (not implemented yet)
    """
    LOGGER.info(f"Building generative model for agent {agent_id}")

    num_states = env.get_num_states()
    num_obs = env.get_num_observations()
    num_actions = env.get_num_actions()

    LOGGER.info(f"  num_states={num_states}, num_obs={num_obs}, num_actions={num_actions}")

    # A: observation likelihood (fully observable, identity)
    A = np.eye(num_obs, num_states)
    LOGGER.debug(f"  A matrix: shape={A.shape}")

    # B: transition dynamics
    B = np.zeros((num_actions, num_states, num_states))

    for s in range(num_states):
        x, y = env.obs_index_to_pos(s)

        for a in range(num_actions):
            # Compute next position given action
            x_next, y_next = x, y

            if a == UP:
                y_next = max(0, y - 1)
            elif a == DOWN:
                y_next = min(env.height - 1, y + 1)
            elif a == LEFT:
                x_next = max(0, x - 1)
            elif a == RIGHT:
                x_next = min(env.width - 1, x + 1)
            elif a == STAY:
                pass

            s_next = env.pos_to_obs_index((x_next, y_next))
            B[a, s_next, s] = 1.0

    LOGGER.debug(f"  B matrix: shape={B.shape}")

    # C: log preferences
    C = np.zeros(num_obs)

    for o in range(num_obs):
        x, y = env.obs_index_to_pos(o)

        if y != env.safe_y:
            # Lava: very low preference
            C[o] = 0.01
        elif x == env.goal_x and y == env.safe_y:
            # Goal: high preference
            C[o] = 10.0
        else:
            # Safe corridor: neutral
            C[o] = 1.0

    # Normalize to log space
    C = np.log(C + 1e-16)
    LOGGER.debug(f"  C (log preferences): shape={C.shape}, range=[{C.min():.2f}, {C.max():.2f}]")

    # D: initial state prior (uniform over start position)
    D = np.zeros(num_states)
    start_pos = env.start_positions[agent_id]
    start_s = env.pos_to_obs_index(start_pos)
    D[start_s] = 1.0

    LOGGER.debug(f"  D (initial prior): shape={D.shape}, start_s={start_s} (pos={start_pos})")

    model = {
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "policies": [],  # TODO: enumerate policies
    }

    LOGGER.info(f"Generative model built for agent {agent_id}")
    return model
