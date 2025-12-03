from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import random

Position = Tuple[int, int]


class CellType:
    EMPTY = 0
    WALL = 1
    LAVA = 2
    GOAL_A = 3
    GOAL_B = 4
    HELL = 5  # e.g., deadlock / catastrophic collision


@dataclass
class GridConfig:
    height: int = 7
    width: int = 11
    slip_prob: float = 0.05  # probability of random slip
    max_steps: int = 50


class GridWorld:
    """
    Simple multi-agent gridworld with:
    - Narrow bottleneck
    - Wider detour
    - Stochastic slips
    - Individual goals for agent A and B
    """

    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "STAY"]
    ACTION_TO_DELTA = {
        "UP": (-1, 0),
        "DOWN": (1, 0),
        "LEFT": (0, -1),
        "RIGHT": (0, 1),
        "STAY": (0, 0),
    }

    def __init__(self, config: Optional[GridConfig] = None):
        self.config = config or GridConfig()
        self.grid = self._build_grid()
        self.agent_ids = ["A", "B"]
        self.agent_positions: Dict[str, Position] = {}
        self.t = 0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def reset(self) -> Dict[str, Dict]:
        """
        Reset environment to starting configuration.
        Returns per-agent observations.
        """
        self.t = 0
        self.agent_positions = {
            "A": (3, 1),  # row, col
            "B": (3, 2),
        }
        return self._get_observations()

    def step(
        self,
        actions: Dict[str, str],
    ) -> Tuple[Dict[str, Dict], Dict[str, float], Dict[str, bool], bool, Dict]:
        """
        actions: dict mapping agent_id -> action string
        Returns:
            observations: dict agent_id -> obs dict
            rewards: dict agent_id -> float
            dones: dict agent_id -> bool
            done_global: bool
            info: dict with debug data
        """
        self.t += 1

        proposed = {
            aid: self._apply_action_with_slip(aid, actions.get(aid, "STAY"))
            for aid in self.agent_ids
        }

        # resolve walls and bounds
        proposed = {
            aid: self._clip_to_valid(pos, self.agent_positions[aid])
            for aid, pos in proposed.items()
        }

        # handle collisions
        new_positions, collision_pairs = self._resolve_collisions(proposed)

        self.agent_positions = new_positions

        rewards, dones, info = self._compute_rewards_and_dones(collision_pairs)

        obs = self._get_observations()
        done_global = all(dones.values()) or self.t >= self.config.max_steps

        info["collision_pairs"] = collision_pairs
        info["t"] = self.t

        return obs, rewards, dones, done_global, info

    # -------------------------------------------------------------------------
    # Internal grid construction
    # -------------------------------------------------------------------------

    def _build_grid(self) -> np.ndarray:
        """
        Build a fixed layout:
        - Start area on the left
        - Narrow bottleneck in middle
        - Wider detour path above/below
        - Goals on the right
        """
        H, W = self.config.height, self.config.width
        grid = np.full((H, W), CellType.EMPTY, dtype=int)

        # Add outer walls
        grid[0, :] = CellType.WALL
        grid[-1, :] = CellType.WALL
        grid[:, 0] = CellType.WALL
        grid[:, -1] = CellType.WALL

        # Vertical walls with one-cell bottleneck
        mid_col = W // 2
        for r in range(1, H - 1):
            # Leave a bottleneck at row 3 (zero-indexed)
            if r == H // 2:
                continue
            grid[r, mid_col] = CellType.WALL

        # Optional lava around bottleneck (just as an example)
        bottleneck_row = H // 2
        lava_cols = [mid_col - 1, mid_col + 1]
        for c in lava_cols:
            if 0 < bottleneck_row < H - 1 and 0 < c < W - 1:
                grid[bottleneck_row, c] = CellType.LAVA

        # Goals for A and B on far right
        grid[bottleneck_row - 1, W - 2] = CellType.GOAL_A
        grid[bottleneck_row + 1, W - 2] = CellType.GOAL_B

        return grid

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _apply_action_with_slip(self, agent_id: str, action: str) -> Position:
        """
        Apply intended action with stochastic slip.
        """
        assert action in self.ACTIONS, f"Unknown action: {action}"
        pos = self.agent_positions[agent_id]

        if random.random() < self.config.slip_prob:
            # slip to a random other action (except STAY to keep it interesting)
            candidate_actions = [a for a in self.ACTIONS if a != "STAY"]
            action = random.choice(candidate_actions)

        dx, dy = self.ACTION_TO_DELTA[action]
        return (pos[0] + dx, pos[1] + dy)

    def _clip_to_valid(self, proposed: Position, fallback: Position) -> Position:
        """
        Enforce bounds and wall constraints.
        """
        r, c = proposed
        H, W = self.config.height, self.config.width

        if r < 0 or r >= H or c < 0 or c >= W:
            return fallback
        cell = self.grid[r, c]
        if cell == CellType.WALL:
            return fallback
        return proposed

    def _resolve_collisions(
        self,
        proposed: Dict[str, Position],
    ) -> Tuple[Dict[str, Position], List[Tuple[str, str]]]:
        """
        If agents try to enter the same cell (or swap), treat as collision.
        """
        new_positions = self.agent_positions.copy()
        collision_pairs: List[Tuple[str, str]] = []

        # Simple rule: if both propose same cell, that cell becomes HELL and both are placed there.
        pos_to_agents: Dict[Position, List[str]] = {}
        for aid, pos in proposed.items():
            pos_to_agents.setdefault(pos, []).append(aid)

        for pos, agents_here in pos_to_agents.items():
            if len(agents_here) == 1:
                aid = agents_here[0]
                new_positions[aid] = pos
            else:
                # collision: everyone moves into a HELL cell at that pos
                for aid in agents_here:
                    new_positions[aid] = pos
                collision_pairs.extend(
                    [(agents_here[i], agents_here[j])
                     for i in range(len(agents_here))
                     for j in range(i + 1, len(agents_here))]
                )

        return new_positions, collision_pairs

    def _compute_rewards_and_dones(
        self,
        collision_pairs: List[Tuple[str, str]],
    ) -> Tuple[Dict[str, float], Dict[str, bool], Dict]:
        rewards = {aid: -0.1 for aid in self.agent_ids}  # step cost
        dones = {aid: False for aid in self.agent_ids}
        info: Dict = {}

        # collisions -> HELL
        if collision_pairs:
            for aid in self.agent_ids:
                pos = self.agent_positions[aid]
                r, c = pos
                # mark cell as HELL
                self.grid[r, c] = CellType.HELL
                rewards[aid] -= 10.0
                dones[aid] = True
            info["event"] = "collision"
            return rewards, dones, info

        for aid in self.agent_ids:
            pos = self.agent_positions[aid]
            r, c = pos
            cell = self.grid[r, c]

            if cell == CellType.LAVA:
                rewards[aid] -= 8.0
                dones[aid] = True
                info[f"event_{aid}"] = "lava"
            elif aid == "A" and cell == CellType.GOAL_A:
                rewards[aid] += 5.0
                dones[aid] = True
                info[f"event_{aid}"] = "goal"
            elif aid == "B" and cell == CellType.GOAL_B:
                rewards[aid] += 5.0
                dones[aid] = True
                info[f"event_{aid}"] = "goal"

        return rewards, dones, info

    def _get_observations(self) -> Dict[str, Dict]:
        """
        For now, give each agent:
        - its own position
        - the local cell type
        - positions of all agents (for ToM; can be coarsened later)
        """
        obs = {}
        for aid in self.agent_ids:
            pos = self.agent_positions[aid]
            r, c = pos
            cell = self.grid[r, c]
            obs[aid] = {
                "self_pos": pos,
                "cell_type": int(cell),
                "all_positions": self.agent_positions.copy(),
            }
        return obs
