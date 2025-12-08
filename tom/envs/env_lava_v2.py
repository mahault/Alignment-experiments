"""
LavaV2Env: Multi-variant lava corridor environment for multi-agent coordination.

This environment supports different corridor layouts to test coordination scenarios.
Key improvements over LavaV1Env:
- Support for multiple layout variants (narrow, wide, bottleneck, risk-reward)
- Configurable starting positions for agents
- Extended observations including other agent's position
- Proper collision detection and termination
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from tom.envs.env_lava_variants import LavaLayout, get_layout


class LavaV2Env:
    """
    Multi-variant lava corridor environment.

    Supports different layouts for testing coordination:
    - "narrow": Single-file corridor (collision unavoidable)
    - "wide": Multi-row corridor (can pass each other)
    - "bottleneck": Wide areas with narrow middle (coordination needed)
    - "risk_reward": Risky fast path vs safe slow detour
    """

    def __init__(
        self,
        layout_name: str = "wide",
        num_agents: int = 2,
        timesteps: int = 20,
        **layout_kwargs
    ):
        """
        Initialize lava corridor environment with specified layout.

        Parameters
        ----------
        layout_name : str
            Layout variant: "narrow", "wide", "bottleneck", "risk_reward"
        num_agents : int
            Number of agents (must match layout's start positions)
        timesteps : int
            Maximum timesteps per episode
        **layout_kwargs : dict
            Additional layout parameters (e.g., width=8)
        """
        self.layout = get_layout(layout_name, **layout_kwargs)
        self.num_agents = num_agents
        self.timesteps = timesteps

        # Validate that layout has enough start positions
        if len(self.layout.start_positions) < num_agents:
            raise ValueError(
                f"Layout {layout_name} only has {len(self.layout.start_positions)} "
                f"start positions but {num_agents} agents requested"
            )

        # Build safe cell lookup
        self.safe_cells_set = set(self.layout.safe_cells)
        self.num_states = self.layout.width * self.layout.height

    @property
    def width(self):
        return self.layout.width

    @property
    def height(self):
        return self.layout.height

    @property
    def goal_x(self):
        return self.layout.goal_pos[0]

    @property
    def goal_y(self):
        return self.layout.goal_pos[1]

    def reset(self, key=None):
        """
        Reset environment to initial state.

        Returns
        -------
        state : dict
            Initial environment state with agent positions
        obs : dict
            Initial observations for each agent (includes other agent positions)
        """
        # Place agents at starting positions
        positions = {}
        for i in range(self.num_agents):
            positions[i] = self.layout.start_positions[i]

        # Create initial state
        state = {
            "env_state": {
                "pos": positions,
                "timestep": 0,
            },
            "timestep": 0,
            "done": False,
        }

        # Generate observations
        obs = self._get_observations(positions)

        return state, obs

    def step(self, state, actions):
        """
        Step environment forward with agent actions.

        Parameters
        ----------
        state : dict
            Current environment state
        actions : dict
            Actions for each agent {agent_id: action_idx}
            Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY

        Returns
        -------
        next_state : dict
            Next environment state
        obs : dict
            Observations for each agent
        reward : float
            Reward (not used in active inference)
        done : bool
            Whether episode is done
        info : dict
            Additional information (collision, lava_hit, goal_reached)
        """
        current_pos = state["env_state"]["pos"]
        timestep = state["timestep"] + 1

        # Apply actions to get new positions
        next_pos = {}
        for agent_id, action in actions.items():
            next_pos[agent_id] = self._apply_action(current_pos[agent_id], action)

        # Check for collisions
        collision = self._check_collision(next_pos)

        # Check for lava hits
        lava_hits = {}
        for agent_id, pos in next_pos.items():
            lava_hits[agent_id] = not self._is_safe(pos)

        # Check for goal reached
        goal_reached = {}
        for agent_id, pos in next_pos.items():
            goal_reached[agent_id] = (pos == self.layout.goal_pos)

        # Episode done if collision, lava hit, all reached goal, or max timesteps
        done = (
            collision
            or any(lava_hits.values())
            or all(goal_reached.values())
            or timestep >= self.timesteps
        )

        # Create next state
        next_state = {
            "env_state": {
                "pos": next_pos,
                "timestep": timestep,
            },
            "timestep": timestep,
            "done": done,
        }

        # Generate observations
        obs = self._get_observations(next_pos)

        # Info dict
        info = {
            "collision": collision,
            "lava_hit": lava_hits,
            "goal_reached": goal_reached,
            "timestep": timestep,
        }

        return next_state, obs, 0.0, done, info

    def _apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Apply action to position.

        Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=STAY
        """
        x, y = pos

        if action == 0:  # UP
            y = max(0, y - 1)
        elif action == 1:  # DOWN
            y = min(self.height - 1, y + 1)
        elif action == 2:  # LEFT
            x = max(0, x - 1)
        elif action == 3:  # RIGHT
            x = min(self.width - 1, x + 1)
        elif action == 4:  # STAY
            pass
        else:
            raise ValueError(f"Invalid action: {action}")

        new_pos = (x, y)

        # Stay in place if new position is lava
        if not self._is_safe(new_pos):
            return new_pos  # Allow moving into lava (will be detected as lava hit)

        return new_pos

    def _is_safe(self, pos: Tuple[int, int]) -> bool:
        """Check if position is safe (not lava)."""
        return pos in self.safe_cells_set

    def _check_collision(self, positions: Dict[int, Tuple[int, int]]) -> bool:
        """Check if any agents are at the same position."""
        pos_list = list(positions.values())
        return len(pos_list) != len(set(pos_list))

    def _get_observations(self, positions: Dict[int, Tuple[int, int]]) -> Dict:
        """
        Generate observations for all agents.

        Each agent observes:
        - "location_obs": Their own position (flat index)
        - "other_obs": Other agent's position (flat index, only for 2-agent case)
        """
        obs = {}

        for agent_id, pos in positions.items():
            my_idx = self.pos_to_idx(pos)

            obs_dict = {
                "location_obs": jnp.array([my_idx], dtype=jnp.int32)
            }

            # For 2-agent case, include other agent's position
            if self.num_agents == 2:
                other_id = 1 - agent_id  # 0->1, 1->0
                other_pos = positions[other_id]
                other_idx = self.pos_to_idx(other_pos)
                obs_dict["other_obs"] = jnp.array([other_idx], dtype=jnp.int32)

            obs[agent_id] = obs_dict

        return obs

    def pos_to_idx(self, pos: Tuple[int, int]) -> int:
        """Convert (x, y) to flat state index."""
        x, y = pos
        return y * self.width + x

    def idx_to_pos(self, idx: int) -> Tuple[int, int]:
        """Convert flat state index to (x, y)."""
        y = idx // self.width
        x = idx % self.width
        return x, y

    def render_state(self, state: dict) -> str:
        """
        Render current state as ASCII art.

        Returns
        -------
        ascii_art : str
            String representation of the environment
        """
        positions = state["env_state"]["pos"]

        lines = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)

                # Check if any agent is at this position
                agents_here = [aid for aid, apos in positions.items() if apos == pos]

                if len(agents_here) > 1:
                    row.append("X")  # Collision
                elif len(agents_here) == 1:
                    row.append(str(agents_here[0]))  # Agent ID
                elif pos == self.layout.goal_pos:
                    row.append("G")  # Goal
                elif self._is_safe(pos):
                    row.append(".")  # Safe cell
                else:
                    row.append("~")  # Lava

            lines.append(" ".join(row))

        return "\n".join(lines)

    def get_layout_info(self) -> dict:
        """
        Get environment layout information for model building.

        Returns
        -------
        layout_info : dict
            Layout information including dimensions, safe cells, goal position
        """
        return {
            "width": self.width,
            "height": self.height,
            "num_states": self.num_states,
            "goal_pos": self.layout.goal_pos,
            "safe_cells": self.layout.safe_cells,
            "layout_name": self.layout.name,
        }
