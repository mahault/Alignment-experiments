"""
TOM-compatible wrapper for LavaCorridorEnv.

This provides a JAX-friendly interface matching the TOM environment API.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Tuple, Any

from src.envs.lava_corridor import LavaCorridorEnv, LavaCorridorConfig


class LavaV1Env:
    """
    TOM-compatible wrapper for LavaCorridor environment.

    This wrapper provides:
    - JAX-compatible reset/step methods
    - Observation extraction for TOM agents
    - State management compatible with TOM rollouts
    """

    def __init__(self, width=4, height=3, num_agents=2, timesteps=10):
        """
        Initialize lava corridor environment.

        Parameters
        ----------
        width : int
            Grid width
        height : int
            Grid height (must be 3 for lava-safe-lava)
        num_agents : int
            Number of agents
        timesteps : int
            Maximum timesteps per episode
        """
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.timesteps = timesteps

        # Create underlying environment
        config = LavaCorridorConfig(
            width=width,
            height=height,
            num_agents=num_agents
        )
        self._env = LavaCorridorEnv(config)

        # Layout information for TOM
        self.safe_y = 1
        self.goal_x = width - 1
        self.num_states = width * height

    def reset(self, key=None):
        """
        Reset environment to initial state.

        Parameters
        ----------
        key : jax.random.PRNGKey, optional
            Random key (not used for deterministic resets)

        Returns
        -------
        state : dict
            Initial environment state
        obs : dict
            Initial observations for each agent
        """
        state, obs = self._env.reset(key)

        # Convert to TOM format
        tom_state = {
            "env_state": state,
            "timestep": 0,
            "done": False,
        }

        tom_obs = self._convert_obs_to_tom(obs)

        return tom_state, tom_obs

    def step(self, state, actions):
        """
        Step environment forward with agent actions.

        Parameters
        ----------
        state : dict
            Current environment state
        actions : dict
            Actions for each agent {agent_id: action_idx}

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
            Additional information
        """
        env_state = state["env_state"]
        timestep = state["timestep"] + 1

        # Step underlying environment
        next_env_state, next_obs, done, info = self._env.step(env_state, actions)

        # Update TOM state
        next_state = {
            "env_state": next_env_state,
            "timestep": timestep,
            "done": done or (timestep >= self.timesteps),
        }

        # Convert observations
        tom_obs = self._convert_obs_to_tom(next_obs)

        # Reward not used in active inference, but return for compatibility
        reward = 0.0

        return next_state, tom_obs, reward, next_state["done"], info

    def _convert_obs_to_tom(self, obs):
        """
        Convert LavaCorridor observations to TOM format.

        Parameters
        ----------
        obs : dict
            Observations from LavaCorridor {agent_id: (x, y)}

        Returns
        -------
        tom_obs : dict
            TOM-format observations {agent_id: {modality: jax.Array}}
        """
        tom_obs = {}

        for agent_id, (x, y) in obs.items():
            # Convert position to flat location index
            location_idx = self.pos_to_idx(x, y)

            # TOM expects dict of modality observations
            tom_obs[agent_id] = {
                "location_obs": jnp.array([location_idx], dtype=jnp.int32)
            }

        return tom_obs

    def pos_to_idx(self, x, y):
        """Convert (x, y) to flat state index."""
        return y * self.width + x

    def idx_to_pos(self, idx):
        """Convert flat state index to (x, y)."""
        y = idx // self.width
        x = idx % self.width
        return x, y

    def get_layout(self):
        """
        Get environment layout information for TOM model building.

        Returns
        -------
        layout : dict
            Layout information including:
            - width, height: grid dimensions
            - num_states: total state space size
            - safe_y: y-coordinate of safe row
            - goal_x: x-coordinate of goal
        """
        return {
            "width": self.width,
            "height": self.height,
            "num_states": self.num_states,
            "safe_y": self.safe_y,
            "goal_x": self.goal_x,
        }
