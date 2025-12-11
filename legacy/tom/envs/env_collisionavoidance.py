import os
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr

import numpy as np

import equinox as eqx
from equinox import Module, field

from typing import Optional
from jaxtyping import Array, PRNGKeyArray

from PIL import Image


# load image assets
IMGS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "images/")
ASSETS = {
    name: np.asarray(Image.open(os.path.join(IMGS_DIR, f"{name}.png")))
    for name in [
        "wasteland",
        "wasteland_agent_1",
        "wasteland_agent_2",
        "wasteland_agent_3",
        "wasteland_agent_1_2",
        "wasteland_agent_1_3",
        "wasteland_agent_2_3",
        "wasteland_agent_1_2_3",
    ]
}

# indexing items for the grid
WASTELAND = 0

class CollisionAvoidanceEnv(Module):
    num_agents: int = field(static=True)
    grid_size: int = field(static=True)
    grid: jnp.ndarray = field(init=False, static=False)
    agent_position: jnp.ndarray
    initial_positions: Optional[jnp.ndarray] = field(static=True, default=None)

    def __init__(self, num_agents, grid_size=3, initial_positions=None):
        self.num_agents = num_agents
        self.grid_size = grid_size
        # add a batch dimennsion of 1 to the grid and agent position
        self.grid = self._initialise_grid()[None, ...]
        self.initial_positions = initial_positions
        self.agent_position = self._initialise_agent_position(self.initial_positions, num_agents, grid_size)[None, ...]


    def _initialise_grid(self):
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)

        grid = grid.at[:, :].set(WASTELAND)

        return grid
    
    def _initialise_agent_position(self, initial_positions, num_agents, grid_size):
        if initial_positions is not None:
            if len(initial_positions) != num_agents:
                raise ValueError(f"Number of agents is {num_agents} but {len(initial_positions)} initial positions were given. ")
            return initial_positions
        
        # if initial_positions is not defined, initialise the agent positions randomly in corners
        rng_key = jr.PRNGKey(0)
        corner_indices = jnp.array([
            0,  # top-left
            grid_size - 1,  # top-right
            grid_size * (grid_size - 1),  # bottom-left
            grid_size * grid_size - 1  # bottom-right
        ])
        agent_position = jr.choice(rng_key, corner_indices, (num_agents,), replace=False)
        return agent_position

    def _move_agent(self, action, agent_index, agent_positions):
        move_offsets = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], 
                                  [-1, -1], [-1, 1], [1, -1], [1, 1]])

        current_position = agent_positions[agent_index]
        x, y = divmod(current_position, self.grid_size)
        dx, dy = move_offsets[action]

        new_x = x + dx
        new_y = y + dy

        # check if the new position is within grid bounds
        is_valid_x = (new_x >= 0) & (new_x < self.grid_size)
        is_valid_y = (new_y >= 0) & (new_y < self.grid_size)
        is_valid_move = is_valid_x & is_valid_y

        # if move is valid, calculate new position; else if going to hell state, stay in current position
        new_position = jnp.where(
            is_valid_move,
            new_x * self.grid_size + new_y,
            current_position
        )

        return new_position
    
    def _get_observation(self):
        # get the x and y coordinates of the agent's position
        location = self.agent_position[0]

        # create observation arrays for location and item state
        location_obs = location.reshape((self.num_agents, 1))

        # add observation of the other agent's location
        other_location_obs = jnp.roll(location, shift=-1).reshape((self.num_agents, 1))
        
        return [location_obs, other_location_obs]
    
    def step(self, rng_key: PRNGKeyArray, actions: Optional[Array] = None):
        # pymdp rollout runs n agents on n env instances, so yields a batch of keys [n, 2]
        # however, we only have one environment, so we can just use the first key
        if rng_key.ndim > 1: rng_key = rng_key[0] 

        new_agent_position = self.agent_position

        if actions is not None:
            for i in range(self.num_agents):
                move_action = actions[i, 0]
                new_agent_position = new_agent_position.at[0, i].set(self._move_agent(move_action, i, new_agent_position[0]))

        new_env = eqx.tree_at(lambda env: env.agent_position, self, new_agent_position)

        observations = new_env._get_observation()

        return observations, new_env

    def reset(self, rng_key: Optional[PRNGKeyArray] = None, state: Optional[Array] = None):
        new_env = eqx.tree_at(
            lambda env: (env.agent_position, env.grid),
            self,
            (
                self._initialise_agent_position(self.initial_positions, self.num_agents, self.grid_size)[None, ...],
                self._initialise_grid()[None, ...],
            ),
        )
        observations = new_env._get_observation()
        return observations, new_env

    def render(self):
        # load images
        agent_position = [divmod(pos, self.grid_size) for pos in self.agent_position[0]]
        
        frame = np.zeros(
            (self.grid_size * 8, self.grid_size * 8, 3), dtype=np.uint8
        )

        # displaying the grid with borders
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # check if any agent is at the current position
                agent_here = [idx for idx, (agent_x, agent_y) in enumerate(agent_position) if agent_x == x and agent_y == y]

                if agent_here:
                    if len(agent_here) > 1:
                        img = ASSETS["wasteland_agent_1_2"]
                    else:
                        # if there's a single agent, choose the appropriate image
                        agent_idx = agent_here[0]
                        img = ASSETS[f"wasteland_agent_{agent_idx + 1}"]
                else:
                    # no agent, use the default image
                    img = ASSETS["wasteland"]

                frame[x * 8 : (x + 1) * 8, y * 8 : (y + 1) * 8] = img

        return frame
