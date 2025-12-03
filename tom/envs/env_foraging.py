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
        "apple",
        "orchard",
        "orchard_agent_1",
        "orchard_agent_2",
        "orchard_agent_3",
        "orchard_agent_1_2",
        "orchard_agent_1_3",
        "orchard_agent_2_3",
        "orchard_agent_1_2_3",
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
APPLE = 1
ORCHARD = 2

class ForagingEnv(Module):
    num_agents: int = field(static=True)
    grid_size: int = field(static=True)
    grid: jnp.ndarray = field(init=False, static=False)
    agent_position: jnp.ndarray
    apple_spawn_rate: float = field(static=True)
    initial_positions: Optional[jnp.ndarray] = field(static=True, default=None)

    def __init__(self, apple_spawn_rate, num_agents, grid_size=3, initial_positions=None):
        self.num_agents = num_agents
        self.grid_size = grid_size
        # add a batch dimennsion of 1 to the grid and agent position
        self.grid = self._initialise_grid()[None, ...]
        self.initial_positions = initial_positions
        self.agent_position = self._initialise_agent_position(self.initial_positions, num_agents, grid_size)[None, ...]
        self.apple_spawn_rate = apple_spawn_rate

    def _initialise_grid(self):
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)

        # set the middle row: wasteland
        grid = grid.at[1, :].set(WASTELAND)

        # set the top row: all orchard cells initially
        grid = grid.at[0, :].set(ORCHARD)

        # set the bottom row: all orchard cells initially
        grid = grid.at[-1, :].set(ORCHARD)
        grid = grid.at[-1, 0].set(APPLE) # set location 6 to have an apple initially
        grid = grid.at[-1, -1].set(APPLE) # set location 8 to have an apple initially

        return grid
    
    def _initialise_agent_position(self, initial_positions, num_agents, grid_size):
        if initial_positions is not None:
            if len(initial_positions) != num_agents:
                raise ValueError(f"Number of agents is {num_agents} but {len(initial_positions)} initial positions were given. ")
            return initial_positions
        
        # if initial_positions is not defined, initialise the agent positions randomly
        rng_key = jr.PRNGKey(0)
        middle_row_indices = jnp.arange(grid_size) + grid_size
        agent_position = jr.choice(rng_key, middle_row_indices, (num_agents,), replace=False)
        return agent_position

    def update_environment(self, rng_key):
        # spawn apples at orchard locations based on apple_spawn_rate
        rng_key, subkey = jr.split(rng_key)
        
        # find all orchard locations (both top and bottom rows)
        orchard_mask = (self.grid[0] == ORCHARD)
        orchard_indices = jnp.where(orchard_mask, size=self.grid_size * self.grid_size, fill_value=-1)
        orchard_coords = jnp.stack([orchard_indices[0], orchard_indices[1]], axis=1)
        num_orchards = jnp.sum(orchard_mask)

        def spawn_apple(rng_key):
            # randomly select an orchard location to spawn apple
            random_idx = jr.randint(rng_key, (), 0, num_orchards)
            selected_coord = orchard_coords[random_idx]
            x, y = selected_coord[0], selected_coord[1]
            # spawn apple at selected orchard location
            new_grid = self.grid.at[0, x, y].set(APPLE)
            return new_grid

        def no_spawn(rng_key):
            return self.grid

        # decide whether to spawn apple based on spawn rate
        spawn_decision = jr.uniform(subkey, shape=())
        
        rng_key, subkey2 = jr.split(rng_key)
        new_grid = lax.cond(
            spawn_decision < self.apple_spawn_rate,
            spawn_apple,
            no_spawn,
            subkey2
        )

        # only spawn if there are orchards available
        final_grid = lax.cond(
            num_orchards > 0,
            lambda g: new_grid,
            lambda g: self.grid,
            new_grid
        )

        self = eqx.tree_at(lambda env: env.grid, self, final_grid)
        return self

    def step(self, rng_key: PRNGKeyArray, actions: Optional[Array] = None):
        # pymdp rollout runs n agents on n env instances, so yields a batch of keys [n, 2]
        # however, we only have one environment, so we can just use the first key
        if rng_key.ndim > 1: rng_key = rng_key[0] 

        rng_key, subkey_env = jr.split(rng_key, 2)
        new_env = self.update_environment(subkey_env)

        new_agent_position = self.agent_position
        rewards = jnp.zeros(self.num_agents, dtype=jnp.int32)

        if actions is not None:
            for i in reversed(range(self.num_agents)):
                move_action, grid_action = actions[i, 0], actions[i, 1]
                new_agent_position = new_agent_position.at[0, i].set(self._move_agent(move_action, i, new_agent_position[0]))
                new_env, reward_array = self._action_agent(i, new_env, new_agent_position[0, i], grid_action)
                rewards = rewards + reward_array

        new_env = eqx.tree_at(lambda env: env.agent_position, new_env, new_agent_position)

        observations = new_env._get_observation(rewards)

        return observations, new_env

    def _move_agent(self, action, agent_index, agent_positions):
        move_offsets = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])

        x, y = divmod(agent_positions[agent_index], self.grid_size)
        dx, dy = move_offsets[action]

        new_x = jnp.clip(x + dx, 0, self.grid_size - 1)
        new_y = jnp.clip(y + dy, 0, self.grid_size - 1)

        new_position = new_x * self.grid_size + new_y

        return new_position

    def _action_agent(self, agent_idx, env, position, action):
        action = jnp.array(action, dtype=jnp.int32)
        x, y = divmod(position, self.grid_size)

        def eat(grid):
            return jax.lax.cond(
                grid[0, x, y] == APPLE,  # if apple,
                lambda g: (g.at[0, x, y].set(ORCHARD), jnp.zeros(self.num_agents, dtype=jnp.int32).at[agent_idx].set(1)),
                lambda g: (g, jnp.zeros(self.num_agents, dtype=jnp.int32)),  # if no apple, nothing changed and no reward
                grid,
            )

        def noop(grid):
            return (grid, jnp.zeros(self.num_agents, dtype=jnp.int32))

        new_grid, reward_array = jax.lax.switch(action, [noop, eat], env.grid)

        return eqx.tree_at(lambda env: env.grid, env, new_grid), reward_array
    
    def _get_observation(self, rewards):
        # get the x and y coordinates of the agent's position
        location = self.agent_position[0]
        x, y = divmod(location, self.grid_size)

        # get the item state directly from the grid
        item_state = self.grid[0, x, y]
        
        # map grid values to observation values for the generative model
        # wasteland (0) -> "wasteland", apple (1) -> "apple", orchard (2) -> "orchard"
        # if in wasteland locations (middle row), always observe "wasteland" (index 0)
        # if in apple/orchard locations, observe the actual state
        is_wasteland = (x == 1)  # middle row
        apple_obs = jnp.where(item_state == APPLE, 1, 2)  # 1 for apple, 2 for orchard
        mapped_item_state = jnp.where(is_wasteland, 0, apple_obs)  # 0 for wasteland, otherwise apple/orchard

        # create observation arrays for location and item state
        location_obs = location.reshape((self.num_agents, 1))
        item_obs = mapped_item_state.reshape((self.num_agents, 1))
        reward_obs = rewards.reshape((self.num_agents, 1))

        return [location_obs, item_obs, reward_obs]

    def reset(self, rng_key: Optional[PRNGKeyArray] = None, state: Optional[Array] = None):
        new_env = eqx.tree_at(
            lambda env: (env.agent_position, env.grid),
            self,
            (
                self._initialise_agent_position(self.initial_positions, self.num_agents, self.grid_size)[None, ...],
                self._initialise_grid()[None, ...],
            ),
        )
        observations = new_env._get_observation(jnp.zeros(self.num_agents, dtype=jnp.int32))
        return observations, new_env

    def render(self):
        # load images
        agent_position = [divmod(pos, self.grid_size) for pos in self.agent_position[0]]
        grid = self.grid[0]
        frame = np.zeros(
            (self.grid_size * 8, self.grid_size * 8, 3), dtype=np.uint8
        )

        # displaying the grid with borders
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_value = grid[x, y]
                img = None

                # check if any agent is at the current position
                agent_here = [idx for idx, (agent_x, agent_y) in enumerate(agent_position) if agent_x == x and agent_y == y]

                if agent_here:
                    # if multiple agents are here, use the combined image
                    if len(agent_here) > 1:
                        # sort agent indices to ensure consistent asset naming
                        agent_here.sort()
                        agent_suffix = "_".join(str(idx + 1) for idx in agent_here)
                        
                        if x == 1:  # wasteland row
                            asset_key = f"wasteland_agent_{agent_suffix}"
                        else:  # orchard/apple rows (top and bottom)
                            asset_key = f"orchard_agent_{agent_suffix}"
                        
                        # check if the asset exists, fallback to single agent if not
                        if asset_key in ASSETS:
                            img = ASSETS[asset_key]
                        else:
                            # fallback: use the first agent's image
                            agent_idx = agent_here[0]
                            if x == 1:  # wasteland row
                                img = ASSETS[f"wasteland_agent_{agent_idx + 1}"]
                            else:  # orchard/apple rows (top and bottom)
                                img = ASSETS[f"orchard_agent_{agent_idx + 1}"]
                    else:
                        # if there's a single agent, choose the appropriate image
                        agent_idx = agent_here[0]
                        if x == 1:  # wasteland row
                            img = ASSETS[f"wasteland_agent_{agent_idx + 1}"]
                        else:  # orchard/apple rows (top and bottom)
                            img = ASSETS[f"orchard_agent_{agent_idx + 1}"]
                else:
                    # no agent, use the default image
                    if x == 1:  # wasteland row
                        img = ASSETS["wasteland"]
                    else:  # orchard/apple rows (top and bottom)
                        if cell_value == APPLE:
                            img = ASSETS["apple"]
                        elif cell_value == ORCHARD:
                            img = ASSETS["orchard"]

                frame[x * 8 : (x + 1) * 8, y * 8 : (y + 1) * 8] = img

        return frame
