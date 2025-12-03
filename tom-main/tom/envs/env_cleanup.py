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
        "river",
        "river_agent_1",
        "river_agent_2",
        "river_agent_3",
        "river_agent_1_2",
        "river_agent_1_3",
        "river_agent_2_3",
        "river_agent_1_2_3",
        "polluted_river",
        "polluted_agent_1",
        "polluted_agent_2",
        "polluted_agent_3",
        "polluted_agent_1_2",
        "polluted_agent_1_3",
        "polluted_agent_2_3",
        "polluted_agent_1_2_3",
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
RIVER_CLEAN = 1
RIVER_DIRTY = 2
APPLE = 3
ORCHARD = 4

class CleanUpEnv(Module):
    """
        Environment for a 3x3 gridworld mimicking the MeltingPot CleanUp task.

        The grid consists of:
        - Top row: polluted or clean river cells (RIVER_DIRTY / RIVER_CLEAN)
        - Middle row: wasteland cells (WASTELAND)
        - Bottom row: orchard cells (ORCHARD) with apples (APPLE)

        Agents can perform the following actions:
        - Move in one of four directions (up, down, left, right)
        - Perform a grid cell action (noop, eat, clean)

        When two tiles are cleaned, the river becomes clean and apples can grow in the orchard.
        However, over time the river becomes polluted again and apples stop growing.

        This is a jax jitable environment that can be used with pymdp. Note this environment 
        will have a batch dimension of 1, but can accept actions with a batch dimension of n agents.
    """
    num_agents: int = field(static=True)
    grid_size: int = field(static=True)
    grid: jnp.ndarray = field(init=False, static=False)
    agent_position: jnp.ndarray
    pollution_rate: float = field(static=True)
    initial_positions: Optional[jnp.ndarray] = field(static=True, default=None)

    def __init__(self, pollution_rate, num_agents, grid_size=3, initial_positions=None):
        self.num_agents = num_agents
        self.grid_size = grid_size
        # add a batch dimennsion of 1 to the grid and agent position
        self.grid = self._initialise_grid()[None, ...]
        self.initial_positions = initial_positions
        self.agent_position = self._initialise_agent_position(self.initial_positions, num_agents, grid_size)[None, ...]
        self.pollution_rate = pollution_rate

    def _initialise_grid(self):
        grid = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)

        # set the middle row: wasteland
        grid = grid.at[1, :].set(WASTELAND)

        # set the top row: all polluted river cells, no clean river cells
        grid = grid.at[0, :].set(RIVER_DIRTY)  

        # set the bottom row: all orchard cells, no apples
        grid = grid.at[-1, :].set(ORCHARD)

        return grid
    
    def _initialise_agent_position(self, initial_positions, num_agents, grid_size):
        if initial_positions is not None:
            if len(initial_positions) != num_agents:
                raise ValueError(f"Number of agents is {num_agents} but {len(initial_positions)} initial positions were given. ")
            return initial_positions
            
        # if initial_positions is not defined, initialise the agent positions randomly
        rng_key = jr.PRNGKey(0)
        agent_position = jr.choice(rng_key, jnp.arange(grid_size*grid_size), (num_agents,), replace=False)
        return agent_position

    def update_environment(self, rng_key):
        '''
        In this function, we update the environment grid based on the pollution rate and the actions of the agents.
        We first check if the environment should be polluted.
        We then add apples to the environment if the pollution is low.
        '''
        rng_key, subkey = jr.split(rng_key)
        pollute = jax.random.uniform(subkey, shape=()) # random number between 0 and 1 to determine if the environment should be polluted

        def increase_pollution(rng_key):
            clean_river_mask = (self.grid[0, 0] == RIVER_CLEAN)  # Boolean mask for clean river cells in the top row
            clean_river_indices = jnp.where(clean_river_mask, size=self.grid_size)[0]  # get indices of clean river cells
            num_clean_river = clean_river_indices.size  # count number of clean river cells

            def make_dirty(grid):
                # select a random clean river cell to become dirty
                random_index = jr.randint(rng_key, (), 0, num_clean_river)

                # create a mask to set the selected indexed clean river cell to become dirty 
                mask = jnp.zeros_like(grid)
                mask = mask.at[0, 0, clean_river_indices[random_index]].set(1) 

                # make the river dirty at the selected index
                grid = grid * (1 - mask) + mask * RIVER_DIRTY 

                return grid

            # if no clean river cells, do nothing
            def no_op(grid):
                return grid
            
            # if there are clean river cells, make one of them dirty
            return jax.lax.cond(
                num_clean_river > 0,
                make_dirty,
                no_op,
                self.grid,
            )

        def nothing_happens_pollution(rng_key):
            return self.grid

        # if randomly generated number (pollute) is greater than the pollution rate, do nothing, otherwise increase pollution in the river
        rng_key, subkey = jr.split(rng_key)
        new_grid = lax.cond(
            pollute > self.pollution_rate, 
            nothing_happens_pollution, 
            increase_pollution, 
            subkey
        )

        # update the environment grid
        self = eqx.tree_at(lambda env: env.grid, self, new_grid)

        # ADD APPLES TO THE ENVIRONMENT IF POLLUTION IS LOW
        num_dirty_river = jnp.sum(new_grid[0, 0] == RIVER_DIRTY) # count number of dirty river cells

        orchard_mask = new_grid[0, -1] == ORCHARD  # Boolean mask for orchard cells in the bottom row
        orchard_indices = jnp.where(orchard_mask, size=self.grid_size)[0]  # get indices of orchard cells
        num_orchard = orchard_indices.size  # count orchard cells
        
        def add_apple(rng_key):
            random_index = jr.randint(rng_key, (), 0, num_orchard) # select a random orchard cell to add an apple to

            mask = jnp.zeros_like(self.grid) 
            mask = mask.at[0, -1, orchard_indices[random_index]].set(1) # set the selected indexed orchard cell to become an apple

            new_grid = self.grid * (1 - mask) + mask * APPLE # add an apple to the selected index

            return new_grid

        def nothing_happens_apple(rng_key):
            return self.grid

        rng_key, subkey = jr.split(rng_key)

        def condition_fn(carry):
            num_dirty_river, num_orchard = carry
            return (num_dirty_river <= 1) & (num_orchard > 0)

        # ADD APPLES TO THE ENVIRONMENT IF POLLUTION IS LOW AND THERE ARE ORCHARD CELLS
        new_grid = lax.cond(
            condition_fn((num_dirty_river, num_orchard)),
            add_apple,
            nothing_happens_apple,
            subkey,
        )

        self = eqx.tree_at(lambda env: env.grid, self, new_grid)

        return self

    def step(self, rng_key: PRNGKeyArray, actions: Array):
        '''
        We first update the environment (add pollution + add apples). 
        Then we carry out the actions of the agents in the environment.
        And then we derive the new observations for the agents (new location, item state, action-dependent reward, locations of other agents appended).
        '''
        # pymdp rollout runs n agents on n env instances, so yields a batch of keys [n, 2]
        # however, we only have one environment, so we can just use the first key
        if rng_key.ndim > 1: rng_key = rng_key[0]

        rng_key, subkey_env = jr.split(rng_key, 2) # create subkeys

        new_agent_position = self.agent_position # get the agent's current position to set the new position
        rewards = jnp.zeros(self.num_agents, dtype=jnp.int32) # initialise the rewards for the agents as zeros
        new_env = self.update_environment(subkey_env) # start with the current environment
        
        # carrying out the actions of the agents
        if actions is not None:
            for i in reversed(range(self.num_agents)):
                move_action, grid_action = actions[i][0], actions[i][1] # get the move and grid actions for the agent
                new_agent_position = new_agent_position.at[0, i].set(self._move_agent(move_action, i, new_agent_position)) # move the agent
                new_env, reward_array = self._action_agent(i, new_env, new_agent_position[0, i], grid_action) # carry out the grid cell action (noop/eat/clean)
                rewards = rewards + reward_array

        # update the environment after all agent actions are complete
        # new_env = current_env.update_environment(subkey_env)
        new_env = eqx.tree_at(lambda env: env.agent_position, new_env, new_agent_position) # update the agent's position in the environment
        
        observations = new_env._get_observation(rewards) # derive the new observations for the agents

        return observations, new_env

    def _move_agent(self, action, agent_index, agent_positions):
        move_offsets = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]])

        x, y = divmod(agent_positions[0, agent_index], self.grid_size)
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

        def clean(grid):
            return jax.lax.cond(
                grid[0, x, y] == RIVER_DIRTY,  # if river_dirty,
                lambda g: (g.at[0, x, y].set(RIVER_CLEAN), jnp.zeros(self.num_agents, dtype=jnp.int32)),  # turn to river_clean and no reward
                lambda g: (g, jnp.zeros(self.num_agents, dtype=jnp.int32)),  # if no river_dirty, nothing changed and no reward
                grid,
            )

        def noop(grid):
            return (grid, jnp.zeros(self.num_agents, dtype=jnp.int32))

        new_grid, reward_array = jax.lax.switch(action, [noop, eat, clean], env.grid) # carry out the action (noop/eat/clean)

        return eqx.tree_at(lambda env: env.grid, env, new_grid), reward_array
    
    def _get_observation(self, rewards):
        # get the x and y coordinates of the agent's position
        location = self.agent_position[0]
        x, y = divmod(location, self.grid_size)

        # get the item state directly from the grid
        item_state = self.grid[0, x, y]

        # create observation arrays for location and item state
        location_obs = location.reshape((self.num_agents, 1))
        item_obs = item_state.reshape((self.num_agents, 1))
        reward_obs = rewards.reshape((self.num_agents, 1))

        ##### STORAGE FOR ToM VERSION #####
        # # add observation of all other agents' locations
        # other_agents_obs = []
        # for i in range(self.num_agents - 1):  # -1 because we don't need to roll for the last agent
        #     other_agents_obs.append(jnp.roll(location, shift=-(i+1)).reshape((self.num_agents, 1)))
        
        # return [location_obs, item_obs, reward_obs] + other_agents_obs
        ####################################

        # returning only one agent's observations for the non-tom version.
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

    def render(self, t=None):
        agent_position = self.agent_position[0]
        grid = self.grid[0]

        if t is not None:
            # render the environment at a specific timestep
            # check if we have a time dimension
            if self.grid.ndim != 4:
                raise ValueError("Grid does not have a time dimension for rendering with t.")
            
            agent_position = agent_position[t]
            grid = grid[t]
        else:
            if self.grid.ndim != 3:
                raise ValueError("Cannot render this environment state, might need a time t set.")


        # load images
        agent_position = [divmod(pos, self.grid_size) for pos in agent_position]

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
                        agent_suffix = "_".join(str(idx + 1) for idx in sorted(agent_here))
                        if x == 0:
                            if cell_value == RIVER_CLEAN:
                                img = ASSETS[f"river_agent_{agent_suffix}"]
                            elif cell_value == RIVER_DIRTY:
                                img = ASSETS[f"polluted_agent_{agent_suffix}"]
                        elif x == self.grid_size - 1:
                            if cell_value == APPLE or cell_value == ORCHARD:
                                img = ASSETS[f"orchard_agent_{agent_suffix}"]
                        elif x == 1:
                            img = ASSETS[f"wasteland_agent_{agent_suffix}"]
                    else:
                        # if there's a single agent, choose the appropriate image
                        agent_idx = agent_here[0]
                        if x == 0:
                            if cell_value == RIVER_CLEAN:
                                img = ASSETS[f"river_agent_{agent_idx + 1}"]
                            elif cell_value == RIVER_DIRTY:
                                img = ASSETS[f"polluted_agent_{agent_idx + 1}"]
                        elif x == self.grid_size - 1:
                            if cell_value == APPLE or cell_value == ORCHARD:
                                img = ASSETS[f"orchard_agent_{agent_idx + 1}"]
                        elif x == 1:
                            img = ASSETS[f"wasteland_agent_{agent_idx + 1}"]
                else:
                    # no agent, use the default image
                    if x == 0:
                        if cell_value == RIVER_CLEAN:
                            img = ASSETS["river"]
                        elif cell_value == RIVER_DIRTY:
                            img = ASSETS["polluted_river"]
                    elif x == self.grid_size - 1:
                        if cell_value == APPLE:
                            img = ASSETS["apple"]
                        elif cell_value == ORCHARD:
                            img = ASSETS["orchard"]
                    elif x == 1:
                        img = ASSETS["wasteland"]

                frame[x * 8 : (x + 1) * 8, y * 8 : (y + 1) * 8] = img

        return frame
