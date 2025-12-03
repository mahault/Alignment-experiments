import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX
from jaxmarl import make

import equinox as eqx
from equinox import Module, field

# note that the channels related to tomatoes have been removed from the mapping as they are not used in overcooked_v1
CHANNEL_MAP = {
    'agent_pos': 0, # agent 0 position
    'agent_1_pos': 1, # agent 1 position
    'facinglocation_start': 2, # start of agent 0's orientation channels
    'facinglocation_end': 6, # end of agent 0's orientation channels
    'agent_1_orientation_start': 6, # start of agent 1's orientation channels
    'agent_1_orientation_end': 10, # end of agent 1's orientation channels
    'pot_idx': 10, # pot locations (static; binary)
    'counter_idx': 11, # counter locations (static; binary)
    'onion_pile_idx': 12, # onion pile locations (static; binary)
    'plate_pile_idx': 14, # plate pile locations (static; binary)
    'goal_idx': 15, # delivery locations (static; binary)
    'onions_in_pot': 16, # how many onions in pot (static; count)
    'onions_in_soup': 18, # how many onions in soup (static; count)
    'pot_cooking_time': 20, # pot cooking time remaining (static; count)
    'soup_ready': 21, # soup/food ready (binary)
    'plate_locations': 22, # plate locations variable (dynamic w agents)
    'onion_locations': 23, # onion locations variable (dynamic w agents)
    'urgency': 25, # urgency flag (binary; based on timesteps remaining)
}


# generate coordinate to location mapping from environment layout
def generate_coord_mapping(layout):
    all_positions = []
    for pos in range(layout["height"] * layout["width"]):
        row = pos // layout["width"]
        col = pos % layout["width"]
        all_positions.append([row, col])
    
    return jnp.array(all_positions)

def extract_object_locations(agent_obs, layout_key, coord_mapping, layout):
    # extracting channel index from layout key
    channel_idx = CHANNEL_MAP[layout_key]
    obj_channel = agent_obs[:, :, channel_idx]
    
    # find out how many of the objects are there from the layout
    object_count = len(layout[layout_key]) if layout_key in layout else 0
    
    # if the object doesn't exist, return an empty array
    if object_count == 0:
        return jnp.array([])
    
    # extract coordinates where the objects are present
    # obj_channel > 0 creates a boolean mask; size sets the number of objects to find; returns a tuple of object coorindates 
    pile_coords = jnp.where(obj_channel > 0, size=object_count, fill_value=-1)
    
    # creates a a list of -1s depending on the number of objects - this will be filled with the location indices so we initialise with -1s as 0 is a valid location
    pile_locations = jnp.full(object_count, -1)
    
    # convert coordinates to location indices in a for loop
    for i in range(object_count):
        row, col = pile_coords[0][i], pile_coords[1][i]
        valid_coords = (row >= 0) & (col >= 0)
        
        # finding which location index corresponds to the the coordinates
        matches = jnp.all(coord_mapping == jnp.array([row, col]), axis=1)
        location_idx = jnp.where(matches, jnp.arange(len(coord_mapping)), -1).max()
        
        # stores the location index in pile_locations list
        pile_locations = pile_locations.at[i].set(
            jnp.where(valid_coords & (location_idx >= 0), location_idx, -1)
        )
    
    return jnp.sort(pile_locations)  # sort for consistent ordering

def extract_agent_obsmodalities(agent_obs, obsmodality, coord_mapping=None, layout=None, goal_delivered_state=0):
    if obsmodality == 'location':
        pos_channel = agent_obs[:, :, CHANNEL_MAP['agent_pos']]
        coords = jnp.where(pos_channel > 0, size=1, fill_value=-1)
        x, y = coords[0][0], coords[1][0]
        
        # find matching coordinates and convert to nonwall location index
        matches = jnp.all(coord_mapping == jnp.array([x, y]), axis=1)
        all_location_idx = jnp.where(matches, jnp.arange(len(coord_mapping)), -1).max()
        
        # convert from all grid locations to nonwall location index as we work in a reduced location state space
        wall_set = set(layout["wall_idx"].tolist())
        nonwall_locations = jnp.array([loc for loc in range(len(coord_mapping)) if loc not in wall_set])
        # find index in nonwall_locations
        matches = (nonwall_locations == all_location_idx)
        nonwall_idx = jnp.where(matches, jnp.arange(len(nonwall_locations)), -1).max()
        
        return jnp.array([nonwall_idx])
    
    elif obsmodality == 'facinglocation':
        # get agent's current position (same code from above) 
        pos_channel = agent_obs[:, :, CHANNEL_MAP['agent_pos']]
        coords = jnp.where(pos_channel > 0, size=1, fill_value=-1)
        agent_x, agent_y = coords[0][0], coords[1][0]
        
        # get agent's facing location (0=north, 1=south, 2=east, 3=west)
        orient_channels = agent_obs[:, :, CHANNEL_MAP['facinglocation_start']:CHANNEL_MAP['facinglocation_end']]
        
        # summing across all dimensions of the orientation channels as only 1 of them will contain the flag
        orient_sums = jnp.sum(orient_channels, axis=(0, 1))
        facinglocation = jnp.argmax(orient_sums)
        
        # calculate facing location based on current position + facing location
        direction_offsets = jnp.array([
            [-1, 0],  # north (up)
            [1, 0],   # south (down)
            [0, 1],   # east (right)
            [0, -1],  # west (left)
        ])
        
        # select the appropriate direction offset using the facing location
        dr_dc = direction_offsets[facinglocation]
        facing_x, facing_y = agent_x + dr_dc[0], agent_y + dr_dc[1]
        
        # convert facing location coordinates to location index
        valid_x = (0 <= facing_x) & (facing_x < layout["height"])
        valid_y = (0 <= facing_y) & (facing_y < layout["width"])
        valid_coords = valid_x & valid_y
        
        facing_coords = jnp.array([facing_x, facing_y])
        matches = jnp.all(coord_mapping == facing_coords, axis=1)
        facing_location = jnp.where(
            valid_coords, 
            jnp.where(matches, jnp.arange(len(coord_mapping)), -1).max(),
            -1 
        )
            
        return jnp.array([facing_location])
    
    elif obsmodality == 'self_carrying':
        # get agent's current position
        pos_channel = agent_obs[:, :, CHANNEL_MAP['agent_pos']]
        coords = jnp.where(pos_channel > 0, size=1, fill_value=-1)
        agent_x, agent_y = coords[0][0], coords[1][0]
        
        # check what items are at agent's position (carried items)
        onion_channel = agent_obs[:, :, CHANNEL_MAP['onion_locations']]
        plate_channel = agent_obs[:, :, CHANNEL_MAP['plate_locations']]
        soup_ready_channel = agent_obs[:, :, CHANNEL_MAP['soup_ready']]
        pot_channel = agent_obs[:, :, CHANNEL_MAP['pot_idx']]
        
        # check if agent is carrying items
        carrying_onion = onion_channel[agent_x, agent_y] > 0
        carrying_plate = plate_channel[agent_x, agent_y] > 0
        
        # check if there's soup ready at agent's location
        soup_at_location = soup_ready_channel[agent_x, agent_y] > 0
        # but make sure it's not at a pot location (pot_done vs plate_full)
        pot_at_location = pot_channel[agent_x, agent_y] > 0
        carrying_dish = soup_at_location & ~pot_at_location
        
        # define carrying elements for consistent indexing
        carrying_elements = ["nothing", "onion", "plate_empty", "plate_full"]
        
        # determine carrying state using named elements
        nothing_idx = carrying_elements.index("nothing")
        onion_idx = carrying_elements.index("onion")
        plate_empty_idx = carrying_elements.index("plate_empty")
        plate_full_idx = carrying_elements.index("plate_full")
        
        # priority: onion > plate_full > plate_empty > nothing
        carrying_state = jnp.where(
            carrying_onion, onion_idx,
            jnp.where(
                carrying_dish, plate_full_idx,
                jnp.where(carrying_plate, plate_empty_idx, nothing_idx)
            )
        )
        
        return jnp.array([carrying_state])
    
    elif obsmodality == 'pot':
        # extract onions in pot channel to determine pot state
        onions_in_pot_channel = agent_obs[:, :, CHANNEL_MAP['onions_in_pot']]
        cooking_time_channel = agent_obs[:, :, CHANNEL_MAP['pot_cooking_time']]
        soup_ready_channel = agent_obs[:, :, CHANNEL_MAP['soup_ready']]
        
        # sum all onions across all pot locations to get total onions in pots
        total_onions_in_pots = jnp.sum(onions_in_pot_channel)
        # get maximum cooking time across all pots (assuming only one pot is cooking at a time)
        max_cooking_time = jnp.max(cooking_time_channel)
        # check if soup is ready
        soup_ready = jnp.max(soup_ready_channel) > 0
        
        # define pot state elements for consistent indexing (6 cooking stages instead of 19)
        pot_elements = [
            "pot_empty", "pot_1onion", "pot_2onions", "pot_3onions",
            "pot_cooking6", "pot_cooking5", "pot_cooking4",
            "pot_cooking3", "pot_cooking2", "pot_cooking1", "pot_done",
        ]
        
        # determine pot state based on soup ready status, cooking time, and onion count
        # map JaxMARL's 20 cooking steps to our 6 cooking stages
        # cooking_time 20-17 -> pot_cooking6
        # cooking_time 16-13 -> pot_cooking5
        # cooking_time 12-9 -> pot_cooking4
        # cooking_time 8-5 -> pot_cooking3
        # cooking_time 4-2 -> pot_cooking2
        # cooking_time 1 -> pot_cooking1
        pot_state = jnp.where(
            soup_ready,
            pot_elements.index("pot_done"),
            jnp.where(
                max_cooking_time > 0,
                # map cooking time to 6-stage cooking states
                jnp.where(
                    max_cooking_time >= 17, pot_elements.index("pot_cooking6"),
                    jnp.where(
                        max_cooking_time >= 13, pot_elements.index("pot_cooking5"),
                        jnp.where(
                            max_cooking_time >= 9, pot_elements.index("pot_cooking4"),
                            jnp.where(
                                max_cooking_time >= 5, pot_elements.index("pot_cooking3"),
                                jnp.where(
                                    max_cooking_time >= 2, pot_elements.index("pot_cooking2"),
                                    pot_elements.index("pot_cooking1")
                                )
                            )
                        )
                    )
                ),
                # not cooking, so determine based on onion count
                jnp.where(
                    total_onions_in_pots == 0, 
                    pot_elements.index("pot_empty"),
                    jnp.where(
                        total_onions_in_pots == 1,
                        pot_elements.index("pot_1onion"),
                        jnp.where(
                            total_onions_in_pots == 2,
                            pot_elements.index("pot_2onions"),
                            pot_elements.index("pot_3onions")
                        )
                    )
                )
            )
        )
        
        return jnp.array([pot_state])
    
    elif obsmodality == 'goal_delivered':
        # use the externally tracked goal delivery state since JaxMARL doesn't expose this directly in observations
        # goal_delivered_state: 0 = goal_empty, 1 = goal_delivered
        
        return jnp.array([goal_delivered_state])
    
    else:
        # handle object modalities (e.g., onion_pile_idx, pot_idx, goal_idx, etc.) if they exist in the layout
        if obsmodality.endswith(('0', '1', '2', '3', '4')):  # currently supports up to 5 objects
            obj_num = int(obsmodality[-1])
            layout_key = obsmodality[:-1]  # remove the number suffix to get layout key
            
            # check if this is a valid layout key that we handle
            valid_layout_keys = ["onion_pile_idx", "plate_pile_idx"]
            # valid_layout_keys = ["onion_pile_idx", "pot_idx", "plate_pile_idx", "goal_idx"]

            if layout_key in valid_layout_keys:
                all_locations = extract_object_locations(agent_obs, layout_key, coord_mapping, layout)
                
                # return the requested object index
                if obj_num < len(all_locations):
                    return jnp.array([all_locations[obj_num]])
                else:
                    return jnp.array([-1])  # object doesn't exist
            else:
                raise ValueError(f"Unknown layout key in obsmodality: {layout_key}")
        else:
            raise ValueError(f"Unknown obsmodality: {obsmodality}")

def generate_obsmodalities_from_layout(layout):
    # only return the dynamic observation modalities
    # static object locations (piles) are already encoded in the model structure
    obsmodalities = ['location', 'facinglocation', 'self_carrying', 'pot', 'goal_delivered']
    
    return obsmodalities

# convert jaxMARL overcooked observations to pymdp format
def obs_conversion(obs, num_agents, layout, rewards=None, _goal_state={'delivered': False}):
    # detect goal delivery based on rewards (DELIVERY_REWARD = 20 in jaxMARL)
    if rewards is not None:
        delivery_reward = 20.0
        reward_array = jnp.array(list(rewards.values()))
        has_delivery = jnp.any(reward_array >= delivery_reward)
        _goal_state['delivered'] = jnp.where(has_delivery, True, _goal_state['delivered'])
    
    # convert boolean to index: false -> 0 (goal_empty), true -> 1 (goal_delivered)
    goal_delivered_state = jnp.asarray(_goal_state['delivered'], dtype=jnp.int32)
    
    agents = [f'agent_{i}' for i in range(num_agents)]
    obsmodalities = generate_obsmodalities_from_layout(layout)
    
    # generate coordinate mapping from layout
    coord_mapping = generate_coord_mapping(layout)
    
    # extract all obsmodalities for all agents
    obs_batched = []
    for om in obsmodalities:
        obsmodality_data = jnp.stack([
            extract_agent_obsmodalities(obs[agent], om, coord_mapping, layout, goal_delivered_state) for agent in agents
        ], axis=0)
        obs_batched.append(obsmodality_data)
    
    return obs_batched

def reset_goal_delivery_tracking():
    obs_conversion.__defaults__ = (None, {'delivered': False})


class OvercookedV1Env(Module):
    """
    wrapper for jaxMARL overcooked_v1 environment to work with pymdp rollout
    """

    num_agents: int = field(static=True)
    max_steps: int = field(static=True)
    layout: dict = field(static=True)
    env_object: object = field(static=True)  
    current_env_state: object = field(default=None, static =False) 
    initiate_inventory: list = field(static=True)

    def __init__(self, num_agents, layout, max_steps, initiate_inventory):
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.layout = layout
        self.env_object = make('overcooked', layout=layout, max_steps=self.max_steps)
        self.current_env_state = None
        self.initiate_inventory = initiate_inventory

    def reset(self, key):
        key = jax.random.PRNGKey(0)
        key, key_r = jax.random.split(key, 2)
        obs_reset, state_reset = self.env_object.reset(key_r)

        # reset goal delivery tracking
        reset_goal_delivery_tracking()

        # modifying the JaxMARL environment state if there is an initial inventory
        if self.initiate_inventory is not None:
            agent_inventory = self.initiate_inventory

            # converting inventory items to indices
            inventory_indices = jnp.array([OBJECT_TO_INDEX[item] for item in agent_inventory])

            # updating the state with new inventories
            new_state = state_reset.replace(agent_inv=inventory_indices)

            # updating the current environment state
            # self.current_env_state = new_state

            # recalculating observations with new inventories
            new_obs = self.env_object.get_obs(new_state)
            state_reset = new_state
            obs_reset_converted = obs_conversion(new_obs, self.num_agents, self.layout)

        # convert observations to pymdp format using layout
        else:
            obs_reset_converted = obs_conversion(obs_reset, self.num_agents, self.layout)
        
        return obs_reset_converted, state_reset
    
    def step(self, rng_key, env_state, actions):
        
        # convert pymdp actions (stacked) to jaxMARL format (Dict)
        jaxmarl_actions = {'agent_0': actions[0, 0], 'agent_1': actions[1, 0]}
        
        # extract a single key from potentially batched rng_key for jaxMARL as it will split it internally
        if rng_key.ndim > 1:
            env_key = rng_key[0]  # take the first key from the batch, maintaining proper key shape
        else:
            env_key = rng_key

        # step the environment using current state
        obs, state, rewards, dones, infos = self.env_object.step(env_key, env_state, jaxmarl_actions)

        # convert observations (Dict) to pymdp format (stacked) using layout
        # pass rewards so obs_conversion can detect goal delivery internally
        obs_converted = obs_conversion(obs, self.num_agents, self.layout, rewards)
        
        return obs_converted, state