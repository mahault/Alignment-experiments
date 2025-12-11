"""
TODO: 
- add the other agent in the focal agent's gen model and input stay action if colliding/swapping locations
- do we need to have preferences increasing in amount of onions in pot? or can the preference values stay the same due to accumulation of utility?
- add the "done" jaxmarl action?
"""


import jax.numpy as jnp

from pymdp.distribution import compile_model
from pymdp.agent import Agent

# turning the model defined into a pymdp agent where model validdation is also conducted
def OvercookedAgent(model, gamma=1.0, batch_size=1):

    # (6 actions of up, down, right, left, stay, interact; 1 batch size; num of state factors)
    num_actions = int(len(model.B[0].batch["actions"]))
    policies = jnp.zeros((num_actions, 1, len(model.B)), dtype=jnp.int32)

    # actions affect location, facing location, and carrying state factors
    for i in range(num_actions):
        policies = policies.at[i, 0, 0:6].set(i) 
    
    # TODO: add the "done" jaxmarl action appropriately
    # policies = policies.at[6, 0, 1].set(2) # action of done currently at 2nd state factor 

    agent = Agent(**model, batch_size=batch_size, policies=policies, learn_A=False, learn_B=False, gamma=gamma, sampling_mode="full")
    return agent


def extract_layout_info(env_layout):
    all_locations = env_layout["height"] * env_layout["width"]
    wall_set = set(env_layout["wall_idx"].tolist())
    
    # create mapping from full grid to moveable (non-wall) locations 
    nonwall_locations = [loc for loc in range(all_locations) if loc not in wall_set]
    nonwall_to_all = jnp.array(nonwall_locations)  # maps nonwall index -> all grid index
    all_to_nonwall = jnp.full(all_locations, -1)  # maps all grid index -> nonwall index
    for nonwall_idx, all_idx in enumerate(nonwall_locations):
        all_to_nonwall = all_to_nonwall.at[all_idx].set(nonwall_idx)
    
    layout_info = {
        "all_locations": all_locations,
        "nonwall_locations": len(nonwall_locations),
        "wall_set": wall_set,
        "nonwall_to_all": nonwall_to_all,
        "all_to_nonwall": all_to_nonwall,
        "objects": {}
    }
    
    # count each type of object in the layout
    object_keys = ["onion_pile_idx", "plate_pile_idx", "pot_idx", "goal_idx"]
    
    for layout_key in object_keys:
        if layout_key in env_layout and len(env_layout[layout_key]) > 0:
            layout_info["objects"][layout_key] = {
                "count": len(env_layout[layout_key]),
                "indices": env_layout[layout_key].tolist()
            }
    
    return layout_info

def generate_obs_modalities(layout_info):
    obsmodalities = {
        "self_location_obs": {
            "size": layout_info["nonwall_locations"],
            "depends_on": ["self_location_state"],
        },
        "self_facinglocation_obs": {
            "size": layout_info["all_locations"],
            "depends_on": ["self_facinglocation_state"],
        },
        "self_carrying_obs": {
            "elements": ["nothing", "onion", "plate_empty", "plate_full"],
            "depends_on": ["self_carrying_state"],
        },
        "pot_obs": {
            "elements": [
                "pot_empty", "pot_1onion", "pot_2onions", "pot_3onions",
                "pot_cooking19", "pot_cooking18", "pot_cooking17", "pot_cooking16",
                "pot_cooking15", "pot_cooking14", "pot_cooking13", "pot_cooking12",
                "pot_cooking11", "pot_cooking10", "pot_cooking9", "pot_cooking8",
                "pot_cooking7", "pot_cooking6", "pot_cooking5", "pot_cooking4",
                "pot_cooking3", "pot_cooking2", "pot_cooking1", "pot_done",
            ],
            "depends_on": ["pot_state"],
        },
        "pot_done_flag_obs": {
            "elements": ["pot_not_done", "pot_done"],
            "depends_on": ["pot_done_flag_state"],
        },
        "goal_delivered_obs": {
            "elements": ["goal_empty", "goal_delivered"],
            "depends_on": ["goal_delivered_state"],
        }
    }

    
    # add modalities for each object type found in layout
    for layout_key, obj_info in layout_info["objects"].items():
        for i in range(obj_info["count"]):
            modality_name = f"{layout_key}{i}_obs"
            obsmodalities[modality_name] = {
                "size": layout_info["all_locations"],
                "depends_on": [f"{layout_key}{i}_state"],
            }
    
    return obsmodalities

def generate_state_factors(layout_info):
    # build dependencies for carrying state - we will append the object state factors
    carrying_depends_on = ["self_carrying_state", "self_facinglocation_state", "pot_done_flag_state", "goal_delivered_state"]
    
    # add all object state factors that are relevant for carrying interactions
    for layout_key, obj_info in layout_info["objects"].items():
        for i in range(obj_info["count"]):
            carrying_depends_on.append(f"{layout_key}{i}_state")
        
    pot_state_depends_on = ["pot_state", "self_facinglocation_state", "self_carrying_state"]
    
    for i in range(layout_info["objects"]["pot_idx"]["count"]):
        pot_state_depends_on.append(f"pot_idx{i}_state")
    
    statefactors = {
        "self_location_state": {
            "size": layout_info["nonwall_locations"],
            "depends_on": ["self_location_state"],
            "controlled_by": ["actions"],
        },
        "self_facinglocation_state": {
            "size": layout_info["all_locations"],
            "depends_on": ["self_facinglocation_state", "self_location_state"],
            "controlled_by": ["actions"],
        },
        "self_carrying_state": {
            "elements": ["nothing", "onion", "plate_empty", "plate_full"],
            "depends_on": carrying_depends_on,
            "controlled_by": ["actions"],
        },
        "pot_state": {
            "elements": [
                "pot_empty", "pot_1onion", "pot_2onions", "pot_3onions",
                "pot_cooking19", "pot_cooking18", "pot_cooking17", "pot_cooking16",
                "pot_cooking15", "pot_cooking14", "pot_cooking13", "pot_cooking12",
                "pot_cooking11", "pot_cooking10", "pot_cooking9", "pot_cooking8",
                "pot_cooking7", "pot_cooking6", "pot_cooking5", "pot_cooking4",
                "pot_cooking3", "pot_cooking2", "pot_cooking1", "pot_done",
                ],
            "depends_on": pot_state_depends_on,
            "controlled_by": ["actions"],
        },
        "pot_done_flag_state": {
            "elements": ["pot_not_done", "pot_done"],
            "depends_on": ["pot_done_flag_state", "pot_state"],
            "controlled_by": ["actions"],
        },
        "goal_delivered_state": {
            "elements": ["goal_empty", "goal_delivered"],
            "depends_on": ["goal_delivered_state", "self_facinglocation_state", "self_carrying_state"],
            "controlled_by": ["actions"],
        },
    }
    
    # add state factors for each object type found in layout
    for layout_key, obj_info in layout_info["objects"].items():
        for i in range(obj_info["count"]):
            state_name = f"{layout_key}{i}_state"
            statefactors[state_name] = {
                "size": layout_info["all_locations"],
                "depends_on": [state_name],
                "controlled_by": ["uncontrollable"], # these are all uncontrollable (static)
            }
    
    return statefactors

# defining the agent's generative model
def OvercookedModel(env_layout):
    
    # analyse layout to determine available objects
    layout_info = extract_layout_info(env_layout)
    
    # helper functions for grid navigation
    def pos_to_coords(pos):
        return pos // env_layout["width"], pos % env_layout["width"]
    
    def coords_to_pos(row, col):
        return row * env_layout["width"] + col
    
    # generate model description based on layout
    obsmodalities = generate_obs_modalities(layout_info)
    statefactors = generate_state_factors(layout_info)
    
    model_description = {
        "observations": obsmodalities,
        "controls": {
            "actions": {
                "elements": ["up", "down", "right", "left", "stay", "interact"],
            },
            "uncontrollable": {
                "elements": ["noop"],
            },
        },
        "states": statefactors,
    }

    model = compile_model(model_description)

    '''
    BUILDING THE A TENSOR
    '''

    # precise observations via identity mappings to state
    model.A["self_location_obs"].data = jnp.eye(layout_info["nonwall_locations"])
    model.A["self_facinglocation_obs"].data = jnp.eye(layout_info["all_locations"])
    model.A["self_carrying_obs"].data = jnp.eye(model.A["self_carrying_obs"].data.shape[0])
    model.A["pot_obs"].data = jnp.eye(model.A["pot_obs"].data.shape[0])
    model.A["pot_done_flag_obs"].data = jnp.eye(model.A["pot_done_flag_obs"].data.shape[0])
    model.A["goal_delivered_obs"].data = jnp.eye(model.A["goal_delivered_obs"].data.shape[0])
    
    # set up identity mappings for all object observation modalities
    for layout_key, obj_info in layout_info["objects"].items():
        for i in range(obj_info["count"]):
            obs_name = f"{layout_key}{i}_obs"
            model.A[obs_name].data = jnp.eye(layout_info["all_locations"])

    '''
    BUILDING THE B TENSOR
    '''

    # creating transitions for nonwall locations only
    action_to_direction = {
        "up": (-1, 0),
        "down": (1, 0), 
        "right": (0, 1),
        "left": (0, -1),
        "stay": (0, 0),
        "interact": (0, 0),
    }
    
    ######### B TENSOR FOR SELF_LOCATION_STATE #########
    for from_nonwall_idx in range(layout_info["nonwall_locations"]):
        # extracting the coordinates of the nonwall location in the larger grid
        from_all_loc = int(layout_info["nonwall_to_all"][from_nonwall_idx])
        from_row, from_col = int(pos_to_coords(from_all_loc)[0]), int(pos_to_coords(from_all_loc)[1])
        
        for action, (dr, dc) in action_to_direction.items():
            to_row, to_col = from_row + dr, from_col + dc
            
            # if target is within bounds (bc the dr dc might result in out of bounds result)
            if (0 <= to_row < env_layout["height"] and 0 <= to_col < env_layout["width"]):
                to_all_loc = int(coords_to_pos(to_row, to_col))
                if to_all_loc not in layout_info["wall_set"]: # and if target is not a wall 
                    to_nonwall_idx = int(layout_info["all_to_nonwall"][to_all_loc])
                    model.B["self_location_state"][to_nonwall_idx, from_nonwall_idx, action] = 1.0 # then move there
                else:
                    # if target is a wall, stay in current location
                    model.B["self_location_state"][from_nonwall_idx, from_nonwall_idx, action] = 1.0
            else:
                # if target coordinates are out of bounds, stay in current location
                model.B["self_location_state"][from_nonwall_idx, from_nonwall_idx, action] = 1.0

    ######### B TENSOR FOR SELF_FACINGLOCATION_STATE #########
    for agent_nonwall_idx in range(layout_info["nonwall_locations"]):
        # extracting the coordinates of the nonwall location in the larger grid
        agent_all_loc = int(layout_info["nonwall_to_all"][agent_nonwall_idx])
        agent_row, agent_col = int(pos_to_coords(agent_all_loc)[0]), int(pos_to_coords(agent_all_loc)[1])
        
        for current_facing_loc in range(layout_info["all_locations"]):
            for action, (dr, dc) in action_to_direction.items():
                if action == "stay" or action == "interact":
                    # when staying, keep the current facing location
                    model.B["self_facinglocation_state"][current_facing_loc, current_facing_loc, agent_nonwall_idx, action] = 1.0
                else:
                    # calculate the location the agent will be facing based on their position and action direction
                    facing_row, facing_col = agent_row + dr, agent_col + dc
                    
                    if (0 <= facing_row < env_layout["height"] and 0 <= facing_col < env_layout["width"]):
                        next_facing_loc = int(coords_to_pos(facing_row, facing_col))
                        # agent faces the location in the direction they're moving to
                        model.B["self_facinglocation_state"][next_facing_loc, current_facing_loc, agent_nonwall_idx, action] = 1.0
                    else:
                        # trying to face out of bounds, keep current facing location
                        model.B["self_facinglocation_state"][current_facing_loc, current_facing_loc, agent_nonwall_idx, action] = 1.0

    ######### B TENSOR FOR SELF_CARRYING_STATE #########
    # counts of each type of object in the layout for multiplying by slice(None)
    # slice(None) is the same as ":" but it allows us to use it in a tuple
    onion_pile_count = layout_info["objects"]["onion_pile_idx"]["count"]
    plate_pile_count = layout_info["objects"]["plate_pile_idx"]["count"]
    pot_count = layout_info["objects"]["pot_idx"]["count"]
    goal_count = layout_info["objects"]["goal_idx"]["count"]
    total_object_count = onion_pile_count + plate_pile_count + pot_count + goal_count
    
    # transitions for movement-based actions: carrying state stays the same
    for action in ["up", "down", "right", "left", "stay"]:
        for carrying_state in range(model.B["self_carrying_state"].data.shape[0]):
            indices = [carrying_state, carrying_state] + [slice(None)] * (3 + total_object_count) + [action]
            model.B["self_carrying_state"][tuple(indices)] = 1.0
    
    # base interact transitions: carrying state stays the same (will be overridden for specific conditions)
    for carrying_state in range(model.B["self_carrying_state"].data.shape[0]):
        indices = [carrying_state, carrying_state] + [slice(None)] * (3 + total_object_count) + ["interact"]
        model.B["self_carrying_state"][tuple(indices)] = 1.0
    
    # pick up onion when facing a location where agent believes an onion pile is, and carrying nothing
    if onion_pile_count > 0:
        for onion_pile_idx in range(onion_pile_count):
            for facing_loc in range(layout_info["all_locations"]):
                # create pile states with this specific onion pile at this location
                pile_states = [slice(None)] * total_object_count
                pile_states[onion_pile_idx] = facing_loc  # this onion pile believed to be at this location
                
                # can pick up onion when facing this location, onion pile is believed to be here, and carrying nothing
                indices = ["onion", "nothing", facing_loc, slice(None), slice(None)] + pile_states + ["interact"]
                model.B["self_carrying_state"][tuple(indices)] = 1.0
                
                # clear the default "stay same" transition for this specific case
                indices_clear = ["nothing", "nothing", facing_loc, slice(None), slice(None)] + pile_states + ["interact"]
                model.B["self_carrying_state"][tuple(indices_clear)] = 0.0

    # pick up plate when facing a location where agent believes a plate pile is, and carrying nothing
    if plate_pile_count > 0:
        for plate_pile_idx in range(plate_pile_count):
            for facing_loc in range(layout_info["all_locations"]):
                # create pile states with this specific plate pile at this location
                pile_states = [slice(None)] * total_object_count
                # onion piles come first, then plate piles
                pile_states[onion_pile_count + plate_pile_idx] = facing_loc  
                
                # can pick up plate when facing this location, plate pile is believed to be here, and carrying nothing
                indices = ["plate_empty", "nothing", facing_loc, slice(None), slice(None)] + pile_states + ["interact"]
                model.B["self_carrying_state"][tuple(indices)] = 1.0
                
                # clear the default "stay same" transition for this specific case
                indices_clear = ["nothing", "nothing", facing_loc, slice(None), slice(None)] + pile_states + ["interact"]
                model.B["self_carrying_state"][tuple(indices_clear)] = 0.0

    # put onion in pot when facing a location where agent believes a pot is, and carrying onion
    for pot_idx in range(pot_count):
        for facing_loc in range(layout_info["all_locations"]):
            # create object states with this specific pot at this location
            object_states = [slice(None)] * total_object_count
            # pot states come after onion piles and plate piles
            object_states[onion_pile_count + plate_pile_count + pot_idx] = facing_loc
            
            # can put onion in pot when facing this location, pot is believed to be here, and carrying onion
            indices = ["nothing", "onion", facing_loc, slice(None), slice(None)] + object_states + ["interact"]
            model.B["self_carrying_state"][tuple(indices)] = 1.0
            
            # clear the default "stay same" transition for this specific case
            indices_clear = ["onion", "onion", facing_loc, slice(None), slice(None)] + object_states + ["interact"]
            model.B["self_carrying_state"][tuple(indices_clear)] = 0.0

    # take soup from done pot when facing a location where agent believes a pot is, carrying plate_empty, and pot is done
    for pot_idx in range(pot_count):
        for facing_loc in range(layout_info["all_locations"]):
            # create object states with this specific pot at this location
            object_states = [slice(None)] * total_object_count
            # pot states come after onion piles and plate piles
            object_states[onion_pile_count + plate_pile_count + pot_idx] = facing_loc
            
            # can take soup when facing this location, pot is believed to be here, carrying plate_empty, and pot is done
            indices = ["plate_full", "plate_empty", facing_loc, "pot_done", slice(None)] + object_states + ["interact"]
            model.B["self_carrying_state"][tuple(indices)] = 1.0
            
            # clear the default "stay same" transition for this specific case
            indices_clear = ["plate_empty", "plate_empty", facing_loc, "pot_done", slice(None)] + object_states + ["interact"]
            model.B["self_carrying_state"][tuple(indices_clear)] = 0.0

    # deliver soup to goal when facing a location where agent believes a goal is, and carrying plate_full
    if goal_count > 0:
        for goal_idx in range(goal_count):
            for facing_loc in range(layout_info["all_locations"]):
                # create object states with this specific goal at this location
                object_states = [slice(None)] * total_object_count
                # goal states come after onion piles, plate piles, and pots
                object_states[onion_pile_count + plate_pile_count + pot_count + goal_idx] = facing_loc
                
                # can deliver soup when facing this location, goal is believed to be here, carrying plate_full, and goal is empty
                indices = ["nothing", "plate_full", facing_loc, slice(None), "goal_empty"] + object_states + ["interact"]
                model.B["self_carrying_state"][tuple(indices)] = 1.0
                
                # clear the default "stay same" transition for this specific case
                indices_clear = ["plate_full", "plate_full", facing_loc, slice(None), "goal_empty"] + object_states + ["interact"]
                model.B["self_carrying_state"][tuple(indices_clear)] = 0.0

    ######### B TENSOR FOR POT_STATE #########
    # default transitions: pot state stays the same for all actions
    for pot_state in range(model.B["pot_state"].data.shape[0]):
        indices = [pot_state, pot_state] + [slice(None)] * (2 + pot_count) + [slice(None)]
        model.B["pot_state"][tuple(indices)] = 1.0

    # transitions for adding onions to pot when agent interacts while carrying onion and facing pot location
    pot_state_names = ["pot_empty", "pot_1onion", "pot_2onions", "pot_3onions"]
    
    for current_state_idx in range(len(pot_state_names) - 1):  # iterate through all transitions except the last state
        current_state = pot_state_names[current_state_idx]
        next_state = pot_state_names[current_state_idx + 1]
        
        for pot_idx in range(pot_count):
            for facing_loc in range(layout_info["all_locations"]):
                # create pot location states with this specific pot at this location
                pot_states = [slice(None)] * pot_count
                pot_states[pot_idx] = facing_loc  
                
                # can put onion in pot when the agent is facing pot location and carrying onion
                indices = [next_state, current_state, facing_loc, "onion"] + pot_states + ["interact"]
                model.B["pot_state"][tuple(indices)] = 1.0
                
                # clear the default "stay same" transition for this specific case
                indices_clear = [current_state, current_state, facing_loc, "onion"] + pot_states + ["interact"]
                model.B["pot_state"][tuple(indices_clear)] = 0.0

    # transitions for starting and progressing cooking when agent interacts while carrying nothing and facing pot location
    # start cooking: pot_3onions -> pot_cooking19
    for pot_idx in range(pot_count):
        for facing_loc in range(layout_info["all_locations"]):
            pot_states = [slice(None)] * pot_count
            pot_states[pot_idx] = facing_loc
            
            # start cooking when interacting with 3-onion pot while carrying nothing
            indices = ["pot_cooking19", "pot_3onions", facing_loc, "nothing"] + pot_states + ["interact"]
            model.B["pot_state"][tuple(indices)] = 1.0
            
            # clear the default "stay same" transition
            indices_clear = ["pot_3onions", "pot_3onions", facing_loc, "nothing"] + pot_states + ["interact"]
            model.B["pot_state"][tuple(indices_clear)] = 0.0

    # progress cooking: pot_cooking19 -> pot_cooking1 -> pot_done (numbers represent time remaining)
    cooking_states = [
        "pot_cooking19", "pot_cooking18", "pot_cooking17", "pot_cooking16",
        "pot_cooking15", "pot_cooking14", "pot_cooking13", "pot_cooking12",
        "pot_cooking11", "pot_cooking10", "pot_cooking9", "pot_cooking8",
        "pot_cooking7", "pot_cooking6", "pot_cooking5", "pot_cooking4",
        "pot_cooking3", "pot_cooking2", "pot_cooking1"
    ]
    
    for i in range(len(cooking_states)):
        current_cooking_state = cooking_states[i]
        next_cooking_state = "pot_done" if i == len(cooking_states) - 1 else cooking_states[i + 1]
        
        for pot_idx in range(pot_count):
            for facing_loc in range(layout_info["all_locations"]):
                pot_states = [slice(None)] * pot_count
                pot_states[pot_idx] = facing_loc
                
                # progress cooking when interacting with cooking pot while carrying nothing
                indices = [next_cooking_state, current_cooking_state, facing_loc, "nothing"] + pot_states + ["interact"]
                model.B["pot_state"][tuple(indices)] = 1.0
                
                # clear the default "stay same" transition
                indices_clear = [current_cooking_state, current_cooking_state, facing_loc, "nothing"] + pot_states + ["interact"]
                model.B["pot_state"][tuple(indices_clear)] = 0.0

    ######### B TENSOR FOR POT_DONE_FLAG_STATE #########
    # pot done flag state depends on pot state
    action_names = ["up", "down", "right", "left", "stay", "interact"]
    pot_state_names = [
        "pot_empty", "pot_1onion", "pot_2onions", "pot_3onions",
        "pot_cooking19", "pot_cooking18", "pot_cooking17", "pot_cooking16",
        "pot_cooking15", "pot_cooking14", "pot_cooking13", "pot_cooking12",
        "pot_cooking11", "pot_cooking10", "pot_cooking9", "pot_cooking8",
        "pot_cooking7", "pot_cooking6", "pot_cooking5", "pot_cooking4",
        "pot_cooking3", "pot_cooking2", "pot_cooking1", "pot_done",
    ]
    
    # clear the tensor first
    model.B["pot_done_flag_state"].data = jnp.zeros_like(model.B["pot_done_flag_state"].data)
    
    # B tensor dimensions: [next_pot_done_flag_state, current_pot_done_flag_state, current_pot_state, action]
    for current_pot_done_flag_state in range(model.B["pot_done_flag_state"].data.shape[1]):
        for current_pot_state in range(model.B["pot_done_flag_state"].data.shape[2]):
            # determine next pot done flag state based on current pot state
            if current_pot_state == pot_state_names.index("pot_done"):
                next_pot_done_flag_state = 1  # "pot_done"
            else:
                next_pot_done_flag_state = 0  # "pot_not_done"
            
            # set transition for all actions
            for action_idx, action in enumerate(action_names):
                model.B["pot_done_flag_state"].data = model.B["pot_done_flag_state"].data.at[next_pot_done_flag_state, current_pot_done_flag_state, current_pot_state, action_idx].set(1.0)

    ######### B TENSOR FOR GOAL_DELIVERED_STATE #########
    # default: goal state stays the same for movement actions
    for action in ["up", "down", "right", "left", "stay"]:
        for goal_state in range(2):  # goal_empty, goal_delivered
            indices = [goal_state, goal_state, slice(None), slice(None), action]
            model.B["goal_delivered_state"][tuple(indices)] = 1.0
    
    # base interact: goal state stays the same (will be overridden for delivery)
    for goal_state in range(2):
        indices = [goal_state, goal_state, slice(None), slice(None), "interact"]
        model.B["goal_delivered_state"][tuple(indices)] = 1.0
    
    # delivery: goal_empty -> goal_delivered when facing goal location with plate_full
    if "goal_idx" in layout_info["objects"]:
        for goal_idx in range(layout_info["objects"]["goal_idx"]["count"]):
            goal_location = layout_info["objects"]["goal_idx"]["indices"][goal_idx]
            
            # when agent faces goal location, carries plate_full, and interacts
            indices = ["goal_delivered", "goal_empty", goal_location, "plate_full", "interact"]
            model.B["goal_delivered_state"][tuple(indices)] = 1.0
            
            # clear default transition for this specific case
            indices_clear = ["goal_empty", "goal_empty", goal_location, "plate_full", "interact"]
            model.B["goal_delivered_state"][tuple(indices_clear)] = 0.0

    ######### B TENSORS FOR OBJECT STATES #########
    # object states transitions are static (uncontrollable)
    for layout_key, obj_info in layout_info["objects"].items():
        for i in range(obj_info["count"]):
            state_name = f"{layout_key}{i}_state"
            for loc in range(layout_info["all_locations"]):
                model.B[state_name][loc, loc, "noop"] = 1.0

    '''
    BUILDING THE C TENSOR.     
    '''

    # model.C["self_location_obs"][5] = 1.0
    model.C["self_carrying_obs"][1] = 1.0 # agent carrying onion
    model.C["self_carrying_obs"][2] = 1.0 # just to distract agent 1 to carry plate
    model.C["pot_obs"][1] = 2.0 # pot has 1 onion
    model.C["pot_obs"][2] = 10.0 # pot has 2 onions
    model.C["pot_obs"][3] = 20.0 # pot has 3 onions
    
    # add preferences for cooking states - higher preference as cooking progresses
    cooking_state_names = [
        "pot_cooking19", "pot_cooking18", "pot_cooking17", "pot_cooking16",
        "pot_cooking15", "pot_cooking14", "pot_cooking13", "pot_cooking12",
        "pot_cooking11", "pot_cooking10", "pot_cooking9", "pot_cooking8",
        "pot_cooking7", "pot_cooking6", "pot_cooking5", "pot_cooking4",
        "pot_cooking3", "pot_cooking2", "pot_cooking1"
    ]
    
    # assign increasing preferences for cooking progression
    base_cooking_preference = 30.0
    for i, cooking_state in enumerate(cooking_state_names):
        # get the index in the full pot_obs elements list
        pot_obs_elements = [
            "pot_empty", "pot_1onion", "pot_2onions", "pot_3onions",
            "pot_cooking19", "pot_cooking18", "pot_cooking17", "pot_cooking16",
            "pot_cooking15", "pot_cooking14", "pot_cooking13", "pot_cooking12",
            "pot_cooking11", "pot_cooking10", "pot_cooking9", "pot_cooking8",
            "pot_cooking7", "pot_cooking6", "pot_cooking5", "pot_cooking4",
            "pot_cooking3", "pot_cooking2", "pot_cooking1", "pot_done",
        ]
        cooking_idx = pot_obs_elements.index(cooking_state)
        # higher preference as we get closer to done (i increases as cooking progresses)
        model.C["pot_obs"][cooking_idx] = base_cooking_preference + i * 2.0
    
    # highest preference for pot_done
    pot_done_idx = pot_obs_elements.index("pot_done")
    model.C["pot_obs"][pot_done_idx] = 100.0
    model.C["pot_done_flag_obs"][1] = 100.0

    model.C["self_carrying_obs"][-1] = 110.0 # carry plate_full
    
    # add highest preference for goal delivery
    model.C["goal_delivered_obs"]["goal_delivered"] = 200.0  # highest preference

    '''
    NORMLISING TENSORS
    '''

    for key in model.A.keys():
        model.A[key].normalize()

    for key in model.B.keys():
        model.B[key].normalize()


    return model
