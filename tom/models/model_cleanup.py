import jax.numpy as jnp
import numpy as np

from pymdp.distribution import compile_model
from pymdp.agent import Agent


def CleanUpAgent(model, gamma=8.0, batch_size=1):

    # (7 actions of noop, up, down, left, right, eat, clean; 1; 8 state factors)
    policies = jnp.zeros((7, 1, 8), dtype=jnp.int32) # action of noop at all state factors

    # move
    for i in range(5):
        policies = policies.at[i, 0, 0].set(i) # actions of moving up, down, left, right for the 1st state factor
    # eat
    policies = policies.at[5, 0, 1].set(1) # action of eat at 2nd state factor so then agent can get a reward when it eats an apple
    policies = policies.at[5, 0, 5:8].set(1) # action of eat at last three state factors re: bottom row
    # clean
    policies = policies.at[6, 0, 1].set(2) # action of clean at 2nd state factor to ensure consistent size and for visualisation
    policies = policies.at[6, 0, 2:5].set(2) # action of clean at 3rd to 5th state factors re: top row

    agent = Agent(**model, batch_size=batch_size, policies=policies, learn_A=False, learn_B=False, gamma=gamma, sampling_mode="full")
    return agent


def CleanUpModel(pollution_rate):
    num_locations = 9
    locations = [str(i) for i in range(num_locations)] + ["hell"]
    item_obs_list = ["wasteland", "river_clean", "river_dirty", "apple", "orchard"]

    model_description = {
        "observations": {
            "location_obs": {"elements": locations,
                             "depends_on": ["location_state"],
            },
            "item_obs": {"elements": item_obs_list,
                         "depends_on": ["location_state", 
                                        "loc0_state", "loc1_state", "loc2_state", 
                                        "loc6_state", "loc7_state", "loc8_state"],
            },
            "reward_obs": {"elements": ["no_reward", "reward"],
                           "depends_on": ["reward_state"],
            },
        },
        "controls": {
            "move": {"elements": ["noop", "up", "down", "left", "right"],
            },
            "actions": {"elements": ["noop", "eat", "clean"],
            },
        },
        "states": {
            "location_state": {"elements": locations,
                               "depends_on": ["location_state"],
                               "controlled_by": ["move"],
            },
            "reward_state": {"elements": ["no_reward", "reward"],
                             "depends_on": ["reward_state", "location_state", 
                                            "loc6_state", "loc7_state", "loc8_state"],
                             "controlled_by": ["actions"],
            },
            "loc0_state": {"elements": ["river_clean", "river_dirty"],
                           "depends_on": ["loc0_state", "location_state"],
                           "controlled_by": ["actions"],
            },
            "loc1_state": {"elements": ["river_clean", "river_dirty"],
                           "depends_on": ["loc1_state", "location_state"],
                           "controlled_by": ["actions"],
            },
            "loc2_state": {"elements": ["river_clean", "river_dirty"],
                           "depends_on": ["loc2_state", "location_state"],
                           "controlled_by": ["actions"],
            },
            "loc6_state": {"elements": ["apple", "orchard"],
                           "depends_on": ["loc6_state", "location_state",
                                          "loc0_state", "loc1_state", "loc2_state"],
                            "controlled_by": ["actions"],
            },
            "loc7_state": {"elements": ["apple", "orchard"],
                           "depends_on": ["loc7_state", "location_state",
                                          "loc0_state", "loc1_state", "loc2_state"],
                           "controlled_by": ["actions"],
            },
            "loc8_state": {"elements": ["apple", "orchard"],
                           "depends_on": ["loc8_state", "location_state",
                                          "loc0_state", "loc1_state", "loc2_state"],
                            "controlled_by": ["actions"],
            },
        },
    }

    model = compile_model(model_description)

    '''
    BUILDING THE A TENSOR
    '''
    model.A["location_obs"].data = jnp.eye(len(locations))

    # if the agent is in locations 3, 4, 5, it will observe wasteland
    model.A["item_obs"]["wasteland", "3", :, :, :, :, :, :] = 1.0
    model.A["item_obs"]["wasteland", "4", :, :, :, :, :, :] = 1.0
    model.A["item_obs"]["wasteland", "5", :, :, :, :, :, :] = 1.0

    # if the agent is in locations 0, 1, 2, it will observe river_clean or river_dirty
    model.A["item_obs"]["river_clean", "0", "river_clean", :, :, :, :, :] = 1.0
    model.A["item_obs"]["river_clean", "1", :, "river_clean", :, :, :, :] = 1.0
    model.A["item_obs"]["river_clean", "2", :, :, "river_clean", :, :, :] = 1.0
    model.A["item_obs"]["river_dirty", "0", "river_dirty", :, :, :, :, :] = 1.0
    model.A["item_obs"]["river_dirty", "1", :, "river_dirty", :, :, :, :] = 1.0
    model.A["item_obs"]["river_dirty", "2", :, :, "river_dirty", :, :, :] = 1.0

    # if the agent is in locations 6, 7, 8, it will observe apple or orchard
    model.A["item_obs"]["apple", "6", :, :, :, "apple", :, :] = 1.0
    model.A["item_obs"]["apple", "7", :, :, :, :, "apple", :] = 1.0
    model.A["item_obs"]["apple", "8", :, :, :, :, :, "apple"] = 1.0
    model.A["item_obs"]["orchard", "6", :, :, :, "orchard", :, :] = 1.0
    model.A["item_obs"]["orchard", "7", :, :, :, :, "orchard", :] = 1.0
    model.A["item_obs"]["orchard", "8", :, :, :, :, :, "orchard"] = 1.0

    model.A["reward_obs"].data = jnp.eye(len(model.A["reward_obs"].data))

    '''
    BUILDING THE B TENSOR
    '''

    # for moving between locations in a 3x3 grid where location 0 is the top left and location 8 is the bottom right
    # (to, from, action)
    valid_transitions = [
        # from 0
        ("0", "0", "noop"),
        ("1", "0", "right"),
        ("3", "0", "down"),
        ("hell", "0", "up"),
        ("hell", "0", "left"),
        # from 1
        ("0", "1", "left"),
        ("1", "1", "noop"),
        ("2", "1", "right"),
        ("4", "1", "down"),
        ("hell", "1", "up"),
        # from 2
        ("1", "2", "left"),
        ("2", "2", "noop"),
        ("5", "2", "down"),
        ("hell", "2", "up"),
        ("hell", "2", "right"),
        # from 3
        ("0", "3", "up"),
        ("3", "3", "noop"),
        ("4", "3", "right"),
        ("6", "3", "down"),
        ("hell", "3", "left"),
        # from 4
        ("1", "4", "up"),
        ("3", "4", "left"),
        ("4", "4", "noop"),
        ("5", "4", "right"),
        ("7", "4", "down"),
        # from 5
        ("2", "5", "up"),
        ("4", "5", "left"),
        ("5", "5", "noop"),
        ("8", "5", "down"),
        ("hell", "5", "right"),
        # from 6
        ("3", "6", "up"),
        ("6", "6", "noop"),
        ("7", "6", "right"),
        ("hell", "6", "left"),
        ("hell", "6", "down"),
        # from 7
        ("4", "7", "up"),
        ("6", "7", "left"),
        ("7", "7", "noop"),
        ("8", "7", "right"),
        ("hell", "7", "down"),
        # from 8
        ("5", "8", "up"),
        ("7", "8", "left"),
        ("8", "8", "noop"),
        ("hell", "8", "right"),
        ("hell", "8", "down"),
        # stay in hell state no matter what action is taken
        ("hell", "hell", "noop"),
        ("hell", "hell", "up"),
        ("hell", "hell", "down"),
        ("hell", "hell", "left"),
        ("hell", "hell", "right"),
    ]
    
    for to_state, from_state, action in valid_transitions:
        model.B["location_state"][to_state, from_state, action] = 1.0

    #  if in location 6, 7, 8, and the agent sees orchard, it does not get a reward regardless of actions
    model.B["reward_state"]["no_reward", "no_reward", "6", "orchard", :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "7", :, "orchard", :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "8", :, :, "orchard", :] = 1.0

    # if in location 6, 7, 8, and the agent does not eat when it sees an apple, it does not get a reward
    model.B["reward_state"]["no_reward", "no_reward", "6", "apple", :, :, "noop"] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "7", :, "apple", :, "noop"] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "8", :, :, "apple", "noop"] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "6", "apple", :, :, "clean"] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "7", :, "apple", :, "clean"] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "8", :, :, "apple", "clean"] = 1.0

    # if in location 6, 7, 8, and the agent eats when it sees an apple, it gets a reward
    model.B["reward_state"]["reward", "no_reward", "6", "apple", :, :, "eat"] = 1.0 
    model.B["reward_state"]["reward", "no_reward", "7", :, "apple", :, "eat"] = 1.0
    model.B["reward_state"]["reward", "no_reward", "8", :, :, "apple", "eat"] = 1.0

    # #  if in location 6, 7, 8, and the agent sees apple or orchard, it goes from reward to no reward regardless of actions
    model.B["reward_state"]["no_reward", "reward", "6", :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "reward", "7", :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "reward", "8", :, :, :, :] = 1.0

    # in all other locations, the agent does not get a reward regardless of actions
    for loc in ["0", "1", "2", "3", "4", "5"]:
        model.B["reward_state"]["no_reward", :, loc, :, :, :, :] = 1.0
    
    # hell gives no reward
    model.B["reward_state"]["no_reward", :, "hell", :, :, :, :] = 1.0

    model.B["reward_state"]["no_reward", "no_reward", "6", "apple", :, :, "eat"] = 0.0
    model.B["reward_state"]["no_reward", "no_reward", "7", :, "apple", :, "eat"] = 0.0
    model.B["reward_state"]["no_reward", "no_reward", "8", :, :, "apple", "eat"] = 0.0


    river_states_locs = ["loc0_state", "loc1_state", "loc2_state"]

    for i, state in enumerate(river_states_locs):
        for agent_location in range(10):
            if i == agent_location:
                # if agent is in the top row (river), and it cleans the cell it is in, the river cell becomes clean
                model.B[state]["river_clean", "river_dirty", agent_location, "clean"] = 1.0
                model.B[state]["river_clean", "river_clean", agent_location, "clean"] = 1.0

                # if the agent does not clean, a dirty river cell has a very small chance of becoming clean - purely to add noise for belief updating
                model.B[state]["river_clean", "river_dirty", agent_location, "noop"] = 0.0
                model.B[state]["river_clean", "river_dirty", agent_location, "eat"] = 0.0

                # if the agent does not clean, the river cell stays dirty
                model.B[state]["river_dirty", "river_dirty", agent_location, "noop"] = 1.0
                model.B[state]["river_dirty", "river_dirty", agent_location, "eat"] = 1.0

                # or the river cell becomes dirty with pollution rate
                model.B[state]["river_dirty", "river_clean", agent_location, "noop"] = pollution_rate 
                model.B[state]["river_dirty", "river_clean", agent_location, "eat"] = pollution_rate
                model.B[state]["river_dirty", "river_clean", agent_location, "clean"] = 0.0 

                model.B[state]["river_clean", "river_clean", agent_location, "noop"] = (1 - pollution_rate)
                model.B[state]["river_clean", "river_clean", agent_location, "eat"] = (1 - pollution_rate)

            else:
                # if the agent is not in the top row (river), the river gets dirty with pollution rate or stays dirty
                model.B[state]["river_dirty", "river_clean", agent_location, :] = pollution_rate
                model.B[state]["river_clean", "river_clean", agent_location, :] = (1 - pollution_rate)
                model.B[state]["river_dirty", "river_dirty", agent_location, :] = 1.0
                model.B[state]["river_clean", "river_dirty", agent_location, :] = 0.0

    orchard_states_locs = ["loc6_state", "loc7_state", "loc8_state"]

    for i, state in enumerate(orchard_states_locs):
        orchard_location = i + 6
        for agent_location in range(10):
            # regardless of the agent's location and actions, apples will grow when pollution level is 0 (=all clean cells) or 1 (=2 clean cells)
            model.B[state]["apple", "orchard", agent_location, "river_clean", "river_clean", "river_clean", :] = 1 / 3
            model.B[state]["orchard", "orchard", agent_location, "river_clean", "river_clean", "river_clean", :] = 1 - (1 / 3)

            model.B[state]["apple", "orchard", agent_location, "river_dirty", "river_clean", "river_clean", :] = 1 / 3
            model.B[state]["apple", "orchard", agent_location, "river_clean", "river_dirty", "river_clean", :] = 1 / 3
            model.B[state]["apple", "orchard", agent_location, "river_clean", "river_clean", "river_dirty", :] = 1 / 3
            model.B[state]["orchard", "orchard", agent_location, "river_dirty", "river_clean", "river_clean", :] = 1 - (1 / 3)
            model.B[state]["orchard", "orchard", agent_location, "river_clean", "river_dirty", "river_clean", :] = 1 - (1 / 3)            
            model.B[state]["orchard", "orchard", agent_location, "river_clean", "river_clean", "river_dirty", :] = 1 - (1 / 3)

            # regardless of the agent's locations and actions, apples will not grow if pollution level is 2 (=1 clean cell) or 3 (=no clean cells)
            model.B[state]["apple", "orchard", agent_location, "river_dirty", "river_dirty", "river_dirty", :] = 0.0
            model.B[state]["orchard", "orchard", agent_location, "river_dirty", "river_dirty", "river_dirty", :] = 1.0

            model.B[state]["apple", "orchard", agent_location, "river_clean", "river_dirty", "river_dirty", :] = 0.0
            model.B[state]["apple", "orchard", agent_location, "river_dirty", "river_clean", "river_dirty", :] = 0.0
            model.B[state]["apple", "orchard", agent_location, "river_dirty", "river_dirty", "river_clean", :] = 0.0
            model.B[state]["orchard", "orchard", agent_location, "river_clean", "river_dirty", "river_dirty", :] = 1.0            
            model.B[state]["orchard", "orchard", agent_location, "river_dirty", "river_clean", "river_dirty", :] = 1.0            
            model.B[state]["orchard", "orchard", agent_location, "river_dirty", "river_dirty", "river_clean", :] = 1.0

            if agent_location == orchard_location: 
                # if the agent is in the bottom row (orchard), and there is an apple in the cell it is in, and it eats it, the apple is gone
                model.B[state]["orchard", "apple", agent_location, :, :, :, "eat"] = 1.0
                # if it does not eat, the apple stays
                model.B[state]["apple", "apple", agent_location, :, :, :, "noop"] = 1.0
                model.B[state]["apple", "apple", agent_location, :, :, :, "clean"] = 1.0
            else: 
                model.B[state]["apple", "apple", agent_location, :, :, :, :] = 1.0

    '''
    BUILDING THE C TENSOR. 
    
    note: no need for D tensor as the agents can start anywhere in the middle row of the grid.
    '''

    model.C["reward_obs"]["reward"] = 10.0
    model.C["location_obs"]["hell"] = -666

    model.D["reward_state"]["no_reward"] = 1.0
    model.D["reward_state"]["reward"] = 0.0

    model.D["loc6_state"]["orchard"] = 1.0
    model.D["loc6_state"]["apple"] = 0.0
    model.D["loc7_state"]["orchard"] = 1.0
    model.D["loc7_state"]["apple"] = 0.0
    model.D["loc8_state"]["orchard"] = 1.0
    model.D["loc8_state"]["apple"] = 0.0

    model.D["loc0_state"]["river_dirty"] = 1.0
    model.D["loc0_state"]["river_clean"] = 0.0
    model.D["loc1_state"]["river_dirty"] = 1.0
    model.D["loc1_state"]["river_clean"] = 0.0
    model.D["loc2_state"]["river_dirty"] = 1.0
    model.D["loc2_state"]["river_clean"] = 0.0

    '''
    broadcasting agent parameters to batch_size
    '''

    model.A["location_obs"].normalize()
    # add noise here to avoid zero-column tensors
    model.A["item_obs"].data += 1e-3
    model.A["item_obs"].normalize()
    model.A["reward_obs"].normalize()
    model.B["location_state"].normalize()
    model.B["reward_state"].normalize()
    model.B["loc0_state"].normalize()
    model.B["loc1_state"].normalize()
    model.B["loc2_state"].normalize()
    model.B["loc6_state"].normalize()
    model.B["loc7_state"].normalize()
    model.B["loc8_state"].normalize()
    model.D["reward_state"].normalize()
    model.D["loc6_state"].normalize()
    model.D["loc7_state"].normalize()
    model.D["loc8_state"].normalize()
    model.D["loc0_state"].normalize()
    model.D["loc1_state"].normalize()
    model.D["loc2_state"].normalize()

    return model
