import jax.numpy as jnp
import numpy as np

from pymdp.distribution import compile_model
from pymdp.agent import Agent

def ForagingAgent(model, gamma=8.0, batch_size=1):

    # (6 actions of noop, up, down, left, right, eat; 1; 8 state factors)
    policies = jnp.zeros((6, 1, 8), dtype=jnp.int32) # action of noop at all state factors

    # move
    for i in range(5):
        policies = policies.at[i, 0, 0].set(i) # actions of moving up, down, left, right for the 1st state factor
    # eat
    policies = policies.at[5, 0, 1].set(1) # action of eat at 2nd state factor so then agent can get a reward when it eats an apple
    policies = policies.at[5, 0, 2:8].set(1) # action of eat at 3rd to last state factors re: top and bottom row
    
    agent = Agent(**model, batch_size=batch_size, policies=policies, learn_A=False, learn_B=False, gamma=gamma, sampling_mode="full")
    return agent


def ForagingModel(apple_spawn_rate):
    num_locations = 9
    locations = [str(i) for i in range(num_locations)] + ["hell"]
    item_obs_list = ["wasteland", "apple", "orchard"]

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
            "actions": {"elements": ["noop", "eat"],
            },
        },
        "states": {
            "location_state": {"elements": locations,
                               "depends_on": ["location_state"],
                               "controlled_by": ["move"],
            },
            "reward_state": {"elements": ["no_reward", "reward"],
                             "depends_on": ["reward_state", "location_state", 
                                            "loc0_state", "loc1_state", "loc2_state",
                                            "loc6_state", "loc7_state", "loc8_state"],
                             "controlled_by": ["actions"],
            },
            "loc0_state": {"elements": ["apple", "orchard"],
                           "depends_on": ["loc0_state", "location_state"],
                           "controlled_by": ["actions"],
            },
            "loc1_state": {"elements": ["apple", "orchard"],
                           "depends_on": ["loc1_state", "location_state"],
                           "controlled_by": ["actions"],
            },
            "loc2_state": {"elements": ["apple", "orchard"],
                           "depends_on": ["loc2_state", "location_state"],
                           "controlled_by": ["actions"],
            },
            "loc6_state": {"elements": ["apple", "orchard"],
                           "depends_on": ["loc6_state", "location_state"],
                            "controlled_by": ["actions"],
            },
            "loc7_state": {"elements": ["apple", "orchard"],
                           "depends_on": ["loc7_state", "location_state"],
                           "controlled_by": ["actions"],
            },
            "loc8_state": {"elements": ["apple", "orchard"],
                           "depends_on": ["loc8_state", "location_state"],
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
    model.A["item_obs"]["apple", "0", "apple", :, :, :, :, :] = 1.0
    model.A["item_obs"]["apple", "1", :, "apple", :, :, :, :] = 1.0
    model.A["item_obs"]["apple", "2", :, :, "apple", :, :, :] = 1.0
    model.A["item_obs"]["orchard", "0", "orchard", :, :, :, :, :] = 1.0
    model.A["item_obs"]["orchard", "1", :, "orchard", :, :, :, :] = 1.0
    model.A["item_obs"]["orchard", "2", :, :, "orchard", :, :, :] = 1.0

    # if the agent is in locations 6, 7, 8, it will observe apple or orchard
    model.A["item_obs"]["apple", "6", :, :, :, "apple", :, :] = 1.0
    model.A["item_obs"]["apple", "7", :, :, :, :, "apple", :] = 1.0
    model.A["item_obs"]["apple", "8", :, :, :, :, :, "apple"] = 1.0
    model.A["item_obs"]["orchard", "6", :, :, :, "orchard", :, :] = 1.0
    model.A["item_obs"]["orchard", "7", :, :, :, :, "orchard", :] = 1.0
    model.A["item_obs"]["orchard", "8", :, :, :, :, :, "orchard"] = 1.0

    model.A["item_obs"].data = model.A["item_obs"].data + 1e-3
    


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



    #  if in orchard locations, and the agent sees orchard, it does not get a reward regardless of actions
    model.B["reward_state"]["no_reward", "no_reward", "0", "orchard", :, :, :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "1", :, "orchard", :, :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "2", :, :, "orchard", :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "6", :, :, :, "orchard", :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "7", :, :, :, :, "orchard", :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "8", :, :, :, :, :, "orchard", :] = 1.0
    
    # if in apple locations, and the agent does not eat when it sees an apple, it does not get a reward
    model.B["reward_state"]["no_reward", "no_reward", "0", "apple", :, :, :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "1", :, "apple", :, :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "2", :, :, "apple", :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "6", :, :, :, "apple", :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "7", :, :, :, :, "apple", :, :] = 1.0
    model.B["reward_state"]["no_reward", "no_reward", "8", :, :, :, :, :, "apple", :] = 1.0
    
    

    # if in apple locations, and the agent eats when it sees an apple, it gets a reward
    model.B["reward_state"]["reward", "no_reward", "0", "apple", :, :, :, :, :, "eat"] = 1.0 
    model.B["reward_state"]["reward", "no_reward", "1", :, "apple", :, :, :, :, "eat"] = 1.0
    model.B["reward_state"]["reward", "no_reward", "2", :, :, "apple", :, :, :, "eat"] = 1.0
    model.B["reward_state"]["reward", "no_reward", "6", :, :, :, "apple", :, :, "eat"] = 1.0 
    model.B["reward_state"]["reward", "no_reward", "7", :, :, :, :, "apple", :, "eat"] = 1.0
    model.B["reward_state"]["reward", "no_reward", "8", :, :, :, :, :, "apple", "eat"] = 1.0
    
    
    # if in orchard/apple locations, and the agent sees apple or orchard, it goes from reward to no reward regardless of actions
    model.B["reward_state"]["no_reward", "reward", "0", :, :, :, :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "reward", "1", :, :, :, :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "reward", "2", :, :, :, :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "reward", "6", :, :, :, :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "reward", "7", :, :, :, :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", "reward", "8", :, :, :, :, :, :, :] = 1.0

    # in all other locations, the agent does not get a reward regardless of actions
    model.B["reward_state"]["no_reward", :, "3", :, :, :, :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", :, "4", :, :, :, :, :, :, :] = 1.0
    model.B["reward_state"]["no_reward", :, "5", :, :, :, :, :, :, :] = 1.0

    # cleaning; if in apple locations, and the agent eats when it sees an apple, you will not go from no reward to no reward (bc you will get a reward)
    model.B["reward_state"]["no_reward", "no_reward", "0", "apple", :, :, :, :, :, "eat"] = 0.0
    model.B["reward_state"]["no_reward", "no_reward", "1", :, "apple", :, :, :, :, "eat"] = 0.0
    model.B["reward_state"]["no_reward", "no_reward", "2", :, :, "apple", :, :, :, "eat"] = 0.0
    model.B["reward_state"]["no_reward", "no_reward", "6", :, :, :, "apple", :, :, "eat"] = 0.0
    model.B["reward_state"]["no_reward", "no_reward", "7", :, :, :, :, "apple", :, "eat"] = 0.0
    model.B["reward_state"]["no_reward", "no_reward", "8", :, :, :, :, :, "apple", "eat"] = 0.0

    # hell gives no reward
    model.B["reward_state"]["no_reward", :, "hell", :, :, :, :] = 1.0

    model.B["reward_state"]["reward", "no_reward", "0", "orchard", :, :, :, :, :, "eat"] = 0.0
    model.B["reward_state"]["reward", "no_reward", "1", :, "orchard", :, :, :, :, "eat"] = 0.0
    model.B["reward_state"]["reward", "no_reward", "2", :, :, "orchard", :, :, :, "eat"] = 0.0
    model.B["reward_state"]["reward", "no_reward", "6", :, :, :, "orchard", :, :, "eat"] = 0.0
    model.B["reward_state"]["reward", "no_reward", "7", :, :, :, :, "orchard", :, "eat"] = 0.0
    model.B["reward_state"]["reward", "no_reward", "8", :, :, :, :, :, "orchard", "eat"] = 0.0



    apple_locs_toprow = ["loc0_state", "loc1_state", "loc2_state"]

    for i, state in enumerate(apple_locs_toprow):
        for agent_location in range(10):
            if i == agent_location:
                # when agent is in the top row
                # orchard -> orchard (no spawn): 1 - apple_spawn_rate
                model.B[state]["orchard", "orchard", agent_location, "noop"] = 1.0 - apple_spawn_rate
                model.B[state]["orchard", "orchard", agent_location, "eat"] = 1.0 - apple_spawn_rate
                # orchard -> apple (spawn): apple_spawn_rate
                model.B[state]["apple", "orchard", agent_location, "noop"] = apple_spawn_rate
                model.B[state]["apple", "orchard", agent_location, "eat"] = apple_spawn_rate
                # apple -> apple (stay, no eat): 1.0
                model.B[state]["apple", "apple", agent_location, "noop"] = 1.0
                # apple -> orchard (eaten): 1.0
                model.B[state]["orchard", "apple", agent_location, "eat"] = 1.0
            else:
                # when agent is elsewhere,
                # orchard -> orchard (no spawn): 1 - apple_spawn_rate
                model.B[state]["orchard", "orchard", agent_location, "noop"] = 1.0 - apple_spawn_rate
                model.B[state]["orchard", "orchard", agent_location, "eat"] = 1.0 - apple_spawn_rate
                # orchard -> apple (spawn): apple_spawn_rate
                model.B[state]["apple", "orchard", agent_location, "noop"] = apple_spawn_rate
                model.B[state]["apple", "orchard", agent_location, "eat"] = apple_spawn_rate
                # apple -> apple (stay): 1.0
                model.B[state]["apple", "apple", agent_location, "noop"] = 1.0
                model.B[state]["apple", "apple", agent_location, "eat"] = 1.0
                # apple -> orchard: never
                model.B[state]["orchard", "apple", agent_location, "noop"] = 0.0
                model.B[state]["orchard", "apple", agent_location, "eat"] = 0.0


    apple_locs_botrow = ["loc6_state", "loc7_state", "loc8_state"]

    for i, state in enumerate(apple_locs_botrow):
        agent_location_idx = i + 6 
        for agent_location in range(10):
            if agent_location_idx == agent_location:
                # when agent is in the bottom row
                # orchard -> orchard (no spawn): 1 - apple_spawn_rate
                model.B[state]["orchard", "orchard", agent_location, "noop"] = 1.0 - apple_spawn_rate
                model.B[state]["orchard", "orchard", agent_location, "eat"] = 1.0 - apple_spawn_rate
                # orchard -> apple (spawn): apple_spawn_rate
                model.B[state]["apple", "orchard", agent_location, "noop"] = apple_spawn_rate
                model.B[state]["apple", "orchard", agent_location, "eat"] = apple_spawn_rate
                # apple -> apple (stay, no eat): 1.0
                model.B[state]["apple", "apple", agent_location, "noop"] = 1.0
                # apple -> orchard (eaten): 1.0
                model.B[state]["orchard", "apple", agent_location, "eat"] = 1.0
            else:
                # when agent is elsewhere
                # orchard -> orchard (no spawn): 1 - apple_spawn_rate
                model.B[state]["orchard", "orchard", agent_location, "noop"] = 1.0 - apple_spawn_rate
                model.B[state]["orchard", "orchard", agent_location, "eat"] = 1.0 - apple_spawn_rate
                # orchard -> apple (spawn): apple_spawn_rate
                model.B[state]["apple", "orchard", agent_location, "noop"] = apple_spawn_rate
                model.B[state]["apple", "orchard", agent_location, "eat"] = apple_spawn_rate
                # apple -> apple (stay): 1.0
                model.B[state]["apple", "apple", agent_location, "noop"] = 1.0
                model.B[state]["apple", "apple", agent_location, "eat"] = 1.0
                # apple -> orchard: never
                model.B[state]["orchard", "apple", agent_location, "noop"] = 0.0
                model.B[state]["orchard", "apple", agent_location, "eat"] = 0.0

    '''
    BUILDING THE C TENSOR. 
    '''

    model.C["reward_obs"]["reward"] = 10.0
    model.C["location_obs"]["hell"] = -666

    '''
    BUILDING THE D TENSOR. 
    '''
    model.D["reward_state"]["reward"] = 0.0
    model.D["reward_state"]["no_reward"] = 1.0

    # top row initial states
    model.D["loc0_state"]["apple"] = 0.5
    model.D["loc0_state"]["orchard"] = 0.5
    model.D["loc1_state"]["apple"] = 0.5
    model.D["loc1_state"]["orchard"] = 0.5
    model.D["loc2_state"]["apple"] = 0.5
    model.D["loc2_state"]["orchard"] = 0.5
    
    # bottom row initial states
    model.D["loc6_state"]["apple"] = 0.5
    model.D["loc6_state"]["orchard"] = 0.5
    model.D["loc7_state"]["apple"] = 0.5
    model.D["loc7_state"]["orchard"] = 0.5
    model.D["loc8_state"]["apple"] = 1.0
    model.D["loc8_state"]["orchard"] = 0.0

    '''
    BROADCASTING AGENT PARAMETERS TO BATCH SIZE
    '''

    model.A["location_obs"].normalize()
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
    
    return model