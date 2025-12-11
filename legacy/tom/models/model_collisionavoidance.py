import jax.numpy as jnp
import numpy as np

from pymdp.distribution import compile_model
from pymdp.agent import Agent


def CollisionAvoidanceAgent(model, gamma=8.0, batch_size=1):

    # (9 actions of noop, up, down, left, right, upleft, upright, downleft, downright; 2 state factors)
    policies = jnp.zeros((9, 1, 2), dtype=jnp.int32) # action of noop at all state factors

    # move
    for i in range(9):
        policies = policies.at[i, 0, 0].set(i) # actions of moving are only for the 1st state factor
        
    agent = Agent(**model, batch_size=batch_size, policies=policies, learn_A=False, learn_B=False, gamma=gamma, sampling_mode="full")
    return agent

def CollisionAvoidanceModel(agent_idx):
    locations = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "hell_state"]

    model_description = {
        "observations": {
            "self_location_obs": {"elements": locations,
                             "depends_on": ["self_location_state"],
            },
            "other_location_obs": {"elements": locations,
                             "depends_on": ["other_location_state"],
            },
        },
        "controls": {
            "move": {"elements": ["noop", "up", "down", "left", "right", "upleft", "upright", "downleft", "downright"],
            },
            "uncontrollable": {"elements": ["noop"],
            },
        },
        "states": {
            "self_location_state": {"elements": locations,
                               "depends_on": ["self_location_state", "other_location_state"],
                               "controlled_by": ["move"],
            },
            "other_location_state": {"elements": locations,
                               "depends_on": ["other_location_state"],
                               "controlled_by": ["uncontrollable"],
            },
        },
    }

    model = compile_model(model_description)
    '''
    BUILDING THE A TENSOR
    '''
    model.A["self_location_obs"].data = jnp.eye(len(locations))
    model.A["other_location_obs"].data = jnp.eye(len(locations))

    '''
    BUILDING THE B TENSOR
    '''

    # for moving between locations in a 3x3 grid where location 0 is the top left and location 8 is the bottom right
    valid_transitions = [
        # (to, from, action)
        # 0 is the top left and 8 is the bottom right
        # from 0
        ("0", "0", "noop"),
        ("hell_state", "0", "up"),
        ("3", "0", "down"),
        ("hell_state", "0", "left"),
        ("1", "0", "right"),
        ("hell_state", "0", "upleft"),
        ("hell_state", "0", "upright"),
        ("hell_state", "0", "downleft"),
        ("4", "0", "downright"),
        
        # from 1
        ("1", "1", "noop"),
        ("hell_state", "1", "up"),
        ("4", "1", "down"),
        ("0", "1", "left"),
        ("2", "1", "right"),
        ("hell_state", "1", "upleft"),
        ("hell_state", "1", "upright"),
        ("3", "1", "downleft"),
        ("5", "1", "downright"),

        # from 2
        ("2", "2", "noop"),
        ("hell_state", "2", "up"),
        ("5", "2", "down"),
        ("1", "2", "left"),
        ("hell_state", "2", "right"),
        ("hell_state", "2", "upleft"),
        ("hell_state", "2", "upright"),
        ("4", "2", "downleft"),
        ("hell_state", "2", "downright"),

        # from 3
        ("3", "3", "noop"),
        ("0", "3", "up"),
        ("6", "3", "down"),
        ("hell_state", "3", "left"),
        ("4", "3", "right"),
        ("hell_state", "3", "upleft"),
        ("1", "3", "upright"),
        ("hell_state", "3", "downleft"),
        ("7", "3", "downright"),

        # from 4
        ("4", "4", "noop"),
        ("1", "4", "up"),
        ("7", "4", "down"),
        ("3", "4", "left"),
        ("5", "4", "right"),
        ("0", "4", "upleft"),
        ("2", "4", "upright"),
        ("6", "4", "downleft"),
        ("8", "4", "downright"),

        # from 5
        ("5", "5", "noop"),
        ("2", "5", "up"),
        ("8", "5", "down"),
        ("4", "5", "left"),
        ("hell_state", "5", "right"),
        ("1", "5", "upleft"),
        ("hell_state", "5", "upright"),
        ("7", "5", "downleft"),
        ("hell_state", "5", "downright"),

        # from 6
        ("6", "6", "noop"),
        ("3", "6", "up"),
        ("hell_state", "6", "down"),
        ("hell_state", "6", "left"),
        ("7", "6", "right"),
        ("hell_state", "6", "upleft"),
        ("4", "6", "upright"),
        ("hell_state", "6", "downleft"),
        ("hell_state", "6", "downright"),

        # from 7
        ("7", "7", "noop"),
        ("4", "7", "up"),
        ("hell_state", "7", "down"),
        ("6", "7", "left"),
        ("8", "7", "right"),
        ("3", "7", "upleft"),
        ("5", "7", "upright"),
        ("hell_state", "7", "downleft"),
        ("hell_state", "7", "downright"),

        # from 8
        ("8", "8", "noop"),
        ("5", "8", "up"),
        ("hell_state", "8", "down"),
        ("7", "8", "left"),
        ("hell_state", "8", "right"),
        ("4", "8", "upleft"),
        ("hell_state", "8", "upright"),
        ("hell_state", "8", "downleft"),
        ("hell_state", "8", "downright"),
    ]

    for to_state, from_state, action in valid_transitions:
        model.B["self_location_state"][to_state, from_state, :, action] = 1.0
    
    # but, if the focal agent's location is the same as the other agent's location, then the focal agent cannot move
    for i in range(len(locations)):
        # first zero out all transitions when self and other are in same location
        model.B["self_location_state"][:, i, i, :] = 0.0
        # then set the transition to stay in same location 
        model.B["self_location_state"][i, i, i, :] = 1.0
    
    # regardless of action, the agent will get stuck in the hell state
    model.B["self_location_state"]["hell_state", "hell_state", :, :] = 1.0

    # for the other agent
    model.B["other_location_state"][:, :, "noop"] = 0.0
    valid_destinations = {}
    for to_state, from_state, _ in valid_transitions:
        if from_state not in valid_destinations:
            valid_destinations[from_state] = []
        if to_state != "hell_state":
            valid_destinations[from_state].append(to_state)

    # set uniform probability for all valid transitions
    for from_state, to_states in valid_destinations.items():
        probability = 1.0 / len(to_states)
        for to_state in to_states:
            model.B["other_location_state"][to_state, from_state, "noop"] = probability
    
    # regardless of action, the agent will get stuck in the hell state
    model.B["other_location_state"]["hell_state", "hell_state", :] = 1.0
    
    '''
    BUILDING THE C TENSOR - agent-specific preferences based on initial positions
    '''
    if agent_idx == 0: 
        model.C["self_location_obs"][8] = 10.0
        model.C["other_location_obs"][0] = 10.0
    elif agent_idx == 1:
        model.C["self_location_obs"][0] = 10.0
        model.C["other_location_obs"][8] = 10.0
    
    model.C["self_location_obs"]["hell_state"] = -100
    model.C["other_location_obs"]["hell_state"] = -100
    

    return model