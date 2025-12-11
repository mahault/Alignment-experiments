import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.nn as nn
import equinox as eqx

from tom.planning.si import si_policy_search, root_idx
from tom.planning.rollout import infer_and_plan_recycle_tree
from tom.planning.si_tom import si_policy_search_tom

from tom.planning.si import Tree
from tom.planning.si_tom import Tree as ToMTree

from pymdp.control import sample_policy
from pymdp.maths import compute_log_likelihood_per_modality, log_stable
from pymdp.algos import all_marginal_log_likelihood
import jax.lax as lax


def to_one_hot(obs, tom_agent, focal_agent_idx):
    o_one_hot = []
    num_agents = obs[0].shape[0]
    
    # ordering the agent indexes using roll such that:
    # focal_agent_idx=0: [0,1] 
    # focal_agent_idx=1: [1,0]
    agent_order = jnp.roll(jnp.arange(num_agents), -focal_agent_idx)
    
    for i in range(len(obs)):  # for each observation modality
        obs_modality = []
        
        for idx in range(num_agents):  # for each agent position
            # Get the actual agent index from the reordered sequence
            agent_idx = agent_order[idx]
            
            # getting the observation mapping for this focal agent and its other agent
            # bc 0th position is always "self" and 1st+ positions are always "others"
            # TODO: this is hardcoded for 2 agents, need to generalise
            mapping = tom_agent.observation_mappings[focal_agent_idx, idx, i]
            
            def uniform_case():
                num_classes = tom_agent.agent_models.num_obs[i]
                return jnp.ones((1, num_classes)) / num_classes
            
            def one_hot_case():
                return nn.one_hot(obs[i][agent_idx], num_classes=tom_agent.agent_models.num_obs[i])
            
            one_hotting = lax.cond(
                mapping == -1,
                uniform_case,
                one_hot_case
            )
            obs_modality.append(one_hotting)
        
        # stack all agents for this observation modality
        o_one_hot.append(jnp.stack(obs_modality, axis=0))
    
    o_one_hot_expanded = [o[None,...] for o in o_one_hot]
    
    return o_one_hot_expanded


def infer_and_belief_share(agent, observation, empirical_prior, world_states):
    # custom impl of fpi where we share likelihood messages of other's world states
    A = agent.A
    A_dependencies = agent.A_dependencies
    #jax.debug.print("empirical prior {q}", q=empirical_prior)

    curr_obs = jtu.tree_map(lambda x: x[0, :, -1, :], observation)

    # jax.debug.print("curr_obs in infer and belief share {o}", o=curr_obs)

    #jax.debug.print("observation {o}", o=curr_obs)
    # vmap over other agents batch dim
    log_likelihoods = jax.vmap(compute_log_likelihood_per_modality)(curr_obs, A)

    # add a small constant to empirical prior to let observations
    # flip strong priors bc of the log(min_val)
    empirical_prior = jtu.tree_map(lambda x: x + 1e-3, empirical_prior)
    log_prior = jtu.tree_map(log_stable, empirical_prior)
    #jax.debug.print("log prior {p}", p=log_prior)
    log_q = jtu.tree_map(jnp.zeros_like, empirical_prior)

    def scan_fn(log_q, iter):
        # do as "normal" factorized fpi implementation
        q = jtu.tree_map(nn.softmax, log_q)
        def marginal_ll(q, log_likelihoods):
            return all_marginal_log_likelihood(q, log_likelihoods, A_dependencies)
        ll = jax.vmap(marginal_ll)(q, log_likelihoods)
        #jax.debug.print("ll {l}", l=ll)

        # but merge LL messages of other's world states into focal world state posterior
        def merge(b, state_idx):
            return jax.lax.cond(jnp.any(world_states == state_idx), lambda : b.at[0].set(b.sum(axis=0)), lambda: b)
        ll = jtu.tree_map(merge, ll, list(range(len(ll))))

        log_q = jtu.tree_map(jnp.add, ll, log_prior)
        return log_q, None

    res, _ = jax.lax.scan(scan_fn, log_q, jnp.arange(agent.num_iter))
    qs = jtu.tree_map(nn.softmax, res)

    #jax.debug.print("posterior {q}", q=qs)
    # expand again to add "time" dim
    qs_expanded = jtu.tree_map(lambda x: x[:, None, :], qs)
    # jax.debug.print("qs_expanded in infer and belief share {q}", q=qs_expanded)
    return qs_expanded


def infer_and_plan_tom(
    focal_agents,
    qs_prev,
    observation,
    action_prev,
    rng_key,
    policy_search,
):

    batch_size = action_prev.shape[0]

    # compute empirical prior from previous action and belief
    def calculate_empirical_prior(action_prev, qs_prev, agent_models):
        return jax.lax.cond(
            jnp.any(action_prev < 0), # we slice out the focal agents action bc we may set the other agents actions masked as -1s
            lambda: agent_models.D,  # if no action is provided, use the agent's D as empirical prior
            lambda: agent_models.update_empirical_prior(action_prev, qs_prev)[0],
        )

    empirical_prior = jax.vmap(calculate_empirical_prior)(action_prev, qs_prev, focal_agents.agent_models)
    # jax.debug.print("empirical prior {e}", e=empirical_prior)

    # calculate new posterior given observations
    # with belief sharing between other agents' posteriors and focal's posterior
    # now vmap over observations too since each focal agent gets its own filtered observations
    qs = jax.vmap(infer_and_belief_share, in_axes=(0, 0, 0, None))(focal_agents.agent_models, observation, empirical_prior, jnp.array(focal_agents.world_states))

    # get posterior over policies
    rng_key, key = jr.split(rng_key)
    qpi, xtra = policy_search(
        focal_agents, qs, key
    )  # compute policy posterior using EFE - uses C to consider preferred outcomes
    # jax.debug.print("qpi {q}", q=qpi)
    # sample action from policy distribution
    keys = jr.split(rng_key, 1 + batch_size)
    rng_key = keys[0]
    # given the batch dims of agent_models, call sample_policy directly

    def sample_action(qpi, key):
        action = sample_policy(focal_agents.agent_models.policies[0, 0], qpi, "stochastic", focal_agents.agent_models.alpha[0, 0], key)
        return action
    action = jax.vmap(sample_action)(qpi, keys[1:])  # sample action for each agent in the batch

    return action, qs, xtra["tree"], xtra["other_trees"]


def rollout(
    focal_agent,
    other_agents,
    env,
    num_timesteps,
    rng_key,
    initial_carry=None,
    focal_agent_tom_policy_search=None,
    other_agent_policy_search=None,
):
    """
    Rollout agents in a multi-agent environment.

    The focal agent will perform Theory of Mind (ToM) reasoning about the other agents.
    The other agents, if added (otherwise, None) will perform sophisticated inference, but not ToM reasoning.

    Returns
    ----------
    last: ``dict``
        dictionary from the last timestep about the rollout, i.e., the final action, observation, beliefs, etc.
    info: ``dict``
        dictionary containing information about the rollout, i.e. executed actions, observations, beliefs, etc.
    env: ``Env``
        environment state after the rollout
    """

    # get the batch_size of the agents
    other_batch_size = other_agents.batch_size if other_agents is not None else 0
    focal_batch_size = focal_agent.batch_size
    batch_size = other_batch_size + focal_batch_size

    # create tree search function for non-ToM agents
    if other_agent_policy_search is None:
        other_agent_policy_search = si_policy_search()

    if focal_agent_tom_policy_search is None:
        focal_agent_tom_policy_search = si_policy_search_tom()

    def step_fn(carry, t):
        # carrying the current timestep's action, observation, beliefs, empirical prior, environment state, and random key
        action = carry["action"]
        observation = carry["observation"]
        other_qs_prev = carry["other_qs"] if other_agents is not None else None
        focal_qs_prev = carry["qs"]
        env_state = carry["env_state"]
        carry_other_agents = carry["other_agents"]
        carry_focal_agent = carry["focal_agent"]
        rng_key = carry["rng_key"]
        other_tree = carry["other_tree"] if other_agents is not None else None
        tree = carry["tree"]

        keys = jr.split(rng_key, 4)
        rng_key = keys[0]  # carry first key

        if other_agents is not None:
            focal_action = action[:focal_batch_size]  # focal agent's action
            other_action = action[focal_batch_size:]  # other agents' actions
            other_observation = [o[focal_batch_size:] for o in observation]  # other agents' observations
            
            # first plan for the other agents
            other_action_next, other_tree_next, xtra = infer_and_plan_recycle_tree(
                carry_other_agents,
                other_tree,
                other_observation,
                other_action,
                keys[1],
                policy_search=other_agent_policy_search,
            )
        else:
            other_action_next = None
            other_tree_next = None

        # plan for the focal agent - scan over focal agents
        
        focal_filtered_observations = jax.vmap(lambda focal_idx: to_one_hot(observation, carry_focal_agent, focal_idx))(
            jnp.arange(focal_batch_size)
        )
        
        if other_agents is not None:
            action = jnp.repeat(action[None, ...], focal_batch_size, axis=0)
        else:
            action = jnp.repeat(action[None, ...], batch_size, axis=0)
        
        focal_action_next, focal_qs, tree_next, focal_other_tree_next = infer_and_plan_tom(
            carry_focal_agent,
            focal_qs_prev,
            focal_filtered_observations,
            action,
            keys[2],
            policy_search=focal_agent_tom_policy_search,
        )
        
        # merge actions of focal agent and other agents
        if other_agents is not None:
            action_next = jnp.concatenate([focal_action_next, other_action_next], axis=0)
        else:
            action_next = focal_action_next
 
        # step environment forward
        observation_next, env_state_next = env.step(
            rng_key=keys[3], env_state=env_state, actions=action_next
        ) 

        # carrying the next timestep's action, observation, beliefs, empirical prior, environment state, and random key

        # get posterior from the tree root
        if other_agents is not None:
            other_qs = jtu.tree_map(lambda x: x[jnp.arange(other_batch_size), jax.vmap(root_idx)(other_tree_next)], other_tree_next.qs)
        else:
            other_qs = None

        # jax.debug.print("focal_qs {f}", f=focal_qs)
        # jax.debug.print("observation_next {o}", o=observation_next)
        # jax.debug.print("action_next {a}", a=action_next)

        carry_dict = {
            "qs": jtu.tree_map(lambda x: x[:, :, -1:, ...], focal_qs),  # keep only latest belief
            "action": action_next,
            "observation": observation_next,
            "env_state": env_state_next,
            "other_agents": carry_other_agents,
            "focal_agent": carry_focal_agent,
            "rng_key": rng_key,
            "tree": tree_next
        }
        
        info_dict = {
            "qs": jtu.tree_map(lambda x: x[:, 0, ...], focal_qs),
            "env_state": env_state,
            "observation": observation,
            "action": action_next,
            "tree": tree_next,
            "focal_other_tree": focal_other_tree_next,
        }
        
        if other_agents is not None:
            carry_dict["other_qs"] = jtu.tree_map(lambda x: x[:, -1:, ...], other_qs)  # keep only latest belief
            carry_dict["other_tree"] = other_tree_next
            info_dict["other_qs"] = jtu.tree_map(lambda x: x[:, 0, ...], other_qs)
            info_dict["other_tree"] = other_tree_next
            
        carry = carry_dict
        info = info_dict

        return carry, info

    if initial_carry is None:
        # initialise first observation from environment
        keys = jr.split(rng_key, 2)
        rng_key = keys[0]
        observation_0, env_state_reset = env.reset(keys[1])

        # specify initial beliefs using D
        focal_qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), focal_agent.agent_models.D)
        
        # put action to -1 to indicate no action taken yet
        if other_agents is not None:
            other_qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), other_agents.D)
            action_0 = -jnp.ones((batch_size, other_agents.policies.shape[-1]), dtype=jnp.int32)
            
            # get initial planning trees to carry, set reset=True
            _, info = other_agent_policy_search(agent=other_agents, qs=other_qs_0, reset=True)
            other_tree = info["tree"]
        else:
            other_qs_0 = None
            action_0 = -jnp.ones((batch_size, focal_agent.agent_models.policies.shape[-1]), dtype=jnp.int32)
            other_tree = None

        _, info = focal_agent_tom_policy_search(agent=focal_agent, qs=focal_qs_0, reset=True)
        tree = info["tree"]

        # set up initial state to carry through timesteps
        initial_carry = {
            "qs": focal_qs_0,
            "action": action_0,
            "observation": observation_0,
            "env_state": env_state_reset,
            "other_agents": other_agents,
            "focal_agent": focal_agent,
            "rng_key": rng_key,
            "tree": tree,
        }
        
        if other_agents is not None:
            initial_carry["other_qs"] = other_qs_0
            initial_carry["other_tree"] = other_tree
            
    # run the active inference loop for num_timesteps using jax.lax.scan
    last, info = jax.lax.scan(step_fn, initial_carry, jnp.arange(num_timesteps + 1))

    info = jtu.tree_map(
        lambda x: x.transpose((1, 0) + tuple(range(2, x.ndim))) if x.ndim >= 2 else x, info
    )  # transpose to have timesteps as first dimension

    return last, info, env
