import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.nn as nn
import equinox as eqx

from tom.planning.si import si_policy_search, root_idx
from tom.planning.rollout import infer_and_plan_recycle_tree
from tom.planning.si_tom import si_policy_search_tom, to_one_hot

from tom.planning.si import Tree
from tom.planning.si_tom import Tree as ToMTree

from pymdp.control import sample_policy
from pymdp.maths import compute_log_likelihood_per_modality, log_stable
from pymdp.algos import all_marginal_log_likelihood



def infer_and_belief_share(agent, observation, empirical_prior, world_states):
    # custom impl of fpi where we share likelihood messages of other's world states
    A = agent.A
    A_dependencies = agent.A_dependencies
    #jax.debug.print("empirical prior {q}", q=empirical_prior)

    curr_obs = jtu.tree_map(lambda x: x[:, -1, :], observation)

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
    return qs_expanded


def infer_and_plan_tom(
    agent,
    qs_prev,
    observation,
    action_prev,
    rng_key,
    policy_search,
):
    # TODO should be forwarding the previous plan tree, while only having access
    # to the focal agent's action and observation

    batch_size = agent.agent_models.A[0].shape[0]

    # compute empirical prior from previous action and belief
    def calculate_empirical_prior(action_prev, qs_prev, agent_models):
        return jax.lax.cond(
            jnp.any(action_prev[0] < 0), # we slice out the focal agents action bc we may set the other agents actions masked as -1s
            lambda: agent_models.D,  # if no action is provided, use the agent's D as empirical prior
            lambda: agent_models.update_empirical_prior(action_prev, qs_prev)[0],
        )

    agents_filtered, _ = eqx.partition(agent.agent_models, filter_spec=lambda leaf: leaf.shape[0] == batch_size)
    empirical_prior = jax.vmap(calculate_empirical_prior)(action_prev, qs_prev, agents_filtered)

    # calculate new posterior given observations
    # add belief sharing between other agents' posteriors and focal's posterior
    qs = jax.vmap(infer_and_belief_share, in_axes=(0, 0, 0, None))(agents_filtered, observation, empirical_prior, jnp.array(agent.world_states))

    # get posterior over policies
    rng_key, key = jr.split(rng_key)
    qpi, xtra = policy_search(
        agent, qs, key
    )  # compute policy posterior using EFE - uses C to consider preferred outcomes
    
    # sample action from policy distribution
    keys = jr.split(rng_key, 1 + batch_size)
    rng_key = keys[0]
    # given the batch dims of agent_models, call sample_policy directly

    def sample_action(qpi, key):
        action = sample_policy(agent.agent_models.policies[0, 0], qpi, "stochastic", agent.agent_models.alpha[0, 0], key)
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
    The other agents will perform sophisticated inference, but not ToM reasoning.

    Returns
    ----------
    last: ``dict``
        dictionary from the last timestep about the rollout, i.e., the final action, observation, beliefs, etc.
    info: ``dict``
        dictionary containing information about the rollout, i.e. executed actions, observations, beliefs, etc.
    env: ``Env``
        environment state after the rollout
    """

    # get the batch_size of the other agents
    other_batch_size = other_agents.batch_size
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
        other_qs_prev = carry["other_qs"]
        focal_qs_prev = carry["qs"]
        env = carry["env"]
        other_agents = carry["other_agents"]
        focal_agent = carry["focal_agent"]
        rng_key = carry["rng_key"]
        other_tree = carry["other_tree"]
        tree = carry["tree"]

        keys = jr.split(rng_key, batch_size + 3)
        rng_key = keys[0]  # carry first key

        focal_action = action[:focal_batch_size]  # focal agent's action
        other_action = action[focal_batch_size:]  # other agents' actions

        focal_observation = [o[:focal_batch_size] for o in observation]  # focal agent's observation
        other_observation = [o[focal_batch_size:] for o in observation]  # other agents' observations

        # first plan for the other agents
        other_action_next, other_tree_next, xtra = infer_and_plan_recycle_tree(
            other_agents,
            other_tree,
            other_observation,
            other_action,
            keys[1],
            policy_search=other_agent_policy_search,
        )

        # plan for the focal agent
        
        # adding partial observability for the focal agent (re: observations and actions)
        # convert observations to one-hots using observation_mappings
        focal_filtered_observation = to_one_hot(observation, focal_agent)
        
        # # create masked action array: focal agent sees its own action, others are masked with -1
        # masked_action = jnp.full_like(action, -1)  # start with all -1s
        # masked_action = masked_action.at[:focal_batch_size].set(focal_action)  # set focal agent's action
        
        focal_action_next, focal_qs, tree_next, focal_other_tree_next = infer_and_plan_tom(
            focal_agent,
            focal_qs_prev,
            focal_filtered_observation,
            action[None, ...], # TODO: try with masked/partial action; cannot do without you are your planning tree where we infer the other agent's actions bc setting as -1s results in empirical prior to be in hell state
            keys[2],
            policy_search=focal_agent_tom_policy_search,
        )

        # merge actions of focal agent and other agents
        # here we assume the focal agent is the first agent in the batch
        action_next = jnp.concatenate([focal_action_next, other_action_next], axis=0)
 
        # step environment forward with chosen action
        observation_next, env_next = env.step(
            rng_key=keys[3:], actions=action_next
        )  # step environment forward with chosen action

        # carrying the next timestep's action, observation, beliefs, empirical prior, environment state, and random key

        # get posterior from the tree root
        other_qs = jtu.tree_map(lambda x: x[jnp.arange(other_batch_size), jax.vmap(root_idx)(other_tree_next)], other_tree_next.qs)

        
        carry = {
            "other_qs": jtu.tree_map(lambda x: x[:, -1:, ...], other_qs),  # keep only latest belief
            "qs": jtu.tree_map(lambda x: x[:, :, -1:, ...], focal_qs),  # keep only latest belief
            "action": action_next,
            "observation": observation_next,
            "env": env_next,
            "other_agents": other_agents,
            "focal_agent": focal_agent,
            "rng_key": rng_key,
            "other_tree": other_tree_next,
            "tree": tree_next
        }
        info = {
            "other_qs": jtu.tree_map(lambda x: x[:, 0, ...], other_qs),
            "qs": jtu.tree_map(lambda x: x[:, 0, ...], focal_qs),
            "env": env,
            "observation": observation,
            "action": action_next,
            "other_tree": other_tree_next,
            "tree": tree_next,
            "focal_other_tree": focal_other_tree_next,
        }

        return carry, info

    if initial_carry is None:
        # initialise first observation from environment
        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        observation_0, env = env.reset(keys[1:])

        # specify initial beliefs using D
        focal_qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), focal_agent.agent_models.D)
        other_qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), other_agents.D)

        # put action to -1 to indicate no action taken yet
        action_0 = -jnp.ones((batch_size, other_agents.policies.shape[-1]), dtype=jnp.int32)

        # get initial planning trees to carry, set reset=True
        _, info = other_agent_policy_search(agent=other_agents, qs=other_qs_0, reset=True)
        other_tree = info["tree"]

        _, info = focal_agent_tom_policy_search(agent=focal_agent, qs=focal_qs_0, reset=True)
        tree = info["tree"]

        # set up initial state to carry through timesteps
        initial_carry = {
            "other_qs": other_qs_0,
            "qs": focal_qs_0,
            "action": action_0,
            "observation": observation_0,
            "env": env,
            "other_agents": other_agents,
            "focal_agent": focal_agent,
            "rng_key": rng_key,
            "other_tree": other_tree,
            "tree": tree,
        }

    # run the active inference loop for num_timesteps using jax.lax.scan
    last, info = jax.lax.scan(step_fn, initial_carry, jnp.arange(num_timesteps + 1))

    info = jtu.tree_map(
        lambda x: x.transpose((1, 0) + tuple(range(2, x.ndim))), info
    )  # transpose to have timesteps as first dimension

    return last, info, env
