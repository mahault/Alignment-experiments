import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.lax as lax
import equinox as eqx

from tom.planning.si import Tree, si_policy_search, root_idx, _remove_orphans


def forward_tree(tree, agent, action, observation):
    """
    Forward the planning tree one timestep given an action and new observation.

    This allows to re-use the tree for planning the next timestep.
    """
    root = root_idx(tree)

    # set all nodes at horizon 1 or lower to -1
    # these are nodes that are now "in the past" but still being reused
    mask = tree.horizon <= 1
    tree = eqx.tree_at(
        lambda x: x.horizon,
        tree,
        (tree.horizon * (1 - mask)) - mask,
    )

    # subtract one horizon on all others
    mask = tree.horizon > 1
    tree = eqx.tree_at(
        lambda x: x.horizon,
        tree,
        tree.horizon * (1 - mask) + mask * (tree.horizon - 1),
    )

    def consider_prune(t, node_idx):

        def prune_observations(t, policy_idx):
            # jax.debug.print("Prune policy node {x}", x=policy_idx)
            # mark the policy node unused
            t = eqx.tree_at(
                lambda x: x.used,
                t,
                t.used.at[policy_idx].set(False),
            )

            def consider_observation(t, obs_idx):

                def consider_root(t, obs_idx):
                    # mark the root by setting horizon to 0
                    # all others should have horizon < 0
                    # jax.debug.print("New root? {x}", x=obs_idx)
                    t = eqx.tree_at(
                        lambda x: x.horizon,
                        t,
                        t.horizon.at[obs_idx].set(0),
                    )
                    return t

                t = lax.cond(
                    (obs_idx >= 0)
                    & (jnp.allclose(t.policy[node_idx], action))
                    & (
                        jnp.allclose(
                            t.observation[obs_idx], jnp.concatenate(observation, axis=-1)
                        )
                    ),
                    consider_root,
                    lambda t, idx: t,
                    t,
                    obs_idx,
                )

                return t, None

            t, _ = lax.scan(consider_observation, t, t.children_indices[policy_idx])
            return t

        # go through all policies and their observation children
        # mark all policy nodes as unused
        # and mark all observation nodes as horizon = 0, except for the new root observation node
        # this will make sure unused observation nodes get pruned as orphans later
        # but reused ones can still be kept (with a negative h value)
        t = lax.cond((node_idx >= 0), prune_observations, lambda t, idx: t, t, node_idx)
        return t, None


    tree_forwarded, _ = lax.scan(consider_prune, tree, tree.children_indices[root])

    def cleanup(tree, tree_forwarded, action, observations):
        # clean unused nodes from the forwarded tree
        tree_forwarded = _remove_orphans(tree_forwarded)
        return tree_forwarded
    
    def reset(tree, tree_forwarded, action, observations):
        # reset the tree to a new initial state
        # here we need to add a 1 batch dim again to agents, action and observations
        # to use the pymdp methods
        agent_expanded = jtu.tree_map(lambda x: x[None, ...], agent)
        action_expanded = action[None, ...]
        observations_expanded = jtu.tree_map(lambda x: x[None, ...], observations)
        qs = jtu.tree_map(lambda x: x[root_idx(tree)][None, ...], tree.qs)
        empirical_prior = jax.lax.cond(
            jnp.any(action < 0),
            lambda: agent_expanded.D,
            lambda: agent_expanded.update_empirical_prior(action_expanded, qs)[0],
        )
        new_qs = agent_expanded.infer_states(observations_expanded, empirical_prior)
        return reset_tree(tree, new_qs, observations)

    tree_forwarded = jax.lax.cond(
        root_idx(tree_forwarded) == -1,
        reset,
        cleanup,
        tree,
        tree_forwarded,
        action,
        observation,
    )

    return tree_forwarded


def reset_tree(tree, qs, obs=None):
    """
    Reset the planning tree to a new initial state.
    """
    qs = jtu.tree_map(lambda x, y: x.at[0].set(y[0]), tree.qs, qs)
    used = tree.used.at[:, :].set(False)
    used = used.at[0].set(True)
    observation = (tree.observation * 0) - 1
    if obs is None:
        observation = observation.at[0].set(0)
    else:
        o = jnp.concatenate(obs, axis=-1)
        observation = observation.at[0].set(o)

    policy = (tree.policy * 0) - 1
    children_indices = (tree.children_indices * 0) - 1
    children_probs = tree.children_probs * 0
    G = tree.G * 0
    horizon = tree.horizon * 0

    tree = eqx.tree_at(
        lambda x: (
            x.qs,
            x.policy,
            x.observation,
            x.G,
            x.used,
            x.horizon,
            x.children_indices,
            x.children_probs,
        ),
        tree,
        (
            qs,
            policy,
            observation,
            G,
            used,
            horizon,
            children_indices,
            children_probs,
        ),
    )
    return tree


def infer_and_plan_recycle_tree(
    agent,
    tree,
    observation,
    action,
    rng_key,
    policy_search,
):
    # forward the tree one step
    agent_filtered, _ = eqx.partition(agent, filter_spec=lambda leaf: leaf.shape[0] == agent.batch_size)
    tree_forwarded = jax.vmap(forward_tree)(tree, agent_filtered, action, observation)

    # compute policy posterior using sophisticated tree search
    qpi, info = policy_search(agent=agent, tree=tree_forwarded)
    tree_next = info["tree"]

    # sample action from posterior distribution
    keys = jr.split(rng_key, agent.batch_size + 1)
    rng_key = keys[0]
    action_next = agent.sample_action(qpi, rng_key=keys[1:])

    return action_next, tree_next, {"qpi": qpi}


def rollout(
    agent,
    env,
    num_timesteps,
    rng_key,
    initial_carry=None,
    policy_search=None,
):
    """
    Rollout an agent in an environment for a number of timesteps.

    This method uses sophisticated planning, and re-uses the planning tree for each timestep.

    Parameters
    ----------
    agent: active inference agent
    env: environment that can step forward and return observations
    num_timesteps: how many timesteps to simulate
    rng_key: random key for sampling

    Returns
    ----------
    last: ``dict``
        dictionary from the last timestep about the rollout, i.e., the final action, observation, beliefs, etc.
    info: ``dict``
        dictionary containing information about the rollout, i.e. executed actions, observations, beliefs, etc.
    env: ``Env``
        environment state after the rollout
    """

    # get the batch_size of the agent
    batch_size = agent.batch_size

    # create tree search function
    if policy_search is None:
        policy_search = si_policy_search()

    def step_fn(carry, t):
        # carrying the current timestep's action, observation, beliefs, empirical prior, environment state, and random key
        action = carry["action"]
        observation = carry["observation"]
        qs_prev = carry["qs"]
        env = carry["env"]
        agent = carry["agent"]
        rng_key = carry["rng_key"]
        tree = carry["tree"]

        keys = jr.split(rng_key, batch_size + 2)
        rng_key = keys[0]  # carry first key

        action_next, tree_next, xtra = infer_and_plan_recycle_tree(
            agent,
            tree,
            observation,
            action,
            keys[1],
            policy_search=policy_search,
        )

        # step environment forward with chosen action
        observation_next, env_next = env.step(
            rng_key=keys[2:], actions=action_next
        )  # step environment forward with chosen action

        # carrying the next timestep's action, observation, beliefs, empirical prior, environment state, and random key

        # get posterior from the tree root
        qs = jtu.tree_map(lambda x: x[jnp.arange(agent.batch_size), jax.vmap(root_idx)(tree_next)], tree_next.qs)

        carry = {
            "action": action_next,
            "observation": observation_next,
            "tree": tree_next,
            "qs": jtu.tree_map(lambda x: x[:, -1:, ...], qs),  # keep only latest belief
            "env": env_next,
            "agent": agent,
            "rng_key": rng_key,
        }
        info = {
            "qs": jtu.tree_map(lambda x: x[:, 0, ...], qs),
            "env": env,
            "observation": observation,
            "action": action_next,
            "tree": tree_next,
        }
        info.update(xtra)

        return carry, info

    if initial_carry is None:
        # initialise first observation from environment
        keys = jr.split(rng_key, batch_size + 1)
        rng_key = keys[0]
        observation_0, env = env.reset(keys[1:])

        # specify initial beliefs using D
        qs_0 = jtu.tree_map(lambda x: jnp.expand_dims(x, -2), agent.D)

        # put action to -1 to indicate no action taken yet
        action_0 = -jnp.ones((agent.batch_size, agent.policies.shape[-1]), dtype=jnp.int32)

        # call tree search function with reset=True to create an initial tree structure
        _, info = policy_search(agent=agent, qs=qs_0, reset=True)
        tree = info["tree"]

        # set up initial state to carry through timesteps
        initial_carry = {
            "qs": qs_0,
            "action": action_0,
            "observation": observation_0,
            "env": env,
            "agent": agent,
            "rng_key": rng_key,
            "tree": tree,
        }

    # run the active inference loop for num_timesteps using jax.lax.scan
    last, info = jax.lax.scan(step_fn, initial_carry, jnp.arange(num_timesteps + 1))

    info = jtu.tree_map(
        lambda x: x.transpose((1, 0) + tuple(range(2, x.ndim))), info
    )  # transpose to have timesteps as first dimension

    return last, info, env
