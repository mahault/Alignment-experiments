import numpy as np

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.lax as lax
import jax.nn as nn
import equinox as eqx

from functools import partial

import pymdp
from pymdp.agent import Agent
from pymdp.control import (
    compute_info_gain,
    compute_expected_utility,
    compute_expected_state,
    compute_expected_obs,
)
from pymdp.maths import log_stable

from tom.planning.si import si_policy_search, _generate_observations, _calculate_probabilities, _remove_orphans
from tom.planning.si import root_idx as si_root_idx

from typing import List


# Helper function to pad a collection of arrays to the same size
def pad_to_max_size(arrays):
    """Pad all arrays in a collection to the maximum size with zeros."""
    if not arrays:
        return arrays, None
    
    max_size = max(arr.shape[0] for arr in arrays)
    padded = []
    masks = []
    
    for arr in arrays:
        size = arr.shape[0]
        if size < max_size:
            padding = jnp.zeros(max_size - size, dtype=arr.dtype)
            padded.append(jnp.concatenate([arr, padding]))
            mask = jnp.concatenate([jnp.ones(size, dtype=bool), jnp.zeros(max_size - size, dtype=bool)])
        else:
            padded.append(arr)
            mask = jnp.ones(size, dtype=bool)
        masks.append(mask)
    
    return padded, masks

# Given a list of padded arrays, get the array corresponding to an index
def get_index_from_padded_collection(collection, masks, idx):
    """
    Select from a collection of padded arrays using a traced index.
    All arrays in collection must have the same shape due to padding.
    Masks indicate which elements are valid in each array.
    """
    # Now all arrays have the same shape, so lax.switch will work
    funcs = [lambda i=i: collection[i] for i in range(len(collection))]
    selected = jax.lax.switch(idx, funcs)
    
    # Also get the corresponding mask
    mask_funcs = [lambda i=i: masks[i] for i in range(len(masks))]
    selected_mask = jax.lax.switch(idx, mask_funcs)
    
    return selected, selected_mask


def si_policy_search_tom(
    horizon=5,
    max_nodes=5000,
    max_branching=10,
    policy_prune_threshold=1 / 16,
    observation_prune_threshold=1 / 16,
    entropy_stop_threshold=0.5,
    efe_stop_threshold=1e10,
    kl_threshold=1e-3,
    prune_penalty=512,
    gamma=1,
    topk_obsspace=10000,
    other_agent_policy_search=None,
):  
    
    @partial(jax.jit, static_argnames=["reset"])
    def search_fn(agent, qs=None, rng_key=None, reset=False):
        # create the initial tree structure
        tree = Tree(
            qs,
            len(agent.agent_models.num_controls),
            len(agent.agent_models.num_obs),
            max_nodes,
            max_branching,
            batch_size=agent.batch_size,
            prune_penalty=prune_penalty
        )
        
        # perform the tree search
        partial_tree_search = partial(
            tree_search_tom,
            horizon=horizon,
            policy_prune_threshold=policy_prune_threshold,
            observation_prune_threshold=observation_prune_threshold,
            entropy_stop_threshold=entropy_stop_threshold,
            efe_stop_threshold=efe_stop_threshold,
            kl_threshold=kl_threshold,
            prune_penalty=prune_penalty,
            gamma=gamma,
            topk_obsspace=topk_obsspace,
            other_agent_policy_search=other_agent_policy_search,
        )

        def scan_tree_search(carry, data):
            tom_agent, tree = data
            tree, other_trees = partial_tree_search(tom_agent, tree)
            return None, (tom_agent, tree, other_trees)

        other_trees = None
        if not reset:
            _, data = lax.scan(scan_tree_search, None, (agent, tree))
            _, tree, other_trees = data
        
        # get the q_pi_marginalized from the root node
        q_pi_marginalized = tree.q_pi_marginalized[jnp.arange(agent.batch_size), jax.vmap(root_idx)(tree)]
        return q_pi_marginalized, {"tree":  tree, "other_trees": other_trees}
    
    return search_fn

class ToMAgent(eqx.Module):

    num_agents: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

    # which state factors are related to self, which to world
    self_states: List = eqx.field(static=True)
    world_states: List = eqx.field(static=True)

    # how to map env observations to agent model observations
    # which observations are for self and which are to model others
    # (num_agents, len(A))
    observation_mappings: jnp.array

    #
    # how to map states of other agents to focal agent
    # i.e. the self location state of another agent can be mapping
    # to a (different) other location state of the focal
    state_mappings: jnp.array

    # copy of the single agent model per agent
    # so we have a batch of batch_size tom agents
    # with each tom_agent having num_agents agent models
    # so resulting agent_models will have shape (batch_size, num_agents, ...)
    agent_models: Agent

    def __init__(self, num_agents, agent_models, self_states, world_states, observation_mappings, state_mappings=None, batch_size=1):
        self.num_agents = num_agents
        self.agent_models = agent_models
        self.self_states = self_states
        self.world_states = world_states
        self.observation_mappings = observation_mappings
        self.state_mappings = state_mappings
        self.batch_size = batch_size


def ToMify(agent: Agent, self_states: List, world_states: List, observation_mappings: List, state_mappings: List = None, batch_size: int = 1):
    """
    Convert a single agent into a Theory of Mind agent.

    This will create a ToMAgent which has multiple agent models,
    one for each agent in the environment, and maps observations
    to the agent's own model and the world model.

    Args:
        agent: The original Agent instance.
        self_states: List of state factors related to the agent itself.
        world_states: List of state factors related to the world.
        observation_mappings: Mapping of observations to agent model observation modality.

    Returns:
        A ToMAgent instance.
    """

    # observation mappings of shape [batch_size, num_agents, num_obs]
    num_agents = observation_mappings.shape[1]

    # note that we are repeating agent.policies here to enable scanning in tom_policy_search
    agentz = Agent(
        [a.repeat(num_agents, axis=0) for a in agent.A],
        [b.repeat(num_agents, axis=0) for b in agent.B],
        [c.repeat(num_agents, axis=0) for c in agent.C] if agent.C is not None else None,
        [d.repeat(num_agents, axis=0) for d in agent.D] if agent.D is not None else None,
        agent.E.repeat(num_agents, axis=0) if agent.E is not None else None,
        pA=[a.repeat(num_agents, axis=0)for a in agent.pA] if agent.pA is not None else None,
        pB=[b.repeat(num_agents, axis=0) for b in agent.pB] if agent.pB is not None  else None,
        H=[h.repeat(num_agents, axis=0) for h in agent.H] if agent.H is not None else None,
        I=[i.repeat(num_agents, axis=0) for i in agent.I] if agent.I is not None else None,
        A_dependencies=agent.A_dependencies,
        B_dependencies=agent.B_dependencies,
        B_action_dependencies=agent.B_action_dependencies,
        num_controls=agent.num_controls,
        control_fac_idx=agent.control_fac_idx,
        policy_len=agent.policy_len,
        policies=agent.policies[None, ...].repeat(num_agents, axis=0),
        gamma=agent.gamma,
        alpha=agent.alpha,
        inductive_depth=agent.inductive_depth,
        inductive_threshold= agent.inductive_threshold,
        inductive_epsilon= agent.inductive_epsilon,
        use_utility=agent.use_utility,
        use_states_info_gain= agent.use_states_info_gain,
        use_param_info_gain= agent.use_param_info_gain,
        use_inductive= agent.use_inductive,
        onehot_obs= agent.onehot_obs,
        action_selection= agent.action_selection,
        sampling_mode= agent.sampling_mode,
        inference_algo= agent.inference_algo,
        num_iter= agent.num_iter,
        batch_size=num_agents,
        learn_A=agent.learn_A,
        learn_B= agent.learn_B,
        learn_C= agent.learn_C,
        learn_D= agent.learn_D,
        learn_E= agent.learn_E,
    )

    # repeat the agent models for each tom agent in batch
    agentz = jtu.tree_map(lambda x: jnp.repeat(x[None], batch_size, axis=0) if isinstance(x, jnp.ndarray) else x, agentz)
    # caveat: we area also (again) repeating "policies" here, which is not ideal,
    # but we still need to have agent_models.policies available when scanning over tom agents

    return ToMAgent(
        num_agents=num_agents,
        agent_models=agentz,
        self_states=self_states,
        world_states=world_states,
        observation_mappings=jnp.asarray(observation_mappings),
        state_mappings=jnp.asarray(state_mappings) if state_mappings is not None else None,
        batch_size=batch_size
    )


class Tree(eqx.Module):
    """
    A tree structure to hold the planning nodes and their data.

    This is an equinox module which allows for JAX transformations like jit.
    Pre-allocates memory for up to `max_nodes` nodes and `max_branching` children per node.
    """

    size: int = eqx.field(static=True)
    num_agents: int = eqx.field(static=True)

    # Which agent's perspective is this node from (n_nodes, 1)
    agent_idx: jnp.ndarray

    # Tree nodes data (n_nodes, n_agents, 1, feature_dims)
    # Node belief states - This will be batched over agents
    qs: List[jnp.ndarray]

    # Likelihood messages of other agents for focal qs
    ll: List[jnp.ndarray]

    # Policy taken if a policy node
    policy: jnp.ndarray
    # Observation expected if an observation node
    observation: jnp.ndarray
    # Node G estimates
    # - for a policy node: G of that policy at that timestep
    # - for an observation node: recursively aggregated G of all children
    G: jnp.ndarray

    # marginalized values (for focal agent's policy nodes)
    q_pi_marginalized: jnp.ndarray
    G_recursive: jnp.ndarray

    # Tree structure bookkeeping
    # Wheter a node is used or not (n_nodes, 1)
    used: jnp.ndarray
    # Which horizon level the node represents (n_nodes, 1)
    horizon: jnp.ndarray
    # Which depth level the node is at (n_nodes, 1)
    depth: jnp.ndarray
    # Indices to children (n_nodes, max_branching), -1 if unused
    children_indices: jnp.ndarray
    # Probabilities for each child (q_pi for policy children, q_o for observation children)
    # (n_nodes, max_branching)
    children_probs: jnp.ndarray

    # index of the other agent's plan tree node we are "in"
    other_tree_idx: jnp.ndarray

    def __init__(
        self,
        qs,
        num_action_modalities,
        num_observation_modalities,
        max_nodes,
        max_branching,
        batch_size=1,
        prune_penalty=512
    ):
        self.size = max_nodes
        self.num_agents = qs[0].shape[-3]

        self.agent_idx = jnp.zeros((batch_size, max_nodes, 1), dtype=jnp.int32)

        self.qs = [jnp.zeros((batch_size, max_nodes, self.num_agents, 1, q.shape[-1])) for q in qs]
        self.ll = [jnp.zeros((batch_size, max_nodes, self.num_agents-1, 1, q.shape[-1])) for q in qs]

        self.policy = -jnp.ones((batch_size, max_nodes, num_action_modalities), dtype=jnp.int32)
        self.observation = -jnp.ones(
            (batch_size, max_nodes, num_observation_modalities), dtype=jnp.int32
        )
        self.G = jnp.zeros((batch_size, max_nodes, 1))

        # marginalized values (for focal agent's policy nodes)
        self.q_pi_marginalized = jnp.zeros((batch_size, max_nodes, max_branching))
        self.G_recursive = jnp.zeros((batch_size, max_nodes, max_branching))

        self.G_recursive = self.G_recursive.at[:, -1, :].set(-prune_penalty)

        self.used = jnp.zeros((batch_size, max_nodes, 1), dtype=jnp.bool)
        self.horizon = jnp.zeros((batch_size, max_nodes, 1), dtype=jnp.int32)
        self.depth = jnp.zeros((batch_size, max_nodes, 1), dtype=jnp.int32)
        self.children_indices = -jnp.ones((batch_size, max_nodes, max_branching), dtype=jnp.int32)
        self.children_probs = jnp.zeros((batch_size, max_nodes, max_branching), dtype=jnp.float32)
        
        self.other_tree_idx = jnp.zeros((batch_size, max_nodes, self.num_agents-1, 1), dtype=jnp.int32)

        # set root node
        # self.qs is of shape [batch_size, max_nodes, num_agents, 1, state_dim]  qs will be [batch_size, num_agents, 1, state_dim]
        self.qs = jtu.tree_map(lambda x, y: x.at[:, 0, ...].set(y), self.qs, qs)
        self.used = self.used.at[:, 0].set(True)
        self.observation = self.observation.at[:, 0].set(0)
        self.agent_idx = self.agent_idx.at[:, 0].set(self.num_agents - 1)

    def __getitem__(self, index):
        """
        Get the node at the given index as a dictionary.

        Works on a non-batched Tree!
        """
        node = {
            "idx": index,
            "agent": self.agent_idx[index, 0],
            "qs": jtu.tree_map(lambda x: x[index:index + 1], self.qs),
            "ll": jtu.tree_map(lambda x: x[index:index + 1], self.ll),
            "G": self.G[index, 0],
            "q_pi_marginalized": self.q_pi_marginalized[index],
            "G_recursive": self.G_recursive[index],
            "horizon": self.horizon[index, 0],
            "depth": self.depth[index, 0],
            "other_tree_idx": self.other_tree_idx[index, :, 0]
        }

        if jnp.any(self.children_indices[index] >= 0):
            node["children"] = [c for c in self.children_indices[index] if c >= 0]
            node["children_probs"] = []
            for i in range(self.children_indices.shape[-1]):
                if self.children_indices[index, i] >= 0:
                    node["children_probs"].append(self.children_probs[index, i])
        else:
            node["children"] = []

        if self.policy[index][0] >= 0:
            node["policy"] = self.policy[index]
        else:
            node["observation"] = self.observation[index : index + 1]

        return node

    def root(self):
        """
        Get the root node of the tree.

        Works on a non-batched Tree!
        """
        root_node = self[root_idx(self)]
        return root_node


def root_idx(tree):
    root_idx = jnp.argwhere(
        (tree.used)[:, 0]
        & (tree.agent_idx == tree.num_agents - 1)[:, 0]
        & (tree.horizon == 0)[:, 0]
        & ~jnp.all(tree.observation == -1, axis=-1),
        size=1,
        fill_value=-1,
    )[0, 0]
    return root_idx


def step(agent, qs, policies, ll, state_mappings=None):
    """
    For a given agent, calculate next qs, qo and G given the current qs and policies.
    Also incorporate belief-sharing likelihood messages ll from other agents
    """

    def _step(q, policy):
        # apply ll messages first, to factor in the effect of other's actions
        # into our own decisions
        ll_sum = jtu.tree_map(lambda x: x.sum(axis=0)[0], ll)
        log_qs = jtu.tree_map(lambda x: jnp.log(x + 1e-10), q)
        qs_updated = jtu.tree_map(lambda lqs, ll: nn.softmax(lqs + ll), log_qs, ll_sum)

        # get our own envisioned next state
        # be aware, we don't know exactly which agent is "right", i.e. if the other agent
        # is at a location, its beliefs about that location are more accurate then ours
        # therefore, we want to precision weigh the ll messages with our own B transitions accordingly
        # similar to calc_likelihood_msgs, we evaluate both policy and noop policy
        # and if the policy didn't affect the state, we upweigh the precision of the other's ll

        # TODO add a flag to have noop policy?
        noop_policy = jnp.zeros_like(policy)
        policy_and_noop = jnp.concatenate([policy[None, ...], noop_policy[None, ...]], axis=0)

        qs_next = jax.vmap(lambda p: compute_expected_state(qs_updated, agent.B, p, agent.B_dependencies))(policy_and_noop)

        log_prev = jtu.tree_map(log_stable, qs_updated)
        log_new = jtu.tree_map(log_stable, qs_next)
        diff_logs = jtu.tree_map(lambda x, y: x - y, log_new, log_prev)
        # filter out effects that would also happen when you did nothing (i.e. env dynamics)
        states_focal_no_impact = jtu.tree_map(lambda x: jnp.allclose(x[0],x[1]), diff_logs)
        states_other_no_impact = jtu.tree_map(lambda x: jnp.allclose(x, 0), ll_sum)
        mask = jtu.tree_map(lambda x,y : 1 - (x & ~y), states_focal_no_impact, states_other_no_impact)

        diff_logs_weighted = jtu.tree_map(lambda x, m: x[0] * m, diff_logs, mask)
        qs_next = jtu.tree_map(lambda x,y: x+y, log_prev, diff_logs_weighted)
        qs_next = jtu.tree_map(nn.softmax, qs_next)

        # don't update mapped states through our own B tensor, but use the one from ll instead
        if state_mappings is not None:
            def map_states(qs, other_agent_idx):
                # Pad both collections to the same size for dynamic indexing
                padded_qs_updated, qs_updated_masks = pad_to_max_size(qs_updated)
                padded_qs_next, qs_next_masks = pad_to_max_size(qs_next)
                
                state_mapping_for_agent = state_mappings[other_agent_idx]
                
                # Build the remapped qs using traced conditionals
                new_qs = []
                for j, mapping_idx in enumerate(state_mapping_for_agent):
                    # Use lax.cond to choose between qs_updated[j] or qs_next[j]
                    # Both branches must return arrays of the same shape (padded)
                    def use_qs_updated(mapping_idx=mapping_idx, j=j):
                        # Return the j-th padded qs_updated value
                        return padded_qs_updated[j]
                    
                    def use_qs_next(mapping_idx=mapping_idx, j=j):
                        # Return the j-th padded qs_next value
                        return padded_qs_next[j]
                    
                    padded_result = lax.cond(
                        mapping_idx >= 0,
                        use_qs_updated,
                        use_qs_next
                    )
                    
                    # Unpad the result back to the original size for this state factor
                    original_size = qs[j].shape[0]
                    result = padded_result[:original_size]
                    new_qs.append(result)
                
                return new_qs, None

            qs_next, _ = lax.scan(map_states, qs_next, jnp.arange(ll[0].shape[0]))
           
        qo = compute_expected_obs(qs_next, agent.A, agent.A_dependencies)
        u = compute_expected_utility(qo, agent.C)
        ig = compute_info_gain(qs_next, qo, agent.A, agent.A_dependencies)
        return qs_next, qo, u, ig

    qs, qo, u, ig = jax.vmap(
        lambda policy: jax.vmap(_step)(qs, policy)
    )(policies)
    G = u + ig
    return qs, qo, G


def _update_node(
    tree,
    idx,
    agent_idx=None,
    qs=None,
    ll=None,
    policy=None,
    observation=None,
    G=None,
    q_pi_marginalized=None,
    G_recursive=None,
    horizon=None,
    depth=None,
    children_indices=None,
    children_probs=None,
    other_tree_idx=None,
):
    """
    Update a node in the planning tree with new data at index `idx`.

    When all nodes are already used, it will not update the node and print a warning.
    """

    def _do_update(tree, idx, agent_idx, qs, ll, policy, observation, G, q_pi_marginalized, G_recursive, horizon, depth, children_indices, children_probs, other_tree_idx):

        if agent_idx is not None:
            agent_idx = tree.agent_idx.at[idx].set(agent_idx)
        else:
            agent_idx = tree.agent_idx   

        qs = (
            jtu.tree_map(lambda x, y: x.at[idx].set(y), tree.qs, qs)
            if qs is not None
            else tree.qs
        )

        ll = (
            jtu.tree_map(lambda x, y: x.at[idx].set(y), tree.ll, ll)
            if ll is not None
            else tree.ll
        )

        # if we set a policy, we also set the observation to -1
        if policy is not None:
            policy = tree.policy.at[idx].set(policy)
            observation = tree.observation.at[idx].set(-1)
        elif observation is not None:
            # if observation is set, we also set the policy to -1
            observation = tree.observation.at[idx].set(observation)
            policy = tree.policy.at[idx].set(-1)
        else:
            policy = tree.policy
            observation = tree.observation

        G = tree.G.at[idx].set(G) if G is not None else tree.G
        q_pi_marginalized = tree.q_pi_marginalized.at[idx].set(q_pi_marginalized) if q_pi_marginalized is not None else tree.q_pi_marginalized
        if G_recursive is not None and len(G_recursive.shape) == 0:
            # G_recursive is a scalar, set it for the first element only
            G_recursive = tree.G_recursive.at[idx, 0].set(G_recursive)
        else:
            G_recursive = tree.G_recursive.at[idx].set(G_recursive) if G_recursive is not None else tree.G_recursive
        horizon = tree.horizon.at[idx].set(horizon) if horizon is not None else tree.horizon
        depth = tree.depth.at[idx].set(depth) if depth is not None else tree.depth

        if children_indices is None:
            children_indices = tree.children_indices
        else:
            # pad with -1
            children_indices = jnp.pad(
                children_indices,
                (0, tree.children_indices.shape[-1] - children_indices.shape[0]),
                constant_values=-1,
            )
            children_indices = tree.children_indices.at[idx].set(children_indices)
        
        if children_probs is None:
            children_probs = tree.children_probs
        else:
            # pad with 0
            children_probs = jnp.pad(
                children_probs,
                (0, tree.children_probs.shape[-1] - children_probs.shape[0]),
                constant_values=0,
            )
            children_probs = tree.children_probs.at[idx].set(children_probs)

        used = tree.used.at[idx].set(True)

        if other_tree_idx is not None:
            other_tree_idx = tree.other_tree_idx.at[idx].set(other_tree_idx)
        else:
            other_tree_idx = tree.other_tree_idx

        tree = eqx.tree_at(
            lambda x: (
                x.agent_idx,
                x.qs,
                x.ll,
                x.policy,
                x.observation,
                x.G,
                x.q_pi_marginalized,
                x.G_recursive,
                x.used,
                x.horizon,
                x.depth,
                x.children_indices,
                x.children_probs,
                x.other_tree_idx,
            ),
            tree,
            (
                agent_idx,
                qs,
                ll,
                policy,
                observation,
                G,
                q_pi_marginalized,
                G_recursive,
                used,
                horizon,
                depth,
                children_indices,
                children_probs,
                other_tree_idx
            ),
        )
        return tree

    def _no_op(tree, idx, agent_idx, qs, ll, policy, observation, G, q_pi_marginalized, G_recursive, horizon, depth, children_indices, children_probs, other_tree_idx):
        jax.debug.print("WARNING: Used up all {x} nodes in the plan tree... at horizon {h}", x=tree.size, h=horizon)
        return tree
    
    return lax.cond(idx < tree.size - 1, _do_update, _no_op, tree, idx, agent_idx, qs, ll, policy, observation, G, q_pi_marginalized, G_recursive, horizon, depth, children_indices, children_probs, other_tree_idx)


def to_one_hot(obs, tom_agent):
    o_one_hot = []
    
    for i in range(len(obs)):  # for each observation modality
        obs_modality = []
        
        for agent_idx in range(obs[i].shape[0]):  # for each agent
            # get the observation mapping for this agent
            mapping = tom_agent.observation_mappings[0, agent_idx, i] 
            
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


@partial(jax.jit, static_argnames=["horizon", "topk_obsspace", "other_agent_policy_search"])
def tree_search_tom(
    tom_agent,
    tree,
    horizon,
    policy_prune_threshold=1 / 16,
    observation_prune_threshold=1 / 16,
    entropy_stop_threshold=0.5,
    efe_stop_threshold=1e10,
    kl_threshold=1e-3,
    prune_penalty=512,
    gamma=1,
    topk_obsspace=10000,
    other_agent_policy_search=None,
):
    """
    Perform a sophisticated inference tree search given an agent and planning tree.

    Keeps expanding the tree until one of the following conditions is met:
    - The horizon is reached.
    - The entropy of the root node's policy distribution is below a threshold.
    - The expected free energy of the root node is below a threshold.

    Args:
        agent: The agent to use for planning.
        tree: The initial planning tree.
        horizon: The maximum horizon to expand the tree.
        policy_prune_threshold: Threshold for pruning policies.
        observation_prune_threshold: Threshold for pruning observations.
        entropy_stop_threshold: Entropy threshold to stop expanding.
        efe_stop_threshold: Expected free energy threshold to stop expanding.
        kl_threshold: KL divergence threshold for reusing nodes.
        prune_penalty: Penalty for pruning a node.
        gamma: Precison of q_pi.

    Returns:
        tree: The expanded planning tree.
    """
    world_states = jnp.array(tom_agent.world_states)
    num_agents = tom_agent.num_agents

    if other_agent_policy_search is None:
        # if not specified, use a SI policy search with the same parameters as the ToM planner
        other_agent_policy_search = si_policy_search(horizon, tree.size, tree.children_indices.shape[-1], policy_prune_threshold, observation_prune_threshold, entropy_stop_threshold, efe_stop_threshold, kl_threshold, prune_penalty, gamma)

    # execute planning for all other agents first, and reuse for building the ToM tree
    # get other agent's beliefs at the root node
    other_qs = jtu.tree_map(lambda x: x[root_idx(tree), 1:], tree.qs)
    # get other agent's beliefs
    other_agents = jtu.tree_map(lambda x: x[1:] if isinstance(x, jnp.ndarray) else x, tom_agent.agent_models)
    other_agents = eqx.tree_at(lambda x: x.policies, other_agents, tom_agent.agent_models.policies[0])

    _, info = other_agent_policy_search(other_agents, other_qs)
    other_trees = info["tree"]

    def _expand_policy_nodes_other(t, node_idx, agent_idx, agent, horizon):
        # jax.debug.print("Expanding policy nodes at node {n} for other agent {a}", n=node_idx, a=agent_idx)
        
        other_tree = jtu.tree_map(lambda x: x[agent_idx - 1], other_trees)
        other_tree_idx = t.other_tree_idx[node_idx, agent_idx - 1, 0]

        # and create nodes for resulting policies
        def really_add(t, idx):
            # jax.debug.print("Adding policy node for other agent {a} from idx {i}", a=agent_idx, i=idx)

            new_idx = jnp.where(
                    ~t.used[:, 0], jnp.arange(t.size), t.size
                ).min()

            # take prev beliefs of node_idx
            # and replace agent_idx with the new policy belief
            prev_beliefs = jtu.tree_map(lambda x: x[node_idx], t.qs)
            updated_beliefs = jtu.tree_map(lambda x,y: y.at[agent_idx].set(x[idx]), other_tree.qs, prev_beliefs)

            # "belief sharing" with focal agent, i.e. focal agent updates its belief about the world states
            # based on the other agent's new posterior
            def calc_likelihood_msgs(prev_beliefs, agent_idx):

                #
                # qs_next using focal b/qs?
                #
                def merge(b, state_idx):
                     return jax.lax.cond(jnp.any(world_states == state_idx), lambda : b[0, 0], lambda: b[agent_idx, 0])
                qs_perspective_switch = jtu.tree_map(merge, prev_beliefs, list(range(len(prev_beliefs))))
                policy = other_tree.policy[idx][None, ...]
                # TODO add a flag to have noop policy?
                noop_policy = jnp.zeros_like(policy)
                policy_and_noop = jnp.concatenate([policy, noop_policy], axis=0)

                focal_B = jtu.tree_map(lambda x: x[0], tom_agent.agent_models.B)
                qs_next = jax.vmap(lambda p: compute_expected_state(qs_perspective_switch, focal_B, p, tom_agent.agent_models.B_dependencies))(policy_and_noop)

                log_prev = jtu.tree_map(log_stable, qs_perspective_switch)
                log_new = jtu.tree_map(log_stable, qs_next)
                diff_logs = jtu.tree_map(lambda x, y: x - y, log_new, log_prev)
                # filter out effects that would also happen when you did nothing (i.e. env dynamics)
                mask = jtu.tree_map(lambda x: 1 - jnp.allclose(x[0],x[1]), diff_logs)
                diff_logs = jtu.tree_map(lambda x, m: x[0] * m, diff_logs, mask)

                # filter likelihood messages for world states only that can be affected by the other agent's actions
                def filter(diff_log, state_idx):
                     return jax.lax.cond(jnp.any(world_states == state_idx), lambda : diff_log, lambda: jnp.zeros_like(diff_log))
                filtered_diff_logs = jtu.tree_map(filter, diff_logs, list(range(len(diff_logs))))

                # if state mappings exist, copy the diff log msg to the correct focal state idx
                if tom_agent.state_mappings is not None:
                    # Pad both collections to the same size for dynamic indexing
                    padded_diff_logs, diff_masks = pad_to_max_size(diff_logs)
                    padded_filtered_logs, filtered_masks = pad_to_max_size(filtered_diff_logs)
                    
                    state_mapping_for_agent = tom_agent.state_mappings[agent_idx-1]
                    
                    # Build the filtered_diff_logs using traced conditionals
                    new_filtered_diff_logs = []
                    for j, mapping_idx in enumerate(state_mapping_for_agent):
                        # Use lax.cond to choose between diff_logs[mapping_idx] or filtered_diff_logs[j]
                        # Both branches must return arrays of the same shape (padded)
                        def use_diff_logs(mapping_idx=mapping_idx, j=j):
                            selected, mask = get_index_from_padded_collection(
                                padded_diff_logs, diff_masks, mapping_idx
                            )
                            # Apply mask to zero out padded elements
                            return jnp.where(mask, selected, 0.0)
                        
                        def use_filtered_logs(mapping_idx=mapping_idx, j=j):
                            # Return the j-th padded filtered log to match shape
                            return padded_filtered_logs[j]
                        
                        padded_result = lax.cond(
                            mapping_idx >= 0,
                            use_diff_logs,
                            use_filtered_logs
                        )
                        
                        # Unpad the result back to the original size for this state factor
                        original_size = filtered_diff_logs[j].shape[0]
                        result = padded_result[:original_size]
                        new_filtered_diff_logs.append(result)
                    
                    filtered_diff_logs = new_filtered_diff_logs

                return filtered_diff_logs
            
            # calculate likelihood messages for focal agent
            ll = calc_likelihood_msgs(prev_beliefs, agent_idx)

            # and add it to the array of lls for all agents
            ll = jtu.tree_map(lambda x, y : x.at[node_idx, agent_idx-1].set(y)[node_idx], t.ll, ll) 

            # Calculate the averaged G of the policy
            # This reflects how "good" the other agent's policy is in terms of absolute expected free energy
            # Watch out, we do use G of index -1, so we cannot fill the final nodes index?
            policy_G = other_tree.G[idx, 0] + jnp.dot(other_tree.G[other_tree.children_indices[idx]][:, 0], other_tree.children_probs[idx])
            t = _update_node(
                t,
                new_idx,
                agent_idx,
                qs=updated_beliefs,
                ll=ll,
                policy=other_tree.policy[idx],
                G=policy_G,
                horizon=horizon + 1,
                depth=t.depth[node_idx, 0] + 1,
                children_indices=jnp.empty((0,)),
                children_probs=jnp.empty((0,)),
                other_tree_idx=t.other_tree_idx[node_idx].at[agent_idx-1].set(idx),
            )
            return t, new_idx

        def do_nothing(t, idx):
            return t, -1

        def consider_add(t, idx):
            return jax.lax.cond(idx >= 0, really_add, do_nothing, t, idx)

        def update_with_policy_children(t, policy_indices, policy_probs):
            t = _update_node(
                t, node_idx, children_indices=policy_indices, children_probs=policy_probs
            )
            return t

        def update_with_policy_dummy(t, policy_indices, policy_probs):
            # jax.debug.print("Expanding ToM but no policies for other agent... add dummy node?!")
            new_idx = jnp.where(
                    ~t.used[:, 0], jnp.arange(t.size), t.size
                ).min()

            # add a dummy node as if we're considering no change in other - one could also expand the other's si tree instead of adding dummy nodes
            t = _update_node(
                t,
                new_idx,
                agent_idx,
                qs=jtu.tree_map(lambda x: x[node_idx], t.qs),
                policy=jnp.zeros_like(t.policy[new_idx]),
                horizon=horizon + 1,
                depth=t.depth[node_idx, 0] + 1,
                children_indices=jnp.empty((0,)),
                children_probs=jnp.empty((0,)),
            )

            policy_indices = policy_indices.at[0].set(new_idx)
            policy_probs = policy_probs.at[0].set(1.0)

            t = _update_node(
                t, node_idx, children_indices=policy_indices, children_probs=policy_probs
            )
            return t

        t, policy_indices = jax.lax.scan(consider_add, t, other_tree.children_indices[other_tree_idx])
        policy_probs = other_tree.children_probs[other_tree_idx]

        t = jax.lax.cond(jnp.all(policy_indices < 0), update_with_policy_dummy, update_with_policy_children, t, policy_indices, policy_probs)

        # jax.debug.print("Other agent {a} q_pi: {q}", a=agent_idx, q=q_pi)

        return t
    
    def _expand_policy_nodes_focal(t, node_idx, agent_idx, agent, horizon):
        # add batch dim of 1 again to the agent, except for policy
        # qs doesnt need an extra batch dim here, since we use step function directly
        qs = jtu.tree_map(lambda x: x[node_idx, agent_idx], t.qs)

        # step function expects an agent without batch dim!
        # calculate expected states, outcomes and free energy for all policies
        # use ll messages from other agents to update qs as well in step()
        ll = jtu.tree_map(lambda x: x[node_idx], t.ll)
        qs_next, qo, G = step(agent, qs, agent.policies, ll, tom_agent.state_mappings)
        q_pi = nn.softmax(G * gamma, axis=0)

        # expand policy nodes
        def add_policy_node(tree, data):
            policy, qs_next, prob, G = data

            def really_add(tree, policy, qs_next, G):
                new_idx = jnp.where(
                    ~tree.used[:, 0], jnp.arange(tree.size), tree.size
                ).min()

                prev_beliefs = jtu.tree_map(lambda x: x[node_idx], t.qs)
                updated_beliefs = jtu.tree_map(lambda x,y: y.at[agent_idx].set(x[node_idx]), qs_next, prev_beliefs)

                tree = _update_node(
                    tree,
                    new_idx,
                    agent_idx,
                    qs=updated_beliefs,
                    policy=policy,
                    G=G,
                    horizon=horizon + 1,
                    depth=tree.depth[node_idx, 0] + 1,
                    children_indices=jnp.empty((0,)),
                    children_probs=jnp.empty((0,)),
                    other_tree_idx=tree.other_tree_idx[node_idx]
                )
                return tree, new_idx
            
            def skip_add(tree, policy, qs_next, G):
                return tree, -1
            
            return lax.cond(prob[0] > policy_prune_threshold, really_add, skip_add, tree, policy, qs_next, G)

        # policies is of shape (n_policies, timesteps (=1), n_actions)
        policies = agent.policies[:, 0, :]
        t, policy_indices = lax.scan(
            add_policy_node, t, (policies, qs_next, q_pi, G)
        )

        # update parent with child indices
        t = _update_node(
            t, node_idx, children_indices=policy_indices, children_probs=q_pi[:, 0]
        )
    
        return t

    def _do_nothing(t, node_idx, agent_idx, agent, horizon):
        return t

    def _expand_first_agent_policy_nodes(tree, node_idx, agent_idx, agent, horizon):
        # expand observation node of focal agent at horizon for "first" other agent
        tree = lax.cond(
            (tree.used[node_idx, 0])
            & (tree.horizon[node_idx, 0] == horizon)
            & (tree.observation[node_idx, 0] >= 0)
            & (tree.agent_idx[node_idx, 0] == num_agents - 1)
            & (tree.used.sum() < tree.size),
            _expand_policy_nodes_other,
            _do_nothing,
            tree,
            node_idx,
            agent_idx,
            agent,
            horizon,
        )
        return tree

    def _expand_other_than_first_policy_nodes(t, node_idx, agent_idx, agent, horizon):
        return lax.cond(agent_idx == 0,
                        _expand_focal_agent_policy_nodes,
                        _expand_other_agent_policy_nodes,
                        t, node_idx, agent_idx, agent, horizon)

    def _expand_other_agent_policy_nodes(tree, node_idx, agent_idx, agent, horizon):
        # expand policy node of previous agent at horizon + 1
        tree = lax.cond(
            (tree.used[node_idx, 0])
            & (tree.horizon[node_idx, 0] == horizon + 1)
            & (tree.policy[node_idx, 0] >= 0)
            & (tree.agent_idx[node_idx, 0] == agent_idx + 1)
            & (tree.used.sum() < tree.size),
            _expand_policy_nodes_other,
            _do_nothing,
            tree,
            node_idx,
            agent_idx,
            agent,
            horizon,
        )
        return tree

    def _expand_focal_agent_policy_nodes(tree, node_idx, agent_idx, agent, horizon):
        # expand policy node of agent 1 at horizon + 1 with policies of focal agent
        tree = lax.cond(
            (tree.used[node_idx, 0])
            & (tree.horizon[node_idx, 0] == horizon + 1)
            & (tree.policy[node_idx, 0] >= 0)
            & (tree.agent_idx[node_idx, 0] == agent_idx + 1)
            & (tree.used.sum() < tree.size),
            _expand_policy_nodes_focal,
            _do_nothing,
            tree,
            node_idx,
            agent_idx,
            agent,
            horizon,
        )
        return tree


    def _expand_observation_nodes(tree, node_idx, agent_idx, agent, horizon):
        # jax.debug.print("Expanding observation nodes at node {n} for agent {a}", n=node_idx, a=agent_idx)

        # get the right qo of node_idx
        all_agents_qs = jtu.tree_map(lambda x: x[node_idx], tree.qs)

        # get other agent self_states with focal agent's world states
        def merge(focal, other, state_idx):
            return jax.lax.cond(jnp.any(world_states == state_idx), lambda : focal, lambda: other)
    
        qs = jtu.tree_map(merge, 
                          jtu.tree_map(lambda x: x[0, 0], all_agents_qs), 
                          jtu.tree_map(lambda x: x[agent_idx, 0], all_agents_qs),
                          list(range(len(all_agents_qs))))
        
        # in case of remappings, now slot in focal agent states into other's qs where appropriate
        other_agent_idx = agent_idx - 1
        if tom_agent.state_mappings is not None:
            def remap_states(qs):
                # Pad both collections to the same size for dynamic indexing
                padded_all_agents_qs_list = [all_agents_qs[i][0, 0] for i in range(len(all_agents_qs))]
                padded_all_agents, agents_masks = pad_to_max_size(padded_all_agents_qs_list)
                padded_qs, qs_masks = pad_to_max_size(qs)
                
                state_mapping_for_agent = tom_agent.state_mappings[other_agent_idx]
                
                # Build the remapped qs using traced conditionals
                new_qs = []
                for j, mapping_idx in enumerate(state_mapping_for_agent):
                    # Use lax.cond to choose between all_agents_qs[mapping_idx] or qs[j]
                    # Both branches must return arrays of the same shape (padded)
                    def use_all_agents(mapping_idx=mapping_idx, j=j):
                        selected, mask = get_index_from_padded_collection(
                            padded_all_agents, agents_masks, mapping_idx
                        )
                        # Apply mask to zero out padded elements
                        return jnp.where(mask, selected, 0.0)
                    
                    def use_qs(mapping_idx=mapping_idx, j=j):
                        # Return the j-th padded qs value to match shape
                        return padded_qs[j]
                    
                    padded_result = lax.cond(
                        mapping_idx >= 0,
                        use_all_agents,
                        use_qs
                    )
                    
                    # Unpad the result back to the original size for this state factor
                    original_size = qs[j].shape[0]
                    result = padded_result[:original_size]
                    new_qs.append(result)
                
                return new_qs

            qs = lax.cond(other_agent_idx >= 0, remap_states, lambda x: x, qs)


        qo = compute_expected_obs(qs, agent.A, agent.A_dependencies)
        qo = jtu.tree_map(lambda x: x[None, ...], qo)

        # now get qs of the agent to infer correct posterior
        # and adjust the batch sizes again :-|
        qs = jtu.tree_map(lambda x: x[agent_idx], all_agents_qs)
        agent = jtu.tree_map(lambda x: x[None] if isinstance(x, jnp.ndarray) else x, agent)
        agent = eqx.tree_at(lambda x: x.policies, agent, agent.policies[0])
        
        # now generate observation nodes like we do in og si_policy_search
        shapes = tuple([o.shape[-1] for o in qo])
        
        # top-k approach: consider most likely k observations per modality
        k = topk_obsspace
        
        def get_topk_for_factor(factor_probs):
            # Use effective k to handle cases where k exceeds modality size bc default k is 10000
            modality_size = factor_probs[0].shape[0]
            k_effective = min(k, modality_size)
            # Get top k indices and their probabilities for this modality
            top_probs, top_indices = jax.lax.top_k(factor_probs[0], k_effective)
            # Renormalise the top probabilities
            top_probs = top_probs / jnp.sum(top_probs)
            return top_indices, top_probs
        
        # Extract top-k data for each modality
        topk_data = [get_topk_for_factor(factor) for factor in qo]
        topk_indices = [data[0] for data in topk_data]
        topk_probs = [data[1] for data in topk_data]
        
        # Calculate actual number of combinations based on effective k values
        k_effective_per_modality = [len(indices) for indices in topk_indices]
        num_combinations = int(np.prod(k_effective_per_modality))
        
        # Generate combinations using imported helper functions
        observations = _generate_observations(shapes, num_combinations, topk_indices)
        probabilities = _calculate_probabilities(num_combinations, topk_probs)

        def add_observation_node(tree, data):
            observation, prob = data

            def consider_add(t):
                # calculate posterior state belief and check if
                # we need to add a new observation node

                # convert to correct dimensions for infer_states
                obs = [o[None, None, ...] for o in observation]

                qs_post = agent.infer_states(obs, qs)
                # remove time dim from qs_post
                qs_post = jtu.tree_map(lambda x: x[:, 0, :], qs_post)

                # TODO check if we already have a node with this belief (or close enough)?
                # and use the KL threshold in ToM as well?!

                new_idx = jnp.where(
                    ~t.used[:, 0], jnp.arange(t.size), t.size
                ).min()

                # jax.debug.print("Add new obs node {x}", x=new_idx)

                # slot the qs_post back into all_agents_qs
                qs_post = jtu.tree_map(lambda x, y: x.at[agent_idx].set(y), all_agents_qs, qs_post)
                
                # in case we are a non-focal agent, we also need to update the other_tree_idx
                def get_other_tree_obs_idx(observation):
                    other_tree_policy_idx = tree.other_tree_idx[node_idx, agent_idx - 1, 0]
                    other_tree_obs_indices = other_trees.children_indices[agent_idx - 1, other_tree_policy_idx]
                    obs_nodes = other_trees.observation[agent_idx - 1, other_tree_obs_indices]
                    check = jax.vmap(lambda o: jnp.allclose(o, observation))(obs_nodes)
                    close_idx = jnp.where(check, size=1, fill_value=-1)[0][0]
                    other_tree_obs_idx = other_tree_obs_indices[close_idx]
                    return other_tree_obs_idx

                other_tree_obs_index = lax.cond(agent_idx > 0,
                        lambda o : tree.other_tree_idx[node_idx].at[agent_idx - 1, 0].set(get_other_tree_obs_idx(o)),
                        lambda o : tree.other_tree_idx[node_idx],
                        observation
                        )

                t = _update_node(
                    t,
                    new_idx,
                    agent_idx,
                    qs=qs_post,
                    observation=observation,
                    horizon=t.horizon[node_idx],
                    depth=t.depth[node_idx, 0] + 1,
                    G=0,
                    children_indices=jnp.empty((0,)),
                    children_probs=jnp.empty((0,)),
                    other_tree_idx=other_tree_obs_index,
                )
                return t, new_idx

            def no_op(t):
                return t, -1

            tree, obs_idx = lax.cond(
                prob > observation_prune_threshold,
                consider_add,
                no_op,
                tree,
            )
            return tree, obs_idx

        tree, obs_indices = lax.scan(
            add_observation_node, tree, (observations, probabilities)
        )

        # update policy parent with child indices
        # get the indices to select, pad with -1
        indices = jnp.where(
            obs_indices >= 0,
            size=tree.children_indices.shape[-1],
            fill_value=-1,
        )[0]
        obs_indices = jnp.concatenate([obs_indices, -jnp.ones(1)])
        obs_indices = obs_indices[indices]

        obs_probabilities = jnp.concatenate([probabilities, jnp.zeros(1)])
        obs_probabilities = obs_probabilities[indices]

        # jax.debug.print("obs_indices: {i}", i=obs_indices)
        # jax.debug.print("obs_probabilities: {p}", p=obs_probabilities)
        tree = _update_node(
            tree,
            node_idx,
            children_indices=obs_indices,
            children_probs=obs_probabilities,
        )
        return tree

    def _expand_focal_agent_observation_nodes(tree, node_idx, agent_idx, agent, horizon):
        # expand policy node of agent 0 at horizon + 1 with observations
        tree = lax.cond(
            (tree.used[node_idx, 0])
            & (tree.horizon[node_idx, 0] == horizon + 1)
            & (tree.policy[node_idx, 0] >= 0)
            & (tree.agent_idx[node_idx, 0] == 0)
            & (tree.used.sum() < tree.size),
            _expand_observation_nodes,
            _do_nothing,
            tree,
            node_idx,
            agent_idx,
            agent,
            horizon,
        )
        return tree

    def _expand_other_agent_observation_nodes(tree, node_idx, agent_idx, agent, horizon):
        # expand policy node of previous agent (agent_idx-1) at horizon + 1 with observations
        tree = lax.cond(
            (tree.used[node_idx, 0])
            & (tree.horizon[node_idx, 0] == horizon + 1)
            & (tree.observation[node_idx, 0] >= 0)
            & (tree.agent_idx[node_idx, 0] == agent_idx -1 )
            & (tree.used.sum() < tree.size),
            _expand_observation_nodes,
            _do_nothing,
            tree,
            node_idx,
            agent_idx,
            agent,
            horizon,
        )
        return tree

    def _backward_node(tree, idx):
        final_other_agent_idx = tree.num_agents - 1
        first_other_agent_idx = 1
        focal_agent_idx = 0

        def _aggregate_G_observation(tree, idx):
            
            def _aggregate_G_observation_children(tree, idx):
                # you are at an observation node, and you need to aggregate the G_recursives of your observation children
                # calulculate the weighted average of the G_recusrives of your children, i.e. weighted by children_probs
                G_recursive = tree.children_probs[idx].dot(tree.G_recursive[tree.children_indices[idx], 0])
                #jax.debug.print("G recursive obs: {g}", g=G_recursive)
                tree = _update_node(tree, idx, G_recursive=G_recursive)
                return tree
            
            def _aggregate_G_observation_final_other(tree, idx):
                # you are the observation node of the final other agent, i.e. agent_idx == num_agents - 1
                # so you are either a leaf node, or you have children that are policy nodes of the final other agent
                # here we need to aggregate the G_recursive of the children, which should now be vectors
                # we als need to update the p_marginalized here, by softmaxing the G_recursives (weighted by gamma)
                G_recursive = tree.children_probs[idx].dot(tree.G_recursive[tree.children_indices[idx]])
                q_pi_marginalized = nn.softmax(G_recursive * gamma)
                G_recursive_scalar = G_recursive.dot(q_pi_marginalized)
                #jax.debug.print("G recursive final other: {g}", g=G_recursive)

                tree = _update_node(
                    tree,
                    idx,
                    G_recursive=G_recursive_scalar,
                    q_pi_marginalized=q_pi_marginalized
                )
                return tree

            tree = lax.cond(
                (tree.agent_idx[idx, 0] == final_other_agent_idx),
                _aggregate_G_observation_final_other,
                _aggregate_G_observation_children,
                tree,
                idx,
            )

            return tree
        
        def _aggregate_G_policy(tree, idx):

            def _aggregate_G_policy_focal(tree, idx):
                # you are at a policy node of the focal agent, i.e. agent_idx == 0
                # your children are observation nodes of the focal agent, which will have G_recursive values
                # calculate the weighted sum of G_recursive values of your children, weighted by children_probs
                # and add your G value to get your G_recursive value
                
                # check if this policy node has any children
                has_children = jnp.any(tree.children_indices[idx] >= 0)
                
                def aggregate_with_children(tree, idx):
                    G_recursive = tree.children_probs[idx].dot(tree.G_recursive[tree.children_indices[idx], 0])
                    G_recursive += tree.G[idx, 0]
                    return tree, G_recursive
                
                def apply_prune_penalty(tree, idx):
                    # apply prune penalty for policy nodes without children
                    G_recursive = tree.children_probs[idx].dot(tree.G_recursive[tree.children_indices[idx], 0])
                    G_recursive -= prune_penalty
                    return tree, G_recursive
                
                tree, G_recursive = lax.cond(
                    has_children,
                    aggregate_with_children,
                    apply_prune_penalty,
                    tree, idx
                )
                #jax.debug.print("G recursive in policy focal: {g}", g=G_recursive)

                
                tree = _update_node(tree, idx, G_recursive=G_recursive)
                return tree
            
            def _aggregate_G_policy_first_other(tree, idx):
                # here we need to turn the G_recursive values of the children into a vector
                # and then we need to update the children_probs by softmaxing the vector (weigthed by gamma)
                
                # check if this policy node has any children
                has_children = jnp.any(tree.children_indices[idx] >= 0)
                
                def aggregate_with_children(tree, idx):
                    G_recursive = tree.G_recursive[tree.children_indices[idx], 0]
                    return tree, G_recursive
                
                def apply_prune_penalty(tree, idx):
                    # apply prune penalty for policy nodes without children
                    G_recursive = tree.G_recursive[tree.children_indices[idx], 0] - prune_penalty
                    return tree, G_recursive
                
                tree, G_recursive = lax.cond(
                    has_children,
                    aggregate_with_children,
                    apply_prune_penalty,
                    tree, idx
                )
                #jax.debug.print("G recursive in policy first other: {g}", g=G_recursive)
                q_pi = nn.softmax(G_recursive * gamma)
                prune_mask = q_pi < policy_prune_threshold
                children_indices = (1-prune_mask) * tree.children_indices[idx] + prune_mask * -1

                tree = _update_node(
                    tree,
                    idx,
                    G_recursive=G_recursive,
                    children_indices=children_indices,
                    children_probs=q_pi
                )
                return tree
            
            def _aggregate_G_policy_other_other(tree, idx):
                # you are at a policy node of an other agent (agent_idx not 0 or 1): take the weighted
                # average of the G_recursive VECTORS of your children, weighted by children_probs
                
                # check if this policy node has any children
                has_children = jnp.any(tree.children_indices[idx] >= 0)
                
                def aggregate_with_children(tree, idx):
                    G_recursive = tree.children_probs[idx].dot(tree.G_recursive[tree.children_indices[idx]])
                    return tree, G_recursive
                
                def apply_prune_penalty(tree, idx):
                    G_recursive = tree.children_probs[idx].dot(tree.G_recursive[tree.children_indices[idx]])
                    G_recursive -= prune_penalty
                    # G_recursive_vector = jnp.full_like(tree.G_recursive[idx], G_recursive)
                    return tree, G_recursive
                
                tree, G_recursive = lax.cond(
                    has_children,
                    aggregate_with_children,
                    apply_prune_penalty,
                    tree, idx
                )

                #jax.debug.print("G recursive in policy other other: {g}", g=G_recursive)

                
                tree = _update_node(
                    tree,
                    idx,
                    G_recursive=G_recursive,
                )
                return tree

            def _aggregate_G_policy_other(tree, idx):
                return lax.cond(
                    (tree.agent_idx[idx, 0] == first_other_agent_idx),
                    _aggregate_G_policy_first_other,
                    _aggregate_G_policy_other_other,
                    tree,
                    idx,
                )

            tree = lax.cond(
                (tree.agent_idx[idx, 0] == focal_agent_idx),
                _aggregate_G_policy_focal,
                _aggregate_G_policy_other,
                tree,
                idx,
            )

            return tree

        tree = lax.cond(
            (tree.observation[idx, 0] >= 0),
            _aggregate_G_observation,
            _aggregate_G_policy,
            tree,
            idx,
        )
        return tree


    def _tree_backward(tree, d):

        def _do_nothing(tree, idx):
            return tree

        def _do_backward(tree, idx):
            tree = lax.cond(
                (tree.used[idx, 0])
                & (tree.depth[idx, 0] == d),
                _backward_node,
                _do_nothing,
                tree,
                idx
            )
            return tree, None

        # update G for all nodes at this horizon
        tree, _ = lax.scan(
            _do_backward, tree, jnp.arange(tree.used.shape[0] - 1, -1, -1)
        )

        # remove orphans if any by pruning
        tree = _remove_orphans(tree)
        return tree, d - 1

    def _expand_horizon(tree, agent, h):
        # jax.debug.print(" ")
        # jax.debug.print("Expanding horizon {h}", h=h)

        def _expand_policies_of_agent(tree, data):
            agent_idx, agent_model = data
            # jax.debug.print("Expanding policies for agent {a} at horizon {h}, prefs {c}", a=agent_idx, h=horizon, c=agent_model.C)

            # for the first (other) agent , i.e. agent_idx == agent.num_agents - 1
            # we have to expand the observation node of agent 0 at horizon
            # for agents idx agent.num_agents - 2 .. 0 
            # we have to expand policy nodes of agent_idx - 1 at horizon + 1

            # note that for agent_idx 0 we don't do belief sharing, but we need it 
            # for the others
            
            # so 3 cases:
            # - agent_idx == agent.num_agents - 1  (expand obs node of agent 0, belief sharing)
            # - agent_idx == 0 (focal agent, expand policy node of agent 1, no belief sharing)
            # - otherwise (expand policy node of agent_idx + 1, belief sharing)

            # agent_idx_to_expand = agent_idx + 1 if agent_idx < agent.num_agents - 1 else 0
            # horizon_to_expand = horizon + 1 if agent_idx < agent.num_agents - 1 else horizon
            # expand_observation = agent_idx == agent.num_agents - 1

            def _expand_node(t, node_idx):
                t = lax.cond(agent_idx == agent.num_agents - 1,
                    _expand_first_agent_policy_nodes,
                    _expand_other_than_first_policy_nodes,
                    t, node_idx, agent_idx, agent_model, h)
                return t, None

            return lax.scan(_expand_node, tree, jnp.arange(tree.used.shape[0]))
        
        # expand policies for each other agent (and do belief sharing), and self
        tree_policies, _  = lax.scan(_expand_policies_of_agent, tree, (jnp.arange(tom_agent.num_agents), tom_agent.agent_models), reverse=True)


        def _expand_observations_of_agent(tree, data):
            agent_idx, agent_model = data
            # jax.debug.print("Expanding observations for agent {a} at horizon {h}", a=agent_idx, h=horizon)

            # we first expand the observation node of the focal agent
            # then we calculate a posterior for the focal
            # then we use the posterior of the focal world states to generate observations for the other agents
            # and calculate posteriors for them
            def _expand_node(t, node_idx):
                t = lax.cond(agent_idx == 0,
                    _expand_focal_agent_observation_nodes,
                    _expand_other_agent_observation_nodes,
                    t, node_idx, agent_idx, agent_model, h)
                return t, None

            return lax.scan(_expand_node, tree, jnp.arange(tree.used.shape[0]))

        # expand observations self, and then other agents
        tree_observations, _  = lax.scan(_expand_observations_of_agent, tree_policies, (jnp.arange(tom_agent.num_agents), tom_agent.agent_models))

        # tree backward
        def backward_pass(tree, agent, h):
            def backward_step(carry, current_d):
                def do_backward(t, cd):
                    return _tree_backward(t, cd)
                
                def skip_backward(t, cd):
                    return t, cd - 1
                
                current_depth = jnp.max(tree.depth)
                
                return lax.cond(
                    current_d <= current_depth,
                    do_backward,
                    skip_backward,
                    carry,
                    current_d
                )

            # Use a static upper bound (horizon * num_agents) and skip nodes beyond actual depth
            max_depth = horizon * tree.num_agents
            # scan from max possible depth down to 0
            tree_bw, _ = lax.scan(backward_step, tree, jnp.arange(max_depth, -1, -1))

            return tree_bw, agent, h + 1

        return lax.cond(tree_observations.used.sum() < tree.used.shape[0] - 1, backward_pass, lambda x,y,z: (tree, agent, horizon), tree_observations, agent, h)


    # expand till horizon
    def continue_expansion(tree, agent, h):
        q_pi = tree.q_pi_marginalized[root_idx(tree)]
        entropy = pymdp.maths.stable_entropy(q_pi)

        return (
            (h < horizon)
            & (jnp.all(q_pi == 0) | (entropy > entropy_stop_threshold))
            & (tree.G[root_idx(tree), 0] < efe_stop_threshold)
        )

    def scan_step(carry, _):
        tree, agent, h, stop_expansion = carry
        
        def do_expand(args):
            tree, agent, h = args
            return _expand_horizon(tree, agent, h)
        
        def do_nothing(args):
            tree, agent, h = args
            return tree, agent, h
        
        # expand if we haven't stopped based on stop criteria (such as entropy or EFE threshold) yet
        tree, agent, h = lax.cond(
            stop_expansion,
            do_nothing,
            do_expand,
            (tree, agent, h)
        )
        
        # check if we should stop next iteration (stay stopped if stop criteria is met)
        stop_expansion = stop_expansion | ~continue_expansion(tree, agent, h)
        
        return (tree, agent, h, stop_expansion), None

    # run scan for max horizon iterations
    initial_h = tree.horizon.max()
    initial_carry = (tree, tom_agent, initial_h, False)
    (tree, tom_agent, _, _), _ = lax.scan(scan_step, initial_carry, None, length=horizon)

    return tree, other_trees
