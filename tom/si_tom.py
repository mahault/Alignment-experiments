# tom/si_tom.py

from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple
import numpy as np

from pymdp.agent import Agent
import jax.numpy as jnp

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def lava_infer_states(
    agent: Agent,
    obs_idx: int,
    qs_prev: Dict[str, Any] | None = None,
    t: int = 0,
) -> None:
    """
    Simple JAX state update for the lava corridor environment.

    Assumptions:
      - Single observation modality (index 0).
      - Single hidden state factor (index 0).
      - A[0].shape == (num_obs, num_states).
      - D[0].shape == (num_states,).
      - qs_prev[k]["qs"] (if provided) is a list with a 1D posterior for this factor.

    We perform one-step Bayes update:

        q(s) ∝ p(o | s) * p_prior(s)

    where p_prior(s) is either:
      - the previous posterior at time t-1 (if available), or
      - the agent's initial prior D[0] (for t == 0 or if no q_prev).
    """
    A0 = agent.A[0]  # jnp.ndarray, shape (num_obs, num_states)
    D0 = agent.D[0]  # jnp.ndarray, shape (num_states,)

    # Choose prior: if we have a previous qs for this agent and t>0, use it;
    # otherwise fall back to the initial D0.
    if t > 0 and qs_prev is not None and "qs" in qs_prev and qs_prev["qs"]:
        # qs_prev["qs"][0] should be a 1D array (num_states,)
        prior = jnp.asarray(qs_prev["qs"][0])
    else:
        prior = D0

    # Likelihood p(o | s) over s for this observation index
    likelihood = A0[obs_idx]  # (num_states,)

    # Unnormalized posterior: p(s | o) ∝ p(o | s) * p_prior(s)
    unnorm = likelihood * prior
    denom = jnp.sum(unnorm)

    # Avoid divide-by-zero: if denom == 0, fall back to uniform.
    qs0 = jnp.where(
        denom > 0.0,
        unnorm / denom,
        jnp.ones_like(unnorm) / unnorm.shape[0],
    )

    # Store as the agent's posterior over the single state factor.
    # This matches the format that Agent.infer_policies expects (list of arrays).
    agent.qs = [qs0]


def run_tom_step(
    agents: List[Agent],
    o: np.ndarray,
    qs_prev: List[Dict[str, Any]] | None,
    t: int,
    learn: bool,
    agent_num: int,
    B_self: np.ndarray,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Run a single Theory-of-Mind step for all K agents.

    This is a factored-out version of the logic originally inside an
    EmpatheticAgent._theory_of_mind, extended to also return:

    - EFE_arr:  shape [K, num_policies]  (expected free energy per policy)
    - Emp_arr:  shape [K, num_policies]  (placeholder for empowerment, currently zeros)

    Args
    ----
    agents: list of pymdp.Agent
        The K agent models used for ToM (self + others).
    o: np.ndarray
        Observation array of shape [K], each entry is an int observation category.
    qs_prev: list of previous ToM results (qs per agent) from last time step, or None at t=0.
    t: int
        Current time index.
    learn: bool
        Whether to update B (transition) based on inferred actions.
    agent_num: int
        Index of the *real* agent in this EmpatheticAgent wrapper (for learning others' actions).
    B_self: np.ndarray
        The "self" B-matrix used for empirical prior updates (same as EmpatheticAgent.B).

    Returns
    -------
    tom_results: list[dict]
        For each k in [0..K-1]:
            {
                "qs":   variational state posterior(s),
                "G":    EFE per policy (np.ndarray[num_policies]),
                "q_pi": policy posterior (np.ndarray[num_policies]),
                "action": chosen action (np.ndarray or scalar)
            }

    EFE_arr: np.ndarray
        Array of shape [K, num_policies], stacking each agent's G.

    Emp_arr: np.ndarray
        Placeholder empowerment array of shape [K, num_policies].
        Currently all zeros; to be replaced later with real empowerment estimates.
    """
    K = len(agents)
    assert o.shape[0] == K, "Observation array must match number of agents."

    LOGGER.debug(f"Running ToM step t={t} for {K} agents, learn={learn}")

    tom_results: List[Dict[str, Any]] = []

    # --- ToM inference for each agent clone ---
    for k in range(K):
        LOGGER.debug(f"  Processing ToM agent {k}, observation={int(o[k])}")

        # 1) Infer hidden states given observation using a direct JAX Bayes update.
        #    We bypass pymdp.Agent.infer_states here because the JAX/vmap path
        #    in v1.0.0_alpha has a batch-axis mismatch with the maths helpers.
        obs_idx = int(o[k])

        this_q_prev = qs_prev[k] if (qs_prev is not None and k < len(qs_prev)) else None
        lava_infer_states(agents[k], obs_idx, qs_prev=this_q_prev, t=t)

        LOGGER.debug(
            f"    Agent {k} state inference complete; "
            f"qs[0].shape={agents[k].qs[0].shape}, "
            f"min={float(agents[k].qs[0].min()):.4f}, "
            f"max={float(agents[k].qs[0].max()):.4f}"
        )

        # 2) Optional learning of B
        if learn:
            _update_B_with_learning(
                agents=agents,
                o=o,
                t=t,
                k=k,
                qs_prev=qs_prev,
                agent_num=agent_num,
                B_self=B_self,
            )
            LOGGER.debug(f"    Agent {k} B-matrix updated")

        # 3) Infer policies & sample action
        agents[k].infer_policies()
        agents[k].sample_action()
        LOGGER.debug(
            f"    Agent {k} policy inference complete, "
            f"sampled action: {agents[k].action}"
        )

        # 4) Store results
        tom_results.append(
            {
                "qs": agents[k].qs,
                "G": agents[k].G,          # shape [num_policies]
                "q_pi": agents[k].q_pi,    # shape [num_policies]
                "action": agents[k].action # typically scalar/array with chosen control
            }
        )

    # --- Build EFE_arr and a placeholder Emp_arr ---
    num_policies = tom_results[0]["G"].shape[0]
    EFE_arr = np.zeros((K, num_policies))
    Emp_arr = np.zeros((K, num_policies))  # will later hold empowerment per agent/policy

    for k in range(K):
        EFE_arr[k] = tom_results[k]["G"]
        # NOTE: Emp_arr[k] will be filled with empowerment estimates later.

    LOGGER.debug(f"ToM step complete: EFE_arr shape={EFE_arr.shape}, Emp_arr shape={Emp_arr.shape}")
    LOGGER.debug(f"  EFE_arr mean={EFE_arr.mean():.4f}, Emp_arr mean={Emp_arr.mean():.4f}")

    return tom_results, EFE_arr, Emp_arr


def _update_B_with_learning(
    agents: List[Agent],
    o: np.ndarray,
    t: int,
    k: int,
    qs_prev: List[Dict[str, Any]] | None,
    agent_num: int,
    B_self: np.ndarray,
) -> None:
    """
    Encapsulate the B-learning logic from EmpatheticAgent._learn(),
    so we can keep run_tom_step self-contained.

    This is structurally equivalent to your existing code, but pulled out.
    """
    if t == 0 or qs_prev is None:
        LOGGER.debug(f"      Skipping B-learning for agent {k}: t=0 or no previous qs")
        return

    # If this is the "real" agent's own model:
    if k == agent_num:
        LOGGER.debug(f"      Agent {k}: updating B with own previous qs")
        agents[k].update_B(qs_prev[k]["qs"])
    else:
        # If this is a ToM agent, infer others' actions from observations.
        # For now we replicate your original heuristic; you can swap this out later.
        inferred_action = _infer_others_action(o=o, agent_num=agent_num, k=k)
        LOGGER.debug(f"      Agent {k}: inferred other's action={inferred_action}")
        agents[k].action = inferred_action
        agents[k].update_B(qs_prev[k]["qs"])


def _infer_others_action(o: np.ndarray, agent_num: int, k: int) -> np.ndarray:
    """
    Brute-force heuristic from your original infer_others_action implementation.
    For K=2 PD-style scenarios:

        if k == 1:
            if o[self.agent_num] in {0, 2}: action = 0 (Cooperate)
            else:                            action = 1 (Defect)
        else:
            if o[self.agent_num] in {0, 1}: action = 0 (Cooperate)
            else:                            action = 1 (Defect)
    """
    if k == 1:
        if o[agent_num] == 0 or o[agent_num] == 2:
            action = np.array([0.])  # Cooperate
        else:
            action = np.array([1.])
    else:
        if o[agent_num] == 0 or o[agent_num] == 1:
            action = np.array([0.])  # Cooperate
        else:
            action = np.array([1.])

    LOGGER.debug(f"        Inferred action for k={k} based on o[{agent_num}]={o[agent_num]}: {action}")
    return action
