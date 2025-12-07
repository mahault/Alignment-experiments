# tom/si_tom.py

from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple
import numpy as np

from pymdp.agent import Agent

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


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

    This is a factored-out version of the logic currently inside
    EmpatheticAgent._theory_of_mind, extended to also return:

    - EFE_arr:  shape [K, num_policies]  (same as before)
    - Emp_arr:  shape [K, num_policies]  (stub for empowerment, currently zeros)

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

        # 1) Infer hidden states given observation
        #
        # IMPORTANT: pass a scalar observation index per modality, and let
        # pymdp.Agent handle conversion to one-hot internally. This matches
        # the standard (unbatched) use of get_likelihood_single_modality.
        obs_idx = int(o[k])
        agents[k].infer_states([obs_idx], empirical_prior=None)
        LOGGER.debug(f"    Agent {k} state inference complete")

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
        LOGGER.debug(f"    Agent {k} policy inference complete, sampled action: {agents[k].action}")

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
