import logging
import numpy as np

# from numpy.typing import ndarray
from pymdp.agent import Agent
from pymdp.control import sample_action
from pymdp.maths import softmax

# Try to import dirichlet_like from pymdp, but fall back to local implementation
try:
    from pymdp.utils import dirichlet_like  # type: ignore[attr-defined]
except ImportError:
    def dirichlet_like(p_array, scale: float = 1.0):
        """
        Local fallback for pymdp.utils.dirichlet_like.

        Given a categorical distribution (or an object-array of such),
        return Dirichlet concentration parameters of the same shape,
        where each categorical vector is normalised and multiplied by `scale`.
        """
        # Object-array case (pymdp.obj_array)
        if isinstance(p_array, np.ndarray) and p_array.dtype == object:
            out = np.empty_like(p_array, dtype=object)
            for idx, arr in enumerate(p_array):
                arr = np.array(arr, dtype=float)
                s = arr.sum()
                if s == 0:
                    # fallback to uniform
                    arr = np.ones_like(arr) / arr.size
                else:
                    arr = arr / s
                out[idx] = arr * scale
            return out

        # Plain ndarray / list case
        arr = np.array(p_array, dtype=float)
        s = arr.sum()
        if s == 0:
            arr = np.ones_like(arr) / arr.size
        else:
            arr = arr / s
        return arr * scale

from tom.si_tom import run_tom_step

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class EmpatheticAgent:
    def __init__(self, config: dict, agent_num: int) -> None:

        self.agent_num = agent_num
        LOGGER.info(f"Initializing EmpatheticAgent {agent_num}")

        # Pull out config parameters for cleaner access within class
        self.A              = config["A"][self.agent_num]
        self.B              = config["B"][self.agent_num]
        self.C              = config["C"][self.agent_num]
        self.D              = config["D"][self.agent_num]
        self.empathy_factor = config["empathy_factor"][self.agent_num]
        self.K              = config["K"]
        self.actions        = config["actions"]
        self.learn          = config["learn"]
        self.policy_len     = config["policy_len"]

        # NEW: optional empowerment weight gamma (how much empowerment matters vs EFE)
        # If not provided in config, default to 1.0
        self.gamma          = config.get("gamma", 1.0)

        LOGGER.debug(f"  Agent {agent_num}: empathy_factor={self.empathy_factor}, gamma={self.gamma}, learn={self.learn}")

        # Get number of observation categories to use in generating observation for t=0
        num_obs_categories = self.A[0][self.agent_num].shape[0]

        # Initialize empty containers for EFE and agents
        self.EFE    = np.zeros(self.K)
        self.agents = []

        # Initialize agent's observation at t=0 by generation using likelihood (A) and state prior
        sampled_obs = np.random.choice(np.arange(num_obs_categories), p=self.A[0] @ self.D[0])
        assert isinstance(sampled_obs, np.int64), "Sampled observation must be an integer."
        self.o_init = np.array([sampled_obs] * self.K)   # Duplicate for each K agents

        # Initialize all K active inference agents in the simulation
        for k in range(self.K):
            if self.learn:
                if config["same_pref"]:
                    self.agents.append(
                        Agent(
                            A=self.A,
                            B=self.B,
                            C=self.C,
                            D=self.D,
                            pB=dirichlet_like(self.B),
                            lr_pB=0.5,
                            policy_len=self.policy_len,
                        )
                    )
                else:
                    self.agents.append(
                        Agent(
                            A=self.A,
                            B=config["B"][k],
                            C=config["C"][k],
                            D=self.D,
                            pB=dirichlet_like(self.B),
                            lr_pB=0.5,
                            policy_len=self.policy_len,
                        )
                    )
            else:
                self.agents.append(
                    Agent(
                        A=self.A,
                        B=self.B,
                        C=self.C,
                        D=self.D,
                        policy_len=self.policy_len,
                    )
                )

        # Get the policy list and number of policies for use later
        self.policies     = self.agents[0].policies
        self.num_policies = len(self.policies)
        self.num_actions  = [len(config["actions"])]

        # Initialize previous variational state posterior with state prior
        self.qs_prev = None

        LOGGER.info(f"  Agent {agent_num} initialized with {self.num_policies} policies and {self.K} ToM agents")

    def step(self, t: int, o: "np.ndarray") -> list:
        """
        Each agent step consists of the following:
        1. Perform theory of mind (ToM) by running K copies of the agent.
        2. Determine agent's overall EFE by using a weighted average of its own EFE
           and all other agents' EFE (empathy).
        3. [TODO] Use weighted VFE to determine variational state posterior.
        4. Use augmented, empathy-weighted EFE (including empowerment if available)
           to determine variational policy posterior, action marginal, and chosen action.
        5. [TODO] Determine emotion state of agent from weighted VFE and EFE.
        """
        LOGGER.debug(f"Agent {self.agent_num} step t={t}")

        if t == 0:
            o = self.o_init
            LOGGER.debug(f"  t=0: using initial observation {o}")

        # Create empty container for storing step results
        step_results = {}

        # Run Theory of Mind step for all K agents
        # This returns: tom_results, EFE_arr [K, num_policies], Emp_arr [K, num_policies]
        tom_results, EFE_arr, Emp_arr = run_tom_step(
            agents=self.agents,
            o=o,
            qs_prev=self.qs_prev,
            t=t,
            learn=self.learn,
            agent_num=self.agent_num,
            B_self=self.B,
        )
        step_results["tom_results"] = tom_results

        # Store previous qs so it can be accessed later for learning
        self.qs_prev = step_results["tom_results"]

        LOGGER.debug(f"  EFE_arr shape: {EFE_arr.shape}, mean: {EFE_arr.mean():.4f}")
        LOGGER.debug(f"  Emp_arr shape: {Emp_arr.shape}, mean: {Emp_arr.mean():.4f}")

        # Calculate agent's expected EFE, augmented by empowerment and empathy
        exp_EFE = self._expected_value_EFE(EFE_arr=EFE_arr, Emp_arr=Emp_arr)
        LOGGER.debug(f"  Expected EFE (augmented): mean={exp_EFE.mean():.4f}, min={exp_EFE.min():.4f}, max={exp_EFE.max():.4f}")

        # Calculate agent's expected Q_pi and action
        exp_q_pi = softmax(exp_EFE)
        p_u = sample_action(
            q_pi=exp_q_pi,
            policies=self.policies,
            num_controls=self.num_actions,
        )
        exp_action = p_u[0]

        # Assemble final step results
        step_results["qs"]         = step_results["tom_results"][0]["qs"]
        step_results["exp_G"]      = exp_EFE
        step_results["exp_q_pi"]   = exp_q_pi
        step_results["exp_p_u"]    = p_u
        step_results["exp_action"] = exp_action

        LOGGER.debug(f"  Agent {self.agent_num} chose action: {exp_action} (action idx: {int(exp_action)})")

        return step_results

    # =========================================================================
    # DEPRECATED METHODS - Replaced by tom.si_tom.run_tom_step
    # Kept for reference only. These are no longer called by step().
    # =========================================================================

    def _theory_of_mind(self, o: "np.ndarray", qs_prev: list, t: int) -> list:
        """
        DEPRECATED: Use tom.si_tom.run_tom_step instead.

        Run K copies of the agent loop by inferring states, policies, and then sampling
        actions.
        """
        tom_results = []

        # Theory of mind simulation for self (agent k = 0) and others (agent k > 0)
        for k in range(self.K):
            # Each ToM agent receives its own observation index
            # Pass observation as array with factor dimension to match A/B's factor axis
            import numpy as np
            obs_idx = int(o[k])
            # Make it a 1D array of length = num_factors (here 1)
            obs_idx_arr = np.array([obs_idx], dtype=np.int32)   # shape (1,)
            self.agents[k].infer_states([obs_idx_arr], empirical_prior=None)

            if self.learn:
                self._learn(o=o, t=t, k=k, qs_prev=qs_prev)

            self.agents[k].infer_policies()
            self.agents[k].sample_action()

            # Add results of simulation to dictionary
            tom_results.append({
                "qs"     : self.agents[k].qs,
                "G"      : self.agents[k].G,
                "q_pi"   : self.agents[k].q_pi,
                "action" : self.agents[k].action,
            })

        return tom_results

    def _learn(self, o:"np.ndarray", t:int, k: int, qs_prev: "np.ndarray"):
        """DEPRECATED: Logic moved to tom.si_tom._update_B_with_learning"""
        if t > 0:
            if k == self.agent_num:
                # If this is the agent's own model
                self.agents[k].update_B(qs_prev[k]["qs"])
            else:
                # If this is a ToM agent, update the B matrix with the action the real agent took
                self.agents[k].action = self.infer_others_action(o, k)
                self.agents[k].update_B(qs_prev[k]["qs"])
        else:
            pass

    def infer_others_action(self, o:"np.ndarray", k: int):
        """
        DEPRECATED: Logic moved to tom.si_tom._infer_others_action

        Infers the action of the other agents based on the current state and this agent's previous action.
        This is used to update the B matrix of the ToM agents.
        """
        # Get the action of the agent at time t-1
        # Brute forced for first experiment
        if k == 1:
            if o[self.agent_num] == 0 or o[self.agent_num] == 2:
                action = np.array([0.])  # Cooperate
            else:
                action = np.array([1.])
        else:
            if o[self.agent_num] == 0 or o[self.agent_num] == 1:
                action = np.array([0.])  # Cooperate
            else:
                action = np.array([1.])
        return action

    def _empirical_prior(self, k: int):
        return np.einsum(
            "ji, i -> j",
            self.B[0][:, :, int(self.agents[k].action.flatten()[0])],
            self.agents[k].qs[0].flatten(),
        )

    def _extract_tom_EFE(self, tom_results: list) -> "np.ndarray":
        """
        DEPRECATED: EFE_arr is now returned directly from tom.si_tom.run_tom_step

        Extracts all EFE calculations for ToM agents
        """
        EFE_arr = np.zeros((self.K, self.num_policies))
        for k in range(self.K):
            EFE_arr[k] = tom_results[k]["G"]
        return EFE_arr

    def _compute_empowerment_matrix(self, tom_results: list) -> "np.ndarray":
        """
        DEPRECATED: Emp_arr is now returned directly from tom.si_tom.run_tom_step

        Placeholder for empowerment (path flexibility) per ToM agent and policy.

        Returns:
            Emp_arr: np.ndarray of shape [K, num_policies]

        For now this returns zeros, so behaviour is unchanged from the original
        empathy-only version. Later, you can replace this with a real computation
        using each agent's A, B, and q_pi to approximate empowerment as
        I(A; O_future) for each policy.
        """
        Emp_arr = np.zeros((self.K, self.num_policies))
        # TODO: compute per-agent, per-policy empowerment here.
        # For example:
        #   for k in range(self.K):
        #       Emp_arr[k] = self._compute_empowerment_for_agent(k, tom_results[k])
        return Emp_arr

    def _expected_value_EFE(self, EFE_arr: 'np.ndarray', Emp_arr: 'np.ndarray' = None) -> "np.ndarray":
        """
        Computes the expected value over *augmented* EFE for each policy by
        weighting the contributions of all simulated agents using the empathy factor.

        For each policy p:
            - EFE_arr[:, p]  are the EFEs for all ToM agents under policy p
            - Emp_arr[:, p]  are their empowerment values under policy p
            - self.empathy_factor is a vector of weights over those agents

        We construct an augmented per-agent EFE:
            augmented_k,p = EFE_k,p - gamma * Emp_k,p

        and then take the empathy-weighted average across k.
        """

        exp_EFE = np.zeros(self.num_policies)

        # If no empowerment provided, treat as zero (reduces to original behaviour).
        if Emp_arr is None:
            Emp_arr = np.zeros_like(EFE_arr)

        gamma = getattr(self, "gamma", 1.0)

        for p in range(self.num_policies):
            # Per-agent augmentation: G - gamma * Emp
            augmented = EFE_arr[:, p] - gamma * Emp_arr[:, p]

            # Empathy-weighted combination across agents
            exp_EFE[p] = augmented @ self.empathy_factor

        return exp_EFE

    def _VFE(self):
        # Expected value of VFE for empathetic agent
        # - Sum_i VFE * empathy_factor
        raise NotImplementedError

    def _emotion_state(self):
        # EFE/VFE -> emotion state
        raise NotImplementedError
