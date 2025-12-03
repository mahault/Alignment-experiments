import logging
import numpy as np

# from numpy.typing import ndarray
from pymdp.agent import Agent
from pymdp.control import sample_action
from pymdp.maths import softmax
from pymdp.utils import dirichlet_like

LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)


class EmpatheticAgent:
    def __init__(self, config: dict, agent_num: int) -> None:
        
        self.agent_num = agent_num
        
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
                    self.agents.append(Agent(A=self.A, B=self.B, C=self.C, D=self.D, 
                                         pB=dirichlet_like(self.B), lr_pB=0.5, policy_len=self.policy_len))
                else:
                    self.agents.append(Agent(A=self.A, B=config["B"][k], C=config["C"][k], D=self.D, 
                                         pB=dirichlet_like(self.B), lr_pB=0.5, policy_len=self.policy_len))
                    
            else:
                # self.agents.append(Agent(A=self.A, B=self.B, C=self.C, D=self.D, 
                #                      use_states_info_gain=False,
                #                      use_param_info_gain=False,
                #                      use_utility=True,
                #                      policy_len=1))
                self.agents.append(Agent(A=self.A, B=self.B, C=self.C, D=self.D, policy_len=self.policy_len))
        
        # Get the policy list and number of policies for use later
        self.policies     = self.agents[0].policies
        self.num_policies = len(self.policies)
        self.num_actions  = [len(config["actions"])]
        
        # Initialize previous variational state posterior with state prior
        self.qs_prev = None
            
    def step(self, t: int, o: "np.ndarray") -> list:
        """ 
        Each agent step consists of the following:
        1. Perform theory of mind (ToM) by running K copies of the agent.
        2. Determine agent's overall EFE by using as weighted average of its own EFE
           and all other agent's EFE.
        3. [TODO] Use weighted VFE to determine variational state posterior.
        3. Use weighted EFE to determine variational policy posterior, action marginal,
           and chosen action.
        4. [TODO] Determine emotion state of agent from weighted VFE and EFE
        """
        if t == 0:
            o = self.o_init
        
        # Create empty container for storing step results
        step_results = {}
        
        # Construct ToM results and add to overall step results
        step_results["tom_results"] = self._theory_of_mind(o=o, qs_prev=self.qs_prev, t=t)
        
        # Store previous qs so it can be accessed later for learning
        self.qs_prev = step_results["tom_results"]
        
        # Calculate agent's expected EFE
        EFE_arr = self._extract_tom_EFE(tom_results=step_results["tom_results"])
        exp_EFE = self._expected_value_EFE(EFE_arr=EFE_arr)
        
        # Calculate agent's expected Q_pi and action
        exp_q_pi = softmax(exp_EFE)
        p_u = sample_action(
                q_pi=exp_q_pi, policies=self.policies, num_controls= self.num_actions)
        exp_action = p_u[0]
        # LOGGER.info(">>>>>>>>>>>>>>>>")
        # LOGGER.info(f"Weighted EFE: {exp_EFE}")
        # LOGGER.info(f"Actual action: {int(exp_action)}")
        # LOGGER.info("===============")
        
        # TODO: Calculate emotion state for the agent
        
        # Assemble final step results
        step_results["qs"]         = step_results["tom_results"][0]["qs"]
        step_results["exp_G"]      = exp_EFE
        step_results["exp_q_pi"]   = exp_q_pi
        step_results["exp_p_u"]    = p_u
        step_results["exp_action"] = exp_action
        
        return step_results
    
    def _theory_of_mind(self, o: "np.ndarray", qs_prev: list, t: int) -> dict:
        """ 
        Run K copies of the agent loop by inferring states, policies, and then sampling
        actions.
        """
        
        tom_results = []
        
        # Theory of mind simulation for self (agent k = 0) and others (agent k > 0)
        for k in range(self.K):
            self.agents[k].infer_states([int(o[k])])   # Each agent gets the same observation

            if self.learn:
                self._learn(o=o, t=t, k=k, qs_prev=qs_prev)
            
            self.agents[k].infer_policies()
            self.agents[k].sample_action()

            #empirical_prior = self._empirical_prior(k=k)
            
            # LOGGER.info(f"Simulation index: {k}")
            # LOGGER.info(f"Empirical_prior: {empirical_prior}")
            # LOGGER.info(f"EFE: {self.agents[k].G}")
            # LOGGER.info(f"Action: {int(self.agents[k].action.flatten()[0])}")
            
            # Add results of simulation to dictionary
            # TODO: Add variational free energy to step results
            tom_results.append({
                "qs"     : self.agents[k].qs,
                "G"      : self.agents[k].G,
                "q_pi"   : self.agents[k].q_pi,
                "action" : self.agents[k].action
            })
            
        return tom_results
    
    def _learn(self, o:"np.ndarray", t:int, k: int, qs_prev: "np.ndarray"):
        if t > 0:
            if k == self.agent_num:
                # If this is the agents model
                self.agents[k].update_B(qs_prev[k]["qs"])
            else:
                # If this is a ToM agent, update the B matrix with the action the real agent took
                self.agents[k].action = self.infer_others_action(o, k)
                #print(f"From state {qs_prev[k]["qs"]} with action {self.agents[k].action} to state {self.agents[k].qs}")
                self.agents[k].update_B(qs_prev[k]["qs"])
        else:
            pass

    def infer_others_action(self, o:"np.ndarray", k: int):
        """ 
        Infers the action of the other agents based on the current state and this agents previous action.
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
        return np.einsum("ji, i -> j", 
                        self.B[0][:, :, int(self.agents[k].action.flatten()[0])], 
                        self.agents[k].qs[0].flatten())
    
    def _extract_tom_EFE(self, tom_results: list) -> "np.ndarray":
        """ Extracts all EFE calculations for ToM agents """
        
        EFE_arr = np.zeros((self.K, self.num_policies))
        for k in range(self.K):
            EFE_arr[k] = tom_results[k]["G"]
            
        return EFE_arr

    def _expected_value_EFE(self, EFE_arr: 'np.ndarray') -> "np.ndarray":
        """ 
        Computes the expected EFE over policies by weighting the EFE of all simulated agents 
        
        For example, if there are three agents, there would be three EFEs for a policy.
        These EFEs are weighted according to the self.empathy_factor
        
        EFE    = [EFE_0, EFE_1, EFE_2]   (EFEs for each agent under policy 1)
        p(EFE) = [0.9  , 0.05 , 0.05]    (Empathy factor weighting for each EFE)
        
        When the expected value, sum_k EFE_k * p(EFE_k), is taken, we get a model average of EFE
        for that particular policy. This method applies this across the entire policy space to 
        give an expected EFE across policies.
        """

        exp_EFE = np.zeros(self.num_policies) 
        
        # Loop over each agent and then calculate the weighted average of EFE For each
        # policy.
        
        for p in range(self.num_policies):
            exp_EFE[p] = EFE_arr[:, p] @ self.empathy_factor    
        return exp_EFE
    
    def _VFE(self):
        # Expected value of VFE for empathetic agent
        # - Sum_i VFE * empathy_factor
        raise NotImplementedError
    
    def _emotion_state(self):
        # EFE/VFE -> emotion state 
        raise NotImplementedError
            