import itertools
import numpy as np

from sim import Sim

class Sweep:
    """ 
    This class setups up hyperparameter sweeps for the simulation.
    This is specifically for the purpose of determining cooperation/defection behavioral outcomes.
    """
    def __init__(self, n_agents: int, config: dict, tuning_vars: list, tuning_settings: list):
        # Main class variables
        self.n_agents    = n_agents
        self.config      = config
        self.tuning_vars = tuning_vars
        
        # Empty container to store results of sweep
        self.results     = []
        
        # Ensure the length of variables in config match the number of agents
        for var in ["A", "B", "C", "D", "empathy_factor"]:
            assert len(self.config[var]) == self.n_agents
        
        # Ensure the tuning variables selected are present in the config
        # Ensure that variable to tune is None
        for var in tuning_vars:
            assert var in list(self.config.keys())
            assert all(elem is None for elem in self.config[var])
            
        assert len(tuning_settings) == self.n_agents
        
        # Determine Cartesian product of tuning settings
        self.combinations = list(itertools.product(tuning_settings[0], tuning_settings[1]))
        
    def run(self):
        
        for var in self.tuning_vars:
            for setting_idx in range(len(self.combinations)):
                print(f"Setting: {setting_idx}")
                for agent in range(self.n_agents):
                    self.config[var][agent] = np.array(self.combinations[setting_idx][agent])
                
                simulation = Sim(config=self.config)
                history = simulation.run(verbose=False)
                self._append_history(history=history)
    
    def _append_history(self, history: dict):
        
        setting_results = []
        for agent in range(self.n_agents):
            actions = history["results"]["action"][:, agent]
            labeled_action = [self.config["actions"][int(u)] for u in actions][-1]
            setting_results.append(labeled_action)
        
        self.results.append("".join(setting_results))
