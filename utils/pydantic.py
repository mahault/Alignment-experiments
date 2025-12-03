""" Pydantic base models for type checks """
# TODO: Consider that a DataClass may be better for this because it
# natively supports numpy array types unlike pydantic

import numpy as np

from pydantic import BaseModel

class ConfigArgs(BaseModel):
    A: np.ndarray                # A array [likelihood / observation model]
    B: np.ndarray                # B array [transition matrix]
    C: np.ndarray                # C array [preferences]
    D: np.ndarray                # D array [initial state prior]
    T: int                       # Number of time steps in simulation
    K: int                       # Number of agents in simulation
    empathy_factor: np.ndarray   # Weighting for each agent versus others
    actions: list                # Number of actions each agent can take
    A_temp_range: list           # Range for A matrix temperature parameter
    
    class Config:
        arbitrary_types_allowed = True