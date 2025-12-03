import numpy as np


def softmax_temperature(x, temp: float) -> "np.ndarray":
    beta = 1 / temp
    return np.exp(np.divide(x, beta)) / np.sum(np.exp(np.divide(x, beta)), axis=0)
