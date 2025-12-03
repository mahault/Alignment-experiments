import numpy as np
import torch

def shannon_entropy(probabilities, axis = None, base=2):
    """Compute Shannon entropy of a probability distribution"""
    return np.round(-np.sum(probabilities * np.log(probabilities + 1e-12) / np.log(base), axis = axis), 4)



def compute_empowerment(
        p_o_given_a: torch.Tensor, 
        tol: float = 1e-8, 
        max_iter = 1000,
        base = 2
    ) -> tuple[torch.Tensor, float]:
    """
    Compute empowerment over p(a) using Blahut–Arimoto algorithm.

    Args:
        p_o_given_a (torch.tensor): of shape [n_actions, n_obs], p(o|a)
    Returns: 
        p(a) (torch.tensor): of shape [n_actions,], optimal action distribution
        empowerment value (float) 
    """
    n_actions, n_obs = p_o_given_a.shape
    p_a = torch.full((n_actions,), 1.0 / n_actions, dtype=torch.double)

    converged = False
    for iter in range(int(max_iter)):
        p_o = (p_a[:, None] * p_o_given_a).sum(0)   # p(o)
        log_ratio = torch.log(p_o_given_a + tol) - torch.log(p_o + tol)   # log q(o|a)/p(o)
        f_a = (p_o_given_a * log_ratio).sum(1)    # expectation over o
        new_p_a = torch.softmax(f_a, dim=0)

        if torch.max(torch.abs(new_p_a - p_a)) < tol: 
            print(f"Converged in {iter} iterations.")
            converged = True
            break
            
        p_a = new_p_a
    if not converged:
        print(f"Warning: Blahut-Arimoto algorithm did not converge in {max_iter} iterations.")
    # Compute empowerment
    p_o = (p_a[:, None] * p_o_given_a).sum(0)
    empowerment = (p_a[:, None] * p_o_given_a * (torch.log(p_o_given_a + tol,) - torch.log(p_o + tol))).sum().item()
    return np.round(p_a, 4), empowerment / np.log(base)  

import matplotlib.pyplot as plt

def plot_distribution(p, title='Distribution', x_labels=None, y_labels=None):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.matshow(p, vmin=0, vmax=1, cmap='hot_r')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.subplots_adjust(wspace=0.4)

    if x_labels is not None:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels)

    if y_labels is not None:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    plt.show()
