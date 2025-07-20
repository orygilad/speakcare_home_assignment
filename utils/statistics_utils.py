from scipy.stats import norm
import numpy as np
import torch
def bayesian_error(mu1, var1, mu2, var2):
    """
    Computes Bayesian error under equal-σ assumption using pooled variance:
        σ_pooled = sqrt((var1 + var2)/2)
    """
    sigma_pooled = np.sqrt((var1 + var2) / 2)
    delta_mu = abs(mu1 - mu2)
    z = delta_mu / (2 * sigma_pooled)
    return norm.cdf(-z)
def compute_confidence(probs, probs_vars):
    top_idx = torch.argmax(probs).item()
    top_mu = probs[top_idx].item()
    top_var = probs_vars[top_idx].item()

    # Zero out top and find second-best
    mean_probs_clone = probs.clone()
    mean_probs_clone[top_idx] = -float('inf')
    second_idx = torch.argmax(mean_probs_clone).item()
    second_mu = probs[second_idx].item()
    second_var = probs_vars[second_idx].item()
    # Replace your previous confidence calc with this:
    p_error = bayesian_error(top_mu, top_var, second_mu, second_var)
    confidence = 1 - p_error
    return top_idx , confidence
