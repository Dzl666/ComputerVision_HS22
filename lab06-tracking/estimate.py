import numpy as np

def estimate(particles, weights):
    """
    :particles: size[num_p, state_dim]
    :weights: size[num_p, 1]
    Return
    :mean state: size[state_dim, 1]
    """
    weighted_p = np.dot(weights.T, particles)
    # normalize before return
    return weighted_p