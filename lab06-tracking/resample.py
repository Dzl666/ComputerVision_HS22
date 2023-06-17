import numpy as np

def resample(particles, weights):
    """
    resample the particles based on their weights, 
    and return these new particles along with their corresponding weights.
    """
    new_p = np.zeros_like(particles)
    new_w = np.zeros_like(weights)
    idx = range(particles.shape[0])
    idx_resample = np.random.choice(idx, particles.shape[0], p=weights[:,-1])
    for i in range(particles.shape[0]):
        new_p[i] = particles[idx_resample[i]]
        new_w[i] = weights[idx_resample[i]]

    return (new_p, new_w/np.sum(new_w))