import numpy as np
import random as rd 

def propagate(particles, height, width, params):
    """
    propagate the particles given the system prediction model (matrix A) and the system model noise
    Make sure that the center of the particle lies inside the frame.

    :particles: [num_p, state_dim]"""
    s_t = np.zeros_like(particles)
    # A deterministic matrix: [state_dim, state_dim]
    A = np.array(
        [[1, 0], 
        [0, 1]]
    )
    if(params["model"] == 1):
        A = np.array(
            [[1,0,0,0],
            [0,1,0,0],
            [1,0,1,0],
            [0,1,0,1]]
        )

    s_t = particles.dot(A)
    for i in range(2):
        for j in range(s_t.shape[0]):
            s_t[j][i] += rd.gauss(0, params["sigma_position"])
    if(params["model"] == 1):
        for i in range(2):
            for j in range(s_t.shape[0]):
                s_t[j][i+2] += rd.gauss(0, params["sigma_velocity"])

    # correct the x_c and y_c that is out of the frame boarder
    s_t[:, 0] = np.maximum(s_t[:, 0], 0)
    s_t[:, 0] = np.minimum(s_t[:, 0], height)
    s_t[:, 1] = np.maximum(s_t[:, 1], 0)
    s_t[:, 1] = np.minimum(s_t[:, 1], width)
    return s_t