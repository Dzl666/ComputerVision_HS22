import numpy as np
import math

from chi2_cost import chi2_cost
from color_histogram import color_histogram

def observe(particles, img, bbox_height, bbox_width, hist_bin, hist, sigma):
    """
    :bbox_height, bbox_width - fixed bounding box size
    :hist - target histogram"""
    height, width, ch = img.shape
    num_p = particles.shape[0]
    chi2_dist = np.zeros((num_p, 1))
    weights = np.zeros_like(chi2_dist)
 
    for i in range(num_p):
        x1 = round(particles[i,0] - bbox_height*0.5)
        y1 = round(particles[i,1] - bbox_width*0.5)
        hist_x = color_histogram(
            min(max(0, x1), height-1), min(max(0, y1), width-1),
            min(max(0, x1+bbox_height), height-1), min(max(0, y1+bbox_width), width-1),
            img, hist_bin)
        chi2_dist[i] = chi2_cost(hist_x, hist) **2
    # Gaussian weight related to chi2 distance
    weights = 1 / (np.sqrt(2*math.pi)*sigma) * np.exp(chi2_dist / (-2* sigma**2))
    return weights / np.sum(weights)