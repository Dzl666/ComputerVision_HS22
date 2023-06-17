import numpy as np
import math
import cv2 as cv
import matplotlib.pyplot as plt

def color_histogram(x1, y1, x2, y2, img, hist_bin):
    """
    :img - h x w x ch
    """
    hist = np.zeros((hist_bin, hist_bin, hist_bin))
    img_color_bin = np.floor(img[x1:x2, y1:y2, :]/256.0 *hist_bin)
    for i in range(hist_bin):
        for j in range(hist_bin):
            for k in range(hist_bin):
                hist[i][j][k] = np.sum(img_color_bin[:, :] == [i,j,k])

    return hist.flatten() / np.sum(hist)