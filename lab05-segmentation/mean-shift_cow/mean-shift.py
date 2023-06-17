import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def gaussian(dist, bandwidth):
    return (1/(bandwidth * math.sqrt(2*torch.pi))) * torch.exp(-0.5*dist / (bandwidth**2))

def distance(x, X):
    return torch.norm(X-x, dim=1)

def update_point(weight, X):
    weighted_X = torch.zeros(3)
    for i in range(3):
        weighted_X[i] = X[:,i].dot(weight) / torch.sum(weight)
    return weighted_X

def meanshift_step(X, bandwidth=0.8):
    X_ = X.clone() # torch.Size([3675, 3])
    for i, x in enumerate(X):
        dist = distance(x, X) # torch.Size([3675])
        weight = gaussian(dist, bandwidth) # torch.Size([3675])
        X_[i] = update_point(weight.cuda(), X) # torch.Size([3])
    return X_


def distance_batch(x, X):
    return torch.norm(X-x, dim=1)

def update_point_batch(weight, X):
    return X.T @ (weight/torch.sum(weight))

def meanshift_step_batch(X, bandwidth=0.8):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance_batch(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point_batch(weight.cuda(), X)
    return X_

def meanshift(X):
    X = X.clone()
    for iters in range(20):
        # print("Iterations: "+str(iters+1))
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X


scale = 0.25    # downscale the image to run faster
# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, channel_axis=2)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(torch.from_numpy(image_lab).cuda()).detach().cpu().numpy()
t = time.time() - t
print ('Elapsed time for mean-shift(vectorized): {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
# resize res-img to original resolution
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=2)
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
