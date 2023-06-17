import numpy as np
import scipy
import cv2

# Harris corner detector
def extract_harris(img, sigma = 1.0, k = 0.05, thresh = 1e-5):
    '''
    Inputs:
    - img:      (h, w) gray-scaled image
    - sigma:    smoothing Gaussian sigma. suggested values: 0.5, 1.0, 2.0
    - k:        Harris response function constant. suggest interval: (0.04 - 0.06)
    - thresh:   scalar value to threshold corner strength. suggested interval: (1e-6 - 1e-4)
    Returns:
    - corners:  (q, 2) numpy array storing the keypoint positions [x, y]
    - C:     (h, w) numpy array storing the corner strength
    '''
    # Convert to float
    img = img.astype(float) / 255.0
    h, w = img.shape[0], img.shape[1]
    # Compute image gradients
    # TODO: implement the computation of the image gradients Ix and Iy here.
    grad_kernel_y = 0.5 * np.array([[-1], [0], [1]])
    grad_kernel_x = 0.5 * np.array([[-1, 0, 1]])
    Ix = scipy.signal.convolve2d(img, grad_kernel_x, mode='same')
    Iy = scipy.signal.convolve2d(img, grad_kernel_y, mode='same')

    # Compute local auto-correlation matrix
    # TODO: compute the auto-correlation matrix here
    IxIx_blur = cv2.GaussianBlur(Ix * Ix, (3,3), sigma, borderType=cv2.BORDER_REPLICATE)
    IxIy_blur = cv2.GaussianBlur(Ix * Iy, (3,3), sigma, borderType=cv2.BORDER_REPLICATE)
    IyIy_blur = cv2.GaussianBlur(Iy * Iy, (3,3), sigma, borderType=cv2.BORDER_REPLICATE)

    # Compute Harris response function
    # TODO: compute the Harris response function C here
    C = np.zeros((h, w))
    C = (IxIx_blur* IyIy_blur - IxIy_blur* IxIy_blur) - k* (IxIx_blur+IyIy_blur) **2
    
    # Detection with threshold
    # TODO: detection and find the corners here
    C_nms = scipy.ndimage.maximum_filter(C, size=3)
    corners_idx = np.where((C_nms == C)&(C > thresh))

    # corners: (q, 2) numpy array
    corners = np.zeros((len(corners_idx[0]), 2))
    corners[:, 0] = corners_idx[1].astype(int)
    corners[:, 1] = corners_idx[0].astype(int)
    cond_edge = (corners[:,0]==0) | (corners[:,0]==w-1) | (corners[:,1]==0) | (corners[:,1]==h-1)
    corners = np.delete(corners, np.where(cond_edge)[0], axis=0)

    return corners, C

