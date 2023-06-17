import numpy as np

def filter_keypoints(img, keypoints, patch_size = 9):
    # TODO: Filter out keypoints that are too close to the edges
    # keypoints: (q, 2)
    h, w = img.shape[0], img.shape[1]
    half_size = np.floor(patch_size / 2.0)
    filter_keypoints = keypoints
    for i in range(keypoints.shape[0]-1, -1, -1):
        point = keypoints[i]
        if point[0]<half_size or point[0]>=w-half_size or point[1]<half_size or point[1]>=h-half_size:
            filter_keypoints = np.delete(filter_keypoints, i, axis=0)

    return filter_keypoints.astype(int)

# The implementation of the patch extraction is already provided here
def extract_patches(img, keypoints, patch_size = 9):
    '''
    Extract local patches for each keypoint
    Inputs:
    - img:          (h, w) gray-scaled images
    - keypoints:    (q, 2) numpy array of keypoint locations [x, y]
    - patch_size:   size of each patch (with each keypoint as its center)
    Returns:
    - desc:         (q, patch_size * patch_size) numpy array. patch descriptors for each keypoint
    '''
    img = img.astype(float) / 255.0
    h, w = img.shape[0], img.shape[1]
    offset = int(np.floor(patch_size / 2.0))
    ranges = np.arange(-offset, offset + 1)
    desc = np.take(img, ranges[:,None] * w + ranges + (keypoints[:, 1] * w + keypoints[:, 0])[:, None, None]) # (q, patch_size, patch_size)
    desc = desc.reshape(keypoints.shape[0], -1) # (q, patch_size * patch_size)
    return desc

