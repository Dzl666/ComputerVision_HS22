import numpy as np

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:    - (q1, feature_dim) descriptor for the first image
    - desc2:    - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:- (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    q1, q2 = desc1.shape[0], desc2.shape[0]
    distances = np.zeros((q1, q2))
    for i in range(q1):
        for j in range(q2):
            distances[i][j] = np.sum((desc1[i,:] - desc2[j,:]) ** 2)
    return distances

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        matches = np.zeros((q1, 2))
        matches[:, 0] = np.array([i for i in range(q1)])
        matches[:, 1] = np.argmin(distances, axis=1)
    elif method == "mutual":
        one_way_idx = np.argmin(distances, axis=1) # each point in img1 to any in img2
        one_way_idx_inv = np.argmin(distances, axis=0)
        matches_mutual = np.zeros((min(q1,q2), 2))
        mutual_cnt = 0
        for i in range(q2):
            inv_idx = one_way_idx_inv[i]
            if one_way_idx[inv_idx] == i:
                matches_mutual[mutual_cnt, :] = np.array([inv_idx, i])
                mutual_cnt += 1
        matches = matches_mutual[0:mutual_cnt-1, :]
    elif method == "ratio":
        one_way_idx = np.argmin(distances, axis=1)
        one_way_val_first = np.min(distances, axis=1)
        one_way_val_second = np.partition(distances, kth=1, axis=1)[:, 1]
        one_way_ratio = one_way_val_first / one_way_val_second
        ratio_idx = (np.where(one_way_ratio < ratio_thresh))[0].astype(int)
        matches = np.zeros((len(ratio_idx), 2))
        matches[:, 0] = ratio_idx
        matches[:, 1] = one_way_idx[ratio_idx]
    else:
        raise NotImplementedError
    return matches.astype(int)

