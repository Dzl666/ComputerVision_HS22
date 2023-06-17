import numpy as np

def Find2D3DCorrespondences(image_name, images, matches, registered_images):
	"""Find (unique) 2D-3D correspondences from 2D-2D correspondences
	"""
	assert(image_name not in registered_images)

	image_kp_idxs = []
	p3D_idxs = []
	for other_image_name in registered_images:
		other_image = images[other_image_name]
		pair_matches = GetPairMatches(image_name, other_image_name, matches)

	for i in range(pair_matches.shape[0]):
		p3D_idx = other_image.GetPoint3DIdx(pair_matches[i,1])
		if p3D_idx > -1:
			p3D_idxs.append(p3D_idx)
			image_kp_idxs.append(pair_matches[i,0])

	# print(f'found {len(p3D_idxs)} points, {np.unique(np.array(p3D_idxs)).shape[0]} unique points')

	# Remove duplicated correspondences
	_, unique_idxs = np.unique(np.array(p3D_idxs), return_index=True)
	image_kp_idxs = np.array(image_kp_idxs)[unique_idxs].tolist()
	p3D_idxs = np.array(p3D_idxs)[unique_idxs].tolist()

	return image_kp_idxs, p3D_idxs

def GetPairMatches(im1, im2, matches):
	"""Make sure we get keypoint matches between the images in the order that we requested
	"""
	if im1 < im2:
		return matches[(im1, im2)]
	else:
		return np.flip(matches[(im2, im1)], 1)


# Update the reconstruction with the new information from a triangulated image
def UpdateReconstructionState(corrs:dict, new_points3D, points3D, images):
	"""
	Add the new points to the set of reconstruction points 
	add the correspondences to the images.
	Be careful to update the point indices to the global indices in the `points3D` array.
	"""
	origin_idx = points3D.shape[0]
	points3D = np.append(points3D, new_points3D, 0)

	# corrs: (im_corrs, new_points3D.shape[0], points3D_t.shape[0])
	for im_name in corrs.keys():
		corrs_kpt_idx = corrs[im_name][0]
		corrs_3Dpoint_start = origin_idx + corrs[im_name][1]
		corrs_3Dpoint_end = origin_idx + corrs[im_name][2]
		# print(f'{corrs_3Dpoint_start}, {corrs_3Dpoint_end}')
		images[im_name].Add3DCorrs(corrs_kpt_idx, list(range(corrs_3Dpoint_start, corrs_3Dpoint_end)))

	return points3D, images