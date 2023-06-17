import numpy as np

from impl.sfm.image import Image
from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1:Image, im2:Image, matches):
	"""
	K: ndarray
	[2759.48, 0.00, 	1520.69
	 0.00, 	  2764.16, 	1006.81
	 0.00, 	  0.00, 	1.00]
	im1, im2: images
	matches: 1-d index of matching points between two images
	"""
	num_kpt = matches.shape[0]
	# Normalize coordinates (to points on the normalized image plane) x_hat = K^-1 * x
	kp1 = MakeHomogeneous(im1.kps[matches[:,0]], ax=1) # [n_kpt, 3] 
	kp2 = MakeHomogeneous(im2.kps[matches[:,1]], ax=1)
	norm_kps1 = (np.linalg.inv(K) @ kp1.T).T # [n_kpt, 3]
	norm_kps2 = (np.linalg.inv(K) @ kp2.T).T

	# Assemble constraint matrix of kp2' * E * kp1 = 0
	constraint_matrix = np.zeros((num_kpt, 9))
	for i in range(num_kpt):
		constraint_matrix[i, :] = np.hstack((
			norm_kps1[i,:] * norm_kps2[i][0],
			norm_kps1[i,:] * norm_kps2[i][1], 
			norm_kps1[i,:]
		))

	# Solve for the nullspace of the constraint matrix
	_, _, vh = np.linalg.svd(constraint_matrix)
	vec_E_hat = vh[-1,:]
	# Reshape the vectorized matrix to it's proper shape again
	E_hat = np.reshape(vec_E_hat, (3,3))
	# the internal constraints of E
	# The first two singular values need to be equal, the third one zero.
	U_E, _, Vh_E = np.linalg.svd(E_hat)
	S_E = np.eye(3, dtype=float)
	S_E[2][2] = 0.0
	E = U_E @ S_E @ Vh_E

	# check if the estimated matrix is correct for kp2' * E * kp1 = 0
	for i in range(num_kpt):
		kp1 = norm_kps1[i, :]
		kp2 = norm_kps2[i, :]
		test = abs(kp2 @ E @ kp1.T)
		assert(test < 0.01)
	return E


def DecomposeEssentialMatrix(E):
	u, _, vh = np.linalg.svd(E)
	# Determine the translation up to sign
	t_hat = u[:,-1]

	W = np.array([
		[0, -1, 0],
		[1, 0, 0],
		[0, 0, 1]
	])
	# Compute the two possible rotations
	R1 = u @ W @ vh
	R2 = u @ W.T @ vh
	# Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
	if np.linalg.det(R1) < 0:
		R1 *= -1
	if np.linalg.det(R2) < 0:
		R2 *= -1

	# Assemble the four possible solutions
	sols = [
		(R1, t_hat),
		(R2, t_hat),
		(R1, -t_hat),
		(R2, -t_hat)
	]
	return sols

def TriangulatePoints(K, im1:Image, im2:Image, matches):
	"""
	Given K, R1, R2, t1, t2, and kpts in each image, get correspond 3D points through triangulate

	Return
	------
	Points3D - size: [num_3D, 3]
	im1_corrs - size: [num_im1_corrs]
	im2_corrs - size: [num_im2_corrs]
	"""
	R1, t1 = im1.Pose()
	R2, t2 = im2.Pose()
	# [3, 4]
	M1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
	M2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

	# Ignore matches that already have a triangulated point
	new_matches = np.zeros((0, 2), dtype=int)
	# !! check for current matches, ignore the 3D points that have found previously
	for i in range(matches.shape[0]):
		p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])# match[i,0] - idx of kpt in img1
		p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
		if p3d_idx1 == -1 and p3d_idx2 == -1:
			new_matches = np.append(new_matches, matches[[i]], 0)

	num_new_matches = new_matches.shape[0]

	points3D = np.zeros((num_new_matches, 3))

	for i in range(num_new_matches):
		# new_matches[i,0] idx of kpt in img1
		kp1 = im1.kps[new_matches[i, 0], :]
		kp2 = im2.kps[new_matches[i, 1], :]

		# H & Z Sec. 12.2
		A = np.array([
			kp1[0] * M1[2] - M1[0],
			kp1[1] * M1[2] - M1[1],
			kp2[0] * M2[2] - M2[0],
			kp2[1] * M2[2] - M2[1]
		])
		_, _, vh = np.linalg.svd(A)
		# homogeneous_point = vh[-1] # [4]
		points3D[i] = HNormalize(vh[-1], ax=0) #[3]

	im1_corrs = new_matches[:,0] # [num_corrs]
	im2_corrs = new_matches[:,1]

	# M1 = K @ np.append(R1.T, -R1.T @ np.expand_dims(t1, 1), 1)
	# M2 = K @ np.append(R2.T, -R2.T @ np.expand_dims(t2, 1), 1)
	# Filter points behind the cameras, points3D - [num_3D, 3]
	# transforming them into each camera space and checking the depth (Z)
	kpt_1 = M1 @ MakeHomogeneous(points3D, ax=1).T # [3, num_match]
	correct_kpt = (kpt_1[2, :] > 0)
	# Filter points behind the first camera
	im1_corrs = im1_corrs[correct_kpt]
	im2_corrs = im2_corrs[correct_kpt]
	points3D = points3D[correct_kpt]
	
	if points3D.shape[0] > 0:
		kpt_2 = M2 @ MakeHomogeneous(points3D, ax=1).T # [3, num_match]
		correct_kpt = (kpt_2[2, :] > 0)
		# Filter points behind the second camera
		im1_corrs = im1_corrs[correct_kpt]
		im2_corrs = im2_corrs[correct_kpt]
		points3D = points3D[correct_kpt, :]

	return points3D, im1_corrs, im2_corrs


def EstimateImagePose(points2D, points3D, K):  
	"""Estimate pose of new image using kpt from the image and correspond 3d points in tracking
	Use points in the normalized image plane, removes the 'K'

	points2D: kpts from the newly register image [num_kpt, 2]
	points3D: common 3D points found so far
	K: K

	Return
	------------
	R, t
	"""
	kpt = MakeHomogeneous(points2D, ax=1) # [num_kpt, 3]
	normalized_points2D = (np.linalg.inv(K) @ kpt.T).T

	constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)
	# We don't use optimization here since we would need to make sure to 
	# only optimize on the se(3) manifold (the manifold of proper 3D poses).
	# Just DLT should give good enough results for this dataset.

	# Solve for the nullspace
	_, _, vh = np.linalg.svd(constraint_matrix)
	P_vec = vh[-1,:]
	P = np.reshape(P_vec, (3, 4), order='C')

	# Make sure we have a proper rotation
	u, _, vh = np.linalg.svd(P[:,:3])
	R = u @ vh
	if np.linalg.det(R) < 0:
		R *= -1

	_, _, vh = np.linalg.svd(P)
	C = np.copy(vh[-1,:])
	t = -R @ (C[:3] / C[3])

	return R, t

def TriangulateImage(K, image_name, registered_images, matches, images):
	"""
	Loop over all registered images and triangulate new points with the new image.
	Make sure to keep track of all new 2D-3D correspondences, also for the registered images
	"""
	image_t:Image = images[image_name]
	new_points3D = np.zeros((0,3))
	corrs_new_img = np.zeros(0)
	corrs = {}
	for img_name_registered in registered_images:
		img_registered:Image = images[img_name_registered]

		e_matches = GetPairMatches(image_name, img_name_registered, matches)

		points3D_t, im1_corrs, im2_corrs = TriangulatePoints(K, image_t, img_registered, e_matches)
		# size: [num_3D, 3], [num_im1_corrs], [num_im2_corrs]
		corrs_new_img = np.hstack((corrs_new_img, im1_corrs))

		# refer to the `local` new point indices here
		# add the index offset before adding the correspondences to the images
		offset = new_points3D.shape[0]
		corrs[img_name_registered] = (im2_corrs, offset, offset + points3D_t.shape[0])
		new_points3D = np.vstack((new_points3D, points3D_t))
		# print(new_points3D.shape[0])
		
	corrs[image_name] = (corrs_new_img.astype(np.int32), 0, new_points3D.shape[0])

	return new_points3D, corrs
  
