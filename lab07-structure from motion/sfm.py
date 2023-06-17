import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np

from utils.vis import Plot3DPoints
from utils.sfm.corrs import Find2D3DCorrespondences, GetPairMatches, UpdateReconstructionState
from utils.sfm.geometry import DecomposeEssentialMatrix, EstimateEssentialMatrix, TriangulatePoints, TriangulateImage, EstimateImagePose
from utils.sfm.image import Image
from utils.sfm.io import ReadFeatureMatches, ReadKMatrix
# from utils.sfm.vis import PlotImages, PlotWithKeypoints, PlotImagePairMatches
from utils.sfm.vis import PlotCameras

def main():
	np.set_printoptions(linewidth=10000, edgeitems=100, precision=3)
	data_folder = '../data'
	image_names = [
		'0000.png', '0001.png', '0002.png', '0003.png', '0004.png',
		'0005.png', '0006.png', '0007.png', '0008.png', '0009.png']
		# , 
	# Read images
	images = {}
	for im_name in image_names:
		images[im_name] = (Image(data_folder, im_name))
	# Read the matches
	matches = {}
	for image_pair in itertools.combinations(image_names, 2):
		matches[image_pair] = ReadFeatureMatches(image_pair, data_folder)
	# read intrinsic matrix
	K = ReadKMatrix(data_folder)

	init_images = [0, 1]
	# ------------------------------------------------------------------------------------
	# Visualize images and features
	# You can comment these lines once you verified that the images are loaded correctly

	# # Show the images
	# PlotImages(images)

	# # Show the keypoints
	# for image_name in image_names:
	# 	PlotWithKeypoints(images[image_name])

	# # Show the feature matches
	# for image_pair in itertools.combinations(image_names, 2):
	# 	PlotImagePairMatches(images[image_pair[0]], images[image_pair[1]], matches[(image_pair[0], image_pair[1])])
	# 	gc.collect()
	# ------------------------------------------------------------------------------------

	e_im1_name = image_names[init_images[0]]
	e_im2_name = image_names[init_images[1]]
	e_im1:Image = images[e_im1_name]
	e_im2:Image = images[e_im2_name]
	# get matching points of im1 and im2
	e_matches = GetPairMatches(e_im1_name, e_im2_name, matches)

	# Estimate Fundamental matrix
	E = EstimateEssentialMatrix(K, e_im1, e_im2, e_matches)
	# Extract the relative pose from the essential matrix. (possible solutions that need to check)
	possible_relative_poses = DecomposeEssentialMatrix(E)

	# ------------------Finding the correct decomposition--------------------------------------
	# For each possible relative pose, try to triangulate points with function TriangulatePoints.
	# We can assume that the correct solution is the one that gives the most points in front of both cameras.
	max_points = 0
	best_pose = -1
	# Be careful not to set the transformation in the wrong direction
	# Note that this pose is assumed to be the transformation from global space to image space
	e_im1.SetPose(np.eye(3), np.zeros(3))
	for R, t in possible_relative_poses:
		# from img1 to img2 R.T, -R.T @ t
		e_im2.SetPose(R, t)
		points3D, im1_corrs, im2_corrs = TriangulatePoints(K, e_im1, e_im2, e_matches)
		if(points3D.shape[0] > max_points):
			max_points = points3D.shape[0]
			best_pose = (R, t)

	# Set the image poses - transformation from global space to image space
	R, t = best_pose
	e_im2.SetPose(R, t)
	# Triangulate initial points
	points3D, im1_corrs, im2_corrs = TriangulatePoints(K, e_im1, e_im2, e_matches)
	# Add the new 2D-3D correspondences to the images
	e_im1.Add3DCorrs(im1_corrs, list(range(points3D.shape[0])))
	e_im2.Add3DCorrs(im2_corrs, list(range(points3D.shape[0])))

	# Keep track of all registered images
	registered_images = [e_im1_name, e_im2_name]
	for reg_im in registered_images:
		print(f'Image {reg_im} sees {images[reg_im].NumObserved()} 3D points')

	# ------------------Map extension--------------------------------------
	# Register new images + triangulate
	# Run until we can register all images
	print("========= Start Map extension =========")#
	while len(registered_images) < len(images):
		for image_name in images:
			if image_name in registered_images:
				continue
			# Find 2D-3D correspondences accorfing to matches 
			# (only return 3D points triangulated by previous image pairs)
			image_kp_idxs, point3D_idxs = Find2D3DCorrespondences(image_name, images, matches, registered_images)

			# With two few correspondences the pose estimation becomes shaky.
			if len(image_kp_idxs) < 50:
				continue# Keep this image for later
			print(f'Register image {image_name} from {len(image_kp_idxs)} correspondences')

			# Estimate new image pose
			R, t = EstimateImagePose(images[image_name].kps[image_kp_idxs], points3D[point3D_idxs], K)

			# Set the estimated image pose in the image and add the correspondences between keypoints and 3D points
			images[image_name].SetPose(R, t)
			images[image_name].Add3DCorrs(image_kp_idxs, point3D_idxs)

			# Triangulate new points wth all previously registered images
			new_points3D, corrs = TriangulateImage(K, image_name, registered_images, matches, images)

			# Update the 3D points and image correspondences
			points3D, images = UpdateReconstructionState(corrs, new_points3D, points3D, images)
			print(f"Current number of 3D Points: {points3D.shape[0]}")

			registered_images.append(image_name)

	# Visualize
	fig = plt.figure()
	ax3d = fig.add_subplot(111, projection='3d')
	Plot3DPoints(points3D, ax3d)
	PlotCameras(images, registered_images, ax3d)

	# Delay termination of the program until the figures are closed
	# Otherwise all figure windows will be killed with the program
	plt.show(block=True)


if __name__ == '__main__':
	main()