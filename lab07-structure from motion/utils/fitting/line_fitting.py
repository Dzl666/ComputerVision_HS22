import numpy as np
from matplotlib import pyplot as plt
import random

np.random.seed(0)
random.seed(0)

def least_square(x,y):
	"""return the least-squares solution
	"""
	# solve A x = b   model kx + b = [x 1] * [k b]' = y
	A = np.vstack((x, np.ones(len(x)))).T
	k, b = np.linalg.lstsq(A, y, rcond=None)[0]
	return k, b

def num_inlier(x,y,k,b,n_samples,thres_dist):
	"""compute the number of inliers and a mask that denotes the indices of inliers
	x, y: data
	k, b: scope and bias calculated by lstsq

	n_samples:
	thres_dist: threshold of inliers in terms of dist between a point and the lstsq line
	"""
	num = 0
	mask = np.zeros(x.shape, dtype=bool)
	inliers = [1 if(np.abs(k*x_i-y_i+b)/np.sqrt(1+k**2) < thres_dist) else 0 
				for x_i, y_i in zip(x, y)]
	num = sum(inliers)
	mask = (np.array(inliers) == 1)
	return num, mask

def ransac(x,y,iter,n_samples,thres_dist,num_subset):
	"""ransac
	1.randomly choose a small subset from the noisy point set ;
	2.compute the least-squares solution for this subset;
	3.compute the number of inliers, if the number exceeds the current best result, update the estimation
	"""
	k_ransac = None
	b_ransac = None
	inlier_mask = None
	best_inliers = 0
	for _ in range(iter):
		subset_idx = random.choices(range(n_samples), k=num_subset)
		x_sub = x[subset_idx]
		y_sub = y[subset_idx]
		# compute the least-squares solution for this subset
		k, b = least_square(x_sub, y_sub)
		num_t, mask = num_inlier(x, y, k, b, n_samples, thres_dist)
		# record the best set
		if(num_t > best_inliers):
			best_inliers = num_t
			inlier_mask = mask
			k_ransac = k
			b_ransac = b

	return k_ransac, b_ransac, inlier_mask

def main():
	iter = 300
	thres_dist = 1
	n_samples = 500
	n_outliers = 50
	# origin model
	k_gt = 1
	b_gt = 10
	num_subset = 5
	x_gt = np.linspace(-10,10,n_samples)
	y_gt = k_gt*x_gt+b_gt
	# add noise
	x_noisy = x_gt+np.random.random(x_gt.shape)-0.5
	y_noisy = y_gt+np.random.random(y_gt.shape)-0.5
	# add outlier
	x_noisy[:n_outliers] = 8 + 10 * (np.random.random(n_outliers)-0.5)
	y_noisy[:n_outliers] = 1 + 2 * (np.random.random(n_outliers)-0.5)

	# least square
	k_ls, b_ls = least_square(x_noisy, y_noisy)

	# ransac
	k_ransac, b_ransac, inlier_mask = ransac(x_noisy, y_noisy, iter, n_samples, thres_dist, num_subset)
	outlier_mask = np.logical_not(inlier_mask)

	print("Estimated coefficients (true, linear regression, RANSAC):")
	print(k_gt, b_gt)
	print(k_ls, b_ls)
	print(k_ransac, b_ransac)

	line_x = np.arange(x_noisy.min(), x_noisy.max())
	line_y_ls = k_ls*line_x+b_ls
	line_y_ransac = k_ransac*line_x+b_ransac

	plt.scatter(
	    x_noisy[inlier_mask], y_noisy[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
	)
	plt.scatter(
	    x_noisy[outlier_mask], y_noisy[outlier_mask], color="gold", marker=".", label="Outliers"
	)
	plt.plot(line_x, line_y_ls, color="navy", linewidth=2, label="Linear regressor")
	plt.plot(
	    line_x,
	    line_y_ransac,
	    color="cornflowerblue",
	    linewidth=2,
	    label="RANSAC regressor",
	)
	plt.legend()
	plt.xlabel("Input")
	plt.ylabel("Response")
	plt.show()

if __name__ == '__main__':
	main()