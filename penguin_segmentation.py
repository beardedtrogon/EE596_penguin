#!/usr/local/bin/python3.7

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from optparse import OptionParser
from sklearn.cluster import KMeans
from random import randrange
from scipy import ndimage
import scipy
import random
import sys

import skimage
from skimage.filters.rank import entropy
from skimage.feature import local_binary_pattern
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.color import label2rgb
from skimage.feature import greycomatrix, greycoprops

from PIL import Image

import pickle

K = 4
max_iterations = 100
attempts=10
eps = 1.0

NUM_COLOR_CLUSTER_REGIONS = 6
NUM_ENTROPY_TEXTURE_REGIONS = 5
NUM_COLOR_RAG_ATRRIBUTES = 8
NUM_ENTROPY_RAG_ATTRIBUTES = 2

RAG_COLOR = [[ [0 for col in range(NUM_COLOR_RAG_ATRRIBUTES)] for col in range(NUM_COLOR_CLUSTER_REGIONS)] for row in range(NUM_COLOR_CLUSTER_REGIONS)]
#RAG_ENTROPY = [[ [0 for col in range(NUM_ENTROPY_RAG_ATTRIBUTES)] for col in range(NUM_ENTROPY_TEXTURE_REGIONS)] for row in range(NUM_ENTROPY_TEXTURE_REGIONS)]
HIST_COLOR_REGIONS = [0 for col in range(NUM_COLOR_CLUSTER_REGIONS)]
HIST_IMAGE = None

BELOW_ADJACENCY = 0
ABOVE_ADJACENCY = 1
LEFT_ADJACENCY  = 2
RIGHT_ADJACENCY = 3

# Create mask of region based on labels
def create_mask(img, labels, num_label, num_rows, num_columns):
	for i in range(num_rows):
		for j in range(num_columns):
			if not labels[i][j] == num_label:
				img[i][j] = [0,0,0]
			else:
				img[i][j] = [255,255,255]

# Create histogram of image or region with a mask
def calc_hist(img, mask):
	# Calculate histogram of entire image
	hist = cv2.calcHist([img], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()
	return hist

# Grey Co-occurence Matrix - calculates dissimilarity and correlation
def texture_CC_region(img, labels, num_label, num_rows, num_columns):
	img_copy = img.copy()

	for i in range(num_rows):
		for j in range(num_columns):
			if not labels[i][j] == num_label:
				img_copy[i][j] = [0,0,0]

	gray = cv2.cvtColor(img_copy.astype('uint8'),cv2.COLOR_BGR2GRAY)

	glcm = greycomatrix(gray, [5], [0],levels=256, symmetric=True, normed=True)
	dissimilar = greycoprops(glcm, 'dissimilarity')[0, 0]
	correlation = greycoprops(glcm, 'correlation')[0, 0]

	return (dissimilar, correlation)

# Calculate percentage of pixels that fulfill relationship wrt centroid of other region
def percentage_pixels(region_centroid, labels, num_label, num_rows, num_columns):
	below_counter = 0
	above_counter = 0
	left_counter = 0
	right_counter = 0
	below_per = 0
	above_per = 0
	left_per = 0
	right_per = 0
	counter = 0

	for r in range(num_rows):
		for c in range(num_columns):
			if labels[r][c] == num_label:
				counter += 1
				if region_centroid[1] > r:
					below_counter += 1
				if region_centroid[1] < r:
					above_counter += 1
				if region_centroid[0] > c:
					left_counter += 1
				if region_centroid[0] < c:
					right_counter += 1

	below_per = below_counter/counter
	above_per = above_counter/counter
	left_per = left_counter/counter
	right_per = right_counter/counter

	return (below_per, above_per, left_per, right_per)

# Calculate the largest CC regions of an image based on size and position in the image
def largest_CC_regions(num_regions, num_needed_regions, stats, max_area, width, height, limits):

	max_values = [0] * num_needed_regions
	indices = [0] * num_needed_regions

	for i in range(num_regions):

		if stats[i][0] < (width/limits):
			continue
		if stats[i][1] == 0:
			continue
		if (stats[i][0] + stats[i][2]) > (width *(limits-1)/limits):
			continue
		if (stats[i][1] + stats[i][3]) == height:
			continue

		if min(max_values) <= stats[i][4] and stats[i][4] < max_area:
			minpos = max_values.index(min(max_values))
			max_values[minpos] = stats[i][4]
			indices[minpos] = i

	return indices

# Calculate the size
def size_CC_region(labels, num_label):
	return sum((x == num_label).sum() for x in labels)

def mean_color_CC_region(labels, num_label, original_image):
	mean_r = 0
	mean_g = 0
	mean_b = 0
	num_pixels = 0

	current_CC = labels == num_label
	index_of_CC = np.where(current_CC == True)

	match_rows = index_of_CC[0]
	match_columns = index_of_CC[1]
	num_pixels = len(match_rows)

	for i in range(num_pixels):
		mean_r += original_image[match_rows[i]][match_columns[i]][0]
		mean_g += original_image[match_rows[i]][match_columns[i]][1]
		mean_b += original_image[match_rows[i]][match_columns[i]][2]

	mean_r = (mean_r / num_pixels)
	mean_g = (mean_g / num_pixels)
	mean_b = (mean_b / num_pixels)

	return (mean_r, mean_g, mean_b)

#def create_texture_cluster_RAG(image, centroids, labels, top_labels, num_rows, num_columns):

def create_color_cluster_RAG(image, centroids, labels, top_labels, num_rows, num_columns):
	global RAG_COLOR
	global HIST_COLOR_REGIONS
	size = []
	mean_color = []
	texture = []
	mask = None

	# calculate attributes of regions
	for num in range(len(top_labels)):
		size.append(size_CC_region(labels, top_labels[num]))
		mean_color.append(mean_color_CC_region(labels, top_labels[num], image))
		texture.append(texture_CC_region(image, labels, top_labels[num], num_rows, num_columns))
		mask = create_mask(image, labels, top_labels[num], num_rows, num_columns)
		HIST_COLOR_REGIONS[num] = calc_hist(image, mask)

	# calculate RAG weights between regions
	for region in range(len(size)):
		for region_to_compare in range(len(size)):

			if region_to_compare == region:
				print(region_to_compare)
				print(region)
				RAG_COLOR[region][region_to_compare][0] = 0
				RAG_COLOR[region][region_to_compare][1] = 0
				RAG_COLOR[region][region_to_compare][2] = 0
				RAG_COLOR[region][region_to_compare][3] = 0
				RAG_COLOR[region][region_to_compare][4] = 0
				RAG_COLOR[region][region_to_compare][5]	= 0
				continue

			# RAG Color attribute #1 - Difference in Size
			RAG_COLOR[region][region_to_compare][0] = size[region] - size[region_to_compare]

			# RAG Color attribute #2 - Difference in Mean Color
			RAG_COLOR[region][region_to_compare][1] = tuple(np.subtract(mean_color[region], mean_color[region_to_compare]))

			# RAG Color attribute #3 - Difference in Gray Co-Matrix (Dissimilar)
			RAG_COLOR[region][region_to_compare][2] = texture[region][0] - texture[region_to_compare][0]

			# RAG Color attribute #4 - Difference in Gray Co-Matrix (Correlation)
			RAG_COLOR[region][region_to_compare][3] = texture[region][1] - texture[region_to_compare][1]

			below, above, left, right = percentage_pixels(centroids[region], labels, top_labels[region_to_compare], num_rows, num_columns)

			# RAG Color attribute #5 - Percentage of Pixels above Centroid of Region
			RAG_COLOR[region][region_to_compare][4] = below

			# RAG Color attribute #6 - Percentage of Pixels below Centroid of Region
			RAG_COLOR[region][region_to_compare][5] = above

			# RAG Color attribute #7 - Percentage of Pixels left of Centroid of Region
			RAG_COLOR[region][region_to_compare][6] = left

			# RAG Color attribute #8 - Percentage of Pixels right of Centroid of Region
			RAG_COLOR[region][region_to_compare][7] = right

def segment_image(K, attemps, max_iterations, eps, image_path):
	global RAG_COLOR
	global HIST_COLOR_REGIONS
	global HIST_IMAGE

	# Datastructures already calculated and saved?
	color_ds_string = os.path.splitext(image_path)[0] + "_color.pkl"
	hist_regions_ds_string = os.path.splitext(image_path)[0] + "_hist_regions.pkl"
	hist_ds_string = os.path.splitext(image_path)[0] + "_hist.pkl"

	# if os.path.exists(color_ds_string) and os.path.exists(hist_regions_ds_string) and os.path.exists(hist_ds_string):
	# 	print("Data structures already created")
	# 	return

	original_image = cv2.imread(image_path)
	img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
	img_copy = img.copy()
	img_copy_2 = img.copy()

	# save histogram of image
	HIST_IMAGE = calc_hist(img, None)

	r,c,d = img.shape

	# Calculate entropy of image
	binary_texture_image = cv2.cvtColor(img.astype('uint8'),cv2.COLOR_BGR2GRAY)
	textured_image = ndimage.gaussian_filter(binary_texture_image, 1.0)
	textured_image = entropy(binary_texture_image, disk(5))

	#threshold entropy filter results
	for i in range(r):
		for j in range(c):
			if textured_image[i][j] < 5:
				textured_image[i][j] = int(0)
			else:
				textured_image[i][j] = int(textured_image[i][j])

	# Perform Gaussian filter on entropy (texture) map
	textured_image = textured_image.astype(np.uint8)
	textured_image = ndimage.gaussian_filter(textured_image, 4.0)
	textured_image = cv2.cvtColor(textured_image,cv2.COLOR_GRAY2RGB)

	text_r, text_c, _ = textured_image.shape

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, eps)

	# Reshape original image for kmeans
	reshaped_img = img.reshape((-1,3))
	reshaped_img = np.float32(reshaped_img)

	# Perform k-means on normal image
	ret,label,center=cv2.kmeans(reshaped_img,K,None,criteria, attempts ,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))

	# Use Gaussian to close gaps
	blur = cv2.GaussianBlur(res2,(15,15),0)

	# Repeate K-means!
	reshaped_img = blur.reshape((-1,3))
	reshaped_img = np.float32(reshaped_img)

	# Perform k-means on normal image
	ret,label,center=cv2.kmeans(reshaped_img,K,None,criteria, attempts ,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center = np.uint8(center)
	res = center[label.flatten()]
	res2 = res.reshape((img.shape))

	# Perform Laplacian of Gaussian Edge Detection and convert back to grayscale
	laplacian = cv2.Laplacian(res2,cv2.CV_64F)
	gray=cv2.cvtColor(laplacian.astype('uint8'),cv2.COLOR_BGR2GRAY)

	for i in range(r):
		for j in range(c):

			# Invert image to have non-edges be non-0
			if not gray[i][j] == 0:
				gray[i][j] = 0
			else:
				gray[i][j] = 128

	connectivity = 4

	# Perform connected components on both texture and original image
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity, cv2.CV_64F)

	# Find top connected component regions in both images
	top_labels = largest_CC_regions(num_labels, NUM_COLOR_CLUSTER_REGIONS, stats, (r*c)/3, c, r, 8)

# For Display Purposes only
##############################################################3
	for num in range(len(top_labels)):
		current_CC = labels == top_labels[num]
		index_of_CC = np.where(current_CC == True)

		match_rows = index_of_CC[0]
		match_columns = index_of_CC[1]
		num_matches = len(match_rows)
		rand_r = random.randint(0,255)
		rand_g = random.randint(0,255)
		rand_b = random.randint(0,255)

		for i in range(num_matches):
			img_copy[match_rows[i]][match_columns[i]] = [rand_r, rand_g, rand_b]

	fig = plt.figure()
	ax1 = fig.add_subplot(2,2,1)
	ax1.imshow(original_image)
	ax2 = fig.add_subplot(2,2,2)
	ax2.imshow(res2)
	#ax3 = fig.add_subplot(2,2,3)
	#plt.imshow(mask, cmap='gray')
	ax4 = fig.add_subplot(2,2,4)
	ax4.imshow(img_copy)
	plt.show()
##############################################################

	# if not os.path.exists(color_ds_string):
	# 	create_color_cluster_RAG(original_image, centroids, labels, top_labels, r, c)
	#
	# color_file = open(color_ds_string, 'wb')
	# hist_regions_file = open(hist_regions_ds_string, 'wb')
	# hist_file = open(hist_ds_string, 'wb')
	# pickle.dump(RAG_COLOR, color_file)
	# pickle.dump(HIST_COLOR_REGIONS, hist_regions_file)
	# pickle.dump(HIST_IMAGE, hist_file)
	# color_file.close()
	# hist_regions_file.close()
	# hist_file.close()

if __name__== "__main__":
	image_path = ""

	parser = OptionParser(usage="usage: %prog folder",
                          version="%prog 1.0")

	parser.add_option("-k", type="int", dest="num")
	parser.add_option("-a", type="int", dest="attempts")
	parser.add_option("-i", type="int", dest="max_iterations")
	parser.add_option("-e", type="float", dest="eps")

	(options, args) = parser.parse_args()

	if len(args) != 1:
		parser.error("wrong number of arguments")
		sys.exit(0)

	if options.num != None:
		K = options.num

	if options.attempts != None:
		attempts = options.attempts

	if options.max_iterations != None:
		max_iterations = options.max_iterations

	if options.eps != None:
		eps = options.eps

	if not os.path.isdir(args[0]):
		parser.error("Please provide folder of images")
		sys.exit(0)

	# First segment, create RAG, and create features
	for f in os.listdir(args[0]):
		filename = os.path.join(args[0], f)
		if os.path.isfile(filename):
			if ".jpg" in filename or ".jpeg" in filename or ".png" in filename:
				print(filename)
				segment_image(K, attempts, max_iterations, eps, filename)
