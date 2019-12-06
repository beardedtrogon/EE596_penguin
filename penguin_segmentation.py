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
NUM_RAG_ATRRIBUTES = 4
NUM_REGION_ATTRIBUTES = 12

RAG = [[ [0 for col in range(NUM_RAG_ATRRIBUTES)] for col in range(NUM_COLOR_CLUSTER_REGIONS)] for row in range(NUM_COLOR_CLUSTER_REGIONS)]
REGION_ATTRIBUTES = [ [0 for col in range(NUM_REGION_ATTRIBUTES)] for col in range(NUM_COLOR_CLUSTER_REGIONS)]

# Create mask of region based on labels
def create_mask(img, labels, num_label, num_rows, num_columns):
	mask = np.zeros(img.shape[:2], np.uint8)
	for i in range(num_rows):
		for j in range(num_columns):
			if not labels[i][j] == num_label:
				mask[i][j] = 0
			else:
				mask[i][j] = 255

	return mask

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
	contrast = greycoprops(glcm, 'contrast')[0, 0]
	homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
	energy = greycoprops(glcm, 'energy')[0, 0]

	return (dissimilar, correlation, contrast, homogeneity, energy)

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

	indices = []
	sizes = [row[4] for row in stats]
	counter = 0

	sorted_sizes = sorted(((v, i) for i, v in enumerate(sizes)), reverse=True)

	for i, (value, index) in enumerate(sorted_sizes):
		if stats[index][0] < (width/limits):
			continue
		if stats[index][1] == 0:
			continue
		if (stats[index][0] + stats[index][2]) > (width *(limits-1)/limits):
			continue
		if (stats[index][1] + stats[index][3]) == height:
			continue
		if value < max_area:
			counter += 1
			indices.append(index)
		if counter == num_needed_regions:
			break

	return indices

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

def create_color_cluster_RAG(image, centroids, stats, labels, top_labels, num_rows, num_columns):
	global RAG
	global REGION_ATTRIBUTES

	# calculate attributes of regions
	for num in range(len(top_labels)):
		size = stats[top_labels[num]][4]
		mean_color = mean_color_CC_region(labels, top_labels[num], image)
		dissimilar, correlation, contrast, homogeneity, energy = texture_CC_region(image, labels, top_labels[num], num_rows, num_columns)
		start_x = stats[top_labels[num]][0]
		start_y = stats[top_labels[num]][1]
		width = stats[top_labels[num]][2]
		height = stats[top_labels[num]][3]

		# Size of region
		REGION_ATTRIBUTES[num][0] = size
		# Mean color of region
		REGION_ATTRIBUTES[num][1] = mean_color
		# Texture - Dissimilarity
		REGION_ATTRIBUTES[num][2] = dissimilar
		# Texture - Correlation
		REGION_ATTRIBUTES[num][3] = correlation
		# Texture - Contrast
		REGION_ATTRIBUTES[num][4] = contrast
		# Texture - Homogeneity
		REGION_ATTRIBUTES[num][5] = homogeneity
		# Texture - Energy
		REGION_ATTRIBUTES[num][6] = energy
		# Centroid of region
		REGION_ATTRIBUTES[num][7] = (centroids[num][0],centroids[num][1])
		# Bounding Box
		REGION_ATTRIBUTES[num][8] = [(start_y, start_x), (start_y, start_x + width), (start_y + height, start_x), (start_y + height, start_x + width)]

		# Create mask of region
		mask = create_mask(image, labels, top_labels[num], num_rows, num_columns)

		# Blue Histogram of region
		REGION_ATTRIBUTES[num][9] = cv2.calcHist([image],[0],mask,[64],[0,256])
		# Green Histogram of region
		REGION_ATTRIBUTES[num][10] = cv2.calcHist([image],[1],mask,[64],[0,256])
		# Red Histogram of region
		REGION_ATTRIBUTES[num][11] = cv2.calcHist([image],[2],mask,[64],[0,256])

		contours,hierarchy = cv2.findContours(mask,2,1)
		cnt1 = contours[0]
		print(cnt1)
		sys.exit(0)

		# print(REGION_ATTRIBUTES[num][9])
		# plt.plot(REGION_ATTRIBUTES[num][9], color='b');
		# plt.plot(REGION_ATTRIBUTES[num][10],color = 'g')
		# plt.plot(REGION_ATTRIBUTES[num][11],color = 'r')
		# plt.show()

	# calculate RAG weights between regions
	for region in range(len(size)):
		for region_to_compare in range(len(size)):

			if region_to_compare == region:
				RAG[region][region_to_compare][0] = 0
				RAG[region][region_to_compare][1] = 0
				RAG[region][region_to_compare][2] = 0
				RAG[region][region_to_compare][3] = 0
				continue

			below, above, left, right = percentage_pixels(centroids[region], labels, top_labels[region_to_compare], num_rows, num_columns)

			# RAG Color attribute #5 - Percentage of Pixels above Centroid of Region
			if below > .5:
				RAG[region][region_to_compare][0] = True
			else:
				RAG[region][region_to_compare][0] = False

			# RAG Color attribute #6 - Percentage of Pixels below Centroid of Region
			if above > .5:
				RAG[region][region_to_compare][1] = True
			else:
				RAG[region][region_to_compare][1] = False

			# RAG Color attribute #7 - Percentage of Pixels left of Centroid of Region
			if left > .5:
				RAG[region][region_to_compare][2] = True
			else:
				RAG[region][region_to_compare][2] = False

			# RAG Color attribute #8 - Percentage of Pixels right of Centroid of Region
			if right > .5:
				RAG[region][region_to_compare][3] = True
			else:
				RAG[region][region_to_compare][3] = False

def segment_image(K, attemps, max_iterations, eps, image_path):
	global RAG
	global REGION_ATTRIBUTES
	global HIST_IMAGE

	# Datastructures already calculated and saved?
	color_ds_string = os.path.splitext(image_path)[0] + "_color.pkl"
	hist_regions_ds_string = os.path.splitext(image_path)[0] + "NUM_REGION_ATTRIBUTES.pkl"

	# if os.path.exists(color_ds_string) and os.path.exists(hist_regions_ds_string) and os.path.exists(hist_ds_string):
	# 	print("Data structures already created")
	# 	return

	original_image = cv2.imread(image_path)
	img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
	img_copy = img.copy()
	img_copy_2 = img.copy()

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

	if not os.path.exists(color_ds_string):
		create_color_cluster_RAG(original_image, centroids, stats, labels, top_labels, r, c)
	#
	# color_file = open(color_ds_string, 'wb')
	# hist_regions_file = open(hist_regions_ds_string, 'wb')
	# hist_file = open(hist_ds_string, 'wb')
	# pickle.dump(RAG, color_file)
	# pickle.dump(REGION_ATTRIBUTES, hist_regions_file)
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
