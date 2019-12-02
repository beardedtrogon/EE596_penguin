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

NUM_COLOR_CLUSTER_REGIONS = 10
NUM_ENTROPY_TEXTURE_REGIONS = 5
NUM_COLOR_RAG_ATRRIBUTES = 8
NUM_ENTROPY_RAG_ATTRIBUTES = 2

RAG_COLOR = [[ [0 for col in range(NUM_COLOR_RAG_ATRRIBUTES)] for col in range(NUM_COLOR_CLUSTER_REGIONS)] for row in range(NUM_COLOR_CLUSTER_REGIONS)] 
RAG_ENTROPY = [[ [0 for col in range(NUM_ENTROPY_RAG_ATTRIBUTES)] for col in range(NUM_ENTROPY_TEXTURE_REGIONS)] for row in range(NUM_ENTROPY_TEXTURE_REGIONS)] 

BELOW_ADJACENCY = 0
ABOVE_ADJACENCY = 1
LEFT_ADJACENCY  = 2
RIGHT_ADJACENCY = 3

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

def percentage_pixels(relationship, region_centroid, labels, num_label, num_rows, num_columns):
	target_counter = 0
	counter = 0

	for r in range(num_rows):
		for c in range(num_columns):
			if labels[r][c] == num_label:
				counter += 1
				if relationship == BELOW_ADJACENCY:
					if region_centroid[1] > r:
						target_counter += 1
				elif relationship == ABOVE_ADJACENCY:
					if region_centroid[1] < r:
						target_counter += 1
				elif relationship == LEFT_ADJACENCY:
					if region_centroid[0] > c:
						target_counter += 1
				elif relationship == RIGHT_ADJACENCY:
					if region_centroid[0] < c:
						target_counter += 1
				else:
					print("Invalid value for relationship parameter")
					return 0

	return (target_counter/counter)

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
	size = []
	mean_color = []
	texture = []

	# calculate attributes of regions
	for num in range(len(top_labels)):
		size.append(size_CC_region(labels, top_labels[num]))
		mean_color.append(mean_color_CC_region(labels, top_labels[num], image))
		texture.append(texture_CC_region(image, labels, top_labels[num], num_rows, num_columns))

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

			# RAG Color attribute #3 - Difference in Gray Co-Matrix
			RAG_COLOR[region][region_to_compare][2] = texture[region][0] - texture[region_to_compare][0]

			# RAG Color attribute #4 - Difference in Gray Co-Matrix
			RAG_COLOR[region][region_to_compare][3] = texture[region][1] - texture[region_to_compare][1]

			# RAG Color attribute #5 - Percentage of Pixels above Centroid of Region
			RAG_COLOR[region][region_to_compare][4] = percentage_pixels(ABOVE_ADJACENCY, centroids[region], labels, top_labels[region_to_compare], num_rows, num_columns)

			# RAG Color attribute #6 - Percentage of Pixels below Centroid of Region
			RAG_COLOR[region][region_to_compare][5] = percentage_pixels(BELOW_ADJACENCY, centroids[region], labels, top_labels[region_to_compare], num_rows, num_columns)

			# RAG Color attribute #7 - Percentage of Pixels left of Centroid of Region
			RAG_COLOR[region][region_to_compare][6] = percentage_pixels(LEFT_ADJACENCY, centroids[region], labels, top_labels[region_to_compare], num_rows, num_columns)

			# RAG Color attribute #8 - Percentage of Pixels right of Centroid of Region
			RAG_COLOR[region][region_to_compare][7] = percentage_pixels(RIGHT_ADJACENCY, centroids[region], labels, top_labels[region_to_compare], num_rows, num_columns)

def segment_image():
	global K
	global attempts
	global max_iterations
	global eps
	global RAG_COLOR

	image_path = ""

	parser = OptionParser(usage="usage: %prog filename",
                          version="%prog 1.0")

	parser.add_option("-k", type="int", dest="num")
	parser.add_option("-a", type="int", dest="attempts")
	parser.add_option("-i", type="int", dest="max_iterations")
	parser.add_option("-e", type="float", dest="eps")

	(options, args) = parser.parse_args()

	if len(args) != 1:
		parser.error("wrong number of arguments")

	if options.num != None:
		K = options.num

	if options.attempts != None:
		attempts = options.attempts

	if options.max_iterations != None:
		max_iterations = options.max_iterations

	if options.eps != None:
		eps = options.eps

	if os.path.exists(args[0]):
		image_path = args[0]

	# Datastructures already calculated and saved?
	color_ds_string = os.path.splitext(image_path)[0] + "_color.pkl"
	texture_ds_string = os.path.splitext(image_path)[0] + "_texture.pkl"

	if os.path.exists(color_ds_string) and os.path.exists(texture_ds_string):
		print("Color and Data structures already created")
		return

	original_image = cv2.imread(image_path)
	img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
	img_copy = img.copy()
	img_copy_2 = img.copy()

	r,c,d = img.shape
	print(img.shape)

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

	# Reshape textured image for kmeans
	reshaped_textured_img = textured_image.reshape((-1,3))
	reshaped_textured_img = np.float32(reshaped_textured_img)

	# Perform k-means on textured image
	ret_texture,label_texture,center_texture=cv2.kmeans(reshaped_textured_img,5,None,criteria, attempts ,cv2.KMEANS_RANDOM_CENTERS)

	# Now convert back into uint8, and make original image
	center_texture = np.uint8(center_texture)
	res_texture = center_texture[label_texture.flatten()]
	res_texture = res_texture.reshape((text_r, text_c,3))

	# Perform Laplacian of Gaussian Edge Detection and convert back to grayscale
	laplacian_texture = cv2.Laplacian(res_texture,cv2.CV_64F)
	gray_texture=cv2.cvtColor(laplacian_texture.astype('uint8'),cv2.COLOR_BGR2GRAY)

	# Reshape original image for kmeans
	reshaped_img = img.reshape((-1,3))
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

			# Invert image to have non-edges be non-0
			if not gray_texture[i][j] == 0:
				gray_texture[i][j] = 0
			else:
				gray_texture[i][j] = 128

	connectivity = 4

	# Perform connected components on both texture and original image
	num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(gray, connectivity, cv2.CV_64F)
	num_labels_2, labels_2, stats_2, centroids_2 = cv2.connectedComponentsWithStats(gray_texture, connectivity, cv2.CV_64F)

	# Find top connected component regions in both images
	top_labels = largest_CC_regions(num_labels, 10, stats, (r*c)/5, c, r, 8)
	top_labels_2 = largest_CC_regions(num_labels_2, 5, stats_2, r*c, c, r, 8)

	color_ds_string = os.path.splitext(image_path)[0] + "_color.pkl"
	afile = open(color_ds_string, 'wb')
	pickle.dump(RAG_COLOR, afile)
	afile.close()

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

	for num in range(len(top_labels_2)):

		current_CC = labels_2 == top_labels_2[num]
		index_of_CC = np.where(current_CC == True)

		match_rows = index_of_CC[0]
		match_columns = index_of_CC[1]
		num_matches = len(match_rows)
		rand_r = random.randint(0,255)
		rand_g = random.randint(0,255)
		rand_b = random.randint(0,255)

		for i in range(num_matches):
			img_copy_2[match_rows[i]][match_columns[i]] = [rand_r, rand_g, rand_b]

	fig = plt.figure()
	ax1 = fig.add_subplot(2,2,1)
	ax1.imshow(original_image)
	ax2 = fig.add_subplot(2,2,2)
	ax2.imshow(res2)
	ax3 = fig.add_subplot(2,2,3)
	ax3.imshow(img_copy_2)
	ax4 = fig.add_subplot(2,2,4)
	ax4.imshow(img_copy)
	plt.show()
##############################################################

	if not os.path.exists(color_ds_string):
		create_color_cluster_RAG(original_image, centroids, labels, top_labels, r, c)

	if not os.path.exists(texture_ds_string):
		create_texture_cluster_RAG(original_image, centroids_2, labels_2, top_labels_2, r, c)

if __name__== "__main__":
	segment_image()