"""
============================================
============================================
Regional maxima to region based segmentation
============================================
============================================

This script combines elements from the scikit-image plot_regional_maxima.py 
and plot_coins_segmentation.py scripts. I've also extracted only the blue 
channel from the original image. The blue channel contains less contrast 
(coming from the green channel mostly), making the segmentation more accurate.
"""

import sys
import os.path

import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter # Will this conflict with prev. line?
from skimage import img_as_float
from skimage import io # Added so I can import my own images
from skimage import morphology
from skimage import img_as_ubyte
from skimage import exposure
from skimage.measure import regionprops
from skimage.color import label2rgb
from skimage.color import rgb2gray
from skimage.morphology import reconstruction
from skimage.filters import sobel


"""
===================
Crop edges function
===================

Crops the right and left sides of an image based on the argument "percent" (0-0.5)
"""

def crop_edges(input_image, percent):
	image = input_image

	crop_column_width = image.shape[1]

	left_crop_column = int(round(percent * crop_column_width))
	right_crop_column =  int(round(crop_column_width - (percent * crop_column_width)))

	cropped_image = image[:, left_crop_column:right_crop_column]

	return cropped_image


"""
===============================
Filter regional maxima function
===============================

(Adapted from scikit-image.org/docs/dev/auto_examples/color_exposure/plot_regional_maxima.html)

Here, we use morphological reconstruction to create a background image, which
we can subtract from the original image to isolate bright features (regional
maxima).

Instead of creating a seed image with maxima along the image border, we can
use the features of the image itself to seed the reconstruction process.
Here, the seed image is the original image minus a fixed value, ``h``.

The final result is known as the h-dome of an image since this tends to 
isolate regional maxima of height ``h``. This operation is particularly useful 
when your images are unevenly illuminated.

h input refers to the h-dome parameter. 0.5 is a good place to start.
"""

def filter_regional_maxima(input_image, input_h):
	# Converting input image to float. Important for subtraction later, which 
	# won't work with uint8.
	image = img_as_float(input_image)

	# Extracting only the blue channel from the image. There's less variation in 
	# the brightness in this channel.
	image = image[:,:,2]

	# --------------------------------------------------------------------------
	# # For testing (sets other channels to zero instead of extracting channel):
	# image[:,:,0] = 0
	# image[:,:,2] = 0
	# io.imsave("/Users/CiderBones/Desktop/image_green.png", image)
	# --------------------------------------------------------------------------

	# Calculating the h-dome
	image = gaussian_filter(image, 1)
	h = input_h
	seed = image - h
	mask = image
	dilated = reconstruction(seed, mask, method='dilation')
	hdome = image - dilated

	return hdome


"""
==================================
Region-based segmentation function
==================================

(Adapted from scikit-image.org/docs/dev/auto_examples/xx_applications/plot_coins_segmentation.html)

Here we segment objects from a background using the watershed transform.

Input should be hdome, the output from the Filter regional maxima function.

The marker_low and marker_high parameters determin which levels of the 
histogram become the background and which become the kernels.

So far I've been using (this will likely vary by image):
marker_low = 20
marker_high = 80
"""

def region_based_segmentation(hdome, marker_low, marker_high):
	# Adjusting the output from filtering regional maxima
	hdome = rgb2gray(hdome) # Converting the output to grayscale
	hdome = img_as_ubyte(hdome) # Converting the float64 output to uint8

	# -----------------------------------------------------------------------------
	# For testing, this shows a histogram of the grayscale image:
	# hist = np.histogram(image, bins=np.arange(0, 256))
	# fig, axes = plt.subplots(1, 2, figsize=(8, 3))
	# axes[0].imshow(hdome, cmap=plt.cm.gray, interpolation='nearest')
	# axes[0].axis('off')
	# axes[1].plot(hist[1][:-1], hist[0], lw=2)
	# axes[1].set_title('histogram of gray values')
	# -----------------------------------------------------------------------------

	# After converting the output to uint8, the histogram is still heavily skewed 
	# to the left. Rescaling the intenity of the image slightly fixes the skew.
	hdome = exposure.rescale_intensity(hdome)

	# First, we find an elevation map using the Sobel gradient of the image.
	elevation_map = sobel(hdome)

	# Next we find markers of the background and the coins based on the extreme
	# parts of the histogram of gray values.
	markers = np.zeros_like(hdome)
	markers[hdome < marker_low] = 1 
	markers[hdome > marker_high] = 2 

	# We use the watershed transform to fill regions of the elevation map starting 
	# from the markers determined above:
	segmentation = morphology.watershed(elevation_map, markers)

	# Filling the holes in the segments, labeling them, and making a pretty overlay.
	segmentation = ndi.binary_fill_holes(segmentation - 1)
	labeled_image, _ = ndi.label(segmentation)
	image_label_overlay = label2rgb(labeled_image, image=hdome)

	return {'labeled_image':labeled_image,
	        'image_label_overlay':image_label_overlay}


"""
============================
Find segment center function
============================

Finds the centers of each object in the segmentation mask, as outputted by 
ndimage watershed transform segmentation (see Region based segmentation function)
"""

def find_centers(input_array):
	array = input_array

	# I think it's converting the first 'array' to just 0 or 1, then takes the 
	# index from the second 'array.' Supposedly if the third argument is not 
	# supplied then it will use all labels greater than zero, but it's not doing 
	# that for me, so here I pull out a list of all the labels to use as the index.
	centers = ndi.center_of_mass(array, array, list(range(1, array.max() + 1)))

	return centers


"""
===========================
Get mean intensity function
===========================

Finds the mean intensity in each channel (RGB) for each object in a mask. 

Note: if you are using a cropped image to do the segmentation, make sure you 
input the same cropped image in this function, not the uncropped image.
"""

def mean_intensity(input_image, input_mask):
	red_image = input_image[:,:,0]
	green_image = input_image[:,:,1]
	blue_image = input_image[:,:,2]

	mask = input_mask

	output_red_intensity = list()
	output_green_intensity = list()
	output_blue_intensity = list()

	for object_num in range(1, (np.max(mask) + 1)):
		red_mean_intensity = np.mean(red_image[mask == object_num])
		green_mean_intensity = np.mean(green_image[mask == object_num])
		blue_mean_intensity = np.mean(blue_image[mask == object_num])

		output_red_intensity.append(red_mean_intensity)
		output_green_intensity.append(green_mean_intensity)
		output_blue_intensity.append(blue_mean_intensity)

	return {'red':output_red_intensity, 
	        'green':output_green_intensity,
	        'blue':output_blue_intensity}


"""
=======================
Putting it all together
=======================
"""

# Importing and cropping the image
raw_image = io.imread(os.path.join(sys.path[0], 'X402x498-2m1.png'))
image = crop_edges(raw_image, 0.15)

# Getting the regional maxima
hdome = filter_regional_maxima(image, 0.5)

# Doing the segmentation
segmentation_output = region_based_segmentation(hdome, 20, 80)
image_segments = segmentation_output['labeled_image']
image_overlay = segmentation_output['image_label_overlay']

# Finding the centers of the segments
segment_centers = find_centers(image_segments)

"""
# -----------------------------------------------------------------------------
# Plotting the centers overlayed on the segmented image
centers_y, centers_x = zip(*segment_centers)
plt.imshow(image_overlay, interpolation='nearest')
plt.scatter(centers_x, centers_y, color="black", marker="+")
plt.show()

# Plotting the centers overlayed on the original image
centers_y, centers_x = zip(*segment_centers)
plt.imshow(image)
plt.scatter(centers_x, centers_y, color="black", marker="+")
plt.show()
# -----------------------------------------------------------------------------
"""

# Getting the mean intensity for all channels
mean_intensity = mean_intensity(image, image_segments)
red_intensity = mean_intensity['red']
green_intensity = mean_intensity['green']
blue_intensity = mean_intensity['blue']


# -----------------------------------------------------------------------------
# Printing the image array contents, for testing
# for x in labeled_image : print(*x, sep=" ")

# Plotting histograms of the mean intensity values
plt.hist(red_intensity, bins = 30)
plt.xlabel('red mean pixel intensity')
plt.ylabel('count')
plt.show()

plt.hist(green_intensity, bins = 30)
plt.xlabel('green mean pixel intensity')
plt.ylabel('count')
plt.show()

plt.hist(blue_intensity, bins = 30)
plt.xlabel('blue mean pixel intensity')
plt.ylabel('count')
plt.show()
# -----------------------------------------------------------------------------