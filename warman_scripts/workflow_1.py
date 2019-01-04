"""
============================================
Regional maxima to region based segmentation
============================================

This script combines elements from the scikit-image plot_regional_maxima.py 
and plot_coins_segmentation.py scripts. I've also extracted only the blue 
channel from the original image. The blue channel contains less contrast 
(coming from the green channel mostly), making the segmentation more accurate.
"""

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
=======================
Defining some functions
=======================
"""

# Crops the right and left sides of an image based on the argument "percent" (0-0.5)
def crop_edges(input_image, percent):
	image = input_image

	crop_column_width = image.shape[1]

	left_crop_column = int(round(percent * crop_column_width))
	right_crop_column =  int(round(crop_column_width - (percent * crop_column_width)))

	cropped_image = image[:, left_crop_column:right_crop_column]

	return cropped_image


# Finds the centers of each object in the segmentation mask
def find_centers(input_array):
	array = input_array

# I think it's converting the first 'array' to just 0 or 1, then takes the 
# index from the second 'array.' Supposedly if the third argument is not 
# supplied then it will use all labels greater than zero, but it's not doing 
# that for me, so here I pull out a list of all the labels to use as the index.
	centers = ndi.center_of_mass(array, array, list(range(1, array.max() + 1)))

	return centers


# Gets the mean green channel intensity from each object in a mask
def mean_intensity(input_image, input_mask):
	image = crop_edges(
		input_image, 0.15)
	image = image[:,:,1]
	mask = input_mask
	output_intensity_list = list()

	for object_num in range(1, (np.max(mask) + 1)):
		mean_intensity = np.mean(image[mask == object_num])
		output_intensity_list.append(mean_intensity)

	return output_intensity_list

"""
=========================
Filtering regional maxima
=========================

(From the tutorial)
Here, we use morphological reconstruction to create a background image, which
we can subtract from the original image to isolate bright features (regional
maxima).

Instead of creating a seed image with maxima along the image border, we can
use the features of the image itself to seed the reconstruction process.
Here, the seed image is the original image minus a fixed value, ``h``.

The final result is known as the h-dome of an image since this tends to 
isolate regional maxima of height ``h``. This operation is particularly useful 
when your images are unevenly illuminated.
"""

# Importing maize image and converting to float. Important for subtraction 
# later, which won't work with uint8
raw_image = io.imread('/Users/CiderBones/Desktop/Laboratory/computer_vision/scikit-image/X402x498-2m1.png')
image = img_as_float(raw_image)

# Extracting only the blue channel from the image. There's less variation in 
# the brightness in this channel.
image = image[:,:,2]

# Cropping 15% off each side of the image
image = crop_edges(image, 0.15)

# -----------------------------------------------------------------------------
# # For testing (sets other channels to zero instead of extracting channel):
# image[:,:,0] = 0
# image[:,:,2] = 0
# io.imsave("/Users/CiderBones/Desktop/image_green.png", image)
# 
# # Testing the image cropping
# print(image.shape)
# image_cropped = crop_edges(image, 0.15)
# print(image_cropped.shape)
# -----------------------------------------------------------------------------

# Calculating the h-dome
image = gaussian_filter(image, 1)
h = 0.5 # Have tried various values for h, 0.5 worked the best for me
seed = image - h
mask = image
dilated = reconstruction(seed, mask, method='dilation')
hdome = image - dilated

# -----------------------------------------------------------------------------
# # Saving the image (only necessary for testing)
# io.imsave("/Users/CiderBones/Desktop/scikit_output_test_new_nogauss.png", hdome)
# -----------------------------------------------------------------------------


"""
=========================
Region-based segmentation
=========================

(From the tutorial)
Here we segment objects from a background using the watershed transform.
"""

# Adjusting the output from filtering regional maxima
hdome = rgb2gray(hdome) # Converting the output to grayscale
hdome = img_as_ubyte(hdome) # Converting the float64 output to uint8

# -----------------------------------------------------------------------------
# For testing:
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
markers[hdome < 20] = 1 # Tried a wide range for these values, will probably 
markers[hdome > 80] = 2 # vary based on the image

# We use the watershed transform to fill regions of the elevation map starting 
# from the markers determined above:
segmentation = morphology.watershed(elevation_map, markers)

# This last method works even better, and the coins can be segmented and
# labeled individually.
segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_image, _ = ndi.label(segmentation)
image_label_overlay = label2rgb(labeled_image, image=hdome)

# -----------------------------------------------------------------------------
# # Printing the image array contents, for testing
# for x in labeled_image : print(*x, sep=" ")
#
# 
# # Testing the center finding function
# centers = find_centers(labeled_image)
# print(*centers, sep = '\n')

# # Plotting the centers overlayed on the segmented image
# centers_y, centers_x = zip(*centers)
# plt.imshow(image_label_overlay, interpolation='nearest')
# plt.scatter(centers_x, centers_y, color="black", marker="+")
# plt.show()

# # Plotting the centers overlayed on the original image
# plot_image = crop_edges(raw_image, 0.15)
# plt.imshow(plot_image)
# plt.scatter(centers_x, centers_y, color="black", marker="+")
# plt.show()

# # Note: this way should work, but the axis are flipped for some reason. I 
# # think there's some inconsistency in how im.show plots the x and y and how 
# # they're arranged in the array.
# # plt.scatter(*zip(*centers))
# # plt.show()

# Testing th mean intensity function
mean_intensity_return = mean_intensity(raw_image, labeled_image)
# print(*mean_intensity_return, sep = '\n')
plt.hist(mean_intensity_return, bins = 30)
plt.xlabel('mean pixel intensity')
plt.ylabel('count')
plt.show()
# -----------------------------------------------------------------------------

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(hdome, cmap=plt.cm.gray, interpolation='nearest')
axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
axes[1].imshow(image_label_overlay, interpolation='nearest')

for a in axes:
    a.axis('off')

plt.tight_layout()

# plt.show()













