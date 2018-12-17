"""
==================================================
Comparing edge-based and region-based segmentation
==================================================

In this example, we will see how to segment objects from a background. We use
the ``coins`` image from ``skimage.data``, which shows several coins outlined
against a darker background.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage import io
from skimage import img_as_ubyte
from skimage import exposure
from skimage.color import rgb2gray

# Loading my image, which is output from the plot_regional_maxima tutorial
image = io.imread('/Users/CiderBones/Desktop/Laboratory/computer_vision/scikit-image/plot_regional_maxima_output.png')
# print(image.ndim)
# print(image.dtype)
image = rgb2gray(image) # Converting the image to grayscale
# print(image.ndim)
# print(image.dtype)
image = img_as_ubyte(image) # Converting the float64 image to uint8
# print(image.dtype)
# After converting the image to uint8, the histogram is still heavily skewed to the left.
# Rescaling the intenity of the image slightly fixes the skew.
image = exposure.rescale_intensity(image)

hist = np.histogram(image, bins=np.arange(0, 256))

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(image, cmap="gray", interpolation='nearest')
axes[0].axis('off')
axes[1].plot(hist[1][:-1], hist[0], lw=2)
axes[1].set_title('histogram of gray values')

######################################################################
#
# Thresholding
# ============
#
# A simple way to segment the coins is to choose a threshold based on the
# histogram of gray values. Unfortunately, thresholding this image gives a
# binary image that either misses significant parts of the coins or merges
# parts of the background with the coins:

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

axes[0].imshow(image > 100, cmap=plt.cm.gray, interpolation='nearest')
axes[0].set_title('image > 100')

axes[1].imshow(image > 150, cmap=plt.cm.gray, interpolation='nearest')
axes[1].set_title('image > 150')

for a in axes:
    a.axis('off')

plt.tight_layout()

######################################################################
# Edge-based segmentation
# =======================
#
# Next, we try to delineate the contours of the coins using edge-based
# segmentation. To do this, we first get the edges of features using the
# Canny edge-detector.

from skimage.feature import canny

edges = canny(image)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(edges, cmap=plt.cm.gray, interpolation='nearest')
ax.set_title('Canny detector')
ax.axis('off')

######################################################################
# These contours are then filled using mathematical morphology.

from scipy import ndimage as ndi

fill_image = ndi.binary_fill_holes(edges)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(fill_image, cmap=plt.cm.gray, interpolation='nearest')
ax.set_title('filling the holes')
ax.axis('off')


######################################################################
# Small spurious objects are easily removed by setting a minimum size for
# valid objects.

from skimage import morphology

image_cleaned = morphology.remove_small_objects(fill_image, 21)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(image_cleaned, cmap=plt.cm.gray, interpolation='nearest')
ax.set_title('removing small objects')
ax.axis('off')

######################################################################
# However, this method is not very robust, since contours that are not
# perfectly closed are not filled correctly, as is the case for one unfilled
# coin above.
#
# Region-based segmentation
# =========================
#
# We therefore try a region-based method using the watershed transform.
# First, we find an elevation map using the Sobel gradient of the image.

from skimage.filters import sobel

elevation_map = sobel(image)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(elevation_map, cmap=plt.cm.gray, interpolation='nearest')
ax.set_title('elevation map')
ax.axis('off')

######################################################################
# Next we find markers of the background and the coins based on the extreme
# parts of the histogram of gray values.

markers = np.zeros_like(image)
markers[image < 25] = 1
markers[image > 60] = 2

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(markers, cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax.set_title('markers')
ax.axis('off')

######################################################################
# Finally, we use the watershed transform to fill regions of the elevation
# map starting from the markers determined above:

segmentation = morphology.watershed(elevation_map, markers)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
ax.set_title('segmentation')
ax.axis('off')

######################################################################
# This last method works even better, and the coins can be segmented and
# labeled individually.

from skimage.color import label2rgb

segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_image, _ = ndi.label(segmentation)
image_label_overlay = label2rgb(labeled_image, image=image)

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(image, cmap=plt.cm.gray, interpolation='nearest')
axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
axes[1].imshow(image_label_overlay, interpolation='nearest')

for a in axes:
    a.axis('off')

plt.tight_layout()

plt.show()
