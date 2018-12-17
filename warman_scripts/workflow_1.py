"""
=============================================
Regional maxima to region based segmentation
=============================================

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
from skimage.color import label2rgb
from skimage.color import rgb2gray
from skimage.morphology import reconstruction
from skimage.filters import sobel

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
image = io.imread('/Users/CiderBones/Desktop/Laboratory/computer_vision/scikit-image/X402x498-2m1.png')
image = img_as_float(image)

# Extracting only the blue channel from the image. There's less variation in 
# the brightness in this channel.
image = image[:,:,2]


# Calculating the h-dome
image = gaussian_filter(image, 1)
h = 0.4 # Have tried various values for h, 0.4 worked the best for me
seed = image - h
mask = image
dilated = reconstruction(seed, mask, method='dilation')
hdome = image - dilated

# Saving the image (only necessary for testing)
#io.imsave("/Users/CiderBones/Desktop/scikit_output_test_new_nogauss.png", hdome)


"""
==================================================
Region-based segmentation
==================================================

(From the tutorial)
Here we segment objects from a background using the watershed transform.
"""

# Adjusting the output from filtering regional maxima
hdome = rgb2gray(hdome) # Converting the output to grayscale
hdome = img_as_ubyte(hdome) # Converting the float64 output to uint8

# For testing:
# hist = np.histogram(image, bins=np.arange(0, 256))

# After converting the output to uint8, the histogram is still heavily skewed 
# to the left. Rescaling the intenity of the image slightly fixes the skew.
hdome = exposure.rescale_intensity(hdome)

# First, we find an elevation map using the Sobel gradient of the image.
elevation_map = sobel(hdome)

# Next we find markers of the background and the coins based on the extreme
# parts of the histogram of gray values.
markers = np.zeros_like(hdome)
markers[hdome < 25] = 1 # Tried a wide range for these values, will probably 
markers[hdome > 60] = 2 # vary based on the image

# We use the watershed transform to fill regions of the elevation map starting 
# from the markers determined above:
segmentation = morphology.watershed(elevation_map, markers)

# This last method works even better, and the coins can be segmented and
# labeled individually.
segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_image, _ = ndi.label(segmentation)
image_label_overlay = label2rgb(labeled_image, image=hdome)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
axes[0].imshow(hdome, cmap=plt.cm.gray, interpolation='nearest')
axes[0].contour(segmentation, [0.5], linewidths=1.2, colors='y')
axes[1].imshow(image_label_overlay, interpolation='nearest')

for a in axes:
    a.axis('off')

plt.tight_layout()

plt.show()












