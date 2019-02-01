#!/usr/bin/env python3

"""
==============================================================================
==============================================================================
=======================          Kernel Cropper         ======================
==============================================================================
==============================================================================

The goal of this script is to take a series of coordinates that mark kernel 
centers in an image and output a series of Imagemagick commands that crop a 
box from the original image containing the kernel. Cell Counter output is in 
xml format, so the first step is to read and parse the xml file into a list 
x/y coordinates of the centers. A box will then be expanded from these 
coordinates based on a box size parameter. The output will be a file of 
ImageMagick commands that will do the crops.
"""

import xml.etree.ElementTree as ET
import numpy as np
import argparse


"""
========================
Setting up the arguments
========================
"""

parser = argparse.ArgumentParser(description='Given xml file of centers, returns ImageMagick commands to crop boxes around centers')
parser.add_argument('-i', '--input_xml_path', type=str, help='Path of input xml file. Should be ImageJ Cell Counter output')
# parser.add_argument('-p', '--image_crop_percentage', type=float, default=0.15, help='Percent width of image to exclude from each side. Between 0 and 0.5.')
parser.add_argument('-b', '--box_size', type=float, default=70, help='Pixel edge length of box')
args = parser.parse_args()


"""
===================
xml parser function
===================

Takes an xml file and returns a list of category and x/y coordinates for each 
kernel.
"""


def parse_xml (input_xml):
	# Make element tree for object
	tree = ET.parse(input_xml)

	# Getting the root of the tree
	root = tree.getroot()

	# Pulling out the name of the image
	image_name_string = (root[0][0].text)

	# Pulling out the fluorescent and non-fluorescent children
	fluorescent = root[1][1]
	nonfluorescent = root[1][2]

	# Setting up some empty lists to move the coordinates from the xml into
	fluor_x = []
	fluor_y = []
	nonfluor_x = []
	nonfluor_y = []

	# # Getting the coordinates of the fluorescent kernels
	for child in fluorescent:
		if child.tag == 'Marker':
			fluor_x.append(child.find('MarkerX').text)
			fluor_y.append(child.find('MarkerY').text)

	# # Getting the coordinates of the non-fluorescent kernels
	for child in nonfluorescent:
		if child.tag == 'Marker':
			nonfluor_x.append(child.find('MarkerX').text)
			nonfluor_y.append(child.find('MarkerY').text)

	# Putting together the results for output
	fluor_coord = np.column_stack((fluor_x, fluor_y))
	nonfluor_coord = np.column_stack((nonfluor_x, nonfluor_y))

	return_list = [fluor_coord, nonfluor_coord, image_name_string]
	return(return_list)



"""
============
Edge cropper
============

This function takes parse_xml output and crops the edges of the ear. This can 
be useful to remove kernels that can be distorted if they're near the edges. 
The function first finds the length of the ear based on the most extreme x 
coordinates. It then removes points that fall within the inputted percentage 
crop.
"""




"""
===========================
Imagemagick command builder
===========================

Takes parse_xml output and builds a list of Imagemagick commands for cropping 
the images.
"""

def command_builder (input_coordinates, input_box_size):
	# Naming/extracting the input variables
	fluor = input_coordinates[0]
	nonfluor = input_coordinates[1]
	image_name = input_coordinates[2]
	box_edge_length = int(input_box_size)

	# Setting up some strings that are the same for every line
	base_command_string = ('convert ' + image_name + ' -crop ')
	box_edge_length_string = (str(box_edge_length) + 'x' + str(box_edge_length))
	fluorescent_path_string = (' ./fluorescent_kernels/')
	nonfluorescent_path_string = (' ./nonfluorescent_kernels/')

	# This is for writing out at the end. Each line will be a ImageMagick 
	# crop command.
	output_string_list = []

	# These commands will make folders in the image folder when the output 
	# script is run.
	output_string_list.append('mkdir -p fluorescent_kernels')
	output_string_list.append('mkdir -p nonfluorescent_kernels')

	# Going through the fluorescent kernels and making the commands
	for index, coord in enumerate(fluor):
		x = int(coord[0])
		y = int(coord[1])

		# Finding the top left corner of a box with edge of box_edge_length 
		# and a center of coord. In this coordinate system, the upper left 
		# corner of the image is 0,0.
		x_box = int(x - (box_edge_length / 2))
		y_box = int(y - (box_edge_length / 2))

		coord_string = (base_command_string + 
			            box_edge_length_string +
			            '+' +
			            str(x_box) +
			            '+' +
			            str(y_box) +
			            fluorescent_path_string +
			            image_name.rstrip('.png') +
			            '_fluor_' +
			            str(index) +
			            '.png')

		output_string_list.append(coord_string)

		print(coord_string)

	# Doing the same thing for the non-fluorescent kernels.
	for index, coord in enumerate(nonfluor):
		x = int(coord[0])
		y = int(coord[1])

		# Finding the top left corner of a box with edge of box_edge_length 
		# and a center of coord. In this coordinate system, the upper left 
		# corner of the image is 0,0.
		x_box = int(x - (box_edge_length / 2))
		y_box = int(y - (box_edge_length / 2))

		coord_string = (base_command_string + 
			            box_edge_length_string +
			            '+' +
			            str(x_box) +
			            '+' +
			            str(y_box) +
			            nonfluorescent_path_string +
			            image_name.rstrip('.png') +
			            '_nonfluor_' +
			            str(index) +
			            '.png')

		####### ADD IF STATEMENT FOR NEGATIVES ##########
		#################################################

		output_string_list.append(coord_string)

		print(coord_string)

	return(output_string_list)



"""
============
List printer
============

Takes command_builder output and prints the list to a file.
"""

def print_list(command_list, image_name_input):
	# Setting up the output file name
	image_name = image_name_input
	image_name_string = (image_name_input.rstrip('.png') +
						 "_commands.sh")

	file = open(image_name_string, "w+")

	for command in command_list:
		file.write("%s\n" % command)

	file.close()


"""
==================
Running the script
==================
"""

# Converts the xml into a list of coordinates (actually a list of two lists of
# coordinates and the name of the file)
coordinates = parse_xml(args.input_xml_path)

# Builds the ImageMagick bash command list
command_list = command_builder(coordinates, args.box_size)

# Prints the commands into a file in the same directory where the script was 
# run.
print_list(command_list, coordinates[2])


















