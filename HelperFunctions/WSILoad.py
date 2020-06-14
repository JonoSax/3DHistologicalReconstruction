'''
This function extracts a tif file of specified resolution 
'''

import os
import numpy as np
import tifffile as tifi
import cv2
from PIL import Image
from glob import glob
import sys
from .Utilities import *

# kernel = 150
# imageSRC =  "Data.nosync/testing/"
# size = 0

# magnification levels of the tif files available
tifLevels = [20, 10, 5, 2.5, 0.625, 0.3125, 0.15625]


def load(imageSRC, imageName = '', size = 0):

    print("\nSTARTING WSILOAD/SEGMENTATION")

    # This moves the quadrants into training/testing data based on the annotations provided
    # Input:    (size), image size to extract, defaults to the largest one
    #           (imageSRC), directory/ies which contain the txt files of the annotation co-ordinates 
    #               as extracted by SegmentLoad.py
    #           (imageName), list of the directories of the quandrated tif files as sectioned by quadrants
    # Output:   (), no output instead it seperates the data into test and training directories as 
    #               done by convention for tensorflow training processes 

    # What needs to happen is there needs to be some kind of recognition of when annotations are coupled
    # IDEA: use the x and y median values to find annotations which are close to each other
    #       calculate the range of the values of each annotation to find which one is inside which 

    # convert ndpi images into tif files of set size

    imagesNDPI = glob(imageSRC + imageName + "*.ndpi")

    for img in imagesNDPI:
        ndpiLoad(size, img)


'''
data = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'

segmentation(200, 4, data, imageName = 'testWSI1')
'''