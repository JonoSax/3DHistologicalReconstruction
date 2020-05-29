'''
This script is extracting the manually segmented sections of the image 
from the WSI and the other information and categorising them into classes
appropriate for the ModelTrainer
'''

import os
import numpy as np
import tifffile as tifi
import cv2
from skimage.measure import block_reduce
from PIL import Image
import glob

kernel = 150
imageSRC =  "TestImage20.tif"

def main(kernel, imageSRC):

    # This function segments the tif image into n x n images 
    # Input:    Square kernel size (pixels)
    #           image source directory            
    # Output:   a directory which contains the segmented sections, each 
    #           named according to their segmented position    

    # type = input("CV2 or PIL processing: ")
    # TO DO: Make this a glob function so that a common name set of the images can be processed (from NDPA)

    # NOTE need to make this dependent on the image input
    outlines = "H653A_9.ndpi.ndpa"

    # find the image by name, agnostic of img type
    # imageDir = glob.glob(image+"*")

    # converting to string

    # size of the square kernel used to segment image

    img = tifi.imread(imageDir)

    height, width, channels = img.shape
    print(height, width, channels)


    try:
        os.mkdir(image + "_tiles")
    except OSError:
        print("\nReplacing existing files\n")

    acr = int(height / kernel)
    up = int(width / kernel)

    # segment tissue
    for h in range(acr):
        for w in range(up):
            segment = img[0 + kernel * h:kernel + kernel * h, 0 + kernel * w:kernel + kernel * w, :]
            # print(image+"_tiles/Segment_"+str(h)+"_"+str(w)+"_"+str(image))
            cv2.imwrite(image + "_tiles/Segment_" + str(h) + "_" + str(w) + "_" + str(imageDir), segment)
            print("height = " + str(h) + "/" + str(up) + ", width = " + str(w) + "/" + +str(acr))


    print("done")