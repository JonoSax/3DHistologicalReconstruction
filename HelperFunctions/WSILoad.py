'''
This function loads in the entire histological image, segmented in the 
specified kernel size with pre-processing
'''

import os
import numpy as np
import tifffile as tifi
import cv2
from skimage.measure import block_reduce
from PIL import Image
from glob import glob

# kernel = 150
# imageSRC =  "Data.nosync/testing/"
# size = 0

def main(kernel, imageSRC, imageName, size, annotations):

    # This function segments the tif image into n x n images 
    # Input:    (kernel), Square kernel size (pixels)
    #           (imageSRC), ndpi source directory
    #           (size), scale image to be extracted from the ndpi file            
    # Output:   (dir), a list of directories which contains the segmented sections, each 
    #           named according to their segmented position    

    # get all the npdi files of interest
    imagesNDPI = glob(imageSRC + imageName + "*.ndpi")

    # convert ndpi images into tif files of set size
    ndpiLoad(size, imagesNDPI)

    # get the name of the tif files extracted
    imagesTIF = glob(imageSRC + "*.tif")

    dirs = list()

    # for each ndpi image found, convert into a tif and segment
    for imageTIF in imagesTIF:

        # read the tif file into a numpy array
        img = tifi.imread(imageTIF)

        # get tif dimenstions --> necessary for the segmentation
        height, width, channels = img.shape

        # create a directory containing the segmentations of the imager
        dir = str(imageTIF + "_tiles")
        dirs.append(dir)

        try:
            os.mkdir(dir)
        except OSError:
            print("\nReplacing existing files\n")

        # create the number of indexes for segmentation 
        acr = int(height / kernel)
        up = int(width / kernel)

        # segment tissue
        for h in range(acr):
            for w in range(up):
                segment = img[0 + kernel * h:kernel + kernel * h, 0 + kernel * w:kernel + kernel * w, :]
                # print(image+"_tiles/Segment_"+str(h)+"_"+str(w)+"_"+str(image))
                cv2.imwrite(imageTIF + "_tiles/Segment_" + str(h) + "_" + str(w) + ".tif", segment)
                print("height = " + str(h) + "/" + str(up) + ", width = " + str(w) + "/" + str(acr))

        print(imageTIF + " done")

    print("done all")  

    return(dirs)



def ndpiLoad(i, src):

    # This function extracts tif files from the raw ndpi files. This uses the 
    # ndpitool from https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/ 
    # Install as necessary. 
    # Input:    (i), magnificataion level to be extracted from the ndpi file
    #           options are 0.15625, 0.3125, 0.625, 1.25, 2.5, 5, 10, 20
    #           (src), list of files to be extracted with set magnification
    # Output:   (), tif file of set magnification, saved in the same directory
    #           (), the tif files extracted is renamed to be simplified
    #           as just [name]_[magnification].tif

    mag = [0.15625, 0.3125, 0.625, 1.25, 2.5, 5, 10, 20][i]

    for s in src:

        os.system("ndpisplit -x" + str(mag) + " " + str(s))

        nameSRC = s.split("/")[-1].split(".")[0]                    # photo name
        dirSRC = s.split(nameSRC + ".ndpi")[0]                      # folder of photo

        extractedName = glob(s.split(".ndpi")[0] + "*z0.tif")[0]    # NOTE, use of z0 is to prevent 
                                                                    # duplication of the same file, however 
                                                                    # if there is z shift then this will fail
        os.rename(extractedName, dirSRC + nameSRC + "_" + str(mag) + ".tif")
