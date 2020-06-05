'''
This function loads in the entire histological image, quadranted in the 
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

def segmentation(annotations, quadrantDirs):

    # This function segments the data based on the annotations provided
    # Input:    (annotations), directory/ies which contain the txt files of the annotation co-ordinates 
    #               as extracted by SegmentLoad.py
    #           (quandrantDirs), list of the directories of the quandrated tif files as sectioned by quadrants
    # Output:   (), no output instead it seperates the data into test and training directories as 
    #               done by convention for tensorflow training processes 

    # What needs to happen is there needs to be some kind of recognition of when annotations are coupled
    # IDEA: use the x and y median values to find annotations which are close to each other
    #       calculate the range of the values of each annotation to find which one is inside which 

    pass

def quadrants(kernel, size, imageSRC, imageName = ''):

    # This function quadrants the tif image into n x n images 
    # Input:    (kernel), Square kernel size (pixels)
    #           (size), scale image to be extracted from the ndpi file         
    #           (imageSRC), ndpi source directory
    #           (imageName), OPTIONAL to specify which samples to process
    # Output:   (dirs), a list of directories which contains the quadranted sections, each 
    #           named according to their quadranted position    

    # get all the npdi files of interest
    imagesNDPI = glob(imageSRC + imageName + "*.ndpi")

    # convert ndpi images into tif files of set size
    ndpiLoad(size, imagesNDPI)

    # get the name of the tif files extracted
    imagesTIF = glob(imageSRC + "*.tif")

    dirs = list()

    # for each ndpi image found, convert into a tif and quadrant
    for imageTIF in imagesTIF:

        # read the tif file into a numpy array
        img = tifi.imread(imageTIF)

        # get tif dimenstions --> necessary for the quadrantation
        height, width, channels = img.shape

        # create a directory containing the quadrants of the images
        dir = str(imageTIF + "_tiles@" + str(kernel) + "x" + str(kernel))
        dirs.append(dir)

        try:
            os.mkdir(dir)
        except OSError:
            print("\nReplacing existing files\n")

        # create the number of indexes for quadrants 
        acr = int(height / kernel)
        up = int(width / kernel)

        # quadrant tissue
        for h in range(acr):
            for w in range(up):
                quadrant = img[0 + kernel * h:kernel + kernel * h, 0 + kernel * w:kernel + kernel * w, :]
                # print(image+"_tiles/quadrant_"+str(h)+"_"+str(w)+"_"+str(image))
                cv2.imwrite(dir + "/quadrant_" + str(h) + "_" + str(w) + ".tif", quadrant)
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
