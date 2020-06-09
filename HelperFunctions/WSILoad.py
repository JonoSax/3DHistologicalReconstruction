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
import sys
from Utilities import *

# kernel = 150
# imageSRC =  "Data.nosync/testing/"
# size = 0

# magnification levels of the tif files available
tifLevels = [0.15625, 0.3125, 0.625, 1.25, 2.5, 5, 10, 20]


def segmentation(kernel, size, imageSRC, imageName = ''):

    print("\nSTARTING WSILOAD/SEGMENTATION\n")

    # This moves the quadrants into training/testing data based on the annotations provided
    # Input:    (annotations), directory/ies which contain the txt files of the annotation co-ordinates 
    #               as extracted by SegmentLoad.py
    #           (quandrantDirs), list of the directories of the quandrated tif files as sectioned by quadrants
    #           (kernel), square kernel size (pixels)
    # Output:   (), no output instead it seperates the data into test and training directories as 
    #               done by convention for tensorflow training processes 

    # What needs to happen is there needs to be some kind of recognition of when annotations are coupled
    # IDEA: use the x and y median values to find annotations which are close to each other
    #       calculate the range of the values of each annotation to find which one is inside which 

    # get the mask directories --> already at the right pixel locations
    masksDirs = glob(imageSRC + imageName + "*" + "_size_" + str(size) + ".mask")

    # specify the root directory where the identified tissue will be stored 
    targetTissueDir = imageSRC + 'targetTissue/'

    # split the data into quadrants
    quadDirs = quadrants(kernel, size, imageSRC, imageName, makeQuads = False)

    # process the segmentations per specimen
    for md in masksDirs:

        # because the mask files are so large, read them in line by line and process per annotation
        maskAnno = open(md)
        argNo = int(maskAnno.readline().replace("ArgNo_", ""))
        entries = int(maskAnno.readline().replace("ListEntries_", ""))
        for annotation in range(entries):

            # iterate through all the annotations of the mask in the txt file
            rows = int(maskAnno.readline().replace("Rows_", ""))
            cols = int(maskAnno.readline().replace("Cols_", ""))
            anno = np.zeros([rows, cols])

            # read in the positions of the mask
            for r in range(rows):
                if r%1000 == 0:
                    print("row: " + str(r))
                values = maskAnno.readline().split(",")

                for c in range(cols):
                    # store the value 
                    anno[r, c] = int(values[c].replace("\n", ""))

            # denseMatrixViewer(anno)

            # get the quadrant position of each downsampled image and the number of pixels that it makes in it
            quad, comp = np.unique((anno /100).astype(int), axis = 0, return_counts = True)

            # create a seperate true label directory 
            for q in range(len(quad)):
                
                # if 90%+ of the pixels (totally arbituary number --> to investigate the effects of this) within a quadrant contain 
                # target tissue, this can be identified as training data, move into appropriate folder
                if comp[q] >= kernel**2 * 0.1:

                    # the exact file name of the tile part
                    quadrantData = imageSRC + imageName + "_" + str(tifLevels[size]) + ".tif_tiles@" + str(kernel) + "x" + str(kernel) + "/quadrant_" + str(quad[q][0]) + "_" + str(quad[q][1]) + ".tif"

                    # move the identified segments into a folder specifically containing true label data
                    trainingDirs(quadrantData, targetTissueDir, 'Vessels', 'K=' + str(kernel) + "_S=" + str(size))

def quadrants(kernel, size, imageSRC, imageName = '', makeQuads = True):

    print("\nSTARTING WSILOAD/QUADRANTS\n")

    # This function quadrants the tif image into n x n images and saves them in a new directory
    # Input:    (kernel), Square kernel size (pixels)
    #           (size), scale image to be extracted from the ndpi file         
    #           (imageSRC), data source directory
    #           (imageName), OPTIONAL to specify which samples to process
    #           (makeQuads), OPTIONAL to prevent the making of the quadrants again, allows for just the dir info
    # Output:   (dirs), a list of directories which contains the quadranted sections, each 
    #           named according to their quadranted position    

    # get all the npdi files of interest
    imagesNDPI = glob(imageSRC + imageName + "*.ndpi")

    # convert ndpi images into tif files of set size
    ndpiLoad(size, imagesNDPI)

    # get the name of the tif files extracted
    imagesTIF = glob(imageSRC + imageName + "*" + str(tifLevels[size]) + ".tif")

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

        # if quandrating of images being completed
        if makeQuads:
            try:
                os.mkdir(dir)
            except OSError:
                print("\nReplacing existing files\n")

            # create the number of indexes for quadrants 
            up = int(height / kernel)
            acr = int(width / kernel)

            # quadrant tissue
            for w in range(acr):
                for h in range(up):
                    quadrant = img[0 + kernel * h:kernel + kernel * h, 0 + kernel * w:kernel + kernel * w, :]
                    # print(image+"_tiles/quadrant_"+str(h)+"_"+str(w)+"_"+str(image))
                    cv2.imwrite(dir + "/quadrant_" + str(w) + "_" + str(h) + ".tif", quadrant)
                    print("width = " + str(w) + "/" + str(acr) + ", height = " + str(h) + "/" + str(up))

            print(imageTIF + " done")

    return(dirs)

def ndpiLoad(sz, src):

    print("\nSTARTING WSILOAD/NDPILOAD\n")

    # This function extracts tif files from the raw ndpi files. This uses the 
    # ndpitool from https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/ 
    # Install as necessary. 
    # Input:    (i), magnificataion level to be extracted from the ndpi file
    #           options are 0.15625, 0.3125, 0.625, 1.25, 2.5, 5, 10, 20
    #           (src), list of files to be extracted with set magnification
    # Output:   (), tif file of set magnification, saved in the same directory
    #           (), the tif files extracted is renamed to be simplified
    #           as just [name]_[magnification].tif

    mag = tifLevels[sz]

    for s in src:

        os.system("ndpisplit -x" + str(mag) + " " + str(s))

        nameSRC = s.split("/")[-1].split(".")[0]                    # photo name
        dirSRC = s.split(nameSRC + ".ndpi")[0]                      # folder of photo

        extractedName = glob(s.split(".ndpi")[0] + "*z0.tif")[0]    # NOTE, use of z0 is to prevent 
                                                                    # duplication of the same file, however 
                                                                    # if there is z shift then this will fail
        os.rename(extractedName, dirSRC + nameSRC + "_" + str(mag) + ".tif")

data = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'

segmentation(100, 4, data, imageName = 'testWSIMod')