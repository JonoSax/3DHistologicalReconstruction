'''
This function extracts a tif file of specified resolution from the ndpi file
'''

import os
from glob import glob
from HelperFunctions.Utilities import *

# kernel = 150
# dataTrain =  "Data.nosync/testing/"
# size = 0

# magnification levels of the tif files available
tifLevels = [20, 10, 5, 2.5, 1.25, 0.625, 0.3125, 0.15625]


def load(dataTrain, imageName = '', size = 0):

    # This moves the quadrants into training/testing data based on the annotations provided
    # Input:    (dataTrain), directory/ies which contain the txt files of the annotation co-ordinates 
    #               as extracted by SegmentLoad.py
    #           (imageName), list of the directories of the quandrated tif files as sectioned by quadrants
    #           (size), image size to extract, defaults to the largest one
    # Output:   (), the tif file at the chosen level of magnification

    # What needs to happen is there needs to be some kind of recognition of when annotations are coupled
    # IDEA: use the x and y median values to find annotations which are close to each other
    #       calculate the range of the values of each annotation to find which one is inside which 

    # convert ndpi images into tif files of set size

    # create a folder for the tif files
    dataTif = dataTrain + str(size) + '/tifFiles/'
    dirMaker(dataTif)

    imagesNDPI = sorted(glob(dataTrain + imageName + "*.ndpi"))
    
    # convert the ndpi into a tif
    for img in imagesNDPI:
        print(nameFromPath(img) + " converted @ " + str(tifLevels[size]) + " res")
        ndpiLoad(size, img, dataTif)


def ndpiLoad(sz, src, dest):

    # This function extracts tif files from the raw ndpi files. This uses the 
    # ndpitool from https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/ 
    # Install as necessary. 
    # Input:    (i), magnificataion level to be extracted from the ndpi file
    #           options are 0.15625, 0.3125, 0.625, 1.25, 2.5, 5, 10, 20
    #           (src), file to be extracted with set magnification
    #           (dest), destination
    # Output:   (), tif file of set magnification, saved in the tifFiles directory

    mag = tifLevels[sz]

    os.system("ndpisplit -x" + str(mag) + " " + str(src))

    nameSRC = src.split("/")[-1].split(".")[0]                    # photo name
    dirSRC = src.split(nameSRC + ".ndpi")[0]                      # folder of photo

    extractedName = glob(src.split(".ndpi")[0] + "*z0.tif")[0]    # NOTE, use of z0 is to prevent 
                                                                # duplication of the same file, however 
                                                                # if there is z shift then this will fail

    imgDir = dest + nameSRC + "_" + str(sz) + ".tif"
    os.rename(extractedName, imgDir)

    
'''
data = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'

segmentation(200, 4, data, imageName = 'testWSI1')
'''