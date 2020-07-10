'''
This function loads in the entire histological image and extracts the roi from the masks
'''

import os
import numpy as np
import tifffile as tifi
import cv2
from PIL import Image
from glob import glob
from .Utilities import *

# magnification levels of the tif files available
tifLevels = [0.15625, 0.3125, 0.625, 1.25, 2.5, 5, 10, 20]


def segmentation(dataTrain, imageName = '', size = 0):

    print("\nSTARTING WSIEXTRACT/SEGMENTATION")

    # This function is extracting the exact annotated area of the vessel from the original tif
    # Input:    (size), image resolution to process
    #           (annotations), directory/ies which contain the txt files of the annotation co-ordinates 
    #               as extracted by SegmentLoad.py
    #           (quandrantDirs), list of the directories of the quandrated tif files as sectioned by quadrants
    #           
    # Output:   (), saves the tissue which has been annotated into a new directory

    # get the mask directories --> already at the right pixel locations
    maskDirs = glob(dataTrain + 'maskFiles/' + imageName + "*_" + str(size) + ".mask")

    # get the tif file directories
    tifDirs = glob(dataTrain + 'tifFiles/' + imageName + "*_" + str(size) + ".tif")

    sampleNames = nameFromPath(tifDirs)

    # combine the directories for referncing
    dirs = list()
    for i in range(len(maskDirs)):
        dirs.append((maskDirs[i], tifDirs[i], sampleNames[i]))

    # specify the root directory where the identified tissue will be stored 
    targetTissueDir = dataTrain + 'targetTissue/'
    try:
        os.mkdir(targetTissueDir)
    except:
        pass

    # process per specimen
    for maskDir, tifDir, sampleName in dirs:

        # read in the mask (not the arguments --> shouldn't be any stored anyway...)
        mask = txtToList(maskDir)[0]

        # read in the tif 
        tif = tifi.imread(tifDir)

        # for each mask annotation get the quadrant positions of the annotations
        for n in range(len(mask)):

            anno = mask[n]

            # get the size of the mask for pixel capture
            xmax = int(anno[:, 0].max())
            ymax = int(anno[:, 1].max())
            xmin = int(anno[:, 0].min())
            ymin = int(anno[:, 1].min())

            target = np.zeros((xmax - xmin + 1, ymax - ymin + 1, 3))

            # read the tif file annotations into a new area
            for x, y in anno.astype(int):
                target[x-xmin, y-ymin, :] = tif[y, x, :]

            cv2.imwrite(targetTissueDir + sampleName + "_" + str(n) + "_vessel.tif", target.astype(np.uint8)) # NOTE ATM this is only for one class, vessels. In the future add arguments for different classes
            
        # create a user viewable image of the WSI and annotated areas
        maskCover2(tifDir, dataTrain + 'maskFiles' + sampleName + "masked", mask)
        print("created " + dataTrain + sampleName + "masked")


