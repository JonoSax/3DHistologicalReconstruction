'''
this script will take the extracted annotations of the tissue and segment them into quadrants of 
tissue for training
'''

import numpy as np
import cv2
from glob import glob
from HelperFunctions.Utilities import *

def quadrant(dataTrain, name = '', size = 0, kernel = 50):

    # this function takes a single image of tissue and divides it into quadrants of target tissue
    # which are then saved in a new training directory
    # Inputs:   (dataTrain), directory of data 
    #           (name), OPTIONAL specific name 
    #           (size), level tif to use, defaults to the largest resolution 
    #           (kernel), kernel size in pixel, defaults to 50
    # Outputs:  (), saves in a new directory the quadranted data which contains 
    #               a threshold % of the target tissue

    # we know that the target tissue is stored in targetTissue directory created
    dirTarget = dataTrain + 'targetTissue/'

    # ensure that a directory for the segmented tissue exists
    dirSegment = dataTrain + 'segmentedTissue/'
    try:
        os.mkdir(dirSegment)
    except:
        pass

    # get all the annotated sections
    segments = glob(dirTarget + name + "*.tif")

    # quadrant each segmented image into kernel size images containing a set % of target
    # tissue
    for s in segments:
        # start a count for the samples found per sample used
        n = 0

        # get the name of the sample
        name = s.split("/")[-1].replace(".tif", "")

        # read in image and get info
        img = cv2.imread(s)
        h, w, c = img.shape

        # kernel the image
        for x in np.arange(0, h-kernel, int(kernel/2)):
            for y in np.arange(0, w-kernel, int(kernel/2)):

                # get the kernelled area
                imgK = img[x:(x+kernel), y:(y+kernel), :]

                # find all the points of the image which are black (this should mean
                # there are areas not of target tissue)
                imgBn = len(np.where(imgK[:, :, :] == (0,0,0))[0])/3

                # this is use to create the formate for denseMatrixViewer
                # imgB = np.unique(np.stack([imgBs[0], imgBs[1]], axis = 1), axis = 0)
                # denseMatrixViewer(imgB)

                # if more than 95% of the points (TBC) within the kernel are target tissue
                # then save the kerneled area as a training image
                if imgBn < kernel ** 2 * 0.05:
                    cv2.imwrite(dirSegment + str(name) + "_" + str(n) + "_s" + str(size) + "_k" + str(kernel) + ".tif", imgK)
                    n+=1

        print(str(n) + " training samples created from " + str(name))
