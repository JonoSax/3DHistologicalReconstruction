'''
this script is performing and measuring the ANHIR registerations
'''

from HelperFunctions.SP_FeatureFinder import feature
from HelperFunctions.Utilities import getMatchingList, denseMatrixViewer, nameFromPath
from HelperFunctions import *
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd

dataHome = '/Volumes/USB/ANHIR/TargetTesting/COAD_08/'
size = 2.5
res = 0.2
cpuNo = 1

def annotateImages(dataHome, size, res):
    '''
    This adds the annotations to the images
    '''

    imgsrc = dataHome + str(size) + "/images/"
    annosrc = dataHome + "landmark/"

    imgs = sorted(glob(imgsrc + "*png"))
    annos = sorted(glob(annosrc + "*"))

    imgsToUse = getMatchingList(annos, imgs)

    for a, i in zip(annos, imgsToUse):
        img = cv2.imread(i)
        anno = pd.read_csv(a)
        for n in anno.index:
            pos = (np.array(anno.loc[n])[1:] * res).astype(int)
            cv2.circle(img, tuple(pos), 1, [0, 255, 0], 20)

        # overwrite the original with the modified
        cv2.imwrite(i, img)

def getTransformedFeatures(dataHome, size):

    '''
    This extracts the annotated features from the annotated
    images 
    '''

    imgsrc = dataHome + str(size) + "/alignedSamples/"

    imgs = sorted(glob(imgsrc + "*png"))

    greenPosAll = {}

    for i in imgs:
        name = nameFromPath(i, 3)
        print(name)
        img = cv2.imread(i)

        imgGreen = np.sum(img * np.array([-1, 1, -1]), axis = 2)

        greenPos = np.where(imgGreen > 100)

        # get the positions of the green points
        gp = np.c_[greenPos[0], greenPos[1]]

        if len(gp) == 0:
            continue

        greenPosAll[name] = gp

    return(greenPosAll)

def getFeatureError(dataHome, size):

    featurePos = getTransformedFeatures(dataHome, size)

    keys = list(featurePos.keys())

    for ref, tar in zip(keys[:-1], keys[1:]):
        refFeatures = featurePos[ref]
        tarFeatures = featurePos[tar]

        denseMatrixViewer([refFeatures, tarFeatures])

if __name__ == "__main__":

    print("\n----------- smallerTif -----------")
    # create jpeg images of all the tifs and a single collated pdf
    # downsize(dataHome, size, res, cpuNo)

    # load the features and annotate the images with them
    # annotateImages(dataHome, size, res)

    print("\n----------- specID -----------")
    # extract the invidiual samples from within the slide
    # specID(dataHome, size, cpuNo)

    print("\n----------- featFind -----------")
    # identify corresponding features between samples 
    # featFind(dataHome, size, cpuNo, featMin = 50, gridNo = 1, dist = 50)

    print("\n----------- AignSegments -----------") 
    # align all the samples
    align(dataHome, size, cpuNo, errorThreshold=np.inf, fullScale=False)

    getFeatureError(dataHome, size)