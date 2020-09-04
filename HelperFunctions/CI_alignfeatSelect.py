'''

This script find features on the samples and aligns them

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
import os
from glob import glob
from multiprocessing import Pool
import multiprocessing

if __name__ != "HelperFunctions.SP_AlignSamples":
    from Utilities import *
    from SP_SampleAnnotator import featSelectArea, featChangePoint
    from SP_SpecimenID import maskMaker, imgStandardiser
    from SP_FeatureFinder import findFeats
    from SP_AlignSamples import shiftFeatures, transformSamples
else:
    from HelperFunctions.Utilities import *
    from HelperFunctions.SP_SampleAnnotator import featSelectArea, featChangePoint
    from HelperFunctions.SP_SpecimenID import maskMaker, imgStandardiser
    from HelperFunctions.SP_FeatureFinder import findFeats
    from HelperFunctions.SP_AlignSamples import shiftFeatures, transformSamples

'''

TODO:   make the alignment process parallel by instead of aligning each image
        to the previous image which is ALIGNED, make each image align to the previous
        image as it and the translation and rotation is summed sequentially for 
        all the previous alignments

'''


def fullMatchingSpec(datasrc, size):

    # this function acesses each segsection that was created and initiates the 
    # full feature identification and alignment process
    # Inputs:   (datasrc), dir of the specimen
    #           (resolution), size of sample 
    # Outputs:  (), aligned samples of the seg sections

    dataSegSections = datasrc + str(size) + "/segSections/"

    segdirs = sorted(glob(dataSegSections + "*"))

    # take a single segment section and perform a complete feature 
    # matching and alignment
    for s in segdirs[:1]:

        fullMatching(s + "/")


def fullMatching(sectiondir):
    
    # this function takes the images from the directories it is pointed to
    # and performas a complete feature and alignment

    # get the seg section images
    imgs = sorted(glob(sectiondir + "*.tif"))[46:]

    # parallelisation info
    cpuCount = int(multiprocessing.cpu_count() * 0.75)
    serialised = False 

    # featfind parameters
    gridNo = 2
    featMin = 20
    dist = 3

    # boolean to save tif after alignment
    saving = False

    # directories to save the modified images
    dataDest = sectiondir + "info/"
    imgDest = sectiondir + "matched/"
    alignedimgDest = sectiondir + "alignedimages/"
    dirMaker(dataDest)
    dirMaker(imgDest)
    
    # perform features matching
    '''
    if serialised: 
        for refsrc, tarsrc in zip(imgs[:-1], imgs[1:]):
            findFeats(refsrc, tarsrc, dataDest, imgDest, gridNo, featMin, dist)

    else:   
        # parallelise with n cores
        with Pool(processes=cpuCount) as pool:
            pool.starmap(findFeats, zip(imgs[:-1], imgs[1:], repeat(dataDest), repeat(imgDest), repeat(gridNo), repeat(featMin), repeat(dist)))

    # perform alignment of the segsections based on the features
    '''

    sampleNames = sorted(nameFromPath(imgs, 3))
    shiftFeatures(sampleNames, dataDest, alignedimgDest)
    # trans form the samples
    if serialised:
        # serial transformation
        for spec in sampleNames:
            transformSamples(spec, sectiondir, dataDest, alignedimgDest, saving, refImg = None)
        
    else:
        # parallelise with n cores
        with Pool(processes=cpuCount) as pool:
            pool.starmap(transformSamples, zip(sampleNames, repeat(sectiondir), repeat(dataDest), repeat(alignedimgDest), repeat(saving), repeat(None)))

    # NOTE have some kind of command to cause the operations to come back in sync here

def featChangeSegPoint(segsrc, img, nopts = 5):

    # this function calls the featChangePoint function to change the features in 
    # an image which has been matched. It needs this because the paths for this are 
    # a little different that whats is used when matching from the original samples

    segInfo = segsrc + 'info/'

    # get all the feats
    tarFeatInfo = sorted(glob(segInfo + "*.tarfeat"))
    refFeatInfo = sorted(glob(segInfo + "*.reffeat"))

    imgs = sorted(glob(segsrc + "*.tif"))

    # get the position of the features 
    featP = np.where(np.array([i.find(img) for i in imgs]) >= 0)[0][0]

    # get the specific reference and target features
    refPath = refFeatInfo[featP]
    tarPath = tarFeatInfo[featP]

    # load the info
    refInfo = txtToDict(refPath, float)[0]
    tarInfo = txtToDict(tarPath, float)[0]

    refImg = cv2.imread(imgs[featP])
    tarImg = cv2.imread(imgs[featP+1])

    refInfoN, tarInfoN = featChangePoint(None, refImg, tarImg, [refInfo, tarInfo], nopts = nopts)

    dictToTxt(refInfoN, refPath, fit = False)
    dictToTxt(tarInfoN, tarPath, fit = False)

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    # datasrc = '/Volumes/USB/H671A_18.5/'
    datasrc = '/Volumes/Storage/H653A_11.3/'
    size = 3

    # featSelectArea(datasrc, size, 2, 0, False)

    fullMatchingSpec(datasrc, size)

    segsrc = datasrc + str(size) + '/segSections/seg0/'

    # this is the target image to be modified
    img = '051_0'

    # featChangeSegPoint(segsrc, img, 8)
    