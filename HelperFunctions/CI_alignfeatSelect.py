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
    from SP_SampleAnnotator import featChangePoint
    from SP_SpecimenID import maskMaker, imgStandardiser
    from SP_FeatureFinder import findFeats
    from SP_AlignSamples import shiftFeatures, transformSamples
else:
    from HelperFunctions.Utilities import *
    from HelperFunctions.SP_SampleAnnotator import featChangePoint
    from HelperFunctions.SP_SpecimenID import maskMaker, imgStandardiser
    from HelperFunctions.SP_FeatureFinder import findFeats
    from HelperFunctions.SP_AlignSamples import shiftFeatures, transformSamples



def alignfeatSelect(datasrc, size):

    dataSegSections = datasrc + str(size) + "/segSections/"

    segdirs = os.listdir(dataSegSections)

    # take a single segment section and perform a complete feature 
    # matching and alignment
    for s in segdirs:

        sectiondir = dataSegSections + s + "/"

        alignsection(sectiondir)


def alignsection(sectiondir):
    
    # this function takes the images from the directories it is pointed to
    # and performas a complete feature and alignment

    imgs = sorted(glob(sectiondir + "*.tif"))
    cpuCount = int(multiprocessing.cpu_count() * 0.75)
    serialised = False

    gridNo = 2
    featMin = 20
    dist = 3
    saving = False

    dataDest = sectiondir + "info/"
    imgDest = sectiondir + "images/"
    alignedimgDest = sectiondir + "alignedimages/"
    dirMaker(dataDest)
    dirMaker(imgDest)

    '''
    if serialised: 
        for refsrc, tarsrc in zip(imgs[:-1], imgs[1:]):
            findFeats(refsrc, tarsrc, dataDest, imgDest, gridNo, featMin, dist)

    else:   
        # parallelise with n cores
        with Pool(processes=cpuCount) as pool:
            pool.starmap(findFeats, zip(imgs[:-1], imgs[1:], repeat(dataDest), repeat(imgDest), repeat(gridNo), repeat(featMin), repeat(dist)))
    ''' 
    sampleNames = sorted(nameFromPath(imgs, 3))

    shiftFeatures(sampleNames, dataDest, alignedimgDest)

    if serialised:
        # serial transformation
        for spec in sampleNames:
            transformSamples(spec, sectiondir, dataDest, alignedimgDest, saving, refImg = None)
        
    else:
        # parallelise with n cores
        with Pool(processes=cpuCount) as pool:
            pool.starmap(transformSamples, zip(sampleNames, repeat(sectiondir), repeat(dataDest), repeat(alignedimgDest), repeat(saving), repeat(None)))


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    datasrc = '/Volumes/USB/H671A_18.5/'
    size = 3

    alignfeatSelect(datasrc, size)