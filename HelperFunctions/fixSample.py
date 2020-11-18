'''

This script fixes individual samples which have a particularly large shift
to fix

'''

import numpy as np
import multiprocessing
from multiprocessing import Pool
from glob import glob
import cv2

if __name__ != "HelperFunctions.fixSample":
    from Utilities import txtToDict, dictToTxt, nameFromPath
    from SP_SampleAnnotator import featSelectArea, featChangePoint
    from SP_AlignSamples import align
else:
    from HelperFunctions.Utilities import txtToDict, dictToTxt, nameFromPath
    from HelperFunctions.SP_SampleAnnotator import featSelectArea, featChangePoint
    from HelperFunctions.SP_AlignSamples import align


def fixit(dataHome, size, cpuNo, sampleIDs, segSection = False):

    # perform manual fitting of samples which did not align properly
    # Inputs:   (dataHome), directory of the specimen
    #           (size), resolution
    #           (cpuNo), parallelisation 
    #           (sampleIDs), strings of sample IDs which were not aligned properly
    #           (imagesrc), specifies where the images are stored, ie either 
    #           in aligned samples or within the segsections directories

    if segSection is not False: imagesrc = "/segSections/" + segSection
    else:   imagesrc = ""

    # get the size specific source of the aligned information information
    dataaligned = dataHome + str(size) + imagesrc + "/alignedSamples/"
    datainfo = dataHome + str(size) + imagesrc + "/info/"
    datamasked = dataHome + str(size) + imagesrc + "/masked/"

    samples = []
    # search for the key word in the name and verify it exists
    for s in sampleIDs:

        while True: 
            samp = glob(dataaligned + "*" + s + "*.png")

            # if the key word returned only one sample
            if len(samp) == 1:
                samples.append(samp[0])
                break

            # if the key word return more than one sample
            if len(samp) > 1:
                print("For " + s + " which sample are you referring to?")
                for sa in samp:
                    print(nameFromPath(sa, 3))
                s = input("Type in one of the above: ")
            
            # if the key word return no samples adjust the sampleID
            if len(samp) == 0:
                s = input("Key word " + s + " returned no results, check sample exists and type it in here: ")

    # ensure none are repeated
    samples = np.unique(samples).tolist()
    
    # perform manual annotation of the samples
    for targetSample in samples:
        print("Processing " + nameFromPath(targetSample, 3))
        reannotator(datainfo, datamasked, targetSample)

    # if there are tif tiles already there, then replace those as well
    saving = len(glob(dataaligned + "*.tif")) > 0

    # if there were samples to align, re-align the entire specimen
    if len(samples) > 0:
        print("     Realigning samples")
        align(dataHome, size, cpuNo, saving, errorThreshold=300)

    else:
        print("     No samples to fix")


def reannotator(infosrc, imgsrc, targetSample, nopts = 8):

    # get all the feats
    tarFeatInfo = sorted(glob(infosrc + "*.tarfeat"))
    refFeatInfo = sorted(glob(infosrc + "*.reffeat"))
    imgs = sorted(glob(imgsrc + "*.png"))

    # get the position of the features 
    featP = np.where(np.array([i.find(nameFromPath(targetSample, 3)) for i in imgs]) >= 0)[0][0]

    # get the specific reference and target features
    # NOTE it is -1 because the images are "one ahead" because they include the 
    # initial reference image
    refPath = refFeatInfo[featP -1]
    tarPath = tarFeatInfo[featP -1]

    # load the info
    refInfo = txtToDict(refPath, float)[0]
    tarInfo = txtToDict(tarPath, float)[0]
    refImg = cv2.imread(imgs[featP-1])
    tarImg = cv2.imread(imgs[featP])

    # change the features
    refInfoN, tarInfoN = featChangePoint(None, refImg, tarImg, [refInfo, tarInfo], nopts = nopts, title="Manually move 8 features to perform re-alignment")

    dictToTxt(refInfoN, refPath, fit = False)
    dictToTxt(tarInfoN, tarPath, fit = False)

if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")

    dataHome = '/Volumes/Storage/H710C_6.1/'
    dataHome = '/Volumes/Storage/H653A_11.3/'

    size = 3
    cpuNo = 6
    
    sampleIDs = ["54_0", "61_0", "80_1", "90_0"]

    fixit(dataHome, size, cpuNo, sampleIDs)