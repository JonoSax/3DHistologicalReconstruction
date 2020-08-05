'''

This script takes seperates a tif image which contains multiple samples and seperates
them, and all their informaion 

'''
if __name__ == "__main__":
    from Utilities import *
else:
    from HelperFunctions.Utilities import *
import numpy as np
import tifffile as tifi
import cv2
from scipy.optimize import minimize
from math import ceil
from glob import glob
import multiprocessing
from multiprocessing import Process, Queue

tifLevels = [20, 10, 5, 2.5, 1.25, 0.625, 0.3125, 0.15625]

def extract(data, name = '', size = 0, extracting = True):

    # This function will take a whole slide tif file at the set resolution and 
    # extract the target image/s from the slide and create a txt file which contains the 
    # co-ordinates of the key features on that image for alignment
    # Inputs:   (data), directories of the tif files of interest
    #           (featDir), directory of the features
    # Outputs:  (), extracts the tissue sample from the slide and aligns them by 
    #           their identified featues

    # get all the segmented information that is to be transformed

    # get the file of the features information 
    dataPin = sorted(glob(data + 'pinFiles/' + name + '*.pin'))
    dataTif = sorted(glob(data + str(size) + '/tifFiles/' + name + '*' + str(size) + '.tif'))
    segSamples = data + str(size) + '/segmentedSamples/'

    # create the dictionary of the directories
    featDirs = dictOfDirs(feat = dataPin, tif = dataTif)

    specimens = nameFromPath(dataTif)

    # parallelise the extraction of the samples and info
    feat = {}
    feats = {}
    corner = {}
    tifShape = {}
    tifShapes = {}
    jobExtract = {}
    jobAdapt = {}
    jobTransform = {}
    q0 = {}
    q1 = {}

    # initialise the segmentextract function
    for spec in specimens:
        q0[spec] = Queue()
        jobExtract[spec] = Process(target=segmentExtract, args = (data, segSamples, featDirs[spec], size, extracting, q0[spec]))     
        jobExtract[spec].start()

    # get the results of segment extract 
    for spec in specimens:
        corner[spec], tifShape[spec] = q0[spec].get()
        jobExtract[spec].join()
        q1[spec] = Queue()
        jobAdapt[spec] = Process(target=featAdapt, args = (data, segSamples, featDirs[spec], corner[spec], size, q1[spec]))
        jobAdapt[spec].start()
        
    # get the results of the featadapt function
    for spec in specimens:
        feat[spec] = q1[spec].get()
        jobAdapt[spec].join()
        
        for k in feat[spec]:
            tifShapes[k] = tifShape[spec][k]

    dictToTxt(tifShapes, segSamples + "all.shape")

    # linear processing, left for debugging
    '''
    for spec in featDirs:
        # for samples with identified features
        # try:
        # extract the single segment
        corner, tifShape = segmentExtract(data, segSamples, featDirs[spec], size, extracting, None)
        
        for t in tifShape.keys():
            tifShapes[t] = tifShape[t]

        # get the feature specific positions
        feat = featAdapt(data, segSamples, featDirs[spec], corner, size)

        for s in feat.keys():
            feats[s] = feat[s]

        # except:
        #    print("No features for " + spec)
    '''

    '''
    # get affine transformation information of the features for optimal fitting
    translateNet, rotateNet, feats = shiftFeatures(feats, segSamples, alignedSamples)
    
    # save the tif shapes, translation and rotation information
    dictToTxt(tifShapes, segSamples + "all.shape")
    dictToTxt(translateNet, segSamples + "all.translated")

    # for the sake of saving, make rotations into lists
    for r in rotateNet:
        rotateNet[r] = [rotateNet[r]]
    dictToTxt(rotateNet, segSamples + "all.rotated")

    # update the specimens found 
    specimens = nameFromPath(list(glob(segSamples + name + "*.feat")))[0:6]
    
    # serial transformation
    for spec in specimens:
        # transformSamples(segmentedSamples[spec], alignedSamples, tifShapes, translateNet, rotateNet, size, extracting)
        transformSamples(segSamples, alignedSamples, spec, alignedSamples, size, extracting)
    
    '''

    # my attempt at parallelising this part of the process. Unfortunately it doesn't work 
    # because the cv2.warpAffine function is objectivePolar fails for AN UNKNOWN REASON
    # when finding the new matrix on the second feature.... unknown specifically but 
    # issues with opencv and multiprocessing are known. 
    '''
    for spec in specimens:
        jobTransform[spec] = Process(target=transformSamples, args = (segmentedSamples[spec], alignedSamples, tifShapes, translateNet, rotateNet, size, extracting)) 
        jobTransform[spec].start()

    for spec in specimens:
        jobTransform[spec].join()
    '''
    print('Extraction complete')

def segmentExtract(data, segSamples, featDir, size, extracting = True, q = None):

    # this funciton extracts the individaul sample from the slice
    # Inputs:   (data), home directory for all the info
    #           (segSamples), destination
    #           (featDir), the dictionary containing the sample specific dirs 
    #           (size), the size of the image being processed
    #           (extracting), boolean whether to load in image and save new one
    # Outputs:  (), saves the segmented image 

    featInfo = txtToDict(featDir['feat'])[0]
    locations = ["top", "bottom", "right", "left"]
    scale = tifLevels[size] / max(tifLevels)

    # create the directory to save the samples
    dirMaker(segSamples)

    tifShape = {}
    keys = list(featInfo.keys())
    bound = {}

    for l in locations:

        # find out what samples are in each
        pos = [k.lower() for k in keys if l in k.lower()]

        # store each array of positions and re-scale them to the size image
        # read in chronological order
        store = {}
        for p in sorted(pos):
            val = p.split("_")[-1]
            store[val] = (featInfo[p]*scale).astype(int)

        # save each co-ordinate point in a dictionary
        bound[l] = store

    # read in the tif image
    tif = tifi.imread(featDir['tif'])
    name = nameFromPath(featDir['tif'])

    areas = {}

    # for each entry extract info --> ASSUMES that there will be the same number of bounding points found
    for p in bound['bottom'].keys():

        # get the y position
        t = bound['top'][p][1]
        b = bound['bottom'][p][1]

        # get the x position
        r = bound['right'][p][0]
        l = bound['left'][p][0]

        yBuff = int((b - t) * 0.05)
        xBuff = int((r - l) * 0.05)

        # create padding
        tb = t - yBuff; bb = b + yBuff
        lb = l - xBuff; rb = r + xBuff

        # ensure bounding box is within the shape of the tif image
        if tb < 0:
            tb = 0
        if bb > tif.shape[0]:
            bb = tif.shape[0]
        if lb < 0:
            lb = 0
        if rb > tif.shape[1]:
            rb = tif.shape[1]
        
        areas[p] = np.array([lb, tb])

        # extract the portion of the tif image defined by the boundary
        tifSeg = tif[tb:bb, lb:rb, :]
        tifShape[name+p] = tifSeg.shape

        if extracting:
            # save the segmented tissue
            print("Extracting " + nameFromPath(featDir['tif']) + p)
            nameP = nameFromPath(featDir['tif']) + p + "_" + str(size) + '.tif'
            tifi.imwrite(segSamples + nameP, tifSeg)
        else:
            print("NOT Extracting, processing " + nameFromPath(featDir['tif']) + p)

    if q is None:
        return(areas, tifShape)
    else:
        q.put([areas, tifShape])

def featAdapt(data, dest, featDir, corners, size, q = None):

    # this function take the features of the annotated features and converts
    # them into the local co-ordinates which are used to align the tissues
    # Inputs:   (data), home directory for all the info
    #           (dest), destination for outputs to be saved
    #           (featDir), the dictionary containing the sample specific dirs
    #           (corners), the global co-ordinate of the top left edge of the img extracted
    #           (size), the size of the image being processed
    # Outputs:  (), save a dictionary of the positions of the annotations relative to
    #               the segmented tissue, saved in the segmented sample folder with 
    #               corresponding name
    #           (specFeatOrder), 

    pinInfo = txtToDict(featDir['feat'])[0]
    scale = tifLevels[size] / max(tifLevels)
    featName = list(pinInfo.keys())
    nameSpec = nameFromPath(featDir['feat'])

    '''
    # remove all the positional arguments
    locations = ["top", "bottom", "right", "left"]
    for n in range(len(featName)):  # NOTE you do need to do this rather than dict.keys() because dictionaries don't like changing size...
        f = featName[n]
        for l in locations:
            m = f.find(l)
            if m >= 0:
                pinInfo.pop(f)
    '''
    # create a dictionary per identified sample 
    specFeatOrder = {}

    # extract only the feat information
    featKey = extractFeatureInfo(pinInfo, 'feat')
    boundKey = extractFeatureInfo(pinInfo, 'bound')
    segSection = extractFeatureInfo(pinInfo, 'segsection')

    # get all the unique segsections 
    segTypes = np.unique([s.split("_")[0] for s in list(pinInfo.keys()) if "seg" in s])
    segSection = {}
    for s in segTypes:
        segSection[s] = extractFeatureInfo(pinInfo, s)

    for f in featKey:
        for fK in featKey[f]:
            featKey[f][fK] = (featKey[f][fK] * scale).astype(int)  - corners[f]
        specFeatOrder[nameSpec+f] = featKey[f]
        dictToTxt(featKey[f], dest + nameFromPath(featDir['tif']) + f + ".feat")

    for f in boundKey:
        for bK in boundKey[f]:
            boundKey[f][bK] = (boundKey[f][bK] * scale).astype(int)  - corners[f]
        dictToTxt(boundKey[f], dest + nameFromPath(featDir['tif']) + f + ".bound")

    for s in segSection:
        for f in segSection[s]:
            for sK in segSection[s][f]:
                segSection[s][f][sK] = (segSection[s][f][sK] * scale).astype(int)  - corners[f]
            dictToTxt(segSection[s][f], dest + nameFromPath(featDir['tif']) + f + "." + s)

    if q is None:
        return(specFeatOrder)
    else:
        q.put(specFeatOrder)

if __name__ == "__main__":
    # dataHome is where all the directories created for information are stored 
    dataTrain = '/Volumes/Storage/H653A_11.3new/'

    # dataTrain = dataHome + 'FeatureID/'
    name = ''
    size = 3

    extract(dataTrain, name, size, True)
