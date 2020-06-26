'''

This script reads from the ndpa file the pins which describe annotatfeatures and 

'''

from Utilities import *
import numpy as np
import tifffile as tifi
import cv2
from scipy.optimize import least_squares

tifLevels = [20, 10, 5, 2.5, 0.625, 0.3125, 0.15625]

def align(data, name = '', size = 0, extracting = True):

    # This function will take a whole slide tif file at the set resolution and 
    # extract the target image/s from the slide and create a txt file which contains the 
    # co-ordinates of the key features on that image for alignment
    # Inputs:   (data), directories of the tif files of interest
    #           (featDir), directory of the features
    # Outputs:  (), extracts the tissue sample from the slide

    # get the file of the features information 
    dataFeat = glob(data + name + 'featFiles/*.feat')[0:2]
    dataTif = glob(data + name + 'tifFiles/*' + str(size) + '.tif')[0:2]

    # create the dictionary of the directories
    featDirs = dictOfDirs(feat = dataFeat, tif = dataTif)

    feats = list()

    for spec in featDirs.keys():

        # extract the single segment
        corners = segmentExtract(data, featDirs[spec], size, False)

        # get the feature specific positions
        feat = featAdapt(data, featDirs[spec], corners, size)

        for f in feat:
            feats.append(f)

    for n in len(feats) - 1:

        # get the reference and target slices to optimise fitting for
        refSample = feats[n]
        tarSample = feats[n+1]

        # NOTE this is the function to minimise the error on, put this into 
        # the scipy.optimize.least_squares function
        def fun(a, b):

            # function calcuating the RMSE between co-ordinate points
            # Inputs:   (a, b), the dictionary of features labelled
            # Outputs:  (s), the scalar value indicating the RMSE beteen all the points

            s = 0
            for i in a.keys():

                # sum up the accumulative error. If there is a missing feature label 
                # just ignore and don't optimise for it
                try:
                    s+= np.sum((a[i] - b[i])**2)
                except:
                    pass

            return(s)

        pass


        

def featAdapt(data, featDir, corners, size):

    # this function take the features of the annotated features and converts
    # them into the local co-ordinates which are used to align the tissues
    # Inputs:   (data), home directory for all the info
    #           (featDir), the dictionary containing the sample specific dirs
    #           (corners), the global co-ordinate of the top left edge of the img extracted
    #           (size), the size of the image being processed
    # Outputs:  (), save a dictionary of the positions of the annotations relative to
    #               the segmented tissue, saved in the segmented sample folder with 
    #               corresponding name

    featInfo = txtToDict(featDir['feat'])[0]
    scale = tifLevels[size] / max(tifLevels)
    featName = list(featInfo.keys())

    # remove all the positional arguments
    locations = ["top", "bottom", "right", "left"]
    for n in range(len(featName)):  # NOTE you do need to do this rather than dict.keys() because dictionaries don't like changing size...
        f = featName[n]
        for l in locations:
            m = f.find(l)
            if m >= 0:
                featInfo.pop(f)

    featKey = list()
    for f in sorted(featInfo.keys()):
        key = f.split("_")
        featKey.append(key)
    
    # create a dictionary per identified sample 
    specFeatInfo = {}
    specFeatOrder = list()
    for v in np.unique(np.array(featKey)[:, 1]):
        specFeatInfo[v] = {}

    # allocate the scaled and normalised sample to the dictionary PER specimen
    for f, p in featKey:
        specFeatInfo[p][f] = (featInfo[f + "_" + p] * scale).astype(int)  - corners[p]
        
    # save the dictionary
    for p in specFeatInfo.keys():
        name = nameFromPath(featDir['tif']) + p + "_" + str(size) + ".feat"
        specFeatOrder.append(specFeatInfo[p])
        dictToTxt(specFeatInfo[p], data + "segmentedSamples/" + name)

    return(specFeatOrder)

def segmentExtract(data, featDir, size, extracting = True):

    # this funciton extracts the individaul sample from the slice
    # Inputs:   (data), home directory for all the info
    #           (featDir), the dictionary containing the sample specific dirs 
    #           (size), the size of the image being processed
    #           (extracting), boolean whether to load in image and save new one
    # Outputs:  (), saves the segmented image 

    featInfo = txtToDict(featDir['feat'])[0]
    locations = ["top", "bottom", "right", "left"]
    scale = tifLevels[size] / max(tifLevels)

    # create the directory to save the samples
    segSamples = data + 'segmentedSamples/'
    try:
        os.mkdir(segSamples)
    except:
        pass


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

        if extracting:

            # extract the portion of the tif image defined by the boundary
            tifSeg = tif[tb:bb, lb:rb, :]

            # save the segmented tissue
            name = nameFromPath(featDir['tif']) + p + "_" + str(size) + '.tif'
            tifi.imwrite(segSamples + name, tifSeg)

    return(areas)

        # plt.imshow(tifSeg); plt.show()


# dataHome is where all the directories created for information are stored 
dataHome = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/'
dataHome = '/Volumes/Storage/'

# dataTrain is where the ndpi and ndpa files are stored 
dataTrain = dataHome + 'HistologicalTraining/'
dataTrain = dataHome + 'FeatureID/'
name = ''
size = 2

align(dataTrain, name, size)
