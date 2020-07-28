'''

This script reads from the ndpa file the pins which describe annotatfeatures and 

NOTE this must be one ONLY per specimen. It won't work between different organs

'''

from .Utilities import *
import numpy as np
import tifffile as tifi
import cv2
from scipy.optimize import minimize
from math import ceil
from glob import glob

tifLevels = [20, 10, 5, 2.5, 1.25, 0.625, 0.3125, 0.15625]

def align(data, name = '', size = 0, extracting = True):

    # This function will take a whole slide tif file at the set resolution and 
    # extract the target image/s from the slide and create a txt file which contains the 
    # co-ordinates of the key features on that image for alignment
    # Inputs:   (data), directories of the tif files of interest
    #           (featDir), directory of the features
    # Outputs:  (), extracts the tissue sample from the slide and aligns them by 
    #           their identified featues

    # get the file of the features information 
    dataFeat = sorted(glob(data + 'pinFiles/' + name + '*.pin'))
    dataTif = sorted(glob(data + '/' + str(size) + '/tifFiles/' + name + '*' + str(size) + '.tif'))
    segSamples = data + str(size) + '/segmentedSamples/'
    alignedSamples = data + str(size) + '/alignedSamples/'
    
    # create the dictionary of the directories
    featDirs = dictOfDirs(feat = dataFeat, tif = dataTif)

    feats = {}
    tifShapes = {}


    for spec in featDirs.keys():
        # for samples with identified features
        # try:
        # extract the single segment
        corners, tifShape = segmentExtract(data, segSamples, featDirs[spec], size, extracting)
        
        for t in tifShape.keys():
            tifShapes[t] = tifShape[t]

        # get the feature specific positions
        feat = featAdapt(data, segSamples, featDirs[spec], corners, size)

        for s in feat.keys():
            feats[s] = feat[s]

        # except:
        #    print("No features for " + spec)
        
    # get affine transformation information of the features for optimal fitting
    translateNet, rotateNet, feats = shiftFeatures(feats, segSamples, alignedSamples)

    # apply the transformations to the samples
    segName = segSamples + name

    # get all the segmented information that is to be transformed
    segmentedSamples = dictOfDirs(
        segments = glob(segName + "*.tif"), 
        bound = glob(segName + "*.bound"), 
        feat = glob(segName + "*.feat"), 
        segsection = glob(segName + "*.segsection"), 
        segsection1 = glob(segName + "*.segsection1"))

    transformSamples(segmentedSamples, alignedSamples, tifShapes, translateNet, rotateNet, size, extracting)
    
    print('Alignment complete')

def segmentExtract(data, segSamples, featDir, size, extracting = True):

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

    return(areas, tifShape)

        # plt.imshow(tifSeg); plt.show()

def featAdapt(data, dest, featDir, corners, size):

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
        dictToTxt(featKey[f], dest + nameFromPath(featDir['tif']) + f + "_" + str(size)  + ".feat")

    for f in boundKey:
        for bK in boundKey[f]:
            boundKey[f][bK] = (boundKey[f][bK] * scale).astype(int)  - corners[f]
        dictToTxt(boundKey[f], dest + nameFromPath(featDir['tif']) + f + "_" + str(size)  + ".bound")

    for s in segSection:
        for f in segSection[s]:
            for sK in segSection[s][f]:
                segSection[s][f][sK] = (segSection[s][f][sK] * scale).astype(int)  - corners[f]
            dictToTxt(segSection[s][f], dest + nameFromPath(featDir['tif']) + f + "_" + str(size)  + "." + s)


    return(specFeatOrder)

def shiftFeatures(feats, dir, alignedSamples):

    # Function takes the images and translation information and adjusts the images to be aligned and returns the translation vectors used
    # Inputs:   (feats), the identified features for each sample
    #           (dir)
    #           (alignedSamples)
    # Outputs:  (translateNet), the translation information to be applied per image
    #           (rotateNet), the rotation information to be applied per image

    translate = {}
    rotate = {}
    featNames = list(feats.keys())

    # store the affine transformations
    translateNet = {}
    rotateNet = {}

    # initialise the first sample with no transformation
    translateNet[featNames[0]] = np.array([0, 0])
    rotateNet[featNames[0]] = 0

    # perform transformations on neighbouring slices and build them up as you go
    for i in range(len(featNames)-1):

        # select neighbouring features to process
        featsO = {}     # feats optimum, at this initialisation they are naive but they will progressively converge 
        for fn in featNames[i:i+2]:
            featsO[fn] = feats[fn]
            
        print("Transforming " + fn + " features")

        # -------- translation only --------
        '''
        translation, featsMod, err = translatePoints(featsMod)
        feats[fn] = featsMod[fn]
        translateNet[fn] = translation[fn]      # assigning rather than iterating over 
        rotateNet[fn] = cv2.getRotationMatrix2D((1, 1), 0, 1)
        '''

        # -------- translation and rotation --------
        translateNet[fn] = np.zeros(2).astype(int)

        featsMod = featsO.copy()
        n = 0
        err1 = 100000
        errorC = 100000
        # for i in range(3):  # This could be a while loop for the error of the rotation
        while errorC > 0:        # once the change in error decreases finish
            print("     Fit " + str(n))
            
            # use the previous error for the next calculation
            err0 = err1

            # get the translation vector
            translation, featsT, err = translatePoints(featsMod)

            # store the accumulative translation vectors 
            translateNet[fn] += translation[fn]

            # find the optimum rotational adjustment
            rotationAdjustment, featsMod, err1 = rotatePoints(featsT)

            # change in error between iterations
            errorC = err0 - err1

            # iterator for printing purposes
            n += 1
        
        # apply ONLY the translation transformations to the original features so that the adjustments are made 
        # to the optimised feature positions
        for f in featsO[fn]:
            featsO[fn][f] -= translateNet[fn]

        # perform a final rotation on the fully translated features to create a SINGLE rotational transformation
        rotationAdjustment, featsFinalMod, errN = rotatePoints(featsO, tol = 1e-8)
        rotateNet[fn] = rotationAdjustment[fn]
        feats[fn] = featsFinalMod[fn]                      # re-assign the new feature positions to fit for

    return(translateNet, rotateNet, feats)

def transformSamples(src, dest, tifShapes, translateNet, rotateNet, feats, size, saving = True):

    # this function takes the affine transformation information and applies it to the samples
    # Inputs:   (src), directories of the segmented samples
    #           (dest), directories to save the aligned samples
    #           (translateNet), the translation information to be applied per image
    #           (rotateNet), the rotation information to be applied per image
    # Outputs   (), saves an image of the tissue with the necessary padding to ensure all images are the same size and roughly aligned if saving is True

    # get the measure of the amount of shift to create the 'platform' which all the images are saved to
    ss = dictToArray(translateNet, int)
    maxSx = np.max(ss[:, 0])
    maxSy = np.max(ss[:, 1])
    minSx = np.min(ss[:, 0])
    minSy = np.min(ss[:, 1]) 

    # make destinate directory
    dirMaker(dest)

    # get the maximum dimensions of all the tif images
    tsa = dictToArray(tifShapes, int)
    mx, my, mc = np.max(tsa, axis = 0)

    # get the dims of the total field size to be created for all the images stored
    xF, yF, cF = (mx + maxSy - minSy, my + maxSx - minSx, mc)       # NOTE this will always be slightly larger than necessary because to work it    
                                                                    # out precisely I would have to know what the max displacement + size of the img
                                                                    # is... this is over-estimating the size needed but is much simpler

    # ---------- apply the transformations onto the images ----------

    # adjust the translations of each image and save the new images with the adjustment
    for n in src:

        info = list(src[n].keys())
        info.remove("segments")
        info.remove("feat")

        dirSegment = src[n]['segments']
        feat = txtToDict(src[n]['feat'])[0]

        # NOTE this is using cv2, probably could convert to using tifi...,.
        field = cv2.imread(dirSegment)
        fy, fx, fc = field.shape

        # debugging commands
        # field = (np.ones((fx, fy, fc)) * 255).astype(np.uint8)      # this is used to essentially create a mask of the image for debugging
        # warp = cv2.warpAffine(field, rotateNet['H653A_48a'], (fy, fx))    # this means there is no rotation applied

        # NOTE featues must be done first by definition to orientate all the othe r
        # information. The formate of the features can vary but it MUST exist in some
        # formate 
        for f in feat:
            feat[f] += np.array([maxSx, maxSy]) - translateNet[n]
        feat = objectivePolar(rotateNet[n], None, False, feat)
        dictToTxt(feat, dest + n + ".feat")
        
        # find the centre of rotation used to align the samples
        centre = findCentre(feat)

        # apply the transformations for all the other types of data as well
        for i in info:
            infoE = txtToDict(src[n][i])[0]
            for f in infoE:
                infoE[f] += np.array([maxSx, maxSy]) - translateNet[n]
            infoE = objectivePolar(rotateNet[n], centre, False, infoE)
            dictToTxt(infoE, dest + n + "." + i)

        # translate the image  
        newField = np.zeros([xF, yF, cF]).astype(np.uint8)      # empty matrix for ALL the images
        xp = maxSx - translateNet[n][0] 
        yp = maxSy - translateNet[n][1] 
        newField[yp:(yp+fy), xp:(xp+fx), :] += field

        # apply the rotational transformation to the image
        # centre = findCentre(dictToArray(feats[n]) + np.array([xp, yp]))

        rot = cv2.getRotationMatrix2D(tuple(centre), -rotateNet[n], 1)
        warped = cv2.warpAffine(newField, rot, (yF, xF))

        try:
            segSect0 = txtToDict(dest + n + ".segsection")[0]
        except:
            segSect0 = {}
            
        try:
            segSect1 = txtToDict(dest + n + ".segsection1")[0]
        except:
            segSect1 = {}

        plotPoints(dest + n + '_alignedAnnotated.jpg', warped, feat, segSect0, segSect1)

        # this takes a while so optional
        if saving:
            cv2.imwrite(dest + n + '.tif', warped)                               # saves the adjusted image at full resolution 

        print("done translation of " + n)

def translatePoints(feats):

    # get the shift of each frame
    # Inputs:   (feats), dictionary of each feature
    # Outputs:  (shiftStore), translation applied
    #           (featsMod), the features after translation
    #           (err), squred error of the target and reference features

    shiftStore = {}
    segments = list(feats.keys())
    shiftStore[segments[0]] = (np.array([0, 0]))
    featsMod = feats.copy()
    ref = feats[segments[0]]

    for i in segments[1:]:

        tar = feats[i]

        # get all the features which both slices have
        tarL = list(tar.keys())
        refL = list(ref.keys())
        feat, c = np.unique(refL + tarL, return_counts=True)
        commonFeat = feat[np.where(c == 2)]

        # create the numpy array with ONLY the common features and their positions
        tarP = {}
        refP = {}
        for cf in commonFeat:
            tarP[cf] = tar[cf]
            refP[cf] = ref[cf]

        # get the shift needed and store
        res = minimize(objectiveCartesian, (0, 0), args=(refP, tarP), method = 'Nelder-Mead', tol = 1e-6)
        shift = np.round(res.x).astype(int)
        err = objectiveCartesian(res.x, tarP, refP)
        shiftStore[i] = shift

        # ensure the shift of frame is passed onto the next frame
        ref = {}
        for t in tar.keys():
            ref[t] = tar[t] - shift

        featsMod[i] = ref

    return(shiftStore, featsMod, err)

def rotatePoints(feats, tol = 1e-6):

    # get the rotations of each frame
    # Inputs:   (feats), dictionary of each feature
    # Outputs:  (rotationStore), affine rotation matrix to rotate the IMAGE --> NOTE 
    #                       rotating the image is NOT the same as the features and this doesn't 
    #                       quite work propertly for some reason...
    #           (featsmod), features after rotation

    rotationStore = {}
    segments = list(feats.keys())
    # rotationStore[segments[0]] = cv2.getRotationMatrix2D((0, 0), 0, 1)  # first segment has no rotational transform
    rotationStore[segments[0]] = 0
    
    featsMod = feats.copy()
    ref = feats[segments[0]]

    for i in segments[1:]:

        tar = feats[i]

        # get all the features which both slices have
        tarL = list(tar.keys())
        refL = list(ref.keys())
        feat, c = np.unique(refL + tarL, return_counts=True)
        commonFeat = feat[np.where(c == 2)]

        # create a dictionary with ONLY the common features and their positions
        tarP = {}
        refP = {}
        for cf in commonFeat:
            tarP[cf] = tar[cf]
            refP[cf] = ref[cf]

        # get the shift needed and store
        res = minimize(objectivePolar, (5), args=(None, True, tarP, refP), method = 'Nelder-Mead', tol = tol) # NOTE create bounds on the rotation
        refR = objectivePolar(res.x, None, False, tar, refP)   # get the affine transformation used for the optimal rotation
        rotationStore[i] = float(res.x)
        err = res.fun

        # reassign this as the new feature
        featsMod[i] = refR

    return(rotationStore, featsMod, err)

# def plotPoints(dir, imgO, points, colour = (255, 0, 0), plot = False):
def plotPoints(dir, imgO, *args):

    # plot circles on annotated points
    # Inputs:   (dir), either a directory (in which case load the image) or the numpy array of the image
    #           (imgO), image directory
    #           (points), dictionary or array of points which refer to the co-ordinates on the image
    # Outputs:  (), saves downsampled jpg image with the points annotated

    # load the image
    if type(imgO) is str:
            imgO = cv2.imread(imgO)

    img = imgO.copy()
    
    si = 50

    # for each set of points add to the image
    for points in args:

        if type(points) is dict:
            points = dictToArray(points)

        for p in points:       # use the target keys in case there are features not common to the previous original 
            pos = tuple(np.round(p).astype(int))
            img = cv2.circle(img, pos, si, (255, 0, 0), si) 
        
        # plot the centre as well
        try:
            img = cv2.circle(img, tuple(findCentre(points)), si, (0, 255, 0), si) 
        except:
            print("No centre drawn")

    # resize the image
    x, y, c = img.shape
    imgResize = cv2.resize(img, (2000, int(2000 * x/y)))

    cv2.imwrite(dir, imgResize, [cv2.IMWRITE_JPEG_QUALITY, 80])

    return (imgResize)

def objectiveCartesian(pos, *args):

    # this function is the error function of x and y translations to minimise error 
    # between reference and target feature co-ordinates
    # Inputs:   (pos), translational vector to optimise
    #           (args), dictionary of the reference and target co-ordinates to fit for
    # Outputs:  (err), the squarred error of vectors given the shift pos

    ref = args[0]   # the first argument is ALWAYS the reference
    tar = args[1]   # the second argument is ALWAYS the target

    tarA = dictToArray(tar, int)
    refA = dictToArray(ref, int)

    # error calcuation
    err = np.sum((refA + pos - tarA)**2)
    # print(str(round(pos[0])) + ", " + str(round(pos[1])) + " evaluated with " + str(err) + " error score")

    return(err)      

def objectivePolar(w, centre, *args):

    # this function is the error function of rotations to minimise error 
    # between reference and target feature co-ordinates    
    # Inputs:   (w), angular translation to optimise
    #           (centre), optional to specify the centre which points are being rotated around
    #           (args), dictionary of the reference and target co-ordinates to fit for 
    #                   boolean on whether to return the error (for optimisation) or rot (for evaluation)
    #                   (minimising), if true then performing optimal fitting of target onto 
    #                   reference. If false then it is just rotating the points given to it as the target
    # Outputs:  (err), the squarred error of vectors given the shift pos
    #           (rot), the affine transform matrix used to (works on images)
    #           (tar), the new features after transforming

    minimising = args[0]
    tar = args[1]   # the second argument is ALWAYS the target, ie the one that is being fitted onto the reference
    if minimising:
        ref = args[2]   # the first argument is ALWAYS the reference, ie the one that isn't rotating
    
    try:
        plotting = args[3]
    except:
        plotting = False
    
    tarN = {}

    # this will shrink the matrices made and all the feature co-ordinates by this 
    # factor in order to reduce the time taken to compute
    # NOTE the more it is scaled the more innacuracies are created, however appears 
    # that it is pretty accurate with a 10 scaling but is also acceptably fast
    scale = 10

    tarA = dictToArray(tar)
    
    # if the centre is not specified, find it from the target points
    if np.sum(centre == None) == 1:
        centre = findCentre(tarA)       # this is the mean of all the features

    # find the centre of the target from the annotated points
    tarA = (tarA/scale).astype(int)
    centre = (centre/scale).astype(int)

    Xmax = int(tarA[:, 0].max())
    Xmin = int(tarA[:, 0].min())
    Ymax = int(tarA[:, 1].max())
    Ymin = int(tarA[:, 1].min())

    y, x = (Ymax - Ymin, Xmax - Xmin)


    # debugging stuff --> shows that the rotational transformation is correct on the features
    # so it would suggest that the centre point on the image is not being matched up well
    
    # create an array to contain all the points found and rotated
    m = 40
    if plotting:
        totalField = np.zeros([y + int(2*m), x + int(2*m), 3])
        cv2.circle(totalField, tuple(centre - [Xmin, Ymin]), 3, (0, 0, 255), 6)
    

    # process per target feature
    tarNames = list(tar)
    tarN = {}

    # adjust the position of the features by w degrees
    # NOTE this is being processed per point instead of all at once because of the fact that rotating points causes
    # a 'blurring' of the new point and to identify what the new point is from this 'blur', we have to be able 
    # to recognise each point. I have decided it is simpler to identify each point by processing each one individually, 
    # rather than doing some kind of k-means clustering rubbish etc. 
    for n in range(len(tarNames)):

        i = tarNames[n]

        # find the feature relative to the centre
        featPos = tarA[n, :] - centre

        # find the max size a matrix would need to be to allow this vector to perform a 360º rotation
        ext = ceil(np.sqrt(np.sum((featPos)**2)))
        fN = featPos + ext          # place the feature within the bounding area ext x ext
        tarB = np.zeros([ext*2+1, ext*2+1])
        tarB[fN[0], fN[1]] = 1


        # create the rotational transformation and apply
        rot = cv2.getRotationMatrix2D((ext, ext), w, 1)
        warp = cv2.warpAffine(tarB, rot, (ext*2, ext*2))

        # get the positions of the rotations
        tarPosWarp = np.array(np.where(warp > 0))                          
        tarPossible = np.stack([tarPosWarp[0, :], tarPosWarp[1, :]], axis = 1)     # array reshaped for consistency


        # get the non-zero values at the found positions
        #   NOTE the rotation causes the sum of a single point (ie 1) to be spread
        #   over several pixels. Find the spread pixel which has the greatest
        #   representation of the new co-ordinate and save that
        v = warp[tuple(tarPosWarp)]
        
        newfeatPos = tarPossible[np.argmax(v)] + centre - ext
        # apply transformation to normalise and extract ONLY the value with the largest portion of its rotation
        tarN[i] = np.round((newfeatPos)*scale).astype(int)

        
        if plotting: 
            cv2.circle(totalField, tuple([featPos[0] + centre[0] + m - Xmin, featPos[1] + centre[1] + m - Ymin]), 3, (255, 0, 0), 6)
            cv2.circle(totalField, tuple([newfeatPos[0]+m - Xmin, newfeatPos[1]+m - Ymin]), 3, (0, 255, 0), 6)
            # print([featPos[0] + centre[0] + m - Xmin, featPos[1] + centre[1] + m - Ymin])
            plt.imshow(totalField); plt.show()        
    # print("w = " + str(w))
    # if optimising, return the error. 
    # if not optimising, return the affine matrix used for the transform
    if minimising:

        tarNa = dictToArray(tarN, int)
        refa = dictToArray(ref, int)
        err = np.sum((tarNa - refa)**2)
        # error calculation
        # print("     err = " + str(err))
        return(err)  

    else:
        return(tarN)

def findCentre(pos):

    # find the mean of an array of points which represent the x and y positions
    # Inputs:   (pos), array
    # Outputs:  (centre), the mean of the x and y points (rounded and as an int)

    if type(pos) == dict:
        pos = dictToArray(pos)

    centre = np.array([np.round(np.mean(pos[:, 0])), np.round(np.mean(pos[:, 1]))]).astype(int)

    return(centre)

'''
# dataHome is where all the directories created for information are stored 
dataHome = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/'
# dataHome = '/Volumes/Storage/'

# dataTrain is where the ndpi and ndpa files are stored 
dataTrain = dataHome + 'HistologicalTraining2/'
# dataTrain = dataHome + 'FeatureID/'
name = ''
size = 3

align(dataTrain, name, size, False)
'''