'''

This script reads from the ndpa file the pins which describe annotatfeatures and 

NOTE this must be one ONLY per specimen. It won't work between different organs

'''

from Utilities import *
import numpy as np
import tifffile as tifi
import cv2
from scipy.optimize import minimize
from math import ceil
from time import clock

tifLevels = [20, 10, 5, 2.5, 0.625, 0.3125, 0.15625]

def align(data, name = '', size = 0, extracting = True):

    # This function will take a whole slide tif file at the set resolution and 
    # extract the target image/s from the slide and create a txt file which contains the 
    # co-ordinates of the key features on that image for alignment
    # Inputs:   (data), directories of the tif files of interest
    #           (featDir), directory of the features
    # Outputs:  (), extracts the tissue sample from the slide and aligns them by 
    #           their identified featues

    # get the file of the features information 
    dataFeat = sorted(glob(data + name + 'featFiles/*.feat'))
    dataTif = sorted(glob(data + name + 'tifFiles/*' + str(size) + '.tif'))

    # create the dictionary of the directories
    featDirs = dictOfDirs(feat = dataFeat, tif = dataTif)

    feats = {}
    tifShapes = {}

    # get the times for each process
    times = {}

    for spec in featDirs.keys():
        # for samples with identified features
        try:
            # extract the single segment
            t_segExtract_a = clock()
            corners, tifShape = segmentExtract(data, featDirs[spec], size, extracting)
            t_segExtract_b = clock()

            times['segmentExtract'] = t_segExtract_b - t_segExtract_a
            
            for t in tifShape.keys():
                tifShapes[t] = tifShape[t]

            # get the feature specific positions
            # NOTE that the list order is in descending order (eg 1a, 1b, 2a, 2b...)
            t_featAdapt_a = clock()
            feat = featAdapt(data, featDirs[spec], corners, size)
            t_featAdapt_b = clock()

            times['featAdapt'] = t_featAdapt_b - t_featAdapt_a

            for s in feat.keys():
                feats[s] = feat[s]

        except:
            print("No features for " + spec)
        
    # get affine transformation information of the features for optimal fitting
    t_shiftFeatures_a = clock()

    translateNet, rotateNet, feats = shiftFeatures(feats)

    t_shiftFeatures_b = clock()
    times['shiftFeatures'] = t_shiftFeatures_b - t_shiftFeatures_a

    # apply the transformations to the samples
    segName = data + name + 'segmentedSamples/*' + str(size) + '.tif'

    # get the segmented images
    dataSegment = dictOfDirs(segments = glob(segName))

    transformSamples(dataSegment, tifShapes, translateNet, rotateNet, feats, size, extracting)
    # print(feats['H653A_48a'])

    
    # shiftFeatures(translationVector, tifShapes, dataSegment, feats, size, saving = True)

    print('Alignment complete')

    # NOTE TODO perform rotational optimisation fit

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
    #           (specFeatOrder), 

    featInfo = txtToDict(featDir['feat'])[0]
    scale = tifLevels[size] / max(tifLevels)
    featName = list(featInfo.keys())
    nameSpec = nameFromPath(featDir['feat'])

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
    specFeatOrder = {}
    for v in np.unique(np.array(featKey)[:, 1]):
        specFeatInfo[v] = {}

    # allocate the scaled and normalised sample to the dictionary PER specimen
    for f, p in featKey:
        specFeatInfo[p][f] = (featInfo[f + "_" + p] * scale).astype(int)  - corners[p]
        
    # save the dictionary
    for p in specFeatInfo.keys():
        name = nameFromPath(featDir['tif']) + p + "_" + str(size) + ".feat"
        specFeatOrder[nameSpec+p] = specFeatInfo[p]
        dictToTxt(specFeatInfo[p], data + "segmentedSamples/" + name)

    return(specFeatOrder)

def shiftFeatures(feats):

    # Function takes the images and translation information and adjusts the images to be aligned and returns the translation vectors used
    # Inputs:   (feats), the identified features for each sample
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

        for i in range(3):  # This could be a while loop for the error of the rotation

            # get the translation vector
            translation, featsT, err = translatePoints(featsMod)

            # store the accumulative translation vectors 
            translateNet[fn] += translation[fn]

            # find the optimum rotational adjustment
            rotationAdjustment, featsMod = rotatePoints(featsT)
        
        # apply ONLY the translation transformations to the original features so that the adjustments are made 
        # to the optimised feature positions
        for f in featsO[fn]:
            featsO[fn][f] -= translateNet[fn]

        # perform a final rotation on the fully translated features to create a SINGLE rotational transformation
        rotationAdjustment, featsFinalMod = rotatePoints(featsO)
        rotateNet[fn] = rotationAdjustment[fn]
        feats[fn] = featsFinalMod[fn]                      # re-assign the new feature positions to fit for
        
        # after the fitting, apply the translation to the features
        # perform a final rotational fitting to get a SINGLE rotational transformation from this optimal position

    return(translateNet, rotateNet, feats)

def transformSamples(dataSegment, tifShapes, translateNet, rotateNet, feats, size, saving = True):

    # this function takes the affine transformation information and applies it to the samples
    # Inputs:   (dataSegment), directories
    #           (translateNet), the translation information to be applied per image
    #           (rotateNet), the rotation information to be applied per image
    # Outputs   (), saves an image of the tissue with the necessary padding to ensure all images are the same size and roughly aligned if saving is True

    # get the measure of the amount of shift to create the 'platform' which all the images are saved to
    ss = dictToArray(translateNet, int)
    maxSy = np.max(ss[:, 0])
    maxSx = np.max(ss[:, 1])
    minSy = np.min(ss[:, 0])
    minSx = np.min(ss[:, 1]) 

    # get the maximum dimensions of all the tif images
    tsa = dictToArray(tifShapes, int)
    mx, my, mc = np.max(tsa, axis = 0)

    # get the dims of the total field size to be created for all the images stored
    xF, yF, cF = (mx + maxSx, my + maxSy, mc)       # NOTE this will always be slightly larger than necessary because to work it    
                                                                    # out precisely I would have to know what the max displacement + size of the img
                                                                    # is... this is over-estimating the size needed but is much simpler

    # ---------- apply the transformations onto the images ----------

    # adjust the translations of each image and save the new images with the adjustment
    for n in dataSegment:

        dirn = dataSegment[n]['segments']
        dirToSave = regionOfPath(dirn) + "annotated_" + n + "_" + str(size)
        
        field = cv2.imread(dirn)

        fx, fy, fc = field.shape

        # debugging commands
        # field = (np.ones((fx, fy, fc)) * 255).astype(np.uint8)      # this is used to essentially create a mask of the image for debugging
        # warp = cv2.warpAffine(field, rotateNet['H653A_48a'], (fy, fx))    # this means there is no rotation applied

        # translate the image  
        newField = np.zeros([xF, yF, cF]).astype(np.uint8)      # empty matrix for ALL the images
        yp = -translateNet[n][0] + maxSy
        xp = -translateNet[n][1] + maxSx
        newField[xp:(xp+fx), yp:(yp+fy), :] += field

        # apply the rotational transformation to the image
        # centre = findCentre(dictToArray(feats[n]) + np.array([xp, yp]))
        centre = findCentre(feats[n])

        rot = cv2.getRotationMatrix2D(tuple(centre), -rotateNet[n], 1)
        warped = cv2.warpAffine(newField, rot, (yF, xF))

        featSpecAdjust = dictToArray(feats[n]) + np.array([maxSy, maxSx])
        plotPoints(dirToSave + '_aligned.jpg', warped, featSpecAdjust)                     # plots features on the image with padding for all image translations 
        
        # this takes a while so optional
        if saving:
            cv2.imwrite(dirToSave + '_aligned.tif', warped)                               # saves the adjusted image at full resolution 

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

def rotatePoints(feats):

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
        res = minimize(objectivePolar, (5), args=(refP, tarP, True), method = 'Nelder-Mead', tol = 1e-6) # NOTE create bounds on the rotation
        rotate, refR = objectivePolar(res.x, refP, tar, False)   # get the affine transformation used for the optimal rotation
        rotationStore[i] = float(rotate)

        # reassign this as the new feature
        featsMod[i] = refR

    return(rotationStore, featsMod)

def plotPoints(dir, imgO, points, plot = False):

    # plot circles on annotated points
    # Inputs:   (dir), either a directory (in which case load the image) or the numpy array of the image
    #           (img), image directory
    #           (points), dictionary or array of points which refer to the co-ordinates on the image
    # Outputs:  (), saves downsampled jpg image with the points annotated

    if type(points) is dict:
        points = dictToArray(points)

    if type(imgO) is str:
        imgO = cv2.imread(imgO)

    img = imgO.copy()

    si = 50
    for p in points:       # use the target keys in case there are features not common to the previous original 
        pos = tuple(np.round(p).astype(int))
        img = cv2.circle(img, pos, si, (255, 0, 0), si) 
    
    # plot the centre as well
    img = cv2.circle(img, tuple(findCentre(points)), si, (0, 255, 0), si) 

    # resize the image
    x, y, c = img.shape
    imgResize = cv2.resize(img, (1000, int(1000 * x/y)))

    if plot:
        plt.imshow(imgResize); plt.show()
    else:
        cv2.imwrite(dir, imgResize,  [cv2.IMWRITE_JPEG_QUALITY, 80])

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

def objectivePolar(w, *args):

    # this function is the error function of rotations to minimise error 
    # between reference and target feature co-ordinates    
    # # Inputs:   (w), angular translation to optimise
    #           (args), dictionary of the reference and target co-ordinates to fit for 
    #                   boolean on whether to return the error (for optimisation) or rot (for evaluation)
    # Outputs:  (err), the squarred error of vectors given the shift pos
    #           (rot), the affine transform matrix used to (works on images)
    #           (tar), the new features after transforming

    ref = args[0]   # the first argument is ALWAYS the reference
    tar = args[1]   # the second argument is ALWAYS the target
    minimising = args[2]
    try:
        plotting = args[3]
    except:
        plotting = False
    tarN = {}

    # this will shrink the matrices made and all the feature co-ordinates by this 
    # factor in order to reduce the time taken to compute
    # NOTE the more it is scaled the more innacuracies are created
    scale = 10

    # find the centre of the target from the annotated points
    tarA = np.round(dictToArray(tar)/scale).astype(int)
    centre = findCentre(tarA)       # this is the mean of all the features

    Xmax = int(tarA[:, 0].max())
    Xmin = int(tarA[:, 0].min())
    Ymax = int(tarA[:, 1].max())
    Ymin = int(tarA[:, 1].min())

    y, x = (Ymax - Ymin, Xmax - Xmin)

    # create an array to contain all the points found and rotated
    m = 40

    # debugging stuff --> shows that the rotational transformation is correct on the features
    # so it would suggest that the centre point on the image is not being matched up well
    
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
        tarB = np.zeros([ext*2, ext*2])
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
    tarNa = dictToArray(tarN, int)
    if minimising:
        # error calculation
        refa = dictToArray(ref, int)
        err = np.sum((tarNa - refa)**2)
        # print("     err = " + str(err))

        return(err)  

    else:
        return(w, tarN)

def findCentre(pos):

    # find the mean of an array of points which represent the x and y positions
    # Inputs:   (pos), array
    # Outputs:  (centre), the mean of the x and y points (rounded and as an int)

    if type(pos) == dict:
        pos = dictToArray(pos)

    centre = np.array([np.round(np.mean(pos[:, 0])), np.round(np.mean(pos[:, 1]))]).astype(int)

    return(centre)


# dataHome is where all the directories created for information are stored 
dataHome = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/'
# dataHome = '/Volumes/Storage/'

# dataTrain is where the ndpi and ndpa files are stored 
dataTrain = dataHome + 'HistologicalTraining2/'
# dataTrain = dataHome + 'FeatureID/'
name = ''
size = 3

align(dataTrain, name, size, True)
