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
    # Outputs:  (), extracts the tissue sample from the slide

    # get the file of the features information 
    dataFeat = sorted(glob(data + name + 'featFiles/*.feat'))[0:3]
    dataTif = sorted(glob(data + name + 'tifFiles/*' + str(size) + '.tif'))[0:3]

    # create the dictionary of the directories
    featDirs = dictOfDirs(feat = dataFeat, tif = dataTif)

    feats = {}
    tifShapes = {}

    # get the times for each process
    times = {}

    for spec in featDirs.keys():

        # extract the single segment
        t_segExtract_a = clock()
        corners, tifShape = segmentExtract(data, featDirs[spec], size, extracting = False)
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

    segName = data + name + 'segmentedSamples/*' + str(size)

    # get the segmented images
    dataSegment = dictOfDirs(segments = glob(segName + '.tif'))

    # get affine transformation information of the features for optimal fitting
    t_shiftFeatures_a = clock()
    translateNet, rotateNet = shiftFeatures(dataSegment, feats)
    t_shiftFeatures_b = clock()
    times['shiftFeatures'] = t_shiftFeatures_b - t_shiftFeatures_a

    # apply the transformations to the samples

    dataSegment = dictOfDirs(segments = glob(segName + '_aligned.tif'))    # use the most recently aligned images

    transformSamples(dataSegment, tifShapes, translateNet, rotateNet, size)
    # print(feats['H653A_48a'])

    
    # shiftFeatures(translationVector, tifShapes, dataSegment, feats, size, saving = True)

    print('Alignment complete')

    pass

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
            name = nameFromPath(featDir['tif']) + p + "_" + str(size) + '.tif'
            tifi.imwrite(segSamples + name, tifSeg)

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

def shiftFeatures(dataSegment, feats):

    # Function takes the images and translation information and adjusts the images to be aligned and returns the translation vectors used
    # Inputs:   (tifShapes), the size of each image being processed
    #           (dataSegment), dictionary of the dirs for the extracted speciment tissues
    #           (feats), the identified features for each sample
    # Outputs:  (translateNet), the translation information to be applied per image
    #           (rotateNet), the rotation information to be applied per image

    # NOTE at the moment the translation and rotation needs to occur all at once.... 
    # REMEMBER that you can do all the translations and rototations on the features 
    # etc and THEN do the image transformations.... 
    # ALSO the standardised image size can be done seperately.....

    # NOTE try and save all the translations and rotations so that the processes can iterate and the FINAL call is to apply to the image sequentially
    translate = list()
    rotate = list()

    # ---------- translate the features and get the transformation info ----------

    
    translate = {}
    rotate = {}

    featNames = list(feats.keys())

    # store the affine transformations
    translateNet = {}
    rotateNet = {}

    # perform transformations on neighbouring slices and build them up as you go
    for i in range(len(featNames)-1):

        # select neighbouring features to process
        featsO = {}     # feats optimum, at this initialisation they are naive but they will progressively converge 
        for fn in featNames[i:i+2]:
            featsO[fn] = feats[fn]

        translateNet[fn] = np.zeros(2).astype(int)

        featsMod = featsO.copy()

        for i in range(3):  # This could be a while loop for the error of the rotation

            # get the translation vector
            translation, featsT, err = translatePoints(featsMod)

            # store the accumulative translation vectors 
            translateNet[fn] += translation[fn]

            # find the optimum rotational adjustment
            rotationAdjustment, featsMod, err = rotatePoints(featsT)

        # apply ONLY the translation transformations to the original features so that the adjustments are made 
        # to the optimised feature positions
        for f in featsO[fn]:
            featsO[fn][f] -= translateNet[fn]

        # perform a final rotation on the fully translated features to create a SINGLE rotational transformation
        rotationAdjustment, featsFinal, errF = rotatePoints(featsO)
        rotateNet[fn] = rotationAdjustment[fn]
        feats[fn] = featsFinal[fn]                      # re-assign the new feature positions to fit for

        # after the fitting, apply the translation to the features
        # perform a final rotational fitting to get a SINGLE rotational transformation from this optimal position

    return(translateNet, rotateNet)

def transformSamples(dataSegment, tifShapes, translateNet, rotateNet, size, saving = True):

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
    xF, yF, cF = (mx + maxSx - minSx, my + maxSy - minSy, mc)       # NOTE this will always be slightly larger than necessary because to work it    
                                                                    # out precisely I would have to know what the max displacement + size of the img
                                                                    # is... this is over-estimating the size needed but is much simpler
 

    # ---------- apply the transformations onto the images ----------

    # adjust the translations of each image and save the new images with the adjustment
    for n in dataSegment:

        dirn = dataSegment[n]['segments']
        dirToSave = regionOfPath(dirn) + n + "_" + str(size)
        
        # only if saving will the whole image be processed
        if saving:
            field = cv2.imread(dirn)
            # plotPoints(dirToSave + '_nomod.jpg', dirn, feats[n])
            # field = (np.zeros(list(field.shape)) * 255).astype(np.uint8)      # this is used to essentially create a mask of the image for debugging

            # put the original image into a standardised shape image
            fieldR = np.zeros([mx, my, mc]).astype(np.uint8)            # empty matrix to store the SINGLE image, standardisation
            w, h, c = field.shape                         
            fieldR[:w, :h, :] += field                                  # add the image INTO the standard window
            # adjust the position of the image within the ENTIRE frame
            newField = np.zeros([xF, yF, cF]).astype(np.uint8)      # empty matrix for ALL the images

            # NOTE this needs to also include the rotational component
            yp = -translationVector[n][0] + maxSy
            xp = -translationVector[n][1] + maxSx
            newField[xp:(xp+mx), yp:(yp+my), :] += fieldR

            # re-assign the shape of the image
            tifShapes[n] = newField.shape

            # plotPoints(dirToSave + '_normfield.jpg', fieldR, feats[n]) # plots features on the image with padding for max image size
            plotPoints(dirToSave + '_allfield.jpg', newField, feats[n])                     # plots features on the image with padding for all image translations 
            cv2.imwrite(dirToSave + '_aligned.tif', newField)                               # saves the adjusted image at full resolution 

            # print(newField.shape)
            # print("x0 = " + str(xp) + " x1 = " + str(xp+mx))
            # print("y0 = " + str(yp) + " y1 = " + str(yp+my))

        print("done translation of " + n)

    return(tifShapes, feats)

def translatePoints(feats):

    # get the shift of each frame
    # Inputs:   (feats), dictionary of each feature
    # Outputs:  (shiftStore), translation applied

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
    # Outputs:  (rotationStore), rotation needed to be applied

    rotationStore = {}
    segments = list(feats.keys())
    rotationStore[segments[0]] = cv2.getRotationMatrix2D((0, 0), 0, 1)  # first segment has no rotational transform
    featsMod = feats.copy()
    ref = feats[segments[0]]

    for i in segments[1:]:

        print("rotating: " + str(i))

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
        rotate, refR, err = objectivePolar(res.x, refP, tarP, False)   # get the affine transformation used for the optimal rotation
        rotationStore[i] = rotate

        # reassign this as the new feature
        featsMod[i] = refR


    return(rotationStore, featsMod, err)

def plotPoints(dir, imgO, points):

    # plot circles on annotated points
    # Inputs:   (dir), either a directory (in which case load the image) or the numpy array of the image
    #           (img), image directory
    #           (points), dictionary of points which refer to the co-ordinates on the image
    # Outputs:  (), saves downsampled jpg image with the points annotated

    if type(imgO) is str:
        imgO = cv2.imread(imgO)

    img = imgO.copy()

    for s in points:       # use the target keys in case there are features not common to the previous original 
        si = 50
        pos = tuple(np.round(points[s]).astype(int))
        img = cv2.circle(img, pos, si, (255, 0, 0), si) 

    x, y, c = img.shape
    imgResize = cv2.resize(img, (1000, int(1000 * x/y)))
    
    cv2.imwrite(dir, img,  [cv2.IMWRITE_JPEG_QUALITY, 20])

def objectiveCartesian(pos, *args):

    # this function is working out the x and y translation to minimise error between co-ordinates
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

    # this function is working out the rotational translation to minimise error between co-ordinates
    # Inputs:   (w), angular translation to optimise
    #           (args), dictionary of the reference and target co-ordinates to fit for 
    #                   boolean on whether to return the error (for optimisation) or rot (for evaluation)
    # Outputs:  (err), the squarred error of vectors given the shift pos
    #           (rot), the affine transform matrix used to (works on images)
    #           (tar), the new features after transforming

    ref = args[0]   # the first argument is ALWAYS the reference
    tar = args[1]   # the second argument is ALWAYS the target
    minimising = args[2]
    tarN = {}

    # this will shrink the matrices made and all the feature co-ordinates by this 
    # factor in order to reduce the time taken to compute
    # NOTE the more it is scaled the more innacuracies are created
    scale = 10

    # find the centre of the target from the annotated points
    tarA = (dictToArray(tar)/scale).astype(int)
    centre = findCentre(tarA)       # this is the mean of all the features

    Xmax = int(tarA[:, 0].max())
    Xmin = int(tarA[:, 0].min())
    Ymax = int(tarA[:, 1].max())
    Ymin = int(tarA[:, 1].min())

    x, y = (Xmax - Xmin + 1, Ymax - Ymin + 1)

    # get the max distance of features
    ext = ceil(np.sqrt(np.max(np.sum((tarA-centre)**2, axis = 1))))  

    # process per target feature
    tarNames = list(tar)
    tarN = {}
    for n in range(len(tarNames)):

        i = tarNames[n]

        featPos = tarA[n, :]


        # find the max size a matrix would need to be to allow this vector to perform a 360º rotation
        ext = ceil(np.sqrt(np.sum((featPos-centre)**2)))
        fN = featPos - centre + ext
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
        
        # apply transformation to normalise and extract ONLY the value with the largest portion of its rotation
        tarN[i] = ((tarPossible[np.argmax(v)] + centre - ext)*scale).astype(int)
    

    # error calculation
    # convert dictionaries to np arrays
    tarNa = dictToArray(tarN, int)
    refa = dictToArray(ref, int)

    err = np.sum((tarNa - refa)**2)
    # print(str(w[0]) + "º evaluated with " + str(err) + " error score")

    # if optimising, return the error. 
    # if not optimising, return the affine matrix used for the transform
    if minimising:
        return(err)  
    else:
        return(rot, tarN, err)

        

def findCentre(pos):

    # find the mean of an array of points which represent the x and y positions
    # Inputs:   (pos), array
    # Outputs:  (centre), the mean of the x and y points (rounded and as an int)

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

align(dataTrain, name, size)
