'''

This script reads from the ndpa file the pins which describe annotatfeatures and 

'''

from Utilities import *
import numpy as np
import tifffile as tifi
import cv2
from scipy.optimize import minimize
from math import ceil

tifLevels = [20, 10, 5, 2.5, 0.625, 0.3125, 0.15625]

def align(data, name = '', size = 0, extracting = True):

    # This function will take a whole slide tif file at the set resolution and 
    # extract the target image/s from the slide and create a txt file which contains the 
    # co-ordinates of the key features on that image for alignment
    # Inputs:   (data), directories of the tif files of interest
    #           (featDir), directory of the features
    # Outputs:  (), extracts the tissue sample from the slide

    # get the file of the features information 
    dataFeat = sorted(glob(data + name + 'featFiles/*.feat'))
    dataTif = sorted(glob(data + name + 'tifFiles/*' + str(size) + '.tif'))

    # create the dictionary of the directories
    featDirs = dictOfDirs(feat = dataFeat, tif = dataTif)

    feats = {}
    tifShapes = {}
    for spec in featDirs.keys():

        # extract the single segment
        corners, tifShape = segmentExtract(data, featDirs[spec], size, extracting = False)
        
        for t in tifShape.keys():
            tifShapes[t] = tifShape[t]

        # get the feature specific positions
        # NOTE that the list order is in descending order (eg 1a, 1b, 2a, 2b...)
        feat = featAdapt(data, featDirs[spec], corners, size)

        for s in feat.keys():
            feats[s] = feat[s]

    segName = data + name + 'segmentedSamples/*' + str(size)

    # get the segmented images
    dataSegment = dictOfDirs(segments = glob(segName + '.tif'))

    # get translation vector of the slice feature shifts 
    for i in range(1):
        print("\nAlignmnet " + str(i))
        tifShapes, feats = shiftFeatures(tifShapes, dataSegment, feats, size, saving = True)
        dataSegment = dictOfDirs(segments = glob(segName + '_aligned.tif'))    # use the most recently aligned images
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

def shiftFeatures(tifShapes, dataSegment, feats, size, saving = False):

    # Function takes the images and translation information and adjusts the images to be aligned and returns the translation vectors used
    # Inputs:   (translationVector), the shift required to align the images
    #           (tifShapes), the size of each image being processed
    #           (dataSegment), dictionary of the dirs for the extracted speciment tissues
    #           (feats), the identified features for each sample
    #           (saving), boolean which controls if the results of the fitting are saved, defaults off
    # Outputs:  (), saves an image of the tissue with the necessary padding to ensure all images are the same size and roughly aligned if saving is True
    #           (feats), the NEW positions of the features after adustment of the translation information
    #           (tifShapes), the NEW size of each image that has been aligned

    # NOTE try and save all the translations and rotations so that the processes can iterate and the FINAL call is to apply to the image sequentially
    translate = list()
    rotate = list()

    # get the maximum dimensions of all the tif images
    tsa = dictToArray(tifShapes, int)
    mx, my, mc = np.max(tsa, axis = 0)

    '''
    translate = {}
    rotate = {}

    # get the maximum dimensions of all the tif images
    tsa = dictToArray(tifShapes, int)
    mx, my, mc = np.max(tsa, axis = 0)

    featNames = list(feats.keys())

    # perform transformations on neighbouring slices and build them up as you go
    for i in range(len(featNames)-1):
        featSection = {}
        for fn in featNames[i:i+2]:
            featSection[fn] = feats[fn]

        translate[fn] += translatePoints(featSection)
    '''

    # find the optimum translation to minimise the error between features
    translationVector = translatePoints(feats)
    translate.append(translationVector)

    # get the measure of the amount of shift to create the 'platform' which all the images are saved to
    ss = dictToArray(translationVector)
    maxSy = np.max(ss[:, 0])
    maxSx = np.max(ss[:, 1])
    minSy = np.min(ss[:, 0])
    minSx = np.min(ss[:, 1]) 

    # translate the feat positions  
    for n in dataSegment:
        for s in feats[n].keys():       # use the target keys in case there are features not common to the previous original 
            feats[n][s] = tuple(np.round(feats[n][s] - translationVector[n] + np.array([maxSy, maxSx]).astype(int)))

    # find the optimum rotational adjustment ot minimise the error between the features
    rotationAdjustment = rotatePoints(feats)
    rotate.append(rotationAdjustment)

    # rotate the feat positions  
    for n in dataSegment:
        for s in feats[n].keys():       # use the target keys in case there are features not common to the previous original 
            feats[n][s] = feats[n][s]

    # get the dims of the total field size to be created for all the images stored
    xF, yF, cF = (mx + maxSx - minSx, my + maxSy - minSy, mc)       # NOTE this will always be slightly larger than necessary because to work it    
                                                                    # out precisely I would have to know what the max displacement + size of the img
                                                                    # is... this is over-estimating the size needed but is much simpler
    
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
            # print(newField.shape)
            # print("x0 = " + str(xp) + " x1 = " + str(xp+mx))
            # print("y0 = " + str(yp) + " y1 = " + str(yp+my))
            newField[xp:(xp+mx), yp:(yp+my), :] += fieldR

            # re-assign the shape of the image
            tifShapes[n] = newField.shape

            # plotPoints(dirToSave + '_normfield.jpg', fieldR, feats[n]) # plots features on the image with padding for max image size
            plotPoints(dirToSave + '_allfield.jpg', newField, feats[n])                     # plots features on the image with padding for all image translations 
            cv2.imwrite(dirToSave + '_aligned.tif', newField)                               # saves the adjusted image at full resolution 

        print("done translation of " + n)

    return(tifShapes, feats)

def translatePoints(feats):

    # get the shift of each frame
    # Inputs:   (feats), dictionary of each feature
    # Outputs:  (shiftStore), translation applied

    shiftStore = {}
    segments = list(feats.keys())
    shiftStore[segments[0]] = (np.array([0, 0]))
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
        res = minimize(objectiveCartesian, (0, 0), args=(tarP, refP), method = 'Nelder-Mead', tol = 1e-6)
        shift = np.round(res.x).astype(int)
        shiftStore[i] = shift

        # ensure the shift of frame is passed onto the next frame
        ref = {}
        for i in tar.keys():
            ref[i] = tar[i] - shift

    return(shiftStore)

def rotatePoints(feats):

    # get the rotations of each frame
    # Inputs:   (feats), dictionary of each feature
    # Outputs:  (rotationStore), rotation needed to be applied

    rotationStore = {}
    segments = list(feats.keys())
    rotationStore[segments[0]] = cv2.getRotationMatrix2D((0, 0), 0, 1)  # first segment has no rotational transform
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
        rotate, ref = objectivePolar(res.x, refP, tarP, False)   # get the affine transformation used for the optimal rotation
        rotationStore[i] = rotate


    return(rotationStore)

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
    err = np.sum((tarA + pos - refA)**2)
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
    scale = 4

    # find the centre of the target from the annotated points
    tarA = (np.array(list(tar.values()))/scale).astype(int)
    centre = findCentre(tarA)

    # create a 2D matrix view of the features
    tarM, shiftT = denseMatrixViewer(tarA, plot = False)     

    # get the max distance of features
    ext = ceil(np.sqrt(np.max(np.sum((tarA-centre)**2, axis = 1))))  

    # create a new matrix to store the target which can then be rotated within
    # NOTE possible to speed this up, scale the matrix down by a factor of 10 etc then multiple everything by the same factor at the end again fro the actual postions
    tarMn = np.zeros(((np.array(tarM.shape) + ext)).astype(int))
    xn, yn = np.array(tarMn.shape).astype(int)         # get the position of the centre of the new matrix
    centreN = tuple(np.array([xn/2, yn/2]).astype(int))     # the centre point of this matrix
    
    # create the rotational transformation 
    rot = cv2.getRotationMatrix2D(centreN, w, 1)

    # process per target feature
    for i in tar:

        x, y = (np.array(tar[i])/scale - centre).astype(int)       # adjust positions relative to the centre

        # re-initialise the matrix for the search
        tarMnC = tarMn.copy()
        tarMnC[x+centreN[0], y+centreN[1]] = 1

        # apply transformation to the original image
        warp = cv2.warpAffine(tarMnC, rot, (yn, xn))

        # get the positions of the rotations
        tarPosWarp = np.where(warp > 0)
        tarPossible = np.array(tarPosWarp)                          
        tarPossible = np.stack([tarPossible[0, :], tarPossible[1, :]], axis = 1)     # array reshaped for consistency

        # get the non-zero values at the found positions
        #   NOTE the rotation causes the sum of a single point (ie 1) to be spread
        #   over several pixels. Find the spread pixel which has the greatest
        #   representation of the new co-ordinate and save that
        v = warp[tarPosWarp]
        
        # apply transformation to normalise
        tarN[i] = ((tarPossible[np.argmax(v)] - centreN + centre)*scale).astype(int)
    

    # error calculation
    # convert dictionaries to np arrays
    tarNa = (np.array(list(tarN.values()))/scale).astype(int)
    refa = (np.array(list(ref.values()))/scale).astype(int)

    err = np.sum((tarNa - refa)**2)
    # print(str(w[0]) + "º evaluated with " + str(err) + " error score")

    # if optimising, return the error. 
    # if not optimising, return the affine matrix used for the transform
    if minimising:
        return(err)  
    else:
        return(rot, tarN)

        

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
