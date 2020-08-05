'''

this script takes feat, bound and segsection information and rotates them to minimise 
the error between slices

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

def align(data, name = '', size = 0, extracting = True):

    # This function will take the extracted sample tif file at the set resolution and 
    # translate and rotate them to minimise the error between slices
    # co-ordinates of the key features on that image for alignment
    # Inputs:   (data), directories of the tif files of interest
    #           (featDir), directory of the features
    # Outputs:  (), extracts the tissue sample from the slide and aligns them by 
    #           their identified featues

    # get the file of the features information 
    segSamples = data + str(size) + '/segmentedSamples/'
    alignedSamples = data + str(size) + '/alignedSamples/'
    dataTif = sorted(glob(data + str(size) + '/tifFiles/' + name + '*' + str(size) + '.tif'))
    dataFeat = sorted(glob(segSamples + name + '*.feat'))

    specimens = sorted(nameFromPath(dataFeat))

    # get affine transformation information of the features for optimal fitting
    shiftFeatures(specimens.copy(), segSamples)
    
    
    # serial transformation
    for spec in specimens:
        # transformSamples(segmentedSamples[spec], alignedSamples, tifShapes, translateNet, rotateNet, size, extracting)
        transformSamples(segSamples, alignedSamples, spec, alignedSamples, size, extracting)
    
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
    print('Alignment complete')

def shiftFeatures(featNames, src):

    # Function takes the images and translation information and adjusts the images to be aligned and returns the translation vectors used
    # Inputs:   (feats), the identified features for each sample
    #           (src)
    #           (alignedSamples)
    # Outputs:  (translateNet), the translation information to be applied per image
    #           (rotateNet), the rotation information to be applied per image

    feats = {}
    for f in featNames:
        feats[f] = txtToDict(src + f + ".feat")[0]


    translate = {}
    rotate = {}

    # store the affine transformations
    translateNet = {}
    rotateNet = {}

    # initialise the first sample with no transformation
    translateNet[featNames[0]] = np.array([0, 0])
    rotateNet[featNames[0]] = [0]

    refFeat = featNames[int(len(featNames)/2)]  # using middle sample for aligning
    refFeat = featNames[0]      # initialise the refFeat as the first sample is sequentially aligning samples

    featNames.remove(refFeat)       # remove whatever feature you are using for aligning

    # perform transformations on neighbouring slices and build them up as you go
    for fn in featNames:
        
        '''
        # select neighbouring features to process
        featsO = {}     # feats optimum, at this initialisation they are naive but they will progressively converge 
        for fn in featNames[i:i+2]:
            featsO[fn] = feats[fn]
        '''

        # align all the images with respect to a single reference frame
        featsO = {}     
        featsO[refFeat] = feats[refFeat]         # MUST USE the first input as reference
        featsO[fn] = feats[fn]                  # MUST USE the second input as target

        refFeat = fn        # re-assign the re-Feat if aligning between slices
            
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
        rotateNet[fn] = [rotationAdjustment[fn]]
        feats[fn] = featsFinalMod[fn]                      # re-assign the new feature positions to fit for

    # save the tif shapes, translation and rotation information
    dictToTxt(translateNet, src + "all.translated")
    dictToTxt(rotateNet, src + "all.rotated")

def transformSamples(segSamples, dest, spec, alignedSamples, size, saving):
    # this function takes the affine transformation information and applies it to the samples
    # Inputs:   (src), directories of the segmented samples
    #           (dest), directories to save the aligned samples
    #           (translateNet), the translation information to be applied per image
    #           (rotateNet), the rotation information to be applied per image
    # Outputs   (), saves an image of the tissue with the necessary padding to ensure all images are the same size and roughly aligned if saving is True

    def adjustPos(src, dest, n, maxPos, translateNet, rotateNet, centre = None):

        # this funciton adjusts the position of features in the txt files and resaves them
        # in the destination location
        # Inputs:   (s), directory of the specimens features to adjust
        #           (dest), directory to save new text file
        #           (n), sample name
        #           (maxPos), field adjustment size
        #           (translateNet), translations of the sample
        #           (rotateNet), rotations of the sample
        #           (centre), position of the centre, if not given will be calculated from input features
        # Outputs:  (), saves the positions with the transformation 
        #           (centre), if the centre is not given then it is to be found from these calculations
        
        # ensure input is a list
        if type(src) != list: src = [src]

        # per inputted file, translate and rotate
        for s in src:
            infoE = txtToDict(s)[0]
            for f in infoE:
                infoE[f] += np.array(maxPos) - translateNet[n]
            infoE = objectivePolar(rotateNet[n][0], centre, False, infoE)
            dictToTxt(infoE, dest + n + "." + s.split(".")[-1])     # from the info in the txt file, rename

        if centre is None:
            return(findCentre(infoE))
        else:
            return(centre)

    # get all the segmented information that is to be transformed
    src = dictOfDirs(
        segments = glob(segSamples + spec + "*.tif"), 
        bound = glob(segSamples + spec + "*.bound"), 
        feat = glob(segSamples +  spec + "*.feat"), 
        segsection = glob(segSamples + spec + "*.segsection*"), 
        tifShapes = segSamples + "all.shape", 
        translateNet = segSamples + "all.translated", 
        rotateNet = segSamples + "all.rotated")

    # load the whole specimen info
    translateNet = txtToDict(src['all']['translateNet'])[0]
    tifShapes = txtToDict(src['all']['tifShapes'])[0]
    rotateNet = txtToDict(src['all']['rotateNet'], float)[0]

    # get the measure of the amount of shift to create the 'platform' which all the images are saved to
    ss = dictToArray(translateNet, int)
    maxSx = np.max(ss[:, 0])
    maxSy = np.max(ss[:, 1])
    minSx = np.min(ss[:, 0])
    minSy = np.min(ss[:, 1]) 
    maxPos = (maxSx, maxSy)

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
    # for n in src:

    n = nameFromPath(src[spec]['feat'])

    dirSegment = src[spec]['segments']

    # NOTE this is using cv2, probably could convert to using tifi...,.
    field = cv2.imread(dirSegment)
    fy, fx, fc = field.shape

    # debugging commands
    # field = (np.ones((fx, fy, fc)) * 255).astype(np.uint8)      # this is used to essentially create a mask of the image for debugging
    # warp = cv2.warpAffine(field, rotateNet['H653A_48a'], (fy, fx))    # this means there is no rotation applied

    # NOTE featues must be done first by definition to orientate all the othe r
    # information. The formate of the features can vary but it MUST exist in some
    # formate 
    
    # find the centre of rotation used to align the samples
    
    centre = None
    for i in ['feat', 'bound', 'segsection']:
        centre = adjustPos(src[spec][i], dest, n, maxPos, translateNet, rotateNet, centre)
    
    print("CENTRE: " + str(centre))

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

    feat = txtToDict(dest + n + ".feat")[0]

    plotPoints(dest + n + '_alignedAnnotatedUpdated.jpg', warped, feat, segSect0, segSect1)

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
            pass

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
        rot = cv2.getRotationMatrix2D((ext, ext), np.float(w), 1)
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

if __name__ == "__main__":
    # dataHome is where all the directories created for information are stored 
    dataTrain = '/Volumes/Storage/H653A_11.3new/'

    # dataTrain = dataHome + 'FeatureID/'
    name = ''
    size = 3

    align(dataTrain, name, size, False)