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
from random import randint
from math import ceil
from glob import glob
import multiprocessing
from multiprocessing import Process, Queue

tifLevels = [20, 10, 5, 2.5, 1.25, 0.625, 0.3125, 0.15625]

# NOTE this should include a function where if there are args in the .feat and .bound 
# files (ie the size of the original image it came from) then re-shape to the new size

'''

TODO
    - make it work as part of the automatic workflow script taking the .feat files
    outputted from FeatFinder 

'''

def align(data, name = '', size = 0, saving = True):

    # This function will take the extracted sample tif file at the set resolution and 
    # translate and rotate them to minimise the error between slices
    # co-ordinates of the key features on that image for alignment
    # Inputs:   (data), directories of the tif files of interest
    #           (featDir), directory of the features
    # Outputs:  (), extracts the tissue sample from the slide and aligns them by 
    #           their identified featues

    # get the file of the features information 
    dataSegmented = data + str(size) + '/tifFiles/'     
    alignedSamples = data + str(size) + '/alignedSamples/'
    segInfo = data + str(size) + '/info/'

    # get the sample slices of the specimen to be processed
    specimenFeats = sorted(nameFromPath(glob(segInfo + "*.feat"), 3))
    specimenTifs = sorted(nameFromPath(glob(dataSegmented + "*.tif")))

    # get affine transformation information of the features for optimal fitting
    # shiftFeatures(specimenFeats, segInfo)
    
    # serial transformation
    for spec in specimenTifs:
        transformSamples(dataSegmented, segInfo, alignedSamples, spec, size, saving)

    # my attempt at parallelising this part of the process. Unfortunately it doesn't work 
    # because the cv2.warpAffine function is objectivePolar fails for AN UNKNOWN REASON
    # when finding the new matrix on the second feature.... unknown specifically but 
    # issues with opencv and multiprocessing are known. 
    '''
    jobs = {}
    for spec in specimens:
        jobs[spec] = Process(target=transformSamples, args = (dataSegmented, segInfo, alignedSamples, spec, size, saving)) 
        jobs[spec].start()

    for spec in specimens:
        jobs[spec].join()
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
    rotateNet[featNames[0]] = [0, 0, 0]

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
        featsO[refFeat] = feats[refFeat].copy()         # MUST USE the first input as reference
        featsO[fn] = feats[fn].copy()                  # MUST USE the second input as target
            
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
        errCnt = 0  

        # once the error is 0 or there are 10 counts of ossiclating between errors
        while (errorC != 0) and (errCnt < 10):        # once the change in error decreases finish
            
            # use the previous error for the next calculation
            err0 = err1
            errorCc = errorC

            # get the translation vector
            translation, featsT, err = translatePoints(featsMod)

            # store the accumulative translation vectors 
            translateNet[fn] += translation[fn]

            # find the optimum rotational adjustment and produce modified feats
            _, featsMod, err1, _ = rotatePoints(featsT)

            # change in errors between iterations, using two errors
            errorC = err0 - err1

            
            # sometimes the calcultions get stuck, ossiclating between two errrors
            # of the same magnitude but opposite sign
            if errorC + errorCc == 0:
                errCnt += 1     # count the number of times these ossiclations occur

            if errCnt < 10:
                print("     Fit " + str(n) + ", ErrorC = " + str(errorC))
            else:
                print("\n\n---- NOTE: FITTING PROCEDUCE DID NOT CONVERGE  ----\n\n")

            # iterator for printing purposes
            n += 1
        
        # apply ONLY the translation transformations to the original features so that the adjustments are made 
        # to the optimised feature positions
        for f in featsO[fn]:
            featsO[fn][f] -= translateNet[fn]

        # perform a final rotation on the fully translated features to create a SINGLE rotational transformation
        rotationAdjustment, featsFinalMod, errN, centre = rotatePoints(featsO, tol = 1e-8, plot = False)
        rotateNet[fn] = [rotationAdjustment[fn], centre[fn][0], centre[fn][1]]  # pass the rotational degree and the centre of rotations
        feats[fn] = featsFinalMod[fn]                      # re-assign the new feature positions to fit for

        refFeat = fn        # re-assign the re-Feat if aligning between slices

    # save the tif shapes, translation and rotation information
    dictToTxt(translateNet, src + "all.translated")
    dictToTxt(rotateNet, src + "all.rotated")

def transformSamples(segSamples, segInfo, dest, spec, size, saving):
    # this function takes the affine transformation information and applies it to the samples
    # Inputs:   (src), directories of the segmented samples
    #           (dest), directories to save the aligned samples
    #           (spec), sample being processed
    #           (saving), boolean whether to save new info
    # Outputs   (), saves an image of the tissue with the necessary padding to ensure all images are the same size and roughly aligned if saving is True

    def adjustPos(infoE, dest, spec, maxPos, translateNet, w, shapeR, t, centre = None):

        # this funciton adjusts the position of features in the txt files and resaves them
        # in the destination location
        # Inputs:   (s), directory of the specimens features to adjust
        #           (dest), directory to save new text file
        #           (spec), sample name
        #           (maxPos), field adjustment size
        #           (translateNet), translations of the sample
        #           (rotateNet), rotations of the sample
        #           (shapeR), the scale factor for the modified images
        #           (t), type of file
        #           (centre), position of the centre, if not given will be calculated from input features
        # Outputs:  (), saves the positions with the transformation 
        #           (centre), if the centre is not given then it is to be found from these calculations


        # adjust the positions based on the fitting process
        for f in infoE:
            infoE[f] = (infoE[f] - translateNet) * shapeR
        infoE = objectivePolar(w, centre, False, infoE) 

        # adjust the positions based on the whole image adjustment
        
        for f in infoE:
            infoE[f] += np.array(maxPos)  
        
        # save the image
        dictToTxt(infoE, dest + spec + "." + t)     # from the info in the txt file, rename
        return(infoE)
        

    # get all the segmented information that is to be transformed
    src = dictOfDirs(
        segments = glob(segSamples + spec + "*.tif"), 
        bound = glob(segInfo + spec + "*.bound"), 
        feat = glob(segInfo +  spec + "*.feat"), 
        segsection = glob(segInfo + spec + "*.segsection*"), 
        tifShapes = segInfo + "all.tifshape", 
        jpgShapes = segInfo + "all.jpgshape",
        translateNet = segInfo + "all.translated", 
        rotateNet = segInfo + "all.rotated")

    # Load the entire image
    fieldWhole = cv2.imread(src[spec]['segments'][0])

    # load the whole specimen info
    translateNet = txtToDict(src['all']['translateNet'][0])[0]
    tifShapes = txtToDict(src['all']['tifShapes'][0])[0]
    jpgShapes = txtToDict(src['all']['jpgShapes'][0])[0]
    rotateNet = txtToDict(src['all']['rotateNet'][0], float)[0]
    specInfo = {}

    # initialise the end position of the tif image to be cropped
    posE = 0

    # for every sample that is extracted from a multi-sample image, process
    for sample in range(len(src[spec]['feat'])):
        featA = txtToDict(src[spec]['feat'][sample])
        specInfo['feat'] = featA[0]
        specInfo['bound'] = txtToDict(src[spec]['bound'][sample])[0]
        for s in src[spec]['segsection']:
            specInfo['segsection' + s] = txtToDict(s)

        sample = nameFromPath(src[spec]['feat'][sample], 3)

        # get the size of jpeg version of the image, some tedious formatting changes to make into
        # an np.array
        jpegSize = np.array(featA[1]['shape'].replace("(", "").replace(")", "").replace(",", "").split()).astype(int)
        
        # get the shapes of the jpeg image and original tif and find the ratio fo their sizes
        shapeO = tifShapes[spec]
        shapeR = np.round((shapeO / jpegSize)[0], 3)

        # the start position is set as the previous end position and the 
        # end position is calculated based on the relative size of the jpeg image
        posS = posE
        posE += int(jpegSize[1] / (jpegSize[0]/shapeO[0]))

        # get the section of the image 
        field = fieldWhole.copy()[:, posS:posE, :]              
        fy, fx, fc = field.shape

        # get the measure of the amount of shift to create the 'platform' which all the images are saved to
        ss = (np.ceil(dictToArray(translateNet, int) * shapeR)).astype(int)     # scale up for the 40% reduction in tif2pdf
        maxSx = np.max(ss[:, 0])
        maxSy = np.max(ss[:, 1])
        minSx = np.min(ss[:, 0])
        minSy = np.min(ss[:, 1]) 
        maxPos = (maxSx, maxSy)

        # get the anlge and centre of rotation used to align the samples
        w = rotateNet[sample][0]
        centre = rotateNet[sample][1:] * shapeR

        # make destinate directory
        dirMaker(dest)

        # ---------- apply the transformations onto the images ----------

        # adjust the translations of each image and save the new images with the adjustment
        # for spec in src:

        # process for feats, bound and segsections. NOTE segsections not always present so 
        # error handling incorporated

        for t in specInfo:
            # adjust all the points
            specInfo[t] = adjustPos(specInfo[t], dest, sample, maxPos, translateNet[sample], w, shapeR, t, centre)

        # translate the image  
        
        # get the maximum dimensions of all the tif images (NOT the jpeg images)
        tsa = dictToArray(jpgShapes, int)

        mx, my, _ = (np.max(tsa, axis = 0) * shapeR).astype(int)

        # get the dims of the total field size to be created for all the images stored
        xF, yF, cF = (mx + maxSy - minSy, my + maxSx - minSx, 3)       # NOTE this will always be slightly larger than necessary because to work it    
                                                                        # out precisely I would have to know what the max displacement + size of the img
                                                                        # is... this is over-estimating the size needed but is much simpler
        xp = int(maxSx - translateNet[sample][0] * shapeR)
        yp = int(maxSy - translateNet[sample][1] * shapeR)

        newField = np.zeros([xF, yF, cF]).astype(np.uint8)      # empty matrix for ALL the images
        newField[yp:(yp+fy), xp:(xp+fx), :] += field


        # apply the rotational transformation to the image
        centre = centre + maxPos

        rot = cv2.getRotationMatrix2D(tuple(centre), -float(w), 1)
        warped = cv2.warpAffine(newField, rot, (xF, yF))

        print("done translation of " + sample)

        # NOTE change the inputs to a single large dictionary
        plotPoints(dest + sample + '_alignedAnnotatedUpdated.jpg', warped, centre, specInfo)

        # this takes a while so optional
        if saving:
            cv2.imwrite(dest + sample + '.tif', warped)                               # saves the adjusted image at full resolution 

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

def rotatePoints(feats, tol = 1e-6, plot = False):

    # get the rotations of each frame
    # Inputs:   (feats), dictionary of each feature
    # Outputs:  (rotationStore), affine rotation matrix to rotate the IMAGE --> NOTE 
    #                       rotating the image is NOT the same as the features and this doesn't 
    #                       quite work propertly for some reason...
    #           (featsmod), features after rotation

    segments = list(feats.keys())

    rotationStore = {}
    centreStore = {}
    featsMod = feats.copy()

    # store the angle of rotation
    rotationStore[segments[0]] = 0

    # store the centre of rotations
    centreStore[segments[0]] = np.array([0, 0])
    
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

        centre = findCentre(tarP)
        
        # get the shift needed and store
        res = minimize(objectivePolar, (-5.0), args=(None, True, tarP, refP), method = 'Nelder-Mead', tol = tol) # NOTE create bounds on the rotation
        refR = objectivePolar(res.x, centre, False, tar, refP)   # get the affine transformation used for the optimal rotation
        rotationStore[i] = float(res.x)

        err = res.fun

        # reassign this as the new feature
        featsMod[i] = refR
        centreStore[i] = centre
    
    if plot: denseMatrixViewer([dictToArray(tarP), dictToArray(refP)], True, 2)

    return(rotationStore, featsMod, err, centreStore)

def plotPoints(dir, imgO, cen, points):

    # plot circles on annotated points
    # Inputs:   (dir), either a directory (in which case load the image) or the numpy array of the image
    #           (imgO), image directory
    #           (cen), rotational centre
    #           (points), dictionary or array of points which refer to the co-ordinates on the image
    # Outputs:  (), saves downsampled jpg image with the points annotated

    # load the image
    if type(imgO) is str:
            imgO = cv2.imread(imgO)

    img = imgO.copy()
    
    si = 50

    # for each set of points add to the image
    for pf in points:

        point = points[pf]

        colour = [randint(0, 1)*255, randint(0, 1) * 255, 0]
        colour2 = [abs(c - 255) for c in colour]
        if type(point) is dict:
            point = dictToArray(point)

        for p in point:       # use the target keys in case there are features not common to the previous original 
            pos = tuple(np.round(p).astype(int))
            img = cv2.circle(img, pos, si, tuple(colour), int(si/2)) 
        
    # plot of the rotation as well using opposite colours
    cen = cen.astype(int)
    # img = cv2.circle(img, tuple(findCentre(points)), si, (0, 255, 0), si) 
    img = cv2.circle(img, tuple(cen), si, tuple(colour2), int(si/2)) 


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
    scale = 5

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
    m = 1
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

        pad = 2

        # find the max size a matrix would need to be to allow this vector to perform a 360º rotation
        ext = ceil(np.sqrt(np.sum((featPos)**2)))
        fN = featPos + ext + pad          # place the feature within the bounding area ext x ext
        tarB = np.zeros([(ext+pad)*2+1, (ext+pad)*2+1])
        tarB[fN[0], fN[1]] = 1

        # create the rotational transformation and apply
        rot = cv2.getRotationMatrix2D((ext+pad+1, ext+pad+1), np.float(w), 1)
        warp = cv2.warpAffine(tarB, rot, ((ext+pad)*2+1, (ext+pad)*2+1))

        # get the positions of the rotations
        tarPosWarp = np.array(np.where(warp > 0))                          
        tarPossible = np.stack([tarPosWarp[0, :], tarPosWarp[1, :]], axis = 1)     # array reshaped for consistency


        # get the non-zero values at the found positions
        #   NOTE the rotation causes the sum of a single point (ie 1) to be spread
        #   over several pixels. Find the spread pixel which has the greatest
        #   representation of the new co-ordinate and save that
        v = warp[tuple(tarPosWarp)]
        
        newfeatPos = tarPossible[np.argmax(v)] + centre - ext - pad
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
    dataSource = '/Volumes/Storage/H653A_11.3new/'
    dataSource = '/Volumes/USB/Testing1/'
    dataSource = '/Volumes/USB/H653/'
    dataSource = '/Volumes/USB/H653A_11.3/'
    dataSource = '/Volumes/USB/H673A_7.6/'


    # dataTrain = dataHome + 'FeatureID/'
    name = ''
    size = 3

    align(dataSource, name, size, True)
