'''

This script contains functions which allow a user to select a ROI on a 
single/multiple samples

These functions in a variety of ways call the roiselector function which is a GUI 
to allow a user to select points. The functionionality of each funcitons is as follows:

    ChangePoint: allows a user to CHANGE the locations of found points corresponding 
    to the matched images and their .feat files

    SelectArea: allows a user to propogate a selected area through a stack of images
    (works with any stack but probs best to use it after aligning tissue)

    SelectPoint: allows a user to SELECT a single point. used to manually annotate
    images during the featfinding period

'''

import cv2
import numpy as np
# import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
import tifffile as tifi
from multiprocessing import Pool, cpu_count
from copy import deepcopy
from itertools import repeat
if __name__.find("HelperFunctions") == -1:
    from Utilities import *
else:
    from HelperFunctions.Utilities import *


# for each fitted pair, create an object storing their key information
class feature:
    def __init__(self, refP = None, tarP = None, dist = None, size = None, res = None):
        # the position of the match on the reference image
        self.refP = refP

        # the position of the match on the target image
        self.tarP = tarP

        # eucledian error of the difference in gradient fields
        self.dist = dist

        # the size of the feature
        self.size = size

        # the resolution index of the image that was processed
        self.res = res

    def __repr__(self):
        return repr((self.dist, self.refP, self.tarP, self.size, self.res))

def featChangePoint(dataSource, ref, tar, featureInfo = None, nopts = 5, ts = 4, title = "Matching"):

    # this fuction brings up a gui which allows for a user to manually CHANGE
    # features on the images. This modifies the original .feat file
    # Inputs:   (dataSource), the directory of where to get the info
    #           (ref, tar), reference and target name of the samples to change
    #           (nopts), number of point to change the annotations of, defaults 5
    #           (ts), text size for plots
    # Outputs:  (matchRef, matchTar), updated ref and target features with new points added
    #           (), also saves the new ref and tar positions in the SAME location

    # if modifying files
    if dataSource is None:
        imgref = ref
        imgtar = tar

        matchRefO = {}
        matchTarO = {}

        # if there are not features found, break
        if len(featureInfo) == 0:
            pass

        # if the info is in a list of objects
        elif type(featureInfo[0]) is not dict:
            for n, f in enumerate(featureInfo):
                matchRefO[n] = f.refP
                matchTarO[n] = f.tarP

        else:
            # if dictionaries are passed in as a list
            matchRefO = featureInfo[0]
            matchTarO = featureInfo[1]

    # if modifying arrays
    else: 

        # get the dirs of the info
        infodirs = dataSource + 'info/'

        # if doing it for the main images
        try:    
            imgdirs = dataSource + 'masked/'

            # get the images, doesn't m
            imgrefdir = glob(imgdirs + ref + ".*")[0]
            imgtardir = glob(imgdirs + tar + ".*")[0]

            imgref = cv2.imread(imgrefdir)
            imgtar = cv2.imread(imgtardir)

        # if doing it for the segSections
        except: 
            imgdirs = dataSource
            imgrefdir = glob(imgdirs + ref + ".*")[0]
            imgtardir = glob(imgdirs + tar + ".*")[0]

            imgref = cv2.imread(imgrefdir)
            imgtar = cv2.imread(imgtardir)

        matchRefdir = infodirs + ref + ".reffeat"
        matchTardir = infodirs + tar + ".tarfeat"

        matchRefO = txtToDict(matchRefdir, float)[0]
        matchTarO = txtToDict(matchTardir, float)[0]

    # automatically set the text size
    ts = imgref.shape[0]/1000

    _, commonFeat = uniqueKeys([matchRefO, matchTarO])

    # create the dictionary with ONLY the common features and their positions
    # for the number of features specified by nofts
    matchRef = {}
    matchTar = {}

    # select only up to the specified number of points to use
    if nopts < len(commonFeat):
        commonFeat = commonFeat[:nopts]

    # if there are less matched points than desired, add some fake ones
    # to move around later
    elif nopts > len(commonFeat):
        for i in range(nopts - len(commonFeat)):
            pt = "added_" + str(i)
            commonFeat.append(pt)
            matchRefO[pt] = np.array([0, 0])
            matchTarO[pt] = np.array([0, 0])

    for cf in commonFeat:
        matchRef[cf] = matchRefO[cf]
        matchTar[cf] = matchTarO[cf]

    # create a standard combined image
    imgs = [imgref, imgtar]

    # get the image dimensions
    imgshapes = []
    for img in imgs:
        imgshapes.append(np.array(img.shape))

    # create a max size field of all images
    xm, ym, cm = np.max(np.array(imgshapes), axis = 0)
    field = np.zeros((xm, ym, cm)).astype(np.uint8)

    # stack the images next to each other to create a combined image
    imgsStand = []
    for img in imgs:
        xr, yr, c = img.shape
        img_s = field.copy(); img_s[:xr, :yr, :] = img
        imgsStand.append(img_s)
    imgCombine = np.hstack(imgsStand)

    featChange = {}
    cv2.startWindowThread()
    for feat in commonFeat:

        # get the pair of features to change
        featChange['ref'] = matchRef[feat]
        featChange['tar'] = matchTar[feat]
        
        for featC in featChange:

            featID = featChange[featC]
        
            # draw on the points
            imgCombineA = annotateImg(imgCombine.copy(), [matchRef, matchTar], ts)

            if featC == 'tar':
                featID += np.array([ym, 0])

            # highlight the point which is being changed
            cv2.circle(imgCombineA, tuple(featID.astype(int)), int(ts*12), (0, 255, 0), int(ts*8))
            
            # get the x and y position from the feature
            y, x = roiselector(imgCombineA, title)

            # if the window is closed to skip selecting a feature, keep the feature
            if np.sum(x) * np.sum(y) == 0:
                yme = featID[0]
                xme = featID[1]
            else:
                yme = np.mean(y)
                xme = np.mean(x)

            # append reference and target information to the original list
            if featC == 'ref':
                matchRef[feat] = np.array((yme, xme))
            elif featC == "tar":
                matchTar[feat] = np.array((yme, xme)) - np.array([ym, 0])

    # save the new manually added positions to the original location, REPLACING the 
    # information

    # return the data in the same format as the input
    if dataSource is None:

        # if the input was an object, return an object
        featInfos = []
        if type(featureInfo[0]) is not dict:
            for f in matchRef:
                featInfos.append(feature(refP = matchRef[f], tarP = matchTar[f], dist = -1, size = 100, res = -1))
            
        else:
            featInfos = [matchRef, matchTar]

    else:
        dictToTxt(matchRef, matchRefdir, shape = imgref.shape, fit = False)
        dictToTxt(matchTar, matchTardir, shape = imgtar.shape, fit = False)
        
        # if the input was a list of dictionaries, return a list of dictionaries
        featInfos = [matchRef, matchTar]

    return(featInfos)

def featSelectArea(datahome, size, feats = 1, sample = 0, normalise = False):

    # this function brings up a gui which allows user to manually selection a 
    # roi on the image. This extracts samples from the aligned tissues and saves them

    cpuCount = int(cpu_count() * 0.75)
    segSections = datahome + str(size) + "/segSections/"

    serialised = False

    for f in range(feats):
        dirMaker(segSections + "seg" + str(f) + "/")

    alignedSamples = datahome + str(size) + "/alignedSamples/"

    # get all the samples to be processed
    samples = sorted(glob(alignedSamples + "*.tif"))

    # get the image to be used as the reference
    if type(sample) == int:
        refpath = samples[sample]
    elif type(sample) == str:
        refpath = glob(alignedSamples + sample + "*.tif")[0]
    
    # load the image
    try:
        img = cv2.cvtColor(tifi.imread(refpath), cv2.COLOR_BGR2RGB)
    except:
        img = cv2.imread(refpath)

    # create a small scale version of the image for color normalisation
    if normalise: imgref = cv2.resize(img, (int(img.shape[1] * 0.1), int(img.shape[0] * 0.1)))
    else: imgref = None

    # extract n feats from the target samples
    x = {}
    y = {}
    for f in range(feats):
        x[f], y[f] = roiselector(img)
        cv2.rectangle(img, (int(x[f][0]), int(y[f][0])), (int(x[f][1]), int(y[f][1])), (255, 255, 255), 40)
        cv2.rectangle(img, (int(x[f][0]), int(y[f][0])), (int(x[f][1]), int(y[f][1])), (0, 0, 0), 20)

    shapes = {}
    if serialised:
        for s in samples:
            name = nameFromPath(s)
            shapes[name] = sectionExtract(s, segSections, feats, x, y, imgref)

    else:
        with Pool(processes=cpuCount) as pool:
            shapes = pool.starmap(sectionExtract, zip(samples, repeat(segSections), repeat(feats), repeat(x), repeat(y), repeat(imgref)))


    # create a dictionary of all the tif shapes. they're all the same size, 
    # its just about ensuring the input into align is consistent
    for i in range(feats):
        imgShapes = {}
        for n, s in enumerate(samples):
            name = nameFromPath(s, 3)
            imgShapes[name] = shapes[n][i]

        dictToTxt(imgShapes, segSections + "seg" + str(i) + "/info/all.tifshape")

def sectionExtract(path, segSections, feats, x, y, ref = None):

    img = cv2.imread(path)
    name = nameFromPath(path, 3)
    sections = []

    segShapes = {}
    # if a reference image is being used to normalise the images
    if (type(ref) is list) or (type(ref) is np.ndarray):
        img = hist_match(img, ref)

    for f in range(feats):

        print(name + " section " + str(f))
        segdir = segSections + "seg" + str(f) + "/"
        section = img[int(y[f][0]):int(y[f][1]), int(x[f][0]):int(x[f][1]), :]
        cv2.imwrite(segdir + name + ".tif", section)
        segShapes[f] = section.shape

    return(segShapes)

def roiselector(img, title = "Matching"):

    # function which calls the gui and get the co-ordinates 
    # Inputs    (img), numpy array image of interest
    # Outputs   (xme, yme), x and y positions on the original image of the points selected
    # ensure that the image is scaled to a resolution which can be used in the sceen
    
    xc, yc, c = img.shape
    r = xc/yc
    # if the height is larger than the width
    if xc > yc/2:
        size = 700
        sizeY = int(size / r)
        sizeX = int(size)

    # if the width is larger than the height
    else:
        size = 1200
        sizeY = int(size) 
        sizeX = int(size * r)

    scale = yc / sizeY

    # perform a search over a reduced size area
    imgr = cv2.resize(img, (sizeY, sizeX))
    roi = cv2.selectROI(title, imgr, showCrosshair=True)

    # get the postions
    y = np.array([roi[1], roi[1] + roi[3]])
    x = np.array([roi[0], roi[0] + roi[2]])

    # scale the positions back to their original size
    y = y * scale
    x = x * scale

    return(x, y)

def annotateImg(imgs, info, ts):

    # this function takes an image and from a list of .feat dictionaries, draws
    # the position and information on and returns the images combined
    # Inputs:   (imgs), image type. if it is a list combine but if it is a single image just use as is
    #           (info), list of .feat dictionaries to annotate onto the images
    #           (ts), text size
    # Outputs:  (imgcombine), the combined image which has all the annotated features

    # if multiple images are being fed in then combine in a uniform array
    if type(imgs) is list:  
        # get the image dimensions
        imgshapes = []
        for img in imgs:
            imgshapes.append(np.array(img.shape))

        # create a max size field of all images
        xm, ym, cm = np.max(np.array(imgshapes), axis = 0)
        field = np.zeros((xm, ym, cm)).astype(np.uint8)

        # stack the images next to each other to create a combined image
        imgsStand = []
        for img in imgs:
            xr, yr, c = img.shape
            img_s = field.copy(); img_s[:xr, :yr, :] = img
            imgsStand.append(img_s)
        imgCombine = np.hstack(imgsStand)

    # if a single image is being used then do all the annotations on this unmodified
    else: 
        imgCombine = imgs
        xm, ym, cm = (np.array(imgCombine.shape) / len(info)).astype(int)

    for m, match in enumerate(info):

        for pos, r in enumerate(match):

            if type(match) is dict:
                r = match[r]

            # enusre that the co-ordinate is in the right format and position 
            if type(r) is np.ndarray: r = tuple((r.astype(int)) + np.array([int(ym * m), 0]))
            else: r = tuple(np.array(r).astype(int) + np.array([int(ym * m), 0]))
            # if type(t) is np.ndarray: t = tuple(t.astype(int) 

            # add the found points as marks
            cv2.circle(imgCombine, r, int(ts*10), (255, 0, 0), int(ts*5))

            # add point info to the image 
            cv2.putText(imgCombine, str(pos), 
                    tuple(r + np.array([20, 0])),
                    cv2.FONT_HERSHEY_SIMPLEX, ts, (255, 255, 255), int(ts*5))

            cv2.putText(imgCombine, str(pos), 
                    tuple(r + np.array([20, 0])),
                    cv2.FONT_HERSHEY_SIMPLEX, ts, (0, 0, 0), int(ts*2.5))

    return(imgCombine)

if __name__ == "__main__":

    '''
    dataSource = '/Volumes/USB/H653/3/masked/'
    nameref = 'H653_01A_0.jpg'
    nametar = 'H653_02A_0.jpg'
    matchRef = {}
    matchTar = {}

    matchRef = {
    'feat_0':np.array([594, 347]),
    'feat_1':np.array([ 254, 1002]),
    'feat_2':np.array([ 527, 1163]),
    'feat_3':np.array([322, 262])
    }

    matchTar = {
    'feat_2':np.array([ 533, 1131]),
    'feat_3':np.array([294, 239]),
    'feat_4':np.array([287, 899]),
    'feat_5':np.array([608, 255])
    }

    featSelectPoint(nameref, nametar, matchRef, matchTar)
    '''

    dataSource = '/Volumes/USB/H653/'
    dataSource = '/Volumes/USB/H673A_7.6/'
    dataSource = '/Volumes/Storage/H653A_11.3new/'
    dataSource = '/Volumes/USB/H653A_11.3/'
    dataSource = '/Volumes/USB/H710C_6.1/'
    dataSource = '/Volumes/USB/H671A_18.5/'



    size = 3

    featChangePoint(r, f, ts = 4)
    # featSelectArea(dataSource, size, 2, 0, False)

    '''
    'H710C_289A+B_0',
    'H710C_304A+B_0',
    'H710C_308A+B_0',
    'H710C_308C_0',  

    ref = [
        'H710C_311C_0'
    ]

    'H710C_289A+B_1',
    'H710C_304A+B_1',
    'H710C_308A+B_1',
    'H710C_309A+B_0',

    tar = [
        'H710C_312A+B_0'
    ]

    for r, f in zip(ref, tar):
        featChangePoint(r, f, ts = 4)
    '''  