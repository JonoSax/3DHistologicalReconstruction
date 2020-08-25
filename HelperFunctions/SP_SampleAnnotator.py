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
from multiprocessing import Process
if __name__.find("HelperFunctions") == -1:
    from Utilities import nameFromPath, dirMaker, txtToDict, dictToTxt, hist_match
else:
    from HelperFunctions.Utilities import nameFromPath, dirMaker, txtToDict, dictToTxt, hist_match


def featChangePoint(ref, tar, ts = 4):

    # this fuction brings up a gui which allows for a user to manually CHANGE
    # features on the images. This modifies the original .feat file
    # Inputs:   (ref, tar), reference and target name of the samples to change
    #           (ts), text size for plots
    # Outputs:  (matchRef, matchTar), updated ref and target features with new points added
    #           (), also saves the new ref and tar positions in the SAME location

    # get the dirs of the info
    imgdirs = dataSource + 'masked/'
    infodirs = dataSource + 'info/'

    imgrefdir = imgdirs + ref + ".png"
    imgtardir = imgdirs + tar + ".png"

    matchRefdir = infodirs + ref + ".feat"
    matchTardir = infodirs + tar + ".feat"
    
    matchRefO = txtToDict(matchRefdir)[0]
    matchTarO = txtToDict(matchTardir)[0]
    matchRef = matchRefO.copy()
    matchTar = matchTarO.copy()

    imgref = cv2.cvtColor(cv2.imread(imgrefdir), cv2.COLOR_BGR2RGB)
    imgtar = cv2.cvtColor(cv2.imread(imgtardir), cv2.COLOR_BGR2RGB)

    # automatically set the text size
    ts = imgref.shape[0]/1000


    # get all the features which both slices have
    refL = list(matchRef.keys())
    tarL = list(matchTar.keys())
    feat, c = np.unique(refL + tarL, return_counts=True)
    commonFeat = feat[np.where(c == 2)]

    # create the dictionary with ONLY the common features and their positions
    refCommon = {}
    tarCommon = {}

    for cf in commonFeat:
        refCommon[cf] = matchRef[cf]
        tarCommon[cf] = matchTar[cf]

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
    for feat in commonFeat:

        # get the pair of features to change
        featChange['ref'] = matchRef[feat]
        featChange['tar'] = matchTar[feat]

        for featC in featChange:

            featID = featChange[featC]
        
            # draw on the points
            imgCombineA = annotateImg(imgCombine.copy(), [refCommon, tarCommon], ts)

            if featC == 'tar':
                featID += np.array([ym, 0])

            # highlight the point which is being changed
            cv2.circle(imgCombineA, tuple(featID.astype(int)), int(ts*12), (0, 255, 0), int(ts*8))
            
            # get the x and y position from the feature
            y, x = roiselector(imgCombineA)
            # if the window is closed to skip selecting a feature, keep the feature
            if np.sum(x) * np.sum(y) == 0:
                yme = featID[0]
                xme = featID[1]
            else:
                yme = int(np.mean(y))
                xme = int(np.mean(x))

            # append reference and target information to the original list
            if featC == 'ref':
                refCommon[feat] = np.array((yme, xme))
            elif featC == "tar":
                tarCommon[feat] = np.array((yme, xme)) - np.array([ym, 0])

        # reassign the modified features to the master dictionary
        matchRef[feat] = refCommon[feat]
        matchTar[feat] = tarCommon[feat] 

    cv2.destroyAllWindows()

    # save the new manually added positions to the original location, REPLACING the 
    # information
    dictToTxt(matchRef, matchRefdir)
    dictToTxt(matchTar, matchTardir)

    return(matchRef, matchTar)

def featSelectArea(datahome, size, feats = 1, sample = 0, normalise = False):

    # this function brings up a gui which allows user to manually selection a 
    # roi on the image. This extracts samples from the aligned tissues and saves them

    segSections = datahome + str(size) + "/segSections/"

    for f in range(feats):
        dirMaker(segSections + "seg" + str(f) + "/")

    alignedSamples = datahome + str(size) + "/alignedSamples/"

    # get all the samples to be processed
    samples = glob(alignedSamples + "*.tif")

    # get the image to be used as the reference
    if type(sample) == int:
        refpath = samples[sample]
    elif type(sample) == str:
        refpath = glob(alignedSamples + sample + "*.tif")[0]
    
    # load the image
    try:
        img = tifi.imread(refpath)
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
        cv2.rectangle(img, (x[f][0], y[f][0]), (x[f][1], y[f][1]), (255, 255, 255), 40)
        cv2.rectangle(img, (x[f][0], y[f][0]), (x[f][1], y[f][1]), (0, 0, 0), 20)

    for s in samples[3:]:
        sectionExtract(segSections, feats, s, x, y, imgref)

    # NOTE only parallelise on the HPC
    '''
    # extract from all the samples the features
    jobs = {}
    for s in samples:
        name = nameFromPath(s)
    
        jobs[name] = Process(target = sectionExtract, args = (segSections, feats, s, x, y))
        jobs[name].start()

    for n in jobs:
        jobs[n].join()
    '''

def featSelectPoint(imgref, imgtar, matchRef, matchTar, feats = 5, ts = 4):

    # this fuction brings up a gui which allows for a user to manually select
    # features on the images. This contributes to making a .feat file
    # Inputs:   (nameref), either the path or the numpy array of the reference image
    #           (nametar), either the path or the numpy array of the target image
    #           (matchRef), any already identified features on the reference image
    #           (matchTar), any already identified features on the target image
    #           (feats), defaults to finding 5 features
    # Outputs:  (matchRef, matchTar), updated ref and target features with new points added

    # get the images with insufficient features
    if type(imgref) == str or type(imgtar) == str:
        imgref = cv2.imread(imgref)
        imgtar = cv2.imread(imgtar)

    ts = imgref.shape[0]/1000

    # add the annotations already on the image
    imgCombine = annotateImg([imgref, imgtar], [matchRef, matchTar], ts)

    n = int(len(matchRef) * 2)
    ym = int(imgCombine.shape[1]/2)
    for i in range(n, feats*2):

        # get the x and y position from the feature
        x, y = roiselector(imgCombine)
        xme = int(np.mean(x))
        yme = int(np.mean(y))

        # add the found points as marks
        imgCombine = cv2.circle(imgCombine, (xme, yme), int(ts*10), (255, 0, 0), int(ts*5))

        # append reference and target information to the original list
        if i%2 == 0:
            obj = "ref"
            matchRef.append(np.array((xme, yme)))
            print("Feat: " + str(matchRef[-1]))
        else:
            obj = "tar"
            matchTar.append(np.array((xme, yme)) - np.array([ym, 0]))
            print("Feat: " + str(matchTar[-1]))

        print(str(i) + " + " + obj)

        # add info to image
        feat = obj + " feat " + str(int(np.floor(i/2)))

        cv2.putText(imgCombine, feat, 
                tuple([xme, yme] + np.array([20, 0])),
                cv2.FONT_HERSHEY_SIMPLEX, ts, (255, 255, 255), int(ts*5))

        cv2.putText(imgCombine, feat, 
                tuple([xme, yme] + np.array([20, 0])),
                cv2.FONT_HERSHEY_SIMPLEX, ts, (0, 0, 0), int(ts*2.5))

    cv2.destroyAllWindows()

    return(matchRef, matchTar)

def sectionExtract(segSections, feats, s, x, y, ref = None):

    img = tifi.imread(s)
    name = nameFromPath(s, 3)
    sections = []

    # if a reference image is being used to normalise the images
    if (type(ref) is list) or (type(ref) is np.ndarray):
        img = hist_match(img, ref)

    for f in range(feats):

        print(name + " section " + str(f))
        segdir = segSections + "seg" + str(f) + "/"
        section = img[y[f][0]:y[f][1], x[f][0]:x[f][1] :]
        tifi.imwrite(segdir + name + ".tif", section)


def roiselector(img):

    # function which calls the gui and get the co-ordinates 
    # Inputs    (img), numpy array image of interest
    # Outputs   (xme, yme), x and y positions on the original image of the points selected
    # ensure that the image is scaled to a resolution which can be used in the sceen
    
    xc, yc, c = img.shape
    r = xc/yc
    size = 700
    scale = yc / (size / r)

    # perform a search over a reduced size area
    roi = cv2.selectROI("image", cv2.resize(img, (int(size / r), size)))

    # get the postions
    y = np.array([roi[1], roi[1] + roi[3]])
    x = np.array([roi[0], roi[0] + roi[2]])

    # scale the positions back to their original size
    y = (np.round(y * scale)).astype(int)
    x = (np.round(x * scale)).astype(int)

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
    dataSource = '/Volumes/Storage/H653A_11.3new/'
    dataSource = '/Volumes/USB/H673A_7.6/'
    dataSource = '/Volumes/USB/H710C_6.1/'


    size = 3

    featSelectArea(dataSource, size, 5, 0, True)

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