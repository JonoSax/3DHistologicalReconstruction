import multiprocessing
from glob import glob
from itertools import repeat
from time import time
from copy import copy as cp

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.interpolate import splev, splprep
from skimage.registration import phase_cross_correlation as pcc
from tensorflow_addons.image import sparse_image_warp
from scipy.optimize import minimize as minimise

from HelperFunctions.SP_AlignSamples import aligner
from HelperFunctions.SP_FeatureFinder import allFeatSearch
from HelperFunctions.Utilities import *


# for each fitted pair, create an object storing their key information
class feature:
    def __init__(self, ID = None, Samp = None, refP = None, tarP = None, dist = None):

        # the feature number 
        self.ID = ID

        # sample which the reference feature is on
        self.Samp = Samp

        # the position of the match on the reference image
        self.refP = refP

        # the position of the match on the target image
        self.tarP = tarP


    def __repr__(self):
            return repr((self.ID, self.refP, self.tarP))

def nonRigidAlign(dirHome, size, cpuNo):

    home = dirHome + str(size)

    imgsrc = home + "/alignedSamples/"
    destRigidAlign = home + "/RealignedSamples/"
    dirfeats = home + "/infoNL/"
    destNLALign = home + "/NLAlignedSamples/"
    destFeatSections = home + "/FeatureSections/"
    
    dirMaker(destRigidAlign)
    dirMaker(dirfeats)
    dirMaker(destNLALign)
    dirMaker(destFeatSections)

    sect = 250           # the proporptional area of the image to search for features
    dist = 30           # distance between the sift features
    featsMin = 10       # min number of samples a feature must pass through for use to NL deform

    # Find the continuous features throught the samples
    contFeatFinder(imgsrc, dirfeats, destRigidAlign, cpuNo = cpuNo, sz = sect, dist = dist)
    
    # perform a rigid alignment on the tracked features
    aligner(imgsrc, dirfeats, destRigidAlign, cpuNo = cpuNo, errorThreshold=100)
    
    # Variables for the feature selectino
    featsMax = 20       # max number of features per sample which meet criteria to be used
    distFeats = 400     # distance (pixels) between the final features
    selectCriteria = 'smooth'       # criteria used to select features (either based on the *smooth*, ie how smooth the 
                                    # the feature is at it moves throught the specimen or *length*, ie priorities longer features)

    # with all the features found, find their trajectory and adjust to create continuity 
    # between samples
    featShaper(destRigidAlign, destFeatSections, featsMin = featsMin, dist = distFeats, maxfeat = featsMax, selectCriteria = selectCriteria, plot = True)
    
    # extract the feature sections
    allFeatExtractor(destRigidAlign, destFeatSections, prefix = "png", scl = 1, sz = sect)

    # extract sections and deform the downsampled images
    nonRigidDeform(destRigidAlign, destNLALign, destFeatSections, prefix = "png")

    # extract the feature sections from the non-linear samples (in their TRUE positions)
    allFeatExtractor(destNLALign, destFeatSections, prefix = "png", scl = 1, sz = sect, realPos=True)

    # extract sections and deform the FULL SCALE images (png is downsampled 
    # by a factor of 0.2)
    nonRigidDeform(destRigidAlign, destNLALign, destFeatSections, scl = 5, prefix = "tif")

def contFeatFinder(imgsrc, destFeat, destImg = None, cpuNo = False, sz = 100, dist = 20, plotting = False):

    '''
    This function takes images and finds features that are continuous
    between the samples

    Inputs:\n
    (imgs), list of directories of the images to be processed in sequential order for feature identifying\n
    (destImg), where to save the image from the plotting
    (cpuNo), number of cores to parallelise the task.\n
    (sz), the equivalent area which a continuous feature will be searched for if this many tiles were created on the section\n
    (dist), minimum pixel distance between detected features\n
    (connect), the final number of features that are tracked to return. These will be the most connected features available\n

    Outputs: \n
    (), saves the feature information. The number of feature paths that are saved is 
    based on the featsMax, where that is the minimum number of features found but at 
    least that many features exist through the most linked connections
    '''

    # ALWAYS use the downscaled version of the images to find continuity 
    # of features
    imgs = sorted(glob(imgsrc + "*.png"))[:3]

    # print(np.where(np.array(nameFromPath(imgs, 3)) == "H710C_364A+B_1"))

    # intialise objects to store and track info
    matchedInfo = []
    allMatchedInfo = {}
    featNo = 0
    refFeats = {}
    continuedFeatures = []

    # sequantially identify new previous features in each sample
    for sampleNo, (refPath, tarPath) in enumerate(zip(imgs[:-1], imgs[1:])):

        # load the images 
        refImg = cv2.imread(refPath)
        refName = nameFromPath(refPath, 3)
        tarName = nameFromPath(tarPath, 3)
        tarImg = cv2.imread(tarPath, 3)

        print(str(sampleNo) + "/" + str(len(imgs)-1) + ", Matching " + tarName + " to " + refName)

        # ---- find new features between the samples ----

        # find spatially coherent sift features between corresponding images
        scales = [0.2, 0.4, 0.6, 0.8, 1]
        startAllFeatSearch = time()
        matchedInfo = allFeatSearch(refImg, tarImg, dist = dist, \
            cpuNo = True, gridNo=3, tol = 0.1, featMin = 50, \
                scales = scales, maxFeats = 150, \
                    name_ref = refName, name_tar = tarName)[0]
        allFeatSearchTime = time() - startAllFeatSearch
        print("     allFeatSearch = " + str(allFeatSearchTime))

        imgCombine = nameFeatures(refImg, tarImg, matchedInfo, scales, combine = True, txtsz=0)
        cv2.imwrite(destFeat + tarName + "-->" + refName + "_raw.png", imgCombine)

        # featMatchimg = nameFeatures(refImg, tarImg, matchedInfo, combine = True)
        # plt.imshow(featMatchimg); plt.show()

        # ---- find the position of the feature in the next slice based on the previous informaiton ----
        startFeatMatch = time()
        '''
        confirmInfo = []
        for m in continuedFeatures + matchedInfo:
            confirmInfo.append(featMatching(m, tarImg, refImg, sz = sz))
        '''

        # parallelsiation seems to take longer...
        if cpuNo == False:
            confirmInfo = []
            for m in continuedFeatures + matchedInfo:
                confirmInfo.append(featMatching(m, tarImg, refImg, sz = sz))
        else:
            # Using Pool
            '''
            with multiprocessing.Pool(processes=cpuNo) as pool:
                confirmInfo = pool.starmap(featMatching, zip(matchedInfo, repeat(tarImg), repeat(refImg), repeat(sz)))
            '''

            # Using Process
            job = []
            qs = {}
            for n, m in enumerate(continuedFeatures + matchedInfo):
                qs[n] = multiprocessing.Queue()
                j = multiprocessing.Process(target=featMatching, args = (m, tarImg, refImg, sz, qs[n]))
                job.append(j)
                j.start()
            for j in job:
                j.join()

            confirmInfo = []
            for q in qs:
                confirmInfo.append(qs[q].get())
        
        featMatchTime = time() - startFeatMatch
        print("     featMatchMaker = " + str(featMatchTime))
        
        # unravel the list of features produced by the continuous features
        # NOTE this is so that the info capturing can work the same with the 
        # multiprocessing and serialisation
        confirmInfos = []
        for info in confirmInfo:
            if info is None:
                continue
            for i in info:
                if i is None:
                    continue
                confirmInfos.append(i)

        # ensure that the features found are still spaitally cohesive
        startMatchMaker = time()
        confirmedFeatures = matchMaker(confirmInfos, dist = dist, tol = 1, cpuNo = True, distCheck=True, spawnPoints=10, angThr=20, distTrh=0.2)
        # confirmedFeatures = confirmInfos.copy()
        matchMakerTime = time() - startMatchMaker
        print("     matchMaker = " + str(matchMakerTime))

        imgCombine = nameFeatures(refImg, tarImg, confirmedFeatures, scales, combine = True, txtsz=0.5)
        cv2.imwrite(destFeat + tarName + "-->" + refName + "_processed.png", imgCombine)
        # featMatchimg = nameFeatures(refImg, tarImg, continuedFeatures, combine = True)
        # plt.imshow(featMatchimg); plt.show()

        # ----- identify the feature ID -------

        # ensure that each feature is identified
        continuedFeatures = []
        for c in confirmedFeatures:
            if c.ID is None:
                # keep track of the feature ID
                c.ID = featNo
                allMatchedInfo[c.ID] = {}
                featNo += 1
            c.Samp = sampleNo
            allMatchedInfo[c.ID][sampleNo] = c

            # create the continued features
            cont = cp(c)
            cont.refP = cont.tarP
            cont.tarP = None
            continuedFeatures.append(cont)

        # create feature dictionaries per sample
        # NOTE try and make this something which saves every iteration
        tarFeats = {}
        ID = []
        for c in confirmedFeatures:
            ID.append(c.ID)

        sortID = np.argsort(ID)

        for s in sortID:
            c = confirmedFeatures[s]
            refFeats[int(c.ID)] = c.refP
            tarFeats[int(c.ID)] = c.tarP

        dictToTxt(refFeats, destFeat + refName + ".reffeat", fit = False, shape = refImg.shape)
        dictToTxt(tarFeats, destFeat + tarName + ".tarfeat", fit = False, shape = tarImg.shape)
        
        # re-assign the target features as reference features
        refFeats = tarFeats

        print("     feats found = " + str(len(confirmedFeatures)) + "/" + str(len(confirmInfos)) + "\n")

    # arrange the data in a way so that it can be plotted in 3D
    samples = len(imgs)
    
    for n in range(1,samples):
        featureNo = len(np.where(np.array([len(allMatchedInfo[mi]) for mi in allMatchedInfo]) == n)[0])
        if featureNo > 0:
            print("Number of " + str(1+n) + "/" + str(samples) + " linked features = " + str(featureNo))

    # create a panda data frame of all the features found for plotting
    df = dictToDF(allMatchedInfo, ["X", "Y", "Zs", "ID"])    

    if plotting:
        # px.line_3d(df, x="X", y="Y", z="Zs", color="ID", title = "All features, unaligned").show()
        plotFeatureProgress(df, imgs, destImg + 'CombinedRough.jpg', sz, 2)

def featShaper(diralign, dirSectdest, featsMin = 5, dist = 100, maxfeat = 20, selectCriteria = "smooth", plot = False):
    '''
    This function modifies the identified features to create continuity of features

        Inputs:\n
    (df), data frame containing the features
    '''

    print("\n--- Creating continuity of features ---")

    # get the new dictionaries, load them into a pandas dataframe
    refFeats = sorted(glob(diralign + "feats/*.reffeat"))
    tarFeats = sorted(glob(diralign + "feats/*.tarfeat"))
    names = nameFromPath(refFeats + tarFeats, 3, unique = True)
    infopds = []

    # read in the ref and tar files and concat into a single padas data frame
    # NOTE beacuse the ref and tarfeatures are the same for all the samples
    # it just has to iteratue through the samples, not the ref/tar feat files
    # NOTE I also know that you can save a dataframe as a CSV but this needs to 
    # have ref and tar files for the aligner and it seems silly to store the 
    # data twice for the sake of reading it in slightly faster.... 
    for f in refFeats + [tarFeats[-1]]:
        info = txtToDict(f, float)[0]
        infopd = pd.DataFrame.from_dict(info, orient = 'index', columns = ['X', 'Y'])
        infopd['Zs'] = names.index(nameFromPath(f, 3))       # add the sample number
        infopd['ID'] = np.array(list(info.keys())).astype(int)
        infopds.append(infopd)
        
    # combine into a single df
    dfAll = pd.concat(infopds)

    # only get the features which meet the minimum sample pass threshold
    featCounts = np.c_[np.unique(dfAll.ID, return_counts = True)]
    featCheck = featCounts[np.where(featCounts[:, 1] > featsMin)]
    dfAllCont = pd.concat([dfAll[dfAll["ID"] == f] for f in featCheck[:, 0]])

    if plot:    
        # px.line_3d(dfAllCont, x="X", y="Y", z="Zs", color="ID", title = "ALL raw features with min continuity").show()
        
        imgs = sorted(glob(diralign + "*.png"))[:4]

        # plotFeatureProgress(dfAllCont[dfAllCont["Zs"] < 4], imgs, diralign + 'CombinedRough.jpg', 1, 30, 2)

    # smooth the features
    featsSm = smoothFeatures(dfAllCont, 1e3, zAxis = "Zs")

    # select the best features
    dfSelectR, dfSelectSM, targetIDs = featureSelector(dfAllCont, featsSm, featsMin = featsMin, dist = dist, maxfeat = maxfeat, cond = selectCriteria)

    plotFeatureProgress(dfSelectR[dfSelectR["Zs"] < 4], imgs, diralign + 'CombinedRough.jpg', 1, 35, 2)

    # fix the features for the missing samples
    dfSelectRFix = fixFeatures(dfSelectR, regionOfPath(diralign, 2))
    dfSelectSMFix = fixFeatures(dfSelectSM, regionOfPath(diralign, 2))

    # re-smooth the fixed features
    dfSelectSMFix2 = smoothFeatures(dfSelectSMFix, smooth = 1e3)

    # 3D plot the smoothed and rough selected features 
    if plot:    px.line_3d(dfSelectRFix, x="X", y="Y", z="Z", color="ID", title = "Raw selected features " + selectCriteria).show()
    # px.line_3d(dfSelectSMFix, x="X", y="Y", z="Z", color="ID", title = "Smoothed selected + fix features").show()
    if plot:    px.line_3d(dfSelectSMFix2, x="X", y="Y", z="Z", color="ID", title = "All Smoothed selected features " + selectCriteria).show()    
    
    # save the data frames as csv files
    dfAll.to_csv(dirSectdest + "all.csv")
    dfAllCont.to_csv(dirSectdest + "rawFeatures.csv")
    featsSm.to_csv(dirSectdest + "smoothFixFeatures.csv")
    dfSelectRFix.to_csv(dirSectdest + "rawFixFeatures.csv")
    dfSelectSMFix2.to_csv(dirSectdest + "smoothFixFeatures.csv")

def allFeatExtractor(imgSrc, dirSectdest, prefix, scl = 1, sz = 0, realPos = False):
    # serialised feature extraction 
    # NOTE this has to be serialised so that the right reference image is used
    
    print("\n--- Extracting of " + prefix + " features ---\n")

    # get the image paths
    imgs = sorted(glob(imgSrc + "*" + prefix))

    # set the destination of the feature sections based on the prefix
    LSectDir = dirSectdest + "linearSect_" + prefix + "_" + str(realPos) + "/"

    # trajectories which are based on fixed samples are "Z" and raw samples are "Zs"
    if realPos:
        zSamp = "Z"
    else:
        zSamp = "Zs"

    if realPos:
        dfRawCont = pd.read_csv(dirSectdest + "smoothFixFeatures.csv")
    else:
        dfRawCont = pd.read_csv(dirSectdest + "rawFeatures.csv")
    
    for n, img in enumerate(imgs):
        print(str(n) + "/" + str(len(imgs)))
        sampdf = dfRawCont[dfRawCont[zSamp] == n].copy()
        sampdf.X *= scl; sampdf.Y *= scl        # scale the positions based on image size
        featExtractor(LSectDir, img, sampdf, dfRawCont, sz, zSamp, prefix = prefix, realPos = realPos)

def nonRigidDeform(diralign, dirNLdest, dirSectdest, scl = 1, prefix = "png"):

    '''
    This transforms the images based on the feature transforms

    Inputs:   
        (diralign), directory of all the aligned info and images
        (dirNLdest), path to save the NL deformed info
        (dirSectdest), path to save the feature sections
        (scl), resolution scale factor
        (prefix), image type used
    
    Outputs:  
        (), warp the images and save
    '''

    print("\n--- NL deformation of " + prefix + " images ---\n")

    dfSelectRFix = pd.read_csv(dirSectdest + "rawFixFeatures.csv")
    dfSelectSMFix2 = pd.read_csv(dirSectdest + "smoothFixFeatures.csv")

    # get the image paths
    imgs = sorted(glob(diralign + "*" + prefix))
    
    # create the dictionary which relates the real Z number to the sample image available
    key = np.c_[np.unique(dfSelectRFix.Z), [np.unique(dfSelectRFix[dfSelectRFix["Z"] == f].Zs)[0] for f in np.unique(dfSelectRFix.Z)]].astype(int)

    # Warp images to create smoothened feature trajectories
    # NOTE the sparse image warp is already highly parallelised so not MP functions

    for Z, Zs in key:
        imgPath = imgs[Zs]
        ImageWarp(Z, imgPath, dfSelectRFix, dfSelectSMFix2, dirNLdest, border = 5, smoother = 0, order = 1, annotate = False, scl = scl)

    '''
    # get the feature extraction from the modified images
    imgsMod = sorted(glob(dirNLdest + "*.png"))

    # for all the NL features SELECTED, extract the feature sections
    for n, img in enumerate(imgsMod):
        sampdf = dfSelectSMFix2[dfSelectSMFix2["Z"] == n]
        featExtractor(NLSectDir, img, sampdf, sz, zSamp = "Z")
    '''
    # plotFeatureProgress([dfSelectR, dfSelectSM], imgsMod, dirNLdest + 'CombinedSmooth.jpg', sz, [3])
    
def featMatching(m, tarImg, refImg, sz = 100, q = None):

    # Identifies the location of a visually similar feature in the next slice
    # Inputs:   (m), feature object of the previous point
    #           (tarImg), image of the next target sample
    #           (refImg), image of the current target sample
    #           (prevImg), image of the previous reference sample
    #           (sz), the proporptional area to analyse
    # Outputs:  (featureInfo), feature object of feature (if identified
    #               in the target image)

    # from the previous sample, use the last known positions as an anitial reference point 
    # to identify the area of the new feature

    x, y, c = tarImg.shape
    tarPAll = []

    # create the new feature object
    featureInfo = cp(m)
    featureInfo.tarP = None
    featureInfo.dist = 0

    l = tile(sz, x, y)/2
    refSect = getSect(refImg, featureInfo.refP, l)

    # if this is a continued feature where there is no target feature then 
    # use the reference feature to spawn the position
    if m.tarP is None:
        tarP = m.refP.copy()

    # if this is a featMatchMaker feature where there is an estimated position of 
    # the target point, use this
    else:
        tarP = m.tarP.copy()

    n = 0
    errorStore = np.inf
    # shiftAll = []
    # errorStoreAll = []
    while True:

        # get the feature sections
        tarSect = getSect(tarImg, tarP, l)

        # if the sections contain no information, break
        if (tarSect == 0).all() or (refSect == 0).all():
            break
        n += 1

        # this loop essentially keeps looking for features until it is "stable"
        # if a "stable" feature isn't found (ie little movement is needed to confirm) 
        # the feature) then eventually the position will tend towards the edges and fail
        # apply conditions: section has informaiton, ref and tar and the same size and threshold area is not empty
        if tarSect.size == 0 or \
            tarSect.shape != refSect.shape or \
                (np.sum((tarSect == 0) * 1) / tarSect.size) > 0.6 or n > 10:
            tarP = None
            break
        else:
            shift, error, _ = pcc(refSect, tarSect, upsample_factor=5)
            # tarP -= np.flip(shift)

            # perform shifting until the position of the image with the minimal error is found
            if error - errorStore > 0:
                # print("Shift = " + str(shiftAccum) + " error = " + str(error))
                break
            tarP -= np.flip(shift)
            # shiftAll.append(shift)
            # errorStoreAll.append(error)
            errorStore = error
    
    '''
    if (abs(np.array(shiftAll))>30).any():#  and n > 3:
        # plotting the change in the target feature position as the PCC optimises
        refSect = getSect(refImg, m.refP, l, True)
        refAll = (getSect(refImg, m.refP, l * 2, True) * 0.5).astype(np.uint8)
        sx = int(np.round(l)); ex = int(np.round(l+2*l))
        refAllC = refAll.copy()
        refAllC[sx:ex, sx:ex] = refSect
        # plt.imshow(np.hstack([refAll, refAllC]));plt.show()

        tarAll = (getSect(tarImg, tarP, l * 2, True) * 0.5).astype(np.uint8)
    
        try:
            tarP = m.tarP.copy()
        except:
            tarP = m.refP.copy()

        for n, s in enumerate(shiftAll):
            x, y = s + l
            tarSect = getSect(tarImg, tarP, l, True)
            tarAllC = tarAll.copy()
            sx = int(np.round(x)); ex = int(np.round(x) + np.round(2*l))
            sy = int(np.round(y)); ey = int(np.round(y) + np.round(2*l))
            tarAllC[sx:ex, sy:ey] = tarSect
            shift, error, _ = pcc(refSect, tarSect, upsample_factor = 5)
            tarP -= np.flip(s)
            print(error)
            print(shift)
            # plt.imshow(np.hstack([refAllC, tarAllC])); plt.show()
            cv2.imwrite("/Users/jonathanreshef/Downloads/tarSectAll"+str(n)+".png", tarAllC)
    '''


    # save specific feature for debugging
    if tarP is not None and False:#and np.sum(np.abs(np.diff(tarAccum, axis = 0))) > 50 :#m.ID == -1:

        print("ShiftAll = " + str(shift))
        print("ID = " + str(m.ID))

        '''
        a, b = moveImg(refSect, tarSectOrig, shift)
        ax1 = plt.subplot(2, 2, 1)
        ax2 = plt.subplot(2, 2, 2)
        ax3 = plt.subplot(2, 2, 3)
        ax4 = plt.subplot(2, 2, 4) 

        ax1.imshow(tarSectOrig)
        ax2.imshow(tarSect)
        ax3.imshow(b)
        ax4.imshow(refSect)
        plt.show()   
        '''
    
    # if all the points passed the conditions, set the target position
    if tarP is not None:
        featureInfo.tarP = tarP # np.mean(tarPAll, axis = 0)
    else:
        featureInfo = None

    '''
    # compare all the images used
    tarImgSectN = getSect(tarImg, featureInfo.tarP, sz)
    plt.imshow(np.hstack([tarImgSectN, tarSect, prevSectBW[0], prevSectBW[1]]), cmap = 'gray'); plt.show()
    '''

    if q == None:
        return([featureInfo])
    else:
        q.put([featureInfo])

def featureSelector(dfR, dfSm, featsMin, dist = 0, maxfeat = np.inf, cond = 'len'):

    '''
    This funciton returns features which match distance, length through 
    the sample and error between the smoothed and raw positions criteria.

    Samples which pass through the most samples are first prioritised
    
    Of these samples, the features which have the lowest errors between the 
    smoothed and raw positions are prioritised

    Each sample is then individually assessed againist all previously found 
    samples to ensure they all meet the distance critiera

    Only from the sample they were found in onwards is the feature propogated

    Inputs:\n
    (dfR), pandas data frame of the raw feature positions\n
    (dfSm), pandas data frame of the smoothed feature positions
    (featsMin), minimum number of samples a feature has to pass through to be used\n
    (dist), minimum distance between features\n
    (maxfeat) max number of features to be returned

    Outputs:\n
    (dfNew), pandas data frame of the features which meet the requirements 
    (targetIDs), the feature IDs that meet the criteria
    '''
    
    # set the maximum iterations
    sampleMax = np.max(dfR["Zs"])
    s = 0
    
    # initialise the searching positions
    featPos = []
    featPosID = []
    featAll = []
    featPosID.append(-1)
    featPos.append(np.array([-1000, -1000]))
    pdAllR = []
    pdAllSm = []
    while s < sampleMax:

        extraLen = featsMin - (featsMin - (sampleMax - s)) * ((sampleMax - s) < featsMin)

        # get all the feature positions on the sample
        sampsdfBoth = []
        n = 0
        # ensure that there are at least 3 features which are continued.
        # if that condition is not met then reduce the extraLen needed
        while len(sampsdfBoth) < 3 and n < extraLen:
            sampdf = dfR[dfR["Zs"] == s + extraLen]

            # create a data frame which contains the features in the current sample as well
            # as the samples further on in the specimen 
            sampHere, sampExtra = dfR[dfR["Zs"] == s + extraLen - n], dfR[dfR["Zs"] == s]
            sampsdfBoth = pd.merge(sampHere, sampExtra, left_on = "ID", right_on = "ID")

            # create a DF which contains all the features which pass through sample s
            try:    
                dfSamp = pd.concat([dfR[dfR["ID"] == f] for f in np.unique(sampsdfBoth.ID)])
                # get the feature IDs and their REMAINING lengths from the current sample
                _, featLen = np.unique(dfSamp[dfSamp["Zs"] >= s + extraLen - n].ID, return_counts = True)

            except: pass
            n += 1


        # calculate the average error per point between the smoothed and raw features
        errorStore = []
        errorStorem = []
        for i in sampsdfBoth.ID:
            rawFeat = np.c_[dfR[dfR["ID"] == i].X, dfR[dfR["ID"] == i].Y]
            smFeat = np.c_[dfSm[dfSm["ID"] == i].X, dfSm[dfSm["ID"] == i].Y]
            error = np.sum((rawFeat - smFeat)**2)/len(smFeat)
            errorStore.append(error)        # get the error of the entire feature
            
        # creat a dataframe with the error and length info of the features at the current sample
        featInfo = pd.DataFrame({'ID': sampsdfBoth.ID, 'X': sampsdfBoth.X_x, 'Y': sampsdfBoth.Y_x, 'error': errorStore, 'len': featLen})

        # sort the data first by length (ie longer values priortised) then by smoothness)
        # featInfoSorted = featInfo.sort_values(by = ['error', 'len'], ascending = [True, False])
        if cond.find('l') > -1: featInfoSorted = featInfo.sort_values(by = ['len', 'error'], ascending = [False, True])
        elif cond.find('s') > -1:   featInfoSorted = featInfo.sort_values(by = ['error', 'len'], ascending = [True, False])

        # evaluate each feature in order of the distance it travels through the samples
        for idF, x, y, err, lenF in featInfoSorted.values:
            si = np.array([x, y])

            # for each feature position, check if it meets the distance criteria and that it
            # also has not been found yet, then append the information
            if (np.sqrt(np.sum((si - featPos)**2, axis = 1)) > dist).all() and len(np.where(featAll == idF)[0]) == 0:
                featPosID.append(idF)   # keep a sample by sample track of features being used
                featAll.append(idF)     # keep a total track of all features being used
                featPos.append(si)      # add in the new feature which meets the criteria

                # add the features from the current sample
                pdAllR.append(dfR[dfR["ID"]==idF][dfR[dfR["ID"]==idF]["Zs"] >= s])
                pdAllSm.append(dfSm[dfSm["ID"]==idF][dfSm[dfSm["ID"]==idF]["Zs"] >= s])

            # remove the initialising points
            if featPosID[0] == -1:
                del featPosID[0]
                del featPos[0]

            if len(featPosID) == maxfeat:
                break

        # create the new DF which contains the features which meet the criteria
        dfNew = pd.concat([dfSm[dfSm["ID"] == f] for f in featPosID])

        # find out how far the minimum feature goes for
        if len(featPosID) < maxfeat:
            # if there are less than the max featnumber then search the next sample
            s += 1
        else:
            # if max features have been found, skip to the sample where the earliest 
            # possible features finishes
            s += np.min([np.max((dfNew[dfNew["ID"] == i]).Zs)-s+1 for i in featPosID])

        # re-initialise all the features which are longer than the 
        # shortest feature
        featPosID = list(np.unique(dfNew[dfNew["Zs"] >= s].ID))
        featPos = list(np.c_[dfNew[dfNew["Zs"] == s].X, dfNew[dfNew["Zs"] == s].Y])
        if len(featPosID) == 0:
            featPosID = []
            featPosID.append(-1)
            featPos = []
            featPos.append(np.array([-1000, -1000]))

    # collate the list of the final features
    finalFeats = np.unique(featAll)

    # get the data frame of all the features found
    dfFinalR = pd.concat(pdAllR)
    dfFinalSm = pd.concat(pdAllSm)

    return(dfFinalR, dfFinalSm, finalFeats)

def ImageWarp(s, imgpath, dfRaw, dfNew, dest, sz = 100, smoother = 0, border = 5, order = 2, annotate = False, scl = 1):

    # perform the non-rigid warp
    # Inputs:   (s), number of the sample
    #           (imgpath), directory of the image
    #           (dfRaw), raw data frame of the feature info
    #           (dfNew), data frame of the smoothed feature info
    #           (dest), directory to save modified image
    #           (scl), rescaling value for the image to match the feature postiions
    # Outputs:  (imgMod), the numpy array of the warped image
    #           (imgFlow), the flow field which was produced to make the 
    #               warped image

    # NOTE I should really add the corner positions for all 
    # samples so that the warping is true, rather than a distorted movement
    # ie create bounds on the images to deform

    # ensure the naming convention is correct 
    prefix = imgpath.split(".")[-1]       # get image prefix
    samp = str(s)
    while len(samp) < 4:
        samp = "0" + samp

    name = nameFromPath(imgpath, 1) + "_" + str(samp)
    print("Warping " +  name)

    # load the correct ID images
    if prefix == "tif":
        img = tifi.imread(imgpath)
    else:
        img = cv2.imread(imgpath)
    x, y, c = img.shape

    # get the sample specific feature info
    rawInfo = dfRaw[dfRaw["Z"] == s]
    smoothInfo = dfNew[dfNew["Z"] == s]

    # merge the info of the features which have the same ID
    allInfo = pd.merge(rawInfo, smoothInfo, left_on = 'ID', right_on = 'ID')

    # get the common feature positions and scale if it necessary 
    rawFeats = np.c_[allInfo.X_x, allInfo.Y_x] * scl
    smFeats = np.c_[allInfo.X_y, allInfo.Y_y] * scl

    # if there is no information regarding the positions to warp images,
    # assume images don't need warping therefore save the original image as final
    if len(rawInfo) == 0 or len(smFeats) == 0:
        cv2.imwrite(dest + name + ".png", img)
        return
        
    # flip the column order of the features for the sparse matrix calculation
    rawf = np.fliplr(np.unique(np.array(rawFeats), axis = 0))
    smf = np.fliplr(np.unique(np.array(smFeats), axis = 0))

    # ensure the inputs are 4D tensors
    tfrawPoints = np.expand_dims(rawf.astype(float), 0)
    tfsmPoints = np.expand_dims(smf.astype(float), 0)
    tftarImg = np.expand_dims(img, 0).astype(float)

    # perform non-rigid deformation on the image
    # NOTE for tif uses excessive memory, consider using HPC...
    imgMod, imgFlow = sparse_image_warp(tftarImg, tfrawPoints, tfsmPoints, \
        num_boundary_points=border, \
            regularization_weight=smoother, \
                interpolation_order=order)

    # print("error = " + str(np.round(np.log(np.sum(np.abs(imgMod - img)/3)), 2)))

    # convert the image and flow field into something useful
    imgMod = np.array(imgMod[0]).astype(np.uint8)
    Flow = np.array(imgFlow[0])
    # plt.imshow(np.sum(np.abs(imgMod - img)/3, axis = 2), cmap = 'gray'); plt.show()

    # print("Max Flow = " + str(np.max(Flow)) + " Min Flow = " + str(np.min(Flow)))

    # add annotations to the image to show the feature position changes
    if annotate:
        # convert Flow tensor into a 3D array so that it can be saved
        # /displayed as some kind of image. 
        imgFlow = np.zeros(imgMod.shape).astype(np.uint8)

        # NOTE visualising the flow field, HOWEVER it is currently relative....
        # change constants multiplier to create a fixed range.
        imgFlow[:, :, 0] = (Flow[:, :, 0] - np.min(Flow[:, :, 0])) / (np.max(Flow[:, :, 0])-np.min(Flow[:, :, 0])) * 255
        imgFlow[:, :, 1] = (Flow[:, :, 1] - np.min(Flow[:, :, 1])) / (np.max(Flow[:, :, 1])-np.min(Flow[:, :, 1])) * 255
            
        l = tile(sz, x, y)/2
        for n, (r, t, id) in enumerate(zip(rawFeats, smFeats, allInfo.ID)):
            rawP = r.astype(int)
            smP = t.astype(int)
            
            cv2.circle(imgMod, tuple(rawP), 12, [0, 0, 255], 8)
            cv2.circle(imgMod, tuple(smP), 12, [255, 0, 0], 8)

            # add featue ID to image
            cv2.putText(imgMod, str(id), tuple(smP + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 5)
            cv2.putText(imgMod, str(id), tuple(smP + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)

            # add key 
            cv2.putText(imgMod, str("Orig"), tuple([100,100]), cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 255, 255], 8)
            cv2.putText(imgMod, str("Orig"), tuple([100,100]), cv2.FONT_HERSHEY_SIMPLEX, 2, [0, 0, 255], 3)

            cv2.putText(imgMod, str("New"), tuple([100,150]), cv2.FONT_HERSHEY_SIMPLEX, 2, [255, 255, 255], 8)
            cv2.putText(imgMod, str("New"), tuple([100,150]), cv2.FONT_HERSHEY_SIMPLEX, 2,  [255, 0, 0], 3)

            pos = np.argsort(rawFeats, axis = 0)[:, 0]
            rawFeatsS = rawFeats[pos]
            smFeatsS = smFeats[pos]

        # draw lines between the points
        '''
        for n in range(len(rawFeatsS) - 1):
            imgMod = drawLine(imgMod, rawFeatsS[n, :], rawFeatsS[n+1, :], colour = [0, 0, 255])
            imgMod = drawLine(imgMod, smFeatsS[n, :], smFeatsS[n+1, :], colour = [255, 0, 0])
        '''    

        # cv2.imshow("Flow + Imgmod", np.hstack([imgFlow, imgMod])); cv2.waitKey(0)

    # save the images
    if prefix == "tif":
        tifi.imwrite(dest + name + ".tif", imgMod)
    else:
        cv2.imwrite(dest + name + ".png", imgMod)

def plotFeatureProgress(dfs, imgAll, dirdest, gridsz = 0, sz = 0, annos = [3]):

    '''
    takes a data frame of the feature info and a list containing the images and 
    returns a picture of all the images and their features 
    Inputs:   (dfs), list of pandas data frames to plot on the same image
              (imgAll), list of the images either as numpy values or file paths
              (dirdst), path of the image to be saved
              (sz), size of the grid to use
    Outputs:  (), image of all the samples with their features and lines connecting them 
                  and a 3d line plot (plotly)

    '''

    if type(dfs) != list:
        dfs = [dfs]

    if type(annos) != list:
        annos = [annos]
    
    for df, anno in zip(dfs, annos):

        # if the input is a list of paths of the images, load the images into a list
        if type(imgAll[0]) is str:
            imgPaths = imgAll.copy()
            imgAll = []
            for i in imgPaths:
                imgAll.append(cv2.imread(i))
            x, y, c = imgAll[0].shape
        
            sampleNo = len(imgAll)
            imgAll = np.hstack(imgAll)

        # make the background white
        imgAll[np.where(np.median(imgAll, axis = 2) == 0)] = 255

        # get the dims of the area search for the features
        l = int(tile(gridsz, x, y)/2)

        imgAll[np.where(np.median(imgAll, axis = 2) == 0), :] = 255
        imgAll[np.where(np.median(imgAll, axis = 2) == 0)] = 255
        # annotate an image of the features and their matches
        imgAll = drawPoints(imgAll, df, sampleNo, zAxis = "Zs", annos = anno, crcSz = sz, l = l)

        # save the linked feature images
        cv2.imwrite(dirdest, imgAll)

def drawPoints(img, df, sampleNos, zAxis = "Z", annos = 3, l = 0, crcSz = 10, shape = None):

    '''
    This function draws the feature positions (1), connecting lines + text (2) and bounding boxes
    (3) for all the features. Using different level flags you can control to what extent 
    the annotations are.

        Inputs: \n
    (img), image to annotate on\n
    (df), pandas dataframe which stores information\n
    (sampleNos), number of samples in the image\n
    (annos), draws features circles (1), connecting lines + numbering also (2) and bounding
    boxes also (3). 0 is no annotations.\n
    (l), length of the sides to draw for the bounding boxes\n
    (crcSz), circle size \n
    (shape), tuple of the shape to force the image into a ceratin shape
    
    Outputs:
        (imgAnno), annotated image
    '''

    x, y, c = img.shape
    y /= sampleNos

    # ensure that the bounding boxes around the features is proporptional to the image size
    blur = int(np.ceil(np.sqrt(x * y / 2e6)))

    for samp, i in enumerate(np.unique(df["ID"])):
        featdf = df[df["ID"] == i]
        tar = None
        for n, (fx, fy, fz) in enumerate(zip(featdf.X, featdf.Y, featdf[zAxis])):

            # NOTE if the sample number = 1 then this means that this is being used for the 
            # bounding annotations rather than the feature tracking so don't shift feataures
            tar = ((np.array([np.round(fx), np.round(fy)])) + np.array([y * fz, 0]) * (sampleNos != 1 * 1)) .astype(int)
            if n == 0 and annos > 0:
                # draw the first point on the first sample it appears as green
                cv2.circle(img, tuple(tar), crcSz, [0, 255, 0], int(crcSz / 2))
                pass

            elif annos > 0:
                # draw the next points as red
                cv2.circle(img, tuple(tar), crcSz, [255, 0, 0], int(crcSz / 2))

                # draw lines between points
                if annos > 1:
                    img = drawLine(img, ref, tar, blur = 10, colour = [255, 0, 0])

            if annos > 2:
                # draw bounding boxes showing the search area of the features
                img = drawLine(img, tar - l, tar + [l, -l], blur = blur, colour=[int(85 * annos), 255, 0])
                img = drawLine(img, tar - l, tar + [-l, l], blur = blur, colour=[int(85 * annos), 255, 0])
                img = drawLine(img, tar + l, tar - [l, -l], blur = blur, colour=[int(85 * annos), 255, 0])
                img = drawLine(img, tar + l, tar - [-l, l], blur = blur, colour=[int(85 * annos), 255, 0])

            # label the images with their id and the number of features present
            if annos > 4:
                cv2.putText(img, str(int(i)), tuple(tar-[10, 5]), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 4)
                cv2.putText(img, str(int(i)), tuple(tar-[10, 5]), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)
            
            # reasign the information
            ref = tar
        
        # make the final point pink
        if tar is not None and annos > 0:
            cv2.circle(img, tuple(tar), crcSz, [0, 255, 255], int(crcSz / 2)) 

    # add the sample number and the number of features on the sample   
    if annos > 1: 
        for s in np.unique(df[zAxis]):
            cv2.putText(img, "Samp_" + str(int(s)) + " no_" + str(len(df[df[zAxis] == s])), tuple(np.array([y * s, 0]).astype(int) + [50, 100]), cv2.FONT_HERSHEY_SIMPLEX, 3, [255, 255, 255], 6)
        
    # resize the image
    if shape is not None:
        img = cv2.resize(img, shape)

    return(img)

def featExtractor(dest, imgPath, infoSamp, infoAll, sz, zSamp = "Zs", prefix = "png", ref = True, realPos = False):

    '''
    This function takes an image and all the feature positions as a pandas data 
    frame and extracts from the image the tissue segment as it would appear in the 
    the 3D reconstruction 

        Inputs: \n
    (dest), home directory to save the information\n
    (imgPath), the path to the single sample to extract the sections from\n
    (info), the pandas data frame of that sample info\n
    (sz), the size of the segment used to identify that feature\n
    (prefix), if the prefix is png or tif process differently to ensure 
    colour scheme is consistent and loading efficiently\n

        Outputs: \n
    (), saves the section in the directory as its feature name\n
    '''


    # get img
    if prefix == "tif":
        img = tifi.imread(imgPath)
    else:
        img = cv2.imread(imgPath)

    x, y, c = img.shape
    l = tile(sz, x, y)/2

    # create a constant shape
    shape = (2000, int(2000 * x/y))
    
    # go through all the info for each sample and extract all the features
    # and save in directories to categorise them into features, named after
    # the sample they were found in
    for _, d in infoSamp.iterrows():

        # get the feature through the entire specimen
        if realPos:
            fullFeat = infoAll[infoAll["ID"] == d.ID]
            fullFeatPos = np.c_[fullFeat.X, fullFeat.Y]
            fullFeatPlate = [np.min(fullFeatPos, axis = 0), np.max(fullFeatPos, axis = 0)]
        else:
            fullFeatPlate = None
        featSect = getSect(img, np.array([d.X, d.Y]), l, False, fullFeatPlate)
        featdir = dest + str(int(d.ID)) + "/"
        made = dirMaker(featdir)

        if made:
            if ref:
                firstSamp = drawPoints(img.copy(), infoSamp[infoSamp["ID"] == d.ID], 1, zSamp, 3, l, crcSz = 0, shape = shape)
                if prefix == "tif":
                    cv2.imwrite(featdir + "_referenceImage.jpg", cv2.cvtColor(firstSamp, cv2.COLOR_BGR2RGB)) 
                else:   
                    cv2.imwrite(featdir + "_referenceImage.jpg", firstSamp) 
        
        if prefix == "tif":
            tifi.imwrite(featdir + nameFromPath(imgPath, 3) + "." + prefix, featSect)
        else:
            cv2.imwrite(featdir + nameFromPath(imgPath, 3) + "." + prefix, featSect)

def fixFeatures(features, home):

    '''this function interpolates the position of missing features based
    on a manual record of how many samples are missing between samples

        Inputs:\n
    (features), dataframe of the features found with the X, Y positions, sample
    position in current stack, NOT the whole one, ID\n
    (home), directory to information (in particular the images and missing Samples info)\n
    (smooth), the amount of smoothing in the b-cubic spline 

        Outputs:\n
    (featuresFixed), dataframe of the features with interpolated positions
    for missing features
    '''
    
    def makeSampKey(missingSampPath, imgs):

        '''
        Gets the csv file containing the missing sample information and 
        the img paths and turns it into a data frame correlating the sample
        image with the position in its REAL vertical position
        '''

        # get the csv file on the missing samples
        try:
            info = open(missingSampPath, "r").read().split("\n")
            # read in only the missing samp info (ignore heading and final line which 
            # contains a space)
            missingSampInfo = np.array([i.split(",")[-1] for i in info[1:-1]]).astype(int)
            print("     " + str(int(np.sum(missingSampInfo))) + " missing samples")
        except:
            # info = np.c_[nameFromPath(imgs[:-1], 3), nameFromPath(imgs[1:], 3), np.zeros(len(imgs)-1).astype(int)]
            missingSampInfo = np.zeros(len(imgs)-1)
            # if there is no missingSampInfo just assume there are no missing samples
            print("     No missing sample information")

            # NOTE access findMissingSamples function

        imgID = []
        imgPath = []
        n = 0
        imgPath.append(imgs[0])
        imgID.append(0) # first feature always included
        for m, i in zip(missingSampInfo.astype(int), imgs[1:]):
            n += m
            n += 1
            imgID.append(int(n))
            imgPath.append(i)
        imgID = np.array(imgID).astype(int)

        # key = pd.DataFrame(np.c_[imgID, np.arange(len(imgID)), imgPath], columns = ["Z", "Sample", "img"])
        key = pd.DataFrame(np.c_[imgID, np.arange(len(imgID))], columns = ["Z", "Zs"])
        key["Zs"] = key.Zs.astype(int)      # convert positions into ints
        key["Z"] = key.Z.astype(int)
        return(key)

    alignedSamples = home + "alignedSamples/"
    info = home + "info/"

    # get the csv file on the missing samples
    missingSampDir = info + "missingSamples1.csv"

    # get all the images
    imgs = sorted(glob(alignedSamples + "*.png"))

    # get the samp to Z keys
    imgID = makeSampKey(missingSampDir, imgs)

    featuresdf = pd.merge(features, imgID, left_on = "Zs", right_on = "Zs").sort_values(by=["Zs", "Z"])
    featureIDs = np.unique(featuresdf.ID)
    # NOTE this is begging to be parallelised.....
    # extrapolate features on missing samples
    info = []
    for ft in featureIDs:

        featdf = featuresdf[featuresdf["ID"] == ft]
        # create a new DF with the SAME columns
        featNdf = pd.DataFrame(columns = featdf.keys())

        for rf, tf in zip(featdf.index[:-1], featdf.index[1:]):
            ref = featdf.loc[rf]
            tar = featdf.loc[tf]

            # get the number of missing samples
            rang = int(tar.Z - ref.Z)

            # use the same information, just iterate the sample number
            for n, r in enumerate(range(rang)):
                # when closer to the reference sample
                if n <= np.floor(rang/2):
                    rN = ref.copy()
                    rN.Z += n 
                    featNdf = featNdf.append(rN)
                # when closer to the target sample
                else:
                    tN = tar.copy()
                    tN.Z += (n - rang)
                    featNdf = featNdf.append(tN)

        featNdf = featNdf.append(tar)

        # store the df
        info.append(featNdf)

    # combine each feature df
    featfix = pd.concat(info)

    return(featfix)

def smoothFeatures(df, smooth = 0, zAxis = "Z"):

    '''
    Perform a cubic-b-spline smoothing over features

        Inputs:\n
    (df), data frame of the raw feature positions\n
    (smooth), the amount to smooth the features by

        Outputs:\n
    (featsSm), data frame of features with smoothed trajectories
    '''

    # create a new dataframe for the smoothed feature positions
    featsStore = []

    for f in np.unique(df["ID"]):
        xp = df[df["ID"] == f].X.astype(float)
        yp = df[df["ID"] == f].Y.astype(float)
        zp = df[df["ID"] == f][zAxis].astype(float)

        # perform a cubic spline fitting over the data
        try:
            tck, u = splprep([xp, yp, zp], s = smooth)
            # num_true_pts = len(z)
            # u_fine = np.linspace(0,1,num_true_pts)        # interpolating between missing points
            # x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            xSm, ySm, _ = splev(u, tck)
        # if there is an error, ie there aren't enough features to perform a new 
        # smoothing operation, then just return the previous values
        except:
            xSm = xp
            ySm = yp
    
        ID = np.array(df[df["ID"] == f].ID)

        # create data frame with info and ensure they are the correct data types
        featsSm = pd.DataFrame(np.c_[xSm, ySm, zp, ID], columns=["X", "Y", zAxis, "ID"])
        featsStore.append(featsSm)

    featsStore = pd.concat(featsStore)

    # ensure the correct data format in the df
    featsStore["X"] = featsStore.X.astype(float)
    featsStore["Y"] = featsStore.Y.astype(float)
    featsStore[zAxis] = featsStore[zAxis].astype(float)
    featsStore["ID"] = featsStore.ID.astype(int)
    
    return(featsStore)

if __name__ == "__main__":

    # with 2 2D images, interpolate between them for given points
    dirHome = [
    '/Volumes/Storage/H710C_6.1/',
    '/Volumes/USB/H671A_18.5/',
    '/Volumes/Storage/H653A_11.3/',
    '/Volumes/USB/H750A_7.0/',
    '/Volumes/USB/H710B_6.1/',
    '/Volumes/USB/H671B_18.5/',
    '/Volumes/USB/H653A_11.3/']


    # NOTE fork is considered unstable but it is an order of magnitude 
    # faster than using spawn.... HOWEVER this only works with the process
    # MP method, not pool.....
    multiprocessing.set_start_method("fork")


    size = 3
    cpuNo = False

    dataHome = '/eresearch/uterine/jres129/BoydCollection/H710C_6.1/'
    dataHome = '/Volumes/USB/H710C_6.1/'
    dataHome = '/Volumes/USB/H653A_11.3/'
    nonRigidAlign(dataHome, size, cpuNo)

    '''
    for d in dirHome:
        # d = '/Volumes/USB/H671B_18.5/'
        nonRigidAlign(d, size, cpuNo)
    '''