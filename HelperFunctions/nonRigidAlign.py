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

if __name__ != "HelperFunctions.nonRigidAlign":
    from SP_AlignSamples import aligner
    from SP_FeatureFinder import allFeatSearch
    from Utilities import *
    from findMissingSamples import findMissingSamples
else:
    from HelperFunctions.SP_AlignSamples import aligner
    from HelperFunctions.SP_FeatureFinder import allFeatSearch
    from HelperFunctions.Utilities import *
    from HelperFunctions.findMissingSamples import findMissingSamples


# for each fitted pair, create an object storing their key information
class feature:
    def __init__(self, ID = None, Samp = None, refP = None, tarP = None, dist = None, res = 0):

        # the feature number 
        self.ID = ID

        # sample which the reference feature is on
        self.Samp = Samp

        # the position of the match on the reference image
        self.refP = refP

        # the position of the match on the target image
        self.tarP = tarP

        self.dist = dist

        self.res = res

    def __repr__(self):
            return repr((self.ID, self.refP, self.tarP))

def nonRigidAlign(dirHome, size, cpuNo = 1, \
    featsMin = 20, dist = 30, featsMax = 20, errorThreshold = 100, \
        distFeats = 400, sect = 100, selectCriteria = "length", \
            flowThreshold = 0.01, featSmoother = 1e12, fixFeatures = False, \
                plot = False, extract = True):

    '''
    Perform a non-rigid alignment of the specimen and extract features 
    from specimen

        Inputs:\n
    (dirHome): \n
    (size): \n
    (cpuNo): \n
    (errorThreshold): maximum per feature error for the linear alignment
    (featsMin): minimum number of samples which a feature has to propogate 
    through to be used in the registeraion \n
    (dist): minimum distance features can be during the feature selection\n
    (featsMax): maximum number of features to be used during the non-linear registeraion\n
    (distFeatures): minimum distance between feataures to be used during non-linear registeraion\n
    (sect): the proporptional area size used for the feature tracking and extraction \n
    (selectCriteria): the type of trajectory used for the NL modelling (prioritising
    either a "smooth" or "length" critiera of the features\n
    (fixFeatures): boolean, if True will read through sample names and 
    attempt to compensate for missing samples but default false so no 
    missing sample interpolation

        Outputs:\n
    (): NL aligned samples
    '''
    
    home = dirHome + str(size)

    imgsrc = home + "/alignedSamples/"
    destRigidAlign = home + "/RealignedSamples/"
    dirfeats = home + "/infoNL/"
    destNLALign = home + "/NLAlignedSamplesSmall/"
    destNLALignBound = home + "/NLAlignedSamplesBound/"
    destFeatSections = home + "/FeatureSections/"

    # parallelisation is performed by process now so it is a boolean
    if cpuNo > 1:
        cpuNo = True
    else:
        cpuNo = False
    
    # Find the continuous features throught the samples
    contFeatFinder(imgsrc, dirfeats, destRigidAlign, cpuNo = cpuNo, sz = sect, dist = dist)
    
    # perform a rigid alignment on the tracked features
    aligner(imgsrc, dirfeats, destRigidAlign, cpuNo = cpuNo, errorThreshold = np.inf)
    
    if fixFeatures:
        findMissingSamples(dirHome, size)
    # with all the features found, find their trajectory and adjust to create continuity 
    # between samples
    
    featShaper(destRigidAlign, destFeatSections, featsMin = featsMin, \
        dist = distFeats, maxfeat = featsMax, selectCriteria = selectCriteria, \
            featSmoother = featSmoother, plot = plot)
    
    # extract sections and deform the downsampled images
    nonRigidDeform(destRigidAlign, destNLALign, destFeatSections, prefix = "png", flowThreshold = flowThreshold)

    # bounding images so that as little background is visiblew
    exactBound(destNLALign, "png", destNLALignBound)

    # extract the feature sections 
    if extract:
        # allFeatExtractor(destRigidAlign, destFeatSections, prefix = "png", scl = 1, sz = sect, realPos = False)
        # allFeatExtractor(destRigidAlign, destFeatSections, prefix = "png", scl = 1, sz = sect, realPos = True)
        allFeatExtractor(destNLALign, destFeatSections, prefix = "png", scl = 1, sz = sect, realPos=False)
        allFeatExtractor(destNLALign, destFeatSections, prefix = "png", scl = 1, sz = sect, realPos=True)


def contFeatFinder(imgsrc, destFeat, destImg, cpuNo = False, sz = 100, dist = 20):

    '''
    This function takes images and finds features that are continuous
    between the samples

    Inputs:\n
    (imgs), list of directories of the images to be processed in sequential order for feature identifying\n
    (destFeat), where to save the tracked features
    (destImg), where to save the realigned images
    (cpuNo), number of cores to parallelise the task.\n
    (sz), the equivalent area which a continuous feature will be searched for if this many tiles were created on the section\n
    (dist), minimum pixel distance between detected features\n
    (connect), the final number of features that are tracked to return. These will be the most connected features available\n

    Outputs: \n
    (), saves the feature information. The number of feature paths that are saved is 
    based on the featsMax, where that is the minimum number of features found but at 
    least that many features exist through the most linked connections
    '''

    dirMaker(destFeat)
    dirMaker(destImg)

    # find the ref and target images needed to complete matching
    imgs = sorted(glob(imgsrc + "*.png"))
    imgRef = imgs[:-1]
    imgTar = imgs[1:]

    # intialise objects to store and track info
    matchedInfo = []
    refFeats = {}
    allMatchedInfo = {}
    continuedFeatures = []
    featNo = 0

    # sequantially identify new previous features in each sample
    for sampleNo, (refPath, tarPath) in enumerate(zip(imgRef, imgTar)):

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
            cpuNo = cpuNo, gridNo=1, tol = 0.1, featMin = 80, \
                scales = scales, maxFeats = 200, \
                    name_ref = refName, name_tar = tarName)[0]
        allFeatSearchTime = time() - startAllFeatSearch
        # print("     allFeatSearch = " + str(allFeatSearchTime))

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
        # print("     featMatchMaker = " + str(featMatchTime))
        
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
        confirmedFeatures = matchMaker(confirmInfos, dist = dist, tol = 1, cpuNo = cpuNo, distCheck=True, spawnPoints=10, angThr=20, distTrh=0.2)
        # confirmedFeatures = confirmInfos.copy()
        matchMakerTime = time() - startMatchMaker
        # print("     matchMaker = " + str(matchMakerTime))

        imgCombine = nameFeatures(refImg, tarImg, confirmedFeatures, scales, combine = True, txtsz=0.5)
        cv2.imwrite(destFeat + tarName + "-->" + refName + "_processed.png", imgCombine)

        # ----- identify the feature ID -------

        # ensure that each feature is identified
        continuedFeatures = []
        featNew = 0
        featCont = 0
        for c in confirmedFeatures:
            if c.ID is None:
                # keep track of the feature ID
                c.ID = featNo
                allMatchedInfo[c.ID] = {}
                featNo += 1
                featNew += 1
            else:
                featCont += 1
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
        print("     New features = " + str(featNew) + ", continued features = " + str(featCont))
    # arrange the data in a way so that it can be plotted in 3D
    samples = len(imgs)
    
    for n in range(1,samples):
        featureNo = len(np.where(np.array([len(allMatchedInfo[mi]) for mi in allMatchedInfo]) == n)[0])
        if featureNo > 0:
            print("Number of " + str(1+n) + "/" + str(samples) + " linked features = " + str(featureNo))

def featShaper(diralign, dirSectdest, featsMin = 5, dist = 100, maxfeat = np.inf, selectCriteria = "smooth", featSmoother = 1e12, plot = False):
    '''
    This function modifies the identified features to create continuity of features

        Inputs:\n
    (diralign), directory containing the images which the deformations were performed on
    (dirSectDest), directory of where the extracted specimens will be stored (and where the 
        feature position will be stored)
    (featsMin), minimum number of features per sample
    (dist), distance between each feature (in pixels)
    (maxFeat), maximum number of features per sample
    (selectCriteria), either prioritise smoothness (compared to the raw vs smooth features) 
        or longer features
    (featSmoother), the smooth parameter for the cubic B-spline
    (plot), boolean. if true then 3D plot feature trajectories

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

    # fix all the features
    featsAllFix = fixFeatures(dfAllCont, regionOfPath(diralign, 2))

    # smooth the features (1e12 is sufficient to smooth the very large interpolation error
    # of H653A but may need to be varied for other specimens...)
    featsSm = smoothFeatures(featsAllFix, featSmoother, zAxis = "Z")
    # px.line_3d(featsSm, x="X", y="Y", z="Zs", color="ID", title = "ALL raw features with min continuity").show()

    # select the best features
    dfSelectR, dfSelectSM, _ = featureSelector(featsAllFix, featsSm, featsMin = featsMin, dist = dist, maxfeat = maxfeat, cond = selectCriteria)

    # 3D plot the smoothed and rough selected features 
    if plot:    
        imgs = sorted(glob(diralign + "*.png"))[:4]
        plotFeatureProgress(dfSelectR[dfSelectR["Zs"] < 4], imgs, diralign + 'CombinedRough.jpg', 1, 35, 2)

        px.line_3d(dfAllCont, x="X", y="Y", z="Zs", color="ID", title = "ALL raw features with min continuity").show()
        px.line_3d(dfSelectR, x="X", y="Y", z="Z", color="ID", title = "Raw selected features " + selectCriteria).show()
        px.line_3d(dfSelectSM, x="X", y="Y", z="Z", color="ID", title = "Smoothed selected features " + selectCriteria).show()    
    
    # save the data frames as csv files
    dfAll.to_csv(dirSectdest + "all.csv")
    dfAllCont.to_csv(dirSectdest + "rawFeatures.csv")
    featsSm.to_csv(dirSectdest + "smoothFixFeatures.csv")
    dfSelectR.to_csv(dirSectdest + "rawSelectedFixFeatures.csv")
    dfSelectSM.to_csv(dirSectdest + "smoothSelectedFixFeatures.csv")

def allFeatExtractor(imgSrc, dirSectdest, prefix, scl = 1, sz = 0, realPos = False):
    # serialised feature extraction 
    # NOTE this has to be serialised so that the right reference image is used
    
    print("\n--- Extracting of " + str(realPos) + " position " + prefix + " features from " + imgSrc + " ---\n")

    # get the image paths
    imgs = sorted(glob(imgSrc + "*" + prefix))

    sectType = imgSrc.split("/")[-2]

    # set the destination of the feature sections based on the image source and prefix
    LSectDir = dirSectdest + sectType + prefix + "_" + str(realPos) + "/"

    # if processing the non linear or linear aligned features
    if imgSrc.find("NL") > -1:
        zSamp = "Z"
        dfRawCont = pd.read_csv(dirSectdest + "smoothFixFeatures.csv")
    else:
        zSamp = "Zs"
        dfRawCont = pd.read_csv(dirSectdest + "rawFeatures.csv")

    # identify the longest 3 features
    ID, IDCount = np.unique(dfRawCont.ID, return_counts = True)
    keyFeats = ID[np.argsort(-IDCount)][:3]

    # for all selected features print out the first and last samples and sections
    keyFeats = -1

    for n, img in enumerate(imgs):
        printProgressBar(n, len(imgs)-1, "feats", "", 0, 20)
        # print(str(n) + "/" + str(len(imgs)-1))
        sampdf = dfRawCont[dfRawCont[zSamp] == n].copy()
        sampdf.X *= scl; sampdf.Y *= scl        # scale the positions based on image size
        featExtractor(LSectDir, img, sampdf, dfRawCont, sz, zSamp, prefix = prefix, realPos = realPos, keyFeats = keyFeats)

def nonRigidDeform(diralign, dirNLdest, dirSectdest, scl = 1, prefix = "png", flowThreshold = 1):

    '''
    This transforms the images based on the feature transforms

    Inputs:   
        (diralign), directory of all the aligned info and images
        (dirNLdest), path to save the NL deformed info
        (dirSectdest), path to save the feature sections
        (scl), resolution scale factor
        (prefix), image type used
        (flowThreshold), maximum magnitude vector for deformation percent of the image diagonal
    
    Outputs:  
        (), warp the images and save
    '''

    print("\n--- NL deformation of " + prefix + " images ---\n")

    featsConfirm = dirSectdest + "featsConfirm/"

    dirMaker(dirNLdest)
    dirMaker(featsConfirm)

    # get info
    if prefix == "png":
        imgs = sorted(glob(diralign + "*" + prefix))
        # create the directory to store the features that were actuall y
        # used for the saved deformation
        dirMaker(featsConfirm)
        dfSelectSMFix = pd.read_csv(dirSectdest + "smoothSelectedFixFeatures.csv")
        dfSelectRFix = pd.read_csv(dirSectdest + "rawSelectedFixFeatures.csv")
    elif prefix == "tif":
        imgs = sorted(glob(diralign + "*" + prefix))
        # get the features which were used during the baseline warping
        smFeats = glob(featsConfirm + "*sm.csv")
        rawFeats = glob(featsConfirm + "*raw.csv")
        dfSelectSMFix, dfSelectRFix = (None, None)
        for sm, r in zip(smFeats, rawFeats):
            smDf = pd.read_csv(sm)
            rDf = pd.read_csv(r)
            # initialise the data frame
            if dfSelectSMFix is None:
                dfSelectSMFix = smDf
                dfSelectRFix = rDf
            else:
                # append the dataframes
                dfSelectSMFix = dfSelectSMFix.append(smDf)
                dfSelectRFix = dfSelectRFix.append(rDf)

    # create the dictionary which relates the real Z number to the sample image available
    key = np.c_[np.unique(dfSelectRFix.Z), [np.unique(dfSelectRFix[dfSelectRFix["Z"] == f].Zs)[0] for f in np.unique(dfSelectRFix.Z)]].astype(int)

    # Warp images to create smoothened feature trajectories
    # NOTE the sparse image warp is already highly parallelised so not MP functions
    for Z, Zs in key:
        # print(str(Z) + "/" + str(len(key)))
        imgPath = imgs[Zs]
        output = ImageWarp(Z, imgPath, dfSelectRFix, dfSelectSMFix, dirNLdest, border = 5, smoother = 0, order = 1, annotate = False, featscl = scl, thr = flowThreshold)
        # during the feature transform of the png image, the final 
        # positions used for warping were used are saved
        if output is not None:
            raw, sm = output 

            # create data frames which contain the necessary information to be used in a 
            # NL deformation
            rawdf = pd.DataFrame(np.c_[raw, np.repeat(Z,len(raw)), np.repeat(Zs,len(raw)), np.arange(len(raw))])
            smdf = pd.DataFrame(np.c_[sm, np.repeat(Z,len(raw)), np.repeat(Zs,len(raw)), np.arange(len(raw))])

            # create the data frames which produced the successful deformations. NOTE that the
            # ID isn't the real ID of the feature, it is instead just so associated corresponding
            # features
            rawdf.to_csv(featsConfirm + nameFromPath(imgPath, 3) + "raw.csv", header = ["Y", "X", "Z", "Zs", "ID"])
            smdf.to_csv(featsConfirm + nameFromPath(imgPath, 3) + "sm.csv", header = ["Y", "X", "Z", "Zs", "ID"])

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
    (maxfeat) max number of features to be returned per sample
    (cond), the priority of the features to select (either prioritising longer or smoothr features)

    Outputs:\n
    (dfNew), pandas data frame of the features which meet the requirements 
    (targetIDs), the feature IDs that meet the criteria
    '''
    
    # set the maximum iterations
    sampleMax = np.max(dfR["Z"])
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
        
        # this is the number of samples ahead of the current one we are looking to ensure features are still present
        extraLen = featsMin - (featsMin - (sampleMax - s)) * ((sampleMax - s) < featsMin)

        # get all the feature positions on the sample
        sampsdfBoth = []
        n = 0
        # ensure that there are at least 3 features which are continued.
        # if that condition is not met then reduce the extraLen needed
        while len(sampsdfBoth) < 3 and n < extraLen:
            # create a data frame which contains the features in the current sample as well
            # as the samples further on in the specimen 
            sampHere, sampExtra = dfR[dfR["Z"] == s + extraLen - n], dfR[dfR["Z"] == s]
            sampsdfBoth = pd.merge(sampHere, sampExtra, left_on = "ID", right_on = "ID")

            # create a DF which contains all the features which pass through sample s
            try:    
                dfSamp = pd.concat([dfR[dfR["ID"] == f] for f in np.unique(sampsdfBoth.ID)])
                # get the feature IDs and their REMAINING lengths from the current sample
                _, featLen = np.unique(dfSamp[dfSamp["Z"] >= s + extraLen - n].ID, return_counts = True)

            except: pass
            n += 1


        # calculate the average error per point between the smoothed and raw features
        errorStore = []
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
                pdAllR.append(dfR[dfR["ID"]==idF][dfR[dfR["ID"]==idF]["Z"] >= s])
                pdAllSm.append(dfSm[dfSm["ID"]==idF][dfSm[dfSm["ID"]==idF]["Z"] >= s])

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
            s += np.min([np.max((dfNew[dfNew["ID"] == i]).Z)-s+1 for i in featPosID])

        # re-initialise all the features which are longer than the 
        # shortest feature
        featPosID = list(np.unique(dfNew[dfNew["Z"] >= s].ID))
        featPos = list(np.c_[dfNew[dfNew["Z"] == s].X, dfNew[dfNew["Z"] == s].Y])
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

def ImageWarp(s, imgpath, dfRaw, dfNew, dest, sz = 100, smoother = 0, border = 5, order = 2, annotate = False, featscl = 1, thr = 0.01, trueName = False):

    # perform the non-rigid warp
    # Inputs:   (s), number of the sample
    #           (imgpath), directory of the image
    #           (dfRaw), raw data frame of the feature info
    #           (dfNew), data frame of the smoothed feature info
    #           (dest), directory to save modified image
    #           (featscl), rescaling value for the image to match the feature postiions
    #           (flowThreshold), the maximum diagonal percent length of a deformation
    #               allowed for a transformation to occur 
    # Outputs:  (imgMod), the numpy array of the warped image
    #           (imgFlow), the flow field which was produced to make the 
    #               warped image

    # scale image to perform deformations to identify the ideal points to use 
    # no minimise flow errors
    imgscl = 0.2

    # ensure the naming convention is correct 
    prefix = imgpath.split(".")[-1]       # get image prefix
    
    if trueName:
        name = nameFromPath(imgpath, 4)
    else:
        samp = str(s)
        while len(samp) < 4:
            samp = "0" + samp
        name = nameFromPath(imgpath, 1) + "_" + str(samp)

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
    rawFeats = np.c_[allInfo.X_x, allInfo.Y_x] * featscl
    smFeats = np.c_[allInfo.X_y, allInfo.Y_y] * featscl

    # if there is no information regarding the positions to warp images,
    # assume images don't need warping therefore save the original image as final
    if len(rawInfo) == 0 or len(smFeats) == 0:
        cv2.imwrite(dest + name + ".png", img)
        return
        
    # flip the column order of the features for the sparse matrix calculation
    rawf = np.fliplr(np.unique(np.array(rawFeats), axis = 0))
    smf = np.fliplr(np.unique(np.array(smFeats), axis = 0))

    tftarImg = np.expand_dims(cv2.resize(img, (int(img.shape[1]*imgscl), int(img.shape[0]*imgscl))), 0).astype(float)
    imgSize = np.sqrt((tftarImg.shape[0]**2 + tftarImg.shape[1]**2))

    # NOTE for fullscale uses lots of memory, consider using HPC...
    # identify the transformation which results in the desired flow threshold
    # NOTE if thr == 1 then there is no need to identify optimal deformations
    while thr < 1:
        imgMod, imgFlow = sparse_image_warp(tftarImg, \
            np.expand_dims(rawf.astype(float)*imgscl, 0), \
                np.expand_dims(smf.astype(float)*imgscl, 0), \
                    num_boundary_points=border, \
                        regularization_weight=smoother, \
                            interpolation_order=order)

        # print("error = " + str(np.round(np.log(np.sum(np.abs(imgMod - img)/3)), 2)))

        # if the largest vector magnitude is larger than 50, this is indicative that
        # the deformation is possible "impossible" so apply a greater smoothing
        # constant and try again
        # get the densefield vector magnitude
        imgFlowMag = np.max(np.sqrt(imgFlow[0, :, :, 0]**2 + imgFlow[0, :, :, 1]**2))
        printProgressBar(np.clip(thr/(imgFlowMag/imgSize), 0, 1), 1, "Warping " + name, "", 0, 20)

        if (imgFlowMag/imgSize)>thr:
            
            pos = np.argmax(np.sum((rawf-smf)**2, axis = 1))
            # remove the individual feature with the largest error contribution
            rawf = np.delete(rawf, pos, axis = 0)
            smf = np.delete(smf, pos, axis = 0)
        else:
            break

    # perform the NL deformation on the original sized image
    tftarImg = np.expand_dims(img, 0).astype(float)

    # perform the actual transformation with the full scale image once
    # key points are known
    imgMod, imgFlow = sparse_image_warp(tftarImg, \
    np.expand_dims(rawf.astype(float), 0), \
        np.expand_dims(smf.astype(float), 0), \
            num_boundary_points=border, \
                regularization_weight=smoother, \
                    interpolation_order=order)
    
    '''
    imgMod, imgFlow = sparse_image_warp(tftarImg, \
            np.expand_dims(rawf.astype(float)*imgscl, 0), \
                np.expand_dims(smf.astype(float)*imgscl, 0), \
                    num_boundary_points=border, \
                        regularization_weight=smoother, \
                            interpolation_order=order)

    imgMod = np.array(imgMod[0]).astype(np.uint8)
    imgFlow = np.array(imgFlow[0]).astype(float)
    imgFlowMag = np.sqrt(imgFlow[:, :, 0]**2 + imgFlow[:, :, 1]**2)
    imgFlowMagNorm = ((imgFlowMag-np.min(imgFlowMag))/(np.max(imgFlowMag) - np.min(imgFlowMag))*255).astype(np.uint8)
    '''
    # add annotations to the image to show the feature position changes
    if annotate:
        # convert Flow tensor into a 3D array so that it can be saved
        # /displayed as some kind of image. 

        # imgFlow = np.mean(imgFlow, axis = 2)

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

    # save the images
    if prefix == "tif":
        tifi.imwrite(dest + name + ".tif", imgMod)
        print(name + " full-scale deformation saved")
        return None

    else:
        dirMaker(dest + "flowMagnitude/")
        cv2.imwrite(dest + name + ".png", imgMod)
        print(name + " baseline deformation saved")

        # based on the final positions of the features found, return these 
        # for the tif transformation so that the flow threshold process
        # doesn't have to occur
        return([rawf, smf])

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

def featExtractor(dest, imgPath, infoSamp, infoAll, sz, zSamp = "Zs", prefix = "png", realPos = False, keyFeats = None):

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
    (realPos), if True then the sections extracted are placed in their relative positions
    from the sample (ie black border around the non-target tissue). If False then the section 
    is perfect extracted

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
            firstSamp = drawPoints(img.copy(), infoSamp[infoSamp["ID"] == d.ID], 1, zSamp, 3, l, crcSz = 0, shape = shape)
            if prefix == "tif":
                cv2.imwrite(featdir + "_referenceImage.jpg", cv2.cvtColor(firstSamp, cv2.COLOR_BGR2RGB)) 
            else:   
                cv2.imwrite(featdir + "_referenceImage.jpg", firstSamp) 
        
        # for the key features draw bounding features around all of them to capture the final feature
        if len(np.where(keyFeats == d.ID)[0]) > 0 or keyFeats:
            finalImg = drawPoints(img.copy(), infoSamp[infoSamp["ID"] == d.ID], 1, zSamp, 3, l, crcSz = 0, shape = shape)
            cv2.imwrite(featdir + "_finalImg.jpg", finalImg) 

        try:
            if prefix == "tif":
                tifi.imwrite(featdir + nameFromPath(imgPath, 3) + "." + prefix, featSect)
            else:
                cv2.imwrite(featdir + nameFromPath(imgPath, 3) + "." + prefix, featSect)
        except:
            pass

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
    missingSampDir = info + "missingSamples.csv"

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

    # NOTE fork is considered unstable but it is an order of magnitude 
    # faster than using spawn.... HOWEVER this only works with the process
    # MP method, not pool.....
    multiprocessing.set_start_method("fork")


    size = 1.25
    cpuNo = 5

    dataHomes = [
    # '/eresearch/uterine/jres129/BoydCollection/H653A_11.3/',
    # '/eresearch/uterine/jres129/BoydCollection/H671A_18.5/',
    # '/eresearch/uterine/jres129/BoydCollection/H671B_18.5/',
    # '/eresearch/uterine/jres129/BoydCollection/H710B_6.1/',
    '/eresearch/uterine/jres129/BoydCollection/test/',
    '/eresearch/uterine/jres129/BoydCollection/H710C_6.1/',
    '/eresearch/uterine/jres129/BoydCollection/H673A_7.6/',
    '/eresearch/uterine/jres129/BoydCollection/H1029A_8.4/'
    '/eresearch/uterine/jres129/BoydCollection/H750A_7.0/',
    ]
    
    dataHomes = [
    # '/Volumes/USB/H653A_11.3/',  
    # '/Volumes/USB/H671A_18.5/',  
    # '/Volumes/USB/H671B_18.5/',  
    '/Volumes/USB/H710C_6.1/',
    '/Volumes/USB/H710B_6.1/',   
    '/Volumes/USB/H673A_7.6/',   
    '/Volumes/USB/H750A_7.0/',   
    '/Volumes/USB/H1029A_8.4/'
    ]
    
    for dataHome in dataHomes:
        name = dataHome.split("/")[-2]
        print(name)
        nonRigidAlign(dataHome, size, cpuNo = cpuNo, \
        featsMin = 10, dist = 30, featsMax = 100, errorThreshold = 200, \
            distFeats = 50, sect = 100, selectCriteria = "length", \
                flowThreshold = 0.1, fixFeatures = False, plot = False)

    '''
    for d in dirHome:
        # d = '/Volumes/USB/H671B_18.5/'
        nonRigidAlign(d, size, cpuNo)
    '''