import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from HelperFunctions.Utilities import *
from HelperFunctions.SP_AlignSamples import aligner
from HelperFunctions.SP_FeatureFinder import allFeatSearch
from tensorflow_addons.image import sparse_image_warp
import plotly.express as px
import multiprocessing
from itertools import repeat
from skimage.registration import phase_cross_correlation as pcc
import pandas as pd
from scipy.interpolate import splprep, splev
from time import time
from copy import copy

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
    destRigidAlign = home + "/RealignedSamplesAll/"
    dirfeats = home + "/infoNL/"
    destNLALign = home + "/NLAlignedSamplesAll2/"
    destFeatSections = home + "/FeatureSectionsAll2/"
    
    dirMaker(destRigidAlign)
    dirMaker(dirfeats)
    dirMaker(destNLALign)
    dirMaker(destFeatSections)

    imgs = sorted(glob(imgsrc + "*.png"))
    sect = 300           # the proporptional area of the image to search for features
    dist = 30           # distance between the sift features
    featsMin = 20       # min number of samples a feature must pass through for use to NL deform

    # Find the continuous features throught the samples
    # contFeatFinder(imgs, dirfeats, destRigidAlign, cpuNo = True, plotting = True, sz = sect, dist = dist, connect = featsMin)
    
    # perform a rigid alignment on the tracked features
    # aligner(imgs, dirfeats, imgsrc, destRigidAlign, cpuNo = False, errorThreshold=100)
    
    featsMax = 20       # max number of samples which meet criteria to be used
    distFeats = 300     # distance (pixels) between the final features

    # perform a non-linear deformation on the samples based on a smoothed trajectory of the features
    # through the samples
    nonRigidDeform(destRigidAlign, destNLALign, destFeatSections, dist = distFeats, sz = sect, featsMin = featsMin, featsMax = featsMax)

def contFeatFinder(imgs, destFeat, destImg = None, cpuNo = False, plotting = False, sz = 100, dist = 20, connect = np.inf):

    '''
    This function takes images and finds features that are continuous
    between the samples

    Inputs:\n
    (imgs), list of directories of the images to be processed in sequential order for feature identifying\n
    (destImg), where to save the image from the plotting
    (cpuNo), number of cores to parallelise the task.\n
    (plotting), boolean whether to plot the feature info through the samples\n
    (sz), the equivalent area which a continuous feature will be searched for if this many tiles were created on the section\n
    (dist), minimum pixel distance between detected features\n
    (connect), the final number of features that are tracked to return. These will be the most connected features available\n

    Outputs: \n
    (), saves the feature information. The number of feature paths that are saved is 
    based on the featsMax, where that is the minimum number of features found but at 
    least that many features exist through the most linked connections
    '''

    # intialise objects to store and track info
    matchedInfo = []
    allMatchedInfo = {}
    featNo = 0

    prevImg1 = None
    prevImg2 = None
    continuedFeatures = []

    # sequantially identify new previous features in each sample
    for sampleNo, (refPath, tarPath) in enumerate(zip(imgs[:-1], imgs[1:])):

        # load the images 
        refImg = cv2.imread(refPath)
        refName = nameFromPath(refPath, 3)
        tarName = nameFromPath(tarPath, 3)
        tarImg = cv2.imread(tarPath, 3)

        print("Matching " + tarName + " to " + refName)

        # ---- find new features between the samples ----

        # find spatially coherent sift features between corresponding images
        startAllFeatSearch = time()
        matchedInfo = allFeatSearch(refImg, tarImg, dist = dist, cpuNo = True, gridNo=30)[0]
        allFeatSearchTime = time() - startAllFeatSearch

        # featMatchimg = nameFeatures(refImg, tarImg, matchedInfo, combine = True)
        # plt.imshow(featMatchimg); plt.show()

        # ---- find the position of the feature in the next slice based on the previous informaiton ----

        startFeatMatch = time()
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
                j = multiprocessing.Process(target=featMatching, args = (m, tarImg, refImg, None, sz, qs[n]))
                job.append(j)
                j.start()
            for j in job:
                j.join()

            confirmInfo = []
            for q in qs:
                confirmInfo.append(qs[q].get())
        featMatchTime = time() - startFeatMatch
        
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
        confirmedFeatures = matchMaker(confirmInfos, dist = dist, tol = 1, cpuNo = True, anchorPoints=5, distCheck=True)
        matchMakerTime = time() - startMatchMaker

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
            cont = copy(c)
            cont.refP = cont.tarP
            cont.tarP = None
            continuedFeatures.append(cont)
        '''
        prevImg2 = prevImg1
        prevImg1 = refImg
        '''

        print("     allFeatSearch = " + str(allFeatSearchTime))
        print("     featMatchMaker = " + str(featMatchTime))
        print("     matchMaker = " + str(matchMakerTime) + "\n")

    # -------- Plotting and info about the features --------

    # arrange the data in a way so that it can be plotted in 3D
    samples = len(imgs)
    
    for n in range(1,samples):
        featureNo = len(np.where(np.array([len(allMatchedInfo[mi]) for mi in allMatchedInfo]) == n)[0])
        if featureNo > 0:
            print("Number of " + str(1+n) + "/" + str(samples) + " linked features = " + str(featureNo))

    # create a panda data frame of all the features found for plotting
    df = dictToDF(allMatchedInfo, ["xPos", "yPos", "Sample", "ID"], min=connect)

    # create feature dictionaries per sample
    for s in range(samples-1):

        refN = nameFromPath(imgs[s], 3)
        tarN = nameFromPath(imgs[s+1], 3)

        # get the features of the reference image
        ref = df[df["Sample"] == s]

        # get the features of the target image
        tar = df[df["Sample"] == s+1]

        refD = {}
        for x, y, s, ID in np.array(ref):
            refD[int(ID)] = np.array([x, y])
        dictToTxt(refD, destFeat + refN + ".reffeat", fit = False, shape = refImg.shape)
        
        tarD = {}
        for x, y, s, ID in np.array(tar):
            tarD[str(int(ID))] = np.array([x, y])
        dictToTxt(tarD, destFeat + tarN + ".tarfeat", fit = False, shape = tarImg.shape)
    
    # plot the position of the features through the samples
    
    if plotting:
        px.line_3d(df, x="xPos", y="yPos", z="Sample", color="ID", title = "All features, unaligned").show()
        plotFeatureProgress(df, imgs, destImg + 'CombinedRough.jpg', sz, 2)

def nonRigidDeform(diralign, dirNLdest, dirSectdest, dist = 100, sz = 0, featsMin = 0, featsMax = None):

    '''
    This function takes the continuous feature sets found in contFeatFinder
    and uses them to non-rigidly warp the 

    Inputs:   
        (diralign), directory of all the aligned info and images
        (dirNLdest), path to save the NL deformed info
        (dirSectdest), path to save the feature sections
        (scl), resolution scale factor
        (s), section size used for feature finding
    
    Outputs:  
        (), warp the images and save
    '''

    LSectDir = dirSectdest + "linearSect/"
    NLSectDir = dirSectdest + "NLSect/"

    # get the image paths
    imgs = sorted(glob(diralign + "*.png"))

    # get the new dictionaries, load them into a pandas dataframe
    refFeats = sorted(glob(diralign + "*.reffeat"))
    tarFeats = sorted(glob(diralign + "*.tarfeat"))
    infopds = []

    # NOTE beacuse the ref and tarfeatures are the same for all the samples
    # it just has to iteratue through the samples, not the ref/tar feat files
    for n, r in enumerate(refFeats + [tarFeats[-1]]):
        info = txtToDict(r, float)[0]
        infopd = pd.DataFrame.from_dict(info, orient = 'index', columns = ['xPos', 'yPos'])
        infopd['Sample'] = n       # add the sample number
        infopd['ID'] = np.array(list(info.keys())).astype(int)
        infopds.append(infopd)
        
    # combine into a single df
    df = pd.concat(infopds)
    # px.line_3d(df, x="xPos", y="yPos", z="Sample", color="ID", title = "Raw aligned features").show()

    # serialised feature extraction (for debugging)
    '''
    for n, img in enumerate(imgs):
        sampdf = df[df["Sample"] == n]
        featExtractor(destSects, img, sampdf, sz)
    '''

    # for all the features identified, extract the found features
    job = []
    for n, img in enumerate(imgs):
        sampdf = df[df["Sample"] == n]
        j = multiprocessing.Process(target=featExtractor, args = (LSectDir, img, sampdf, sz))
        job.append(j)
        j.start()
    for j in job:
        j.join()
    
    # create the 3D plot of the aligned features
    # px.line_3d(df, x="xPos", y="yPos", z="Sample", color="ID", title = "All aligned features").show()
    
    # pick the best features for warping --> this is important to ensure that the 
    # features are not too close to eachother or else they will cause impossible 
    # warping
    shape = cv2.imread(imgs[0]).shape

    # create a new dataframe for the smoothed feature positions
    featsSm = pd.DataFrame(columns=["xPos", "yPos", "Sample", "ID"])

    p = 0       # create data frame count 
    for f in np.unique(df["ID"]):
        xp = df[df["ID"] == f].xPos
        yp = df[df["ID"] == f].yPos
        z = df[df["ID"] == f].Sample

        # if the number of samples the feature passes through is more than 
        # the number of images being processed, don't include
        if np.max(z) > len(imgs): 
            sampRange = np.where(z < len(imgs))[0]
            if len(sampRange) < 3:
                continue
            xp = xp[sampRange]
            yp = yp[sampRange]
            z = z[sampRange]

        num_true_pts = len(z)

        # perform a cubic spline fitting over the data
        
        '''
        xSm = np.linspace(np.array(xp)[0], np.array(xp)[0], len(xp))
        ySm = np.linspace(np.array(yp)[0], np.array(yp)[0], len(yp))

        '''
        # this could possibly be used to interpolate between slices
        # to find missing ones!
        tck, u = splprep([xp, yp, z], s = 1e8)
        # u_fine = np.linspace(0,1,num_true_pts)
        # x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        xSm, ySm, _ = splev(u, tck)
        

        ID = np.array(df[df["ID"] == f].ID)
        for x, y, z, i in zip(xSm, ySm, z, ID):
            # add info to new dataframe AND rescale the featues
            # to the original image size
            featsSm.loc[int(p)] = [x, y, int(z), int(i)]
            p += 1

    # select the best features
    dfSelectR, dfSelectSM, targetIDs = featureSelector(df, featsSm, featsMin = featsMin, dist = dist, maxfeat = featsMax)

    # 3D plot the smoothed and rough selected features 
    px.line_3d(dfSelectR, x="xPos", y="yPos", z="Sample", color="ID", title = "Raw selected features").show()
    px.line_3d(dfSelectSM, x="xPos", y="yPos", z="Sample", color="ID", title = "Smoothed selected features").show()
        
    # Warp images to create smoothened feature trajectories
    # NOTE the sparse image warp is already highly parallelised so not MP functions
    for imgpath in imgs:
        s = imgs.index(imgpath)
        ImageWarp(s, imgpath, dfSelectR, dfSelectSM, dirNLdest, border = 2, smoother = 0, order = 1)

    imgsMod = sorted(glob(dirNLdest + "*.png"))

    # for all the NL features SELECTED, extract them
    for n, img in enumerate(imgsMod):
        sampdf = dfSelectSM[dfSelectSM["Sample"] == n]
        featExtractor(NLSectDir, img, sampdf, sz)
    
    plotFeatureProgress([dfSelectR, dfSelectSM], imgsMod, dirNLdest + 'CombinedSmooth.jpg', sz, [3])

def featMatching(m, tarImg, refImg, prevImg = None, sz = 100, q = None):

    # Identifies the location of a feature in the next slice
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
    featureInfo = feature()
    featureInfo.refP = m.refP.copy()
    featureInfo.tarP = None
    featureInfo.dist = 0
    featureInfo.ID = m.ID

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
    shiftAccum = 0
    while True:

        # get the feature sections
        tarSect = getSect(tarImg, tarP, l)
        n += 1

        # this loop essentially keeps looking for features until it is "stable"
        # if a "stable" feature isn't found (ie little movement is needed to confirm) 
        # the feature) then eventually the position will tend towards the edges and fail
        # apply conditions: section has informaiton, ref and tar and the same size and threshold area is not empty
        if tarSect.size == 0 or \
            tarSect.shape != refSect.shape or \
                (np.sum((tarSect == 0) * 1) / tarSect.size) > 0.6 or \
                    n > 10:
            tarP = None
            break
        else:
            shift, error, _ = pcc(refSect, tarSect, upsample_factor=5)
            # tarP -= np.flip(shift)
            if error - errorStore > 0:
                # print("Shift = " + str(shiftAccum) + " error = " + str(error))
                break
            tarP -= np.flip(shift)
            errorStore = error

    # save specific feature for debugging
    if tarP is not None and m.ID == -1:

        # get the new feature 
        tarSectNew = getSect(tarImg, tarP, l)

        print("ShiftAll = " + str(shift))
        print("ID = " + str(m.ID))

        # ensure the name of the id is standardised 
        sampName = str(m.Samp)
        while len(sampName) < 4:
            sampName = "0" + sampName

        plt.imshow(np.hstack([refSect, tarSect, tarSectNew])); plt.show()

        cv2.imwrite('/Volumes/Storage/H653A_11.3/3/RealignedSamplesVSmallN/featMatchR'+str(m.ID)+'sampO'+str(sampName)+'.jpg', refSect)
        cv2.imwrite('/Volumes/Storage/H653A_11.3/3/RealignedSamplesVSmallN/featMatch'+str(m.ID)+'sampO'+str(sampName)+'.jpg', tarSect)
        cv2.imwrite('/Volumes/Storage/H653A_11.3/3/RealignedSamplesVSmallN/featMatch'+str(m.ID)+'sampM'+str(sampName)+'.jpg', tarSectNew)
        
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

def featureSelector(dfR, dfSm, featsMin, dist = 0, maxfeat = np.inf):

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
    sampleMax = np.max(dfR["Sample"])
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
            sampdf = dfR[dfR["Sample"] == s + extraLen]

            # create a data frame which contains the 
            sampHere, sampExtra = dfR[dfR["Sample"] == s + extraLen - n], dfR[dfR["Sample"] == s]
            sampsdfBoth = pd.merge(sampHere, sampExtra, left_on = "ID", right_on = "ID")

            # create a DF which contains all the features which pass through sample s
            try:    
                dfSamp = pd.concat([dfR[dfR["ID"] == f] for f in np.unique(sampsdfBoth.ID)])
                # get the feature IDs and their REMAINING lengths from the current sample
                _, featLen = np.unique(dfSamp[dfSamp["Sample"] >= s + extraLen - n].ID, return_counts = True)

            except: pass
            n += 1


        # calculate the average error per point between the smoothed and raw features
        errorStore = []
        errorStorem = []
        for i in sampsdfBoth.ID:
            rawFeat = np.c_[dfR[dfR["ID"] == i].xPos, dfR[dfR["ID"] == i].yPos]
            smFeat = np.c_[dfSm[dfSm["ID"] == i].xPos, dfSm[dfSm["ID"] == i].yPos]
            error = np.sum((rawFeat - smFeat)**2)/len(smFeat)
            errorStore.append(error)        # get the error of the entire feature
            
        # creat a dataframe with the error and length info
        featInfo = pd.DataFrame({'ID': sampsdfBoth.ID, 'xPos': sampsdfBoth.xPos_x, 'yPos': sampsdfBoth.yPos_y, 'error': errorStore, 'len': featLen})

        # sort the data first by length (ie longer values priortised) then by smoothness)
        # featInfoSorted = featInfo.sort_values(by = ['error', 'len'], ascending = [True, False])
        featInfoSorted = featInfo.sort_values(by = ['len', 'error'], ascending = [False, True])

        # get the positions of all the features on sample s ordered based on the number 
        # of samples it passes through 
        sampInfo = np.c_[sampsdfBoth.xPos_x, sampsdfBoth.yPos_y]

        # evaluate each feature in order of the distance it travels through the samples
        for idF, xPos, yPos, err, lenF in featInfoSorted.values:
            si = np.array([xPos, yPos])

            # for each feature position, check if it meets the distance criteria and that it
            # also has not been found yet, then append the information
            if (np.sqrt(np.sum((si - featPos)**2, axis = 1)) > dist).all() and len(np.where(featAll == idF)[0]) == 0:
                featPosID.append(idF)   # keep a sample by sample track of features being used
                featAll.append(idF)     # keep a total track of all features being used
                featPos.append(si)      # add in the new feature which meets the criteria

                # add the features from the current sample
                pdAllR.append(dfR[dfR["ID"]==idF][dfR[dfR["ID"]==idF]["Sample"] >= s])
                pdAllSm.append(dfSm[dfSm["ID"]==idF][dfSm[dfSm["ID"]==idF]["Sample"] >= s])

            # remove the initialising points
            if featPosID[0] == -1:
                del featPosID[0]
                del featPos[0]

            if len(featPosID) == maxfeat:
                break

        # create the new DF which contains the features which meet the criteria
        dfNew = pd.concat([dfSm[dfSm["ID"] == f] for f in featPosID])

        # find out how far the minimum feature goes for
        s += np.min([np.max((dfNew[dfNew["ID"] == i]).Sample)-s+1 for i in featPosID])

        # re-initialise all the features which are longer than the 
        # shortest feature
        featPosID = list(np.unique(dfNew[dfNew["Sample"] >= s].ID))
        featPos = list(np.c_[dfNew[dfNew["Sample"] == s].xPos, dfNew[dfNew["Sample"] == s].yPos])

    # collate the list of the final features
    finalFeats = np.unique(featAll)

    # get the data frame of all the features found
    dfFinalR = pd.concat(pdAllR)
    dfFinalSm = pd.concat(pdAllSm)

    return(dfFinalR, dfFinalSm, finalFeats)

def ImageWarp(s, imgpath, dfRaw, dfNew, dest, sz = 100, smoother = 0, border = 5, order = 2):

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

    name = nameFromPath(imgpath, 3)
    print("Warping " +  name)

    # load the correct ID images
    img = cv2.imread(imgpath)
    x, y, c = img.shape

    # get the sample specific feature info
    rawInfo = dfRaw[dfRaw["Sample"] == s]
    smoothInfo = dfNew[dfNew["Sample"] == s]

    # merge the info of the features which have the same ID
    allInfo = pd.merge(rawInfo, smoothInfo, left_on = 'ID', right_on = 'ID')

    # get the common feature positions
    rawFeats = np.c_[allInfo.xPos_x, allInfo.yPos_x]
    smFeats = np.c_[allInfo.xPos_y, allInfo.yPos_y]
    
    '''
    # add points in between features found, NOTE this definitely doesn't work!
    # sort the feature into order from top to bottom
    rawFeatsS = rawFeats[np.argsort(rawFeats[:, 1])]
    smFeatsS = smFeats[np.argsort(rawFeats[:, 1])]

    rawFeatsM = []
    for (xs, ys), (xe, ye) in zip(rawFeatsS[:-1], rawFeatsS[1:]):
        rawFeatsM.append(np.c_[np.linspace(xs, xe, 10), np.linspace(ys, ye, 10)])
    rawFeatsMC = np.concatenate(rawFeatsM)

    smFeatsm = []
    for (xs, ys), (xe, ye) in zip(smFeatsS[:-1], smFeatsS[1:]):
        smFeatsm.append(np.c_[np.linspace(xs, xe, 10), np.linspace(ys, ye, 10)])
    smFeatsMC = np.concatenate(smFeatsm)
    '''

    '''
    for nr, ns in zip(rawFeatsMC, smFeatsMC):
        cv2.circle(img, tuple(nr.astype(int)), 10, [255, 0, 0], 5)
        cv2.circle(img, tuple(ns.astype(int)), 10, [0, 255, 0], 3)
    '''

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

    # perform non-rigid deformation on the original sized image
    imgMod, imgFlow = sparse_image_warp(tftarImg, tfrawPoints, tfsmPoints, num_boundary_points=border, regularization_weight=smoother, interpolation_order=order)

    # convert the image and flow field into something useful
    imgMod = np.array(imgMod[0]).astype(np.uint8)
    Flow = np.array(imgFlow[0])

    # convert Flow tensor into a 3D array so that it can be saved
    # /displayed as some kind of image. 
    '''
    imgFlow = np.zeros(imgMod.shape)
    imgFlow[:, :, 0] = Flow[:, :, 0]
    imgFlow[:, :, 1] = Flow[:, :, 1]
    '''

    imgA = img.copy()
    imgModA = imgMod.copy()
    l = tile(sz, x, y)/2
    for n, (r, t, id) in enumerate(zip(rawFeats, smFeats, allInfo.ID)):
        rawP = r.astype(int)
        smP = t.astype(int)
        
        imgAS = getSect(imgA, rawP, l, bw = False)
        imgModAS = getSect(imgModA, smP, l, bw = False)

        # cv2.imwrite(dest + name + "feat" + str(id) + "sampR" + str(s) + ".jpg", imgAS)       # raw sections
        # cv2.imwrite(dest + name + "feat" + str(id) + "sampS" + str(s) + ".jpg", imgModAS)    # smooth sections

        # plt.imshow(np.hstack([imgAS, imgModAS])); plt.show()
        
        cv2.circle(imgA, tuple(smP), 8, [255, 0, 0], 4)
        cv2.circle(imgA, tuple(rawP), 8, [0, 0, 255], 4)

        cv2.circle(imgModA, tuple(rawP), 8, [0, 0, 255], 4)
        cv2.circle(imgModA, tuple(smP), 8, [255, 0, 0], 4)

        # add featue ID to image
        cv2.putText(imgModA, str(id), tuple(smP + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 5)
        cv2.putText(imgModA, str(id), tuple(smP + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)

    imgMod = imgModA
    # save the images
    cv2.imwrite(dest + name + ".png", imgMod)

def plotFeatureProgress(dfs, imgAll, dirdest, sz = 0, annos = [3]):

    # takes a data frame of the feature info and a list containing the images and 
    # returns two graphics. One is a picture of all the images and their features and 
    # the other is a 3D plot of the features only
    # Inputs:   (dfs), list of pandas data frames to plot on the same image
    #           (imgAll), list of the images either as numpy values or file paths
    #           (dirdst), path of the image to be saved
    #           (sz), size of the grid to use
    # Outputs:  (), image of all the samples with their features and lines connecting them 
    #               and a 3d line plot (plotly)

    # create a 3D plot of the feature progression of ALIGNED only samples

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

        # get the dims of the area search for the features
        l = int(tile(sz, x, y)/2)

        # annotate an image of the features and their matches
        imgAll = drawPoints(imgAll, df, sampleNo, anno, l)

        # save the linked feature images
        cv2.imwrite(dirdest, imgAll)

def drawPoints(img, df, sampleNos, annos = 3, l = 0):

    '''
    This function draws the feature positions (1), connecting lines + text (2) and bounding boxes
    (3) for all the features. Using different level flags you can control to what extent 
    the annotations are.

    Inputs: 
        (img), image to annotate on
        (df), pandas dataframe which stores information
        (sampleNos), number of samples in the image
        (l), length of the sides to draw for the bounding boxes
        (annos), draws features circles (1), connecting lines + numbering also (2) and bounding
        boxes also (3). 0 is no annotations. The different levels of annotations also 
        changes the colour of the lines etc
    
    Outputs:
        (imgAnno), annotated image
    '''

    x, y, c = img.shape
    y /= sampleNos

    for samp, i in enumerate(np.unique(df["ID"])):
        featdf = df[df["ID"] == i]
        tar = None
        for n, (fx, fy) in enumerate(zip(featdf.xPos, featdf.yPos)):

            tar = ((np.array([np.round(fx), np.round(fy)])) + np.array([y * n, 0])).astype(int)
            if n == 0 and annos > 0:
                # draw the first point on the first sample it appears as green
                cv2.circle(img, tuple(tar), 9, [int(85 * annos), 255, 255 - int(85 * annos)], int(annos * 2))

            elif annos > 0:
                # draw the next points as red
                cv2.circle(img, tuple(tar), 9, [int(85 * annos), 255, 255 - int(85 * annos)], int(annos * 2))

                # draw lines between points
                if annos > 1:
                    img = drawLine(img, ref, tar, colour = [255, 0, 0])

            if annos > 2:
                # draw bounding boxes showing the search area of the features
                if (tar-l == np.array([2686.0, 1052.0])).all():
                    print("WAIG")
                img = drawLine(img, tar - l, tar + [l, -l], blur = 2, colour=[int(85 * annos), 255, 0])
                img = drawLine(img, tar - l, tar + [-l, l], blur = 2, colour=[int(85 * annos), 255, 0])
                img = drawLine(img, tar + l, tar - [l, -l], blur = 2, colour=[int(85 * annos), 255, 0])
                img = drawLine(img, tar + l, tar - [-l, l], blur = 2, colour=[int(85 * annos), 255, 0])

            # label the images with their id and the number of features present
            if annos > 0:
                cv2.putText(img, str(int(i)), tuple(tar-[10, 5]), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 4)
                cv2.putText(img, str(int(i)), tuple(tar-[10, 5]), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)
            
            # reasign the information
            ref = tar
        
        # make the final point pink
        if tar is not None and annos > 0:
            cv2.circle(img, tuple(tar), 9, [255, int(85 * annos), 255], int(annos * 2)) 

    # add the sample number and the number of features on the sample   
    if annos > 1: 
        for s in np.unique(df["Sample"]):
            cv2.putText(img, "Samp_" + str(int(s)) + " no_" + str(len(df[df["Sample"] == s])), tuple(np.array([y * s, 0]).astype(int) + [50, 100]), cv2.FONT_HERSHEY_SIMPLEX, 3, [255, 255, 255], 6)
        
    return(img)

def featExtractor(dest, imgPath, info, sz):

    '''
    This function takes an image and all the feature positions as a pandas data 
    frame and extracts from the image the tissue segment and saves it in the 
    correct directory

        Inputs: \n
    (dest), home directory to save the information\n
    (imgPath), the path to the single sample to extract the sections from\n
    (info), the pandas data frame of that sample info\n
    (sz), the size of the segment used to identify that feature

        Outputs: \n
    (), saves the section in the directory as its feature name\n
    '''

    # get the sample number
    samp = np.array(info.Sample)[0]

    # get info
    img = cv2.imread(imgPath)
    x, y, c = img.shape
    l = tile(sz, x, y)/2
    

    # go through all the info for each sample and extract all the features
    # and save in directories to categorise them into features, named after
    # the sample they were found in
    for i, d in info.iterrows():
        featSect = getSect(img, np.array([d.xPos, d.yPos]), l, False)
        featdir = dest + str(int(d.ID)) + "/"
        made = dirMaker(featdir)
        if made:
            firstSamp = drawPoints(img.copy(), info[info["ID"] == d.ID], 1, 3, l)
            cv2.imwrite(featdir + "_referenceImage.jpg", firstSamp, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
        cv2.imwrite(featdir + nameFromPath(imgPath, 3) + ".png", featSect)

if __name__ == "__main__":

    # with 2 2D images, interpolate between them for given points
    dirHome = '/Volumes/USB/H671B_18.5/'
    dirHome = '/Volumes/USB/Test/'
    dirHome = '/Volumes/Storage/H710C_6.1/'
    dirHome = '/Volumes/Storage/H653A_11.3/'

    # NOTE fork is considered unstable but it is an order of magnitude 
    # faster than using spawn.... HOWEVER this only works with the process
    # MP method, not pool.....
    multiprocessing.set_start_method("fork")


    size = 3
    cpuNo = True

    nonRigidAlign(dirHome, size, cpuNo)
