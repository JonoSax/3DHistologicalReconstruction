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
from scipy.signal import savgol_filter as svf
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from time import time

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
    destRigidAlign = home + "/RealignedSamplesSmall/"
    dirfeats = home + "/infoNL/"
    destNLALign = home + "/NLAlignedSamplesVSmall/"

    dirMaker(destRigidAlign)
    dirMaker(dirfeats)
    dirMaker(destNLALign)

    imgs = sorted(glob(imgsrc + "*.png"))
    sect = 50           # the proporptional area of the image to search for features
    dist = 20           # distance between the sift features
    featsMin = 10       # min number of samples a feature must pass through for use to NL deform
    featsMax = 10       # max number of samples which meet criteria to be used
    distFeats = 200     # distance (pixels) between the final features

    # Find the continuous features throught the samples
    # contFeatFinder(imgs, dirfeats, destRigidAlign, cpuNo = True, plotting = True, sz = sect, dist = dist, connect = featsMin)
    
    # perform a rigid alignment on the tracked features
    # aligner(imgs, dirfeats, imgsrc, destRigidAlign, cpuNo=False, errorThreshold=50)
    
    # perform a non-linear deformation on the samples based on a smoothed trajectory of the features
    # through the samples
    nonRigidDeform(destRigidAlign, destRigidAlign, destNLALign, cpuNo = False, dist = distFeats, sz = sect, featsMax = featsMax)

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

    # use the first image in the sequence as the reference image
    refImg = cv2.imread(imgs[0])
    refName = nameFromPath(imgs[0], 3)

    # all the image are the same size
    x, y, c = refImg.shape

    prevImg1 = None
    prevImg2 = None
    continuedFeatures = []

    # sequantially identify new previous features in each sample
    for sampleNo, tarPath in enumerate(imgs[1:]):

        # load the resized image 
        tarName = nameFromPath(tarPath, 3)
        tarImg = cv2.imread(tarPath, 3)

        print("Matching " + tarName + " to " + refName)

        # ---- find the position of the feature in the next slice based on the previous informaiton ----

        startFeatMatch = time()
        if cpuNo == False:
            confirmInfo = []
            for m in matchedInfo:
                confirmInfo.append(featMatching(m, allMatchedInfo, tarImg, refImg, [prevImg1, prevImg2], sz))

        else:
            # Using Pool
            '''
            with multiprocessing.Pool(processes=cpuNo) as pool:
                confirmInfo = pool.starmap(featMatching, zip(matchedInfo, repeat(tarImg), repeat(refImg), repeat(sz)))
            '''

            # Using Process
            job = []
            qs = {}
            for n, m in enumerate(matchedInfo):
                qs[n] = multiprocessing.Queue()
                j = multiprocessing.Process(target=featMatching, args = (m, allMatchedInfo, tarImg, refImg, [prevImg1, prevImg2], sz, qs[n]))
                job.append(j)
                j.start()
            confirmInfo = []
            for j in job:
                j.join()

            for q in qs:
                confirmInfo.append(qs[q].get())

        featMatchTime = time() - startFeatMatch
        
        # unravel the list of features produced by the continuous features
        confirmInfos = []
        for info in confirmInfo:
            if info is None:
                continue
            for i in info:
                confirmInfos.append(i)


        # ensure that the features found are still spaitally cohesive
        startMatchMaker = time()
        continuedFeatures = matchMaker(confirmInfos, dist = dist, tol = 1, cpuNo = cpuNo, anchorPoints=5, distCheck=False)
        matchMakerTime = time() - startMatchMaker

        # featMatchimg = nameFeatures(refImg, tarImg, continuedFeatures, combine = True)
        # plt.imshow(featMatchimg); plt.show()

        # ---- find new features between the samples ----

        # find spatially coherent sift features between corresponding images
        startAllFeatSearch = time()
        matchedInfo = allFeatSearch(refImg, tarImg, continuedFeatures, dist = dist, cpuNo = cpuNo, gridNo=30)[0]
        allFeatSearchTime = time() - startAllFeatSearch

        # ensure that each feature is identified
        for m in matchedInfo:
            if m.ID is None:
                # keep track of the feature ID
                m.ID = featNo
                allMatchedInfo[m.ID] = {}
                featNo += 1
            m.Samp = sampleNo
            allMatchedInfo[m.ID][sampleNo] = m

        # featMatchimg = nameFeatures(refImg, tarImg, matchedInfo, combine = True)
        # plt.imshow(featMatchimg); plt.show()

        # reasign the target info as the reference info
        refName = tarName
        prevImg2 = prevImg1
        prevImg1 = refImg
        refImg = tarImg

        print("     featMatchMaker = " + str(featMatchTime))
        print("     matchMaker = " + str(matchMakerTime))
        print("     allFeatSearchTime = " + str(allFeatSearchTime) + "\n")

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
        plotFeatureProgress(df, imgs, destImg + 'CombinedRough.jpg', sz, "Unaligned", 2)

def nonRigidDeform(dirimgs, dirfeats, dirdest, cpuNo = False, dist = 100, sz = 0, featsMax = None):

    '''
    This function takes the continuous feature sets found in contFeatFinder
    and uses them to non-rigidly warp the 

    Inputs:   
        (dirimgs), directory of all the images
        (dirfeats), directory of the features 
        (dirdest), path to save the NL deformed info
        (cpuNo), cores to use or False to serialise
        (scl), resolution scale factor
        (s), section size used for feature finding
    
    Outputs:  
        (), warp the images and save
    '''

    # get the image paths
    imgs = sorted(glob(dirimgs + "*.png"))[:30]

    # get the new dictionaries, load them into a pandas dataframe
    refFeats = sorted(glob(dirfeats + "*.reffeat"))
    tarFeats = sorted(glob(dirfeats + "*.tarfeat"))
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

    # pick the best features for warping --> this is important to ensure that the 
    # features are not too close to eachother or else they will cause impossible 
    # warping
    shape = cv2.imread(imgs[0]).shape
    df, targetIDs = featureSelector(df, shape, dist = dist, maxfeat = featsMax)

    # create a new dataframe for the smoothed feature positions
    featsSm = pd.DataFrame(columns=["xPos", "yPos", "Sample", "ID"])

    p = 0       # create data frame count 
    for f in targetIDs:
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
        tck, u = splprep([xp, yp, z], s = 1000)
        # u_fine = np.linspace(0,1,num_true_pts)
        # x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
        xSm, ySm, _ = splev(u, tck)
        

        ID = np.array(df[df["ID"] == f].ID)
        for x, y, z, i in zip(xSm, ySm, z, ID):
            # add info to new dataframe AND rescale the featues
            # to the original image size
            featsSm.loc[p] = [x, y, z, i]
            p += 1

    # taking the features found and performing non-rigid alignment
    infoStore = []
    
    if cpuNo is not False:
        with multiprocessing.Pool(processes=cpuNo) as pool:
            pool.starmap(ImageWarp, zip(np.arange(len(imgs)), imgs, repeat(df), repeat(featsSm), repeat(dirdest), repeat(1e6)))

    else:
        # NOTE double chckec that i'm even modifying the images!!
        for s, imgpath in enumerate(imgs):
            ImageWarp(s, imgpath, df, featsSm, dirdest, border = 2, smoother = 0, order = 2)
    
    imgsMod = sorted(glob(dirdest + "*.png"))
    
    plotFeatureProgress([df, featsSm], imgsMod, dirdest + 'CombinedSmooth.jpg', sz, ["Rigid fit", "Non-Linear + Smoothed"], [1, 3])

def featMatching(m, allMatchedInfo, tarImg, refImg, prevImg = None, sz = 50, q = None):

    # Identifies the location of a feature in the next slice
    # Inputs:   (m), feature object of the previous point
    #           (tarImg), image of the next target sample
    #           (refImg), image of the current target sample
    #           (prevImg), image of the previous reference sample
    #           (sz), the equivalent number of windows to create
    # Outputs:  (featureInfo), feature object of feature (if identified
    #               in the target image)

    # from the previous sample, use the last known positions as an anitial reference point 
    # to identify the area of the new feature

    x, y, c = tarImg.shape
    l = tile(sz, x, y)

    tarSectBW = getSect(tarImg, m.tarP, l)

    # get the section from the previous reference image
    prevSectBW = []

    # get the identified feature from the reference image
    refSect = getSect(refImg, m.tarP, l)
    prevSectBW.append(refSect)

    keys = list(allMatchedInfo[m.ID].keys())[:len(prevImg)]

    # get the feature from the previous images
    for img, k in zip(prevImg, keys):

        if type(img) == type(None):
            continue
        # get all the previous information
        pm = allMatchedInfo[m.ID][k]
        prevSectBW.append(getSect(img, pm.refP, l))

    # if a significant majority of the section is foreground, 
    # calculate the shift and create the feature object

    # if there is no image available return 
    if tarSectBW.size == 0:
        allfeatureInfo = None

    elif (np.sum((tarSectBW == 0) * 1) / tarSectBW.size) < 0.7:

        # minimise the error of the feature positon between the target, current and previous
        # samples
        # shift = featSearch(tarSectBW, prevSectBW)
        shift = featSearch(tarSectBW, [refSect])

        # only consider the feature in the prevous sample
        # shift, _, _ = pcc(prevSectBW[0], tarSectBW, upsample_factor=20)
            
        # create the new feature object for the feature in the target image
        featureInfo = feature()
        featureInfo.refP = m.tarP 
        featureInfo.tarP = m.tarP - np.flip(shift)
        featureInfo.dist = 0
        featureInfo.ID = m.ID
        allfeatureInfo = [featureInfo]

    # if mostly background then return nothing
    else:
        allfeatureInfo = None
    
    '''
    # compare all the images used
    tarImgSectN = getSect(tarImg, featureInfo.tarP, sz)
    plt.imshow(np.hstack([tarImgSectN, tarSectBW, prevSectBW[0], prevSectBW[1]]), cmap = 'gray'); plt.show()
    '''

    if q == None:
        return(allfeatureInfo)
    else:
        q.put(allfeatureInfo)

def featSearch(tar, prev):

    # this function is minimising the error between the searches for a feature
    # in the next target slice by comparing the target features with the current and 
    # previous reference slice. This is in order to reduce sudden shifts in positon 
    # Inputs:   (tar), target sample area
    #           (prev), all previous target sample areas
    # Outputs:  (shift), the pixel shift required to minimise errors

    # NOTE: key thing is to minimse the amount of shift required in the target image
    # to minimise the error value of the convolution match

    for p in prev:
        if tar.shape != p.shape:
            # if the shape of the inputs aren't the same then the feature is too close to the edge
            # so return a large shift which will cause it to be eliminated in matchmaker
            return(np.array([10000, 10000]))

    shift, error = objectiveFeatPos(None, tar, prev, False)

    # NOTE I don't have a good way to minimise the error yet.... 
    # shiftOpt = minimize(objectiveFeatPos, (0, 0), args = (tar, prev, True), method = 'Nelder-Mead', tol = 1e-6)
    # shift = shiftOpt.x

    return(shift)

def objectiveFeatPos(move, *args):

    '''
    Function that can either be minimised or used to calculate a shift and error value for
    the cross-correlation between sample sections
    Inputs:   (move), value to minimise
              (args), the respective areas of searching
                  0 is the target
                  1 arg is a list of previous inputs 
                  2 is a boolean whether its being used in an optimising (true) or solving
                  (false) scheme
    Outputs:  (counterShift), error between the difference between the movement required
                  and the average shift of the features
    '''

    tar = args[0]       # target image
    prev = args[1]      # list of reference images    
    opt = args[2]       # optimisation boolean

    # get the optimised shifts of the target image compared to both the tar and all previous 
    # secions

    '''
    errorAccum = 0
    if type(prev) != list:
        prev = [prev]
    for n, p in enumerate(prev):
        if p.shape != tar.shape:
            continue
        errorAccum += moveImgError(p, tar, move, upsampleFactor=1) * (1/(n+1)**2)
    '''
    
    shiftPrev = []
    errorPrev = []
    if type(prev) != list:
        prev = [prev]
    for n, p in enumerate(prev):
        if p.shape != tar.shape:
            continue

        if opt:
            # move the target image relative to the reference image
            p, tar = moveImgError(p, tar, move, upsampleFactor=20)

        # get optimal position for a cross-corrleation shift and its error
        shift, error, _ = pcc(p, tar, return_error=True, upsample_factor=20)

        # perform a weighted averaging of the samples
        for s, e in zip(repeat(shift, len(prev)-n), repeat(error, len(prev)-n)):
            shiftPrev.append(s)
            errorPrev.append(e)
    
    # find the average shift position from all previous matches
    shiftAvg = np.mean(shiftPrev, axis = 0)
    errorAvg = np.mean(errorPrev, axis = 0)

    if opt:
        # create the error value to minmimise, NOTE this could be squared to penalise large changes?
        counterError = np.sum(abs(move - shiftAvg))
        return(errorAvg)
    else:
        return(shiftAvg, errorAvg)

def featureSelector(df, shape, dist = 0, maxfeat = np.inf):

    '''
    This function finds features which are clustered around the centre and returns only those which 
    are a threshold distance away from other features and up to a certain number

    Inputs:\n
    (df), pandas data frame\n
    (shape), shape of the images\n
    (sz), proporption of the image to use for spacing out the features\n
    (maxfeat) max number of features to be returned

    Outputs:\n
    (dfNew), pandas data frame of the features which meet the requirements 
    (targetIDs), the feature IDs that meet the criteria
    '''
    
    # set the maximum iterations
    sampleMax = np.max(df["Sample"])
    s = 0
    
    # initialise the searching positions
    featPos = []
    featPosID = []
    featPosID.append(-1)
    featPos.append(np.array([-1000, -1000]))
    featAll = []
    while s < sampleMax:

        # get all the feature positions on the sample
        sampdf = df[df["Sample"] == s]

        # get the position information into a numpy matrix ordered based upon the lenght 
        # of the features

        # create a DF which contains all the features which pass through sample s
        dfSamp = pd.concat([df[df["ID"] == f] for f in np.unique(sampdf.ID)])

        # get the feature IDs and their remaining lengths from the current sample
        featID, featLen = np.unique(dfSamp[dfSamp["Sample"] >= s].ID, return_counts = True)

        # order the features based on their length
        featSort = np.argsort(-featLen)
        featIDS = featID[featSort]
        featLenS = featLen[featSort]

        # get the positions of all the features on sample s ordered based on the number 
        # of samples it passes through 
        sampInfo = np.c_[sampdf.xPos, sampdf.yPos][featSort]

        # evaluate each feature in order of the distance it travels through the samples
        for fi, si in zip(featIDS, sampInfo):
            # np.c_[df[df["ID"] == n].xPos, df[df["ID"] == n].yPos]

            # for each feature position, check if it meets the distance criteria 
            # and then append the information
            if (np.sqrt(np.sum((si - featPos)**2, axis = 1)) > dist).all():
                featPosID.append(fi)
                featPos.append(si)
                featAll.append(fi)      # append the feature

            # remove the initialising points
            if featPosID[0] == -1:
                del featPosID[0]
                del featPos[0]

            if len(featPosID) == maxfeat:
                break

        # create the new DF which contains the features which meet the criteria
        dfNew = pd.concat([df[df["ID"] == f] for f in featPosID])

        # find out how far the minimum feature goes for
        s += np.min([np.max((dfNew[dfNew["ID"] == i]).Sample)-s+1 for i in featPosID])

        # re-initialise all the features which are longer than the 
        # shortest feature
        featPosID = list(np.unique(dfNew[dfNew["Sample"] >= s].ID))
        featPos = list(np.c_[dfNew[dfNew["Sample"] == s].xPos, dfNew[dfNew["Sample"] == s].yPos])


    # collate the list of the final features
    finalFeats = np.unique(featAll)

    # get the data frame of all the features found
    dfFinal = pd.concat([df[df["ID"] == f] for f in finalFeats])

    return(dfFinal, finalFeats)

def ImageWarp(s, imgpath, dfRaw, dfNew, dest, smoother = 0, border = 5, order = 2):

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
    # x, y, c = (np.array(img.shape) * scl).astype(int)
    # img = cv2.resize(img, tuple([y, x]))

    # get the sample specific feature info
    refInfo = dfRaw[dfRaw["Sample"] == s]
    tarInfo = dfNew[dfNew["Sample"] == s]

    # merge the info of the features which have the same ID
    allInfo = pd.merge(refInfo, tarInfo, left_on = 'ID', right_on = 'ID')

    # get the common feature positions
    refFeats = np.c_[allInfo.xPos_x, allInfo.yPos_x]
    tarFeats = np.c_[allInfo.xPos_y, allInfo.yPos_y]

    # if there is no information regarding the positions to warp images,
    # assume images don't need warping therefore save the original image as final
    if len(refFeats) == 0 or len(tarFeats) == 0:
        cv2.imwrite(dest + name + ".png", img)
        return
        
    # flip the column order of the features for the sparse matrix calculation
    reff = np.fliplr(np.unique(np.array(refFeats), axis = 0))
    tarf = np.fliplr(np.unique(np.array(tarFeats), axis = 0))

    # ensure the inputs are 4D tensors
    tfrefPoints = np.expand_dims(reff.astype(float), 0)
    tftarPoints = np.expand_dims(tarf.astype(float), 0)
    tftarImg = np.expand_dims(img, 0).astype(float)

    # perform non-rigid deformation on the original sized image
    imgMod, imgFlow = sparse_image_warp(tftarImg, tfrefPoints, tftarPoints, num_boundary_points=border, regularization_weight=smoother, interpolation_order=order)

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
    sp = 50
    for n, (r, t) in enumerate(zip(refFeats, tarFeats)):
        refP = r.astype(int)
        tarP = t.astype(int)
        
        imgAS = getSect(imgA, refP, sp, bw = False)
        imgModAS = getSect(imgModA, tarP, sp, bw = False)

        cv2.imwrite(dest + name + "feat" + str(n) + "sampN" + str(s) + ".jpg", imgAS)
        cv2.imwrite(dest + name + "feat" + str(n) + "sampM" + str(s) + ".jpg", imgModAS)

        # plt.imshow(np.hstack([imgAS, imgModAS])); plt.show()
        
        cv2.circle(imgA, tuple(tarP), 8, [255, 0, 0], 4)
        cv2.circle(imgA, tuple(refP), 8, [0, 0, 255], 4)

        cv2.circle(imgModA, tuple(refP), 8, [0, 0, 255], 4)
        cv2.circle(imgModA, tuple(tarP), 8, [255, 0, 0], 4)
        
    img = imgA
    imgMod = imgModA

    imgComb = np.hstack([imgA, imgModA])
    cv2.imwrite(dest + name + "comb.jpg", imgComb)

    # save the images
    cv2.imwrite(dest + name + ".png", imgMod)

def plotFeatureProgress(dfs, imgAll, dirdest, sz = 0, titles = [""], annos = [3]):

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

    if type(titles) != list:
        titles = [titles]
    
    for df, title, anno in zip(dfs, titles, annos):
        px.line_3d(df, x="xPos", y="yPos", z="Sample", color="ID", title = title).show()
    
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
        (l), length fo the sides to draw for the bounding boxes
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
        for n, (fx, fy, s) in enumerate(zip(featdf.xPos, featdf.yPos, featdf.Sample)):

            tar = ((np.array([np.round(fx), np.round(fy)])) + np.array([y * s, 0])).astype(int)
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

def getSect(img, mpos, l, bw = True):

    # get the section of the image based off the feature object and the tile size
    # Inputs:   (img), numpy array of image
    #           (mpos), position
    #           (s), tile size 
    # Outputs:  (imgSect), black and white image section

    x, y, c = img.shape      

    # get target position from the previous match, use this as the 
    # position of a possible reference feature in the next image
    yp, xp = mpos.astype(int)
    xs = np.clip(xp-l, 0, x); xe = np.clip(xp+l, 0, x)
    ys = np.clip(yp-l, 0, y); ye = np.clip(yp+l, 0, y)
    sect = img[xs:xe, ys:ye]

    # NOTE turn into black and white to minimise the effect of colour
    if bw:
        sectImg = np.mean(sect, axis = 2).astype(np.uint8)
    else:
        sectImg = sect

    return(sectImg)


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
