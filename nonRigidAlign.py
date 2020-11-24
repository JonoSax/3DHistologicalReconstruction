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
    destRigidAlign = home + "/RealignedSamplesV100/"
    dirfeats = home + "/infoNL/"
    destNLALign = home + "/NLAlignedSamplesV100/"

    dirMaker(destRigidAlign)
    dirMaker(dirfeats)
    dirMaker(destNLALign)

    '''
    imgs = sorted(glob(imgsrc + "*.png"))[:20]

    imgs = sorted(glob(destNLALign + "*feat0sampR*"))
    orig = cv2.imread(imgs[10])
    new = cv2.imread(imgs[11])
    shift, _, _ = pcc(orig, new, upsample_factor=1)
    a, b = moveImg(orig, new, shift)
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.imshow(np.hstack([orig, new]))
    ax2 = plt.subplot(2, 1, 2)
    ax2.imshow(np.hstack([a, b]))
    plt.show()

    shift, _, _ = pcc(a, b, upsample_factor=1)
    print(shift)    
    '''

    imgs = sorted(glob(imgsrc + "*.png"))[:100]
    sect = 300           # the proporptional area of the image to search for features
    dist = 50           # distance between the sift features
    featsMin = 5       # min number of samples a feature must pass through for use to NL deform

    # Find the continuous features throught the samples
    # contFeatFinder(imgs, dirfeats, destRigidAlign, cpuNo = True, plotting = True, sz = sect, dist = dist, connect = featsMin)
    
    # perform a rigid alignment on the tracked features
    aligner(imgs, dirfeats, imgsrc, destRigidAlign, cpuNo = False, errorThreshold=50)
    
    featsMax = 8       # max number of samples which meet criteria to be used
    distFeats = 250     # distance (pixels) between the final features

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
            confirmInfo = []
            for j in job:
                j.join()

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
        confirmedFeatures = matchMaker(confirmInfos, dist = dist, tol = 1, cpuNo = cpuNo, anchorPoints=5, distCheck=False)
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
        plotFeatureProgress(df, imgs, destImg + 'CombinedRough.jpg', sz, 2)

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
    imgs = sorted(glob(dirimgs + "*.png"))

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

    # create the 3D plot of the aligned features
    px.line_3d(df, x="xPos", y="yPos", z="Sample", color="ID", title = "All aligned features").show()
    
    # pick the best features for warping --> this is important to ensure that the 
    # features are not too close to eachother or else they will cause impossible 
    # warping
    shape = cv2.imread(imgs[0]).shape
    df, targetIDs = featureSelector(df, shape, dist = dist, maxfeat = featsMax)

    # create the 3D plot of the selected aligned features
    px.line_3d(df, x="xPos", y="yPos", z="Sample", color="ID", title = "Selected aligned features").show()

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
            featsSm.loc[int(p)] = [x, y, z, i]
            p += 1

    # 3D plot the smoothed features 
    px.line_3d(featsSm, x="xPos", y="yPos", z="Sample", color="ID", title = "Smoothed features").show()
    
    # taking the features found and performing non-rigid alignment
    infoStore = []
    
    # NOTE the sparse image warp is already highly parallelised
    if cpuNo is not False:
        with multiprocessing.Pool(processes=cpuNo) as pool:
            pool.starmap(ImageWarp, zip(np.arange(len(imgs)), imgs, repeat(df), repeat(featsSm), repeat(dirdest), repeat(1e6)))

    else:
        # NOTE double chckec that i'm even modifying the images!!
        for s, imgpath in enumerate(imgs):
            ImageWarp(s, imgpath, df, featsSm, dirdest, border = 2, smoother = 0, order = 2)
    
    imgsMod = sorted(glob(dirdest + "*.png"))
    
    plotFeatureProgress([df, featsSm], imgsMod, dirdest + 'CombinedSmooth.jpg', sz, [3])

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

    # get the feature sections
    tarSect = getSect(tarImg, tarP, l)

    # apply conditions: section has informaiton, ref and tar and the same size and threshold area is not empty
    if tarSect.size == 0 or tarSect.shape != refSect.shape or (np.sum((tarSect == 0) * 1) / tarSect.size) > 0.3:
        tarP = None
    else:
        shift, _, _ = pcc(refSect, tarSect, upsample_factor=20)
        # tarP -= np.flip(shift)
        tarP -= shift

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

        cv2.imwrite(dest + name + "feat" + str(id) + "sampR" + str(s) + ".jpg", imgAS)       # raw sections
        cv2.imwrite(dest + name + "feat" + str(id) + "sampS" + str(s) + ".jpg", imgModAS)    # smooth sections

        # plt.imshow(np.hstack([imgAS, imgModAS])); plt.show()
        
        cv2.circle(imgA, tuple(smP), 8, [255, 0, 0], 4)
        cv2.circle(imgA, tuple(rawP), 8, [0, 0, 255], 4)

        cv2.circle(imgModA, tuple(rawP), 8, [0, 0, 255], 4)
        cv2.circle(imgModA, tuple(smP), 8, [255, 0, 0], 4)

        # add featue ID to image
        cv2.putText(imgModA, str(id), tuple(smP + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 5)
        cv2.putText(imgModA, str(id), tuple(smP + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)
        
    # create images combining the unwarped and warped images
    imgComb = np.hstack([imgA, imgModA])
    cv2.imwrite(dest + name + "comb.jpg", imgComb)

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
    yp, xp = np.round(mpos).astype(int)
    xs = int(np.clip(xp-l, 0, x)); xe = int(np.clip(xp+l, 0, x))
    ys = int(np.clip(yp-l, 0, y)); ye = int(np.clip(yp+l, 0, y))
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
    dirHome = '/Volumes/Storage/H653A_11.3/'
    dirHome = '/Volumes/Storage/H710C_6.1/'

    # NOTE fork is considered unstable but it is an order of magnitude 
    # faster than using spawn.... HOWEVER this only works with the process
    # MP method, not pool.....
    multiprocessing.set_start_method("fork")


    size = 3
    cpuNo = True

    nonRigidAlign(dirHome, size, cpuNo)
