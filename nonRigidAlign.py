import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from HelperFunctions.Utilities import nameFromPath, matchMaker, nameFeatures, drawLine, dirMaker, drawLine
from HelperFunctions.SP_AlignSamples import aligner
from tensorflow_addons.image import sparse_image_warp
import plotly.express as px
import multiprocessing
from itertools import repeat
from skimage.registration import phase_cross_correlation as pcc
import pandas as pd
from scipy.signal import savgol_filter as svf
from scipy.interpolate import splprep, splev


# for each fitted pair, create an object storing their key information
class feature:
    def __init__(self, refP = None, tarP = None, dist = None, trainIdx = None, descr = None, ID = None, res = 0):
        # the position of the match on the reference image
        self.refP = refP

        # the position of the match on the target image
        self.tarP = tarP

        # eucledian error of the difference in gradient fields
        self.dist = dist

        # the index value of this feature from the sift search
        self.trainIdx = trainIdx

        # gradient descriptors
        self.descr = descr

        # the feature number 
        self.ID = ID

        # resolution of the image used
        self.res = res


    def __repr__(self):
            return repr((self.dist, self.refP, self.tarP, self.descr))

def nonRigidAlign(dirHome, size, cpuNo):

    home = dirHome + str(size)

    src = home + "/pngImages/"
    destRigidAlign = home + "/NLalignedSamples/"
    destFeats = home + "/infoNL/"

    dirMaker(destRigidAlign)
    dirMaker(destFeats)

    imgs = sorted(glob(src + "*.png"))[:10]
    scale = 0.2
    win = 5
    sect = 500

    # NOTE save feats as txt some how...
    feats = contFeatFinder(imgs, destRigidAlign, r = scale, plotting = True, win = win, s = sect)
    
    # aligner(imgs, destFeats, imgs, destRigidAlign, cpuNo=5)
    
    nonRigidDeform(feats, imgs, destRigidAlign, cpuNo, scale, win, sect)

def contFeatFinder(imgs, dest, cpuNo = False, r = 1, plotting = False, win = 9, s = 100):

    # This function takes images and finds features that are continuous
    # between the samples
    # Inputs:   (imgs), list of directories of the images to be processed
    #               in sequential order for feature identifying
    #           (cpuNo), number of cores to parallelise the task. 
    #           (r), factor to resize images
    #           (plotting), boolean whether to plot the feature info through
    #               the samples
    #           (s), the equivalent area which a continuous feature will be searched for 
    #               if this many tiles were created on the section
    #           NOTE it appears ATM that it is faster to serialise...
    # Outputs:  (df), panda data frame which contains the position of the 
    #               features in each sample and the ID of the feature

    # initialise the feature finding objects 
    sift =  cv2.xfeatures2d.SIFT_create() 
    bf = cv2.BFMatcher()

    # intialise objects to store and track info
    matchedInfo = []
    allMatchedInfo = {}
    featNo = 0

    # use the first image in the sequence as the reference image
    refImg = cv2.imread(imgs[0])
    refName = nameFromPath(imgs[0])

    # all the image are the same size
    x, y, c = (np.array(refImg.shape) * r).astype(int)

    # resize the ref img
    refImg = cv2.resize(refImg, (int(y), int(x)))

    # compute all the features 
    kp_ref, des_ref = sift.detectAndCompute(cv2.resize(refImg, (y, x)), None)

    # sequantially identify new previous features in each sample
    for sampleNo, tarPath in enumerate(imgs[1:]):

        # load the resized image 
        tarName = nameFromPath(tarPath)
        tarImg = cv2.resize(cv2.imread(tarPath), (int(y), int(x)))

        print("Matching " + tarName + " to " + refName)

        # ---- identify features which were in the previous sample ----

        if cpuNo is False:
            confirmInfo = []
            for m in matchedInfo:
                confirmInfo.append(featMatching(m, tarImg, refImg, s))

        else:
            with multiprocessing.Pool(processes=cpuNo) as pool:
                confirmInfo = pool.starmap(featMatching, zip(matchedInfo, repeat(tarImg), repeat(refImg), repeat(s)))

        # unravel the list of features produced by the continuous features
        confirmInfos = []
        for info in confirmInfo:
            if info is None:
                continue
            for i in info:
                confirmInfos.append(i)

        continuedFeatures = matchMaker(confirmInfos, dist = 20, tol = 1, cpuNo = False)
        # featMatchimg = nameFeatures(refImg, tarImg, continuedFeatures, combine = True)
        # plt.imshow(featMatchimg); plt.show()

        # ---- find new features in the sample ----

        # find the all matching features in each slice
        kp_tar, des_tar = sift.detectAndCompute(tarImg, None)
        matches = bf.match(des_ref, des_tar)

        # convert the points into the feature object
        resInfo = []
        for m in matches:
            featureInfo = feature()
            # store the feature information as it appears on the original sized image
            featureInfo.refP = np.array(kp_ref[m.queryIdx].pt) 
            featureInfo.tarP = np.array(kp_tar[m.trainIdx].pt) 
            featureInfo.dist = np.array(m.distance)
            featureInfo.trainIdx = m.trainIdx
            featureInfo.descr = np.array(des_tar[m.trainIdx])
            resInfo.append(featureInfo)

        # find related features between both images
        # use a larger distance and tolerance between the features
        # in order to find as many features as possibl ethat are distributed 
        # across the entire sample 
        # use the confirmed features as a starging point
        matchedInfo = matchMaker(resInfo, continuedFeatures, dist = 20, tol = 0.2, cpuNo = False, r = 10)

        # ensure that each feature is identified
        for m in matchedInfo:
            if m.ID is None:
                # keep track of the feature ID
                m.ID = featNo
                allMatchedInfo[m.ID] = {}
                featNo += 1
            allMatchedInfo[m.ID][sampleNo] = m

        '''
        for aMi in allMatchedInfo[16]:
            m = allMatchedInfo[16][aMi]
            print("ref = " + str(np.round(m.refP)) + " tar = " + str(np.round(m.tarP)))
        '''
        # featMatchimg = nameFeatures(refImg, tarImg, matchedInfo, combine = True)
        # plt.imshow(featMatchimg); plt.show()

        # reasign the target info as the reference info
        refName = tarName
        kp_ref, des_ref = kp_tar, des_tar 
        refImg = tarImg
        

    # -------- Plotting and info about the features --------

    # arrange the data in a way so that it can be plotted in 3D
    maxNo = np.max([len(allMatchedInfo[mi]) for mi in allMatchedInfo])
    
    for n in range(maxNo):
        print("Number of " + str(1+n) + "/" + str(maxNo) + " linked features = " + str(len(np.where(np.array([len(allMatchedInfo[mi]) for mi in allMatchedInfo]) == n+1)[0])))
    
    imgStack = []
    for i in imgs:
        imgStack.append(cv2.resize(cv2.imread(i), (y, x)))

    # create a data frame of all the features found 
    df = pd.DataFrame(columns=["xPos", "yPos", "Sample", "ID"])
    i = 0
    for m in allMatchedInfo:
        # if there are less than win specified connected features don't process it
        if len(allMatchedInfo[m]) <= win:
            continue
        for nm, v in enumerate(allMatchedInfo[m]):
            info = allMatchedInfo[m][v]
            # only for the first iteration append the reference position
            if nm == 0:
                df.loc[i] = [info.refP[0], info.refP[1], int(v), int(m)]
                i += 1
                
            df.loc[i] = [info.tarP[0], info.tarP[1], int(v + 1), int(m)]
            i += 1

    # plot the position of the features through the samples
    if plotting:
        plotFeatureProgress(df, imgs, dest + 'CombinedRough.png', r, s)

    return(df)

def nonRigidDeform(df, imgs, dest, cpuNo = False, r = 1, win = 9, sz = 100):

    # This function takes the continuous feature sets found in contFeatFinder
    # and uses them to non-rigidly warp the 
    # Inputs:   (df), panda dataframe of the feature positions and feature ID
    #           (imgs), list of the directories of the images
    #           (dest), destination path to save info
    #           (cpuNo), cores to use or False to serialise
    #           (r), resolution scale factor
    #           (win), window length for feature path filtering
    #           (s), section size used for feature finding
    # Outputs:  (), warp the images and save

    # get the number of features found
    featNo = np.max(df["ID"])

    # ensure the window length is odd
    if win%2 != 1:
        win-=1

    # create a new dataframe for the smoothed feature positions
    featsSm = pd.DataFrame(columns=["xPos", "yPos", "Sample", "ID"])

    # use only the five features with the most connections
    ID, IDCount = np.unique(np.array(df["ID"]), return_counts = True)
    targetIDs = ID[np.argsort(-IDCount)]

    p = 0
    for f in targetIDs:
        xp = df[df["ID"] == f].xPos
        yp = df[df["ID"] == f].yPos
        z = df[df["ID"] == f].Sample
        num_true_pts = len(z)

        # perform a savgol_filter over the data
        if len(xp) > win:
            '''
            '''
            # xSm = np.linspace(np.array(xp)[0], np.array(xp)[0], len(xp))
            # ySm = np.linspace(np.array(yp)[0], np.array(yp)[0], len(yp))

            tck, u = splprep([xp, yp, z], s = 10)

            # this could possibly be used to interpolate between slices
            # to find missing ones!
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
            infoStore = pool.starmap(ImageWarp, zip(np.arange(len(imgs)), imgs, repeat(df), repeat(featsSm), repeat(dest), repeat(r)))
        for iM in infoStore:
            infoStore.append(iM[0])
            # flowStore.append(iF) --> not using the flow ATM

    else:
        for s, imgpath in enumerate(imgs):
            img, flow = ImageWarp(s, imgpath, df, featsSm, dest, r)
            infoStore.append(img)

    plotFeatureProgress(featsSm, infoStore, dest + 'CombinedSmooth.png', r, sz)

def featMatching(m, tarImg, refImg = None, sz = 50):

    # Identifies the location of a feature in the next slice
    # Inputs:   (m), feature object of the previous point
    #           (tarImg), image of the next target sample
    #           (refImg), debugging stuff
    #           (sz), the equivalent number of windows to create
    # Outputs:  (featureInfo), feature object of feature (if identified
    #               in the target image)


    # get target position from the previous match, use this as the 
    # position of a possible reference feature 
    yp, xp = (m.refP).astype(int)
    x, y, c = tarImg.shape
    s = tile(sz, x, y)
    xs = np.clip(xp-s, 0, x); xe = np.clip(xp+s, 0, x)
    ys = np.clip(yp-s, 0, y); ye = np.clip(yp+s, 0, y)
    tarSect = tarImg[xs:xe, ys:ye]
    refSect = refImg[xs:xe, ys:ye]

    # if the point being searched for is black (ie background), 
    # don't return a value
    '''
    if (tarImg[xp, yp] == 0).all():
        plt.imshow(tarSect); plt.show()
        return
    '''

    # find all the features within this section of the new target image using cross-correlation
    refSectBW = np.mean(refSect, axis = 2).astype(np.uint8)
    tarSectBW = np.mean(tarSect, axis = 2).astype(np.uint8)

    # if a significant majority of the section is background, 
    # dont' return a value
    if (np.sum((tarSectBW == 0) * 1) / tarSectBW.size) > 0.2:
        # plt.imshow(tarSect); plt.show()
        return


    # cross-correlate to within 1/10th of a pixel
    shift, error, phasediff = pcc(refSectBW, tarSectBW, upsample_factor=10)
    '''
    if m.ID == 2 or m.ID == 9 or m.ID == 11:
        cv2.circle(tarSectBW, tuple((m.tarP - np.flip(shift) - [ys, xs]).astype(int)), 3, 255, 5)
        cv2.circle(tarSectBW, tuple((m.tarP - np.flip(shift) - [ys, xs]).astype(int)), 3, 0, 2)
        cv2.circle(refSectBW, tuple((m.tarP - [ys, xs]).astype(int)), 3, 255, 5)
        cv2.circle(refSectBW, tuple((m.tarP - [ys, xs]).astype(int)), 3, 0, 2)
        plt.imshow(np.hstack([refSectBW, tarSectBW]), cmap = 'gray'); plt.show()
    '''

    # create the new feature object
    featureInfo = feature()
    featureInfo.refP = m.tarP 
    featureInfo.tarP = m.tarP + np.flip(shift)
    featureInfo.dist = error
    featureInfo.ID = m.ID
    allfeatureInfo = [featureInfo]

    return(allfeatureInfo)

def ImageWarp(s, imgpath, dfRaw, dfNew, dest, r = 1):

    # perform the non-rigid warp
    # Inputs:   (s), number of the sample
    #           (imgpath), directory of the image
    #           (dfRaw), raw data frame of the feature info
    #           (dfNew), data frame of the smoothed feature info
    #           (dest), directory to save modified image
    #           (r), rescaling value for the image to match the feature postiions
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
    # x, y, c = (np.array(img.shape) * r).astype(int)
    # img = cv2.resize(img, tuple([y, x]))

    # get the sample specific feature info
    refInfo = dfRaw[dfRaw["Sample"] == s]
    tarInfo = dfNew[dfNew["Sample"] == s]

    # merge the info of the features which have the same ID
    allInfo = pd.merge(refInfo, tarInfo, left_on = 'ID', right_on = 'ID')

    # get the common feature positions
    refFeats = np.c_[allInfo.xPos_x, allInfo.yPos_x]/r
    tarFeats = np.c_[allInfo.xPos_y, allInfo.yPos_y]/r

    # create the bounds on the image edges
    bound = []
    bound.append(np.array([0, 0]))
    bound.append(np.array([0, y]))
    bound.append(np.array([x, 0]))
    bound.append(np.array([x, y]))
    
    for b in bound:
        refFeats = np.insert(refFeats, 0, b, axis = 0)
        tarFeats = np.insert(tarFeats, 0, b, axis = 0)
    

    # ensure the inputs are 4D tensors
    tfrefPoints = np.expand_dims(refFeats.astype(float), 0)
    tftarPoints = np.expand_dims(tarFeats.astype(float), 0)
    tftarImg = np.expand_dims(img, 0).astype(float)

    # perform non-rigid deformation on the original sized image
    imgMod, imgFlow = sparse_image_warp(tftarImg, tfrefPoints, tftarPoints)

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

    # save the images
    cv2.imwrite(dest + name + ".png", imgMod)

    # return a numpy array of the image and the flow field
    return(imgMod, Flow)

def plotFeatureProgress(df, imgAll, dirdest, r = 1, sz = 0):

    # takes a data frame of the feature info and a list containing the images and 
    # returns two graphics. One is a picture of all the images and their features and 
    # the other is a 3D plot of the features only
    # Inputs:   (df), pandas data frame of the info
    #           (imgAll), list of the images either as numpy values or file paths
    #           (dirdst), path of the image to be saved
    #           (r), resolution scale
    #           (sz), size of the grid to use
    # Outputs:  (), image of all the samples with their features and lines connecting them 
    #               and a 3d line plot (plotly)


    # if the input is a list of paths of the images, load the images into a list
    if type(imgAll[0]) is str:
        imgPaths = imgAll.copy()
        imgAll = []
        for i in imgPaths:
            imgAll.append(cv2.imread(i))
    
    sampleNo = len(imgAll)
    x, y, c = imgAll[0].shape
    imgAll = np.hstack(imgAll)

    # get the dims of the area search for the features
    l = tile(sz, x, y)

    # annotate an image of the features and their matches
    for i in np.unique(df["ID"]).astype(int):
        featdf = df[df["ID"] == i]
        tar = None
        for n, (fx, fy, s) in enumerate(zip(featdf.xPos, featdf.yPos, featdf.Sample)):
            tar = (np.array([np.round(fx/r), np.round(fy/r)]) + [y * s, 0]).astype(int)
            if n == 0:
                # draw the first point on the first sample it appears as green
                cv2.circle(imgAll, tuple(tar), 10, [0, 255, 0], 6)
            else:
                # draw the next points as red and draw lines to the previous points
                imgAll = drawLine(imgAll, ref, tar, colour = [255, 0, 0])
                cv2.circle(imgAll, tuple(tar), 10, [0, 0, 255], 6)

            imgAll = drawLine(imgAll, tar - l, tar + [l, -l], blur = 2, colour=[0, 255, 0])
            imgAll = drawLine(imgAll, tar - l, tar + [-l, l], blur = 2, colour=[0, 255, 0])
            imgAll = drawLine(imgAll, tar + l, tar - [l, -l], blur = 2, colour=[0, 255, 0])
            imgAll = drawLine(imgAll, tar + l, tar - [-l, l], blur = 2, colour=[0, 255, 0])

            cv2.putText(imgAll, str(i), tuple(tar-[10, 5]), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 4)
            cv2.putText(imgAll, str(i), tuple(tar-[10, 5]), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)
            ref = tar
        # make the final point pink
        if tar is not None:
            cv2.circle(imgAll, tuple(tar), 10, [255, 0, 255], 6)

    # save the linked feature images
    cv2.imwrite(dirdest, imgAll)
    
    # create a 3D plot of the feature progression
    px.line_3d(df, x="xPos", y="yPos", z="Sample", color="ID").show()

def tile(sz, x, y):

    # gets the size of the area to search for in an image
    # Inputs:   (sz), tile proporption of the image
    #           (x, y), img dims
    # Outputs:  (s), the square length needed

    # if the size is 0 then don't create a tile
    if sz == 0:
        return(0)

    s = int(np.round(np.sqrt(x*y/sz)))   # border lenght of a tile to use

    return(s)


if __name__ == "__main__":

    # with 2 2D images, interpolate between them for given points
    dirHome = '/Volumes/USB/H671B_18.5/'
    dirHome = '/Volumes/USB/Test/'
    dirHome = '/Volumes/Storage/H710C_6.1/'


    size = 3
    cpuNo = False

    nonRigidAlign(dirHome, size, cpuNo)
