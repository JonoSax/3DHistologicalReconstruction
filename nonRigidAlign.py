from scipy.interpolate import BSpline as bs
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from Utilities import nameFromPath, matchMaker, nameFeatures, drawLine
from tensorflow_addons.image import sparse_image_warp
from copy import deepcopy
import multiprocessing
from itertools import repeat

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

featureInfo = feature()

def nonRigidAlign(dirHome, cpuNo):
    imgs = sorted(glob(dirHome + "*.png"))[:5]

    sift =  cv2.xfeatures2d.SIFT_create() 
    bf = cv2.BFMatcher()
    r = 0.5
    des_refConfirm = []
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


    # the window around the section
    s = 50

    # compute all the features 
    kp_ref, des_ref = sift.detectAndCompute(cv2.resize(refImg, (y, x)), None)

    for sampleNo, tarPath in enumerate(imgs[1:]):

        # get all the features from the 
        tarName = nameFromPath(tarPath)
        tarImg = cv2.imread(tarPath)
        tarImg = cv2.resize(tarImg, (int(y), int(x)))

        print("Matching " + tarName + " to " + refName)

        # ---- identify the continuity of features ----

        if cpuNo is False:
            for m in matchedInfo:
                confirmInfo.append(findConFeats(m, tarImg, sift, bf, s))

        else:
            with multiprocessing.Pool(processes=cpuNo) as pool:
                confirmInfo = pool.starmap(findConFeats, zip(matchedInfo, repeat(tarImg), repeat(s)))

        # unravel the list of features produced by the continuous features
        confirmInfos = []
        for info in confirmInfo:
            if info is None:
                continue
            for i in info:
                confirmInfos.append(i)

        continuedFeatures = matchMaker(confirmInfos, dist=20, tol = 1, cpuNo = cpuNo)
        # featMatchimg = nameFeatures(refImg, tarImg, continuedFeatures, combine = True)
        # plt.imshow(featMatchimg); plt.show()

        # ---- find new features in the sample ----

        # find the all matching features in each slice
        kp_tar, des_tar = sift.detectAndCompute(tarImg, None)
        matches = bf.match(des_ref, des_tar)

        # convert the points into the feature object
        resInfo = []
        def_ref = []
        for m in matches:
            featureInfo = feature()
            # store the feature information as it appears on the original sized image
            featureInfo.refP = np.array(kp_ref[m.queryIdx].pt) 
            featureInfo.tarP = np.array(kp_tar[m.trainIdx].pt) 
            featureInfo.dist = np.array(m.distance)
            featureInfo.trainIdx = m.trainIdx
            featureInfo.descr = np.array(des_tar[m.trainIdx])
            resInfo.append(deepcopy(featureInfo))

        # find related features between both images
        # use a larger distance and tolerance between the features
        # in order to find as many features as possibl ethat are distributed 
        # across the entire sample 
        # use the confirmed features as a starging point
        matchedInfo = matchMaker(resInfo, continuedFeatures, dist=20, tol = 0.2, cpuNo = cpuNo)

        for m in matchedInfo:
            if m.ID is None:
                # keep track of the feature ID
                m.ID = featNo
                allMatchedInfo[m.ID] = {}
                featNo += 1

            allMatchedInfo[m.ID][sampleNo] = m

        # featMatchimg = nameFeatures(refImg, tarImg, matchedInfo, combine = True)
        # plt.imshow(featMatchimg); plt.show()
        # get the descriptors from the target sample which were matched with 
        # the reference sample

        # reasign the target features as the reference features
        refName = tarName
        kp_ref, des_ref = kp_tar, des_tar 
        refImg = tarImg


    # arrange the data in a way so that it can be plotted in 3D
    maxNo = np.max([len(allMatchedInfo[mi]) for mi in allMatchedInfo])
    
    for n in range(maxNo):
        print("Number of " + str(1+n) + " linked features = " + str(len(np.where(np.array([len(allMatchedInfo[mi]) for mi in allMatchedInfo]) == n+1)[0])))
    
    imgStack = []
    for i in imgs:
        imgStack.append(cv2.resize(cv2.imread(i), (y, x)))

    imgAll = np.hstack(imgStack)

    for i in allMatchedInfo:
        if len(allMatchedInfo[i]) > 3:
            start = list(allMatchedInfo[i].keys())[0]
            ref = allMatchedInfo[i][start].refP + [y * start, 0]
            cv2.circle(imgAll, tuple(ref.astype(int)), 10, [0, 255, 0], 6)
            for n, m in enumerate(allMatchedInfo[i]):
                tar = allMatchedInfo[i][m].tarP + [y * (start+n+1), 0]
                imgAll = drawLine(imgAll, ref, tar, colour = [255, 0, 0])
                cv2.circle(imgAll, tuple(tar.astype(int)), 10, [0, 0, 255], 6)
                ref = tar

            cv2.circle(imgAll, tuple(tar.astype(int)), 10, [255, 0, 255], 6)

    plt.imshow(imgAll); plt.show()

    '''
    for m in allMatchedInfo:
        if len(allMatchedInfo[m]) > 3:
            start = list(allMatchedInfo[m].keys())[0]
            ref = allMatchedInfo[m][start].refP 
            cv2.circle(imgAll, tuple(ref.astype(int)), 20, [255, 0, 0], 10)
            for n, i in enumerate(allMatchedInfo[m]):
                tar = allMatchedInfo[m][i].tarP + [y * n, 0]
                cv2.circle(imgAll, tuple(tar.astype(int)), 20, [255, 0, 0], 10)
                imgAll = drawLine(imgAll, ref, tar)
                ref = tar
    '''
    # plt.imshow(imgAll);plt.show()

    cv2.imwrite('/Users/jonathanreshef/Downloads/imgN.png', imgAll)
    
    ax = plt.axes(projection='3d')
    xdata = []
    ydata = []
    zdata = []
    for n, matchedData in enumerate(allMatchedInfo):
        for v in matchedData:
            xdata.append(v.tarP[0])
            ydata.append(v.tarP[1])
            zdata.append(n)
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    plt.show()
    


    # taking the features found and performing non-rigid alignment
    '''

    # add the features into a list 
    # adjust the position of the features for the change in resolution 
    # of image used in the feature finding and create 
    refPoints = []
    tarPoints = []
    for m in matchedInfo:
        refPoints.append(m.refP / r)
        tarPoints.append(m.tarP / r)

    # create tensors of the featues and images
    tfrefPoints = np.expand_dims(np.array(refPoints), 0)
    tftarPoints = np.expand_dims(np.array(tarPoints), 0)
    tftarImg = np.expand_dims(tarImg, 0).astype(float)

    # perform non-rigid deformation on the original sized image
    imgMod = sparse_image_warp(tftarImg, tfrefPoints, tftarPoints)

    # adjust the original tif image with the flow shape rescaled 
    # to the tif image
    # imgModFull = dense_image_warp(tftarImgFull, FLOW)

    # reasign the new reference image as the image which has just been 
    # adjusted 
    refImg = np.array(imgMod[0])[0]
    refName = tarName
    '''

def findConFeats(m, tarImg, s = 50):

    # Identifies the location of a feature in the next slice
    # Inputs:   (m), feature object of the previous point
    #           (tarImg), image of the next target sample
    #           (s), the window around the section
    #           (sift), SIFT method
    #           (bf), bf method
    # Outputs:  (featureInfo), feature object of feature (if identified
    #               in the target image)


    # get target position from the previous match, use this as the 
    # position of a possible reference feature 
    yp, xp = (m.tarP).astype(int)
    x, y, c = tarImg.shape
    xs = np.clip(xp-s, 0, x); xe = np.clip(xp+s, 0, x)
    ys = np.clip(yp-s, 0, y); ye = np.clip(yp+s, 0, y)
    tarSect = tarImg[xs:xe, ys:ye]

    sift =  cv2.xfeatures2d.SIFT_create() 
    bf = cv2.BFMatcher()

    # plt.imshow(np.hstack([refSect, tarSect])); plt.show()

    # find all the features within this section of the new target image
    kp_feat, des_feat = sift.detectAndCompute(tarSect, None)

    if des_feat is None:
        return

    # of all the features found in this section, find the top 10 best
    # fit features
    fm = bf.match(des_feat, np.expand_dims(m.descr, 0))
    featMatch = sorted(fm, key=lambda fm: fm.distance)[:20]

    # store the feature information (NOTE swapping the ref and tar positions
    # because the bf match was reversed to get more features to search from)
    allfeatureInfo = []
    for f in featMatch:
        featureInfo = feature()
        featureInfo.refP = m.tarP 
        featureInfo.tarP = np.array(kp_feat[f.queryIdx].pt) + [ys, xs]
        featureInfo.dist = np.array(f.distance)
        featureInfo.trainIdx = f.queryIdx
        featureInfo.descr = np.array(des_feat[f.queryIdx])
        featureInfo.ID = m.ID
        allfeatureInfo.append(featureInfo)

    return(allfeatureInfo)

if __name__ == "__main__":

    # with 2 2D images, interpolate between them for given points
    dirHome = '/Volumes/USB/Test/3/alignedSamples/'
    cpuNo = 6
    nonRigidAlign(dirHome, cpuNo)
