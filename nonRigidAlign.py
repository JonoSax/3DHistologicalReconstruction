import numpy as np
import matplotlib.pyplot as plt
import cv2
from glob import glob
from HelperFunctions.Utilities import *
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
    def __init__(self, refP = None, tarP = None, dist = None, ID = None):

        # the feature number 
        self.ID = ID

        # the position of the match on the reference image
        self.refP = refP

        # the position of the match on the target image
        self.tarP = tarP

        # eucledian error of the difference in gradient fields
        self.dist = dist


    def __repr__(self):
            return repr((self.ID, self.refP, self.tarP, self.dist))

def nonRigidAlign(dirHome, size, cpuNo):

    home = dirHome + str(size)

    imgsrc = home + "/pngImages/"
    destRigidAlign = home + "/RealignedSamples/"
    dirfeats = home + "/infoNL/"
    destNLALign = home + "/NLAlignedSamples/"

    dirMaker(destRigidAlign)
    dirMaker(dirfeats)
    dirMaker(destNLALign)

    imgs = sorted(glob(imgsrc + "*.png"))[:100]
    scale = 0.2
    win = 1
    sect = 500
    dist = 20

    # NOTE save feats as txt some how...
    contFeatFinder(imgs, dirfeats, destRigidAlign, cpuNo = False, scl = scale, plotting = True, sz = sect, dist = dist)

    aligner(imgs, dirfeats, imgsrc, destRigidAlign, cpuNo=False)
    
    nonRigidDeform(destRigidAlign, dirfeats, destNLALign, cpuNo, scale, win, sect)

def contFeatFinder(imgs, destFeat, destImg = None, cpuNo = False, scl = 1, plotting = False, sz = 100, dist = 20):

    # This function takes images and finds features that are continuous
    # between the samples
    # Inputs:   (imgs), list of directories of the images to be processed
    #               in sequential order for feature identifying
    #           (destImg), where to save the image from the plotting
    #           (cpuNo), number of cores to parallelise the task. 
    #           (scl), factor to resize images
    #           (plotting), boolean whether to plot the feature info through
    #               the samples
    #           (sz), the equivalent area which a continuous feature will be searched for 
    #               if this many tiles were created on the section
    #           (dist), minimum distance between detected features
    #           NOTE it appears ATM that it is faster to serialise...
    # Outputs:  (), saves the feature objects for each path of the info

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
    x, y, c = (np.array(refImg.shape) * scl).astype(int)

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
                confirmInfo.append(featMatching(m, tarImg, refImg, sz))

        else:
            with multiprocessing.Pool(processes=cpuNo) as pool:
                confirmInfo = pool.starmap(featMatching, zip(matchedInfo, repeat(tarImg), repeat(refImg), repeat(sz)))

        # unravel the list of features produced by the continuous features
        confirmInfos = []
        for info in confirmInfo:
            if info is None:
                continue
            for i in info:
                confirmInfos.append(i)

        continuedFeatures = matchMaker(confirmInfos, dist = dist, tol = 1, cpuNo = False)
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
            resInfo.append(featureInfo)

        # find related features between both images
        # use a larger distance and tolerance between the features
        # in order to find as many features as possibl ethat are distributed 
        # across the entire sample 
        # use the confirmed features as a starging point
        matchedInfo = matchMaker(resInfo, continuedFeatures, dist = dist, tol = 0.2, cpuNo = False, anchorPoints = 10)

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
    maxNo = len(imgs)
    
    for n in range(1,maxNo):
        print("Number of " + str(1+n) + "/" + str(maxNo) + " linked features = " + str(len(np.where(np.array([len(allMatchedInfo[mi]) for mi in allMatchedInfo]) == n)[0])))
    
    imgStack = []
    for i in imgs:
        imgStack.append(cv2.resize(cv2.imread(i), (y, x)))

    # create a panda data frame of all the features found for plotting
    df = dictToDF(allMatchedInfo, ["xPos", "yPos", "Sample", "ID"], min=3, scl = scl)

    # create feature dictionaries per sample
    for s in range(maxNo-1):

        refN = nameFromPath(imgs[s], 3)
        tarN = nameFromPath(imgs[s+1], 3)

        # get the features of the reference image
        ref = df[df["Sample"] == s]

        # get the features of the target image
        tar = df[df["Sample"] == s+1]

        refD = {}
        for x, y, s, ID in np.array(ref):
            refD[str(int(ID))] = np.array([x, y])
        dictToTxt(refD, destFeat + refN + ".reffeat", fit = False)
        
        tarD = {}
        for x, y, s, ID in np.array(tar):
            tarD[str(int(ID))] = np.array([x, y])
        dictToTxt(tarD, destFeat + tarN + ".tarfeat", fit = False)

    # plot the position of the features through the samples
    if plotting:
        plotFeatureProgress(df, imgs, destImg + 'CombinedRough.jpg', scl, sz)

def nonRigidDeform(dirimgs, dirfeats, dirdest, cpuNo = False, scl = 1, win = 9, sz = 100):

    # This function takes the continuous feature sets found in contFeatFinder
    # and uses them to non-rigidly warp the 
    # Inputs:   (dirimgs), directory of all the images
    #           (dirfeats), directory of the features 
    #           (dirdest), path to save the NL deformed info
    #           (cpuNo), cores to use or False to serialise
    #           (scl), resolution scale factor
    #           (win), window length for feature path filtering
    #           (s), section size used for feature finding
    # Outputs:  (), warp the images and save

    # get the image paths
    imgs = glob(dirimgs + "*.png")

    # get the new dictionaries, load them into a pandas dataframe
    refFeats = glob(dirfeats + "*.reffeat")
    tarFeats = glob(dirfeats + "*.reffeat")
    infopds = []

    # NOTE beacuse the ref and tarfeatures are the same for all the samples
    # it just has to iteratue through the samples, not the ref/tar feat files
    for n, r in enumerate(refFeats + [tarFeats[-1]]):
        info = txtToDict(r, float)[0]
        infopd = pd.DataFrame.from_dict(info, orient = 'index', columns = ['xPos', 'yPos'])
        infopd['Sample'] = int(n)        # add the sample number
        infopd['ID'] = info.keys()
        infopds.append(infopd)
        
    # combine into a single df
    df = pd.concat(infopds)
    

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


            # this could possibly be used to interpolate between slices
            # to find missing ones!
            tck, u = splprep([xp, yp, z], s = 100)
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
            infoStore = pool.starmap(ImageWarp, zip(np.arange(len(imgs)), imgs, repeat(df), repeat(featsSm), repeat(dest), repeat(scl)))
        for iM in infoStore:
            infoStore.append(iM[0])
            # flowStore.append(iF) --> not using the flow ATM

    else:
        for s, imgpath in enumerate(imgs):
            img, flow = ImageWarp(s, imgpath, df, featsSm, dirdest, scl)
            infoStore.append(img)

    plotFeatureProgress(featsSm, infoStore, dirdest + 'CombinedSmooth.jpg', scl, sz)

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
    s = int(tile(sz, x, y)/2)
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
    if (np.sum((tarSectBW == 0) * 1) / tarSectBW.size) > 0.5:
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
    featureInfo.tarP = m.tarP - np.flip(shift)
    featureInfo.dist = error
    featureInfo.ID = m.ID
    allfeatureInfo = [featureInfo]

    return(allfeatureInfo)

def ImageWarp(s, imgpath, dfRaw, dfNew, dest, scl = 1):

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

def plotFeatureProgress(df, imgAll, dirdest, scl = 1, sz = 0):

    # takes a data frame of the feature info and a list containing the images and 
    # returns two graphics. One is a picture of all the images and their features and 
    # the other is a 3D plot of the features only
    # Inputs:   (df), pandas data frame of the info
    #           (imgAll), list of the images either as numpy values or file paths
    #           (dirdst), path of the image to be saved
    #           (scl), resolution scale
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
    l = int(tile(sz, x, y)/2)

    # annotate an image of the features and their matches
    for samp, i in enumerate(np.unique(df["ID"])):
        featdf = df[df["ID"] == i]
        tar = None
        for n, (fx, fy, s) in enumerate(zip(featdf.xPos, featdf.yPos, featdf.Sample)):

            tar = ((np.array([np.round(fx), np.round(fy)])) + np.array([y * s, 0])).astype(int)
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
        
        cv2.putText(imgAll, "Samp_" + str(samp) + " no_" + str(len(df[df["Sample"] == samp])), tuple(np.array([y * samp, 0]).astype(int) + [50, 100]), cv2.FONT_HERSHEY_SIMPLEX, 3, [255, 255, 255], 6)
        
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
