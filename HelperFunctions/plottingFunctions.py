'''
This script contains all the plotting funcitons for the diagrams used in 
my thesis
'''

import cv2
import numpy as np
from glob import glob
from random import uniform
from time import time
import tifffile as tifi

if __name__ != "HelperFunctions.plottingFunctions":
    from Utilities import *
    from SP_AlignSamples import align
    from SP_FeatureFinder import featFind
else:
    from HelperFunctions.Utilities import *
    from HelperFunctions.SP_AlignSamples import align
    from HelperFunctions.SP_FeatureFinder import featFind

def plottingFeaturesPerRes(IMGREF, name, matchedInfo, scales, circlesz = 1):

    '''
    this plotting funciton gets the features that have been produced per resolution 
    and combines them into a single diagram

        Inputs:\n
    IMGREF, image to plot the features on\n
    name, name of the sample\n
    matchedInfo, list of all the feature objects\n
    sclaes, the image scales used in the search\n
    circlesz, circle sz (put to 0 to remove circles)\n

    '''

    imgRefS = []
    sclInfoAll = []
    for n, scl in enumerate(scales):

        # get the position 
        sclInfo = [matchedInfo[i] for i in list(np.where([m.res == n for m in matchedInfo])[0])]

        if len(sclInfo) == 0: 
            continue
        else:
            sclInfoAll += sclInfo

        # for each resolution plot the points found
        imgRefM = cv2.resize(IMGREF.copy(), (int(IMGREF.shape[1] * scl), int(IMGREF.shape[0] * scl)))
        # downscale then upscale just so that the image looks like the downsample version but can 
        # be stacked
        imgRefM = cv2.resize(imgRefM, (int(IMGREF.shape[1]), int(IMGREF.shape[0])))
        

        imgRefM, _ = nameFeatures(imgRefM.copy(), imgRefM.copy(), sclInfoAll, circlesz=circlesz, combine = False, txtsz=0)
        
        imgRefM = cv2.putText(imgRefM, "Scale = " + str(scl), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, [255, 255, 255], thickness = 14)
        imgRefM = cv2.putText(imgRefM, "Scale = " + str(scl), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 0],  thickness = 6)
        
        cv2.imwrite("/Users/jonathanreshef/Downloads/" + name + "_" + str(scl) + ".png", imgRefM)
        
        imgRefS.append(imgRefM)

    imgRefS = np.hstack(imgRefS)
    cv2.imwrite("/Users/jonathanreshef/Downloads/" + name + ".png", imgRefS)
    # plt.imshow(imgRefS); plt.show()

def colourDistributionHistos():

    '''
    Compares the distributions of the colour channels before and after
    colour normalisations compared to the reference image
    '''


    imgtarSmallOrig = cv2.imread(imgtarSmallpath)

    colours = ['b', 'g', 'r']
    z = [np.arange(10), np.zeros(10)]
    ylim = 0.2

    # ----- Plot the origianl vs reference histogram plots -----

    blkDsh, = plt.plot(z[0], z[1], "k:")    # targetO
    blkDot, = plt.plot(z[0], z[1], "k--")    # targetMod
    blkLn, = plt.plot(z[0], z[1], "k")    # reference
    
    for c, co in enumerate(colours):
        o = np.histogram(imgtarSmallOrig[:, :, c], 32, (0, 256))   # original
        r = np.histogram(imgref[:, :, c], 32, (0, 256)) # reference
        v = np.histogram(imgtarSmall[:, :, c], 32, (0, 256))   # modified
        v = np.ma.masked_where(v == 0, v)

        maxV = np.sum(r[0][1:])        # get the sum of all the points
        plt.plot(o[1][2:], o[0][1:]/maxV, co + ":", linewidth = 2)
        plt.plot(v[1][2:], v[0][1:]/maxV, co + "--", linewidth = 2)
        plt.plot(r[1][2:], r[0][1:]/maxV, co, linewidth = 1)

    plt.legend([blkDsh, blkDot, blkLn], ["TargetOrig", "TargetMod", "Reference"])
    plt.ylim([0, ylim])
    plt.xlabel("Pixel value", fontsize = 14)
    plt.ylabel("Pixel distribution",  fontsize = 14)
    plt.title("Histogram of colour profiles",  fontsize = 18)
    plt.show()

    # ----- Plot the modified vs reference histogram plots -----
    
    for c, co in enumerate(colours):
        v = np.histogram(imgr[:, :, c], 32, (0, 256))[0][1:]   # modified
        v = np.ma.masked_where(v == 0, v)
        plt.plot(v/maxV, co + "--", linewidth = 2)
        r = np.histogram(imgref[:, :, c], 32, (0, 256))[0][1:] # reference
        plt.plot(r/maxV, co, linewidth = 1)

    z = np.arange(10)

    plt.legend([blkDsh, blkLn], ["TargetMod", "Reference"])
    plt.ylim([0, ylim])
    plt.xlabel("Pixel value", fontsize = 14)
    plt.ylabel("Pixel distribution",  fontsize = 14)
    plt.title("Histogram of colour profiles\n modified vs reference",  fontsize = 18)
    plt.show()

def imageModGenerator():

    '''
    This function generates images with a random translation and rotation
    from an original image. Used to feed into the featurefind and align process
    to validate they work.
    '''

    def imgTransform(img, maxSize, maxRot = 90):
    
        '''
        From an original image make a bunch of new images with random 
        rotation and translation within a maximum sized area
        '''
        
        ip = img.shape[:2]
        angle = uniform(-maxRot, maxRot)/2

        xM, yM = (maxSize - img.shape[:2])/2
        translate = np.array([uniform(-xM, xM), uniform(-yM, yM)])
        
        rot = cv2.getRotationMatrix2D(tuple(np.array(ip)/2), float(angle), 1)
        warpedImg = cv2.warpAffine(img, rot, (ip[1], ip[0]))

        # get the position of the image at the middle of the image
        p = np.array((maxSize - ip)/2 + translate).astype(int)

        plate = np.zeros(np.insert(maxSize, 2, 3)).astype(np.uint8)
        plate[p[0]:p[0]+ ip[0], p[1]:p[1]+ ip[1]] = warpedImg

        return(plate) 

    dataHome = "/Users/jonathanreshef/Documents/2020/Masters/Thesis/Diagrams/alignerDemonstration/"

    imgSrc = dataHome + "3/BirdFace.png"
    img = cv2.imread(imgSrc)

    dirMaker(dataHome + "3/masked/")

    # get the reference image in the centre of the maxSize
    x, y = img.shape[:2]
    xM, yM = (np.array([x, y]) * 1.2).astype(int)
    xMi = int((xM-x)/2); yMi = int((yM-y)/2)
    imgN = np.zeros([xM, yM, 3]).astype(np.uint8)
    imgN[xMi:xMi+x, yMi:yMi+y, :] = img

    # create the centred image to align all samples to
    # cv2.imwrite(dataHome + "3/masked/modImg000.png", imgN)

    for i in range(20):
        imgN = imgTransform(img, np.array([xM, yM]), maxRot=360)
        name = str(i + 1)
        while len(name) < 3:
            name = "0" + name
        # cv2.imwrite(dataHome + "3/masked/modImg" + name + ".png", imgN)

    featFind(dataHome, "3", 1, 10, 1, 1)
    align(dataHome, "3", 1)

def triangulatorPlot(img, matchedInfo):

    '''
    Plots the features of samples and visualises the triangulation 
    calculation
    '''
                
    def triangulator(img, featurePoints, anchorPoints, feats = 5, crcsz = 5):

        '''
        Plot the triangulation of the features in the sample 
        from the anchor feautres

            Inputs:\n
        (img), image\n
        (featurePoints), list of ALL feature points\n
        (anchorPoints), number of points in featurePoints which are used for 
        the anchor\n
        (feats), maximum number of non-anchor points to plot

            Output:\n
        (imgAngles), image with the feature and its points annotated
        '''

        # get the anchor points and the points found from these
        anchors = featurePoints[:int(anchorPoints)]
        points = featurePoints[int(anchorPoints):]

        imgAngles = img.copy()
        col = [[255, 0, 0], [0, 255, 255], [255, 0, 255], [0, 255, 0], [255, 255, 0]]

        for n, p in enumerate(points[:feats]):
            # draw the point of interest
            cv2.circle(imgAngles, tuple(p.astype(int)), crcsz, [0, 0, 255], crcsz*2)
            for n1, a1 in enumerate(anchors):

                # draw the anchor points
                cv2.circle(imgAngles, tuple(a1.astype(int)), crcsz, [255, 0, 0], crcsz*2)
                for n2, a2 in enumerate(anchors):
                    if n1 == n2: 
                        continue
                    # draw the lines between the features
                    imgAngles = drawLine(imgAngles, p, a1, colour=col[n])
                    imgAngles = drawLine(imgAngles, p, a2, colour=col[n])
        
        return(imgAngles)
                    
    # get the reference and target points 
    refPts = [m.refP for m in matchedInfo]

    # annotate the image with the feature info
    for i in range(5):
        refAngles = triangulator(img, refPts, 5, i, crcsz = 12)

        cv2.imshow('angledImages', refAngles); cv2.waitKey(0)
    cv2.destroyWindow('angledImages')

def siftTimer():

    '''
    Timing the SIFT operator for various octave numbers and images
    resolutions 
    '''

    imgPath = '/Volumes/USB/H653A_11.3/3/masked/H653A_002_0.png'
    imgOrig = cv2.imread(imgPath)

    # perform sift search on multiple different resolutions of sift
    for n in range(1, 10):

        img = cv2.resize(imgOrig.copy(), (int(imgOrig.shape[1]/n), int(imgOrig.shape[0]/n)))

        for i in range(10, 11):
            sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = i) 

            a = time()
            for _ in range(10):
                s = sift.detectAndCompute(img, None)[0]
            fin = time()-a

            print("ImgSz = " + str(np.round(1/n, 2)) + " Octaves = " + str(i) + " Time: " + str(fin) + " feats = " + str(len(s)))

def vesselPositionsOnMaskedImgs():

    '''
    This is getting the masked positions as manually annotated from 
    the samples after they have been seperated by specimen ID. 
    Saved as pixel positions on the full resolution image
    '''

    src = '/Volumes/USB/H653A_11.3/2.5/'
    masks = src + 'NLAlignedSamples/'

    downsampleImgs = glob(masks + "*.png")
    fullImgs = sorted(glob(masks + "*.tif"))

    for d in downsampleImgs:

        name = nameFromPath(d, 3)

        print("Processing " + name)

        # penalise non-green colours
        img = np.mean(cv2.imread(d) * np.array([-1, 1, -1]), axis = 2)

        maskPos = np.where(img > 50); maskPos = np.c_[maskPos[1], maskPos[0]]

        # ensure the dense reconstructions are the original size
        maskPos = np.insert(maskPos, 0, np.array([0, 0]), axis = 0)
        maskPos = np.insert(maskPos, 0, np.flip(img.shape[:2]), axis = 0)

        vessels = denseMatrixViewer([maskPos], plot = False, point = True)[0]
        cv2.imwrite(masks + name + "_vessels.png", vessels)
        
        listToTxt([maskPos], masks + name + "vessels.txt")

if __name__ == "__main__":

    # imageModGenerator()

    # siftTimer()

    # vesselPositionsOnMaskedImgs()