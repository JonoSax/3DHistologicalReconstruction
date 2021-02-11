'''
This script contains all the plotting funcitons for the diagrams used in 
my thesis
'''

import cv2
import numpy as np
if __name__ != "HelperFunctions.plottingFunctions":
    from Utilities import *
else:
    from HelperFunctions.Utilities import *

def plottingFeaturesPerRes(IMGREF, name, matchedInfo, scales, circlesz = 1):

    '''
    this plotting funciton gets the features that have been produced per resolution 
    and combines them into a single diagram
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