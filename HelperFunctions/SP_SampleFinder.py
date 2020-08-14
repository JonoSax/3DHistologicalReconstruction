'''

This funciton allows a user to select a ROI on a single/multiple samples
which is then used to identify all other matching featues on all other samples

This uses low resolution jpeg images which have been created by the tif2dfp function
in order to reduce the computational load

'''

import cv2
import numpy as np
# import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
import tifffile as tifi
if __name__ == "__main__":
    from Utilities import nameFromPath, dirMaker
else:
    from HelperFunctions.Utilities import nameFromPath, dirMaker

def featSelectArea(datahome, sample, feats = 5):

    # this function brings up a gui which allows user to manually selection a 
    # roi on the image. This extracts samples from the aligned tissues and saves them


    segSections = datahome + "segSections/"

    for f in range(feats):
        dirMaker(segSections + "seg" + str(f) + "/")

    alignedSamples = datahome + "alignedSamples/"

    img = glob(alignedSamples + sample + "*.tif")[0]

    samples = glob(alignedSamples + "*.tif")

    if type(img) == str:
        img = tifi.imread(img)

    # extract n feats from the target samples
    x = {}
    y = {}
    for f in range(feats):
        x[f], y[f] = roiselector(img)
        cv2.rectangle(img, (x[f][0], y[f][0]), (x[f][1], y[f][1]), (255, 255, 255), 40)
        cv2.rectangle(img, (x[f][0], y[f][0]), (x[f][1], y[f][1]), (0, 0, 0), 20)

    # extract from all the samples the features
    for s in samples:
        name = nameFromPath(s)
        img = tifi.imread(s)
        for f in range(feats):
            segdir = segSections + "seg" + str(f) + "/"
            section = img[y[f][0]:y[f][1], x[f][0]:x[f][1] :]
            tifi.imwrite(segdir + name + ".tif", section)


def featSelectPoint(imgref, imgtar, matchRef, matchTar, feats = 5):

    # this fuction brings up a gui which allows for a user to manually select
    # features on the images. This contributes to making a .feat file
    # Inputs:   (nameref), either the path or the numpy array of the reference image
    #           (nametar), either the path or the numpy array of the target image
    #           (matchRef), any already identified features on the reference image
    #           (matchTar), any already identified features on the target image
    #           (feats), defaults to finding 5 features
    # Outputs:  (matchRef, matchTar), updated ref and target features with new points added

    # get the images with insufficient features
    if type(imgref) == str or type(imgtar) == str:
        imgref = cv2.imread(imgref)
        imgtar = cv2.imread(imgtar)

    # combine the images
    # get the image dimensions
    xr, yr, cr = imgref.shape
    xt, yt, ct = imgtar.shape
    xm, ym, cm = np.max(np.array([(xr, yr, cr), (xt, yt, ct)]), axis = 0)
        
    # create a max size field of both images
    field = np.zeros((xm, ym, cm)).astype(np.uint8)

    # re-assign the images to the left of this standard field
    img_refC = field.copy(); img_refC[:xr, :yr, :] = imgref
    img_tarC = field.copy(); img_tarC[:xt, :yt, :] = imgtar

    # combine images
    imgCombine = np.hstack([img_refC, img_tarC])
    xc, yc, c = imgCombine.shape

    n = 0           # iterator counter

    # draw on the already found points
    for i, (r, t) in enumerate(zip(matchRef, matchTar)):

        # enusre that the co-ordinate is in the right format and position 
        if type(r) is np.ndarray: r = tuple(r.astype(int))
        if type(t) is np.ndarray: t = tuple(t.astype(int) + np.array([ym, 0]))
        else: t = tuple(np.array(t).astype(int) + np.array([ym, 0]))
        
        # get the ref and target points and adjust for the hstack
        # r = tuple(r.astype(int))
        # t = tuple(t.astype(int) + np.array([ym, 0]))

        # add the found points as marks
        cv2.circle(imgCombine, r, 30, (255, 0, 0), 10)

        cv2.putText(imgCombine, "ref feat " + str(i), 
                tuple(r + np.array([20, 0])),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 20)

        cv2.putText(imgCombine, "ref feat " + str(i), 
                tuple(r + np.array([20, 0])),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)

        cv2.circle(imgCombine, t, 30, (255, 0, 0), 10)

        cv2.putText(imgCombine, "tar feat " + str(i), 
            tuple(t + np.array([20, 0])),
            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 20)

        cv2.putText(imgCombine, "tar feat " + str(i), 
            tuple(t + np.array([20, 0])),
            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)

        n+=2    

    # Have a UI to select features from a raw image
    # NOTE this image should be pre-selected and be an input into a function
    r = xc/yc

    for i in range(n, feats*2):

        # get the x and y position from the feature
        x, y = roiselector(imgCombine)
        xme = int(np.mean(x))
        yme = int(np.mean(y))

        # add the found points as marks
        imgCombine = cv2.circle(imgCombine, (xme, yme), 30, (255, 0, 0), 10)

        # append reference and target information to the original list
        if i%2 == 0:
            obj = "ref"
            matchRef.append(np.array((xme, yme)))
            print("Feat: " + str(matchRef[-1]))
        else:
            obj = "tar"
            matchTar.append(np.array((xme, yme)) - np.array([ym, 0]))
            print("Feat: " + str(matchTar[-1]))

        print(str(i) + " + " + obj)

        feat = obj + " feat " + str(int(np.floor(i/2)))

        cv2.putText(imgCombine, feat, 
                tuple([xme, yme] + np.array([20, 0])),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 20)

        cv2.putText(imgCombine, feat, 
                tuple([xme, yme] + np.array([20, 0])),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)

    cv2.destroyAllWindows()

    return(matchRef, matchTar)


def roiselector(img):

    # function which calls the gui and get the co-ordinates 
    # Inputs    (img), numpy array image of interest
    # Outputs   (xme, yme), x and y positions on the original image of the points selected
    # ensure that the image is scaled to a resolution which can be used in the sceen
    
    xc, yc, c = img.shape
    r = xc/yc
    size = 700
    scale = yc / (size / r)

    # perform a search over a reduced size area
    roi = cv2.selectROI("image", cv2.resize(img, (int(size / r), size)))

    # get the postions
    y = np.array([roi[1], roi[1] + roi[3]])
    x = np.array([roi[0], roi[0] + roi[2]])

    # scale the positions back to their original size
    y = (np.round(y * scale)).astype(int)
    x = (np.round(x * scale)).astype(int)

    return(x, y)

if __name__ == "__main__":

    nameref = '/Volumes/USB/H653/3/masked/H653_01A.jpg'
    nametar = '/Volumes/USB/H653/3/masked/H653_02A.jpg'
    matchRef = {}
    matchTar = {}

    matchRef = [(1747, 2655), (1313, 1966), (649, 2438)]
    matchTar = [(1776, 2662), (1348, 1950), (657, 2414)]

    # featSelectPoint(nameref, nametar, matchRef, matchTar)

    alignedimghome = '/Volumes/USB/H653/3/'
    sample = 'H653_01A'

    featSelectArea(alignedimghome, sample)

