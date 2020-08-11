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

def featSelect(nameref, nametar, matchRef, matchTar, feats = 5):

    # this fuction brings up a gui which allows for a user to manually select
    # features on the images
    # Inputs:   (nameref), path of the reference image
    #           (nametar), path of the target image
    #           (matchRef), any already identified features on the reference image
    #           (matchTar), any already identified features on the target image
    #           (feats), defaults to finding 5 features
    # Outputs:  (matchRef, matchTar), updated ref and target features with new points added

    # get the images with insufficient features
    imgref = cv2.imread(nameref)
    imgtar = cv2.imread(nametar)

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
    imgCombine = np.hstack([img_refC, img_refC])
    xc, yc, c = imgCombine.shape

    rois = list()   # store all points
    n = 0           # iterator counter
    for i, (r, t) in enumerate(zip(matchRef, matchTar)):
        
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


    # imgROI = cv2.imread(imgs[np.where(np.array([sample in i for i in imgs]) == True)[0][0]])

    # Have a UI to select features from a raw image
    # NOTE this image should be pre-selected and be an input into a function

    r = xc/yc

    scale = xc/(1000 * r)

    for i in range(n, feats*2):
        # perform a search over a reduced size area
        roi = cv2.selectROI("image", cv2.resize(imgCombine, (1000, int(1000 * r))))

        # get the postions
        y = np.array([roi[1], roi[1] + roi[3]])
        x = np.array([roi[0], roi[0] + roi[2]])

        # scale the positions back to their original size
        yme = int(np.mean(y) * scale)
        xme = int(np.mean(x) * scale)

        # add the found points as marks
        imgCombine = cv2.circle(imgCombine, (xme, yme), 30, (255, 0, 0), 10)

        # append reference and target information to the original list
        if i%2 == 0:
            obj = "ref"
            matchRef.append(np.array((xme, yme)))
        else:
            obj = "tar"
            matchTar.append(np.array((xme, yme)) - np.array([ym, 0]))

        print(str(i) + " + " + obj)

        feat = obj + " feat " + str(int(np.floor(i/2)))

        cv2.putText(imgCombine, feat, 
                tuple([xme, yme] + np.array([20, 0])),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 20)

        cv2.putText(imgCombine, feat, 
                tuple([xme, yme] + np.array([20, 0])),
                cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 10)

        rois.append(roi)

    cv2.destroyAllWindows()


    return(matchRef, matchTar)

if __name__ == "__main__":

    nameref = '/Volumes/USB/H653/3/masked/H653_01A.jpg'
    nametar = '/Volumes/USB/H653/3/masked/H653_02A.jpg'
    matchRef = {}
    matchTar = {}

    matchTar = [(4794, 2869), (4897, 3562), (4856, 958)]
    matchRef = [(1928, 2865), (2040, 3536), (1949, 927)]

    featSelect(nameref, nametar, matchRef, matchTar)