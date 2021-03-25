'''
this script is performing and measuring the ANHIR registerations
'''

from math import comb
from numpy.lib.arraysetops import unique
from numpy.lib.recfunctions import rec_drop_fields
from pandas.core.frame import DataFrame
from scipy._lib.six import b
from HelperFunctions.SP_FeatureFinder import feature
from HelperFunctions.Utilities import getMatchingList, denseMatrixViewer, nameFromPath, drawLine
from HelperFunctions import *
from nonRigidAlign import nonRigidAlign, nonRigidDeform
from HelperFunctions.SP_AlignSamples import aligner
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import scipy

def annotateImages(dataHome, size, res):
    '''
    This adds the annotations to the images
    '''

    imgsrc = dataHome + str(size) + "/images/"
    annosrc = dataHome + "landmark/"

    imgs = sorted(glob(imgsrc + "*png"))
    annos = sorted(glob(annosrc + "*"))

    imgsToUse = getMatchingList(annos, imgs)

    for a, i in zip(annos, imgsToUse):
        img = cv2.imread(i)
        anno = pd.read_csv(a)
        for n in anno.index:
            pos = (np.array(anno.loc[n])[1:] * res).astype(int)
            cv2.circle(img, tuple(pos), 1, [0, 3*(n+1), 0], 20)

        # overwrite the original with the modified
        cv2.imwrite(i, img)

def getTransformedFeatures(dataHome, size, imgsrc):

    '''
    This extracts the annotated features from the annotated
    images 
    '''
    print("----- " + imgsrc + " -----")
    imgsrc = dataHome + str(size) + "/" + imgsrc + "/"
    annosrc = dataHome + "landmark/"

    annos = sorted(glob(annosrc + "S*"))

    imgs = sorted(glob(imgsrc + "*png"))
    imgsToUse = getMatchingList(annos, imgs)

    greenPosAll = {}
    greenPosNo = []

    for i, a in zip(imgsToUse, annos):
        name = nameFromPath(i, 3)
        img = cv2.imread(i)

        # get the positions of the features
        imgGreen = np.sum(img * np.array([-1, 1, -1]), axis = 2)
        greenPos = np.where(imgGreen > 0)
        gp = np.c_[greenPos[0], greenPos[1]]

        if len(gp) == 0:
            continue

        featsNo = len(pd.read_csv(a))

        # NOTE hardcoded because there is one feature which was removed from the 
        # sample during the specimenID and it's way to hard to compensate for that
        if name.find("7") > -1:
            featsNo -= 1

        greenPosNo.append(featsNo)

        greenPosAll[name] = gp

    return(greenPosAll, greenPosNo)

def getFeatureError(dataHome, size, imgSrc, plot = False):

    featurePos, featureNo = getTransformedFeatures(dataHome, size, imgSrc)

    imgdir = dataHome + str(size) + "/" + imgSrc + "/"
    imgs = sorted(glob(imgdir + "*png"))

    annosImgs = getMatchingList(list(featurePos.keys()), imgs)

    keys = list(featurePos.keys())
    names = []
    featureErrors = []

    for refImg, tarImg, refP, tarP, refNo, tarNo in zip(annosImgs[:-1], annosImgs[1:], keys[:-1], keys[1:], featureNo[:-1], featureNo[1:]):
        name = tarP + "_" + imgSrc
        print(name)
        names.append(name)

        refFeatures = featurePos[refP]
        tarFeatures = featurePos[tarP]

        ref = cv2.imread(refImg)
        tar = cv2.imread(tarImg)

        # get the positions of the features
        # ref[np.where(np.sum(ref * np.array([-1, 1, -1]), axis = 2) > 0)]

        # get the point of the feature
        refPoint = KMeans(refNo).fit(refFeatures).cluster_centers_
        tarPoint = KMeans(tarNo).fit(tarFeatures).cluster_centers_

        vrS = []
        trS = []
        refImgc = ref.copy()
        tarImgc = tar.copy()
        for r, t in zip(refPoint, tarPoint):
            vr = ref[tuple(r.astype(int))].astype(int)
            tr = tar[tuple(t.astype(int))].astype(int)

            # cv2.circle(refImgc, tuple(np.flip(r).astype(int)), 5, [255, 0, 0], 3)
            # cv2.circle(tarImgc, tuple(np.flip(t).astype(int)), 5, [255, 0, 0], 3)
            
            vrS.append(vr)
            trS.append(tr)

        #cv2.imshow("test", np.hstack([refImgc, tarImgc])); cv2.waitKey(0)
        
        # sort one axis to assist in matching
        # refPoint = refPoint[np.argsort(np.array(vrS)[:, 1])]
        # tarPoint = tarPoint[np.argsort(np.array(trS)[:, 1])]

        refMatchStore = []
        tarMatchStore = []
        
        # Use the relative distance of points to infer which ones are matches
        for n, tr in enumerate(trS):
            # find the feature match with the smallest distance
            error = np.sum((tr-vrS)**2, axis = 1)
            pos = np.argmin(error)

            # if the error is larger than 0 then we have not got a match
            if np.min(error) > 0: 
                continue

            if np.sum((refPoint[pos] - tarPoint[n])**2) > 100000:
                continue

            # create the target feature and its position in the list matches
            # the feature in the reference list
            refMatchStore.append(refPoint[pos])
            tarMatchStore.append(tarPoint[n])

            # remove the entry that was identified
            vrS = np.delete(vrS, pos, 0)
            refPoint = np.delete(refPoint, pos, 0)

        if plot:
            # matrix, shift = denseMatrixViewer([refMatchStore, tarMatchStore], plot = False)
            combImg = np.hstack([refImgc, tarImgc])
            for r, t in zip(refMatchStore, tarMatchStore):
                combImg = drawLine(combImg, np.flip(r), np.flip(t) + [ref.shape[1], 0], blur = 4)
                # matrix = drawLine(matrix, t-shift, r-shift)

            plt.imshow(combImg); plt.show()

            # plt.imshow(matrix); plt.show()

        # calculate the error between the features
        error=np.sum((np.array(refMatchStore) - np.array(tarMatchStore))**2, axis = 1)

        featureErrors.append(error)

    # convert into a dataframe
    errordf = pd.DataFrame(featureErrors).T
    errordf.columns = names

    return(errordf)

def processErrors(dataHome, size, datasrc, plot):

    '''
    Get the per feature error as a csv, save it and provide some quick stats
    '''

    annosrc = dataHome + "landmark/"

    names = nameFromPath(sorted(glob(annosrc + "S*")))

    # get the features and calculate the error between them
    featureErrors = getFeatureError(dataHome, size, datasrc, plot)

    # save as a csv
    featureErrors.to_csv(dataHome + "landmark/" + datasrc + ".csv")

def quickStats(dfPath): 

    # read in the df
    df = pd.read_csv(dfPath, index_col=0)
    name = nameFromPath(dfPath)
    print("\n---- " + name + " ----")
    # Provide some quick stats
    names = list(df.keys())

    for e, name in zip(df.T.values, names):
        # remove nans
        e = e[~np.isnan(e)]

        dist = np.round(np.sqrt(np.median(e)), 2)
        std = np.round(np.sqrt(np.std(e)), 2)
    
        print(name + ": " + str(dist) + "±" + str(std))

    return(df)

def featureErrorAnalyser(dataHome):

    # this is the hard coded order of the processing (ie image processing goes from
    # masked to aligned, then aligned to re...)
    processOrder = ["maskedSamples", "alignedSamples", "NLalignedSamples"]

    dataSrc = dataHome + "landmark/"
    dfs = sorted(glob(dataSrc + "*Samples.csv"))

    infoAll = []
    for d in dfs:
        infodf = quickStats(d)
        infoAll.append(np.sqrt(infodf))

    df = pd.concat(infoAll)

    names = sorted(nameFromPath(list(df.keys()), 4))
    ids = np.unique(nameFromPath(names, 3))

    print("---- pValues ----")
    for i in ids:
        for p0, p1 in zip(processOrder[:-1], processOrder[1:]):
            p0df = df[i + "_" + p0]
            p1df = df[i + "_" + p1]

            pV = scipy.stats.ttest_ind(p0df, p1df, nan_policy = 'omit').pvalue
            print(i + ": " + p0 + "-->" + p1 + " = " + str(np.round(pV, 4)))

    # plot the distribution of the errors
    # initialise
    info = []
    idstore = None
    names = []

    for n in names:
        name = n.split("_")[-1]
        id = nameFromPath(n)
        if idstore is None or id == idstore:
            info.append(df[n])
            names.append(name)
        else:
            plt.hist(info)
            plt.legend(names)
            plt.title(idstore)
            plt.show()
            info = []; info.append(df[n])
            names = []; names.append(id)
        idstore = id

    plt.hist(info)
    plt.legend(names)
    plt.title(id)
    plt.show()
            

    print("")

if __name__ == "__main__":
    
    dataHome = '/Volumes/USB/ANHIR/TargetTesting/COAD_08/'
    size = 3
    res = 0.2
    cpuNo = 1

    # transform the target images first 
    size = 3
    downsize(dataHome, size, res, cpuNo)
    specID(dataHome, size, cpuNo)
    '''
    featFind(dataHome, size, cpuNo, featMin = 50, gridNo = 1, dist = 50)
    align(dataHome, size, cpuNo, errorThreshold=500, fullScale=False)
    nonRigidAlign(dataHome, size, cpuNo, featsMin = 3, errorThreshold=500, selectCriteria="smooth", distFeats=200)
    
    input("Hit any key to continue AFTER copying the found info into the new folder")
    '''

    size = 2.6
    src = dataHome + str(size) + "/"
    # apply the feataures onto the images and deform them exactly the same as 
    # the target images
    '''
    annotateImages(dataHome, size, res)
    specID(dataHome, size, cpuNo, None)
    # linear alignment
    aligner(src + '/maskedSamples/', src + '/info/', src + '/alignedSamples/', cpuNo, errorThreshold = 500)
    
    # alignment with the NL features
    aligner(src + '/alignedSamples/', src + '/infoNL/', src + '/ReAlignedSamples/', cpuNo, errorThreshold = 500)
    
    # NL alignment
    nonRigidDeform(dataHome + str(size) + "/RealignedSamples/", \
        dataHome + str(size) + "/NLAlignedSamples/", \
            dataHome + str(size) + "/FeatureSections/", \
                prefix = "png")
    input("Press enter after renaming the NL samples")
    '''
    # get the errors of the linear aligned features 
    # processErrors(dataHome, size, "maskedSamples", False)
    # processErrors(dataHome, size, "alignedSamples", False)
    # processErrors(dataHome, size, "RealignedSamples", False)
    # processErrors(dataHome, size, "NLalignedSamples", False)

    featureErrorAnalyser(dataHome)