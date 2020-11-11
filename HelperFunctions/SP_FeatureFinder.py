'''

This script automatically finds features between different layers of tissue,
works best on images processed by SP_SpecimenID

NOTE this script functionally depends on cv2 which would be implemented a LOT 
faster on C++ --> consider re-writing for speed

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os
from multiprocessing import Pool
import multiprocessing
from itertools import repeat
from copy import copy, deepcopy
if __name__ != "HelperFunctions.SP_FeatureFinder":
    from Utilities import *
    from SP_SampleAnnotator import featChangePoint, matchMaker
else:
    from HelperFunctions.Utilities import *
    from HelperFunctions.SP_SampleAnnotator import featChangePoint, matchMaker

# for each fitted pair, create an object storing their key information
class feature:
    def __init__(self, refP = None, tarP = None, dist = None, size = None, res = -1, ID = None):
        # the position of the match on the reference image
        self.refP = refP

        # the position of the match on the target image
        self.tarP = tarP

        # eucledian error of the difference in gradient fields
        self.dist = dist

        # the size of the feature
        self.size = size

        # the resolution index of the image that was processed
        self.res = res

        # Feature number
        self.ID = ID

    def __repr__(self):
            return repr((self.dist, self.refP, self.tarP, self.size, self.res))

'''
TODO:
    - Find a method to pre-process the image to further enhance the sift operator
        UPDATE I think it is unlikely I will find a way to radically improve SIFT. It is 
        easier to just allow it to do the best it can and add manual points on sections which 
        don't match

    - EXTRA FOR EXPERTS: conver this into C++ for superior speed

    - Make the sift operator work over each image individually and collect all 
    the key points and descriptors for ONLY sections which contain a threshold level
    of no black points (ie nothing). All of these descriptors are then combined into
    a single list and put into the BF matcher where the rest of the script continues etc.
        The benefit of this is that it will mean that there doesn't need to be any 
        hard coded gridding ---> tried this, doesn't work because the structures in 
        the tissue are so repetitive that there is almost guaranteed to be a good match 
        in a non-spatially relevant area of the tissue which causes a false match
'''


def featFind(dataHome, name, size, cpuNo = False):
    
    # this is the function called by main. Organises the inputs for findFeats

    # get the size specific source of information
    datasrc = dataHome + str(size) + "/"

    # gets the images for processing
    imgsrc = datasrc + "masked/"

    # specify where the outputs are saved
    dataDest = datasrc + "info/"
    imgDest = datasrc + "matched/"
    dirMaker(dataDest)
    dirMaker(imgDest)

    # set the parameters
    gridNo = 2
    featMin = 40
    dist = 10

    # get the images
    imgs = sorted(glob(imgsrc + "*.png"))

    if cpuNo is False:
        # serialisation (mainly debuggin)
        for refsrc, tarsrc in zip(imgs[:-1], imgs[1:]):
            findFeats(refsrc, tarsrc, dataDest, imgDest, gridNo, featMin, dist)

    else:
        # parallelise with n cores
        with Pool(processes=cpuNo) as pool:
            pool.starmap(findFeats, zip(imgs[:-1], imgs[1:], repeat(dataDest), repeat(imgDest), repeat(gridNo), repeat(featMin), repeat(dist)))
        
def findFeats(refsrc, tarsrc, dataDest, imgdest, gridNo, featMin = 20, dist = 50):

    '''
    This script finds features between two sequential samples (based on their
    name) that correspond to biologically the same location. 
    NOTE this is done on lower res jpg images to speed up the process as this is 
    based on brute force methods. IT REQUIRES ADAPTING THE FEAT 
    TO THE ORIGINAL SIZE TIF FILES --> info is stored as args in the .feat files
    It is heavily based on the cv2.SIFT function
    
    Inputs:   
        (imgsrc), source of the pre-processed images (from SP_SpecimenID)
        (dataDest), the location to save the txt files
        (imgdest), location to save the images which show the matching process
        (gridNo), number of grids (along horizontal axis) to use to analyse images
        (featMin), minimum number of features to apply per image
        (dist), the minimum distance between features in pixels

    Outputs: 
        (), saves .feat files for each specimen which correspond to the neighbouring
        two slices (one as the reference and one as the target)
        (), jpg images which show where the features were found bewteen slices
    '''

    # counting the number of features found
    matchRefDict = {}    
    matchTarDict = {}
    bf = cv2.BFMatcher()   
    sift = cv2.xfeatures2d.SIFT_create()    # NOTE this required the contrib module --> research use only

    name_ref = nameFromPath(refsrc, 3)
    name_tar = nameFromPath(tarsrc, 3)
    imgName = name_ref + " <-- " + name_tar
    print("-Matching " + name_tar + " to " + name_ref)

    # load in the images
    img_refMaster = cv2.imread(refsrc)
    img_tarMaster = cv2.imread(tarsrc)

    # make a copy of the images to modify
    img_refC = img_refMaster.copy()
    img_tarC = img_tarMaster.copy()

    # normalise for all the colour channels
    # fig, (bx1, bx2, bx3) = plt.subplots(1, 3)
    for c in range(img_tarC.shape[2]):
        img_tarC[:, :, c] = hist_match(img_tarC[:, :, c], img_refC[:, :, c])

    # get the image dimensions
    xr, yr, cr = img_refMaster.shape
    xt, yt, ct = img_tarMaster.shape
    xm, ym, cm = np.max(np.array([(xr, yr, cr), (xt, yt, ct)]), axis = 0)
        
    # create a max size field of both images
    field = np.zeros((xm, ym, cm)).astype(np.uint8)

    # create an object to store all the feature information
    featureInfo = feature()

    # store all feature objects which describe the confirmed matches
    matchedInfo = []

    # specify these are automatic annotations
    manualAnno = 0

    # perform a multi-zoom fitting procedure
    # It is preferable to use a lower resolution image because it means that features that
    # are larger on the sample are being found and the speed of processing is significantly faster.
    # However if there are not enough features per a treshold
    scales = [0.1, 0.2, 0.3, 0.5, 0.8, 1]
    searching = True
    manualPoints = []

    # find all the spatially cohesive features in the samples
    matchedInfo, xrefDif, yrefDif, xtarDif, ytarDif, scl = allFeatSearch(img_refMaster, img_tarMaster, scales = scales, \
        dist = dist, featMin = featMin, name_ref = name_ref, gridNo = gridNo)
    x, y = int(xt*scl), int(yt*scl)

    # ---------- update and save the found features ---------

    # if the sift search worked then update the dictionary
    # update the dictionaries
    for fn, kp in enumerate(matchedInfo):

        # adjust for the initial standardisation of the image
        # and for the scale of the image
        refAdj = kp.refP - np.array([yrefDif, xrefDif]) / scl
        tarAdj = kp.tarP - np.array([ytarDif, xtarDif]) / scl

        matchedInfo[fn].refP = refAdj
        matchedInfo[fn].tarP = tarAdj
        matchedInfo[fn].ID = fn
        
        # create feature dictionary
        name = "feat_" + str(fn) + "_scl_" + str(kp.res)
        matchRefDict[name] = refAdj
        matchTarDict[name] = tarAdj

    # store the positions of the identified features for each image as 
    # BOTH a reference and target image. Include the image size this was 
    # processed at

    # ---------- create a combined image of the target and reference image matches ---------

    # make pictures to show the features found
    txtsz = 0.5

    # img_refC = cv2.resize(img_refO, (yr, xr))
    # img_tarC = cv2.resize(img_tarO, (yt, xt))

    # annotate the image with the feature info
    img_refC, img_tarC = nameFeatures(img_refC, img_tarC, matchedInfo)
    
    # plot the features found per resolution
    '''
    img_refF = field.copy(); img_refF[:xr, :yr] = img_refC
    img_tarF = field.copy(); img_tarF[:xt, :yt] = img_tarC
    plt.imshow(np.hstack([img_refF, img_tarF])); plt.show()
    '''

    print("     " + imgName + " has " + str(len(matchRefDict)) + " features, scale = " + str(scl))

    # print a combined image showing the matches
    img_refF = field.copy(); img_refF[:xr, :yr] = img_refC
    img_tarF = field.copy(); img_tarF[:xt, :yt] = img_tarC
    # plt.imshow(np.hstack([img_refF, img_tarF])); plt.show()

    imgCombine = np.hstack([img_refF, img_tarF])

    # put text for the number of features
    cv2.putText(img = imgCombine, text = str("Feats found = " + str(len(matchTarDict))), 
        org = (50, 50),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = txtsz*4, color = (255, 255, 255), thickness = int(txtsz*12))
    cv2.putText(img = imgCombine, text = str("Feats found = " + str(len(matchTarDict))), 
    org = (50, 50),
    fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = txtsz*4, color = (0, 0, 0), thickness = int(txtsz*6))

    cv2.imwrite(imgdest + "/" + name_ref + " <-- " + name_tar + ".jpg", imgCombine)

    dictToTxt(matchRefDict, dataDest + name_ref + ".reffeat", shape = img_refMaster.shape, fit = False)
    dictToTxt(matchTarDict, dataDest + name_tar + ".tarfeat", shape = img_tarMaster.shape, fit = False)

# ------------ HARD CODED SPECIMEN SPECIFIC FEATURES ------------

def imgPlacement(name_spec, img_ref, img_tar):

    '''
    this function takes the name of the specimen (target just because...) and 
    performs hardcoded placements of the images within the field. This is okay because
    each sample has its own processing quirks and it's definitely easier to do 
    it like this than work it out properly
    
    Inputs:   
        (name_tar), name of the specimen
        (img_ref, img_tar), images to place
    
    Outputs:  
        (x/y ref/tar Dif), the shifts used to place the images
        (img_ref, img_tar), adjusted images
    '''

    # get the image dimensions, NOTE this is done in the main function but I 
    # didn't want to feed all those variables into this function... seems very messy
    xr, yr, cr = img_ref.shape
    xt, yt, ct = img_tar.shape
    xm, ym, cm = np.max(np.array([(xr, yr, cr), (xt, yt, ct)]), axis = 0)
    
    # create a max size field of both images
    field = np.zeros((xm, ym, cm)).astype(np.uint8)

    # something for the bottom right
    if name_spec == 'H653A' or name_spec == 'H710C':
        # these are the origin shifts to adapt each image
        xrefDif = xm-xr
        yrefDif = ym-yr
        xtarDif = xm-xt
        ytarDif = ym-yt

        # re-assign the images to the left of the image (NOTE this is for H563A which has
        # been segmented and the samples are very commonly best aligned on the left side)
        img_refF = field.copy(); img_refF[-xr:, -yr:] = img_ref
        img_tarF = field.copy(); img_tarF[-xt:, -yt:, :] = img_tar
    
    # something in the middle
    elif name_spec == 'H1029A':
        # position the further right, lowest point of the each of the target and 
        # reference images at the bottom right positions of the fields
        pos = np.where(img_tar != 0)
        xmaxt = np.max(pos[1])
        ymaxt = pos[0][np.where(pos[1] == xmaxt)[0]][-1]

        pos = np.where(img_ref != 0)
        xmaxr = np.max(pos[1])
        ymaxr = pos[0][np.where(pos[1] == xmaxr)[0]][-1]

        img_tarp = img_tar[:ymaxt, :xmaxt]
        img_refp = img_ref[:ymaxr, :xmaxr]

        xrp, yrp, c = img_refp.shape
        xtp, ytp, c = img_tarp.shape

        xm, ym, cm = np.max(np.array([(xrp, yrp, c), (xtp, ytp, c)]), axis = 0)
        fieldp = np.zeros((xm, ym, cm)).astype(np.uint8)

        xrefDif = xm-xrp
        yrefDif = ym-yrp
        xtarDif = xm-xtp
        ytarDif = ym-ytp

        img_refF = fieldp.copy(); img_refF[-xrp:, -yrp:, :] = img_refp
        img_tarF = fieldp.copy(); img_tarF[-xtp:, -ytp:, :] = img_tarp
        
    elif name_spec == 'H710B':
        # put the image in the middle of the field
        xrefDif = int((xm-xr) / 2)
        yrefDif = int((ym-yr) / 2)
        xtarDif = int((xm-xt) / 2)
        ytarDif = int((ym-yt) / 2)

        img_refF = field.copy(); img_refF[xrefDif:xrefDif+xr, yrefDif:yrefDif+yr, :] = img_ref
        img_tarF = field.copy(); img_tarF[xtarDif:xtarDif+xt, ytarDif:ytarDif+yt, :] = img_tar


    # if not specifically hardcoded, just place in the top left
    else:
        xrefDif = 0
        yrefDif = 0
        xtarDif = 0
        ytarDif = 0

        img_refF = field.copy(); img_refF[:xr, :yr, :] = img_ref
        img_tarF = field.copy(); img_tarF[:xt, :yt, :] = img_tar
        
    return(xrefDif, yrefDif, xtarDif, ytarDif, img_refF, img_tarF)
    
def allFeatSearch(imgRef, imgTar, matchedInfo = [], scales = [0.2, 0.3, 0.5, 0.8, 1], dist = 1, featMin = 20, name_ref = None, gridNo = 1, sc = 20, cpuNo = False, tol = 0.05, spawnPoints = 10, anchorPoints = 5):

    '''
    Find the spatially cohesive features in an image

    Inputs:     
        (img*), numpy array of images
        (matchedInfo), previously identified features to act as anchor points for 
        spatial cohesiveness
        (scales), list of resolution multiplication factors to use to search for features
        (dist), minimum distance between features when searching
        (featMin), minimum number of features which must be found
        (name_ref), name of the specimen. Hardcoded image placements to help sift. 
        Not necessary but if you know where the image could kind of go is useful...
        (gridNo), grid sizes to segment the sift searches
        (sc), overlap of sc pixels around the grid for sift searches
        (cpuNo)
        (tol)
        (spawnPoints)
        (anchorPoints)

    Outputs:    
        (matchedInfo), feature positions
        (xrefDif, yrefDif, xtarDif, ytarDif) , the shifts use for each image to align
        (scl), final scale resolution used to find threshold features
    '''

    # initialise the bf and sift 
    bf = cv2.BFMatcher()   
    sift = cv2.xfeatures2d.SIFT_create()    # NOTE this required the contrib module --> research use only
    
    # initialise
    searching = True
    manualPoints = []
    featureInfo = feature()
    manualAnno = 0

    # continue to perform matching until quit. This allows for the manual features to 
    # be included as part of the fitting process
    while searching:

        # perform feature detection at multiple scales to initially identify large
        # features which can be used to find more robust spatially cohesive features at
        # higher resoltions, and to speed up the operations if the min features to be found
        # are met in the low resolution images
        for scln, scl in enumerate(scales):

            # store the scale specific information
            resInfo = []

            # load the image at the specified scale
            img_refO = cv2.resize(imgRef, (int(imgRef.shape[1]* scl), int(imgRef.shape[0]*scl)))
            img_tarO = cv2.resize(imgTar, (int(imgTar.shape[1]* scl), int(imgTar.shape[0]*scl)))

            # provide specimen specific image placement
            xrefDif, yrefDif, xtarDif, ytarDif, img_ref, img_tar = imgPlacement(nameFromPath(name_ref, 1), img_refO, img_tarO)

            x, y, c = img_ref.shape
            pg = int(np.round(x/gridNo))        # create a grid which is pg x pg pixel size

            # perform a sift operation over the entire image and find all the matching 
            # features --> more for demo purposes on why the below method is implemented
            '''
            kp_ref, des_ref = sift.detectAndCompute(img_ref,None)
            kp_tar, des_tar = sift.detectAndCompute(img_tar,None)
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des_ref,des_tar, k=2)     
            # cv2.drawMatchesKnn expects list of lists as matches.
            img3 = cv2.drawMatchesKnn(img_ref,kp_ref,img_tar,kp_tar,matches, None, flags=2)
            '''

            # ---------- identify features on the reference and target image ---------

            # The reason why a scanning method over the images is implemented, rather than
            # just letting sift work across the full images, is because the nature of the 
            # samples (biological tissue) means there are many repeating structures which has
            # lead to feature matching in non-sensible locations. This scanning method assumes
            # there is APPROXIMATE sample placement (ie there are samples in the middle of the 
            for c in range(int(np.ceil(x/pg))):
                for r in range(int(np.ceil(y/pg))):

                    # extract a small overlapping grid from both images
                    startX = np.clip(int(c*pg-sc), 0, x)
                    endX = np.clip(int((c+1)*pg+sc), 0, x)
                    startY = np.clip(int(r*pg-sc), 0, y)
                    endY = np.clip(int((r+1)*pg+sc), 0, y)

                    # extract a small grid from both image
                    imgSect_ref = img_ref[c*pg:(c+1)*pg, r*pg:(r+1)*pg, :]
                    imgSect_tar = img_tar[startX:endX, startY:endY, :]  # NOTE target area search is expaneded

                    # if the proporption of information within the slide is black (ie background)
                    # is more than a threshold, don't process
                    if (np.sum((imgSect_ref==0)*1) >= imgSect_ref.size*0.6) and (gridNo != 1): #or (np.sum((imgSect_tar>0)*1) <= imgSect_tar.size):
                        continue

                    # plt.imshow(imgSect_ref); plt.show()
                    # get the key points and descriptors of each section
                    kp_ref, des_ref = sift.detectAndCompute(imgSect_ref,None)
                    kp_tar, des_tar = sift.detectAndCompute(imgSect_tar,None)

                    # only further process if there are matches found in both samples
                    if (des_ref is not None) and (des_tar is not None):
                        # identify strongly identifiable features in both the target and 
                        # reference tissues
                        matches = bf.match(des_ref, des_tar)

                        # get all the matches, adjust for the window used 
                        # get all the matches, adjust for the window used 
                        for m in matches:

                            # store the feature information as it appears on the original sized image
                            featureInfo.refP = (kp_ref[m.queryIdx].pt + np.array([r*pg, c*pg])) / scl
                            featureInfo.tarP = (kp_tar[m.trainIdx].pt + np.array([startY, startX])) / scl
                            featureInfo.dist = m.distance
                            featureInfo.size = kp_tar[m.trainIdx].size
                            featureInfo.res = scln
                            
                            # store the information specific to this resolution
                            resInfo.append(deepcopy(featureInfo))

            # find the spatially cohesive features
            if len(resInfo) > 0:
                matchedInfo += deepcopy(manualPoints)
                # matchedInfo = matchMaker(resInfo, matchedInfo, manualAnno > 0, dist)
                matchedInfo = matchMaker(resInfo, matchedInfo, manualAnno > 0, dist * scl, cpuNo, tol, spawnPoints, anchorPoints)

                for n, m in enumerate(matchedInfo):
                    matchedInfo[n].dist = 0.01 * n # preserve the order of the features
                                                            # fit but ensure that it is very low to 
                                                            # indicate that it is a "confirmed" fit
                    matchedInfo[n].size = 100


                # plotting all the features found for this specific resoltuion
                '''
                ir = img_ref.copy()
                it = img_tar.copy()
                for i in resInfo:
                    cv2.circle(ir, tuple((i.refP*scl).astype(int)), 3, (255, 0, 0))
                    cv2.circle(it, tuple((i.tarP*scl).astype(int)), 3, (255, 0, 0))

                plt.imshow(np.hstack([ir, it])); plt.show()
                '''

            # if the number of features founds exceeds the minimum then break
            if len(matchedInfo) >= featMin:
                searching = False
                break

        # If after searching through up until the full resolution image there is not a 
        # threshold number of features found provide manual annotations up until that featMin
        if len(matchedInfo) < featMin:

            if manualAnno < 2:
                # NOTE use these to then perform another round of fitting. These essentially 
                # become the manual "anchor" points for the spatial coherence to work with. 
                print("\n\n!!!Not enough matches!!!!\n\n")
                manualPoints = featChangePoint(None, img_ref, img_tar, matchedInfo, nopts = 2, title = "Select 2 pairs of features to assist the automatic process")
                manualAnno += 1
                matchedInfo = []

            else:
                # if automatic process is still not working, just do 
                # the whole thing manually
                print("\n\n------- Manually annotating " + name_tar + " -------\n\n")
                matchedInfo = featChangePoint(None, img_ref, img_tar, matchedInfo, nopts = 8, title = "Automatic process failed, select 8 pairs of features for alignment")
                searching = False

    return(matchedInfo, xrefDif, yrefDif, xtarDif, ytarDif, scl)
          


if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn')

    dataSource = '/Volumes/USB/Testing1/'
    dataSource = '/Volumes/Storage/H653A_11.3new/'
    dataSource = '/Volumes/Storage/H653A_11.3/'
    dataSource = '/Volumes/USB/H671A_18.5/'
    dataSource = '/Volumes/USB/H1029A_8.4/'
    dataSource = '/Volumes/USB/H673A_7.6/'
    dataSource = '/Volumes/USB/H671A_18.5/'
    dataSource = '/Volumes/USB/H710B_6.1/'
    dataSource = '/Volumes/Storage/H710C_6.1/'
    dataSource = '/Volumes/USB/Test/'

    name = ''
    size = 3
    cpuNo = False

    featFind(dataSource, name, size, cpuNo)
