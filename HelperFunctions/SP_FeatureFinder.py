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
    from SP_SampleAnnotator import featChangePoint
else:
    from HelperFunctions.Utilities import *
    from HelperFunctions.SP_SampleAnnotator import featChangePoint

# for each fitted pair, create an object storing their key information
class feature:
    def __init__(self, refP = None, tarP = None, dist = None, size = None, res = None):
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

    def __repr__(self):
            return repr((self.dist, self.refP, self.tarP, self.size, self.res))

'''
TODO: explanation of what happens here.....

'''

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

    # This script finds features between two sequential samples (based on their
    # name) that correspond to biologically the same location. 
    # NOTE this is done on lower res jpg images to speed up the process as this is 
    # based on brute force methods. IT REQUIRES ADAPTING THE FEAT 
    # TO THE ORIGINAL SIZE TIF FILES --> info is stored as args in the .feat files
    # It is heavily based on the cv2.SIFT function
    # Inputs:   (imgsrc): source of the pre-processed images (from SP_SpecimenID)
    #           (dataDest): the location to save the txt files
    #           (imgdest): location to save the images which show the matching process
    #           (gridNo): number of grids (along horizontal axis) to use to analyse images
    #           (featMin): minimum number of features to apply per image
    #           (dist): the minimum distance between features in pixels
    # Outputs:  (): .feat files for each specimen which correspond to the neighbouring
    #               two slices (one as the reference and one as the target)
    #               jpg images which show where the features were found bewteen slices

    # counting the number of features found
    matchRefDict = {}    
    matchTarDict = {}
    bf = cv2.BFMatcher()   
    sift = cv2.xfeatures2d.SIFT_create()    # NOTE this required the contrib module --> research use only

    name_ref = nameFromPath(refsrc, 3)
    name_tar = nameFromPath(tarsrc, 3)
    imgName = name_ref + " <-- " + name_tar
    print("\nMatching " + name_tar + " to " + name_ref)

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
            img_refO = cv2.resize(img_refMaster, (int(img_refMaster.shape[1]* scl), int(img_refMaster.shape[0]*scl)))
            img_tarO = cv2.resize(img_tarMaster, (int(img_tarMaster.shape[1]* scl), int(img_tarMaster.shape[0]*scl)))

            # provide specimen specific image placement
            xrefDif, yrefDif, xtarDif, ytarDif, img_ref, img_tar = imgPlacement(nameFromPath(name_ref, 1), img_refO, img_tarO)

            x, y, c = img_ref.shape
            pg = int(np.round(x/gridNo))        # create a grid which is pg x pg pixel size
            sc = 20                          # create an overlap of sc pixels around this grid

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
            for c in range(int(np.ceil(x/pg))+1):
                for r in range(int(np.ceil(y/pg))+1):

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
                    if (np.sum((imgSect_ref==0)*1) >= imgSect_ref.size*0.6): #or (np.sum((imgSect_tar>0)*1) <= imgSect_tar.size):
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
                            featureInfo.tarP = (kp_tar[m.trainIdx].pt + + np.array([startY, startX])) / scl
                            featureInfo.dist = m.distance
                            featureInfo.size = kp_tar[m.trainIdx].size
                            featureInfo.res = scln
                            
                            # store the information specific to this resolution
                            resInfo.append(deepcopy(featureInfo))

            # find the spatially cohesive features
            if len(resInfo) > 0:
                matchedInfo += deepcopy(manualPoints)
                matchedInfo = matchMaker(matchedInfo, resInfo, manualAnno > 0, dist)

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
                print("\n\n!!!" + imgName + " doesn't have enough matches!!!!\n\n")
                manualPoints = featChangePoint(None, img_ref, img_tar, matchedInfo, nopts = 2, title = "Select 2 pairs of features to assist the automatic process")
                manualAnno += 1
                matchedInfo = []

            else:
                # if automatic process is still not working, just do 
                # the whole thing manually
                print("\n\n------- Manually annotating " + name_tar + " -------\n\n")
                matchedInfo = featChangePoint(None, img_ref, img_tar, matchedInfo, nopts = 8, title = "Automatic process failed, select 8 pairs of features for alignment")
                searching = False
                
    # ---------- update and save the found features ---------

    # if the sift search worked then update the dictionary
    # update the dictionaries
    names = []
    for fn, kp in enumerate(matchedInfo):

        # adjust for the initial standardisation of the image
        # and for the scale of the image
        refAdj = kp.refP - np.array([yrefDif, xrefDif]) / scl
        tarAdj = kp.tarP - np.array([ytarDif, xtarDif]) / scl

        matchedInfo[fn].refP = refAdj
        matchedInfo[fn].tarP = tarAdj
        
        # add matched feature,
        name = "feat_" + str(fn) + "_scl_" + str(kp.res)
        names.append(name)
        matchRefDict[name] = refAdj
        matchTarDict[name] = tarAdj

    # store the positions of the identified features for each image as 
    # BOTH a reference and target image. Include the image size this was 
    # processed at

    # ---------- create a combined image of the target and reference image matches ---------

    # make pictures to show the features found
    txtsz = 0.5

    # different colour for each resolution used to find features
    colours = [(255, 0, 0), (255, 0, 255), (255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 255, 255)]
    
    # img_refC = cv2.resize(img_refO, (yr, xr))
    # img_tarC = cv2.resize(img_tarO, (yt, xt))
    for i, nF in enumerate(matchedInfo):

        # if there is no match info just assign it to 0 (ie was a manual annotaiton)
        try: md = int(nF.dist); ms = np.round(nF.size, 2)
        except: md = np.inf; ms = np.inf
        
        name = names[i]

        # mark the feature
        newref = nF.refP.astype(int)
        newtar = nF.tarP.astype(int)

        cv2.circle(img_refC, tuple(newref.astype(int)), int(txtsz*10), colours[nF.res], int(txtsz*6))
        cv2.circle(img_tarC, tuple(newtar.astype(int)), int(txtsz*10), colours[nF.res], int(txtsz*6))

        # add the feature number onto the reference image
        cv2.putText(img = img_refC, text = str(name), 
        org = tuple(newref + np.array([-50, 15])),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = txtsz, color = (255, 255, 255), thickness = int(txtsz*10))

        cv2.putText(img = img_refC, text = str(name), 
        org = tuple(newref + np.array([-50, 15])),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = txtsz, color = (0, 0, 0), thickness = int(txtsz*4))
        
        # add the feature number and dist + size info
        text = str(name + ", d: " + str(md) + ", s: " + str(ms))
        text = str(name)

        cv2.putText(img = img_tarC, text = text, 
        org = tuple(newtar + np.array([-50, 15])),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = txtsz, color = (255, 255, 255), thickness = int(txtsz*10))

        cv2.putText(img = img_tarC, text = text, 
        org = tuple(newtar + np.array([-50, 15])),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = txtsz, color = (0, 0, 0), thickness = int(txtsz*4))


    # plot the features found per resolution
    '''
    img_refF = field.copy(); img_refF[:xr, :yr] = img_refC
    img_tarF = field.copy(); img_tarF[:xt, :yt] = img_tarC
    plt.imshow(np.hstack([img_refF, img_tarF])); plt.show()
    '''

    # draw the grid lines on the images
    for rp in range(int(np.ceil(y/pg))):
        # horizontal line
        r = int(rp * pg / scl)
        cv2.line(img_refC, (r, 0), (r, xm), (255, 255, 255), 4, 1)
        cv2.line(img_refC, (r, 0), (r, xm), (0, 0, 0), 2, 1)
        cv2.line(img_tarC, (r, 0), (r, xm), (255, 255, 255), 4, 1)
        cv2.line(img_tarC, (r, 0), (r, xm), (0, 0, 0), 2, 1)
    
    for cp in range(int(np.ceil(x/pg))):
        # vertical line
        c = int(cp * pg / scl)
        cv2.line(img_refC, (0, c), (ym, c), (255, 255, 255), 4, 1)
        cv2.line(img_refC, (0, c), (ym, c), (0, 0, 0), 2, 1)
        cv2.line(img_tarC, (0, c), (ym, c), (255, 255, 255), 4, 1)
        cv2.line(img_tarC, (0, c), (ym, c), (0, 0, 0), 2, 1)

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

    dictToTxt(matchRefDict, dataDest + name_ref + ".reffeat", shape = img_refO.shape, fit = False)
    dictToTxt(matchTarDict, dataDest + name_tar + ".tarfeat", shape = img_tarO.shape, fit = False)

def matchMaker(matchedInfo, resInfo, manual, dist = 50, featNo = None):

    # ---------- NOTE the theory underpinning this function ----------
    # The combination of individual features which produces the most matched features 
    # due to being the most spatially coherent with each other, most likely contain 
    # the features that represent actual biological structures becasue biological 
    # structures between individual samples which are physically close to each other 
    # are also spatailly coherent
    
    # this takes lists of all the information from the SIFT feature identification 
    # and bf matching and returns only n number of points which match the criteria
    # Inputs:   (matchInfo*), points that have already been found
    #           (resInfo), all the information returned by the sift operator after being
    #               brute forced matched for that specific resolution
    #           (manual), boolean as to whether the features that have been inputted 
    #               include manually annotated features. If True then treat the manual
    #               annotations as "ground truth" and don't overwrite
    #           (dist), sets the minimium distance between each feature. A larger distance 
    #               reduces the number of features that can be found but will also ensure 
    #               that features are  better spread across the sample, rather than being pack 
    #               arond a single strong feature
    #           (featNo), minimum number of features to find in the images. depreceated
    # Outputs:  (infoStore), the set of features and corresponding information that have been 
    #               found to have the most spatial coherence

    def findbestfeatures():

        # find the two best features ensuring that they are located 
        # a reasonable distane away

        # get the best features and remove from the list of sorted features
        matchInfo = []
        matchInfo.append(allInfo[0]); del allInfo[0]
        
        # get the second best feature, ensuring that it is an acceptable distance away from 
        # the best feature (sometimes two really good matches are found pretty much next
        # to each other which isn't useful for fitting)
        for n, i in enumerate(allInfo):

            # if the distance between the next best feature is less than 100 
            # pixels, don't use it
            if (np.sqrt(np.sum((matchInfo[0].refP - i.refP)**2)) < dist) or (np.sqrt(np.sum((matchInfo[0].tarP - i.tarP)**2)) < dist):
                continue
            
            # if the 2nd best feature found meets the criteria append and move on
            else:
                matchInfo.append(i); del allInfo[n]
                break

        return(matchInfo)

    def findgoodfeatures():

        # find new features in the ref and target tissue which are positioned 
        # in approximately the same location RELATIVE to the best features found already

        matchInfoN = []

        # append the already found matching info
        matchInfoN += matchInfo

        # from all the remaining features, find the ones that meet the characteristics:
        #   - The new feature is a distance away from all previous features
        #   - The new features on each sample are within a threshold angle and distance
        #   difference relative to the best features found 
        noFeatFind = 0  # keep track of the number of times a match has not been found
        for an, i in enumerate(allInfo):
            
            # if a featNo criteria is set, continue looking until the target number of features is 
            # met (or if there are no new features found for a duration of the search, bottom break)
            if featNo is int:
                if len(allInfo) >= featNo:
                    break

            # if the difference between the any of the already found feature is less than dist 
            # pixels, don't use it
            repeated = False

            for mi in matchInfoN:
                if (np.sqrt(np.sum((mi.refP - i.refP)**2)) < dist) or (np.sqrt(np.sum((mi.tarP - i.tarP)**2)) < dist):
                    repeated = True
                    break           

            if repeated:
                continue

            # get the relative angles of the new features compared to the best features 
            # found. Essentially triangulate the new feature relative to all the previously 
            # found features. This is particularly important as the further down the features
            # list used, the worse the sift match is so being in a position relative to all 
            # the other features found becomes a more important metric of fit
            # print("FEAT BEING PROCESSED")
            angdist = []
            ratiodist = []
            # use, up to, the top n best features: this is more useful/only used if finding
            # LOTS of features as this fitting procedure has a O(n^2) time complexity 
            # so limiting the search to this sacrifices limited accuracy for significant 
            # speed ups
            for n1, mi1 in enumerate(matchInfoN[:5]):
                for n2, mi2 in enumerate(matchInfoN[:5]):

                    # if the features are repeated, don't use it
                    if n1 == n2:
                        continue

                    # find the angle for the new point and all the previously found point
                    newrefang = findangle(mi1.refP, mi2.refP, i.refP)
                    newtarang = findangle(mi1.tarP, mi2.tarP, i.tarP)

                    # store the difference of this new point relative to all the ponts
                    # previously found
                    angdist.append(abs(newrefang - newtarang))

                # get the distances of the new points to the best feature
                newrefdist = np.sqrt(np.sum((mi1.refP - i.refP)**2))
                newtardist = np.sqrt(np.sum((mi1.tarP - i.tarP)**2))

                # finds how much larger the largest distance is compared to the smallest distance
                ratiodist.append((newtardist/newrefdist)**(1-((newrefdist>newtardist)*2)) - 1)

            # if the new feature is than 5 degress off and within 5% distance each other 
            # from all the previously found features then append 
            if (np.array(angdist) < 5/180*np.pi).all() and (np.array(ratiodist) < 0.05).all():
            # NOTE using median is a more "gentle" thresholding method. allows more features
            # but the standard of these new features is not as high
            # if np.median(angdist) < 180/180*np.pi and np.median(ratiodist) < 1:
                # add the features
                matchInfoN.append(i)
                noFeatFind = 0
            else:
                # if more than 5% of all the features are investigated and there are no
                # new features found, break the matching process (unlikely to find anymore
                # good features)
                noFeatFind += 1
                if noFeatFind > int(len(allInfo) * 0.05):
                    break

        return(matchInfoN)

    # store the initial feature found (ensures that there aren't less features 
    # than what we start with. Important for the manual annotations might have more 
    # features than the automatic process)

    # if there is a repeated feature, delete it (this comes from adding manual features)
    for n1, mi1 in enumerate(matchedInfo):
        for n2, mi2 in enumerate(matchedInfo):

            # if checking the same feature don't use it
            if n1 == n2:
                continue

            if (mi1.tarP == mi2.tarP).all() or (mi1.refP == mi2.refP).all():
                del matchedInfo[n2]

    infoStore = matchedInfo

    # create a list containing all the confirmed matched points and current ponts of interest
    allInfo = matchedInfo + resInfo

    # sort the information based on the distance
    allInfo = sorted(allInfo, key=lambda allInfo: allInfo.dist)

    # append the next n number of best fit features to the matches but 
    # ONLY if their angle from the two reference features is within a tolerance 
    # range --> this heavily assumes that the two best fits found are actually 
    # good features...
    # try up to 10 times: NOTE this is important because it reduces the reliance on the assumption 
    # that feature with lowest distance score is in fact and actual feature. This instead allows
    # for the feature finding process to rely more on the coherence of all the other 
    # features relative to each other
    for fits in range(10):
        # get the two best features 
        matchInfo = findbestfeatures()

        # find features which are spatially coherent relative to the best feature for both 
        # the referenc and target image and with the other constrains
        matchInfoN = findgoodfeatures()

        # add the 2nd best feature back into all the info as it was removed and can 
        # still be used
        allInfo.insert(0, matchInfo[1])

        # Store the features found and if more features are found with a different combination
        # of features then save that instead
        if len(matchInfoN) > len(infoStore): 
            # re-initialise the matches found
            infoStore = matchInfoN

        # if there are no more features to search through, break
        if (len(allInfo) < 3): # or (len(infoStore) > 200): 
            break
              
        # print(str(fits) + " = " + str(len(matchInfoN)) + "/" + str(len(resInfo)))
            
    # denseMatrixViewer([infoStore.refP, infoStore.tarP], True)

    return(infoStore)


# ------------ HARD CODED SPECIMEN SPECIFIC FEATURES ------------

def imgPlacement(name_spec, img_ref, img_tar):

    # this function takes the name of the specimen (target just because...) and 
    # performs hardcoded placements of the images within the field. This is okay because
    # each sample has its own processing quirks and it's definitely easier to do 
    # it like this than work it out properly
    # Inputs:   (name_tar), name of the specimen
    #           (img_ref, img_tar), images to place
    # Outputs:  (x/y ref/tar Dif), the shifts used to place the images
    #           (img_ref, img_tar), adjusted images

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
    

if __name__ == "__main__":
    
    multiprocessing.set_start_method('spawn')

    dataSource = '/Volumes/USB/Testing1/'
    dataSource = '/Volumes/Storage/H653A_11.3new/'
    dataSource = '/Volumes/Storage/H653A_11.3/'
    dataSource = '/Volumes/USB/H671A_18.5/'
    dataSource = '/Volumes/USB/H1029A_8.4/'
    dataSource = '/Volumes/USB/Test/'
    dataSource = '/Volumes/USB/H673A_7.6/'
    dataSource = '/Volumes/USB/H671A_18.5/'
    dataSource = '/Volumes/USB/H710B_6.1/'
    dataSource = '/Volumes/Storage/H710C_6.1/'

    name = ''
    size = 3
    cpuNo = 6

    featFind(dataSource, name, size, cpuNo)
