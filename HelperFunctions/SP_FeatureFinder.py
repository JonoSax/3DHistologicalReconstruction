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
if __name__ != "HelperFunctions.SP_FeatureFinder":
    from Utilities import *
    from SP_SampleAnnotator import featSelectPoint
else:
    from HelperFunctions.Utilities import *
    from HelperFunctions.SP_SampleAnnotator import featSelectPoint

# set the multiprocessing type (only do it if it's not spawned)
if __name__ != "__mp_main__": multiprocessing.set_start_method('spawn')

'''
TODO: explanation of what happens here.....

'''

'''
TODO:
    - Find a method to pre-process the image to further enhance the sift operator
        UPDATE I think it is unlikely I will find a way to radically improve SIFT. It is 
        easier to just allow it to do the best it can and add manual points on sections which 
        don't match

    - Apply the sift operators over the target image ONCE and then organise these 
    into the appropriate grids for searching --> NOTE this is only useful when it is 
    on a reference image which I have tried and it doesn't work very well

    - EXTRA FOR EXPERTS: conver this into C++ for superior speed

    - Make the sift operator work over each image individually and collect all 
    the key points and descriptors for ONLY sections which contain a threshold level
    of no black points (ie nothing). All of these descriptors are then combined into
    a single list and put into the BF matcher where the rest of the script continues etc.
        The benefit of this is that it will mean that there doesn't need to be any 
        hard coded gridding 
'''


def featFind(dataHome, name, size):
    
    # this is the function called by main. Organises the inputs for findFeats
    
    # use only 3/4 the CPU resources availabe
    cpuCount = int(multiprocessing.cpu_count() * 0.75)
    serialise = False

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
    gridNo = 6
    featNo = None
    dist = 30

    # get the images
    imgs = sorted(glob(imgsrc + "*.png"))

    if serialise:
        # serialisation (mainly debuggin)
        for refsrc, tarsrc in zip(imgs[:-1], imgs[1:]):
            findFeats(refsrc, tarsrc, dataDest, imgDest, gridNo, featNo, dist)

    else:
        # parallelise with n cores
        with Pool(processes=cpuCount) as pool:
            pool.starmap(findFeats, zip(imgs[:-1], imgs[1:], repeat(dataDest), repeat(imgDest), repeat(gridNo), repeat(featNo), repeat(dist)))
        

def findFeats(refsrc, tarsrc, dataDest, imgdest, gridNo = 1, featNo = None, dist = 50):

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
    #           (featNo): number of features to apply per image
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

    print("Matching " + name_tar + " to " + name_ref)

    # load in the images
    img_refO = cv2.imread(refsrc)
    img_tarO = cv2.imread(tarsrc)

    # get the image dimensions
    xr, yr, cr = img_refO.shape
    xt, yt, ct = img_tarO.shape
    xm, ym, cm = np.max(np.array([(xr, yr, cr), (xt, yt, ct)]), axis = 0)
        
    # create a max size field of both images
    field = np.zeros((xm, ym, cm)).astype(np.uint8)

    # provide specimen specific image placement
    xrefDif, yrefDif, xtarDif, ytarDif, img_ref, img_tar = imgPlacement(nameFromPath(name_ref, 1), img_refO, img_tarO)

    # normalise for all the colour channels
    # fig, (bx1, bx2, bx3) = plt.subplots(1, 3)
    for c in range(img_tar.shape[2]):
        img_tar[:, :, c] = hist_match(img_tar[:, :, c], img_ref[:, :, c])
    
    x, y, c = img_ref.shape
    pg = int(np.round(y/gridNo, -1))     # pixel grid size
    sc = 0.2    # the extra 1D length size of the target section

    matchDistance = []
    matchRef = []
    matchTar = []
    matchSize = []

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

    # iterate through a pixel grid of pg ** 2 x c size
    # NOTE the target section is (pg + 2sc) ** 2 x c in size --> idea is that the
        # target secition will have some significant shift therefore should look in
        # a larger area
    # The reason why a scanning method over the images is implemented, rather than
    # just letting sift work across the full images, is because the nature of the 
    # samples (biological tissue) means there are many repeating structures which has
    # lead to feature matching in non-sensible locations. This scanning method assumes
    # there is APPROXIMATE sample placement (ie there are samples in the middle of the 
    # slice)
    allMatches = []
    allrefpt = []
    alltarpt = []
    alldistance = []
    allsize = []
    for c in range(1, int(np.ceil(x/pg)) - 1):
        for r in range(1, int(np.ceil(y/pg)) - 1):

            # extract a small grid from both image
            imgSect_ref = img_ref[c*pg:(c+1)*pg, r*pg:(r+1)*pg, :]
            imgSect_tar = img_tar[int((c-sc)*pg):int((c+1+sc)*pg), int((r-sc)*pg):int((r+1+sc)*pg), :]  # NOTE target area search is expaneded

            # if the proporption of information within the slide is black (ie background)
            # is more than a threshold, don't process
            if (np.sum((imgSect_ref==0)*1) >= imgSect_ref.size*0.1): #or (np.sum((imgSect_tar>0)*1) <= imgSect_tar.size):
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
                allMatches.append(bf.match(des_ref, des_tar))

                # get all the matches, adjust for the window used 
                for m in matches:
                    allrefpt.append(kp_ref[m.queryIdx].pt + np.array([r*pg, c*pg]))
                    alltarpt.append(kp_tar[m.trainIdx].pt + + np.array([(r-sc)*pg, (c-sc)*pg]))
                    allsize.append(kp_tar[m.trainIdx].size)
                    alldistance.append(m.distance)

    # if there are more than featNo matches then pick the BEST of all those features
    if featNo is None or len(alltarpt) > featNo:
        matchRef, matchTar, matchSize, matchDistance = matchMakerN(allrefpt, alltarpt, allsize, alldistance, featNo)
        print(name_tar + " has " + str(len(matchTar)) + " features")

    # if there are less than the specific number of features, create a GUI to perform 
    # manual matching
    elif len(alltarpt) < featNo:
        matchRef, matchTar = featSelectPoint(img_ref, img_tar, [], [], 3)

    # ---------- update and save the found features ---------

    # update the dictionaries
    for fn, (kr, kt) in enumerate(zip(matchRef, matchTar)):

        # add matched feature, adjust for the initial standardisation of the image
        matchRefDict["feat_" + str(fn)] = kr - np.array([yrefDif, xrefDif])
        matchTarDict["feat_" + str(fn)] = kt - np.array([ytarDif, xtarDif])

    # store the positions of the identified features for each image as 
    # BOTH a reference and target image. Include the image size this was 
    # processed at
    dictToTxt(matchRefDict, dataDest + "/" + name_ref + ".reffeat", shape = img_refO.shape, fit = False)
    dictToTxt(matchTarDict, dataDest + "/" + name_tar + ".tarfeat", shape = img_tarO.shape, fit = False)

    # ---------- create a combined image of the target and reference image matches ---------

    # make pictures to show the features found
    img_refC = img_refO.copy()
    img_tarC = img_tarO.copy()
    txtsz = 1
    for i, nF in enumerate(matchTarDict):

        # if there is no match info just assign it to 0 (ie was a manual annotaiton)
        try: md = int(matchDistance[i]); ms = np.round(matchSize[i], 2)
        except: md = np.inf; ms = np.inf

        # mark the feature
        newref = matchRefDict[nF].astype(int)
        tar = matchTarDict[nF].astype(int)

        cv2.circle(img_refC, tuple(newref.astype(int)), int(txtsz*10), (255, 0, 0), int(txtsz*6))
        cv2.circle(img_tarC, tuple(tar.astype(int)), int(txtsz*10), (255, 0, 0), int(txtsz*6))

        # add the feature number onto the image
        cv2.putText(img_refC, str(nF), 
        tuple(newref + np.array([-50, 50])),
        cv2.FONT_HERSHEY_SIMPLEX, int(txtsz), (255, 255, 255), int(txtsz*10))
        cv2.putText(img_refC, str(nF), 
        tuple(newref + np.array([-50, 50])),
        cv2.FONT_HERSHEY_SIMPLEX, int(txtsz), (0, 0, 0), int(txtsz*4))
        
        text = str(nF + ", d: " + str(md) + ", s: " + str(ms))

        cv2.putText(img_tarC, text,
        tuple(tar + np.array([-200, 50])),
        cv2.FONT_HERSHEY_SIMPLEX, int(txtsz), (255, 255, 255), int(txtsz*10))
        cv2.putText(img_tarC, text, 
        tuple(tar + np.array([-200, 50])),
        cv2.FONT_HERSHEY_SIMPLEX, int(txtsz), (0, 0, 0), int(txtsz*4))

    # draw the grid lines on the images
    for r in range(0, y, pg):
        # horizontal line
        cv2.line(img_refC, (r, 0), (r, x), (255, 255, 255), 4, 1)
        cv2.line(img_refC, (r, 0), (r, x), (0, 0, 0), 2, 1)
        cv2.line(img_tarC, (r, 0), (r, x), (255, 255, 255), 4, 1)
        cv2.line(img_tarC, (r, 0), (r, x), (0, 0, 0), 2, 1)
    
    for c in range(0, x, pg):
        # vertical line
        cv2.line(img_refC, (0, c), (y, c), (255, 255, 255), 4, 1)
        cv2.line(img_refC, (0, c), (y, c), (0, 0, 0), 2, 1)
        cv2.line(img_tarC, (0, c), (y, c), (255, 255, 255), 4, 1)
        cv2.line(img_tarC, (0, c), (y, c), (0, 0, 0), 2, 1)

    # print a combined image showing the matches
    img_refF = field.copy(); img_refF[:xr, :yr] = img_refC
    img_tarF = field.copy(); img_tarF[:xt, :yt] = img_tarC
    cv2.imwrite(imgdest + "/" + name_ref + " <-- " + name_tar + ".jpg", np.hstack([img_refF, img_tarF]))

    print("     Done matching " + name_ref + " to " + name_tar)

def matchMakerN(allrefpt, alltarpt, allsize, alldistance, featNo = None, dist = 50):

    # this takes lists of all the information from the sift feature identification 
    # and bf matching and returns only n number of points which match the criteria
    # Inputs:   (all*), all the information returned by the sift operator after being
    #               brute forced matched
    #           (featNo), the number of features to find on the samples. If None then 
    #               this will return as many features as it can. 
    #               NOTE THIS IS NOT RECOMMENDED because if the initial features found which 
    #               are used to find all the other features aren't good, then this will return
    #               only bad features. It is recommended to use a threshold featNo. A samples 
    #               should be able to return a lot of features with this function IF the 
    #               sample is in good condition and the features being found are useful. 
    #               Setting featNo to None means shit features can be used. 
    #           (dist), sets the minimium distance between each feature. A larger distance 
    #               reduces the number of features that can be found but will also ensure 
    #               that features are  better spread across the sample, rather than being pack 
    #               arond a single strong feature
    # Outputs:  (match*), all the features and corresponding information that have been 
    #               found to be a good match

    def findbestfeatures():

        matchRef = []
        matchTar = []
        matchSize = []
        matchDistance = []

        # find the two best features ensuring that they are located 
        # a reasonable distane away

        # get the best features and remove from the list of sorted features
        matchRef.append(allrefpt[0]); del allrefpt[0]
        matchTar.append(alltarpt[0]); del alltarpt[0]
        matchSize.append(allsize[0]); del allsize[0]
        matchDistance.append(alldistance[0]); del alldistance[0]

        # get the second best feature, ensuring that it is an acceptable distance away from 
        # the best feature (sometimes two really good matches are found pretty much next
        # to each other which isn't useful for fitting)
        for n, (r, t, s, d) in enumerate(zip(allrefpt, alltarpt, allsize, alldistance)):

            # if the distance between the next best feature is less than 100 
            # pixels, don't use it
            if np.sqrt(np.sum(matchRef - r)**2) < d and np.sqrt(np.sum(matchTar - t)**2) < d:
                continue
            
            # if the 2nd best feature found meets the criteria append and move on
            else:
                matchRef.append(r); del allrefpt[n]
                matchTar.append(t); del alltarpt[n]
                matchSize.append(s); del allsize[n]
                matchDistance.append(d); del alldistance[n]
                break

        return(matchRef, matchTar, matchSize, matchDistance)

    def findgoodfeatures():

        # find new features in the ref and target tissue which are positioned 
        # in approximately the same location RELATIVE to the best features found already

        matchRefn = []
        matchTarn = []
        matchSizen = []
        matchDistancen = []

        # apped the best matches to the new matches:
        for r, t, s, d in zip(matchRef, matchTar, matchSize, matchDistance):

            matchRefn.append(r)
            matchTarn.append(t)
            matchSizen.append(s)
            matchDistancen.append(d)

        # from all the remaining features, find the ones that meet the characteristics:
        #   - The new feature is a distance away from all previous features
        #   - The new features on each sample are within a threshold angle and distance
        #   difference relative to the best features found 
        noFeatFind = 0  # keep track of the number of times a match has not been found
        for r, t, s, d in zip(allrefpt, alltarpt, allsize, alldistance):
            
            # if a featNo criteria is set, continue looking until the target number of features is 
            # met (or if there are no new features found for a duration of the search, bottom break)
            if featNo is int:
                if len(matchTarn) >= featNo:
                    break

            # if the difference between the any of the already found feature is less than 100 
            # pixels, don't use it
            repeated = False
            for mr, tr in zip(matchRefn, matchTarn):
                if np.sqrt(np.sum(mr - r)**2) < dist or np.sqrt(np.sum(tr - t)**2) < dist:
                    repeated = True
                    # print('new')
                    break

            if repeated:
                # print('repeated')
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
            for n1, (m1, t1) in enumerate(zip(matchRefn[:5], matchTarn[:5])):
                for n2, (m2, t2) in enumerate(zip(matchRefn[:5], matchTarn[:5])):

                    # if the features are repeated, don't use it
                    if n1 == n2:
                        continue

                    # find the angle for the new point and all the previously found point
                    newrefang = findangle(m1, m2, r)
                    newtarang = findangle(m1, m2, t)

                    # store the difference of this new point relative to all the ponts
                    # previously found
                    angdist.append(abs(newrefang - newtarang))

                # get the distances of the new points to the best feature
                newrefdist = np.sqrt(np.sum((m1 - r)**2))
                newtardist = np.sqrt(np.sum((t1 - t)**2))

                # finds how much larger the largest distance is compared to the smallest distance
                ratiodist.append((newtardist/newrefdist)**(1-((newrefdist>newtardist)*2)) - 1)

            # if the new feature is than 5 degress off and within 5% distance each other 
            # from all the previously found features then append 
            if (np.array(angdist) < 5/180*np.pi).all() and (np.array(ratiodist) < 0.05).all():
            # NOTE using median is a more "gentle" thresholding method. allows more features
            # but the standard of these new features is not as high
            # if np.median(angdist) < 180/180*np.pi and np.median(ratiodist) < 1:
                # add the features
                matchRefn.append(r)
                matchTarn.append(t)
                matchSizen.append(s)
                matchDistancen.append(d)
                noFeatFind = 0
            else:
                # if more than 2% of all the features are investigated and there are no
                # new features found, break the matching process (unlikely to find anymore
                # good features)
                noFeatFind += 1
                if noFeatFind > int(len(alltarpt) * 0.05):
                    break

        return(matchRefn, matchTarn, matchSizen, matchDistancen)

    # sort the features based on distance
    sort = np.argsort(np.array(alldistance))

    # sort the information by ascending distance value
    allrefpt = list(np.array(allrefpt)[sort])
    alltarpt = list(np.array(alltarpt)[sort])
    allsize = list(np.array(allsize)[sort])
    alldistance = list(np.array(alldistance)[sort])

    # append the next n number of best fit features to the matches but 
    # ONLY if their angle from the two reference features is within a tolerance 
    # range --> this heavily assumes that the two best fits found are actually 
    # good features...
    # try up to 5 times
    for fits in range(5):
        # get the two best features 
        matchRef, matchTar, matchSize, matchDistance = findbestfeatures()

        # find features which are spatially coherent relative to the best feature for both 
        # the referenc and target image and with the other constrains
        matchRefn, matchTarn, matchSizen, matchDistancen = findgoodfeatures()

        # If no feature number is used, just find as many features as possible with the 
        # "best" features. NOTE if the best features weren't good then all the subsequent features 
        # found won't be good. It is not recommended to use None for featNo
        if featNo is None:
            if len(matchTarn) >= 10:
                # there is an expectaion that None will find lots of features... so set the bar kind of high
                return(matchRefn, matchTarn, matchSizen, matchDistancen)
                
        # if the required number of features was found then break 
        elif len(matchTarn) >= featNo:
            return(matchRefn, matchTarn, matchSizen, matchDistancen)

        # if the number of features found doesn't meet the conditions, then perform 
        # another fitting procedure (up until 5 iteration) but save the best found features 
        # so far to be returned at the end
        if fits == 0: pass
        elif (len(matchTarn) > len(info['tar'])): pass
        else: continue

        info = {
            "ref": matchRefn, 
            "tar": matchTarn, 
            "size": matchSizen, 
            "dist": matchDistancen
        }

        # if insufficient new features were not found then find new "best" features and 
        # begin the process again. NOTE that the chance of actually finding more features
        # is unlikely if these "best" features were actually the best, but there is a chance
        # that the "best" features were not a good match so this process allows the user to 
        # to 
            
        # shows what features were found relative to each other
        # denseMatrixViewer([matchRefn, matchTarn], True)

    # if multiple rounds of fitting occured, then return the best features found
    return(info['ref'], info['tar'], info['size'], info['dist'])


# ------------ HARD CODED SPECIMEN SPECIFIC FEATURES ------------

def imgPlacement(name_spec, img_refO, img_tarO):

    # this function takes the name of the specimen (target just because...) and 
    # performs hardcoded placements of the images within the field. This is okay because
    # each sample has its own processing quirks and it's definitely easier to do 
    # it like this than work it out properly
    # Inputs:   (name_tar), name of the specimen
    #           (img_refO, img_tarO), images to place
    # Outputs:  (x/y ref/tar Dif), the shifts used to place the images
    #           (img_ref, img_tar), adjusted images

    # get the image dimensions, NOTE this is done in the main function but I 
    # didn't want to feed all those variables into this function... seems very messy
    xr, yr, cr = img_refO.shape
    xt, yt, ct = img_tarO.shape
    xm, ym, cm = np.max(np.array([(xr, yr, cr), (xt, yt, ct)]), axis = 0)
    
    # create a max size field of both images
    field = np.zeros((xm, ym, cm)).astype(np.uint8)

    # something for the bottom right
    if name_spec == 'H653A' or name_spec == 'H710B':
        # these are the origin shifts to adapt each image
        xrefDif = xm-xr
        yrefDif = ym-yr
        xtarDif = xm-xt
        ytarDif = ym-yt

        # re-assign the images to the left of the image (NOTE this is for H563A which has
        # been segmented and the samples are very commonly best aligned on the left side)
        img_ref = field.copy(); img_ref[-xr:, -yr:] = img_refO
        img_tar = field.copy(); img_tar[-xt:, -yt:, :] = img_tarO
    
    # specific H1029A positioning
    elif name_spec == 'H1029A':
        # position the further right, lowest point of the each of the target and 
        # reference images at the bottom right positions of the fields
        pos = np.where(img_tarO != 0)
        xmaxt = np.max(pos[1])
        ymaxt = pos[0][np.where(pos[1] == xmaxt)[0]][-1]

        pos = np.where(img_refO != 0)
        xmaxr = np.max(pos[1])
        ymaxr = pos[0][np.where(pos[1] == xmaxr)[0]][-1]

        img_tarp = img_tarO[:ymaxt, :xmaxt]
        img_refp = img_refO[:ymaxr, :xmaxr]

        xrp, yrp, c = img_refp.shape
        xtp, ytp, c = img_tarp.shape

        xm, ym, cm = np.max(np.array([(xrp, yrp, c), (xtp, ytp, c)]), axis = 0)
        fieldp = np.zeros((xm, ym, cm)).astype(np.uint8)

        xrefDif = xm-xrp
        yrefDif = ym-yrp
        xtarDif = xm-xtp
        ytarDif = ym-ytp

        img_ref = fieldp.copy(); img_ref[-xrp:, -yrp:, :] = img_refp
        img_tar = fieldp.copy(); img_tar[-xtp:, -ytp:, :] = img_tarp
        
    elif name_spec == 'H710C':
        # put the image in the middle of the field
        xrefDif = int((xm-xr) / 2)
        yrefDif = int((ym-yr) / 2)
        xtarDif = int((xm-xt) / 2)
        ytarDif = int((ym-yt) / 2)

        img_ref = field.copy(); img_ref[xrefDif:xrefDif+xr, yrefDif:yrefDif+yr, :] = img_refO
        img_tar = field.copy(); img_tar[xtarDif:xtarDif+xt, ytarDif:ytarDif+yt, :] = img_tarO


    # if not specifically hardcoded, just place in the top left
    else:
        xrefDif = 0
        yrefDif = 0
        xtarDif = 0
        ytarDif = 0

        img_ref = field.copy(); img_ref[:xr, :yr, :] = img_refO
        img_tar = field.copy(); img_tar[:xt, :yt, :] = img_tarO
        
    return(xrefDif, yrefDif, xtarDif, ytarDif, img_ref, img_tar)

if __name__ == "__main__":

    dataSource = '/Volumes/USB/Testing1/'
    dataSource = '/Volumes/USB/H653A_11.3/'
    dataSource = '/Volumes/USB/H671B_18.5/'
    dataSource = '/Volumes/USB/H673A_7.6/'
    dataSource = '/Volumes/Storage/H653A_11.3new/'
    dataSource = '/Volumes/USB/H710C_6.1/'

    

    name = ''
    size = 3

    featFind(dataSource, name, size)
