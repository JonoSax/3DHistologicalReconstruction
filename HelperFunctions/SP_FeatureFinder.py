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
from multiprocessing import Process
if __name__ == "__main__":
    from Utilities import listToTxt, dictToTxt, nameFromPath, dirMaker, dictToArray
    from SP_SampleFinder import featSelectPoint
else:
    from HelperFunctions.Utilities import listToTxt, dictToTxt, nameFromPath, dirMaker, dictToArray
    from HelperFunctions.SP_SampleFinder import featSelectPoint

'''

Parameters:

dist, the error between descriptors. The lower the number the more similar the 
gradient descriptors for the sift operations between images are

sz, the size of the feature found by the sift operator. The larger the size the 
more prominent the feature is in the image

no, the number of tiles to make along the horizontal axis (NOTE this uses square tiles
so will be a different number vertically)

featNo, number of features that are found per image. NOTE more features do not necessarily 
mean better fitting as the optimising algorithms may find weird local minimas rather than 
a global minima with fewer features. If the automatic feature detection doesn't find 
enough features then a GUI will allow the user to select features until the featNo
is satisfied.


'''

'''
TO DO:
    - Find a method to pre-process the image to further enhance the sift operator
        UPDATE I think it is unlikely I will find a way to radically improve SIFT. It is 
        easier to just allow it to do the best it can and add manual points on sections which 
        don't match

    - Apply the sift operators over the target image ONCE and then organise these 
    into the appropriate grids for searching --> NOTE this is only useful when it is 
    on a reference image which I have tried and it doesn't work very well.

    - Probably create a function before this which extracts the images out of their 
    bounded areas first, then process (would also allow for iterative sample extraction)
        - after doing 

    - EXTRA FOR EXPERTS: conver this into C++ for superior speed
'''


def featFind(dataHome, name, size):
    
    # this is the function called by main. Organises the inputs for findFeats

    # get the size specific source of information
    datasrc = dataHome + str(size) + "/"
    # datasrc = '/Volumes/USB/'

    # gets the images for processing
    imgsrc = datasrc + "masked/"
    # imgsrc = datasrc + "IndividualImages/"

    # specify where the outputs are saved
    infodest = datasrc + "info/"
    imgdest = datasrc + "matched/"

    findFeats(imgsrc, infodest, imgdest, dist = 250, sz = 8, gridNo = 15, featNo = 5)

    '''
    # for parallelisation
    jobs = {}
    for spec in specimens:
        findFeats(imgsrc, infodest, imgdest, spec)
        # NOTE some of the sample don't have many therefore shouldn't process
        # jobs[spec] = Process(target=findFeats, args = (dataSource, spec))
        # jobs[spec].start()
    '''
    '''
    for spec in specimens:
        jobs[spec].join()
    '''

def hist_match(source, template):
    """
    Courtesy of https://stackoverflow.com/questions/31490167/how-can-i-transform-the-histograms-of-grayscale-images-to-enforce-a-particular-r/31493356#31493356
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    # NOTE this is done here rather than in SpecimenID because it only works well
    # when the sample is very well identified

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # remove the effect of black (it is working on a masked image)
    s_counts[0] = 0
    t_counts[0] = 0

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def findFeats(dataSource, dataDest, imgdest, dist = 250, sz = 10, gridNo = 10, featNo = 5):

    # This script finds features between two sequential samples (based on their
    # name) that correspond to biologically the same location. 
    # NOTE this is done on lower res jpg images to speed up the process as this is 
    # based on brute force methods. IT REQUIRES ADAPTING THE FEAT AND BOUND POSITIONS
    # TO THE ORIGINAL SIZE TIF FILES --> info is stored as args in the .feat and .bound files
    # It is heavily based on the cv2.SIFT function
    # Inputs:   (dataSource): source of the pre-processed images (from SP_SpecimenID)
    #           (dataDest): the location to save the txt files
    #           (dist): the error between the match of features
    #           (imgdest): location to save the images which show the matching process
    #           (sz): size of the sift feature to use in processing
    #           (gridNo): number of grids (along horizontal axis) to use to analyse images
    #           (featNo): number of features to apply per image
    # Outputs:  (): .feat files for each specimen which correspond to the neighbouring
    #               two slices (one as the reference and one as the target)
    #               .bound files which are the top/bottom/left/right positions with the image
    #               jpg images which show where the features were found bewteen slices

    dirMaker(imgdest)
    dirMaker(dataDest)

    # get the images
    imgs = sorted(glob(dataSource + "/*"))

    # counting the number of features found
    noFeat = 0
    matchRefDict = {}       # the ref dictionary is re-allocated... a little difference from target
    bf = cv2.BFMatcher()
    for n in range(len(imgs)-1):

        # initialise the target dictionary
        matchTarDict = {}

        name_ref = nameFromPath(imgs[n], 3)
        name_tar = nameFromPath(imgs[n+1], 3)

        print("Matching " + name_tar + " to " + name_ref)

        # load in the images
        img_refO = cv2.imread(imgs[n])
        img_tarO = cv2.imread(imgs[n+1])

        # find the boundary of the selected image (find the top/bottom/left/right 
        # most points which bound the image)
        boundRef = {}
        pos = np.vstack(np.where(img_refO[:, :, 0] != 0))
        top, left = np.argmin(pos, axis = 1)
        bottom, right = np.argmax(pos, axis = 1)
        boundRef['top'] = np.flip(pos[:, top])
        boundRef['bottom'] = np.flip(pos[:, bottom])
        boundRef['left'] = np.flip(pos[:, left])
        boundRef['right'] = np.flip(pos[:, right])
        # store the boundary of the image based on the mask
        dictToTxt(boundRef, dataDest + "/" + name_ref + ".bound", shape = str(img_refO.shape))

        boundTar = {}
        pos = np.vstack(np.where(img_tarO[:, :, 0] != 0))
        top, left = np.argmin(pos, axis = 1)
        bottom, right = np.argmax(pos, axis = 1)
        boundTar['top'] = np.flip(pos[:, top])
        boundTar['bottom'] = np.flip(pos[:, bottom])
        boundTar['left'] = np.flip(pos[:, left])
        boundTar['right'] = np.flip(pos[:, right])

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

        # Initiate SIFT detector
        # NOTE this required the contrib module --> research use only
        sift = cv2.xfeatures2d.SIFT_create()
        
        x, y, c = img_ref.shape
        p = int(np.round(y/gridNo, -1))     # pixel grid size
        sc = 0.2    # the extra 1D length size of the target section

        matchDistance = []
        matchRef = []
        matchTar = []
        matchSize = []

        # iterate through a pixel grid of p ** 2 x c size
        # NOTE the target section is (p + 2sc) ** 2 x c in size --> idea is that the
            # target secition will have some significant shift therefore should look in
            # a larger area
        # The reason why a scanning method over the images is implemented, rather than
        # just letting sift work across the full images, is because the nature of the 
        # samples (biological tissue) means there are many repeating structures which has
        # lead to feature matching in non-sensible locations. This scanning method assumes
        # there is APPROXIMATE sample placement (ie there are samples in the middle of the 
        # slice) --> ATM this is taking the whole slide with a mask on, however there is no
        # reason why the bounding method cannot be applied before this step to extract the 
        # images from the slide. This would make the likelihood of the central placement on
        # the sample more likely + reduce computation given the more precise area created
        for c in range(1, int(np.ceil(x/p)) - 1):
            for r in range(1, int(np.ceil(y/p)) - 1):

                # extract a small grid from both image
                imgSect_ref = img_ref[c*p:(c+1)*p, r*p:(r+1)*p, :]

                # NOTE for the target, the sift dectection should be performed ONCE
                # and then this selection processes occurs over the calculated sift points
                imgSect_tar = img_tar[int((c-sc)*p):int((c+1+sc)*p), int((r-sc)*p):int((r+1+sc)*p), :]  # NOTE target area search is expaneded

                # if the entire contains very little info (ie less than 1/3 of the image contains
                # target tissue) don't process
                if (np.sum((imgSect_ref>0)*1) <= imgSect_ref.size*0.9): #or (np.sum((imgSect_tar>0)*1) <= imgSect_tar.size):
                    continue
                # plt.imshow(imgSect_ref); plt.show()
                # get the key points and descriptors of each section
                kp_ref, des_ref = sift.detectAndCompute(imgSect_ref,None)
                kp_tar, des_tar = sift.detectAndCompute(imgSect_tar,None)

                # create lists to store section specific match finding info
                kp_keep_ref = []
                des_keep_ref = []
                kp_keep_tar = []
                des_keep_tar = []
                size_keep_tar = []

                # only further process if there are matches found in both samples
                if (des_ref is not None) and (des_tar is not None):
                    # identify strongly identifiable features in both the target and 
                    # reference tissues
                    for kpi, desi in zip(kp_ref, des_ref):
                        # set a minimum size for the feature match
                        if kpi.size > sz:
                            # extract the position of the found feature and adjust
                            # back to the global size of the original image 
                            kp_keep_ref.append(np.array(kpi.pt) + np.array([r*p, c*p]))

                            # store the descriptor
                            des_keep_ref.append(desi)

                    # only consider points which have a significant size
                    for kpi, desi in zip(kp_tar, des_tar):
                        if kpi.size > sz:
                            # NOTE if the range of search for targets is larger then the adjust needs to match as well
                            kp_keep_tar.append(np.array(kpi.pt) + np.array([int((r-sc)*p), int((c-sc)*p)])) 
                            des_keep_tar.append(desi)
                            size_keep_tar.append(kpi.size)

                    # if there are key points found, bf match
                    if len(des_keep_ref) * len(des_keep_tar) > 0:
                        des_keep_ref = np.array(des_keep_ref)
                        des_keep_tar = np.array(des_keep_tar)
                        matches = bf.match(des_keep_ref, des_keep_tar)

                        m_info = {}
                        m_info['distance'] = []
                        m_info['ref'] = []
                        m_info['tar'] = []
                        m_info['size'] = []
                    
                        # if a match is found, get the pair of points
                        for m in matches:
                            m_info['distance'].append(m.distance)
                            m_info['ref'].append(kp_keep_ref[m.queryIdx])
                            m_info['tar'].append(kp_keep_tar[m.trainIdx])
                            m_info['size'].append(size_keep_tar[m.trainIdx])
                        
                        # only confirm points which have a good match
                        bestMatch = np.argmin(np.array(m_info['distance']))
                        # NOTE this match value is chosen based on observations.... 
                        # lower scores mean the matches are better (which results in fewer
                        # matches found). 
                        if m_info['distance'][bestMatch] < dist:
                            matchDistance.append(m_info['distance'][bestMatch])
                            matchRef.append(m_info['ref'][bestMatch])
                            matchTar.append(m_info['tar'][bestMatch])
                            matchSize.append(m_info['size'][bestMatch])


        if len(matchTar) < featNo:
            matchRef, matchTar = featSelectPoint(img_ref, img_tar, matchRef, matchTar, featNo)

        # if there are more than 5 matches then pick the 5 most appropriate matches
        else:
            bestMatches = matchMaker(matchTar, matchDistance, featNo)
            
            # select only the five best matches
            matchRef = np.array(matchRef)[bestMatches]
            matchTar = np.array(matchTar)[bestMatches]
            matchDistance = np.array(matchDistance)[bestMatches]
            matchSize = np.array(matchSize)[bestMatches]

        # ---------- create a combined image of the target and reference image matches ---------

        # update the dictionaries
        newFeats = []
        ts = 1
        for kr, kt in zip(matchRef, matchTar):
            
            # get the features in the correct formate
            featRef = tuple(kr.astype(int))
            featTar = tuple(kt.astype(int))

            # add matched feature, adjust for the initial standardisation of the image
            matchRefDict["feat_" + str(noFeat)] = kr.astype(int) - np.array([yrefDif, xrefDif])
            matchTarDict["feat_" + str(noFeat)] = kt.astype(int) - np.array([ytarDif, xtarDif])

            newFeats.append("feat_" + str(noFeat))
            noFeat += 1     # continuously iterate through feature numbers


        # add in the features
        for i, n in enumerate(newFeats):

            # if there is no match info just assign it to 0 (ie was a manual annotaiton)
            try: md = matchDistance[i]; ms = matchSize[i]
            except: md = 0; ms = 0

            # mark the feature
            newref = matchRefDict[n]
            tar = matchTarDict[n]

            cv2.circle(img_refO, tuple(newref), 20, (255, 0, 0), 8)
            cv2.circle(img_tarO, tuple(tar), 20, (255, 0, 0), 8)

            # add the feature number onto the image
            cv2.putText(img_refO, str(n), 
            tuple(newref + np.array([-50, 50])),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 15)
            cv2.putText(img_refO, str(n), 
            tuple(newref + np.array([-50, 50])),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
            
            text = str(n + ", d: " + str(int(md)) + ", s: " + str(np.round(ms, 2)))

            cv2.putText(img_tarO, text,
            tuple(tar + np.array([-200, 50])),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 15)
            cv2.putText(img_tarO, text, 
            tuple(tar + np.array([-200, 50])),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)

        # store the positions of the identified features for each image as 
        # BOTH a reference and target image. Include the image size this was 
        # processed at
        dictToTxt(matchRefDict, dataDest + "/" + name_ref + ".feat", shape = str(img_refO.shape))

        # draw the grid lines on the ref image
        for r in range(0, y, p):
            # horizontal line
            cv2.line(img_refO, (r, 0), (r, x), (255, 255, 255), 4, 1)
            cv2.line(img_refO, (r, 0), (r, x), (0, 0, 0), 2, 1)
        
        for c in range(0, x, p):
            # vertical line
            cv2.line(img_refO, (0, c), (y, c), (255, 255, 255), 4, 1)
            cv2.line(img_refO, (0, c), (y, c), (0, 0, 0), 2, 1)

        # print a combined image showing the matches
        img_refF = field.copy(); img_refF[:xr, :yr] = img_refO
        img_tarF = field.copy(); img_tarF[:xt, :yt] = img_tarO
        cv2.imwrite(imgdest + "/" + name_ref + " <-- " + name_tar + ".jpg", np.hstack([img_refF, img_tarF]))
    
        # ---------------- write the individual reference and target images ----------

        # add in the boundaries
        for p in boundRef:
            cv2.rectangle(img_refO, tuple(boundRef[p] - 20 ), tuple(boundRef[p] + 20 ), (0, 255, 0), 50)
            cv2.putText(img_refO, str(p), 
            tuple(boundRef[p] + np.array([20, 20])),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            
            cv2.rectangle(img_tarO, tuple(boundTar[p] - 20), tuple(boundTar[p] - 20), (0, 255, 0), 50)
            cv2.putText(img_tarO, str(p), 
            tuple(boundTar[p] + np.array([20, 20])),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # add in the NEW features
        for p in newFeats:
            cv2.circle(img_refO, tuple(matchRefDict[p]), 20, (255, 0, 0), 8)
            cv2.circle(img_tarO, tuple(matchTarDict[p]), 20, (255, 0, 0), 8)

        # draw the centre of the features found
        cv2.circle(img_tarO, tuple(np.mean(dictToArray(matchTarDict), axis = 0).astype(int)), 20, (0, 255, 0), 8)
        cv2.imwrite(imgdest + "/" + name_ref + "_ref.jpg", img_refO)
        cv2.imwrite(imgdest + "/" + name_tar + "_tar.jpg", img_tarO)

        # re-assign the target dictionary now as the ref dictioary
        matchRefDict = matchTarDict

    # at the very end, print the targ4et features found
    dictToTxt(matchTarDict, dataDest + "/" + name_tar + ".feat", shape = str(img_tarO.shape))

    boundTar = {}
    pos = np.vstack(np.where(img_tarO[:, :, 0] != 0))
    top, left = np.argmin(pos, axis = 1)
    bottom, right = np.argmax(pos, axis = 1)
    boundTar['top'] = pos[:, top]
    boundTar['bottom'] = pos[:, bottom]
    boundTar['left'] = pos[:, left]
    boundTar['right'] = pos[:, right]
    # store the boundary of the image based on the mask
    dictToTxt(boundTar, dataDest + "/" + name_tar + ".bound", shape = str(img_tarO.shape))

def matchMaker(matchTar, matchDistance, n = 5):

    # this function takes all the matches which meet the criteria and chooses only the 
    # best matches for the fitting procedures of AlignSamples
    # Inputs:   (matchTar), the list of positions on the target image which have been matched
    #           (matchDistance), the error of the descriptors for each match
    #           (n), number of samples to be selected as the best samples to choose from, defulats as 5
    # Outputs:  (bestMatches), returns the positions of the best matches in all the lists

    # create a copy so that I don't much up the original array
    matchTarSort = matchTar.copy()

    # there features should be found (best match, 2 centres) and if more features are to be
    # found then it is on top of this
    extra = n - 4

    # create a list of the best match positions
    bestMatches = list()

    # get the ordered list
    ordered = np.argsort(matchDistance)

    # pick the two best features 
    bestMatches.append(ordered[0])
    bestMatches.append(ordered[1])

    # pick the features (2) which are in the middle vertically and horizontally
    # ensure that there is an odd number length of the array so that the median can be found
    if len(matchTar) % 2 == 0:
        matchTarM = np.vstack([matchTarSort, np.array([0, 0])])
    else: 
        matchTarM = matchTarSort

    for i in range(extra):
        # get the position of the median on either the x and y axis
        p = np.where(np.array(matchTarM)[:, i] == np.median(matchTarM, axis = 0)[i])[0][0]

        # if the match is the same as the previously added one, don't add it again but 
        # note that an extra match will need to be found
        if len(np.where(bestMatches == p)[0]) == 0: bestMatches.append(p)

    middle = []
    for p in bestMatches:
        # get the positions of the points found so far
        middle.append(np.array(matchTarM)[p, :])
   
    # from all the points so far, find the middle
    middle = np.mean(middle, axis = 0)

    # re-assign the best matched position to be a middle position. this is done instead of 
    # deleting so that the positional arguments are not mucked around
    # matchTarSort[ordered[0]] = middle

    # pick the points furtherest away from the middle
    err = []
    for i in matchTarSort:
        # get the error between points
        err.append(np.sum((i - middle)**2))

    # get the positions which have the largest error and therefore are the furthest away 
    # from the middle
    errSort = np.argsort(err)
    i = 0
    while len(bestMatches) < n:
        # get the point which is of interest
        p = int(errSort[-(i + 1)])

        # if point has not been used then add it to the bestmatches
        if len(np.where(np.array(bestMatches) == p)[0]) == 0:
            bestMatches.append(int(errSort[-(i + 1)]))

        i += 1

    return(bestMatches)

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
    if name_spec == 'bottom right?':
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
    if name_spec == 'H1029A':
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
        
    elif name_spec == 'H653A':
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
    # dataSource = '/Volumes/USB/IndividualImages/'
    dataSource = '/Volumes/USB/H653/'
    # dataSource = '/Volumes/USB/H1029a/'
    dataSource = '/Volumes/Storage/H653A_11.3new/'
    

    name = ''
    size = 3

    featFind(dataSource, name, size)
