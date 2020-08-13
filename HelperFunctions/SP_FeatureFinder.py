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
    from SP_SampleFinder import featSelect
else:
    from HelperFunctions.Utilities import listToTxt, dictToTxt, nameFromPath, dirMaker, dictToArray
    from HelperFunctions.SP_SampleFinder import featSelect


'''
TO DO:
    - Find a method to pre-process the image to further enhance the sift operator
        UPDATE I think it is unlikely I will find a way to radically improve SIFT. It is 
        easier to just allow it to do the best it can and add manual points on sections which 
        don't match

    - Create function so if there are less than x points, create GUI which will allow manual 
    ID of features

    - Apply the sift operators over the target image ONCE and then organise these 
    into the appropriate grids for searching

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

    # get the directories which are masked
    specimens = []
    for i in os.listdir(imgsrc):
        if (i.find("masked")>0): 
            specimens.append(i) 

    # for parallelisation
    jobs = {}

    findFeats(imgsrc, infodest, imgdest, '')

    '''
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

def findFeats(dataSource, dataDest, imgdest, spec):

    # This script finds features between two sequential samples (based on their
    # name) that correspond to biologically the same location. 
    # NOTE this is done on lower res jpg images to speed up the process as this is 
    # based on brute force methods. IT REQUIRES ADAPTING THE FEAT AND BOUND POSITIONS
    # TO THE ORIGINAL SIZE TIF FILES --> info is stored as args in the .feat and .bound files
    # It is heavily based on the cv2.SIFT function
    # Inputs:   (dataSource): source of the pre-processed images (from SP_SpecimenID)
    #           (dataDest): the location to save the txt files
    #           (imgdest): location to save the images which show the matching process
    #           (spec): the specific specimen to process
    # Outputs:  (): .feat files for each specimen which correspond to the neighbouring
    #               two slices (one as the reference and one as the target)
    #               .bound files which are the top/bottom/left/right positions with the image
    #               jpg images which show where the features were found bewteen slices

    dirMaker(imgdest)

    # get the masked images
    imgs = sorted(glob(dataSource + spec + "/*.jpg"))[0:2]

    # counting the number of features found
    noFeat = 0
    matchRefDict = {}       # the ref dictionary is re-allocated... a little difference from target
    bf = cv2.BFMatcher()
    for n in range(len(imgs)-1):

        # initialise the target dictionary
        matchTarDict = {}

        name_ref = nameFromPath(imgs[n])
        name_tar = nameFromPath(imgs[n+1])

        print("Matching " + name_tar + " to " + name_ref)

        # load in the images
        img_refO = cv2.imread(imgs[n])
        img_tarO = cv2.imread(imgs[n+1])

        # make the images gray scale
        # this doesn't make much of a difference..... mostly the same features are found as in 
        # the colour image but often less are found.... For some reason there are some samples 
        # where there are features found when there weren't any before..... 
        # img_refO = np.expand_dims(cv2.cvtColor(img_refO, cv2.COLOR_BGR2GRAY), -1)
        # img_tarO = np.expand_dims(cv2.cvtColor(img_tarO, cv2.COLOR_BGR2GRAY), -1)
        

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
        dictToTxt(boundRef, dataDest + spec + "/" + name_ref + ".bound", shape = str(img_refO.shape))

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
                
        # these are the origin shifts to adapt each image
        '''
        xrefDif = int((xm-xr)/2)
        yrefDif = int((ym-yr)/2)
        xtarDif = int((xm-xt)/2)
        ytarDif = int((ym-yt)/2)

        # re-assign the images into the middle of this standard field
        img_ref = field.copy(); img_ref[xrefDif: xrefDif + xr, yrefDif: yrefDif + yr, :] = img_refO
        img_tar = field.copy(); img_tar[xtarDif: xtarDif + xt, ytarDif: ytarDif + yt, :] = img_tarO
        '''
        
        xrefDif = 0
        yrefDif = 0
        xtarDif = 0
        ytarDif = 0

        # re-assign the images to the left of the image (NOTE this is for H563A which has
        # been segmented and the samples are very commonly best aligned on the left side)
        img_ref = field.copy(); img_ref[:xr, :yr] = img_refO
        img_tar = field.copy(); img_tar[:xt, :yt, :] = img_tarO
        
        '''
        xrefDif = xm-xr
        yrefDif = ym-yr
        xtarDif = xm-xt
        ytarDif = ym-yt

        # re-assign the images to the left of the image (NOTE this is for H563A which has
        # been segmented and the samples are very commonly best aligned on the left side)
        img_ref = field.copy(); img_ref[-xr:, -yr:] = img_refO
        img_tar = field.copy(); img_tar[-xt:, -yt:, :] = img_tarO
        '''

        # normalise for all the colour channels
        # fig, (bx1, bx2, bx3) = plt.subplots(1, 3)
        for c in range(img_tar.shape[2]):
            img_tar[:, :, c] = hist_match(img_tar[:, :, c], img_ref[:, :, c])

            # the median value EXCLUDING the 0's
            # medTar = np.median(img_tar[(np.where(img_tar[:, :, c] > 0))[0], (np.where(img_tar[:, :, c] > 0))[1], c])
            # medRef = np.median(img_ref[(np.where(img_ref[:, :, c] > 0))[0], (np.where(img_ref[:, :, c] > 0))[1], c])

            # make the colour range extreme
            # img_tar[:, :, c] = np.clip((img_tar[:, :, c] - medTar) / (np.max(img_tar[:, :, c]) - medTar) * 255, 0, 255)
            # img_ref[:, :, c] = np.clip((img_ref[:, :, c] - medRef) / (np.max(img_ref[:, :, c]) - medRef) * 255, 0, 255)

        # emphasise edges --> NOTE this doesn't work.... investigate how SIFT
        # works and how to pre-process images to enhance its effectiveness
        
        '''
        edgestar = cv2.Canny(img_tar, 100, 200)
        edgesref = cv2.Canny(img_tar, 100, 200)

        img_tar = np.clip(img_tar * np.expand_dims(((edgestar/255 - 1) * -1).astype(np.uint8), -1), 0, 255)
        img_ref = np.clip(img_ref * np.expand_dims(((edgesref/255 - 1) * -1).astype(np.uint8), -1), 0, 255)
        '''

        '''
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title("target")
        ax1.imshow(img_tar)
        ax2.set_title("reference")
        ax2.imshow(img_ref)
        plt.show()
        '''

        # Initiate SIFT detector
        # NOTE this required the contrib module --> research use only
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors of the whole image with SIFT
        # kp_ref, des_ref = sift.detectAndCompute(img_ref,None)
        # kp_tar, des_tar = sift.detectAndCompute(img_tar,None)
        
        x, y, c = img_ref.shape
        p = 100     # pixel grid size
        sc = 0.5    # the extra 1D length size of the target section

        matchInfo = []
        matchRef = []
        matchTar = []

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

                # NOTE is there room for further IP before being analysed???
                '''
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                ax1.imshow(imgSect_ref)
                ax2.imshow(imgSect_tar)

                # edge detection
                imgSect_ref = cv2.Canny(imgSect_ref, 100, 200)
                imgSect_tar = cv2.Canny(imgSect_tar, 100, 200)
                ax3.imshow(imgSect_ref, cmap = 'gray')
                ax4.imshow(imgSect_tar, cmap = 'gray')
                plt.show()
                '''
                
                # get the key points and descriptors of each section
                kp_ref, des_ref = sift.detectAndCompute(imgSect_ref,None)
                kp_tar, des_tar = sift.detectAndCompute(imgSect_tar,None)

                '''
                imgSect_refKEYS = cv2.drawKeypoints(imgSect_ref,kp_ref,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                imgSect_tarKEYS = cv2.drawKeypoints(imgSect_tar,kp_tar,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                fig, (ax1, ax2) = plt.subplots(1, 2)
                ax1.imshow(imgSect_refKEYS)
                ax2.imshow(imgSect_tarKEYS)
                plt.show()
                '''

                # create lists to store section specific match finding info
                kp_keep_ref = []
                des_keep_ref = []
                kp_keep_tar = []
                des_keep_tar = []

                # only further process if there are matches found in both samples
                if (des_ref is not None) and (des_tar is not None):
                    # identify strongly identifiable features in both the target and 
                    # reference tissues
                    for kpi, desi in zip(kp_ref, des_ref):
                        # set a minimum size for the feature match
                        if kpi.size > 10:
                            # extract the position of the found feature and adjust
                            # back to the global size of the original image 
                            kp_keep_ref.append(np.array(kpi.pt) + np.array([r*p, c*p]))

                            # store the descriptor
                            des_keep_ref.append(desi)

                    # only consider points which have a significant size
                    for kpi, desi in zip(kp_tar, des_tar):
                        if kpi.size > 10:
                            # NOTE if the range of search for targets is larger then the adjust needs to match as well
                            kp_keep_tar.append(np.array(kpi.pt) + np.array([int((r-sc)*p), int((c-sc)*p)])) 
                            des_keep_tar.append(desi)

                    # if there are key points found, bf match
                    if len(des_keep_ref) * len(des_keep_tar) > 0:
                        des_keep_ref = np.array(des_keep_ref)
                        des_keep_tar = np.array(des_keep_tar)
                        matches = bf.match(des_keep_ref, des_keep_tar)

                        m_info = {}
                        m_info['distance'] = []
                        m_info['ref'] = []
                        m_info['tar'] = []
                    
                        # if a match is found, get the pair of points
                        for m in matches:
                            m_info['distance'].append(m.distance)
                            m_info['ref'].append(kp_keep_ref[m.queryIdx])
                            m_info['tar'].append(kp_keep_tar[m.trainIdx])
                        
                        # only confirm points which have a good match
                        bestMatch = np.argmin(np.array(m_info['distance']))
                        # NOTE this match value is chosen based on observations.... 
                        # lower scores mean the matches are better (which results in fewer
                        # matches found). 
                        if m_info['distance'][bestMatch] < 250:
                            matchInfo.append(m_info['distance'][bestMatch])
                            matchRef.append(m_info['ref'][bestMatch])
                            matchTar.append(m_info['tar'][bestMatch])

        # if there are more than 5 matches, only select the 5 best
        if len(matchInfo) > 20:
            # get the ordered list
            ordered = np.argsort(matchInfo)

            # select only the five best matches
            matchRef = np.array(matchRef)[ordered[:20]]
            matchTar = np.array(matchTar)[ordered[:20]]

        # if there are less than two matches, perform a manual feature identification process
        # NOTE would be ideal if this process was saved until the end....
        if len(matchTar) <= 5:
            matchRef, matchTar = featSelect(img_ref, img_tar, matchRef, matchTar)

        # add annotations to where the matches have been found
        newFeats = []
        for kr, kt in zip(matchRef, matchTar):
            
            featRef = tuple(kr.astype(int))

            # draw a circle around the common feature
            cv2.circle(img_ref, featRef, 20, (255, 0, 0), 8)

            # add the feature number onto the image
            cv2.putText(img_ref, str(noFeat), 
            tuple(featRef + np.array([20, 0])),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
            cv2.putText(img_ref, str(noFeat), 
            tuple(featRef + np.array([20, 0])),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

            # add matched feature, adjust for the initial standardisation of the image
            matchRefDict["feat_" + str(noFeat)] = kr.astype(int) - np.array([yrefDif, xrefDif])

            # do the same thing for the target image
            featTar = tuple(kt.astype(int))

            cv2.circle(img_tar, featTar, 20, (255, 0, 0), 8)

            cv2.putText(img_tar, str(noFeat), 
            tuple(featTar + np.array([20, 0])),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 6)
            cv2.putText(img_tar, str(noFeat), 
            tuple(featTar + np.array([20, 0])),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            matchTarDict["feat_" + str(noFeat)] = kt.astype(int) - np.array([ytarDif, xtarDif])

            newFeats.append("feat_" + str(noFeat))
            noFeat += 1     # continuously iterate through feature numbers

        # store the positions of the identified features for each image as 
        # BOTH a reference and target image. Include the image size this was 
        # processed at
        dictToTxt(matchRefDict, dataDest + spec + "/" + name_ref + ".feat", shape = str(img_refO.shape))

        # print a combined image showing the matches
        cv2.imwrite(imgdest + spec + "/" + name_ref + " <-- " + name_tar + ".jpg", np.hstack([img_ref, img_tar]))
    
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

        # add in the features
        for p in newFeats:
            cv2.circle(img_refO, tuple(matchRefDict[p]), 20, (255, 0, 0), 8)
            cv2.circle(img_tarO, tuple(matchTarDict[p]), 20, (255, 0, 0), 8)

        # draw the centre of the features found
        cv2.circle(img_tarO, tuple(np.mean(dictToArray(matchTarDict), axis = 0).astype(int)), 20, (0, 255, 0), 8)
        cv2.imwrite(imgdest + spec + "/" + name_ref + "_ref.jpg", img_refO)
        cv2.imwrite(imgdest + spec + "/" + name_tar + "_tar.jpg", img_tarO)

        # re-assign the target dictionary now as the ref dictioary
        matchRefDict = matchTarDict

    # at the very end, print the target features found
    dictToTxt(matchTarDict, dataDest + spec + "/" + name_tar + ".feat", shape = str(img_tarO.shape))

    boundTar = {}
    pos = np.vstack(np.where(img_tarO[:, :, 0] != 0))
    top, left = np.argmin(pos, axis = 1)
    bottom, right = np.argmax(pos, axis = 1)
    boundTar['top'] = pos[:, top]
    boundTar['bottom'] = pos[:, bottom]
    boundTar['left'] = pos[:, left]
    boundTar['right'] = pos[:, right]
    # store the boundary of the image based on the mask
    dictToTxt(boundTar, dataDest + spec + "/" + name_tar + ".bound", shape = str(img_tarO.shape))

if __name__ == "__main__":

    dataSource = '/Volumes/USB/Testing1/'
    # dataSource = '/Volumes/USB/IndividualImages/'
    dataSource = '/Volumes/USB/H653/'
    

    name = ''
    size = 3

    featFind(dataSource, name, size)
