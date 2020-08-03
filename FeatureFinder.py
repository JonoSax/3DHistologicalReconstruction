'''

This script finds features between different layers of tissue

'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
import os

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

dataSource = '/Volumes/USB/InvididualImagesMod2/'

specimens = os.listdir(dataSource)

for spec in specimens:  

    imgs = sorted(glob(dataSource + spec + "/*"))

    for n in range(len(imgs)-1):

        img_ref = cv2.imread(imgs[n])
        img_tar = cv2.imread(imgs[n+1])

        # normalise for all the colour channels
        # fig, (bx1, bx2, bx3) = plt.subplots(1, 3)
        for c in range(3):
            img_tar[:, :, c] = hist_match(img_tar[:, :, c], img_ref[:, :, c])

        # Initiate SIFT detector
        # NOTE this required the contrib module --> research use only
        sift = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        # kp_ref, des_ref = sift.detectAndCompute(img_ref,None)
        # kp_tar, des_tar = sift.detectAndCompute(img_tar,None)
        
        y, x, c = img_ref.shape
        p = 100     # pixel grid size
        bf = cv2.BFMatcher()

        matchRef = []
        matchTar = []

        for c in range(1, int(np.ceil(y/p)) - 1):
            for r in range(1, int(np.ceil(x/p)) - 1):
                # extract a small grid from both image
                imgSect_ref = img_ref[c*p:(c+1)*p, r*p:(r+1)*p, :]

                # NOTE would be good to expand the target search
                imgSect_tar = img_tar[(c-1)*p:(c+2)*p, (r-1)*p:(r+2)*p, :]

                # if the entire contains very little info (ie less than 1/3 of the image contains
                # target tissue) don't process
                if (np.sum((imgSect_ref>0)*1) <= p**2 / 3) or (np.sum((imgSect_tar>0)*1) <= p**2 / 3):
                    continue

                # get the key points and descriptors of each section
                kp_ref, des_ref = sift.detectAndCompute(imgSect_ref,None)
                kp_tar, des_tar = sift.detectAndCompute(imgSect_tar,None)

                # create lists to store section specific match finding info
                kp_keep_ref = []
                des_keep_ref = []
                kp_keep_tar = []
                des_keep_tar = []
                if (des_ref is not None) and (des_tar is not None):
                    # identify strongly identifiable features in both the target and 
                    # reference tissues
                    for kpi, desi in zip(kp_ref, des_ref):
                        # set a minimum size for the feature match
                        if kpi.size > 10:
                            # extract the position of the found feature and adjust
                            # back to the global size 
                            kp_keep_ref.append(np.array(kpi.pt) + np.array([r*p, c*p]))

                            # store the descriptor
                            des_keep_ref.append(desi)

                    for kpi, desi in zip(kp_tar, des_tar):
                        if kpi.size > 10:
                            # NOTE if the range of search for targets is larger then the adjust needs to match as well
                            kp_keep_tar.append(np.array(kpi.pt) + np.array([(r-1)*p, (c-1)*p])) 
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
                        if m_info['distance'][bestMatch] < 200:
                            matchRef.append(m_info['ref'][bestMatch])
                            matchTar.append(m_info['tar'][bestMatch])


        # add annotations to where the matches have been found
        for k in matchRef:
            cv2.circle(img_ref, tuple(k.astype(int)), 20, (0, 0, 255), 8)

        for k in matchTar:
            cv2.circle(img_tar, tuple(k.astype(int)), 20, (0, 0, 255), 8)
        
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(img_ref); 
        ax2.imshow(img_tar); 
        plt.show()