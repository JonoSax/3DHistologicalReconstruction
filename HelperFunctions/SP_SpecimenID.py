'''
This function automatically creates a mask around the target specimen and seperates multiple 
samples into seperate images.
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os 
from glob import glob
from multiprocessing import Pool
import multiprocessing
from itertools import repeat
if __name__ != "HelperFunctions.SP_SpecimenID":
    from Utilities import *
else:
    from HelperFunctions.Utilities import *

'''
TODO: explanation of what happens here.....

-  Investigate if this now works for a single sift operator applied over the 
entire image, rather than segmenting the image into grids

'''

# NOTE can probably depreciate specID and make sectionSelecter the main function
def specID(dataHome, size, cpuNo = False, imgref = 'refimg.png'):

    # get the size specific source of information
    datasrc = dataHome + str(size) + "/"

    # get the reference image path
    if imgref is not None:
        refimgPath = getSampleName(dataHome, imgref)
        imgref = cv2.imread(refimgPath)

    # gets the images for processing
    sectionSelecter(datasrc, cpuNo, imgref)


def sectionSelecter(datasrc, cpuNo = False, imgref = None, plot = False):

    '''
    This function creates a mask which is trying to selectively surround
    ONLY the target tissue and normalises the image colours

        Inputs:

    (spec), the specific sample being processed\n
    (datasrc), the location of the jpeg images (as extracted)\n
    (cpuNo), number of cores to use for parallelisations\n
    (refimg), the reference image to use for colour normalisations. If set to None
    will not perform this\n

        Outputs:\n

    (), create the down-sampled and full scale tif images with their respecitve 
    samples extracted and with their colours normalised against a reference image 
    '''

    imgsmallsrc = datasrc + "images/"
    imgbigsrc = datasrc + "tifFiles/"

    # create the directory where the masked files will be created
    imgMasked = datasrc + "maskedSamples/"
    imgMasks = imgMasked + "masks/"
    imgPlots = imgMasked + "plot/"
    dirMaker(imgMasks)
    dirMaker(imgPlots)
    
    # get all the images 
    imgsmall = sorted(glob(imgsmallsrc + "*.png"))
    imgbig = sorted(glob(imgbigsrc + "*.tif"))
    
    print("\n   #--- SEGMENT OUT EACH IMAGE AND CREATE MASKS ---#")
    # serialised
    if cpuNo == 1:
        for idir in imgsmall:    
            maskMaker(idir, imgMasks, imgPlots)

    else:
        # parallelise with n cores
        with Pool(processes=cpuNo) as pool:
            pool.starmap(maskMaker, zip(imgsmall, repeat(imgMasks), repeat(imgPlots), repeat(plot)))
    
    print("\n   #--- APPLY MASKS ---#")
    
    # get the directories of the new masks
    masks = sorted(glob(imgMasks + "*.pbm"))

    # use the first image as the reference for colour normalisation
    # NOTE use the small image as it is faster but pretty much the 
    # same results

    # imgref = None
    # serialised
    if cpuNo == 1:
        for m in masks:
            imgStandardiser(imgMasked, m, imgsmallsrc, imgbigsrc, imgref)

    else:
        # parallelise with n cores
        with Pool(processes=cpuNo) as pool:
            pool.starmap(imgStandardiser, zip(repeat(imgMasked), masks, repeat(imgsmallsrc), repeat(imgbigsrc), repeat(imgref)))

    print('Info Saved')

def maskMaker(idir, imgMasked = None, imgplot = False, plot = False):     

    # this function loads the desired image extracts the target sample:
    # Inputs:   (img), the image to be processed
    #           (imgplot), boolean whether to show key processing outputs, defaults false
    # Outputs:  (im), mask of the image  

    #     figure.max_open_warning --> fix this to not get plt warnings

    # use numpy to allow for parallelisation
    try:
        imgO = np.round(np.mean(cv2.imread(idir), 2)).astype(np.uint8)
    except:
        print("FAILED: " + idir)
        return
    # imgO = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY) 

    name = nameFromPath(idir)

    print(name + " masking")
    rows, cols = imgO.shape

    img = imgO.copy()

    # ----------- specimen specific modification -----------
    
    # H653 specimen specific mod
    if name.find("H653") >= 0:
        '''
        H653 has bands from the plate which the specimen is stored on which causes a lot of 
        disruption to the img and compared to the amount of tissue in these bands, is more of an
        issue 
        '''

        img[:int(cols * 0.08), :] = np.median(img)
        img[-int(cols * 0.05):, :] = np.median(img)

    if name.find("H710B") >= 0:
        # remove some of the bottom row
        img[-int(cols*0.05):, :] = np.median(img)
    
    if name.find("H710C") >= 0:
        # remove a little bit of the left hand side of the image 
        img[:, -int(rows*0.05):] = np.median(img)

    if name.find("H673A") >= 0:
        # remove some of the bottome
        img[-int(cols * 0.08):, :] = np.median(img)

    if name.find("H671A") >= 0:
        # remove some of the top and bottom
        img[:int(cols*0.05), :] = np.median(img)
        img[-int(cols*0.05):, :] = np.median(img)

    if name.find("H750") >= 0:

        # remove some of the top and bottom
        img[:int(cols*0.07), :] = np.median(img)
        img[-int(cols*0.1):, :] = np.median(img)

    # ----------- background remove filter -----------

    # find the colour between the two peak value distributions 
    # this is threshold between the background and the foreground
    lBin = 20
    hBin = len(np.unique(img))
    rBin = hBin/lBin
    histVals, histBins = np.histogram(img, lBin)

    # the background is the maximum pixel value
    backPos = np.argmax(histVals)

    # the end of the foreground is at the inflection point of the pixel count
    diffback = np.diff(histVals[:backPos])
    try:
        forePos = np.where(np.diff(diffback) < 0)[0][-1] + 1
    except:
        forePos = backPos - 2

    # find the local minima between the peaks on a higher resolution histogram profile
    histValsF, histBinsF = np.histogram(img, hBin)
    backVal = int(np.round((forePos + 1) * rBin + np.argmin(histValsF[int(np.round(forePos + 1) * rBin):int(np.round(backPos * rBin))])))
    background = histBinsF[backVal]

    '''    
    plt.plot(histBins[1:], histVals); 
    plt.xlabel('pixelValue')
    plt.ylabel('pixelCount')
    plt.title(name + " intensity histogram profile")
    plt.semilogy(histBins[backPos+1], histVals[backPos], marker="o")
    plt.show()  
    '''

    # accentuate the colour
    im_accentuate = img.copy()
    b = background
    im_binary = (((im_accentuate - b) < 0)*1).astype(np.uint8)
    # im_accentuate = (a * np.tanh((im_accentuate - a + b) * 3) + a).astype(np.uint8)
    # im_accentuate = (im_accentuate - np.min(im_accentuate)) / (np.max(im_accentuate) - np.min(im_accentuate)) * 255
    
    # ----------- smoothing -----------
    # plt.imshow(im_accentuate, cmap = 'gray'); plt.show()
    # plt.scatter(histinfo[1][:-1], histinfo[0]); plt.show()
    # create kernel
    kernelb = np.ones([5, 5])
    kernelb /= np.sum(kernelb)

    # apply 
    img_smooth = cv2.filter2D(im_accentuate,-1,kernelb)
    img_smooth = cv2.erode(img_smooth, (3, 3), iterations=1)

    # ----------- adaptative binarising -----------

    # threshold to form a binary mask
    v = int((np.median(img_smooth) + np.mean(img_smooth))/2)
    im_binary = (((img_smooth<v) * 255).astype(np.uint8)/255).astype(np.uint8) #; im = ((im<=200) * 0).astype(np.uint8)  
    im_binary = cv2.dilate(im_binary, (5, 5), iterations=20)      # build edges back up
    
    # ----------- single feature ID -----------
    
    # create three points to use depending on what works for the flood fill. One 
    # in the centre and then two in the upper and lower quater along the vertical line
    points = []
    try:
        for x in np.arange(0.25, 1, 0.25):
            for y in np.arange(0.25, 1, 0.25):
                binPos = np.where(im_binary==1)
                pointV = binPos[0][np.argsort(binPos[0])[int(len(binPos[0])*y)]]
                vPos = np.where(binPos[0] == pointV)
                pointH = binPos[1][vPos[0]][np.argsort(binPos[1][vPos[0]])[int(len(vPos[0])*x)]]
                points.append(tuple([int(pointH), int(pointV)]))   # centre point
    except:
        print("     " + name + " FAILED")
        return

    # flood fill all the points found and if it is significant (ie more than a 
    # threshold % of the image is filled) keep if
    im_id = im_binary * 0
    for point in points:
        im_search = (cv2.floodFill(im_binary.copy(), None, point, 255)[1]/255).astype(np.uint8)
        if np.sum(im_search) > im_search.size * 0.05:
            im_id += im_search

    # ensure that im_id is only a mask of 0 and 1
    im_id = ((im_id>0)*1).astype(np.uint8)

    # perform an errosion on a flipped version of the image
    # what happens is that all the erosion/dilation operations work from the top down
    # so it causes an accumulation of "fat" at the bottom of the image. this removes it
    im_id = cv2.rotate(cv2.dilate(cv2.rotate(im_id, cv2.ROTATE_180), (5, 5), iterations = 10), cv2.ROTATE_180)

    # save the mask as a .pbm file
    cv2.imwrite(imgMasked + name + ".pbm", im_id)
    print("     " + name + " Masked")
    # plot the key steps of processing
    if plot:
        # create sub plotting capabilities
        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # ax1.subplot(3, 1, 1)
        ax1.semilogy(histBinsF[1:], histValsF)
        ax1.semilogy(histBinsF[backVal+1], histValsF[backVal], marker="o", color = [0, 0, 1])
        ax1.semilogy(histBins[1:], histVals)
        ax1.semilogy(histBins[forePos +1], histVals[forePos], marker="o", color = [1, 0, 0])
        ax1.semilogy(histBins[backPos +1], histVals[backPos], marker="o", color = [1, 0, 0])
        ax1.title.set_text("Histogram profile")
        ax1.set(xlabel='Pixel bin (start)', ylabel='Pixel count')

        '''
        plt.subplot(3, 2, 3)
        plt.imshow(255-im_accentuate, cmap = 'gray')
        plt.axis("off")
        plt.title('accentuated colour')
        '''

        # turn the mask into an RGB image (so that the circle point is clearer)
        im_binary3d = (np.ones([im_binary.shape[0], im_binary.shape[1], 3]) * np.expand_dims(im_binary * 255, -1)).astype(np.uint8)
        for point in points:
            cv2.circle(im_binary3d, tuple(point), 100, (255, 0, 0), 20)

        # ax2.subplot(3, 2, 4)
        ax2.imshow(im_binary3d)
        ax2.axis("off")
        ax2.title.set_text("centreFind Mask")

        # ax3.subplot(3, 2, 5)
        ax3.imshow(im_id, cmap = 'gray')
        ax3.axis("off")
        ax3.title.set_text("identified sample ")

        # imgMod = imgO * np.expand_dims(im_id, -1)
        imgMod = imgO * im_id

        # draw the bounding box of the image being extracted
        # segment out the image
        extract = bounder(im_id)
        for n in extract:
            x, y = extract[n]
            for i in range(2):
                cv2.line(imgMod, (x[i], y[i]), (x[1], y[0]), (255, 0, 0), 10)
                cv2.line(imgMod, (x[i], y[i]), (x[0], y[1]), (255, 0, 0), 10)

        # ax4.subplot(3, 2, 6)
        ax4.imshow(imgMod, cmap = 'gray') 
        ax4.axis("off")
        ax4.title.set_text("masked image")
        f.tight_layout(pad = 1)
        # plt.show()
        plt.savefig(imgplot + name + ".jpg")
        plt.clf()

def imgStandardiser(destPath, maskpath, smallsrcPath, bigsrcPath, imgRef):

    # this applies the mask created to the lower resolution and full 
    # resolution images and creates information needed for the alignment
    # an area of the largest possible dimension of all the images
    # Inputs:   (maskPath), path of the sample mask
    #           (imgbigpath), path of the tif image of the samples
    #           (imgsmallpath), path of the reduced sized image
    #           (imgref), reference image for colour normalisation
    # Outputs:  (), saves image at destination with standard size and mask if inputted

    # get info to place all the images into a standard size to process (if there
    # is a mask)

    name = nameFromPath(maskpath)
    print(name + " modifying")

    try:    
        imgsmallpath = glob(smallsrcPath + name + "*.png")[0]
        imgbigpath = glob(bigsrcPath + name + "*.tif")[0]
    except: 
        return
   
    imgsmall = cv2.imread(imgsmallpath)
    imgbig = cv2.imread(imgbigpath)
    try:
        ratio = int(np.round(imgbig.shape[0] / imgsmall.shape[0], 2))  # get the upscale size of the images
    except:
        print("---- FAILED " + name + " ----")
        return

    # if there are just normal masks, apply them
    if maskpath is not None:

        # read in the raw mask
        try:
            mask = (cv2.imread(maskpath)/255).astype(np.uint8)
        except:
            print("     FAILED: " + maskpath)
            return

        # get the bounding positional information
        extract = bounder(mask[:, :, 0])

        id = 0
        for n in extract:

            if mask is None or imgbig is None or imgsmall is None:
                print("\n\n!!! " + name + " failed!!!\n\n")
                break

            # get the co-ordinates
            x, y = extract[n]

            # extract only the mask containing the sample
            maskE = mask[y[0]:y[1], x[0]:x[1], :]

            # if each of the mask section is less than 20% of the entire mask 
            # area, it probably isn't a sample and is not useful
            if maskE.size < mask.size * 0.2 / len(extract) or np.sum(maskE) == 0:
                continue

            # create a a name
            newid = name + "_" + str(id)

            # extract only the image which contains the sample
            imgsmallsect = imgsmall[y[0]:y[1], x[0]:x[1], :]

            # adjust for the original size image
            xb, yb = np.array(extract[n]) *  ratio
            imgbigsect = imgbig[:, :, :3][yb[0]:yb[1], xb[0]:xb[1], :]

            # expand the dims so that it can multiply the original image
            maskS = cv2.resize(maskE, (imgsmallsect.shape[1], imgsmallsect.shape[0])).astype(np.uint8)
            maskB = cv2.resize(maskE, (imgbigsect.shape[1], imgbigsect.shape[0])).astype(np.uint8)

            # apply the mask to the images 
            imgsmallsect *= maskS
            imgbigsect *= maskB

            if imgRef is not None:
                imgsmallsect = imgNormColour(imgsmallsect, imgRef)#, imgbigsect)

            # write the new images
            cv2.imwrite(destPath + newid + ".png", imgsmallsect)
            # tifi.imwrite(destPath + newid + ".tif", imgbigsect)

            id += 1

            print("     " + newid + " made")

def imgNormColour(imgtarSmallpath, imgref, imgtarFullpath = None):

    '''
    Normalises all the colour channels of an image

        Inputs:\n

    (imgtarSmalldir), downsampled image path to normalise the colours for
    (imgref), image as array which has the colour properties to match
    (imgtarFullpath), full scale tif image to normalise the colour for

        Outputs:\n  

    (), over-writes the old images wit the new ones
    '''

    if type(imgtarFullpath) == str: 
        print("Normalising " + nameFromPath(imgtarSmallpath, 3))
        imgtarSmall = cv2.imread(imgtarSmallpath)

        # if the input is an image just re-assing
    else:
        imgtarSmall = imgtarSmallpath

    # if converting the tif image as well, load it and create a reference image 
    # using rgb (tifi) not bgr (cv2)
    if type(imgtarFullpath) == str: 
        imgtarFull = tifi.imread(imgtarFullpath)
        
    elif imgtarFullpath is not None:
        imgtarFull = imgtarFullpath

    imgRefRGB = cv2.cvtColor(imgref, cv2.COLOR_BGR2RGB)

    for c in range(3):
        imgtarSmall[:, :, c], normColourKey = hist_match(imgtarSmall[:, :, c], imgref[:, :, c])   

        # if there is a full scale image, those colours as well
        if imgtarFullpath is not None:

            # performing a full image normalisations
            imgtarFull[:, :, c], _ = hist_match(imgtarFull[:, :, c], imgRefRGB[:, :, c])

            # using the normcolourkey 
            # imgtarFull[:, :, c] = hist_match(imgtarFull[:, :, c], imgref[:, :, c], normColourKey)

    '''
    cv2.imwrite(imgtarSmallpath, imgtarSmall)
    if imgtarFullpath is not None:
        tifi.imwrite(imgtarFullpath, imgtarFull)
    '''

    if imgtarFullpath is not None:
        return(imgtarSmall, imgtarFull)
    else:
        return(imgtarSmall)
        
if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    dataSource = '/Volumes/USB/Testing1/'
    dataSource = '/Volumes/USB/IndividualImages/'
    dataSource = '/Volumes/USB/H671B_18.5/'
    dataSource = '/Volumes/Storage/H653A_11.3/'
    dataSource = '/Volumes/USB/H1029A_8.4/'
    dataSource = '/Volumes/USB/Test/'
    dataSource = '/Volumes/USB/H750A_7.0/'
    dataSource = '/Volumes/USB/H671A_18.5/'
    dataSource = '/Volumes/USB/H710B_6.1/'
    dataSource = '/Volumes/USB/H710C_6.1/'
    dataSource = '/Volumes/USB/H653A_11.3/'
    dataSource = '/Volumes/USB/H673A_7.6/'

    size = 3
    cpuNo = 6
        
    specID(dataSource, size, cpuNo)