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
def specID(dataHome, name, size, cpuNo = False):

    # get the size specific source of information
    datasrc = dataHome + str(size) + "/"

    # gets the images for processing
    sectionSelecter(name, datasrc, cpuNo)


def sectionSelecter(spec, datasrc, cpuNo = False):

    # this function creates a mask which is trying to selectively surround
    # ONLY the target tissue
    # Inputs:   (spec), the specific sample being processed
    #           (datasrc), the location of the jpeg images (as extracted 
    #               by tif2pdf)

    imgsmallsrc = datasrc + "images/"
    imgbigsrc = datasrc + "tifFiles/"

    # create the directory where the masked files will be created
    imgMasked = datasrc + "masked/"
    imgMasks = imgMasked + "masks/"
    imgPlots = imgMasked + "plot/"
    dirMaker(imgMasks)
    dirMaker(imgPlots)

    # get all the images 
    imgsmall = sorted(glob(imgsmallsrc + spec + "*.png"))
    imgbig = sorted(glob(imgbigsrc + spec + "*.tif"))
    
    print("\n   #--- SEGMENT OUT EACH IMAGE AND CREATE MASKS ---#")
    # serialised
    if cpuNo is False:
        for idir in imgsmall:    
            maskMaker(idir, imgMasks, imgPlots)

    else:
        # parallelise with n cores
        with Pool(processes=cpuNo) as pool:
            pool.starmap(maskMaker, zip(imgsmall, repeat(imgMasks), repeat(imgPlots)))
    
    print("\n   #--- APPLY MASKS ---#")
    
    # get the directories of the new masks
    masks = sorted(glob(imgMasks + "*.pbm"))

    # use the first image as the reference for colour normalisation
    # NOTE use the small image as it is faster but pretty much the 
    # same results
    try: imgref = cv2.imread(imgsmall[1])
    except: imgref = None
    
    # serialised
    if cpuNo is False:
        tifShape = {}
        jpegShape = {}
        info = []
        for m, iB, iS in zip(masks, imgbig, imgsmall):
            name = nameFromPath(iB)
            info.append(imgStandardiser(imgMasked, m, iB, iS, imgref))

    else:
        # parallelise with n cores
        with Pool(processes=cpuNo) as pool:
            info = pool.starmap(imgStandardiser, zip(repeat(imgMasked), masks, imgbig, imgsmall, repeat(imgref)))

        # extract the tif and jpeg info
        tifShape = {}
        jpegShape = {}
        for i in info:
            tifShape.update(i[0])
            jpegShape.update(i[1])
    
    dictToTxt(tifShape, datasrc + "info/all.tifshape")
    dictToTxt(jpegShape, datasrc + "info/all.jpgshape")
    print('Info Saved')
    
    # NOTE this takes ages on the tifs.....
    '''
    print("\n   #--- NORMALISE COLOURS ---#")
    # NOTE this is done seperately from the masking so that the colour 
    # normalisation is done on masked images, rather than images on slides
    # get all the masked images 
    imgsmallmasked = sorted(glob(imgMasked + "*png"))
    imgbigmasked = sorted(glob(imgMasked + "*tif")) 

    imgref = cv2.imread(imgsmallmasked[1])

    if cpuNo is False:
        # normalise the colours of the images
        for imgtar in imgsmallmasked + imgbigmasked:
            imgNormColour(imgtar, imgref)
    else:
        with Pool(processes=cpuNo) as pool: 
            pool.starmap(imgNormColour, zip(imgsmallmasked + imgbigmasked, repeat(imgref)))

    # create the all.shape information file
    '''

def maskMaker(idir, imgMasked = None, imgplot = None):     

    # this function loads the desired image and processes it as follows:
    #   0 - GrayScale image
    #   1 - Hard coded specimen specific transforms
    #   2 - Low pass filter (to remove noise)
    #   3 - Passes through a tanh filter to accentuate the darks and light, adaptative
    #   4 - Smoothing function
    #   5 - binarising funciton, adaptative
    #   6 - single feature identification with flood fill
    # Inputs:   (img), the image to be processed
    #           (imgplot), boolean whether to show key processing outputs, defaults false
    # Outputs:  (im), mask of the image  

    #     figure.max_open_warning --> fix this to not get plt warnings

    # use numpy to allow for parallelisation
    imgO = np.mean(cv2.imread(idir), 2)
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


    
    # ----------- low pass filter -----------

    # low pass filter
    r = 30
    f = np.fft.fft2(img.copy())
    fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 20*np.log(np.abs(fshift))
    crow, ccol = int(rows/2), int(cols/2)
    fshiftL = fshift
    filterL = np.zeros([rows, cols])
    filterL[r:-r, r:-r] = 1
    fshiftL *= filterL
    f_ishiftL = np.fft.ifftshift(fshiftL)
    img_backLow = np.fft.ifft2(f_ishiftL)
    img_lowPassFilter = np.abs(img_backLow)
    img_lowPassFilter = (img_lowPassFilter / np.max(img_lowPassFilter) * 255).astype(np.uint8)

    # ----------- background remove filter -----------

    # find the colour between the two peak value distributions 
    # this is threshold between the background and the foreground
    scl = 5
    histVals, histBins = np.histogram(img, 20)

    # the background is the maximum pixel value
    backPos = np.argmax(histVals)

    # the end of the foreground is at the inflection point of the pixel count
    diffback = np.diff(histVals[:backPos])
    try:
        forePos = np.where(np.diff(diffback) < 0)[0][-1] + 1
    except:
        forePos = backPos - 2

    # find the local minima between the peaks on a higher resolution histogram profile
    histValsF, histBinsF = np.histogram(img, 100)
    backVal = (forePos + 1) * 5 + np.argmin(histValsF[(forePos + 1) * 5 :backPos * 5])
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
    im_accentuate = img_lowPassFilter.copy()
    a = 127.5           # this sets the tanh to plateau at 0 and 255 (pixel intensity range)
    b = a - background  # this moves the threshold point to where the background is found
    im_accentuate = (a * np.tanh((im_accentuate - a + b) * 3) + a).astype(np.uint8)
    im_accentuate = (im_accentuate - np.min(im_accentuate)) / (np.max(im_accentuate) - np.min(im_accentuate)) * 255

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
    for x in np.arange(0.25, 1, 0.25):
        for y in np.arange(0.25, 1, 0.25):
            binPos = np.where(im_binary==1)
            pointV = binPos[0][np.argsort(binPos[0])[int(len(binPos[0])*y)]]
            vPos = np.where(binPos[0] == pointV)
            pointH = binPos[1][vPos[0]][np.argsort(binPos[1][vPos[0]])[int(len(vPos[0])*x)]]
            points.append(tuple([int(pointH), int(pointV)]))   # centre point

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
    
    # segment out the image
    extract = bounder(im_id)

    # save the mask as a .pbm file
    cv2.imwrite(imgMasked + name + ".pbm", im_id)
    print("     " + name + " Masked")
    # plot the key steps of processing
    if imgplot is not None:
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
        ax1.set(xlabel='Pixel bin', ylabel='Pixel count')

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
        for n in extract:
            x, y = extract[n]
            for i in range(2):
                cv2.line(imgMod, (x[i], y[i]), (x[1], y[0]), (255, 0, 0), 10)
                cv2.line(imgMod, (x[i], y[i]), (x[0], y[1]), (255, 0, 0), 10)


        # ax4.subplot(3, 2, 6)
        ax4.imshow(imgMod, cmap = 'gray') 
        ax4.axis("off")
        ax4.title.set_text("masked image")
        f.tight_layout(pad = 0.1)
        # plt.show()
        plt.savefig(imgplot + name + ".jpg")
        plt.clf()

def bounder(im_id):
    
    # this function extracts the co-ordinates which are to be used to bound
    # the mask and image
    # Inputs:   (im_id), the binary image
    # Outputs:  (extractA), co-ordinates of bounding positions corresponding to 
    #           the binary image

    def edgefinder(im, max = False):
        
        # this function find the occurence of the up and down edges
        # indicating the start and end of an image
        # Inputs:   (im): binary image 
        #           (vertical): boolean, if it is vertical then it changes this 
        #           from finding multple starts and stops to finding only the max
        #           positions (used to find vertical points in a known continuous 
        #           structure)
        # Outputs:  (up, down): positions of the edges

        # find where the edges are and get the mid points between samples
        # (ignore the start and finish points)

        # convert the image into a 1d array 
        count = (np.sum((im), axis = 0)>0)*1      # this robustly flattens the image into a 1D object
        l = len(count)

        # get the edges
        down = np.where(np.diff(count) == -1)[0]
        up = np.where(np.diff(count) == 1)[0]

        # check there are values in up and down
        # if there is an 'up' occuring before a 'down', remove it 
        # (there has to be an image before a midpoint occurs)
        if len(up) * len(down) > 0:
            if up[0] > down[0]:
                up = np.insert(up, 0, 0)
            if down[-1] < up[-1]:
                down = np.insert(down, 0, l) 
        # if there is no start or stop just add the start and end
        if len(up) == 0:
            up = np.insert(up, 0, 0)
        if len(down) == 0:
            down = np.insert(down, 0, l)

        # ensure the order of points
        down = np.sort(down)
        up = np.sort(up)

        if max:
            down = np.max(down)
            up = np.min(up)

        return(up, down)
    
    rows, cols = im_id.shape
    
    # flatten the mask and use this to figure out how many samples there 
    # are and where to split them
    imgr = cv2.resize(im_id, (100, 100))
    resized = cv2.erode(imgr, (3, 3), iterations=5)

    # plt.imshow(resize); plt.show()
    # resize = cv2.erode(im_id, (3, 3), iterations = 5)
    x, y = resized.shape

    start, end = edgefinder(resized)

    extractA = {}
    extractS = {}
    # find the horizontal start and stop positions of each sample
    for n, (s, e) in enumerate(zip(start, end)):
        extractA[n] = []
        extractS[n] = []
        sampH = np.clip(np.array([s, e+1]).astype(int), 0, y)    # +- 3 to compensate for erosion (approx)
        extractA[n].append((sampH * cols / y).astype(int))
        extractS[n].append(sampH)
    
    # find the vertical stop an start positions of each sample
    for ext in extractS:
        x0, x1 = extractS[ext][0]
        imgsect = resized[:, x0:x1]
        bottom, top = edgefinder(cv2.rotate(imgsect, cv2.ROTATE_90_COUNTERCLOCKWISE), True)
        sampV = np.clip(np.array([bottom-5, top+1]).astype(int), 0, x)   # +- 3 to compensate for erosion (approx)
        extractA[ext].append((sampV * rows / x).astype(int))

    return(extractA)

def imgStandardiser(destPath, maskpath, imgbigpath, imgsmallpath, imgref):

    # this applies the mask created to the lower resolution and full 
    # resolution images and creates information needed for the alignment
    # an area of the largest possible dimension of all the images
    # Inputs:   (maskPath), path of the sample mask
    #           (imgbigpath), path of the tif image of the samples
    #           (imgsmallpath), path of the reduced sized image
    #           (imgref), reference image for colour normalisation
    # Outputs:  (), saves image at destination with standard size and mask if inputted
    #           (jpgShapes, tifShapes), image shapes of the small and large images

    # get info to place all the images into a standard size to process (if there
    # is a mask)

    name = nameFromPath(imgsmallpath, 3)
    print(name + " modifying")
    imgsmall = cv2.imread(imgsmallpath)
    imgbig = 1 #cv2.imread(imgbigpath)
    tifShape = {}
    jpegShape = {}
    # ratio = np.round(imgsmall.shape[0]/imgbig.shape[0], 2)

    # ----------- HARD CODED SPECIMEN SPECIFIC MODIFICATIONS -----------

    # if there are just normal masks, apply them
    if maskpath is not None:

        # read in the raw mask
        mask = (cv2.imread(maskpath)/255).astype(np.uint8)

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
            if maskE.size < mask.size * 0.2 / len(extract):
                continue

            # create a a name
            newid = name + "_" + str(id)

            # extract only the image which contains the sample
            imgsmallsect = imgsmall[y[0]:y[1], x[0]:x[1], :]

            # adjust for the original size image
            # xb, yb = (extract[n]/ratio).astype(int)
            # imgbigsect = imgbig[yb[0]:yb[1], xb[0]:xb[1], :]

            # expand the dims so that it can multiply the original image
            maskS = cv2.resize(maskE, (imgsmallsect.shape[1], imgsmallsect.shape[0])).astype(np.uint8)
            # maskB = cv2.resize(maskE, (imgbigsect.shape[1], imgbigsect.shape[0])).astype(np.uint8)

            # apply the mask to the images 
            imgsmallsect *= maskS
            # imgbigsect *= maskB

            # write the new images
            cv2.imwrite(destPath + newid + ".png", imgsmallsect)
            # cv2.imwrite(imgMasked + newid + ".tif", imgbigsect)

            id += 1
            
            # save the new image dimensions
            # tifShape[newid] = imgbigsect.shape
            jpegShape[newid] = imgsmallsect.shape

            print("     " + newid + " made")

    return([tifShape, jpegShape])

def imgNormColour(imgtarpath, imgref):

    # normalises all the colour channels of an image
    # Inputs:   (imgtarpath), image path to change
    #           (imgref), image as array which has the colour properties to match
    # Outputs:  (), re-saves the image

    print("Normalising " + imgtarpath.split("/")[-1])
    imgtar = cv2.imread(imgtarpath)

    for c in range(3):
        imgtar[:, :, c] = hist_match(imgtar[:, :, c], imgref[:, :, c])

    cv2.imwrite(imgtarpath, imgtar)


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    dataSource = '/Volumes/USB/Testing1/'
    dataSource = '/Volumes/USB/IndividualImages/'
    dataSource = '/Volumes/USB/H653A_11.3/'
    dataSource = '/Volumes/USB/H671B_18.5/'
    dataSource = '/Volumes/Storage/H653A_11.3/'
    dataSource = '/Volumes/USB/H1029A_8.4/'
    dataSource = '/Volumes/USB/Test/'
    dataSource = '/Volumes/USB/H750A_7.0/'
    dataSource = '/Volumes/USB/H671A_18.5/'
    dataSource = '/Volumes/USB/H673A_7.6/'
    dataSource = '/Volumes/USB/H710B_6.1/'
    dataSource = '/Volumes/Storage/H710C_6.1/'

    name = ''
    size = 3
    cpuNo = 6
        
    specID(dataSource, name, size, cpuNo)