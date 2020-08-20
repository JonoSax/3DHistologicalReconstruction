'''
This function automatically creates a mask around the target specimen and seperates multiple 
samples into seperate images.
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os 
from glob import glob
from PIL import Image
from multiprocessing import Process, Queue
if __name__ == "__main__":
    from Utilities import nameFromPath, dirMaker, dictToTxt
else:
    from HelperFunctions.Utilities import nameFromPath, dirMaker, dictToTxt

'''

TO DO:

    - have this file create create multiple .bound files if there are multiple specimens

    - NOTE this fails to be useful for:
        images with multiple samples in them
        if the samples are not the correct orientation (masterMask prevent outlier images
        form being processed)

    - make this work if a single sample within a specimen is selected (ie the spec in 
    selectionSelection works for a string input as well as a list)

'''

def specID(dataHome, name, size):

    # get the size specific source of information
    datasrc = dataHome + str(size) + "/"

    # gets the images for processing
    
    # imgsrc = '/Volumes/USB/IndividualImages/'

    sectionSelecter(name, datasrc)


def sectionSelecter(spec, datasrc):

    # this function creates a mask which is trying to selectively surround
    # ONLY the target tissue
    # Inputs:   (spec), the specific sample being processed
    #           (datasrc), the location of the jpeg images (as extracted 
    #               by tif2pdf)

    imgsrc = datasrc + "images/"
    # imgsrc = '/Volumes/USB/IndividualImages/temporaryH710A/'
    # datasrc = '/Volumes/USB/IndividualImages/temporaryH710A/'

    # create the directory where the masked files will be created
    imgMasked = datasrc + "masked/"
    dirMaker(imgMasked)

    imgs = sorted(glob(imgsrc + spec + "*.*"))

    masksStore = {}
    splitStore = {}
    imgsStore = {}
    maskShapes = []
    q = {}
    jobs = {}

    scale = 0.5

    print("Processing " + spec)

    # NOTE this can probably be parallelised 
    for i in imgs:
        name = nameFromPath(i)
        q[name] = Queue()

        # read in the image and downsample
        imgO = cv2.imread(i)
        imgsStore[name] = imgO
        imgShapes = imgO.shape

        # downsample the image for masking
        # NOTE uses PIL object instead of cv2 so multiprocessing can work
        # img = np.array(Image.fromarray(imgO).resize((int(x*scale), int(y*scale))))
        # img = cv2.resize(imgO, (int(x*scale), int(y*scale)))

        img = imgO  # NOTE it looks like the mask making is REALLY good at full res
        
        # create the mask for the individual image
        mask, split = maskMaker(name, img, 15, False)
        
        # plt.imshow(cv2.cvtColor(imgMasked, cv2.COLOR_BGR2RGB)); plt.show()

        # store mask info to standardise mask size
        masksStore[name] = mask
        splitStore[name] = split
        maskShapes.append(mask.shape)

    '''
    for name in nameFromPath(imgs):
        mask = q[name].get()
        print(name + " got")
        jobs[name].join()
        masksStore[name] = mask
        maskShapes.append(mask.shape)
    '''

    print(spec + "   Masks created")

    masksStore, maskShapes = masterMaskMaker(name, maskShapes, masksStore)

    # NOTE this should be optional.....
    # apply the mask to all the images and save
    jpgShapes = {}
    for i in imgs:
        name = nameFromPath(i)
        jpgShapes = imgStandardiser(i, imgMasked, splitStore[name], jpgShapes, masksStore[name], maskShapes)
        print(name + " modified")

    # store the jpg shapes
    dictToTxt(jpgShapes, datasrc + "info/all.jpgshape")

def maskMaker(name, imgO, r, plotting = False, q = None):     

    # this function loads the desired image and processes it as follows:
    #   0 - GrayScale image
    #   1 - TODO, hard coded specimen specific transforms
    #   2 - Low pass filter (to remove noise)
    #   3 - Passes through a tanh filter to accentuate the darks and light, adaptative
    #   4 - Smoothing function
    #   5 - binarising funciton, adaptative
    #   6 - single feature identification with flood fill
    # Inputs:   (img), the image to be processed
    #           (r), cut off frequency for low pass filter
    #           (plotting), boolean whether to show key processing outputs, defaults false
    # Outputs:  (im), mask of the image  

    # ----------- grayscale -----------

    print(name + " masking")

    # make image grayScale
    img = cv2.cvtColor(imgO, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape

    # ----------- specimen specific modification -----------
    
    # H653 specimen specific mod
    if name.find("H653") >= 0:
        '''
        H653 has bands from the plate which the specimen is stored on which causes a lot of 
        disruption to the img and compared to the amount of tissue in these bands, is more of an
        issue 
        '''

        img[:int(cols * 0.08), :] = 255
        img[-int(cols * 0.05):, :] = 255
        a = 150

    if name.find("H710B") >= 0:

        # remove some of the bottom row
        img[-int(cols*0.05):, :] = np.median(img)
        a = np.mean(img)

    
    # ----------- low pass filter -----------

    # low pass filter
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

    # ----------- tanh filter -----------

    # accentuate the colour
    im_accentuate = img_lowPassFilter.copy()
    # a = 127.5           # this sets the tanh to plateau at 0 and 255 (pixel intensity range)
    b = a - np.median(im_accentuate) # sample specific adjustments, 
                                        # NOTE this method is just based on observation, no 
                                        # actual theory... seems to work. Key is that it is 
                                        # sample specific
    im_accentuate = (a * np.tanh((im_accentuate - a + b) * 3 / a) + a).astype(np.uint8)
    im_accentuate = (im_accentuate - np.min(im_accentuate)) / (np.max(im_accentuate) - np.min(im_accentuate)) * 255

    # ----------- smoothing -----------

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
    im_binary = cv2.dilate(im_binary, (5, 5), iterations=10)      # build edges back up

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
    im_id = cv2.rotate(cv2.dilate(cv2.rotate(im_id, cv2.ROTATE_180), (5, 5), iterations = 5), cv2.ROTATE_180)
    
    # flatten the mask and use this to figure out how many samples there 
    # are and where to split them
    count = (np.sum(cv2.resize(im_id, (20, 5)), axis = 0)>0)*1      # this robustly flattens the image into a 1D object
    # plt.imshow(np.expand_dims(count, 0)); plt.show()

    edgeup = False
    edgedown = False
    midp = []
    midp.append(0)
    for n in range(len(count)-1):
        xo = int(count[n])
        x1 = int(count[n+1])
        if x1 - xo == 1:
            if edgedown:
                edgeup = True
            mide = n
        elif xo - x1 == 1:
            edgedown = True
            mids = n
        # get the middle of the images
        if edgeup and edgedown:
            midp.append(int((mide + mids)/2 * cols / 20))
            edgeup = False
            edgedown = False
    midp.append(cols)
    midp = np.array(midp)

    # plot the key steps of processing
    if plotting:
        # create sub plotting capabilities
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        imgMod = imgO * np.expand_dims(im_id, -1)

        ax1.imshow(im_accentuate, cmap = 'gray')
        ax1.set_title('accentuated colour')

        # turn the mask into an RGB image (so that the circle point is clearer)
        im_binary3d = (np.ones([im_binary.shape[0], im_binary.shape[1], 3]) * np.expand_dims(im_binary * 255, -1)).astype(np.uint8)
        for point in points:
            cv2.circle(im_binary3d, tuple(point), 100, (255, 0, 0), 20)

        ax2.imshow(im_binary3d)
        ax2.set_title("centreFind Mask")

        ax3.imshow(im_id, cmap = 'gray')
        ax3.set_title("identified sample ")

        ax4.imshow(imgMod) 
        ax4.set_title("masked image")
        plt.show()
    
    '''
    # convert mask into 3D array and rescale for the original image
    im = cv2.resize(im, (int(x), int(y)))
    im = np.expand_dims(im, -1)
    '''

    if q is None:
        return(im_id, midp)
    else:
        q.put(im_id, midp)

def masterMaskMaker(name, maskShapes, masksStore): 

    # for specific specimens, a master mask (combining all the masks together)
    # is useful to reduce outlier mask outlines
    if name.find("H653") >=0 or name.find("1029a") >= 0:
        # Standardise the size of the mask to create a mastermask
        maskShapes = np.array(maskShapes)
        yM, xM = np.max(maskShapes, axis = 0)
        fieldO = np.zeros([yM, xM]).astype(np.uint8)
        maskAll = fieldO.copy()
        for m in masksStore:
            field = fieldO.copy()
            mask = masksStore[m]
            y, x = mask.shape
            ym0 = int((yM - y) / 2)
            xm0 = int((xM - x) / 2)

            # replace the mask with a centred and standardised sized version
            field[ym0:(ym0 + y), xm0:(xm0 + x)] += mask
            masksStore[m] = field
            maskAll += field 

        # create a master mask which is made of only the masks which appear in at least
        # 1/3 of all the other masks --> this is to reduce the appearance of random masks
        masterMask = maskAll > len(masksStore) / 3

    else:
        maskShapes = None        # if not doing a master mask don't need the shapes
        masterMask = 1

    # ------------------------------------------------------------------

    # apply the master mask to all the specimen masks 
    for m in masksStore:
        masksStore[m] *= masterMask  

    return(masksStore, maskShapes)

def imgStandardiser(imgDir, imgMasked, split, jpgShapes, maskS = None, imgShapes = None):

    # this gets all the images in a directory and applies a mask to isolate 
    # only the target tissue
    # an area of the largest possible dimension of all the images
    # Inputs:   (dest), destination directory for all the info
    #           (imgsrc), directory containg the images
    #           (split), the positions to split the images if there are multiple samples
    #           (jpgShapes), dictionary to store all the jpg image shapes that are segemented out, it is recursive
    #           (src), source destination for all the info
    #           (mask), masks to apply to the image. 
    #               If NOT inputted then the image saved is just resized
    #               If inputted then it should be a dictionary and mask is applied
    # Outputs:  (), saves image at destination with standard size and mask if inputted
    #           (jpgShapes), jpeg image shapes 

    # get info to place all the images into a standard size to process (if there
    # is a mask)

    name = nameFromPath(imgDir)
    img = cv2.imread(imgDir)

    # if there is a master mask, process
    if imgShapes is not None:

        imgShapes = np.array(imgShapes)
        yM, xM = np.max(imgShapes, axis = 0)
        fieldO = np.zeros([yM, xM, 3]).astype(np.uint8)

        field = fieldO.copy()
        y, x, z = img.shape

        ym0 = int((yM - y) / 2)
        xm0 = int((xM - x) / 2)

        # place the image in the centre of the field
        field[ym0:(ym0 + y), xm0:(xm0 + x), :z] = img
        img = field

        # process if there is a mask
        # repositin the mask to the new image (using PIL for multiprocessing)
        # maskS = np.array(Image.fromarray(maskS).resize((int(xM), int(yM))))

    # if there are just normal masks, apply them
    if maskS is not None:

        # expand the dims so that it can multiply the original image
        maskS = np.expand_dims(maskS, -1)
        img *= maskS

    if imgShapes is not None:
        
        # extract the image from the centre of the field back into its original position
        img = img[ym0:(ym0 + y), xm0:(xm0 + x), :z]

    # save the segmented images
    for n in range(len(split) - 1): 
        xo = split[n]
        x1 = split[n+1]
        imgSect = img[:, xo:x1, :]
        cv2.imwrite(imgMasked + name + "_" + str(n) + ".jpg", imgSect)
        jpgShapes[name + "_" + str(n)] = imgSect.shape

    return(jpgShapes)


if __name__ == "__main__":

    dataSource = '/Volumes/USB/Testing1/'
    # dataSource = '/Volumes/USB/IndividualImages/'
    dataSource = '/Volumes/Storage/H653A_11.3new/'
    name = ''
    size = 3
        
    specID(dataSource, name, size)
    # iterate through each specimen and perform feature mapping between each sample
    # NOTE tried to parallelise but once again cv2 is a huge hang up.....
