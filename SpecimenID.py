'''
This function creates a mask around the target specimen
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os 
from glob import glob
from HelperFunctions.Utilities import nameFromPath, dirMaker
from multiprocessing import Process
from PIL import Image

def sectionSelecter(spec, dataSource, dataDestination):

    # gets the 

    imgs = glob(dataSource + spec + "/*.jpg")
    dest = dataDestination + spec + "/"
    src = dataSource + spec + "/"

    masksStore = {}
    imgsStore = {}
    maskShapes = []

    scale = 0.05

    print("Processing " + spec)

    for i in imgs:
        name = nameFromPath(i)

        # read in the image and downsample
        imgO = cv2.imread(i)
        imgsStore[name] = imgO
        y, x, c = imgO.shape

        # NOTE uses PIL object instead of cv2 so multiprocessing can work
        img = np.array(Image.fromarray(imgO).resize((int(x*scale), int(y*scale))))
        # img = cv2.resize(imgO, (int(x*scale), int(y*scale)))

        # create the mask for the individual image
        mask = maskMaker(img, 15)

        # plt.imshow(cv2.cvtColor(imgMasked, cv2.COLOR_BGR2RGB)); plt.show()

        # store mask info to standardise mask size
        masksStore[name] = mask
        maskShapes.append(mask.shape)

    print("     Masks created")

    # Standardise the size of the mask
    maskShapes = np.array(maskShapes)
    yM, xM = np.max(maskShapes, axis = 0)
    fieldO = np.zeros([yM, xM]).astype(np.uint8)
    maskAll = fieldO.copy()
    MasksStandard = {}
    for m in masksStore:
        field = fieldO.copy()
        mask = masksStore[m]
        y, x = mask.shape
        ym0 = int((yM - y) / 2)
        xm0 = int((xM - x) / 2)

        # replace the mask with a centred and standardised sized version
        field[ym0:(ym0 + y), xm0:(xm0 + x)] += mask
        MasksStandard[m] = field
        maskAll += field 

    # create a master mask which is made of only the masks which appear in at least
    # 1/3 of all the other masks --> this is to reduce the appearance of random masks
    masterMask = maskAll > len(masksStore) / 3

    # apply the master mask to all the specimen masks 
    for m in masksStore:
        MasksStandard[m] *= masterMask

    # apply the mask to all the images and save
    imgStandardiser(dest, src, MasksStandard)
    print("     Images modified")

def maskMaker(imgO, r, plotting = False):     

    # this function loads the desired image and processes it as follows:
    #   1 - GrayScale image
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

    # make image grayScale
    img = cv2.cvtColor(imgO, cv2.COLOR_BGR2GRAY)
    
    # ----------- low pass filter -----------

    # low pass filter
    f = np.fft.fft2(img.copy())
    fshift = np.fft.fftshift(f)
    # magnitude_spectrum = 20*np.log(np.abs(fshift))
    rows, cols = img.shape
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
    a = np.mean(im_accentuate)
    a = 127.5           # this sets the tanh to plateau at 0 and 255 (pixel intensity range)
    b = a - np.median(im_accentuate) # sample specific adjustments, 
                                        # NOTE this method is just based on observation, no 
                                        # actual theory... seems to work. Key is that it is 
                                        # sample specific
    im_accentuate = im_accentuate / np.max(im_accentuate) * 255
    im_accentuate = (a * np.tanh((im_accentuate - a + b) * 3 / a) + a).astype(np.uint8)

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

    # ----------- single feature ID -----------

    # find a point which is in the sample of interest. This assumes that the sample will 
    # be the most prominent feature of the mask therefore eroding until a critical value will
    # only leave pixels which belong to the target
    # NOTE add padding so that the erode can work for all positions
    b = 3
    im_centreFind = cv2.copyMakeBorder(im_binary.copy(), b, b, b, b, cv2.BORDER_CONSTANT, value = 0)
    storeC = 1
    while (np.sum(im_centreFind)-storeC > 100) and (storeC > 0):
        store0 = np.sum(im_centreFind)
        im_centreFind = cv2.erode(im_centreFind, (b, b))
        storeC = store0 - np.sum(im_centreFind)
        

    # pick the lowest point found (from observation, there appears to be more noise near the 
    # origin of the image [top left visually] so this is just another step to make a more robust
    # spec finder)
    points = np.where(im_centreFind > 0) 
    point = (points[1][-1] - b, points[0][-1] - b)

    # only use the mask for the target sample 
    im_id = (cv2.floodFill(im_binary.copy(), None, point, 255)[1]/255).astype(np.uint8)
    im_id = cv2.dilate(im_id, (5, 5), iterations=3)      # build edges back up
    
    # plot the key steps of processing
    if plotting:
        # create sub plotting capabilities
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        cv2.circle(imgO, tuple(point), 3, (255, 0, 0), 2)
        imgMod = imgO * im_id

        ax1.imshow(im_accentuate, cmap = 'gray')
        ax1.set_title('accentuated colour')

        ax2.imshow(im_centreFind, cmap = 'gray')
        ax2.set_title("centreFind Mask")

        ax3.imshow(im_id[:, :, 0], cmap = 'gray')
        ax3.set_title("identified sample ")

        ax4.imshow(imgMod, cmap = 'gray') 
        ax4.set_title("masked image")
        plt.show()
    
    '''
    # convert mask into 3D array and rescale for the original image
    im = cv2.resize(im, (int(x), int(y)))
    im = np.expand_dims(im, -1)
    '''
    return(im_id)
        
def imgStandardiser(dest, src, mask = None):

    # this gets all the images in a directory and standardises their size and applies a mask
    # by placing the original image in the middle of a field which encompasses 
    # an area of the largest possible dimension of all the images
    # Inputs:   (dest), destination directory for all the info
    #           (src), source destination for all the info
    #           (mask), masks to apply to the image. 
    #               If NOT inputted then the image saved is just resized
    #               If inputted then it should be a dictionary and mask is applied
    # Outputs:  (), saves image at destination with standard size and mask if inputted

    dirMaker(dest)

    imgDirs = glob(src + "/*.jpg")

    img = []
    for i in imgDirs:
        img.append(cv2.imread(i))

    imgShape = []
    for i in img:
        imgShape.append(i.shape)

    imgShape = np.array(imgShape)

    yM, xM, zM = np.max(imgShape, axis = 0)

    fieldO = np.zeros([yM, xM, zM]).astype(np.uint8)

    for i, idir in zip(img, imgDirs):

        name = nameFromPath(idir)

        field = fieldO.copy()

        y, x, z = i.shape

        ym0 = int((yM - y) / 2)
        xm0 = int((xM - x) / 2)

        # place the image in the centre of the field
        field[ym0:(ym0 + y), xm0:(xm0 + x), :z] = i

        if mask:
            # rescale the mask to the image size (using PIL for multiprocessing)
            maskS = np.array(Image.fromarray(mask[name]).resize((int(xM), int(yM))))

            # expand the dims so that it can multiply the original image
            maskS = np.expand_dims(maskS, -1)
            field *= maskS

        cv2.imwrite(dest + name + ".jpg", field)


if __name__ == "__main__":
        
    dataSource = '/Volumes/USB/InvididualImages/'
    dataDestination = '/Volumes/USB/InvididualImagesMod2/'

    specimens = os.listdir(dataSource)
    # specimens = ['temporaryH653']

    # specimens = ['temporaryH653']

    jobs = {}

    # iterate through each specimen and perform feature mapping between each sample
    # NOTE tried to parallelise but once again cv2 is a huge hang up.....
    for spec in specimens:    

        # sectionSelecter(spec, dataSource, dataDestination)      
        jobs[spec] = Process(target=sectionSelecter, args = (spec, dataSource, dataDestination))
    
    for spec in specimens:
        jobs[spec].start()
