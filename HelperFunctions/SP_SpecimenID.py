'''
This function creates a mask around the target specimen and seperates multiple 
samples into seperate images
'''

if __name__ == "__main__":
    from Utilities import nameFromPath, dirMaker, dictToTxt
else:
    from HelperFunctions.Utilities import nameFromPath, dirMaker, dictToTxt
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os 
from glob import glob

from multiprocessing import Process
from PIL import Image

def specID(dataHome, name, size):


    # get the size specific source of information
    datasrc = dataHome + str(size) + "/"

    # gets the images for processing
    imgsrc = datasrc + "images/"
    imgsrc = '/Volumes/USB/IndividualImages/'

    # get the directories which are NOT masked
    specimens = []
    for i in os.listdir(imgsrc):
        if (i.find("masked")<0): 
            specimens.append(i) 
    
    # iterate through each specimen and perform feature mapping between each sample
    # NOTE tried to parallelise but once again cv2 is a huge hang up.....
    for spec in specimens[2:3]:
        sectionSelecter(spec, imgsrc)
        
    # this always dies on my mac.... only use on HPC
    '''
    jobs = {}
    for n in range(int(len(specimens)/4)):
        # multiprocess in batches to avoid destroying my memory!
        # see if concurrent processing perhaps???
        print("\nLot: " + str(n))

        specRange = specimens[n*4:(n+1)*4]

        for spec in specRange:    
            # sectionSelecter(spec, dataSource, dataDestination)      
            jobs[spec] = Process(target=sectionSelecter, args = (spec, dataSource, dataDestination))
            print("process " + spec + " allocated")
            jobs[spec].start()
            print("process " + spec + " started")
        
        for spec in specRange:
            print("process " + spec + " joined")
            jobs[spec].join()
    '''

def sectionSelecter(spec, datasrc):

    # this function creates a mask which is trying to selectively surround
    # ONLY the target tissue
    # Inputs:   (spec), the specific sample being processed
    #           (datasrc), the location of the jpeg images (as extracted 
    #               by tif2pdf)

    imgsrc = datasrc + spec

    # create the directory where the masked files will be created
    imgMasked = imgsrc + "masked/"
    dirMaker(imgMasked)

    imgs = sorted(glob(imgsrc + "/*.jpg"))

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
        try:
            mask = maskMaker(img, 15)
        # if a mask can't be made, just return the whole image
        except:
            mask = np.ones([img.shape[0], img.shape[1]]).astype(np.uint8)
            print(name + " has no mask")

        # plt.imshow(cv2.cvtColor(imgMasked, cv2.COLOR_BGR2RGB)); plt.show()

        # store mask info to standardise mask size
        masksStore[name] = mask
        maskShapes.append(mask.shape)

    print(spec + " Masks created")

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
    imgStandardiser(imgs, imgMasked, MasksStandard)
    print(spec + " Images modified")

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

    # ----------- single feature ID -----------

    # find a point which is in the sample of interest. This assumes that the sample will 
    # be the most prominent feature of the mask therefore eroding until a critical value will
    # only leave pixels which belong to the target
    # NOTE add padding so that the erode can work for all positions
    b = 3
    im_centreFind = cv2.copyMakeBorder(im_binary.copy(), b, b, b, b, cv2.BORDER_CONSTANT, value = 0)
    storeC = 1
    while (np.sum(im_centreFind) > 100) and (storeC > 0):
        store0 = np.sum(im_centreFind)
        im_centreFind = cv2.erode(im_centreFind, (b, b))
        storeC = store0 - np.sum(im_centreFind)
        im_centreFindStore = im_centreFind
        
    if np.sum(im_centreFind) == 0:
        im_centreFind = im_centreFindStore

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
        
def imgStandardiser(imgDirs, imgMasked, mask = None):

    # this gets all the images in a directory and applies a mask to isolate 
    # only the target tissue
    # an area of the largest possible dimension of all the images
    # Inputs:   (dest), destination directory for all the info
    #           (imgsrc), directory containg the images
    #           (src), source destination for all the info
    #           (mask), masks to apply to the image. 
    #               If NOT inputted then the image saved is just resized
    #               If inputted then it should be a dictionary and mask is applied
    # Outputs:  (), saves image at destination with standard size and mask if inputted

    # place all the images into a standard size to process
    img = []
    for i in imgDirs:
        img.append(cv2.imread(i))

    imgShape = []
    for i in img:
        imgShape.append(i.shape)

    imgShape = np.array(imgShape)

    yM, xM, zM = np.max(imgShape, axis = 0)

    fieldO = np.zeros([yM, xM, zM]).astype(np.uint8)

    bound = {}

    # apply the mask to all the images then save as its original size
    for i, idir in zip(img, imgDirs):

        name = nameFromPath(idir)

        field = fieldO.copy()

        y, x, z = i.shape

        ym0 = int((yM - y) / 2)
        xm0 = int((xM - x) / 2)

        # place the image in the centre of the field
        field[ym0:(ym0 + y), xm0:(xm0 + x), :z] = i

        # process if there is a mask
        if (type(mask) != None) & (np.sum(mask[name]) > 1):
            # rescale the mask to the image size (using PIL for multiprocessing)
            maskS = np.array(Image.fromarray(mask[name]).resize((int(xM), int(yM))))
            
            # expand the dims so that it can multiply the original image
            maskS = np.expand_dims(maskS, -1)
            field *= maskS

        imageSection = field[ym0:(ym0 + y), xm0:(xm0 + x), :z]

        cv2.imwrite(imgMasked + name + ".jpg", imageSection)


if __name__ == "__main__":

    dataSource = '/Volumes/USB/Testing1/'
    dataSource = '/Volumes/USB/IndividualImages/'
    name = ''
    size = 3
        
    specID(dataSource, name, size)
    # iterate through each specimen and perform feature mapping between each sample
    # NOTE tried to parallelise but once again cv2 is a huge hang up.....
