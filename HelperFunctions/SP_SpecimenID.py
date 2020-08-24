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
from multiprocessing import Process, Queue, Manager
import tifffile as tifi
if __name__ == "__main__":
    from Utilities import nameFromPath, dirMaker, dictToTxt, txtToDict
else:
    from HelperFunctions.Utilities import nameFromPath, dirMaker, dictToTxt, txtToDict

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

    imgsmallsrc = datasrc + "images/"
    imgbigsrc = datasrc + "tifFiles/"

    # imgsrc = '/Volumes/USB/IndividualImages/temporaryH710A/'
    # datasrc = '/Volumes/USB/IndividualImages/temporaryH710A/'

    # create the directory where the masked files will be created
    imgMasked = datasrc + "masked/"
    dirMaker(imgMasked)

    imgsmall = sorted(glob(imgsmallsrc + spec + "*.png"))
    imgbig = sorted(glob(imgbigsrc + spec + "*.tif"))


    masksStore = {}
    splitStore = {}
    get = {}
    q = {}
    jobs = {}

    scale = 0.5

    print("Processing " + spec)

    # Create a mask from a LOWER RESOLUTION IMAGE --> uses less ram
    
    return_dict = Manager().dict()
    for i in imgsmall:
        name = nameFromPath(i, 3)

        # read in the image and downsample
        img = cv2.imread(i)

        # downsample the image for masking
        # NOTE uses PIL object instead of cv2 so multiprocessing can work
        # img = np.array(Image.fromarray(imgO).resize((int(x*scale), int(y*scale))))
        
        # img = imgO  # NOTE it looks like the mask making is REALLY good at full res
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # create the mask for the individual image
        # split = maskMaker(name, img, 30, True, imgMasked, False)

        # parallelise
        jobs[name] = Process(target = maskMaker, args = (name, img, 30, True, imgMasked, False, return_dict))
        jobs[name].start()

    for name in jobs:
        jobs[name].join()
        
    for name in return_dict.keys():
        splitStore[name] = return_dict[name]

    print(spec + "   Masks created")

    dictToTxt(splitStore, imgMasked + "all.splitstore")

    masks = sorted(glob(imgMasked + "*.pbm"))

    # APPLY THE MASK TO THE LOW AND HIGH RESOLUTION IMAGES
    # Note this is serialised only because it will destroy my ram.... 
    # loading tif files is a big no no
    tifShape = {}
    jpegShape = {}
    for iB, iS, m in zip(imgbig, imgsmall, masks):
        name = nameFromPath(iB)
        tifShape, jpegShape = imgStandardiser(iB, iS, imgMasked, tifShape, jpegShape, m)
        print(name + " modified")

    # create the all.shape information file
    dictToTxt(tifShape, datasrc + "info/all.tifshape")
    dictToTxt(jpegShape, datasrc + "info/all.jpgshape")
    print('Info Saved')

def maskMaker(name, img, r, split = True, imgMasked = None, plotting = False, return_dict = None):     

    # this function loads the desired image and processes it as follows:
    #   0 - GrayScale image
    #   1 - Hard coded specimen specific transforms
    #   2 - Low pass filter (to remove noise)
    #   3 - Passes through a tanh filter to accentuate the darks and light, adaptative
    #   4 - Smoothing function
    #   5 - binarising funciton, adaptative
    #   6 - single feature identification with flood fill
    # Inputs:   (img), the image to be processed
    #           (r), cut off frequency for low pass filter
    #           (plotting), boolean whether to show key processing outputs, defaults false
    # Outputs:  (im), mask of the image  

    print(name + " masking")
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
        b = 140

    if name.find("H710B") >= 0:

        # remove some of the bottom row
        img[-int(cols*0.05):, :] = np.median(img)
    
        b = np.mean(img)
    
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

    # ----------- background remove filter -----------

    histinfo = np.histogram(img, 20)
    hist = histinfo[0]
    diffback = np.diff(hist)
    background = histinfo[1][np.argmax(diffback)-1]

    # accentuate the colour
    im_accentuate = img_lowPassFilter.copy()
    a = 127.5           # this sets the tanh to plateau at 0 and 255 (pixel intensity range)
    b = a - background                                    # NOTE this method is just based on observation, no 
                                        # actual theory... seems to work. Key is that it is 
                                        # sample specific
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
    
    extract = np.array([])
    if split:
        # flatten the mask and use this to figure out how many samples there 
        # are and where to split them
        resize = cv2.erode(cv2.resize(im_id, (100, 100)), (3, 3), iterations=5)
        count = (np.sum((resize), axis = 0)>0)*1      # this robustly flattens the image into a 1D object
        # plt.imshow(resize); plt.show()

        # find where the edges are and get the mid points between samples
        # (ignore the start and finish points)
        down = np.where(np.diff(count) == -1)[0]
        up = np.where(np.diff(count) == 1)[0]

        # check there are values in up and down
        # if there is an 'up' occuring before a 'down', remove it 
        # (there has to be an image before a midpoint occurs)
        if len(up) * len(down) > 0:
            if up[0] > down[0]:
                up = np.insert(up, 0, 0)
            if down[-1] < up[-1]:
                down = np.insert(down, 0, 100) 
        # if there is no start or stop just add the start and end
        if len(up) == 0:
            up = np.insert(up, 0, 0)
        if len(down) == 0:
            down = np.insert(down, 0, 100)

        # ensure the order of points
        down = np.sort(down)
        up = np.sort(up)
        
        # find the start and stop of each sample, perfectly extracting them
        for d, u in zip(down, up):
            samp = (np.array([u, d]) * cols / 100)
            extract = np.hstack([extract, samp]).astype(int)
        
        extract = list(extract)

    # save the mask as a .pbm file
    cv2.imwrite(imgMasked + name + ".pbm", im_id)

    # plot the key steps of processing
    if plotting:
        # create sub plotting capabilities
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

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

        imgMod = imgO * np.expand_dims(im_id, -1)
        
        for x0, x1 in extract:
            cv2.line(imgMod, (x0, 0), (x1, rows), (255, 0, 0), 20)

        ax4.imshow(imgMod) 
        ax4.set_title("masked image")
        # plt.show()
        plt.savefig(imgMasked + 'plot' + name + ".jpg")
    
    '''
    # convert mask into 3D array and rescale for the original image
    im = cv2.resize(im, (int(x), int(y)))
    im = np.expand_dims(im, -1)
    '''

    if return_dict is None:
        return(extract)
    else:
        return_dict[name] = extract
        print("put " + name)

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

def imgStandardiser(imgbigDir, imgsmallDir, imgMasked, tifShape, jpegShape, maskdir = None):

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

    name = nameFromPath(imgsmallDir, 3)
    imgsmall = cv2.imread(imgsmallDir)
    imgbig = tifi.imread(imgbigDir)

    split = txtToDict(imgMasked + "all.splitstore")[0][name]

    ratio = np.round(imgsmall.shape[0]/imgbig.shape[0], 2)


    # ----------- HARD CODED SPECIMEN SPEICIFIC MODIFICATIONS -----------

    if (name.lower().find("a+b") == -1) & (name.lower().find("h710c") >= 0):
        imgbig = cv2.rotate(imgbig, cv2.ROTATE_180)

    # if there are just normal masks, apply them
    if maskdir is not None:

        mask = cv2.imread(maskdir)/255

        # expand the dims so that it can multiply the original image
        maskS = cv2.resize(mask, (imgsmall.shape[1], imgsmall.shape[0])).astype(np.uint8)
        maskB = cv2.resize(mask, (imgbig.shape[1], imgbig.shape[0])).astype(np.uint8)

        # apply the mask to the images 
        imgsmall *= maskS
        imgbig *= maskB

    # save the segmented images
    if len(split) > 0:
        for n in range(int(len(split)/2)):
            # get the pairs of start and end points 
            xo = split[int(2*n)]
            x1 = split[int(2*n + 1)]

            imgSectSmall = imgsmall[:, xo:x1, :]
            newid = name + "_" + str(n)

            # if more than 90% of the image is black then it is not useful information
            if np.where(imgSectSmall == 0)[0].size > (imgSectSmall.size) * 0.9:
                continue

            imgSectBig = imgbig[:, int(xo/ratio):int(x1/ratio), :]
            
            cv2.imwrite(imgMasked + newid + ".png", imgSectSmall)
            tifi.imwrite(imgMasked + newid + ".tif", imgSectBig)

            tifShape[newid] = imgSectBig.shape
            jpegShape[newid] = imgSectSmall.shape

    # if there are no segmentations, save as is
    else:
        cv2.imwrite(imgMasked + name + ".png", imgsmall)
        tifi.imwrite(imgMasked + name + ".tif", imgbig)

        tifShape[name] = imgsmall.shape
        jpegShape[name] = imgbig.shape

    return(tifShape, jpegShape)


if __name__ == "__main__":

    dataSource = '/Volumes/USB/Testing1/'
    # dataSource = '/Volumes/USB/IndividualImages/'
    dataSource = '/Volumes/USB/H653A_11.3/'
    dataSource = '/Volumes/Storage/H653A_11.3new/'
    dataSource = '/Volumes/USB/H673A_7.6/'
    dataSource = '/Volumes/USB/H710C_6.1/'
    name = ''
    size = 3
        
    specID(dataSource, name, size)
    # iterate through each specimen and perform feature mapping between each sample
    # NOTE tried to parallelise but once again cv2 is a huge hang up.....
