'''
This function creates a mask around the target specimen
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os 
from glob import glob
from Utilities import nameFromPath 

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

    # create sub plotting capabilities
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

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
    storeC = 1000
    while (np.sum(im_centreFind) > 100) and (storeC > 0):
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
    im_id = np.expand_dims(cv2.dilate(im_id, (5, 5), iterations=3), -1)      # build edges back up
    
    # plot the key steps of processing
    if plotting:
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
        

dataSource = '/Volumes/USB/InvididualImages/'
dataDestination = '/Volumes/USB/InvididualImages/'

specimens = os.listdir(dataSource)[12:]

# specimens = ['temporaryH653']

scale = 0.05

# iterate through each specimen and perform feature mapping between each sample
for spec in specimens:

    imgs = glob(dataSource + spec + "/*.jpg")

    # based on the assumption all images are the same size (they should be)
    x, y = cv2.imread(imgs[0], 0).shape
    maskAll = np.zeros([int(x * scale), int(y * scale)])

    for i in imgs:
        print(nameFromPath(i))

        # img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) 

        img = cv2.imread(i)
        y, x, c = img.shape
        img = cv2.resize(img, (int(x*scale), int(y*scale)))

        mask = maskMaker(img, 15)

        # summate all the masks generated
        # maskAll += mask

        mask = np.expand_dims(cv2.resize(mask, (x, y)), -1)

        # plt.imshow(imgO * mask, cmap = 'gray'); plt.show()

        print('done')

    # create a mask which is made of the positions where the sample was positively
    # identifed at least 1/3 of the time
    masterMask = (maskAll > len(imgs)/3).astype(np.uint8)
    masterMask = (cv2.resize(masterMask, (x, y))*255).astype(np.uint8)
    cv2.imwrite(dataSource + spec + "/mastermask.tif", masterMask)

    # ap
    '''
    for i in imgs:
        name = nameFromPath(i)
        print(name)
        img = cv2.imread(i) * (np.expand_dims(masterMask, -1)/255).astype(np.uint8)
        cv2.imwrite(dataSource + spec + "/masked_" + name + ".jpg", img)
    '''


    print('test')
