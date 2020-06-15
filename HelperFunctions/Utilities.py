'''
This contains a collection of commonly used functions I have written to perform
menial tasks not directly related to the extraction of relevant information
'''

import numpy as np
import matplotlib.pyplot as plt
from shutil import copy
import os
import cv2
import tifffile as tifi
from glob import glob

# magnification levels of the tif files available
tifLevels = [20, 10, 5, 2.5, 0.625, 0.3125, 0.15625]



def trainingDirs(data, target, label, *args):

    print("\nSTARTING UTILITIES/TRAININGDIRS")

    # This function takes data and copies it from that location into a new directory containing all the training data
    # of the true labels
    # Inputs:   (data), the directory of the data
    #           (target), the location of the directory to save the data (either pre-existing or new)
    #           (label), the label for the data being moved (either into an existing folder or new)
    #           (*args), sub-directories that can be created
    # Outputs:  (), the directory is populated with true data labels to be used

    # create the target tissue folder 
    try:
        os.mkdir(target)
    except:
        pass

    # create the label folder
    try:
        os.mkdir(target + label)
    except:
        pass

    # create subdirectories (optional)
    dir = target + label
    for d in args:
        dirn = dir + "/" + d
        try:
            os.mkdir(dir + "/" + d)
        except OSError:
            print("\nReplacing existing files\n")
        dir = dirn


    # copy the data into created folders
    copy(data, dirn)
    print("STARTING UTILITIES/TRAININGDIRS\n")

def listToTxt(data, dir, **kwargs):

    print("\nSTARTING UTILITIES/LISTTOTXT")

    # Converts a list of information into a txt folder with the inputted name
    # Inputs:   (data), the single list to be saved
    #           (dir), the exact name and path which this list will be saved as
    #           (*args), inputs that appear at the top of the saved file
    # Outputs:  (), txt file saved in directory 

    # txt layout
    '''
    ArgumentNo_[number of addition arguments to read in]
    Arg_1_[some data]
    Arg_2_[some data]
    ...
    Rows_[X number of rows]
    Cols_[Y number of columns]
    [x0y0],[x0y1],[x0y2],[x0yY],...
    [x1y0],[x1y1],[x1y2],[x1yY],...
    [xXyY],...
    EndData
    '''

    f = open(dir, 'w')

    # declar
    f.write("ArgNo_" + str(len(kwargs)) + "\n")

    argK = list()
    argV = list()

    # ensure that the exact directory being specified exists, if not create it
    dirSplit = dir.split("/")
    dirToMake = ""
    for d in range(dir.count("/")):
        dirToMake += str(dirSplit[d] + "/")
        try:
            os.mkdir(dirToMake)
        except:
            pass


    # get optional arguments
    for k in kwargs.keys():
        argK.append(k)

    for v in kwargs.values():
        argV.append(v)

    # write the arguments at the top of the file
    for i in range(len(kwargs)):
        f.write(argK[i] + "_" + argV[i] + "\n")        
    
    f.write("ListEntries_" + str(len(data)) + "\n")
    for i in range(len(data)):
        sample = data[i]
        X, Y = sample.shape
        f.write("Rows_" + str(X) + "\n")
        f.write("Cols_" + str(Y) + "\n")

        for x in range(X):
            for y in range(Y):
                f.write(str(sample[x, y]))
                if (y+1)%Y:
                    f.write(",")
                else:
                    f.write("\n")

    f.write("EndData")

    f.close()
    print("ENDING UTILITIES/LISTTOTXT\n")

def txtToList(dir):

    print("\nSTARTING UTILITIES/TXTTOLIST")

    # Reads in a text file which was saved with the listToTxt function
    # Inputs:   (dir), the name of a single file
    # Outputs:  (dataMain), a list containing the data
    #           (dataArgs), a dictionary containing the argument data

    f = open(dir, 'r')

    # argument numbers
    argNo = int(f.readline().replace("ArgNo_", ""))

    # store the arguments in a dictionary
    args = {}
    for i in range(argNo):
        arg = f.readline().split("_")
        args[arg[0]] = arg[1].replace("\n", "")

    allList = list()

    # use to verify all the information has been collected

    sampleList = list()
    listNo = int(f.readline().replace("ListEntries_", ""))
    for n in range(listNo):
        rows = int(f.readline().replace("Rows_", ""))
        cols = int(f.readline().replace("Cols_", ""))
        storedData = np.zeros([rows, cols])
        for r in range(rows):
            values = f.readline().split(",")
            for c in range(cols):
                storedData[r, c] = int(values[c].replace("\n", ""))

        sampleList.append(storedData)

    print("ENDING UTILITIES/TXTTOLIST\n")
    return(sampleList, args)

def denseMatrixViewer(coords):

    print("\nSTARTING UTILITIES/DENSEMATRIXVIEWER")

    # This function takes in a numpy array of co-ordinates in a global space and turns it into a local sparse matrix 
    # which can be view with matplotlib
    # Inputs:   (coords), the coordinates
    # Outputs:  (), produces a plot to view

    Xmax = int(coords[:, 0].max())
    Xmin = int(coords[:, 0].min())
    Ymax = int(coords[:, 1].max())
    Ymin = int(coords[:, 1].min())

    coordsNorm = coords - [Xmin, Ymin]

    area = np.zeros([Xmax - Xmin + 1, Ymax - Ymin + 1])

    X, Y = coords.shape
    for x in range(X):
        xp, yp = coordsNorm[x, :].astype(int)
        area[xp, yp] = 1

    plt.imshow(area, cmap = 'gray')
    plt.show()

    print("STARTING UTILITIES/DENSEMATRIXVIEWER\n")

def quadrantLines(dir, dirTarget, kernel):
    print("\nSTARTING UTILITIES/QUADRANTLINES")

    # This function adds the quadrant lines onto the tif file
    # Inputs:   (dir), the SPECIFIC name of the original tif image 
    #           (dirTarget), the location to save the image
    #           (kernel), kernel size
    # Outputs:  (), re-saves the image with quadrant lines drawn over it

    imgO = tifi.imread(dir)
    hO, wO, cO = imgO.shape

    # if the image is more than 6 megapixels downsample 
    if hO * wO >= 30 * 10 ** 6:
        aspectRatio = hO/wO
        imgR = cv2.resize(imgO, (6000, int(6000*aspectRatio)))
    else:
        imgR = imgO

    h, w, c = imgR.shape

    # scale the kernel to the downsampled image
    scale = h/hO
    kernelS = int(kernel * scale)

    wid = np.arange(0, w, kernelS)
    hgt = np.arange(0, h, kernelS)

    # draw verticl lines
    for x in wid:
        cv2.line(imgR, (x, 0), (x, h), (0, 0, 0), thickness=1)

    # draw horizontal lines
    for y in hgt:
        cv2.line(imgR, (0, y), (w, y), (0, 0, 0), thickness=1)

    newImg = dirTarget + "mod.jpeg"
    cv2.imwrite(newImg, imgR, [cv2.IMWRITE_JPEG_QUALITY, 80])
    # cv2.imshow('kernel = ' + str(kernel), imgR); cv2.waitKey(0)

    print("ENDING UTILITIES/QUADRANTLINES\n")
    return(newImg, scale)

def maskCover(scale, imgTarget, masks):

    print("\nSTARTING UTILITIES/MASKCOVER")

    # This function adds the masks onto the tif file
    # Inputs:   (scale), the scale factor for the downsampled image
    #           (imgTarget), the image which the masks will be added to
    #           (mask), mask information
    # Outputs:  (), re-saves the image with the mask area as an inverse colour area
    
    # read in both original and target image

    imgM = cv2.imread(imgTarget)

    for mask in masks:

        for x, y in mask:
            # inverse colours of mask areas
            imgM[y, x, :] = 255 - imgM[y, x, :]

    cv2.imwrite(imgTarget, imgM)


    print("ENDING UTILITIES/MASKCOVER\n")

def ndpiLoad(sz, src):

    print("\nSTARTING WSILOAD/NDPILOAD")

    # This function extracts tif files from the raw ndpi files. This uses the 
    # ndpitool from https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/ 
    # Install as necessary. 
    # Input:    (i), magnificataion level to be extracted from the ndpi file
    #           options are 0.15625, 0.3125, 0.625, 1.25, 2.5, 5, 10, 20
    #           (src), file to be extracted with set magnification
    # Output:   (), tif file of set magnification, saved in the same directory
    #           (), the tif files extracted is renamed to be simplified
    #           as just [name]_[magnification].tif

    mag = tifLevels[sz]

    os.system("ndpisplit -x" + str(mag) + " " + str(src))

    nameSRC = src.split("/")[-1].split(".")[0]                    # photo name
    dirSRC = src.split(nameSRC + ".ndpi")[0]                      # folder of photo

    extractedName = glob(src.split(".ndpi")[0] + "*z0.tif")[0]    # NOTE, use of z0 is to prevent 
                                                                # duplication of the same file, however 
                                                                # if there is z shift then this will fail
    os.rename(extractedName, dirSRC + nameSRC + "_" + str(sz) + ".tif")
    
    print("ENDING WSILOAD/NDPILOAD\n")