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

    # This function takes data and copies it from that location into a new directory containing all the training data
    # of the true labels
    # Inputs:   (data), the directory of the data
    #           (target), the location of the directory to save the data (either pre-existing or new)
    #           (label), the label for the data being moved (either into an existing folder or new)
    #           (*args), sub-directories that can be created
    # Outputs:  (), the directory is populated with true data labels to be used

    # create the target tissue folder 
    dir = target + label
    dirMaker(dir)

    # create subdirectories (optional)
    for d in args:
        dirn = dir + "/" + d
        dirMaker(dirn)
        dir = dirn


    # copy the data into created folders
    copy(data, dirn)

def listToTxt(data, dir, **kwargs):

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
    # ensure that the exact directory being specified exists, if not create it
    dirMaker(dir)
    f = open(dir, 'w')

    # declar
    f.write("ArgNo_" + str(len(kwargs)) + "\n")

    argK = list()
    argV = list()


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

def dirMaker(dir):

    # creates directories (including sub-directories)
    # Input:    (dir), path to be made
    # Output:   (), all sub-directories necessary are created

    # ensure that the exact directory being specified exists, if not create it
    dirSplit = dir.split("/")
    dirToMake = ""
    for d in range(dir.count("/")):
        dirToMake += str(dirSplit[d] + "/")
        try:
            os.mkdir(dirToMake)
        except:
            pass

def txtToList(dir):

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

    return(sampleList, args)

def dictToTxt(data, dir, **kwargs):
    # This function saves a dictionary as a txt file
    # Converts a dictinoary into a txt file with the inputted name
    # Inputs:   (data), the single list to be saved
    #           (dir), the exact name and path which this list will be saved as
    #           (*args), inputs that appear at the top of the saved file
    # Outputs:  (), txt file saved in directory 
    
    # ensure that the exact directory being specified exists, if not create it
    dirMaker(dir)


    f = open(dir, 'w')

    # declar
    f.write("ArgNo_" + str(len(kwargs)) + "\n")

    argK = list()
    argV = list()

    # ensure that the exact directory being specified exists, if not create it
    dirMaker(dir)

    # get optional arguments
    for k in kwargs.keys():
        argK.append(k)

    for v in kwargs.values():
        argV.append(v)

    # write the arguments at the top of the file
    for i in range(len(kwargs)):
        f.write(argK[i] + "_" + argV[i] + "\n")        
    
    f.write("Entries:" + str(len(data.keys())) + "\n")
    for n in data.keys():
        f.write(n + ":")
        for v in data[n]:
            f.write(str(v) + " ")
        f.write("\n")
    
    f.close()

def txtToDict(dir):

    # Reads in a text file which was saved with the dictToTxt function
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

    # use to verify all the information has been collected

    sampleDict = {}
    dictNo = int(f.readline().replace("Entries:", ""))
    for n in range(dictNo):
        info = f.readline().split(":")
        key = info[0]
        data = info[1].replace("\n", "")

        # save data as a dictionary
        sampleDict[key] = np.array(data.split(" ")[0:-1]).astype(int)


    return(sampleDict, args)

def denseMatrixViewer(coords, plot = True):

    # This function takes in a numpy array of co-ordinates in a global space and turns it into a local sparse matrix 
    # which can be view with matplotlib
    # Inputs:   (coords), the coordinates
    #           (plot), boolean to control plotting
    # Outputs:  (), produces a plot to view
    #           (area), the array 

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

    if plot:
        plt.imshow(area, cmap = 'gray')
        plt.show()

    shift = (Xmin, Ymin)

    return(area, shift)

def quadrantLines(dir, dirTarget, kernel):

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

    return(newImg, scale)

def maskCover(dir, dirTarget, masks):

    # This function adds add the mask outline to the image
    # Inputs:   (dir), the SPECIFIC name of the tif image the mask was made on
    #           (dirTarget), the location to save the image
    #           (kernel), kernel size
    # Outputs:  (), re-saves the image with mask of the vessels drawn over it

    imgO = tifi.imread(dir)
    imgO = cv2.cvtColor(imgO, cv2.COLOR_RGB2BGR)
    hO, wO, cO = imgO.shape

    # if the image is more than 70 megapixels downsample 
    if hO * wO >= 100 * 10 ** 6:
        aspectRatio = hO/wO
        imgR = cv2.resize(imgO, (10000, int(10000*aspectRatio)))
    else:
        imgR = imgO

    h, w, c = imgR.shape

    # scale the kernel to the downsampled image
    scale = h/hO
    for mask in masks:
        maskN = np.unique((mask * scale).astype(int), axis = 0)
        for x, y in maskN:
            # inverse colours of mask areas
            imgR[y, x, :] = 255 - imgR[y, x, :]

    newImg = dirTarget + ".jpeg"
    cv2.imwrite(newImg, imgR, [cv2.IMWRITE_JPEG_QUALITY, 80])
    # cv2.imshow('kernel = ' + str(kernel), imgR); cv2.waitKey(0)

    return(newImg, scale)

def dataPrepare0(imgDir):

    # this function prepares an array of data for the network
    # MAKE A NEW ONE FOR EACH METHOD OF PROCESSING
    # Inputs:   (imgDir), a list containing the image directories of data
    # Outputs:  (arrayP), the array of data now standardised for processing

    array = np.array([cv2.imread(fname, 0) for fname in imgDir]) 

    # normalise data to be within a range of 0 to 1 for each image
    arrayP = np.array([(a-np.min(a))/(np.max(a) - np.min(a)) for a in array])

    # ensure a 4d tensor is created
    while len(arrayP.shape) < 4:
        arrayP = np.expand_dims(arrayP, -1)

    return(arrayP)

def nameFromPath(paths, n = 2):
    # this function extracts the name from a path
    # Inputs:   (paths), either a list or string of paths 
    #           (n), number of 
    # Outputs:  (names), elist of the names from the paths

    pathStr = False
    if type(paths) is str:
        paths = [paths]
        pathStr = True

    names = list()
    for path in paths:
        # choose the last part of the path and the suffix
        name = path.split("/")[-1].split(".")[0]

        # each '_' indicates a new piece of information in the name
        # if there is at least one underscore then seperate the information
        # defaults to saving only the first piece of underscored information 
        # (this is often the specific sample within the specimen, the next 
        # _ will often be the size of the sample used)
        if len(name.split("_")) > 1:
            name = "_".join(name.split("_")[0:n])
        
        names.append(name)

    # if the path input is a string, it will expect an output of a string as well
    if pathStr:
        names = names[0]

    return names

def regionOfPath(paths, n = 1):
    # this function extracts the user defined position of the (ie what is the path 1 directory up, 2....)
    # Inputs:   (paths), either a list or string of paths 
    #           (n), how many levels of directories up you want to move, defaults to moving up one level
    # Outputs:  (names), elist of the names from the paths

    pathStr = False
    if type(paths) is str:
        paths = [paths]
        pathStr = True

    names = list()
    for path in paths:

        # select the region of the path you are interested in
        name = "/".join(path.split("/")[:-n]) + "/"
        
        names.append(name)

    # if the path input is a string, it will expect an output of a string as well
    if pathStr:
        names = names[0]

    return names

def dictOfDirs(**kwargs):

    # this function takes a list of directories and a key name and puts them 
    # into a dictionary which is named after the sample name
    # Inputs:   (kwargs), needs the file type name and then a list of directories
    # Outputs:  (dictToWrite), a dictinoary which contains dictionaries of the samples
    #               and each of the files associated with it

    dictToWrite = {}

    f = kwargs.keys()

    # get all the names
    names = list()
    for k in kwargs.keys():
        names += nameFromPath(kwargs[k])

    # get all the unique names
    names = np.unique(np.array(names))


    for n in names:
        dictToWrite[n] = {}

    for k in kwargs.keys():
        path = kwargs[k]

        for p in path:
            dictToWrite[nameFromPath(p)][k] = p


    return(dictToWrite)

def dictToArray(d, type = float):

    # converts a dictionary into a nparray. This works for only 1 layered dictionaries into a 2D matrix
    # Input:    (d), dictionary
    #           (type), number type to use
    # Output:   (l), array

    l = np.array(list(d.values())).astype(type)

    return(l)