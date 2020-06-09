'''
This contains a collection of commonly used functions I have written to perform
menial tasks not directly related to the extraction of relevant information
'''

import numpy as np
import matplotlib.pyplot as plt
from shutil import copy
import os

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
    # Inputs:   (data), the list to be saved
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
    # Inputs:   (dir), the name/s of the file/s
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

    plt.imshow(area)
    plt.show()

    print("STARTING UTILITIES/DENSEMATRIXVIEWER\n")

def quadrantLines(dir, kernel):
    print("\nSTARTING UTILITIES/QUADRANTLINES")

    # This function adds the quadrant lines onto the tif file
    # Inputs:   (dir), the SPECIFIC name of the image 
    #           (kernel), kernel size
    # Outputs:  (), re-saves the image with quadrant lines drawn over it

    print("ENDING UTILITIES/QUADRANTLINES\n")

def maskCover(dir, mask):
    print("\nSTARTING UTILITIES/QUADRANTLINES")

    # This function adds the masks onto the tif file
    # Inputs:   (dir), the SPECIFIC name of the image 
    #           (mask), mask information
    # Outputs:  (), re-saves the image with the mask area as an inverse colour area

    print("ENDING UTILITIES/QUADRANTLINES\n")
