'''
This contains a collection of commonly used functions I have written to perform
menial tasks not directly related to the extraction of relevant information
'''

import numpy as np

def listToTxt(data, dir, **kwargs):
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

def txtToList(dir):
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

    return(sampleList, args)



def annotationsReader(annotationDirs):

    # This function reads in the annotations extracted by SegmentLoad and creates a mask
    # Input:    (annotationdir), list containing the directories of the annotations
    # Output:   (annotationsDict), dictionary containing a list of numpy arrays of the annotations each
    #               name after the directory location of the .pos file

    # ensure input is as a list 
    if type(annotationDirs) is not list:
        store = annotationDirs
        annotationDirs = list()
        annotationDirs.append(store)

    annotationsDict = {}

    # for every ndpa file
    for a in annotationDirs:

        coords = list()
        f = open(a)
        annotationNo = int(f.readline().replace("NUMBER_OF_ANNOTATIONS=", ""))
        for n in range(annotationNo):
            no = int(f.readline().replace("Annotation:", ""))

            # perform a check to ensure the annotations are being read in correctly
            if n != no:
                sys.exit("The file is being read incorrectly, perhpas it has been interferred with")
            
            points = int(f.readline().replace("Entries:", ""))
            pos = np.zeros([points, 2])

            for p in range(points):
                line = f.readline().replace("\n", "").split(",")
                pos[p, 0], pos[p, 1] = line

            pos = pos.astype(int)
            coords.append(pos)

        annotationsDict[a] = coords
        return(annotationsDict)
          