'''

this creates a mask from the annotations which indicates the 
regions which are the annotate tissue

'''

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from skimage.segmentation import flood_fill
from Utilities import *

# magnification levels of the tif files available
tifLevels = [0.15625, 0.3125, 0.625, 1.25, 2.5, 5, 10, 20]


def maskCreator(size, specimenSRC, segmentName = ''):

    print("\nSTARTING MASKMAKER/MASKCREATOR\n")

    # TODO: Seperate into two functions the mask building and target tissue identification

    # This function takes the manual annotations and turns them into a dense matrix which 
    # has identified all the pixel which the annotations encompass at the user chosen scale
    # Inputs:   (size), the user chosen scale which refers to the zoom level of the tif file extracted
    #           (specimenAnnotations), list of the annotations as loaded by annotationsReaders
    # Outputs:  (maskDirs), directories to the files which contains all pixel locations for the
    #           annotations of the  highest resolultion tif file 

    specimenDir = glob(specimenSRC + segmentName + "*.pos")

    scale = tifLevels[size] / max(tifLevels)

    for specimen in specimenDir:

        # open the manual annotations file
        annoSpec, argsDict = txtToList(specimen)
        
        # find the encomappsed areas of each annotations
        denseAnnotations = maskFinder(annoSpec, scale)
        
        # of the identified areas, find the roi between overlapping ares
        targetTissue = roiFinder(denseAnnotations)

        # save the mask as a txt file of all the pixel co-ordinates of the target tissue
        listToTxt(targetTissue, str(specimenDir[0] + "_size_" + str(size) + ".mask"))
    
    print("ENDING MASKMAKER/MASKCREATOR\n")

def coordMatch(array1, array2):

    # This function removes the values of the smaller array from the larger array to identify the roi.
    # Inputs:   (largerArray), numpy.array which contains vector values. MUST be the larger dimension 
    #           (smallerArray), numpy.array which contains vector values. MUST be the smaller dimension 
    # Outputs:  (roi), essentailly the areas which are unique to the larger array

    # get each array entry repeated once and count its occurence
    uniq, count = np.unique(np.concatenate([array1, array2]), return_counts=True, axis = 0)

    # only pass through the entires which occur once --> removes the points of overlap
    roi = uniq[np.where(count == 1)]

    return(roi)

def maskFinder(annoSpec, scale):

    # This function takes the manual annotations and turns them into a mask which 
    # encompasses the inner area
    # Inputs:   (annoSpec), the annotations for a single specimen
    #           (scale), the scaling factor to apply between the raw data and the chosen 
    #               tif size file to analyse
    # Outputs:  (denseAnnotations), a list containing the global co-ordinate position of ONLY
    #               the true values of the mask

    # --- perform mask building per annotation
    denseAnnotations = list()

    for n in range(len(annoSpec)):

        print("\nAnnotation " + str(n) + "/" + str(len(annoSpec)))

        # process per annotation
        annotation = annoSpec[n]

        # scale the annotation
        annotationScaled = annotation * scale

        # shift the annotation to a (0, 0) origin 
        xminO = int(annotationScaled[:, 0].min())
        yminO = int(annotationScaled[:, 1].min())
        annotationU, posU = np.unique((annotationScaled - [xminO, yminO]).astype(int), axis = 0, return_index = True)           

        # get the properties of the new scaled grid
        xmax = int(annotationScaled[:, 0].max())
        ymax = int(annotationScaled[:, 1].max())
        xmin = int(annotationScaled[:, 0].min())
        ymin = int(annotationScaled[:, 1].min())

        # after np.unique the entries are ordered by value which breaks the interpolation step as it works on the 
        # assumption of sequential points. So re-order the now downsampled hand annotations
        posM, annotationM = zip(*sorted(zip(posU, annotationU)))
        annotationM = np.array(annotationM)

        # scaled grid
        grid = np.zeros([xmax-xmin+3, ymax-ymin+3])     # NOTE: added 3 so that the entire bounds of the co-ordinates are stored... 

        # --- Interpolate between each annotated point creating a continuous border of the annotation
        x_p, y_p = annotationM[-1]      # initialise with the last point for continuity 
        for x, y in annotationM:

            # number of points needed for a continuous border
            num = int(sum(np.abs(np.subtract((x, y),(x_p, y_p)))))

            # generating the x and y region points
            x_r = np.linspace(x, x_p, num+1)
            y_r = np.linspace(y, y_p, num+1)

            # perform a "blurring" operation to ensure the outline of the mask is solid
            extrapolated21 = np.stack([x_r+1, y_r], axis = 1)
            extrapolated01 = np.stack([x_r, y_r+1], axis = 1)
            extrapolated11 = np.stack([x_r+1, y_r], axis = 1)
            extrapolated10 = np.stack([x_r+1, y_r+2], axis = 1)
            extrapolated12 = np.stack([x_r+2, y_r+2], axis = 1)

            extrapolated = np.concatenate([extrapolated21,
                                            extrapolated01,
                                            extrapolated11,
                                            extrapolated10,
                                            extrapolated12,]).astype(int)

            # set these points as true
            for x_e, y_e in extrapolated:
                grid[x_e, y_e] = 1
            
            # save previous point to i
            x_p, y_p = x, y
            
            #print("x_r: " + str(x_r))
            # print("y_r: " + str(y_r) + "\n")
            # plt.imshow(grid); plt.show()

        # --- fill in the outline with booleans
        # perform a preliminary horizontal search
        for x in range(grid.shape[0]):
            edges = np.zeros([2, 2])
            startFound = False  
            edgeFound = False 

            # for every y point
            for y in range(1, grid.shape[1]-1):

                # check if there is a raising edge point on this row
                if (grid[x, y+1] == 0) & (grid[x, y] == 1):
                    edges[0, :] = [x, y]
                    startFound = True
                    
                # check if there is a falling edge point on this row
                elif (grid[x, y-1] == 0) & (grid[x, y] == 1) & startFound:
                    edges[1, :] = [x, y]
                    
                    # find the middle of the edge --> assumed this will be a pixel within the roi
                    roi0 = tuple(np.mean(edges, axis = 0).astype(int))
                    edgeFound = True

                    
                    # AT THIS POINT WE HAVE POTENTIALLY FOUND AN EDGE. HOWEVER DUE
                    # TO HUMAN BS WHEN DRAWING THE ANNOTATIONS WE NEED TO CHECK OVER 
                    # THIS EDGE WITH ANOTHER METHOD --> CONFIRM THE EXISTENCE OF THE 
                    # EDGE IN THE VERTICAL PLANE AS WELL
                    # perform a vertical search to confirm roi
                    startFound = False
                    for x in range(1, grid.shape[0]-1):
                        yroi0 = roi0[1]
                        # check if there is a raising edge point on this column
                        if (grid[x+1, yroi0] == 0) & (grid[x, yroi0] == 1):
                            edges[0, :] = [x, yroi0]
                            startFound = True
                            
                        # check if there is a falling edge point on this column
                        elif (grid[x-1, yroi0] == 0) & (grid[x, yroi0] == 1) & startFound:
                            edges[1, :] = [x, yroi0]
                            
                            # find the middle of the edge --> assumed this will be a pixel within the roi
                            roi = tuple(np.mean(edges, axis = 0).astype(int))
                            break
                    

            if edgeFound:
                break

        # floodfill in the entirety of this encompassed area (flood fill) 
        try:
            gridN = flood_fill(grid, roi, 1)
        except:
            gridN = grid
            print("     Flood not performed on annotaiton " + str(n))

        # --- save the mask identified in a dense form and re-position into the SCALED global space
        denseGrid = np.stack(np.where(gridN == 1), axis = 1) + [xmin, ymin]
        # denseMatrixViewer(denseGrid)
        # plt.imshow(grid); plt.show()
        denseAnnotations.append(denseGrid)
    
    # return a list which contains all the true pixel positions of the circled areas
    return(denseAnnotations)

def roiFinder(denseAnnotations):

    # This function takes the dense list of mask value positions from maskFinder and 
    # identifies which pair of these masks overlap and then identifies the roi target
    # tissue between the areas
    # Inputs:   (annoSpec), the dense list of co-ords for the mask from maskFinder
    # Outputs:  (targetTissue), dense list of co-ords for ONLY the areas between the 
    #               annotated masks
    
    # perform a boundary search to investigate which masks are within which sections
    annoID = np.arange(len(denseAnnotations))
    noAnnos = len(annoID)

    targetTissue = list()

    # --- identify the target tissue between paired annotations
    while len(annoID) > 0:
        s = annoID[0]
        print("\nAnnotation " + str(s) + "/" + str(noAnnos))
        annoID = np.delete(annoID, np.where(annoID == s)[0][0])      # as you find matches remove from array search
        
        # annotation to perform search
        search = denseAnnotations[s]
        sXmax = search[:, 0].max()
        sXmin = search[:, 0].min()
        sYmax = search[:, 1].max()
        sYmin = search[:, 1].min()

        # need to check if search is either the inside or outside mask
        for m in annoID:
            match = denseAnnotations[m]
            mXmax = match[:, 0].max()
            mXmin = match[:, 0].min()
            mYmax = match[:, 1].max()
            mYmin = match[:, 1].min()

            # logic operators to check if there is an encompassed area
            mInsideBool = (mXmax <= sXmax) & (mYmax <= sYmax) & (mXmin >= sXmin) & (mYmin >= sYmin)
            mOutsideBool = (mXmax >= sXmax) & (mYmax >= sYmax) & (mXmin <= sXmin) & (mYmin <= sYmin)

            # check if match is inside the search
            if mInsideBool or mOutsideBool:
                # print("inside: " + str(mInsideBool) + " outside: " + str(mOutsideBool))
                annoID = np.delete(annoID, np.where(annoID == m)[0][0])
                annotatedROI = coordMatch(search, match)
                print("anno " + str(s) + " matches " + str(m))
                break
            
            # if no match is found assumed that there is no annotated centre
            else:
                annotatedROI = search
                print("     anno " + str(s) + " is not matched with anno " + str(m))

        # view the roi
        # denseMatrixViewer(annotatedROI)

        targetTissue.append(annotatedROI)

    return(targetTissue)

# data directory
data = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'
size = 2
kernel = 100
name = 'testWSI1'
# Extract the manual co-ordinates of the annotated tissue
# SegmentLoad.readndpa(data)

# create the masks of the annotationes
maskCreator(size, data, name)
