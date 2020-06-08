'''

this creates a mask from the annotations which indicates the 
regions which are the annotate tissue

'''

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from skimage.segmentation import flood_fill
from .Utilities import *


def segmentedAreas(segmentSRC, segmentName = ''):

    # This function create a mask of the annotations which encompass the target tissue for the target resolution and kernel size.
    # Input:    (kernel), Square kernel size (pixels)
    #           (imageSRC), data source directory
    #           (imageName), OPTIONAL to specify which samples to process
    # Output:   (), saves a mask of the images identifying the annotated tissue

    specimenDir = glob(segmentSRC + segmentName + "*.pos")

    # get the masks of each of the annotations
    specMask = maskCreator(specimenDir)



def maskCreator(specimenDir):

    # This function takes the manual annotations and turn them into a dense matrix which 
    # has identified all the pixel which the annotations encompass
    # Inputs:   (specimenAnnotations), list of the annotations as loaded by annotationsReaders
    # Outputs:  (maskDirs), directories to the files which contains all pixel locations for the
    #           annotations of the  highest resolultion tif file 

    for specimen in specimenDir:

        # open the manual annotations file
        annoSpec, argsDict = txtToList(specimen)

        denseAnnotations = list()
            
        # --- perform mask building per annotation
        for n in range(len(annoSpec)):
            print("annotation no: " + str(n))

            # process per annotation
            annotation = annoSpec[n]

            # shift the annotation to a (0, 0) origin
            xmin = int(annotation[:, 0].min())
            ymin = int(annotation[:, 1].min())
            annotationM = annotation - [xmin, ymin]

            # create a grid array for interpolation for the annotation of interest
            xmax = int(annotation[:, 0].max())
            ymax = int(annotation[:, 1].max())
            grid = np.zeros([xmax-xmin+3, ymax-ymin+3])     # NOTE: added one so that the entire bounds of the co-ordinates are stored... 

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
            gridN = flood_fill(grid, roi, 1)

            # --- save the mask identified in a dense form and re-position into global space
            denseGrid = np.stack(np.where(gridN == 1), axis = 1) + [xmin, ymin]
            denseAnnotations.append(denseGrid)
        
        # perform a boundary search to investigate which masks are within which sections
        annoID = np.arange(len(denseAnnotations))
        targetTissue = list()

        # --- identify the target tissue between paired annotations
        while len(annoID) > 0:
            s = annoID[0]
            print("\nAnnotation " + str(s))
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
                    break
                
                # if no match is found assumed that there is no annotated centre
                else:
                    annotatedROI = search
                    print("there is no identified centre")

            # view the roi
            denseMatrixViewer(annotatedROI)

            targetTissue.append(annotatedROI)

        # save the mask as a txt file of all the pixel co-ordinates of the target tissue
        listToTxt(targetTissue, str(specimenDir[0] + ".mask"))
            
        # save the complete mask of the specimen as a txt file

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