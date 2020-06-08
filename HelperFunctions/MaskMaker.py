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


def segmentedAreas(kernel, segmentSRC, segmentName = ''):

    # This function create a mask of the annotations which encompass the target tissue.
    # Input:    (kernel), Square kernel size (pixels)
    #           (imageSRC), data source directory
    #           (imageName), OPTIONAL to specify which samples to process
    # Output:   (), saves a mask of the images identifying the annotated tissue

    specimenDir = glob(segmentSRC + segmentName + "*.pos")

    # get the masks of each of the annotations
    specMask = maskCreator(specimenDir)

    pass

def maskCreator(specimenDir):

    # This function takes the manual annotations and turn them into a dense matrix which 
    # has identified all the pixel which the annotations encompass
    # Inputs:   (specimenAnnotations), list of the annotations as loaded by annotationsReaders
    # Outputs:  (maskDirs), directories to the files which contains all pixel locations for the
    #           annotations of the  highest resolultion tif file 

    for specimen in specimenDir:

        annoSpec, argsDict = txtToList(specimen)

        denseAnnotations = list()
            

        # perform mask building per annotation
        # for n in range(len(annoSpec)):
        for n in range(6):
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
        
        # --- all annotation mask created an added to list denseAnnotations

        # perform a boundary search to investigate which masks are within which sections
        annoID = np.arange(len(denseAnnotations))
        targetTissue = list()

        for s in annoID:
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
                    print("inside: " + str(mInsideBool) + " outside: " + str(mOutsideBool))
                    annotatedROI = coordMatch(search, match)
                    break
                
                # if no match is found assumed that there is no annotated centre
                else:
                    # 
                    annotatedROI = search
                    print("there is no identified centre")
            annoID = np.delete(annoID, np.where(annoID == m)[0][0])

            # denseMatrixViewer(annotatedROI)
            targetTissue.append(annotatedROI)

        # save the mask as a txt file of pixel co-ordinates
        listToTxt(annotationsMask, str(specimenDir[0] + ".mask"))

    pass
            
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



search = np.array([[22894,  6368],
       [22894,  6369],
       [22894,  6370],
       [22894,  6371],
       [22894,  6372],
       [22894,  6373],
       [22894,  6374],
       [22894,  6375],
       [22894,  6376],
       [22894,  6377],
       [22894,  6378],
       [22894,  6379],
       [22894,  6380],
       [22894,  6381],
       [22894,  6382],
       [22894,  6383],
       [22894,  6384],
       [22894,  6385],
       [22894,  6386],
       [22894,  6387],
       [22894,  6388],
       [22894,  6389],
       [22894,  6390],
       [22894,  6391],
       [22894,  6392],
       [22894,  6393],
       [22894,  6394],
       [22895,  6353],
       [22895,  6354],
       [22895,  6355],
       [22895,  6356],
       [22895,  6357],
       [22895,  6358],
       [22895,  6359],
       [22895,  6360],
       [22895,  6361],
       [22895,  6362],
       [22895,  6363],
       [22895,  6364],
       [22895,  6365],
       [22895,  6366],
       [22895,  6367],
       [22895,  6368],
       [22895,  6369],
       [22895,  6370],
       [22895,  6371],
       [22895,  6372],
       [22895,  6373],
       [22895,  6374],
       [22895,  6375],
       [22895,  6376],
       [22895,  6377],
       [22895,  6378],
       [22895,  6379],
       [22895,  6380],
       [22895,  6381],
       [22895,  6382],
       [22895,  6383],
       [22895,  6384],
       [22895,  6385],
       [22895,  6386],
       [22895,  6387],
       [22895,  6388],
       [22895,  6389],
       [22895,  6390],
       [22895,  6391],
       [22895,  6392],
       [22895,  6393],
       [22895,  6394],
       [22895,  6395],
       [22895,  6396],
       [22895,  6397],
       [22895,  6398],
       [22895,  6399],
       [22895,  6400],
       [22895,  6401],
       [22895,  6402],
       [22895,  6403],
       [22895,  6404],
       [22895,  6405],
       [22895,  6406],
       [22895,  6407],
       [22895,  6408],
       [22895,  6409],
       [22895,  6410],
       [22895,  6411],
       [22895,  6412],
       [22895,  6413],
       [22895,  6414],
       [22895,  6415],
       [22895,  6416],
       [22895,  6417],
       [22895,  6418],
       [22895,  6419],
       [22895,  6420],
       [22895,  6421],
       [22895,  6422],
       [22895,  6423],
       [22895,  6424],
       [22895,  6425]])

'''
match = np.array([[22820,  6297],
       [22820,  6298],
       [22820,  6299],
       [22820,  6300],
       [22820,  6301],
       [22820,  6302],
       [22820,  6303],
       [22820,  6304],
       [22820,  6305],
       [22820,  6306],
       [22820,  6307],
       [22820,  6308],
       [22820,  6309],
       [22820,  6310],
       [22820,  6311],
       [22820,  6312],
       [22821,  6272],
       [22821,  6273],
       [22821,  6274],
       [22821,  6275],
       [22821,  6276],
       [22821,  6277],
       [22821,  6278],
       [22821,  6279],
       [22821,  6280],
       [22821,  6281],
       [22821,  6282],
       [22821,  6283],
       [22821,  6284],
       [22821,  6285],
       [22821,  6286],
       [22821,  6287],
       [22821,  6288],
       [22821,  6289],
       [22821,  6290],
       [22821,  6291],
       [22821,  6292],
       [22821,  6293],
       [22821,  6294],
       [22821,  6295],
       [22821,  6296],
       [22821,  6297],
       [22821,  6298],
       [22821,  6299],
       [22821,  6300],
       [22821,  6301],
       [22821,  6302],
       [22821,  6303],
       [22821,  6304],
       [22821,  6305]])
'''

match = search[70:100]

# coordMatch(search, match)


segmentSRC = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'

segmentedAreas(100, segmentSRC)