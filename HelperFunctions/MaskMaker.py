'''

this creates a mask from the annotations which indicates the 
regions which are the annotate tissue

'''

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from skimage.segmentation import flood_fill



def annotationsReader(annotationDirs):

    # This function reads in the annotations extracted by SegmentLoad
    # Input:    (annotationdir), list containing the directories of the annotations
    # Output:   (annotations), dictionary containing a list of numpy arrays of the annotations. each
    #               name after the directory location of the .pos file

    annotations = {}


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

        annotations[a] = coords
        return(annotations)
                
def segmentedAreas(kernel, segmentSRC, segmentName = ''):

    # This function create a mask of the annotations which encompass the target tissue.
    # Input:    (kernel), Square kernel size (pixels)
    #           (imageSRC), data source directory
    #           (imageName), OPTIONAL to specify which samples to process
    # Output:   (), saves a mask of the images identifying the annotated tissue

    specimenDir = glob(segmentSRC + segmentName + "*.pos")
    specimenAnnotations = annotationsReader(specimenDir)

    # create a mask for all the specimens
    for a in specimenAnnotations:
        print("annotation analysed: " + str(a))
        annoSpec = specimenAnnotations[a]

        # perform mask building per annotation
        for n in range(len(annoSpec)):
            print("annotation no: " + str(n))

            # process per annotation
            annotation = annoSpec[n]

            # shift the annotation to a (0, 0) origin
            xmin = annotation[:, 0].min()
            ymin = annotation[:, 1].min()
            annotationM = annotation - [xmin, ymin]

            # create a grid array for interpolation for the annotation of interest
            xmax = annotation[:, 0].max()
            ymax = annotation[:, 1].max()
            grid = np.zeros([xmax-xmin+3, ymax-ymin+3])     # NOTE: added one so that the entire bounds of the co-ordinates are stored... 

            # interpolate between each annotated point creating a continuous border of the annotation
            x_p, y_p = annotationM[-1]
            for x, y in annotationM:

                # number of points needed for a continuous border
                num = sum(np.abs(np.subtract((x, y),(x_p, y_p))))

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

            edges = np.zeros([2, 2])

            # perform a preliminary horizontal search
            for x in range(grid.shape[0]):
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

                        '''
                        # AT THIS POINT WE HAVE POTENTIALLY FOUND AN EDGE. HOWEVER DUE
                        # TO HUMAN BS WHEN DRAWING THE ANNOTATIONS WE NEED TO CHECK OVER 
                        # THIS EDGE WITH ANOTHER METHOD --> CONFIRM THE EXISTENCE OF THE 
                        # EDGE IN THE VERTICAL PLANE AS WELL
                        # perform a vertical search to confirm roi
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
                        '''

                if edgeFound:
                    break

            # fill in the entirety of this encompassed area (flood fill)
            gridN = flood_fill(grid, roi0, 1)
            plt.imshow(gridN); plt.show()            
            plt.imshow(grid); plt.show()

            # add this annotation to the mask

            pass


        # perform global subtractions to remove the non-ROIs
        
        # save the complete mask of the specimen as a txt file

        pass

    pass


segmentSRC = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'

segmentedAreas(100, segmentSRC)