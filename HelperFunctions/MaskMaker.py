'''

this creates a mask from the annotations which indicates the 
regions which are the annotate tissue

'''

import os
import numpy as np
from glob import glob


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
    # Output:   ()

    annotationsDir = glob(segmentSRC + segmentName + "*.pos")
    annotations = annotationsReader(annotationsDir)

    for a in annotationsDir:
        specimen = annotations[a]

        for n in range(len(specimen)):

            # process per annotation
            annotation = specimen[n]

            # perform interpolation from a normalised co-ordinate system
            xmin = annotation[:, 0].min()
            ymin = annotation[:, 1].min()
            annotationM = annotation - [xmin, ymin]

            # create a grid array for interpolation
            xmax = annotation[:, 0].max()
            ymax = annotation[:, 1].max()
            grid = np.zeros([xmax-xmin, ymax-ymin])

            # identify all the points between each annotated point 
            x_p, y_p = annotationM[0]
            for x, y in annotationM[1:-1]:
                
                m = (y - y_p) / (x - x_p)
                if x > x_p:
                    g = -1
                else:
                    g = 1
                x_r = np.arange(x, x_p, g)

                grid[x, y] = 1
                x_p, y_p = x, y

            # fill in the entirety of this encompassed area

            # save the complete pixel locations of the annotations

            pass

    pass


segmentSRC = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'

segmentedAreas(100, segmentSRC)