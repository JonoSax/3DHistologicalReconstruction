'''
This script loads in the NDPA files which have the manually segmented 
sections and correlates these positions with array position on the WSI 
that has just been loaded
'''

import numpy as np
from glob import glob
import openslide 

def readndpa(annotationsSRC, speciment = ''):

    # This function reads a NDPA file and extracts the co-ordinate of the hand drawn points and converts
    # them into their pixel location equivalents
    # Input:    (annotationsSRC), Directory for the ndpi files
    #           (annotationName), the specimen name/s to specifically investigate (optional, if not
    #                               chosen then will investigate the entire directory)
    # Output:   A list containing numpy arrays which have the recorded positions of the drawn points on the slide. Each
    #           entry to the list refers to a section drawn

    # get the directories of all the specimes of interest
    names = glob(annotationsSRC + specimen + "*.ndpa")

    posAll = list()

    for name in names:

        file = open(name)

        # find the number of sections identified in slice --> Used to validate that the search is completed
        sections = open(name).read().count("ndpviewstate id=")

        # extract all info from text file line by line
        doco = list(file.readlines())

        # declare list and array
        posA = list()
        pos = np.empty([0, 2])
        l = 0

        # NOTE the skips in this loop are hard coded because as far as i can tell
        # they are enitrely predictable 
        while l < len(doco):

                if "<point>" in doco[l]:

                    x = doco[l+1]
                    y = doco[l+2]

                    x = x.replace("<x>", ""); x = int(x.replace("</x>\n", ""))
                    y = y.replace("<y>", ""); y = int(y.replace("</y>\n", ""))

                    pos = np.vstack((pos, [x, y]))

                    l += 4  # jump 4 line to the next set of co-ordinates

                elif "</pointlist>" in doco[l]:
                    posA.append(pos.astype(int))
                    pos = np.empty([0, 2])

                    l+=18   # jump 18 lines to the next section of co-ordinates

                else:
                    l+=1    # if no info found, just iterate through

        print(str(len(posA)/sections*100)+"% of section " + name + " found")
        posAll.append(posA)

    # get the ndpi properties of all the specimes of interest
    xShift, yShift, xResolution, yResolution = normaliseNDPA(names)

    for name in names:


    return(posAll)

def normaliseNDPA(data):

    # This functions reads the properties of the ndpi files and extracts the properties 
    # of the file.
    # Inputs:   (data), directory containing all the ndpi files of interest
    # Output:   (xShift), the x position on the image (in nm) of the co-ordinate systems origin
    #           (yShift), the y position on the image (in nm) of the co-ordinate systems origin
    #           (xResolution), the x scalar of unit measurement (nm in this case) to pixel
    #           (yResolution), the x scalar of unit measurement (nm in this case) to pixel

    # data = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'
    # data = '/Volumes/Storage/H653A_11.3 new/'

    slicesDir = glob(str(data+"*.ndpi"))

    xShift = list()
    yShift = list()
    xResolution = list()
    yResolution = list()

    for slicedir in slicesDir:
        slideProperties = openslide.OpenSlide(slicedir).properties
        xShift.append(int(slideProperties['hamamatsu.XOffsetFromSlideCentre']))
        yShift.append(int(slideProperties['hamamatsu.YOffsetFromSlideCentre']))
        xResolution.append(int(slideProperties['tiff.XResolution']))
        yResolution.append(int(slideProperties['tiff.YResolution']))

    return (xShift, yShift, xResolution, yResolution)