'''
This script loads in the NDPA files which have the manually segmented 
sections and correlates these positions with array position on the WSI 
that has just been loaded
'''

import numpy as np
from glob import glob
import openslide 

# simple dictionary used to convert to SI unit between the different scales in the ndpi and ndpa files
# It is using millimeters as the base unit (ie multiple of 1)
unitDict = {
    'nanometers':1*10**-6, 
    'micrometers':1*10**-3,
    'millimeters':1*10**-0,
    'centimeters':1*10**1,
    'nanometer':1*10**-6, 
    'micrometer':1*10**-3,
    'millimeter':1*10**-0,
    'centimeter':1*10**1
}

def savendpa(coordinates):

    # This function takes the extracted ndpa co-ordinates and saves them in a text file so that 
    # it can be called by WSILoaded.py
    # Input:    (coordinates), list containing all the processed co-ordinates 
    # Output:   (), files saved per specimen containing the co-ordinates

    pass

def readndpa(annotationsSRC, specimen = ''):

    # This function reads a NDPA file and extracts the co-ordinate of the hand drawn points and converts
    # them into their pixel location equivalents
    # Input:    (annotationsSRC), Directory for the ndpi files
    #           (annotationName), the specimen name/s to specifically investigate (optional, if not
    #                               chosen then will investigate the entire directory)
    # Output:   A list containing numpy arrays which have the recorded positions of the drawn points on the slide. Each
    #           entry to the list refers to a section drawn

    # get the directories of all the specimes of interest
    ndpaNames = glob(annotationsSRC + specimen + "*.ndpa")
    ndpiNames = glob(annotationsSRC + specimen + "*.ndpi")

    posAll = list()

    # go through each ndpa file of interest and extract the RAW co-ordinates into a list.
    # this list is indexed as follows:
    # posAll --> for all the co-ordinates
    #       posA --> for a single ndpa file
    #               co-ordinates0 --> for a single annotation
    #               co-ordinates1
    #               ....
    #       ...
    for name in ndpaNames:

        file = open(name)

        # find the number of sections identified in slice --> Used to validate that the search is completed
        sections = open(name).read().count("ndpviewstate id=")

        # extract all info from text file line by line
        doco = list(file.readlines())

        # declare list and array
        posA = list()
        pos = np.empty([0, 2])
        l = 0

        print("\nThis is the specimen: " + str(name) + "\n")

        # NOTE the skips in this loop are hard coded because as far as i can tell
        # they are enitrely predictable --> ONLY WORKS IF THERE ARE ONLY FREE HAND ANNOTATIONS
        while l < len(doco):

            # get the unit co-ordinates
            if "<coordformat>" in doco[l]:

                unit = doco[l]

                unitStr = unit.replace("<coordformat>", "").replace("</coordformat>\n", "").replace(" ", "")

                unit = unitDict[unitStr]

                l+=12

            elif "<point>" in doco[l]:                # indicates an annotated point

                x = doco[l+1]
                y = doco[l+2]

                x = x.replace("<x>", ""); x = int(x.replace("</x>\n", "")) * unit       # convert the positions into mm
                y = y.replace("<y>", ""); y = int(y.replace("</y>\n", "")) * unit       # convert the positions into mm
                pos = np.vstack((pos, [x, y])) 
                l += 4  # jump 4 line to the next set of co-ordinates

            elif "</pointlist>" in doco[l]:
                posA.append(pos)        #end of the annotation detected 
                pos = np.empty([0, 2])

                l+=6   # jump 18 lines to the next section of co-ordinates

            else:
                l+=1    # if no info found, just iterate through

        print(str(len(posA)/sections*100)+"% of section " + name + " found")
        posAll.append(posA)

    # get the ndpi properties of all the specimes of interest
    xShift, yShift, xRes, yRes, xDim, yDim = normaliseNDPA(ndpiNames)

    # apply the necessary transformations to extracted co-ordinates to convert to pixel represnetations
    # with the origin in the top left corner of the image
    for spec in range(len(ndpaNames)):

        # NOTE: These should all now be in mm..... work with that

        centreShift = np.hstack([xShift[spec], yShift[spec]])
        topLeftShift = np.hstack([xDim[spec]/2, yDim[spec]/2])
        scale = np.hstack([xRes[spec], yRes[spec]])
        posSpec = posAll[spec]

        for i in range(len(posAll[spec])):
            stack = ((posSpec[i] - centreShift + topLeftShift ) * scale).astype(int)
            posAll[spec][i] = np.unique(stack, axis=0)          # remove any duplicate co-ordinates

        

    return(posAll)

def normaliseNDPA(slicesDir):

    # This functions reads the properties of the ndpi files and extracts the properties 
    # of the file.
    # Inputs:   (data), list of directory/ies containing all the ndpi files of interest
    # Output:   (xShift),       # x shift of the slide scanner centre from image centre in physical units (nm)
    #           (yShift),       # y shift of the slide scanner centre from image centre in physical units (nm)
    #           (xResolution),  # x scale of physical units to pixels
    #           (yResolution),  # y scale of physical units to pixels
    #           (xDim),         # width (pixels) of highest resolution of tif stored
    #           (yDim),         # height (pixels) of highest resolution of tif stored 

    # data = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'
    # data = '/Volumes/Storage/H653A_11.3 new/'

    xShift = list()
    yShift = list()
    xResolution = list()
    yResolution = list()
    xDim = list()
    yDim = list()


    for slicedir in slicesDir:
        slideProperties = openslide.OpenSlide(slicedir).properties                          # all ndpi properties
        unit = unitDict[slideProperties['tiff.ResolutionUnit']]                             # get the unit multiplier so that all units are in mm
        xShift.append(int(slideProperties['hamamatsu.XOffsetFromSlideCentre']) * 10**-6)    # assumed nm, converted to mm
        yShift.append(int(slideProperties['hamamatsu.YOffsetFromSlideCentre']) * 10**-6)    # assumed nm, converted to mm
        xRes = int(slideProperties['tiff.XResolution']) / unit                              # scale is unit dependent
        yRes = int(slideProperties['tiff.YResolution']) / unit                              # scale is unit dependent                       
        xDim.append(int(slideProperties['openslide.level[0].width'])/xRes)                  # assumed always in pixels   
        yDim.append(int(slideProperties['openslide.level[0].height'])/yRes)                 # assumed always in pixels

        xResolution.append(xRes)
        yResolution.append(yRes)  

    return (xShift, yShift, xResolution, yResolution, xDim, yDim)