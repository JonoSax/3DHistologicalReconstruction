'''
This script loads in the NDPA files which have the manually segmented 
sections and correlates these positions with the pixel position on the WSI 
that has just been loaded. 

It then saves these co-ordinates as a .pos file 

'''

import numpy as np
from glob import glob
import openslide 
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
from .Utilities import *

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

def readndpa(annotationsSRC, specimen = ''):

    print("\nSTARTING SEGMENTLOAD/READNDPA\n")

    # This function reads a NDPA file and extracts the co-ordinate of the hand drawn points and converts
    # them into their pixel location equivalents
    # Input:    (annotationsSRC), Directory for the ndpi files
    #           (annotationName), the specimen name/s to specifically investigate (optional, if not
    #                               chosen then will investigate the entire directory)
    # Output:   A list containing numpy arrays which have the recorded positions of the drawn points on the slide. Each
    #           entry to the list refers to a section drawn

    # get the directories of all the specimes of interest
    ndpaNames = glob(annotationsSRC + specimen + "*.ndpa")
    ndpiNames = glob(annotationsSRC + specimen + "*.ndpi")      # need this for the properties of the image

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

        # ndpa file
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
        # they are enitrely predictable --> ONLY WORKS IF THERE ARE ONLY FREE HAND ANNOTATIONS
        while l < len(doco):

            # get the unit used in the co-ordinates
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
        file.close()
        posAll.append(posA)

    # NOTE: Probably could combine the loop which reads in the information per specimen and the loop
    # which saves it in the .pos format 

    # get the ndpi properties of all the specimes of interest
    xShift, yShift, xRes, yRes, xDim, yDim = normaliseNDPA(ndpiNames)

    # apply the necessary transformations to extracted co-ordinates to convert to pixel represnetations
    # with the origin in the top left corner of the image and save them as a txt file for all npda files
    for spec in range(len(ndpaNames)):

        # create txt file which contains these co-ordinates
        # f = open(str(ndpaNames[spec]) + ".pos", 'w')

        stacks = list()

        # create the file name
        dirSave = str(ndpaNames[spec] + ".pos")

        # npdi properties
        centreShift = np.hstack([xShift[spec], yShift[spec]])
        topLeftShift = np.hstack([xDim[spec]/(2 * xRes[spec]), yDim[spec]/(2 * yRes[spec])])
        scale = np.hstack([xRes[spec], yRes[spec]])
        posSpec = posAll[spec]
        # f.write("NUMBER_OF_ANNOTATIONS=" + str(len(posSpec)) + "\n")
        for i in range(len(posAll[spec])):
            
            # co-ordinate transformation
            stack = ((posSpec[i] - centreShift + topLeftShift ) * scale).astype(int)
            
            # save into a list
            stacks.append(stack)

        # save the entire list as a txt file per utilities saving structure
        listToTxt(stacks, dirSave, Entries = str(len(posAll[spec])), xDim = str(xDim[spec]), yDim = str(yDim[spec]))

    print("Co-ordinates extracted and saved in " + annotationsSRC)

def normaliseNDPA(slicesDir):

    print("\nSTARTING SEGMENTLOAD/NORMALISENDPA\n")

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
        xDim.append(int(slideProperties['openslide.level[0].width']))                       # assumed always in pixels   
        yDim.append(int(slideProperties['openslide.level[0].height']))                      # assumed always in pixels

        xResolution.append(xRes)
        yResolution.append(yRes)  

    return (xShift, yShift, xResolution, yResolution, xDim, yDim)