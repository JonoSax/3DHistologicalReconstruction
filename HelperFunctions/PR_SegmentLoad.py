'''
This script loads in the NDPA files which have the manually segmented 
sections and converts all the positions into their corresponding pixel positions

This is done for annotations (.pos) and for specific points (.pin)
'''

import numpy as np
from glob import glob
import openslide 
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
from HelperFunctions.Utilities import *
from multiprocessing import process

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

def readannotations(dataTrain, specimen = '', alwaysProcess = True):

    # This function reads a NDPA file and extracts the co-ordinate of the hand drawn points and converts
    # them into their pixel location equivalents
    # Input:    (dataTrain), Directory for the ndpi files
    #           (specimen), the specimen name/s to specifically investigate (optional, if not
    #                               chosen then will investigate the entire directory)
    #           (alwaysProcess), boolean whether to write over existing .pin and .pos files
    # Output:   (), Saves a txt file of the recorded positions of the:
    #                   annotations, saved in posFiles
    #                   pins, saved in pinFiles

    # pos and feat files are raw from the ndpa so don't depend on the chosen size of
    # the tif file to process. If these files are present then don't perform extraction 
    if ('posFiles' in os.listdir(dataTrain)) and ('pinFiles' in os.listdir(dataTrain)) and alwaysProcess is False:
        print("Pos and Feat files already extracted")

    else:
        # get the directories of all the specimes of interest
        ndpaNames = sorted(glob(dataTrain + specimen + "*.ndpa"))
        ndpiNames = sorted(glob(dataTrain + specimen + "*.ndpi"))      # need this for the properties of the image

        specDict = dictOfDirs(ndpa = ndpaNames, ndpi = ndpiNames)

        # go through each ndpa file of interest and extract the RAW co-ordinates into a list.
        # this list is indexed as follows:
        # posAll --> for all the co-ordinates
        #       posA --> for a single ndpa file
        #               co-ordinates0 --> for a single annotation
        #               co-ordinates1
        #               ....
        #       ...
        for spec in specDict:

            ndpaPath = specDict[spec]['ndpa']
            ndpiPath = specDict[spec]['ndpi']

            # get the drawn annotations
            posA = getAnnotations(spec, ndpaPath)
            info = readlandmarks(spec, ndpaPath)

            # get the ndpi properties of all the specimes of interest
            xShift, yShift, xRes, yRes, xDim, yDim = normaliseNDPA(ndpiPath)

            # apply the necessary transformations to extracted co-ordinates to convert to pixel represnetations
            # with the origin in the top left corner of the image and save them as a txt file for all npda files

            # create txt file which contains these co-ordinates
            # f = open(str(ndpaNames[spec]) + ".pos", 'w')

            stacks = list()

            # npdi properties
            centreShift = np.hstack([xShift, yShift])
            topLeftShift = np.hstack([xDim/(2 * xRes), yDim/(2 * yRes)])
            scale = np.hstack([xRes, yRes])

            # f.write("NUMBER_OF_ANNOTATIONS=" + str(len(posSpec)) + "\n")
            for posSpec in posA:
                
                # co-ordinate transformation
                stack = ((posSpec - centreShift + topLeftShift ) * scale).astype(int)
                
                # save into a list
                stacks.append(stack)

            for feat in info.keys():
                info[feat] = ((info[feat] - centreShift + topLeftShift ) * scale).astype(int)

            # save the entire list as a txt file per utilities saving structure
            if len(stacks) > 0:
                listToTxt(stacks, dataTrain + 'posFiles/' + spec + ".pos", Entries = str(len(posSpec)), xDim = str(xDim), yDim = str(yDim))
            if len(info) > 0:
                dictToTxt(info, dataTrain + 'pinFiles/' + spec + ".pin")

def readlandmarks(spec, ndpaPath):
    
    # This function reads a NDPA file and extracts the co-ordinate of the landmarks and converts
    # them into their pixel location equivalents
    # Input:    (ndpaPath), Directory for the ndpa files
    # Output:   (), Saves a txt file of the landmkars positions on the slide. Each
    #           entry to the list refers to a section drawn. Saved in the folder, landmarkFiles

    # ndpa file
    file = open(ndpaPath)

    # extract all info from text file line by line
    doco = list(file.readlines())

    # find the number of annotations drawn in the slide --> Used to validate that the search is completed
    pins = open(ndpaPath).read().count('type="pin"')

    # declare list and array
    info = {}
    l = 0
    # NOTE the skips in this loop are hard coded because as far as i can tell
    # they are enitrely predictable --> ONLY WORKS IF THERE ARE ONLY FREE HAND ANNOTATIONS
    while l < len(doco):

        # identifty the pin
        if 'type="pin"' in doco[l]:

            # get the title
            l-=10
            title = (doco[l].replace("<title>", "").replace("</title>\n", "").replace(" ", "")).lower()

            # ensure names are correct
            if len(title.split("_")) <= 2:

                # ensure that the feat name is split into three portions
                if title.find("feat") >= 0:
                    title = 'feat_' + title.split('feat')[-1]

                # add a boundary name to the type
                elif title.find("top") >= 0 or  title.find("bottom") >= 0 or title.find("right") >= 0 or title.find("left") >= 0:
                    title = 'bound_' + title
                    pass


            # get the unit used
            l+=2
            unitStr = doco[l].replace("<coordformat>", "").replace("</coordformat>\n", "").replace(" ", "")
            unit = unitDict[unitStr]

            # the the co-ordinate of the pin
            l+=9
            x = doco[l]
            y = doco[l+1]

            x = x.replace("<x>", ""); x = int(x.replace("</x>\n", "")) * unit       # convert the positions into mm
            y = y.replace("<y>", ""); y = int(y.replace("</y>\n", "")) * unit       # convert the positions into mm

            # sace info in dictionary
            info[title] = np.array([x, y])

        else:
            l += 1

    file.close()

    try:
        print(str(len(info)/pins*100)+"% of pins found from " + spec)
    except:
        print("0 pins found from " + spec)

    return(info)

def getAnnotations(spec, ndpaPath):

    # This function specifically reads the drawn annotations on the file
    # Input:    (ndpaPath), ndpafile path
    # Outputs:  (posA), list of co-ordinates of the annotations

    # ndpa file
    file = open(ndpaPath)

    # extract all info from text file line by line
    doco = list(file.readlines())

    # find the number of annotations drawn in the slide --> Used to validate that the search is completed
    sections = open(ndpaPath).read().count('type="freehand"')

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
    
    # check to assess if all points found 
    try:
        print(str(len(posA)/sections*100)+"% of annotations found from " + spec)
    except:
        print("0 annotations found from " + spec)

    file.close()

    return(posA)

def readXML(ndpaPath):
    
    # NOTE this could be a good place for a more generic xml reader where each annotation
    # is read in and then based on what type of data it is, stored differently

    # ndpa file
    file = open(ndpaPath)

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
        if "<ndpviewstate" in doco[l]:
            pass

    pass
    
def normaliseNDPA(sliceDir):

    # This functions reads the properties of the ndpi files and extracts the properties 
    # of the file.
    # Inputs:   (data), list of directory/ies containing all the ndpi files of interest
    # Output:   (xShift),       # dictionary of x shift of the slide scanner centre from image centre in physical units (nm)
    #           (yShift),       # dictionary of y shift of the slide scanner centre from image centre in physical units (nm)
    #           (xResolution),  # dictionary of x scale of physical units to pixels
    #           (yResolution),  # dictionary of y scale of physical units to pixels
    #           (xDim),         # dictionary of width (pixels) of highest resolution of tif stored
    #           (yDim),         # dictionary of height (pixels) of highest resolution of tif stored 

    # data = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'
    # data = '/Volumes/Storage/H653A_11.3 new/'
    
    slicedirName = nameFromPath(sliceDir)

    slideProperties = openslide.OpenSlide(sliceDir).properties                          # all ndpi properties
    unit = unitDict[slideProperties['tiff.ResolutionUnit']]                             # get the unit multiplier so that all units are in mm
    xShift = (int(slideProperties['hamamatsu.XOffsetFromSlideCentre']) * 10**-6)    # assumed nm, converted to mm
    yShift = (int(slideProperties['hamamatsu.YOffsetFromSlideCentre']) * 10**-6)    # assumed nm, converted to mm
    xRes = int(slideProperties['tiff.XResolution']) / unit                              # scale is unit dependent
    yRes = int(slideProperties['tiff.YResolution']) / unit                              # scale is unit dependent                       
    xDim = (int(slideProperties['openslide.level[0].width']))                       # assumed always in pixels   
    yDim = (int(slideProperties['openslide.level[0].height']))                      # assumed always in pixels

    xResolution = xRes
    yResolution = yRes

    return (xShift, yShift, xResolution, yResolution, xDim, yDim)