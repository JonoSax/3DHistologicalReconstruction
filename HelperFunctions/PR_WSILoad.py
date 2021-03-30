'''
This function extracts a tif file of specified resolution from the ndpi file
'''

import os
from glob import glob
from multiprocessing import Pool
from itertools import repeat
if __name__ != "HelperFunctions.PR_WSILoad":
    from Utilities import *
else:
    from HelperFunctions.Utilities import *

# kernel = 150
# dataTrain =  "Data.nosync/testing/"
# size = 0

# magnification levels of the tif files available
tifLevels = [20, 10, 5, 2.5, 1.25, 0.625, 0.3125, 0.15625]

def WSILoad(dataTrain, size, cpuNo = False):

    # this is the function called by main. Organises the inputs for findFeats
    ndpis = sorted(glob(dataTrain + "*.ndpi"))
    ndpas = sorted(glob(dataTrain + "*.ndpa"))
    # ensure all ndpi files have at least a two digit category number (prevents 
    # false identification)
    for ndpi, ndpa in zip(ndpis, ndpas):
        name = nameFromPath(ndpi).replace(" ", "")
        home = regionOfPath(ndpi).replace(" ", "\ ")    # remove the spaces from final name
        no = name.split("_")

        # if there are less than two digits in the name, add a 0
        while len(no[-1]) < 4:
            no[-1] = "0" + no[-1]

        name = no[0] + "_" + no[-1]

        # if the name has been modified rename it
        if ndpi.replace(" ", "\ ") != home + name:
            os.system("mv " + ndpi.replace(" ", "\ ") + " " + home + name + ".ndpi")
            os.system("mv " + ndpa.replace(" ", "\ ") + " " + home + name + ".ndpi.ndpa")
        else:
            print(name + " well named")

    specimens = sorted(glob(dataTrain + "*.ndpi"))
    # load(dataTrain, '', size)

    if cpuNo > 1: 
        with Pool(processes = cpuNo) as pool: 
            pool.starmap(load, zip(specimens, repeat(dataTrain), repeat(size)))
    else:
        for s in specimens:
            load(s, dataTrain, size)

    '''
    # parallelise work
    jobs = {}

    for spec in specimens:
        jobs[spec] = Process(target=load, args=(dataTrain, spec, size))
        jobs[spec].start()

    for spec in specimens:
        jobs[spec].join()
    
    '''

def load(dirimg, dataTrain, size = 0):

    # This moves the quadrants into training/testing data based on the annotations provided
    # Input:    (dataTrain), directory/ies which contain the txt files of the annotation co-ordinates 
    #               as extracted by SegmentLoad.py
    #           (dirimg), path of the tif files 
    #           (size), image size to extract, defaults to the largest one
    # Output:   (), the tif file at the chosen level of magnification

    # What needs to happen is there needs to be some kind of recognition of when annotations are coupled
    # IDEA: use the x and y median values to find annotations which are close to each other
    #       calculate the range of the values of each annotation to find which one is inside which 

    # convert ndpi images into tif files of set size

    # create a folder for the tif files
    dataTif = dataTrain + str(size) + '/tifFiles/'
    dirMaker(dataTif)
    

    ndpiLoad(size, dirimg, dataTif)
    print(nameFromPath(dirimg) + " converted @ " + str(size) + " res")

def ndpiLoad(mag, src, dest):

    # This function extracts tif files from the raw ndpi files. This uses the 
    # ndpitool from https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/ 
    # Install as necessary. 
    # Input:    (mag), magnificataion level to be extracted from the ndpi file
    #           options are 0.15625, 0.3125, 0.625, 1.25, 2.5, 5, 10, 20
    #           (src), file to be extracted with set magnification
    #           (dest), destination
    # Output:   (), tif file of set magnification, saved in the tifFiles directory

    # nameSRC = src.split("/")[-1].split(".")[0]                    # photo name
    nameSRC = nameFromPath(src)
    # dirSRC = src.split(nameSRC + ".ndpi")[0]                      # folder of photo
    dirSRC = regionOfPath(src)

    # ensure correct path name
    src = src.replace(" ", "\ ")

    os.system("ndpisplit -x" + str(mag) + " " + str(src))

    name = nameFromPath(src)

    extractedName = dirSRC + name + "_x" + str(mag) + "_z0.tif"    # NOTE, use of z0 is to prevent 
                                                                # duplication of the same file, however 
                                                                # if there is z shift then this will fail

    imgDir = dest + nameSRC + "_" + str(mag) + ".tif"
    try:
        os.rename(extractedName, imgDir)

    except:
        print("     " + name + " didn't work, trying glob")
        extractedName = glob(dirSRC + name + "*.tif")[0]
        # extractedName = getSampleName(dirSRC, name)
        os.rename(extractedName, imgDir)

    
if __name__ == "__main__":

    dataTrain = '/Volumes/USB/IndividualImages/temporaryH653/'
    dataTrain = '/Volumes/USB/H653A_11.3/'
    name = ''
    size = 2.5
    cpuNo = 3

    WSILoad(dataTrain, size, cpuNo)
    