'''
Function that gets the tif images and identifies common samples and combines them 
into a pdf of chronological order
'''

import tifffile as tifi
import cv2 
import numpy as np
from glob import glob
import os
import img2pdf as i2p
from multiprocessing import Pool
import multiprocessing
from itertools import repeat
if __name__ != "HelperFunctions.SP_tif2pdf":
    from Utilities import *
else:
    from HelperFunctions.Utilities import *


def smallerTif(dataHome, name, size, scale = 0.3, cpuNo = False):

    # from ndpi files, extract lower resolution images storing them and then 
    # create a pdf file from all of them

    files = sorted(glob(dataHome + str(size) + "/tifFiles/*.tif"))
    allSpecimens = {}
    allSpecimens[nameFromPath(files[0], 1)] = {}

    path = dataHome + str(size) + "/"

    # this is creating a dictionary of all the sample paths per specimen
    for n in files:
        allSpecimens[nameFromPath(files[0], 1)][nameFromPath(n).split("_")[-1]] = n

    # get a dictionary of all the sample to process
    # allSpecimens = sampleCollector(dataHome, size)

    for spec in allSpecimens:
        pdfCreator(allSpecimens[spec], spec, path, scale, cpuNo, False)

def pdfCreator(specificSample, spec, path, scale, cpuNo, remove = False):
    # this function takes the directory names and creates a pdf of each sample 
    # Inputs:   (sampleCollections), dictionary containing all the dir names of the imgs
    #           (spec), specimen of interest
    # Outputs:  (), creates a pdf per sample, also has to create a temporary folder
    #               for jpeg images but that is deleted at the end

    # create a temporary folder for the jpeg images per sample
    dataTemp = path + 'temporary' + spec + '/'
    dataTemp = path + "images/"

    dirMaker(dataTemp)


    # order the dictionary values 
    order = list(specificSample.keys())
    orderS = [o.split()[0].split("_")[-1] for o in order]      # seperates by spaces
    orderN = np.array([''.join(i for i in o if i.isdigit()) for o in orderS]).astype(int)
    orderI = np.argsort(orderN)
    order = np.array(order)[orderI]

    dirStore = list()

    # create an ordered list of the sample directories
    c = 0   # count for user to observe progress
    tifShape = {}

    if cpuNo is False:
        for n, name in zip(order, nameFromPath(specificSample)):
            tifShape[name] = miniSample(specificSample[n], dataTemp, scale, n)
        
    else:
        with Pool(processes=cpuNo) as pool:
            info = pool.starmap(miniSample, zip(list(specificSample.values()), repeat(dataTemp), repeat(scale)))

    # NOTE the order should be the same as the tif files because it is collected 
    # with glob in the same way and the sample name is preserved (?? verify...)
    # Just FYI on why this is so convoluted, it is because the file names can often
    # be stupid names so there is bit of a process for getting them right!
    dirStore = sorted(glob(dataTemp + spec + "*.png"))
    allsmallSamples = {}
    for n in dirStore:
        allsmallSamples[nameFromPath(n).split("_")[-1]] = n

    # create a list of ordered directories to create the pdf
    dirStore = []
    for n in order:
        dirStore.append(allsmallSamples[n])

    '''
    # combine all the sample images to create a single pdf 
    with open(dataPDF + spec + "NotScaled.pdf","wb") as f:
        pass
        # f.write(i2p.convert(dirStore))
    dataPDF = path + "pdfStore/"
    dirMaker(dataPDF)
    print("PDF writing complete for " + spec + "!\n")
    # remove the temporary jpg files
    if remove:
        for d in dirStore:
            os.remove(d)

        # remove the temporary dir
        os.rmdir(dataTemp)
        print("Removing " + dataTemp)
    '''

def miniSample(sample, dataTemp, scale, n = None):

    # this function loads a single image, downsizes it and annotates it 
    # Inputs:   (dataTemp), directory to store the images in
    #           (sample), the directory of the tif file
    #           (scale), the factor to downscale the image
    #           (n), sample categorical name
    # Outputs:  (), saves a lower res image of the tif file with spec info on it
    #           (img.shape), returns the downsampled img shape

    # load in the tif files and create a scaled down version
    # imgt = tifi.imread(specificSample[n])

    spec = nameFromPath(sample) 
    try:
        imgt = cv2.imread(sample)       # NOTE the tif files being read in aren't too big anymore...
    except:
        print(spec + " failed")
        return([])

    if imgt is None:
        print(sample)
        return([])
    # (n.lower().find("d") >= 0) & (spec.lower().find("h710b") >= 0) or \

    img = cv2.resize(imgt, (int(imgt.shape[1] * scale),  int(imgt.shape[0] * scale)))

    # add the sample name to the image (top left corner)
    if n is not None:
        cv2.putText(img, spec + "_" + str(n), 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (0, 0, 0),
            2)
    # create a temporary jpg image and store the dir
    tempName = dataTemp + spec + '.png'
    cv2.imwrite(tempName, img)
    print("Specimen: " + spec + " downsampled")
    # tifi.imwrite(tempName, img)

    return(imgt.shape)

def sampleCollector(dataHome, size):

    # this function collects all the sample tif images and organises them
    # in order to process properly. Pretty much it takes care of the fact
    # that the samples are often named poorly and turns them into a nice 
    # dictionary
    # Inputs:   (dataHome), source directory
    #           (size), specific size image to process
    # Outputs:  (sampleCollection), a nice dictionary which categorises
    #           each tif image into its specimen and then orders them based on 
    #           their position in the stack

    samples = glob(dataHome + str(size) + "/tifFiles/" + "*.tif")

    sampleCollections = {}
    specimens = list()

    # create a dictionary containing all the specimens and their corresponding sample
    for spec in samples:
        specID, no = nameFromPath(spec).split("_")

        # attempt to get samples
        try:
            # ensure that the value can be quantified. If its in the wrong 
            # then it will require manual adjustment to process
            int(no)
            # ensure that the naming convention allows it to be ordered
            while (len(no) < 3):
                no = "0" + no
        except:
            # create a txt file of these files
            print("sample " + specID + no + " is not processed")
            continue

        # create the dictionary as you go
        try:
            sampleCollections[specID][no] = spec
        except:
            sampleCollections[specID] = {}
            sampleCollections[specID][no] = spec
            pass

    return(sampleCollections)

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    # dataHome = '/Volumes/resabi201900003-uterine-vasculature-marsden135/Boyd collection/ConvertedNDPI/'
    dataSource = '/Volumes/USB/Testing1/'
    dataSource = '/Volumes/USB/H653/'
    dataSource = '/Volumes/USB/H653A_11.3/'
    dataSource = '/Volumes/USB/H673A_7.6/'
    dataSource = '/Volumes/USB/H671A_18.5/'
    dataSource = '/Volumes/USB/H671B_18.5/'
    dataSource = '/Volumes/Storage/H653A_11.3/'
    dataSource = '/Volumes/USB/H710C_6.1/'
    dataSource = '/Volumes/USB/H1029A_8.4/'
    dataSource = '/Volumes/USB/H710B_6.1/'
    dataSource = '/Volumes/USB/Test/'



    size = 3
    name = ''
    scale = 0.2
    cpuNo = 6

    smallerTif(dataSource, name, size, scale, cpuNo)
    
    
    
    
