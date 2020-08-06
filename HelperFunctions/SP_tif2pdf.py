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
from multiprocessing import Process
from time import perf_counter as clock
from HelperFunctions.Utilities import nameFromPath, dirMaker

def smallerTif(dataHome, name, size):

    scale = 0.2

    # get a dictionary of all the sample to process
    allSamples = sampleCollector(dataHome, size)

    for spec in allSamples:
        pdfCreator(allSamples, spec, size, scale, False)

def pdfCreator(sampleCollections, spec, size, scale, remove = True):
    # this function takes the directory names and creates a pdf of each sample 
    # Inputs:   (sampleCollections), dictionary containing all the dir names of the imgs
    #           (spec), specimen of interest
    # Outputs:  (), creates a pdf per sample, also has to create a temporary folder
    #               for jpeg images but that is deleted at the end

    # create a temporary folder for the jpeg images per sample
    dataTemp = dataHome + 'temporary' + spec + '/'
    dataTemp = dataHome + str(size) + "/images/" + spec + "/"
    dataPDF = dataHome + "pdfStore/"

    dirMaker(dataTemp)
    dirMaker(dataPDF)

    specificSample = sampleCollections[spec]


    # order the dictionary values 
    order = list(specificSample.keys())
    orderS = [o.split()[0] for o in order]
    orderN = np.array([''.join(i for i in o if i.isdigit()) for o in orderS]).astype(int)
    orderI = np.argsort(orderN)

    order = np.array(order)[orderI]
    dirStore = list()

    # create an ordered list of the sample directories
    c = 0   # count for user to observe progress

    startTime = clock()
    for n in order:

        print("Specimen: " + spec + ", Sample " + str(c) + "/" + str(len(order)))
        # load in the tif files and create a scaled down version
        specificSample[n]
        imgt = tifi.imread(specificSample[n])
        # scale = imgt.shape[1]/imgt.shape[0]
        # img = cv2.resize(imgt, (int(1000*scale), 1000))
        img = cv2.resize(imgt, (int(imgt.shape[1] * scale),  int(imgt.shape[0] * scale)))

        # for sample H710C, all the c samples are rotated
        if (n.lower().find("c") >= 0) & (spec.lower().find("h710c") >= 0):
            img = cv2.rotate(img, cv2.ROTATE_180)

        # add the sample name to the image (top left corner)
        cv2.putText(img, spec + "_" + str(n), 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (0, 0, 0),
            2)
        # create a temporary jpg image and store the dir
        tempName = dataTemp + spec + "_" + str(n) + '.jpg'
        cv2.imwrite(tempName, img)
        dirStore.append(tempName)
        c += 1

        # provide some timing information
        endTime = clock()
        timeOfProcess = endTime - startTime
        timeLeft = timeOfProcess/c * (len(order) - c)
        if c%5 == 0:
            print("     Sample " + spec + " has " + str(timeLeft) +  " secs to go")
    
    # comment the above loop and uncomment below to process already processed images
    # without having to re-process all images
    # dirStore = sorted(glob(dataTemp + spec + "*.jpg"))

    # combine all the sample images to create a single pdf 
    with open(dataPDF + spec + "NotScaled.pdf","wb") as f:
        f.write(i2p.convert(dirStore))
    print("PDF writing complete for " + spec + "!\n")
    # remove the temporary jpg files
    if remove:
        for d in dirStore:
            os.remove(d)

        # remove the temporary dir
        os.rmdir(dataTemp)
        print("Removing " + dataTemp)

    # research drive access via VPN

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
            # NOTE create a txt file of these files
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
    # dataHome = '/Volumes/resabi201900003-uterine-vasculature-marsden135/Boyd collection/ConvertedNDPI/'
    dataHome = '/Volumes/USB/Testing1/'
    size = 3
    name = ''

    sampleExtractor(dataHome, size, name)
    
    
    
    
