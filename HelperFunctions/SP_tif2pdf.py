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
if __name__ == "__main__":
    from Utilities import nameFromPath, dirMaker, dictToTxt, dictOfDirs
else:
    from HelperFunctions.Utilities import nameFromPath, dirMaker, dictToTxt, dictOfDirs

def smallerTif(dataHome, name, size, scale = 0.3):

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
        pdfCreator(allSpecimens[spec], spec, path, scale, False)

def pdfCreator(specificSample, spec, path, scale, remove = False):
    # this function takes the directory names and creates a pdf of each sample 
    # Inputs:   (sampleCollections), dictionary containing all the dir names of the imgs
    #           (spec), specimen of interest
    # Outputs:  (), creates a pdf per sample, also has to create a temporary folder
    #               for jpeg images but that is deleted at the end

    # create a temporary folder for the jpeg images per sample
    dataTemp = path + 'temporary' + spec + '/'
    dataTemp = path + "images/"
    dataPDF = path + "pdfStore/"

    dirMaker(dataTemp)
    dirMaker(dataPDF)


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
    
    # NOTE creating the jpeg files should be very parallelised and at the end
    # the order of the files shoudl be sorted out and the pdf created
    for n, name in zip(order, nameFromPath(specificSample)):
        tifShape[name] = miniSample(dataTemp, specificSample[n], scale, n)

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

    # combine all the sample images to create a single pdf 
    with open(dataPDF + spec + "NotScaled.pdf","wb") as f:
        pass
        # f.write(i2p.convert(dirStore))
    print("PDF writing complete for " + spec + "!\n")
    # remove the temporary jpg files
    if remove:
        for d in dirStore:
            os.remove(d)

        # remove the temporary dir
        os.rmdir(dataTemp)
        print("Removing " + dataTemp)

    # research drive access via VPN

def miniSample(dataTemp, sample, scale, n):

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
        imgt = tifi.imread(sample)
    except:
        imgt = cv2.cvtColor(cv2.imread(sample), cv2.COLOR_BGR2RGB)

    print("Specimen: " + spec + " downsampled")

    # NOTE right here could be a SINGLE function which takes the info and 
    # determines if there are any hard coded rules to apply
    # for sample H710C, all the c samples are rotated
    if (n.lower().find("c") >= 0) & (spec.lower().find("h710c") >= 0) or \
        (n.lower().find("d") >= 0) & (spec.lower().find("h710b") >= 0) or \
        (n.lower().find("c") >= 0) & (spec.lower().find("h671b") >= 0):
        imgt = cv2.rotate(imgt, cv2.ROTATE_180)
        tifi.imwrite(sample, imgt)


    # scale = imgt.shape[1]/imgt.shape[0]
    # img = cv2.resize(imgt, (int(1000*scale), 1000))
    img = cv2.resize(imgt, (int(imgt.shape[1] * scale),  int(imgt.shape[0] * scale)))

    # add the sample name to the image (top left corner)
    cv2.putText(img, spec + "_" + str(n), 
        (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1,
        (0, 0, 0),
        2)
    # create a temporary jpg image and store the dir
    tempName = dataTemp + spec + '.png'
    cv2.imwrite(tempName, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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

    # NOTE I wonder if this works better with higher resolution images as a rule 
    # or just as an observation?

    # dataHome = '/Volumes/resabi201900003-uterine-vasculature-marsden135/Boyd collection/ConvertedNDPI/'
    dataSource = '/Volumes/USB/Testing1/'
    dataSource = '/Volumes/USB/H653/'
    dataSource = '/Volumes/USB/H653A_11.3/'
    dataSource = '/Volumes/USB/H673A_7.6/'
    dataSource = '/Volumes/USB/H710B_6.1/'
    dataSource = '/Volumes/USB/H671B_18.5/'


    size = 3
    name = ''
    scale = 0.2

    smallerTif(dataSource, name, size, scale)
    
    
    
    
