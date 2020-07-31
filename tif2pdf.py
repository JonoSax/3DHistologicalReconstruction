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


def nameFromPath(paths):
    # this function extracts the name from a path
    # Inputs:   (paths), either a list or string of paths 
    # Outputs:  (names), elist of the names from the paths

    pathStr = False
    if type(paths) is str:
        paths = [paths]
        pathStr = True

    names = list()
    for path in paths:
        # choose the last part of the path and the suffix
        name = path.split("/")[-1].split(".")[0]

        # if there is a size parameter, remove the last part of the name 
        # NOTE this is hard coded where each slide is named as [sampleName]_[sampleNo]
        if len(name.split("_")) > 2:
            name = "_".join(name.split("_")[0:2])
        
        names.append(name)

    # if the path input is a string, it will expect an output of a string as well
    if pathStr:
        names = names[0]

    return names

# create temporary jpg file of all the tif images
def pdfCreator(sampleCollections, s, remove = True):
    # this function takes the directory names and creates a pdf of each sample 
    # Inputs:   (sampleCollections), dictionary containing all the dir names of the imgs
    #           (s), specimen of interest
    # Outputs:  (), creates a pdf per sample, also has to create a temporary folder
    #               for jpeg images but that is deleted at the end

    # create a temporary folder for the jpeg images per sample
    dataTemp = dataHome + 'temporary' + s + '/'
    print("Making " + dataTemp)
    try:
        os.mkdir(dataTemp)
    except:
        pass

    specificSample = sampleCollections[s]


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

        print("Specimen: " + s + ", Sample " + str(c) + "/" + str(len(order)))
        # load in the tif files and create a scaled down version
        imgt = tifi.imread(specificSample[n])
        # scale = imgt.shape[1]/imgt.shape[0]
        scale = 0.03
        # img = cv2.resize(imgt, (int(1000*scale), 1000))
        img = cv2.resize(imgt, (int(imgt.shape[1] * scale),  int(imgt.shape[0] * scale)))

        # for sample H710C, all the c samples are rotated
        if (n.lower().find("c") >= 0) & (s.lower().find("h710c") >= 0):
            img = cv2.rotate(img, cv2.ROTATE_180)

        # add the sample name to the image (top left corner)
        cv2.putText(img, spec + "_" + str(n), 
            (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (0, 0, 0),
            2)
        # create a temporary jpg image and store the dir
        tempName = dataTemp + spec + str(n) + '.jpg'
        cv2.imwrite(tempName, img)
        dirStore.append(tempName)
        c += 1

        endTime = clock()

        timeOfProcess = endTime - startTime

        timeLeft = timeOfProcess/c * (len(order) - c)

        if c%5 == 0:
            print("     Sample " + s + " has " + str(timeLeft) +  " secs to go")
    
    # comment the above loop and uncomment below to process already processed images
    # without having to re-process all images
    # dirStore = sorted(glob(dataTemp + spec + "*.jpg"))

    # combine all the sample images to create a single pdf 
    with open(dataPDF + s + "NotScaled.pdf","wb") as f:
        f.write(i2p.convert(dirStore))
    print("PDF writing complete for " + s + "!\n")
    # remove the temporary jpg files
    if remove:
        for d in dirStore:
            os.remove(d)

        # remove the temporary dir
        os.rmdir(dataTemp)
        print("Removing " + dataTemp)

    # research drive access via VPN

# dataHome = '/Volumes/resabi201900003-uterine-vasculature-marsden135/Boyd collection/ConvertedNDPI/'
dataHome = '/eresearch/uterine/jres129/Boyd collection/ConvertedNDPI/'

samples = glob(dataHome + "*.tif")

sampleCollections = {}
specimens = list()

# create a dictionary containing all the specimens and their corresponding sample
for s in samples:
    spec, no = nameFromPath(s).split("_")

    # attempt to get samples
    try:
        no = (no)
    except:
        # NOTE create a txt file of these files
        print("sample " + spec + no + " is not processed")
        continue

    # create the dictionary as you go
    try:
        sampleCollections[spec][no] = s
    except:
        sampleCollections[spec] = {}
        sampleCollections[spec][no] = s
        pass

dataPDF = dataHome + "pdfStore/"
try:
    os.mkdir(dataPDF)
except:
    pass

samples = list(sampleCollections.keys())

# specific samples
# samples = ['H710C']  #, 'H710A', 'H653A']


for s in sampleCollections:
    Process(target=pdfCreator, args=(sampleCollections, s, False)).start()

'''
for s in samples:
    pdfCreator(sampleCollections, s)
'''