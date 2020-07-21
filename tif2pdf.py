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


# research drive access via VPN
dataHome = '/Volumes/resabi201900003-uterine-vasculature-marsden135/Boyd collection/ConvertedNDPI/'
# dataHome = '/eresearch/uterine/jres129/Boyd collection/ConvertedNDPI/'

samples = glob(dataHome + "*.tif")

sampleCollections = {}
specimens = list()

# create a dictionary containing all the specimens and their corresponding sample
for s in samples:
    spec, no = nameFromPath(s).split("_")

    try:
        no = int(no)
    except:
        # NOTE create a txt file of these files
        print("sample " + spec + no + " is not processed")
        continue

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

dataTemp = dataHome + 'temporary/'
try:
    os.mkdir(dataTemp)
except:
    pass

# create temporary jpg file of all the tif images
for s in sampleCollections:
    print("Creating " + s)
    order = np.sort(list(sampleCollections[s].keys()))
    dirStore = list()

    # create an ordered list of the sample directories
    for n in order:
        print("     Sample " + str(n) + "/" + str(np.max(order)))

        # load in the tif files and create a scaled down version
        imgt = tifi.imread(sampleCollections[s][n])
        scale = imgt.shape[1]/imgt.shape[0]
        img = cv2.resize(img, (int(1000*scale), 1000))

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

    # combine all the sample images to create a single pdf 
    print("Writing PDF")
    with open(dataPDF + s + ".pdf","wb") as f:
        f.write(i2p.convert(dirStore))
    pritn("PDF writing complete!\n")
    # remove the temporary jpg files
    for d in dirStore:
        os.remove(d)

# remove the temporary dir
os.rmdir(dataTemp)