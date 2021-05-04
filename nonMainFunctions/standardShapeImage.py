'''

Create new images which are all the same size (mainly used to showcase naive reconstructions)

'''

import cv2
from glob import glob 
import numpy as np
from HelperFunctions.Utilities import nameFromPath, dirMaker
from multiprocessing import Pool
from itertools import repeat

def standardiser(d, path):
    specname = d.split("/")[-2]
    print(specname)
    dest = d + "/3/maskedSamplesNormSize/"
    dirMaker(dest, True)
    d += path
    imgs = glob(d + "*.png")

    shapes = []
    for i in imgs:
        shape = cv2.imread(i).shape
        shapes.append(shape)
    maxShape = np.max(np.array(shapes), axis = 0)

    print(specname + " has max shape " + str(maxShape))
    plate = np.zeros(maxShape).astype(np.uint8)

    for i in imgs:
        sampname = nameFromPath(i, 3)
        img = cv2.imread(i)
        if sampname.lower().find("c") > -1:
            pass #img = cv2.rotate(img, cv2.ROTATE_180)
        s = img.shape
        imgR = plate.copy()
        imgR[:s[0], :s[1], :] = img
        cv2.imwrite(dest + sampname + ".png", imgR)

if __name__ == "__main__":

    destsrc = '/Volumes/USB/'
    dataHomes = glob(destsrc + "H*")
    path = "/3/maskedSamples/"

    cpuNo = 1

    
    standardiser('/Volumes/USB/H671B_18.5/', path)
    
    '''
    with Pool(processes=cpuNo) as pool:
        pool.starmap(standardiser,zip(dataHomes, repeat(path)))
    '''