'''
This function standardises the size of all the images in a directory
'''

import numpy as np
import cv2
from glob import glob
from Utilities import nameFromPath, dirMaker, txtToDict, dictToArray
import os
import tifffile as tifi
from multiprocessing import Process

dataSource = '/Volumes/USB/H673A_7.6/3/'
dataMasked = dataSource + "masked/"
dataMoved = dataSource + 'moved/'

dirMaker(dataMoved)

imgShapes = dictToArray(txtToDict('/Volumes/USB/H673A_7.6/3/info/all.tifshape')[0])

imgShape = np.array(imgShapes)
yM, xM, zM = np.max(imgShape, axis = 0).astype(int)

fieldO = np.zeros([yM, xM, zM]).astype(np.uint8)

imgDirs = glob(dataMasked + "*.tif")

for i in imgDirs:

    name = nameFromPath(i)
    img = tifi.imread(i)

    field = fieldO.copy()

    y, x, z = img.shape

    # place the image to the right of the new field
    field[-y:, -x:, :] = img

    tifi.imwrite(dataMoved + name + ".tif", field)

    print(name + " saved")
