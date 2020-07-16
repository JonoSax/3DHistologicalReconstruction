'''
This script takes the aligned tif files and arranges them into a n-dimensional 
tif file 
'''

from .Utilities import nameFromPath, dirMaker
import numpy as np
from glob import glob
import cv2
import tifffile as tifi

def stack(dataTrain, name, size):

    specDirs = sorted(glob(dataTrain + str(size) + '/alignedSamples/' + name + '*.tif'))[0:2]
    stackedDir = dataTrain + '/' + str(size) + '/'

    dims = len(specDirs)

    # all the tif files will be the same size
    img = tifi.imread(specDirs[0])
    
    specName = nameFromPath(specDirs[0], 1)

    x, y, c = img.shape

    stack = np.zeros([x, y, c, dims])

    for s, d in zip(specDirs, range(dims)):
        print("Adding " + nameFromPath(s))
        img = tifi.imread(s)
        stack[:, :, :, d] = img

    tifi.imwrite(stackedDir + specName + "_stacked.tif", stack)

    print(stack.shape)

'''
dataHome = '/Volumes/USB/H653A_11.3new/'
name = ''
size = 3
stack(dataHome, name, size)
'''