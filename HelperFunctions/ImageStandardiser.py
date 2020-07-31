'''
This function standardises the size of all the images in a directory
'''

import numpy as np
import cv2
from glob import glob
from Utilities import nameFromPath, dirMaker
import os
from multiprocessing import Process

def imgStandardiser(dataDestination, dataSource, sample):

    dirMaker(dataDestination + sample + "/")

    imgDirs = glob(dataSource + sample + "/*.jpg")

    img = []
    for i in imgDirs:
        img.append(cv2.imread(i))

    imgShape = []
    for i in img:
        imgShape.append(i.shape)

    imgShape = np.array(imgShape)

    yM, xM, zM = np.max(imgShape, axis = 0)

    fieldO = np.zeros([yM, xM, zM]).astype(np.uint8)

    for i, idir in zip(img, imgDirs):

        name = nameFromPath(idir)

        field = fieldO.copy()

        y, x, z = i.shape

        ym0 = int((yM - y) / 2)
        xm0 = int((xM - x) / 2)

        # place the image in the middle of the new field
        field[ym0:(ym0 + y), xm0:(xm0 + x), :z] = i

        cv2.imwrite(dataDestination + sample + "/" + name + ".jpg", field)

        print(name + " saved")

dataSource = '/Volumes/Storage/SegmentPDF/InvididualImages/'
dataDestination = '/Volumes/USB/InvididualImagesMod/'

specimens = os.listdir(dataSource)

for sample in specimens:

    imgStandardiser(dataDestination, dataSource, sample)

    # NOTE this crashes my computer....
    # Process(target = imgStandardiser, args = (dataDestination, dataSource, sample)).start()
