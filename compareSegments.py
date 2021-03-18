'''

Load in hand annotated and automatiaclly segmented samples and 
quantify the overlap of each area

'''


import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

from HelperFunctions.Utilities import *

imgsrc = '/Volumes/USB/H653A_11.3/3/SegmentationTest/'
segsrc = '/Volumes/USB/H653A_11.3/3/SegmentationEvals/'

# manually segmented images
annoImgs = sorted(glob(imgsrc + "*anno.png"))[2:]

# result of the automatic segmentation 
segImgs = sorted(glob(segsrc + "*anno*.png"))

# get the evaluated images 
evalsegs = []
for a in annoImgs:  

    name = nameFromPath(a, 3)

    annoImg = cv2.imread(a)

    np.unique(annoImg, return_counts=True)

    # find the green part of the image (decidua)
    deciduaAnno = np.clip((np.max(annoImg * np.array([-1, 1, -1]), axis = 2)>200)*1, 0, 255)
    deciduaAnnoCount = np.sum(deciduaAnno)
    # find the blue part of the image (villous tree)
    villousAnno = np.clip((np.max(annoImg * np.array([1, -1, -1]), axis = 2)>200)*1, 0, 255)
    villousAnnoCount = np.sum(villousAnno)
    # find the red part of the image (myometrium)
    myoAnno = np.clip((np.max(annoImg * np.array([-1, -1, 1]), axis = 2)>200)*1, 0, 255)
    myoAnnoCount = np.sum(myoAnno)

    # get the segmented images which refer to the annotated image of interest
    segImgsa = np.where(np.array(nameFromPath(segImgs, 3)) == nameFromPath(a, 3))[0]

    # NOTE 0 is the decidua, 1 is myometrium and 2 is villous
    deciduaSegCount = []
    villousSegCount = []
    myoSegCount = []
    for s in segImgsa:

        segImg = cv2.imread(segImgs[s])

        deciduaSeg = np.clip((np.max(segImg * np.array([-1, 1, -1]), axis = 2)>200)*1, 0, 255)
        villousSeg = np.clip((np.max(segImg * np.array([1, -1, -1]), axis = 2)>200)*1, 0, 255)
        myoSeg = np.clip((np.max(segImg * np.array([-1, -1, 1]), axis = 2)>200)*1, 0, 255)
            
        # get the segmentation value
        deciduaSegCount.append(np.sum(deciduaSeg))
        villousSegCount.append(np.sum(villousSeg))
        myoSegCount.append(np.sum(myoSeg))

    '''
    this table outlines the % of correctly identified pixels when the segmentation 
    is attempting to identify that pixel type. For example:
    
                 -- Actual -- 
                  dec  myo  vil
            dec: [0.64 0.3  0.02]
    Predict myo: [0.03 0.91 0.  ]
            vil: [0.34 0.05 0.58]

    
    when the segmentation is identifying the myometrium it correctly identifies
    91% of the actual area (as determined by hand annotations) however it 
    incorrectly identifies 30% of the decidua and 5% of the villous trees as well
    '''

    print("\n----" + name + "----")
    print("            -- ACUTAL % overlap -- ")
    print("               dec  myo  vil")
    print("         dec: " + str(np.round(np.array(deciduaSegCount) / deciduaAnnoCount, 2)))
    print("Predict  myo: " + str(np.round(np.array(myoSegCount) / myoAnnoCount, 2)))
    print("         vil: " + str(np.round(np.array(villousSegCount)/ villousAnnoCount, 2)))
