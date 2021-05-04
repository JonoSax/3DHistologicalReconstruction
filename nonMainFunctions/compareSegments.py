'''

Load in hand annotated and automatiaclly segmented samples and 
quantify the overlap of each area

'''


import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

from HelperFunctions.Utilities import *

model = "ResNet101"
model = "VGG16"
imgsrc = '/Volumes/USB/H653A_11.3/3/SegmentationManual/'
segsrc = '/Volumes/USB/H653A_11.3/3/SegmentationsEvals' + model + "/"

# manually segmented images
annoImgs = sorted(glob(imgsrc + "*anno.png"))

# result of the automatic segmentation 
segImgs = sorted(glob(segsrc + "*anno*.png"))


# get the evaluated images 
evalsegs = []

darkRatioStore = []
decratioStore = [] 
myoratioStore = [] 
vilratioStore = []  
for a in annoImgs:  

    name = nameFromPath(a, 3)

    annoImg = cv2.imread(a)

    # get the null information
    darkAnno = (np.sum(annoImg, axis =2)==0)*1
    darkAnnoCount = np.sum(darkAnno)

    # find the green part of the image (decidua)
    # deciduaAnno1 = (annoImg[:, :, 1]>100)*1
    deciduaAnno = (np.sum(annoImg[:, :, [0, 2]]<100, axis = 2)>1)-darkAnno
    deciduaAnnoCount = np.sum(deciduaAnno)
    
    # find the red part of the image (myometrium)
    # myoAnno = (annoImg[:, :, 2]>100)*1
    myoAnno = (np.sum(annoImg[:, :, [0, 1]]<100, axis = 2)>1)-darkAnno
    myoAnnoCount = np.sum(myoAnno)

    # find the blue part of the image (villous tree)
    # villousAnno = (annoImg[:, :, 0]>100)*1
    villousAnno = (np.sum(annoImg[:, :, [1, 2]]<100, axis = 2)>1)-darkAnno
    villousAnnoCount = np.sum(villousAnno)

    # get the segmented images which refer to the annotated image of interest
    segImgsa = np.where(np.array(nameFromPath(segImgs, 3)) == nameFromPath(a, 3))[0]

    # NOTE 0 is the decidua, 1 is myometrium and 2 is villous
    darkSegCount = []
    deciduaSegCount = []
    myoSegCount = []
    villousSegCount = []
    segImgComb = np.zeros(annoImg.shape)
    for s in segImgsa:

        segImg = cv2.imread(segImgs[s])

        segImgComb+=segImg

        # create the tissue images
        darkSeg = (np.sum(segImg, axis = 2) == 150)*1

        # NOTE these aren't summing up to one so perhaps change the 
        # colours on the segmentations to be better identifiable??
        deciduaSeg = (segImg[:, :, 1]>200)*1
        # deciduaSeg = (np.sum(segImg[:, :, [0, 1]]<100, axis = 2)>0)# -darkSeg
        myoSeg = (segImg[:, :, 2]>200)*1
        # myoSeg = (np.sum(segImg[:, :, [0, 2]]<100, axis = 2)>0)# -darkSeg
        villousSeg = (segImg[:, :, 0]>200)*1
        # villousSeg = (np.sum(segImg[:, :, [1, 2]]<100, axis = 2)>0)# -darkSeg
        

        # get the segmentation proporptions
        darkSegCount.append(np.sum(darkSeg))
        deciduaSegCount.append(np.sum(deciduaSeg))
        myoSegCount.append(np.sum(myoSeg))
        villousSegCount.append(np.sum(villousSeg))

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

    darkRatio = np.round( (np.array(darkSegCount)/ darkAnnoCount) / (np.sum(darkSegCount)/darkAnnoCount), 2)
    decratio = np.round( (np.array(deciduaSegCount)/ deciduaAnnoCount) / (np.sum(deciduaSegCount)/ deciduaAnnoCount), 2)
    myoratio = np.round( (np.array(myoSegCount)/ myoAnnoCount) / (np.sum(myoSegCount)/ myoAnnoCount), 2)
    vilratio = np.round( (np.array(villousSegCount)/ villousAnnoCount) / (np.sum(villousSegCount)/ villousAnnoCount), 2)

    darkRatioStore.append(darkRatio)
    decratioStore.append(decratio)
    myoratioStore.append(myoratio)
    vilratioStore.append(vilratio)

    print("\n           ----" + name + "----")
    print("            -- ACUTAL % overlap -- ")
    print("               dark  dec  myo  vil  sum")
    print("        dark: " + str(darkRatio))
    print("         dec: " + str(decratio))
    print("Predict  myo: " + str(myoratio))
    print("         vil: " + str(vilratio))

darkRatioStoreInfo =np.c_[np.min(darkRatioStore, axis = 0), np.median(darkRatioStore, axis = 0), np.max(darkRatioStore, axis = 0)]
decratioStoreInfo = np.c_[np.min(decratioStore, axis = 0), np.median(decratioStore, axis = 0), np.max(decratioStore, axis = 0)]
myoratioStoreInfo = np.c_[np.min(myoratioStore, axis = 0), np.median(myoratioStore, axis = 0), np.max(myoratioStore, axis = 0)]
vilratioStoreInfo = np.c_[np.min(vilratioStore, axis = 0), np.median(vilratioStore, axis = 0), np.max(vilratioStore, axis = 0)]

print("\n           ---- ALL ---- ")
print("            -- ACUTAL % overlap med, (min, max) -- ")
print("               dark  dec  myo  vil  sum")

print("dark,", end = " ")
for mi, me, ma in darkRatioStoreInfo:
    # print(str(d) + "±" + str(s), end = " ")
    print('%.2f (%.2f-%.2f),' % (me, mi, ma), end = " ")

print("\n dec,", end = " ")
for mi, me, ma in decratioStoreInfo:
    # print(str(d) + "±" + str(s), end = " ")
    print('%.2f (%.2f-%.2f),' % (me, mi, ma), end = " ")


print("\n myo,", end = " ")
for mi, me, ma in myoratioStoreInfo:
    # print(str(d) + "±" + str(s), end = " ")
    print('%.2f (%.2f-%.2f),' % (me, mi, ma), end = " ")


print("\n vil,", end = " ")
for mi, me, ma in vilratioStoreInfo:
    # print(str(d) + "±" + str(s), end = " ")
    print('%.2f (%.2f-%.2f),' % (me, mi, ma), end = " ")

