'''

This funciton allows a user to select a ROI on a single/multiple samples
which is then used to identify all other matching featues on all other samples

This uses low resolution jpeg images which have been created by the tif2dfp function
in order to reduce the computational load

'''

import cv2
import numpy as np
# import tensorflow as tf
from glob import glob

dataTrain = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/HistologicalTraining2/'

sampleImgs = dataTrain + 'temporaryH710/'

imgs = glob(sampleImgs + "*.jpg")

sample = "H710C31"

imgROI = cv2.imread(imgs[np.where(np.array([sample in i for i in imgs]) == True)[0][0]])

# Have a UI to select features from a raw image
# NOTE this image should be pre-selected and be an input into a function

rois = cv2.selectROIs("image", imgROI)
cv2.destroyAllWindows()

imgStore = []
for roi in rois:
    y0 = roi[1]
    y1 = roi[1] + roi[3]
    x0 = roi[0]
    x1 = roi[0] + roi[2]
    imgSelected = imgROI[y0:y1, x0:x1, :]
    imgSelected.append(imgStore)
    
    # cv2.imshow("img", imgSelected); cv2.waitKey(0)

# NOTE once the ROI are selected, what the processed image looks like should be shown
# and it allows the user to confirm if they like how the features look
    # can the hyperparameters of the image adjustments be user specified as well???
    # should the ROI be a pin or a region???

# pre-process all the images so that a mask like file is created which highlights the features

# create a ML model which is trained on many variation on these images
# NOTE 

# apply the model to all the images and identify where in the image it is 

# find the position of the features, in particular a single point and from this identified 
# position, and save this as a txt file using the dictToText function.
# NOTE the feature locations need to be scaled back up to the correct resolution

print('test')

