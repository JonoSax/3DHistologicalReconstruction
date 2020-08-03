'''
this script takes the model trained and uses it on data of your choise to identify the classes 
of tissue
'''

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile as tifi
from glob import glob
from HelperFunctions.Utilities import *

def main(dataHome, dataAssess, savedModel, class2feat):
    
    # evaluates the model on a tif image and identifies vessels in the tissue
    # Inputs:   (dataHome), directory of all the data
    #           (dataAssess), the directory containing all the tif images to be analysed 
    #           (savedModel), model location
    #           (class2feat), dictionary to relate the class in tf to the original feature

    # load the saved model
    print("Model used\n" + str(savedModel.split("/")[-1])) 
    model =  tf.keras.models.load_model(savedModel)
    
    # get the dims of the KERNEL used for model training (ignoring the number of images per input, assumes 4D input)
    hk, wk, ck = model.input.shape[1:4]

    # get all the images
    imgs = glob(dataAssess + "*")
    for img in imgs:
        # wsi = tifi.imread(img)
        wsi = cv2.imread(img)

        # load in the processed image data
        wsi = dataPrepare0([img])[0]

        # get the WSI dimensions to process
        hw, ww, cw = wsi.shape

        ws = 20                                     # width shuffle of the qudrant
        hs = 20                                     # height shuffle of the quadrant
        quadW = int(ww/ws) - int(wk/ws) + 1         # number of horizontal searches ([number of possible shuffles in WSI] - [shuffle in a single kernel] + [initiasl search])
        quadH = int(hw/hs) - int(hk/hs) + 1         # number of vertical searches

        # with naive quadranting, create a matrix to store the results 
        quadLabels = np.zeros([quadH, quadW])

        # NOTE ATM this is absolutely not working....... 
        # either the model is shit (which it could be but it did train well....)
        # or I am feeding the information into the predictor wrong (see NOTE in ModelTrainer
        # about creating a function for inputs)

        # process along the entire WSI quadranting the image
        for x in range(quadH):
            quadwsi = np.zeros([quadW, wk, hk, ck])
            for y in range(quadW):

                # normalise the results as well
                quadwsi[x, :, :, :] = wsi[x*ws:x*ws+wk, y*hs:y*hs+hk, :]

            # perform prediction
            results = model.predict(quadwsi)
            print("results: " + str(np.around(results, 2)))

            # store the label most likely associated with the data
            quadLabels[x, :] = np.argmax(results, axis = 1)

            if (hw == hk) & (ww == wk):
                # if the image is the same size as the kernel then the whole data is represented in a single label
                print("    Label = " + img.split("/")[-1] + " Predicted = " + str(class2feat[int(quadLabels[0])]) + "\n")
            
            else:
                # if the image is not the same size, just trace the progress
                print("x=" + str(x) + "/" + str(quadH) + "\n")

        

        # plt.imshow(quadLabels); plt.show()
        '''
        for x in np.arange(0, ww, wk):
            # create a quadrant of image values to pass to the predictor
            quadwsi = np.zeros([int(ww/wk), wk, hk, 3])
            for y in np.arange(0, hw, hk):
                quadwsi[int(x/wk), :, :, :] = wsi[x:x+wk, y:y+hk, :]

            # get predictions
            results = model.predict(quadwsi)

            # find the the class with the highest probability 
            quadLabels[int(x/wk), :] = np.argmax(results, axis = 1)
        '''

            



    print('test')
'''
name = 'sh_(50, 50, 3)ep_10ba_32'
dataHome = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'
dataAssess = dataHome
main(dataHome, dataAssess, name)

savedModel.split("(")[-1].split(")")[0].split(",").astype(int)
'''

# ModelEvaluater.main(dataTrain, dataAssess, modelDir, class2feat)
