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

def main(dataDir, imgDir, savedModel, class2feat):
    
    # evaluates the model on a tif image and identifies vessels in the tissue
    # Inputs:   (dataDir), directory of all the data
    #           (imgDir), the directory containing all the tif images to be analysed 
    #           (savedModel), model location
    #           (class2feat), dictionary to relate the class in tf to the original feature

    # load the saved model
    model =  tf.keras.models.load_model(savedModel)
    
    # get the dims of the input ignoring the number of images per input
    hk, wk, ck = model.input.shape[1:4]

    # get all the images
    imgs = glob(imgDir + "*.tif")
    for img in imgs:
        wsi = tifi.imread(img)

        # get the wsi dimensions to process
        hw, ww, cw = wsi.shape

        ws = 20             # width shuffle of the qudrant
        hs = 20             # height shuffle of the quadrant
        quadW = int(ww/ws)  # number of horizontal searches
        quadH = int(hw/hs)  # number of vertical searches

        # with naive quadranting, create a matrix to store the results 
        quadLabels = np.zeros([quadH, quadW])

        # NOTE ATM this is absolutely not working....... 
        # either the model is shit (which it could be but it did train well....)
        # or I am feeding the information into the predictor wrong (see NOTE in ModelTrainer
        # about creating a function for inputs)

        # process along the entire WSI quadranting the image
        for x in range(quadH-int(hk/hs)):
            quadwsi = np.zeros([quadW, wk, hk, 3])
            for y in range(quadW-int(wk/hs)):
                quadwsi[x, :, :, :] = wsi[x*ws:x*ws+wk, y*hs:y*hs+hk, :]/255

            results = model.predict(quadwsi)
            quadLabels[x, :] = np.argmax(results, axis = 1)
            print("x=" + str(x) + "/" + str(quadH))

        

        print('done')
        plt.imshow(quadLabels); plt.show()
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
dataDir = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'
imgDir = dataDir
main(dataDir, imgDir, name)

savedModel.split("(")[-1].split(")")[0].split(",").astype(int)
'''