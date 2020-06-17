'''
This script creates n number of images which are purelsy noise
'''

import numpy as np
import cv2
import os


def main(dataTrain, kernel, number):

    # Inputs:   (dataTrain), directory of data
    #           (kernel), size of the filter used throughout the entire script
    #           (number), number of randomised images to create

    # the images are stored in the segmentedTissue folder because this is accessed by the DataGenerator and split into training
    # and testing folders. NOTE THIS ONLY WORKS BECAUSE THE NAME OF THE CLASS IS STORED IN THE IMAGE NAME
    dataNoise = dataTrain + 'segmentedTissue/'

    for n in range(number):
        noiseImg = (np.random.random((kernel, kernel, 3))*255).astype(np.uint8)

        cv2.imwrite(dataNoise + 'noise_n_' + str(n) + ".tif", noiseImg)

# dataHome is where all the directories created for information are stored 
dataHome = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/'

# dataTrain is where the ndpi and ndpa files are stored 
dataTrain = dataHome + 'HistologicalTraining/'

main(dataTrain, 50, 5)