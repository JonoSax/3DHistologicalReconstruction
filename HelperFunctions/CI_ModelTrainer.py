'''
This script is the NN which is training and validating on the data generated
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import cv2
from glob import glob
import tifffile as tifi
from datetime import datetime
from HelperFunctions.Utilities import *

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *



def train(dataTrain, name = "", save = True, epoch = 10, batch_size = 32, train = True):

    # This script is training on the data from the annotations extracted and quadranted
    # Inputs:   (dataTrain), data source
    #           (name), optional for a specific model version
    #           (epoch), epoch to train for defaults at 10
    #           (batch_size), training parameter defaults at 32
    #           (train), boolean as to whether the model should be trained. 
    #               if false then model won't be re-trained/saved so only outputs returned
    # Outputs:  (), saves the model which has been trained
    #           (modelName), the full path of the model being saved (or at least where it was saved if train = False)
    #           (class2feat), dictionary to convert the classes to the original features
    
    # set the locaion to save the model
    modelDir = dataTrain + "savedModels/"

    # get the data for the model
    X_train, X_valid, Y_train, Y_valid, class2feat = dataProcess(dataTrain, train)

    # name of the model (include the time it was created)
    modelLabel = modelDir + name + "_sh_" + str(X_train[0].shape) + "ep_" + str(epoch) + "ba_" + str(batch_size) + "t_" 
    
    # add time stamp
    modelName = modelLabel + datetime.now().strftime("%Y_%m_%d_%H_%M")

    if train == True:
        print("---Training model---")

        # create the model topology
        model = neuralNet(X_train[0].shape, len(class2feat))

        # train the weights of the NN and create the fully function NN
        model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            epochs=epoch,
            validation_data=(X_valid, Y_valid),
        )

        try:
            os.mkdir(modelDir)
        except:
            pass

        # save the model
        if save:
            model.save(modelName)
    else:
        print("---Not training model---")

        # get the newest model with the training parameters of interest
        modelDirs = glob(modelLabel + "*")
        
        # get the names of the models
        names = list()
        for m in modelDirs: names.append(m.split("t_")[-1])

        # find the dates they were created
        dates = list()
        for n in names: dates.append(datetime.strptime(n, '%Y_%m_%d_%H_%M'))

        # select the newest model
        modelName = modelDirs[np.argmax(dates)]



    return(modelName, class2feat)

    # NOTE: to do,  
    #               do a prediction on non-target tissue and see how we can tets over the entire image

def neuralNet(shape, noFeatures):

    # this function creates the model topology for the training
    # Inputs:   (shape), dimensions of the data being fed into the model
    #           (noFeatures), number of features being identified
    # Outputs:  (model), the model which has been "bolted" together and is ready for training

    i = tf.keras.layers.Input(shape = shape)  #create input shape, flexible!
    print("Input created, " + str(i))

    #create the CNN layers --> make enough layers so that the final representation is smaller than 4x4
    n = 5
    c = tf.keras.layers.Conv2D(2**n, (3, 3), strides=2, activation='relu')(i)
    while c.shape[1] > 4:
        n+=1
        c = tf.keras.layers.Conv2D(2**n, (3, 3), strides=2, activation='relu')(c)

    print("CNN is complete")

    # create the dense layer
    
    d = Flatten()(c)
    '''
    d = Dropout(0.2)(d)
    d = Dense(512, activation='relu')(d)
    d = Dropout(0.2)(d)
    d = Dense(256, activation='relu')(d)
    d = Dropout(0.2)(d)
    d = Dense(128, activation='relu')(d)
    d = Dropout(0.2)(d)'''
    d = Dense(64, activation='relu')(d)
    d = Dropout(0.2)(d)
    d = Dense(32, activation='relu')(d)
    d = Dropout(0.2)(d)
    d = Dense(noFeatures, activation='softmax')(d)

    print("DNN is complete")

    model = Model(i, d)

    print("Model stitching complete")

    model.summary()

    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    return(model)

def dataProcess(dataTrain, train):

    # this function processes the training data for tensorflow training
    # Inputs:   (dataTrain), directory of data
    #           (train), boolean. If false then won't process the image data and only return the dictionary
    # Outputs:  (X_train), normalised images for training (return [] when train = False)
    #           (X_valid), normalised images for validation (return [] when train = False)
    #           (Y_train), associated class for training (return [] when train = False)
    #           (Y_valid), associated class for validation (return [] when train = False)
    #           (class2feat), dictionary used to convert the processed classes back into the original features in ModelEvaluater

    dataSet = 'segmentedTissueSorted'

    # get the directories for all the training images for each class
    train_path = dataTrain + str(dataSet) + '/trainT/'
    valid_path = dataTrain + str(dataSet) + '/testT/'

    '''
    Create a function here which create an image of pure noise 
    '''

    # get the names of the features (excluding any hidden files)
    features = [f for f in os.listdir(train_path) if not f.startswith('.')]
    features.append("other")

    # create dictionaries to convert between classes and tensorflow notation
    class2feat = { i : features[i] for i in range(len(features)) }
    feat2class = dict(zip(features, np.arange(len(features))))

    # NOTE this is outside the if statement as atleast one image is needed so a model imput size can be calculated
    # trainImages = glob(train_path + '/*/*.tif')        # get the training file paths
    trainImages = glob(train_path + '/*/*.png')        # get the training file paths
    
    if train == True:

        # NOTE these are inside train because they are only needed if fully training
        # validImages = glob(valid_path + '/*/*.tif')        # get the validation file paths
        validImages = glob(valid_path + '/*/*.png')        # get the validation file paths

        folders = glob(train_path + '/*')                  # number of folders
        storeModels = './savedModels/'

        # training info
        # IMAGE_SIZE = tifi.imread(trainImages[0]).shape                       # assumes that the first image will represent all (good assumption) --> NOTE this is 3D
        IMAGE_SIZE = cv2.imread(trainImages[0]).shape                       # assumes that the first image will represent all (good assumption) --> NOTE this is 3D


        print("\n   Setting up the data")
        Ntrain = len(trainImages)
        Nvalid = len(validImages)
        # initialising arrays for the data to be populated in
        Y_train = np.zeros(Ntrain)
        Y_valid = np.zeros(Nvalid)

        # extract the classifications from each image for the training and validation data
        print("    Y_train, " + str(Y_train.shape))
        for i in range(Ntrain): 
            # for each class identified by its names, store as a different value
            for f in features:
                # find the class in the NAME 
                # when found break and this is the label for training
                if trainImages[i].split("/")[-1].find(f) > 0:
                    break
            Y_train[i] = int(feat2class[f])

        print("    Y_valid, " + str(Y_valid.shape))
        for i in range(Nvalid): 

            # for each class identified by its names, store as a different value
            for f in features:
                if validImages[i].split("/")[-1].find(f) > 0:
                    break
            Y_valid[i] = int(feat2class[f])

        # NOTE this should be made into a function so that for both training and for 
        # predicting the inputted formated into the model is the same

        # tif images into numpy arrays, converting from RGG to grayscale and normalising
        X_train = dataPrepare0(trainImages)
        print("    X_train, " + str(X_train.shape))

        X_valid = dataPrepare0(validImages)
        print("    X_valid, " + str(X_valid.shape))
    
    else:
        # laod two images just so that the size of the input can be calculated in the name when train=False
        X_train = dataPrepare0(trainImages[0:2])
        X_valid = []
        Y_train = []
        Y_valid = []

    return(X_train, X_valid, Y_train, Y_valid, class2feat)

'''
dataTrain = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'

modelName, class2feat = train(dataTrain, name = "textTrain", save = False, train = True)

print(class2feat)
'''