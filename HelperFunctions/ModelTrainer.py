'''
This script is the NN which is training and validating on the data generated
'''
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from PIL import Image
from glob import glob
import tifffile as tifi

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *



def train(imageSRC, epoch = 10, batch_size = 32):

    # This script is training on the data from the annotations extracted and quadranted
    # Inputs:   (imageSRC), data source
    #           (epoch), epoch to train for defaults at 10
    #           (batch_size), training parameter defaults at 32
    # Outputs:  (), saves the model which has been trained
    
    print("\nGetting inputs\n")

    # get the data for the model
    X_train, X_valid, Y_train, Y_valid, class2feat, feat2class = dataProcess(imageSRC)

    # create the model topology
    model = neuralNet(X_train[0].shape, len(class2feat))

    # train the weights of the NN and create the fully function NN
    model.fit(
        X_train, Y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_data=(X_valid, Y_valid),
    )

    modelDir = imageSRC + "savedModels/"
    try:
        os.mkdir(modelDir)
    except:
        pass

    # save the model
    model.save(modelDir + "ep_" + str(epoch) + "ba_" + str(batch_size))

    # NOTE: to do,  make the model a seperate funciton (seperates the data process and topology)
    #               do a prediction on non-target tissue and see how we can tets over the entire image

def neuralNet(shape, features):

    # this function creates the model topology for the training
    # Inputs:   (shape), dimensions of the data being fed into the model
    # Outputs:  (model), the model which has been "bolted" together and is ready for training

    i = tf.keras.layers.Input(shape = shape)  #create input shape, flexible!
    print("Input created, " + str(i))

    #create the CNN layers
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation='relu')(i)
    y = tf.keras.layers.Conv2D(64, (3, 3), strides=2, activation='relu')(x)
    z = tf.keras.layers.Conv2D(128, (3, 3), strides=2, activation='relu')(y)

    print("CNN is complete")

    # create the dense layer
    d = Flatten()(z)
    d = Dropout(0.2)(d)
    d = Dense(512, activation='relu')(d)
    d = Dropout(0.2)(d)
    d = Dense(256, activation='relu')(d)
    d = Dropout(0.2)(d)
    d = Dense(128, activation='relu')(d)
    d = Dropout(0.2)(d)
    d = Dense(64, activation='relu')(d)
    d = Dropout(0.2)(d)
    d = Dense(features + 1, activation='softmax')(d)

    print("DNN is complete")

    model = Model(i, d)

    print("Model stitching complete")

    model.summary()

    model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    return(model)

def dataProcess(imageSRC):

    dataSet = 'segmentedTissueSorted'

    # get the directories for all the training images for each class
    train_path = imageSRC + str(dataSet) + '/train'
    valid_path = imageSRC + str(dataSet) + '/test'

    features = glob(train_path + "*")

    # create dictionaries to convert between classes and tensorflow notation
    class2feat = { i : features[i-1] for i in range(1, len(features) + 1) }
    feat2class = dict(zip(features, np.arange(1, len(features)+1)))

    # classes = os.listdir(train_path)                        # get the classes
    trainImages = glob(train_path + '/*/*.tif')        # get the training file paths
    validImages = glob(valid_path + '/*/*.tif')        # get the validation file paths
    folders = glob(train_path + '/*')                  # number of folders
    storeModels = './savedModels/'

    # training info
    IMAGE_SIZE = tifi.imread(trainImages[0]).shape                       # assumes that the first image will represent all (good assumption) --> NOTE this is 3D

    print("\n   Setting up the data")
    Ntrain = len(trainImages)
    Nvalid = len(validImages)
    # initialising arrays for the data to be populated in
    Y_train = np.zeros(Ntrain)
    Y_valid = np.zeros(Nvalid)

    # extract the classifications --> NOTE these have been converted from b16 into b10
    print("    Y_train, " + str(Y_train.shape))
    for i in range(Ntrain): 
        # for each class identified by its names, store as a different value
        for f in features:
            if trainImages[i].find(f) > 0:
                break
        Y_train[i] = int(feat2class[f])

    print("    Y_valid, " + str(Y_valid.shape))
    for i in range(Nvalid): 

        # for each class identified by its names, store as a different value
        for f in features:
            if validImages[i].find(f) > 0:
                break
        Y_valid[i] = int(feat2class[f])

    # tif images into numpy arrays, converting from RGG to grayscale and normalising
    X_train = np.array([np.array(Image.open(fname))/255 for fname in trainImages])
    print("    X_train, " + str(X_train.shape))

    X_valid = np.array([np.array(Image.open(fname))/255 for fname in validImages])
    print("    X_valid, " + str(X_valid.shape))

    return(X_train, X_valid, Y_train, Y_valid, class2feat, feat2class)



data = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'

train(data)