'''
This is the main script extracting the image values from the WSI which are 
identified from manual segmenetaiton and training a NN on this data for 
segment identification

'''

from HelperFunctions import *
from glob import glob

# ---------- THINGS TO DO ----------
# Make it so that the directories of the slices and the annotated slices are all in a single callable object, rather than seperate variables
# Make is so that every function saves something so that once a step is complete, that function can be commented
    # out but the next function is only calling the saved output of the function --> this means the script
    # is essentially a full workflow but is also not dependent on every sequential step being run

# User input
'''
Folder location of slices
Extent of training (epochs, batch)
'''

# data directory
data = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'

# Extract the manual co-ordinates of the annotated tissue
# SegmentLoad.readndpa(data)

# create the masks of the annotationes
# MaskMaker.segmentedAreas(data)

# Load in the WSI segments, seperate into target tissue and non-target
WSILoad.quadrants(100, 0, data, 'testWSIMod')

# NOTE this could possible go into the WSILoad function
# Perform pre-processing on the WSI to highlight features/remove background+artifacts 
WSIPreProcessing.main()

# Creating the training data
DataGenerator.main()

# Training the model
ModelTrainer.main()

