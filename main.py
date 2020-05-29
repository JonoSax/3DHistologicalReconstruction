'''
This is the main script extracting the image values from the WSI which are 
identified from manual segmenetaiton and training a NN on this data for 
segment identification
'''

from HelperFunctions import *
import glob

# User input
'''
Folder location of slices
Extent of training (epochs, batch)
'''

data = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'

# NOTE functions need to be able to deal with a list of names
slicesDIr = glob(str(data+"*.ndpi"))
annotationsDir = slices = glob(str(data+"*.ndpa"))

# Load in the WSI
segments = WSILoad.main(slicesDir)

# Load in the locations of identified slices
annotations = SegmentLoad.readndpa(annotationsDir)

# Perform pre-processing on the WSI to highlight features/remove background+artifacts 
WSIPreProcessing.main()

# Creating the training data
DataGenerator.main()

# Training the model
ModelTrainer.main()

