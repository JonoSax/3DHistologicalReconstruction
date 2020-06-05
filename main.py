'''
This is the main script extracting the image values from the WSI which are 
identified from manual segmenetaiton and training a NN on this data for 
segment identification
'''

from HelperFunctions import *
from glob import glob

# ---------- THINGS TO DO ----------
# Make it so that the directories of the slices and the annotated slices are all in a single callable object, rather than seperate variables

# User input
'''
Folder location of slices
Extent of training (epochs, batch)
'''
# data directory
data = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/testing/'

kernel = 150

# NOTE functions need to be able to deal with a list of names
slicesDir = glob(str(data+"*.ndpi"))

test = slicesDir[0].split(data)[-1].split(".ndpi") 
annotationsDir = slices = glob(str(data+"*.ndpa"))

# get a list of all the names of the specimens represented in the ndpi (and ndpa) files
names = list()
for file in glob(data + "*.ndpi"): 
    names.append(file.split(data)[-1].split(".ndpi")[0])           

SegmentLoaded = list()
WSILoaded = list()

for name in names:

    # Load in the locations of identified slices
    SegmentLoaded.append(SegmentLoad.readndpa(data, name))

    # Load in the WSI segments, seperate into target tissue and non-target
    WSILoaded.append(WSILoad.main(kernel, data, name, 0, annotations))

# NOTE this could possible go into the WSILoad function
# Perform pre-processing on the WSI to highlight features/remove background+artifacts 
WSIPreProcessing.main()

# Creating the training data
DataGenerator.main()

# Training the model
ModelTrainer.main()

