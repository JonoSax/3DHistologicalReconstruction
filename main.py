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

# dataHome is where all the directories created for information are stored 
dataHome = '/Volumes/Storage/'

# research drive access from HPC
# dataHome = '/eresearch/uterine/jres129/All material for Boyd paper/Results Boyd paper/Arcuates and Radials/NDPI segmentations/H653A_11.3\ new/'

# research drive access via VPN
dataHome = '/Volumes/resabi201900003-uterine-vasculature-marsden135/All material for Boyd paper/Results Boyd paper/Arcuates and Radials/NDPI segmentations/'

# dataTrain is where the ndpi and ndpa files are stored 
dataTrain = dataHome + 'H653A_11.3new/'

# data directory containing the wsi images to be assessed
dataAssess = dataHome + "samples/"

size = 3
kernel = 50
name = ''
portion = 0.2

# NOTE: update directories used between dataHome and dataTrain

# Extract all the manual co-ordinates of the annotated tissue
SegmentLoad.readannotations(dataTrain, name)


## create the masks of the annotationes
# MaskMaker.maskCreator(dataTrain, name, size)

'''
## from the wsi, get the target tif resolution
WSILoad.load(dataTrain, name, size)
# WSILoad.load(dataAssess, name, size)

# extract the individual specmimens
SegmentID.align(dataTrain, name, size)

## Extract the target tissue from the tif files 
WSIExtract.segmentation(dataTrain, name, size)
## create quadrants of the target tissue from the extracted tissue
# targetTissue.quadrant(dataTrain, name, size, kernel)

# Creating the training data --> NOTE every time it does this it creates a replaces the previous testing/training data
DataGenerator.main(dataTrain, portion, 'vessel')

# Training the model --> note this should never be commented out, only set to False
modelDir, class2feat = ModelTrainer.train(dataTrain, name = 'text', epoch=4, train = True)

# ModelEvaluater.main(dataTrain, dataAssess, modelDir, class2feat)

'''
