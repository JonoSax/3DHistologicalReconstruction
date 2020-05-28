'''
This is the main script extracting the image values from the WSI which are 
identified from manual segmenetaiton and training a NN on this data for 
segment identification
'''

import HelperFunctions as HF
import numpy as np

# User input
'''
Folder location of slices
Extent of training (epochs, batch)
'''

# Load in the WSI
HF.WSILoad.main()

# Load in the locations of identified slices
HF.SegmentLoad.main()

# Perform pre-processing on the WSI to highlight features/remove background+artifacts 
HF.WSIPreProcessing.main()

# Creating the training data
HF.DataGenerator.main()

# Training the model
HF.ModelTrainer.main()

