'''
This is the main script extracting the image values from the WSI which are 
identified from manual segmenetaiton and training a NN on this data for 
segment identification

'''
from HelperFunctions import *
from HelperFunctions.Utilities import nameFromPath
from glob import glob
import multiprocessing
from multiprocessing import Process
from time import perf_counter


# ---------- THINGS TO DO ----------
# make the parallelisation imbedded in the functions rather than called in main. 
    # ENSURE that parallelisation is an option to allow for easier debugging


'''
Folder location of slices
Extent of training (epochs, batch)
'''

# dataHome is where all the directories created for information are stored 
dataTrain = '/Volumes/Storage/H653A_11.3new/'

# dataTrain = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/HistologicalTraining2/'

# research drive access from HPC
# dataTrain = '/eresearch/uterine/jres129/AllmaterialforBoydpaper/ResultsBoydpaper/ArcuatesandRadials/NDPIsegmentations/'

# research drive access via VPN
# dataTrain = '/Volumes/resabi201900003-uterine-vasculature-marsden135/All material for Boyd paper/Results Boyd paper/Arcuates and Radials/NDPI segmentations/'

# get all the ndpi files that are to be processed
specimens = sorted(nameFromPath(glob(dataTrain + "*.ndpi")))

size = 3
kernel = 50
name = ''
portion = 0.2

# create the dictionary of the jobs to perform
jobs = {}

# tasks that are being parallelised in this script call
tasksDone = ['PR_SegmentLoad', 'PR_WSILoad', 'SP_MaskMaker', 'CI_WSIExtract']

# all tasks the can be parallelised
allTasks = ['PR_SegmentLoad', 'PR_WSILoad', 'SP_MaskMaker', 'CI_WSIExtract', 'CI_SegmentExtraction', 'CI_targetTissue']

# create dictionary containg the jobs to be done
for t in tasksDone:
    jobs[t] = {}

# create the jobs for parallelisation
for s in specimens:

    # these are all the tasks which can be parallelised ATM. They will ONLY be performed if
    # loaded into the tasksDone list
    for t in tasksDone:
        if t == 'PR_SegmentLoad':
            ## Extract the raw annotation and feature information
            jobs[t][s] = Process(target=PR_SegmentLoad.readannotations, args=(dataTrain, s))

        elif t == 'PR_WSILoad':
            ## Extract the tif file of the given size 
            jobs[t][s] = Process(target=PR_WSILoad.load, args=(dataTrain, s, size))

        elif t == 'SP_MaskMaker':
            ## Create the masks of the identified vessels for the given size chosen
            jobs[t][s] = Process(target=SP_MaskMaker.maskCreator, args=(dataTrain, s, size))

        elif t == 'CI_WSIExtract':
            ## Extract the identified vessels from the samples
            jobs[t][s] = Process(target=CI_WSIExtract.segmentation, args=(dataTrain, s, size))

        elif t == 'CI_targetTissue':
            ## create quadrants of the target tissue from the extracted tissue
            jobs[t][s] = Process(target=CI_targetTissue.quadrant, args=(dataTrain, s, size, kernel))

# Run the function in parallel, sequentially
for t in jobs:
    print("\n----------- " + t + " -----------")
    for s in jobs[t]:
        jobs[t][s].start()
    
    for s in jobs[t]:
        jobs[t][s].join()

# serialise for debugging
if len(tasksDone) == 1:

    PR_SegmentLoad.readannotations(dataTrain, name)

    PR_WSILoad.load(dataTrain, name, size)

    SP_MaskMaker.maskCreator(dataTrain, name, size)

    CI_WSIExtract.segmentation(dataTrain, name, size)



## Align each specimen to reduce the error between slices, 
# NOTE: either parallelise within this function or create a new function for the extraction of the slices
print("\n----------- SegmentExtract -----------")
SP_SingleSegmentExtract.extract(dataTrain, name, size, True)      # extracting the individual slices can technically
                                            # be parallelised, but the fitting must be sequential

print("\n----------- AignSegments -----------")
SP_AlignSamples.align(dataTrain, name, size, True)  

print("\n----------- FeatureExtraction -----------")
# propogate a segSection feature selected through an entire stack of aligned samples
CI_FeatureExtraction.extract(dataTrain, name, size)



'''
# Creating the training data --> NOTE every time it does this it creates a replaces the previous testing/training data
DataGenerator.main(dataTrain, portion, 'vessel')

# Training the model --> note this should never be commented out, only set to False
modelDir, class2feat = ModelTrainer.train(dataTrain, name = 'text', epoch=4, train = True)

# ModelEvaluater.main(dataTrain, dataAssess, modelDir, class2feat)

'''
