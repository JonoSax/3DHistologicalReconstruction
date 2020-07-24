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
# from concurrent.futures import ProcessPoolExecutor as executor
import concurrent.futures

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
dataHome = '/Volumes/USB/H653A_11.3new/'

dataHome = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/HistologicalTraining2/'

# research drive access from HPC
# dataHome = '/eresearch/uterine/jres129/AllmaterialforBoydpaper/ResultsBoydpaper/ArcuatesandRadials/NDPIsegmentations/'

# research drive access via VPN
# dataHome = '/Volumes/resabi201900003-uterine-vasculature-marsden135/All material for Boyd paper/Results Boyd paper/Arcuates and Radials/NDPI segmentations/'

# dataTrain is where the ndpi and ndpa files are stored 
dataTrain = dataHome

# get all the ndpi files that are to be processed
specimens = sorted(nameFromPath(glob(dataTrain + "*.ndpi")))

# data directory containing the wsi images to be assessed
dataAssess = dataHome + "samples/"

size = 3
kernel = 50
name = ''
portion = 0.2

# create the dictionary of the jobs to perform
jobs = {}

# tasks that are being parallelised in this script call
tasksDone = ['SegmentLoad', 'WSILoad', 'MaskMaker', 'WSIExtract']

# all tasks the can be parallelised
allTasks = ['SegmentLoad', 'WSILoad', 'MaskMaker', 'WSIExtract', 'targetTissue']

# create dictionary containg the jobs to be done
for t in tasksDone:
    jobs[t] = {}

# create the jobs for parallelisation
for s in specimens:

    # these are all the tasks which can be parallelised ATM. They will ONLY be performed if
    # loaded into the tasksDone list
    for t in tasksDone:
        if t == 'SegmentLoad':
            ## Extract the raw annotation and feature information
            jobs[t][s] = Process(target=SegmentLoad.readannotations, args=(dataTrain, s))

        elif t == 'WSILoad':
            ## Extract the tif file of the given size 
            jobs[t][s] = Process(target=WSILoad.load, args=(dataTrain, s, size))

        elif t == 'MaskMaker':
            ## Create the masks of the identified vessels for the given size chosen
            jobs[t][s] = Process(target=MaskMaker.maskCreator, args=(dataTrain, s, size))

        elif t == 'WSIExtract':
            ## Extract the identified vessels from the samples
            jobs[t][s] = Process(target=WSIExtract.segmentation, args=(dataTrain, s, size))

        elif t == 'targetTissue':
            ## create quadrants of the target tissue from the extracted tissue
            jobs[t][s] = Process(target=targetTissue.quadrant, args=(dataTrain, s, size, kernel))

# Run the function in parallel, sequentially
for t in jobs:
    print("\n----------- " + t + " -----------")
    for s in jobs[t]:
        jobs[t][s].start()
    
    for s in jobs[t]:
        jobs[t][s].join()

'''
SegmentLoad.readannotations(dataTrain, name)

WSILoad.load(dataTrain, name, size)

MaskMaker.maskCreator(dataTrain, name, size)

WSIExtract.segmentation(dataTrain, name, size)
'''
## Align each specimen to reduce the error between slices
print("\n----------- segmentID -----------")
SegmentID.align(dataTrain, name, size)      # extracting the individual slices can technically
                                            # be parallelised, but the fitting must be sequential


# NOTE to do
# create a function which will extract from the orientated tissue, the segSections annoated
# accordingly: SegSection_tr_# and SegSection_bl_# (where # is a or b)
# these will be extracted as tif images and put into normalised image sizes to perform a 
# small scale 3D volume segmentation


# creat a stack from the aligned images
print("\n----------- stack -----------")
# stackAligned.stack(dataTrain, name, size)
## Extract the target tissue from the tif files 

# targetTissue.quadrant(dataTrain, name, size, kernel)

'''
# Creating the training data --> NOTE every time it does this it creates a replaces the previous testing/training data
DataGenerator.main(dataTrain, portion, 'vessel')

# Training the model --> note this should never be commented out, only set to False
modelDir, class2feat = ModelTrainer.train(dataTrain, name = 'text', epoch=4, train = True)

# ModelEvaluater.main(dataTrain, dataAssess, modelDir, class2feat)

'''
