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
name = 'H653A_09'
portion = 0.2

# create the dictionary of the jobs to perform
jobs = {}

# tasks being parallelised
tasks = ['SegmentLoad', 'WSILoad', 'MaskMaker', 'WSIExtract', 'targetTissue']
jobs[tasks[0]] = {}
jobs[tasks[1]] = {}
jobs[tasks[2]] = {}
jobs[tasks[3]] = {}
# jobs[tasks[4]] = {}


'''
# create the jobs for parallelisation, ENSURING the jobs are done in the correct order
with concurrent.futures.ProcessPoolExecutor() as executor:
    ## Extract the raw annotation and feature information
    [executor.submit(SegmentLoad.readannotations, dataTrain, s) for s in specimens]

    ## Extract the tif file of the given size 
    [executor.submit(WSILoad.load, dataTrain, s, size) for s in specimens]

    ## Create the masks of the identified vessels for the given size chosen
    [executor.submit(MaskMaker.maskCreator, dataTrain, s, size) for s in specimens]

    ## Extract the identified vessels from the samples
    [executor.submit(WSIExtract.segmentation, dataTrain, s, size) for s in specimens]

## create quadrants of the target tissue from the extracted tissue
# jobs[tasks[4]][s] = Process(target=targetTissue.quadrant, args=(dataTrain, s, size, kernel))
'''


# create the jobs for parallelisation, ENSURING the jobs are done in the correct order
for s in specimens[0:2]:
    ## Extract the raw annotation and feature information
    jobs[tasks[0]][s] = Process(target=SegmentLoad.readannotations, args=(dataTrain, s))

    ## Extract the tif file of the given size 
    jobs[tasks[1]][s] = Process(target=WSILoad.load, args=(dataTrain, s, size))

    ## Create the masks of the identified vessels for the given size chosen
    jobs[tasks[2]][s] = Process(target=MaskMaker.maskCreator, args=(dataTrain, s, size))

    ## Extract the identified vessels from the samples
    jobs[tasks[3]][s] = Process(target=WSIExtract.segmentation, args=(dataTrain, s, size))

    ## create quadrants of the target tissue from the extracted tissue
    # jobs[tasks[4]][s] = Process(target=targetTissue.quadrant, args=(dataTrain, s, size, kernel))

# Each function in parallel, sequentially
time = {}
for t in jobs:
    print("\n----------- " + t + " -----------")
    for s in jobs[t]:
        jobs[t][s].start()
    
    for s in jobs[t]:
        jobs[t][s].join()

## Align each specimen to reduce the error between slices
print("\n----------- segmentID -----------")
SegmentID.align(dataTrain, name, size)

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
