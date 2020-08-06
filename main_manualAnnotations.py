'''

This script specifically work for processing the samples when there are maunal
annotations.

Processing includes:
    - Extracting the sample from the ndpi file at the given zoom
    - Extracting any hand annotated drawings of the vessels and locations
    identified for alignment
    - Aligning the tissue
    - Extracting any particular features as defined by segSection

'''

from HelperFunctions import *

# dataHome is where all the directories created for information are stored 
dataTrain = '/Volumes/Storage/H653A_11.3new/'

# dataTrain = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/HistologicalTraining2/'

# research drive access from HPC
# dataTrain = '/eresearch/uterine/jres129/AllmaterialforBoydpaper/ResultsBoydpaper/ArcuatesandRadials/NDPIsegmentations/'

# research drive access via VPN
# dataTrain = '/Volumes/resabi201900003-uterine-vasculature-marsden135/All material for Boyd paper/Results Boyd paper/Arcuates and Radials/NDPI segmentations/'


size = 3
kernel = 50
name = ''
portion = 0.2

print("\n----------- SegLoad -----------")
# extract the manual annotations
SegLoad(dataTrain, name)

print("\n----------- WSILoad -----------")
# extract the tif file of the specified size
WSILoad(dataTrain, name, size)

print("\n----------- maskMaker -----------")
# from the manually annotated blood vessels, make them into masks
maskMaker(dataTrain, name, size)

print("\n----------- WSIExtract -----------")
# extract ONLY the blood vessels from the sample (masked)
WSIExtract(dataTrain, name, size)

print("\n----------- SegmentExtract -----------")
# extract only the sample from the whole slide
# if there are multiple specimens, seperate these into a, b, c... samples (including info)
sampleExtract(dataTrain, name, size, True)

print("\n----------- AignSegments -----------")
# align all the specimens 
align(dataTrain, name, size, True)  

print("\n----------- FeatureExtraction -----------")
# propogate an annotated feature through the aligned tissue and extract
featExtract(dataTrain, name, size)
