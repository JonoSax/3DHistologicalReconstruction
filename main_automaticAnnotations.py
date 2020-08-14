'''

This script performs normal tissue extraction but is specifically designed to 
align tissues which have no annotations

Processing includes:
    - Extracting the sample from the ndpi file at the given zoom
    - Creating features for each tissue sample 
    - Aligning the tissue
    - Extracting any particular features as defined by segSection (or )

'''

from HelperFunctions import *

# dataHome is where all the directories created for information are stored 
dataTrain = '/Volumes/USB/H653/'

# dataTrain = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/HistologicalTraining2/'

# research drive access from HPC
# dataTrain = '/eresearch/uterine/jres129/AllmaterialforBoydpaper/ResultsBoydpaper/ArcuatesandRadials/NDPIsegmentations/'

# research drive access via VPN
# dataTrain = '/Volumes/resabi201900003-uterine-vasculature-marsden135/All material for Boyd paper/Results Boyd paper/Arcuates and Radials/NDPI segmentations/'


size = 3
kernel = 50
name = ''
portion = 0.2

print("\n----------- WSILoad -----------")
# extract the tif file of the specified size
WSILoad(dataTrain, name, size)

print("\n----------- smallerTif -----------")
# create jpeg images of all the tifs and a single collated pdf
smallerTif(dataTrain, name, size)

print("\n----------- specID -----------")
# identifies the sample within the slide from the jpegs created
specID(dataTrain, name, size)

print("\n----------- featFind -----------")
# identifies corresponding features per samples 
featFind(dataTrain, name, size)

print("\n----------- SegmentExtract -----------") # --> NOTE add size of all the shapes and make these functions adjust the size of the files for this image size
# extract only the sample from the whole slide
# if there are multiple specimens, seperate these into a, b, c... samples (including info)
# sampleExtract(dataTrain, name, size, True)

print("\n----------- AignSegments -----------") # --> NOTE add size of all the shapes and make these functions adjust the size of the files for this image size
# align all the specimens 
align(dataTrain, name, size, True)

# NOTE to complete
# Identify a feature to propogate through the entire tissue
# SP_SampleFinder()

print("\n----------- FeatureExtraction -----------")
# propogate an annotated feature through the aligned tissue and extract
# featExtract(dataTrain, name, size)

# NOTE run the results of the feat extraction THROUGH the align function again to 
# provide the fine scale alignment for the tissue segment
'''

How this will work is for the first slice where the feature is chosen, in the next slice
a slightly larger area is selected to perform feature mapping from the original tissue. 

    NOTE I think it would be ideal for the section selected if you can go and essentailly 
    outline EXACTLY what the issue is, rather than just using a block of tissue, so that 
    subsequent feature matching is on the actual target tissue rather than just noise as well

Once this new image is found the amount it has shift from the ORIGINAL image is used to expaned
the search area and this new feature is identifed. 

In the next slice the NEW feature is used to identify the NEXT translation of the slice to find 
the next expanded search area. 

This will involve performing a sift operation on each area and performing a translation (probably 
not rotation.... ) of the tissue and using this translation to expand the search area AND align 
the current tissue sample

THIS ASSUMES THAT THE Z-SHIFT IS REASONABLY CONSISTENT BETWEEN SLICES

'''