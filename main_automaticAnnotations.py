'''

This script performs normal tissue extraction but is specifically designed to 
align tissues which have no annotations

Processing includes:
    - Extracting the sample from the ndpi file at the given zoom
    - Creating features for each tissue sample 
    - Aligning the tissue
    - Allowwing for manual extracting of any structures in the samples

'''

import multiprocessing
from HelperFunctions import *

# dataHome is where the ndpi files are stored
dataHome = '/Volumes/USB/H1029a/'
dataHome = '/Volumes/USB/H673A_7.6/'
dataHome = '/Volumes/USB/H710C_6.1/'
dataHome = '/Volumes/Storage/H653A_11.3new/'
dataHome = '/Volumes/USB/H710B_6.1/'
dataHome = '/Volumes/USB/Test/'


# research drive access via VPN
# dataHome = '/Volumes/resabi201900003-uterine-vasculature-marsden135/Boyd collection/H1029A_8.4/'

# resolution scale to use (0 is full resolution, 1 is half etc.)
size = 3

# resolution of the smaller image to be extratcted from the full scale image
res = 0.2

# NOTE depreceated, needs to be removed
name = ''

# number of cores to use. If set to False then will serialise and allow
# for debugging (if 1 is used it still processes with the multiprocessing
# functions)
cpuNo = 6

# number of features to extract from the aligned samples for higher
# resolution analysis
features = 5

if __name__ == "__main__":

    multiprocessing.get_start_method('spawn')

    print("\n----------- WSILoad -----------")
    # extract the tif file of the specified size
    WSILoad(dataHome, name, size)

    print("\n----------- smallerTif -----------")
    # create jpeg images of all the tifs and a single collated pdf
    # smallerTif(dataHome, name, size, res, cpuNo)

    print("\n----------- specID -----------")
    # extract the invidiual sample from within the slide
    # specID(dataHome, name, size, cpuNo)

    print("\n----------- featFind -----------")
    # identify corresponding features between samples 
    featFind(dataHome, name, size, cpuNo)

    print("\n----------- AignSegments -----------") 
    # align all the samples 
    align(dataHome, name, size, cpuNo)

    print("\n----------- FeatureExtraction -----------")
    # from the whole sample, select features and create a propogated stack 
    fullMatchingSpec(dataHome, size, features, cpuNo)
