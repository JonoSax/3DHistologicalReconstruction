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
dataHome = '/Volumes/Storage/H653A_11.3new/'
dataHome = '/Volumes/USB/H750A_7.0/'
dataHome = '/Volumes/USB/H671A_18.5/'
dataHome = '/Volumes/USB/H710B_6.1/'
dataHome = '/Volumes/USB/H673A_7.6/'
dataHome = '/Volumes/Storage/H653A_11.3/'
dataHome = '/Volumes/Storage/H710C_6.1/'
dataHome = '/Volumes/USB/Test/'

# NOTE 
#   bug check when there are no images


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
features = 3

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')
    
    print("\n----------- WSILoad ---------")
    # extract the tif file of the specified size
    WSILoad(dataHome, name, size)
    
    print("\n----------- smallerTif -----------")
    # create jpeg images of all the tifs and a single collated pdf
    downsize(dataHome, size, res, cpuNo)

    print("\n----------- specID -----------")
    # extract the invidiual sample from within the slide
    specID(dataHome, name, size, cpuNo)

    print("\n----------- featFind -----------")
    # identify corresponding features between samples 
    featFind(dataHome, name, size, cpuNo)
    print("\n----------- AignSegments -----------") 
    # align all the samples
    savingTif = True
    align(dataHome, size, cpuNo, savingTif)
    
    print("\n----------- FixSamples -----------")
    # fix any samples which were not aligned properly 
    print("Examine the samples (for example as as a stack in ImageJ) and assess if there are any samples poorly aligned. Type the name of the problematic samples here, then press enter twice to continue")
    samples = []
    n = 0
    while True:
        sample = input("Sample name " + str(n) + ": ")
        if sample == "":
            break
        samples.append(sample)
        n += 1
    fixit(dataHome, size, cpuNo, samples)

    print("\n----------- FeatureExtraction -----------")
    # from the whole sample, propogate user chosen features and align these 
    fullMatchingSpec(dataHome, size, features, cpuNo)
