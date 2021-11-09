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

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')

    '''
    dataHomes = [
    # '/Volumes/USB/H653A_11.3/',
    # '/Volumes/USB/H671A_18.5/',
    # '/Volumes/USB/H671B_18.5/',
    # '/Volumes/USB/H673A_7.6/',
    # '/Volumes/USB/H710B_6.1/',
    # '/Volumes/USB/H710C_6.1/',
    # '/Volumes/USB/H750A_7.0/',
    # '/Volumes/USB/H1029A_8.4/'
    ]
    '''
    dataHomes = [
    # '/eresearch/uterine/jres129/BoydCollection/H653A_11.3/',
    # '/eresearch/uterine/jres129/BoydCollection/H671A_18.5/',
    # '/eresearch/uterine/jres129/BoydCollection/H671B_18.5/',
    # '/eresearch/uterine/jres129/BoydCollection/H710B_6.1/',
    # '/eresearch/uterine/jres129/BoydCollection/H710C_6.1/',
    # '/eresearch/uterine/jres129/BoydCollection/H673A_7.6/',
    # '/eresearch/uterine/jres129/BoydCollection/H750A_7.0/',
    # '/eresearch/uterine/jres129/BoydCollection/H1029A_8.4/'
    '/eresearch/uterine-vasculature-marsden135/mmad610/BoydCollection/mm_test/'
    ]

    # zoom level to use
    size = 1.25

    # resolution of the baseline image to be extratcted from the full scale image
    res = 0.4

    # number of cores to use. If set to 1 then will serialise and allow
    # for debugging (if 1 is used it still processes with the multiprocessing
    # functions) 
    cpuNo = 20

    errorThreshold = 300

    for dataHome in dataHomes:

        print("\n\n #############--- Processing " + dataHome.split("/")[-2] + " ---#############\n\n")
        
        print("\n----------- WSILoad ---------")
        # extract the tif file of the specified size
        WSILoad(dataHome, size, cpuNo)
        
        print("\n----------- smallerTif -----------")
        # create jpeg images of all the tifs and a single collated pdf
        downsize(dataHome, size, res, cpuNo)
        
        print("\n----------- specID -----------")
        # extract the invidiual sample from within the slide
        specID(dataHome, size, cpuNo)

        print("\n----------- featFind -----------")
        # identify corresponding features between samples 
        featFind(dataHome, size, cpuNo, featMin = 40, dist = 50)

        print("\n----------- AignSegments -----------") 
        # align all the samples
        align(dataHome, size, cpuNo, errorThreshold=errorThreshold)
        
        print("\n----------- FixSamples -----------")
        # fix any samples which were not aligned properly 
        n = 0
        while True:
            samples = []
            while True:
                print("\nExamine the samples (for example as as a stack in ImageJ) and assess ", \
                    "if there are any samples poorly aligned. Type the name of the problematic", \
                    "TARGET samples here, then press enter twice to continue. If there are", \
                    "no problematic samples just press enter twice")
                sample = input("Sample name " + str(n) + ": ")
                if sample == "":
                    break
                samples.append(sample)
                n += 1
            if len(samples) > 0:
                fixit(dataHome, size, cpuNo, samples, errorThreshold)
            else:
                break
        
        # change the multiprocessing method for the nonRigidAlign
        multiprocessing.set_start_method("fork", force=True)
        
        # perform a non-rigid alignment
        nonRigidAlign(dataHome, size, cpuNo = cpuNo, \
        featsMin = 10, dist = 30, featsMax = 100, errorThreshold = 200, \
            distFeats = 50, sect = 100, selectCriteria = "length", \
                flowThreshold = 0.05, fixFeatures = False, plot = False)
    
