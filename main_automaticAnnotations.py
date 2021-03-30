'''

This script performs normal tissue extraction but is specifically designed to 
align tissues which have no annotations

Processing includes:
    - Extracting the sample from the ndpi file at the given zoom
    - Creating features for each tissue sample 
    - Aligning the tissue
    - Allowwing for manual extracting of any structures in the samples

'''

from HelperFunctions.Utilities import nameFromPath
import multiprocessing
from HelperFunctions import *

if __name__ == "__main__":

    multiprocessing.set_start_method('spawn')


    # dataHome is where the ndpi files are stored
    dataHome = '/Volumes/USB/Testing/'
    dataHome = '/eresearch/uterine/jres129/BoydCollection/H710C_6.1/'
    dataHome = '/Volumes/USB/ANHIR/TargetTesting/COAD_17/'
    dataHome = '/Volumes/USB/H1029a/'
    dataHome = '/Volumes/USB/H750A_7.0/'
    dataHome = '/Volumes/USB/H671A_18.5/'
    dataHome = '/Volumes/USB/H673A_7.6/'
    dataHome = '/Volumes/USB/H653A_11.3/'
    dataHome = '/Volumes/USB/H653A_11.3/'
    dataHome = '/Volumes/USB/H710C_6.1/'
    dataHome = '/Volumes/USB/H710B_6.1/'

    dataHomes = [
    # '/Volumes/USB/H653A_11.3/',
    # '/Volumes/USB/H671A_18.5/',
    # '/Volumes/USB/H671B_18.5/',
    # '/Volumes/USB/H710B_6.1/',
    # '/Volumes/USB/H710C_6.1/',
    # '/Volumes/USB/H673A_7.6/',
    # '/Volumes/USB/H750A_7.0/',
    '/Volumes/USB/H1029A_8.4/'
    ]

    # research drive access via VPN
    # dataHome = '/Volumes/resabi201900003-uterine-vasculature-marsden135/Boyd collection/H1029A_8.4/'

    # zoom level to use
    size = 3

    # resolution of the baseline image to be extratcted from the full scale image
    res = 0.2

    # number of cores to use. If set to False then will serialise and allow
    # for debugging (if 1 is used it still processes with the multiprocessing
    # functions) 
    cpuNo = 4    

    for dataHome in dataHomes:

        print("\n\n #############--- Processing " + dataHome.split("/")[-1] + " ---#############\n\n")
        
        print("\n----------- WSILoad ---------")
        # extract the tif file of the specified size
        # WSILoad(dataHome, size, cpuNo)
        
        print("\n----------- smallerTif -----------")
        # create jpeg images of all the tifs and a single collated pdf
        # downsize(dataHome, size, res, cpuNo)

        print("\n----------- specID -----------")
        # extract the invidiual sample from within the slide
        # specID(dataHome, size, cpuNo)

        print("\n----------- featFind -----------")
        # identify corresponding features between samples 
        # featFind(dataHome, size, cpuNo, featMin = 40, gridNo = 1, dist = 50)

        print("\n----------- AignSegments -----------") 
        # align all the samples
        align(dataHome, size, cpuNo, fullScale=False, errorThreshold=300)
        
        print("\n----------- FixSamples -----------")
        # fix any samples which were not aligned properly 
        print("Examine the samples (for example as as a stack in ImageJ) and assess ", \
            "if there are any samples poorly aligned. Type the name of the problematic", \
                "TARGET samples here, then press enter twice to continue. If there are", \
                    "no problematic samples just press enter twice")
        n = 0
        while True:
            samples = []
            while True:
                sample = input("Sample name " + str(n) + ": ")
                if sample == "":
                    break
                samples.append(sample)
                n += 1
            if len(samples) > 0:
                fixit(dataHome, size, cpuNo, samples)
            else:
                break
        
    
    print("\n----------- FeatureExtraction -----------")
    # from the whole sample, propogate user chosen features and align these 
    # NOTE this needs to be updated to use EITHER linear or NL images + 
    # use PCC to help identify better feature trajectory 
    # number of features to extract from the aligned samples for higher
    # resolution analysis
    # eatures = input("How many features do you want to manually extract: ")
    # fullMatchingSpec(dataHome, size, features, cpuNo)
    
