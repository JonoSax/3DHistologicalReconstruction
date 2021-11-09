'''
This script takes all the modifications that were performed on the baseline image
and applies them to the full-scale tif images
'''

from HelperFunctions.nonRigidAlign import nonRigidDeform
from HelperFunctions.Utilities import getSampleName, exactBound
from HelperFunctions.SP_AlignSamples import getSpecShift, transformSamples
from HelperFunctions.SP_SpecimenID import imgStandardiser
from multiprocessing import Pool
from glob import glob
import cv2
from itertools import repeat

if __name__ == "__main__":

    size = 1.25
    res = 0.4
    scale = 1/res
    cpuNo = 20

    dataHomes = [
    # '/Volumes/USB/H653A_11.3/',
    # '/Volumes/USB/H671A_18.5/',
    # '/Volumes/USB/H671B_18.5/',
    # '/Volumes/USB/H673A_7.6/',
    # '/Volumes/USB/H710B_6.1/',
    #'/Volumes/USB/H710C_6.1/',
    # '/Volumes/USB/H750A_7.0/',
    # '/Volumes/USB/H1029A_8.4/'
    '/eresearch/uterine-vasculature-marsden135/mmad610/BoydCollection/mm_test/'
    ]
    
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
    

    for d in dataHomes:

        # define all the directories where the necessary information is stored
        datasrc = d + str(size) + "/"
        imgsrc = datasrc + "tifFiles/"
        baselineAlignedSample = datasrc + "alignedSamples/"
        imgMaskedSamples = datasrc + "fullScaleMaskedSamples/"
        featureInfoPath = datasrc + "info/"
        baselineReAlignedSample = datasrc + "RealignedSamples/"
        alignedSamples = datasrc + "fullScaleAlignedSamples/"
        NLfeatureInfoPath = datasrc + "infoNL/"
        RealignedSamples = datasrc + "fullScaleReAlignedSamples/"
        featureStore = datasrc + "FeatureSections/"
        NLAlignedSamples = datasrc + "fullScaleNLAlignedSamples/"
        NLAlignedSamplesBound = datasrc + "/fullScaleNLAlignedSamplesBound/"
        
        # apply the masks onto the tif images and normalise the colours
        print("------ Masking ------")
        imgref = cv2.imread(getSampleName(d, "refimg.png"))
        imgMasked = datasrc + "maskedSamples/masks/"
        masks = sorted(glob(imgMasked + "*.pbm"))
        
        with Pool(processes = cpuNo) as pool:
            pool.starmap(imgStandardiser, zip(repeat(imgMaskedSamples), masks, repeat(imgsrc), repeat(imgref), repeat(scale)))

        print("------ Linear regisration ------")
        # perform the first iteration of the aligned samples
        maxShape, minShift = getSpecShift(featureInfoPath)
        samples = sorted(glob(imgMaskedSamples + "*.tif"))
        with Pool(processes = cpuNo) as pool:
            pool.starmap(transformSamples, zip(samples, repeat(maxShape), repeat(minShift), repeat(featureInfoPath), repeat(alignedSamples), repeat(False), repeat(2.5)))

        # perform the re-linear alignment with the NL features
        print("------ Linear re-registration ------")
        maxShape, minShift = getSpecShift(NLfeatureInfoPath)
        samples = sorted(glob(alignedSamples + "*.tif"))
        with Pool(processes = cpuNo) as pool:
            pool.starmap(transformSamples, zip(samples, repeat(maxShape), repeat(minShift), repeat(NLfeatureInfoPath), repeat(RealignedSamples), repeat(False), repeat(2.5)))

        # perform NL warping
        print("------ NL registration ------")
        nonRigidDeform(RealignedSamples, NLAlignedSamples, featureStore, scl = scale, prefix = "tif")
        exactBound(NLAlignedSamples, "tif", dest = NLAlignedSamplesBound)
