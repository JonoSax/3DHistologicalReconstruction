'''
This script takes all the modifications that were performed on the baseline image
and applies them to the full-scale tif images
'''

from nonRigidAlign import nonRigidAlign, nonRigidDeform
from HelperFunctions.Utilities import nameFromPath, txtToDict
from HelperFunctions.SP_AlignSamples import getSpecShift, transformSamples
from HelperFunctions.SP_SpecimenID import imgStandardiser
import multiprocessing
from HelperFunctions import *
from glob import glob
import cv2
import tifffile as tifi

if __name__ == "__main__":

    size = 3

    dataHomes = [
    '/Volumes/USB/H653A_11.3/',
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
    ]
    '''

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

        
        '''
        # apply the masks onto the tif images and normalise the colours
        imgref = cv2.imread(d + "refimg.png")
        imgMasked = datasrc + "maskedSamples/masks/"
        masks = sorted(glob(imgMasked + "*.pbm"))
        for m in masks:
            imgStandardiser(imgMaskedSamples, m, imgsrc, imgref)
        '''
        
        # perform the first iteration of the aligned samples
        maxShape, minShift = getSpecShift(featureInfoPath)
        samples = sorted(glob(imgMaskedSamples + "*.tif"))
        for sample in samples:
            transformSamples(sample, maxShape, minShift, featureInfoPath, alignedSamples, False)

        # bound the images the SAME as the baseline aligned images
        bound = txtToDict(baselineAlignedSample + "_boundingPoints.txt", int)
        imgs = glob(alignedSamples + "*.tif")
        x, y = bound
        for i in imgs:
            name = nameFromPath(i)
            img = cv2.imread(i)
            imgB = img[y[0]:y[1], x[0]:x[1], :]
            tifi.imwrite(i, imgB)

        # perform the re-linear alignment with the NL features
        maxShape, minShift = getSpecShift(NLfeatureInfoPath)
        samples = sorted(glob(alignedSamples + "*.tif"))
        for sample in samples:
            transformSamples(sample, maxShape, minShift, featureInfoPath, RealignedSamples, False)

        # bound the images the SAME as the baseline realigned images
        bound = txtToDict(baselineReAlignedSample + "_boundingPoints.txt", int)
        imgs = glob(RealignedSamples + "*.tif")
        x, y = bound
        for i in imgs:
            name = nameFromPath(i)
            img = cv2.imread(i)
            imgB = img[y[0]:y[1], x[0]:x[1], :]
            tifi.imwrite(i, imgB)

        # perform NL warping
        nonRigidDeform(RealignedSamples, NLAlignedSamples, featureStore, scl = 1, prefix = "tif")

