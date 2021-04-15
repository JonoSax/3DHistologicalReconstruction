'''
This script allows for labelling, training and segmentation of tissue.
'''

import numpy as np
import pandas as pd
from HelperFunctions.Utilities import dictToTxt, dirMaker, nameFromPath, printProgressBar, txtToDict
from glob import glob
from random import random, shuffle
from shutil import copyfile
import tensorflow as tf
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dropout, Dense
from multiprocessing import Pool
from itertools import repeat
import matplotlib.pyplot as plt

def labelData(imgDirs, classes = ["m", "d", "v"]):

    '''
    Label all the features which have been extracted
    '''

    # create a dictionary which contains all the labels made
    labels = {}

    featIDs = glob(imgDirs + "*")
    featIDOrder = np.argsort([-len(glob(f + "/*.png")) for f in featIDs])

    labelStore = []

    # label the features based on length (ie the longest features are labelled first)
    for n in featIDOrder:
        f = featIDs[n]
        featID = nameFromPath(f)
        refImg = cv2.imread(f + "/_referenceImage.jpg")
        finImg = cv2.imread(f + "/_finalImg.jpg")
        sects = glob(f + "/*.png")

        # get 5 sections to show (shuffle each time)
        imgs = []
        shuffle(sects)
        sectsToShow = sects[:5]
        for s in sectsToShow:
            img = cv2.imread(s)
            x, y, c = img.shape
            imgPlate = np.zeros([x+20, y+20, c]).astype(np.uint8)
            imgPlate[10:-10, 10:-10, :] = img
            imgs.append(imgPlate)

        sectsStack = np.vstack(imgs)

        ys, xs, _ = sectsStack.shape
        yr, xr, _ = refImg.shape

        xrs = int(xs * yr/ys)

        # resize the section stack to be the same height as the reference image
        sectsStackR = cv2.resize(sectsStack, (xrs, yr))
        imgComb = np.hstack([refImg, finImg, sectsStackR])

        imgComb = cv2.cvtColor(imgComb, cv2.COLOR_BGR2RGB)

        # show the tissue being labelled
        _, ax = plt.subplots()
        plt.axis("off")    
        plt.imshow(imgComb)
        ax.set_title("First sample, last sample, random selection of features. Close this window and type in the terminal the key for the tissue types when ready.")
        plt.show()

        print("\nThere are " + str(len(sects)) + " sections in feature ID " + featID)
        labelVals, labelCount = np.unique(sum(list(labels.values()), []), return_counts = True)
        print("There are " + str(labelCount) + " for " + str(labelVals))
        print("What tissue types are present in feature " + str(featID))
        print("type DONE if you want to save and stop labelling")
        for c in classes:
            print(c + ", ", end = "")
        featInputs = input(" : ")
        # if there is no matching tissue type then dont add to labels
        if featInputs == "":
            continue
        elif featInputs.lower() == "done":
            break
        featSplit = featInputs.split(",")
        feats = [f.replace(" ", "") for f in featSplit]
        labels[featID] = feats
        labelStore += feats

        # NOTE if there was adaptive learning, where as features are trained the 
        # the approximations of what they might be are used to further assist trainig


    dictToTxt(labels, imgDirs + "labels.txt")
    print("Labels saved as: " + imgDirs + "labels.txt")

def getLabels(imgDir, infoDir):

    '''
    Get the labels from the annotationed folders and convert into a sparse matrix

        Inputs:\n
    imgDir, directory of the images to perform the search for labels on \n
    infoDir, directory to save the csv of the labels

        Outputs:\n
    valK, pandas dataframe of the labels, also saved as a csv
    '''

    # NOTE ssh into the drive and do os.listdir for each specimen
    # All the folder names
    vals = ['3505', '2842', '1668', '3045', '315_v,b', '3604', '554_d,g,b', '118_d,g', '743', '3_m,b', '1033', '3052', '2897', '1588', '3472', '597_d,v,b', '550_v,b', '1595', '667_v,b', '2916', '3305', '2777', '44_v,b', '588_d,v,g,b', '3411', '2883', '836', '3367', '878', '3093', '1954', '2799', '874', '3053', '1081', '2303', '3426', '1640', '3473', '225_d,g', '695_d,v,g,b', '333_v,b', '3247', '2818', '1275', '3294', '937', '862', '979', '33_d,g', '2964', '3488', '699_d,v,g,b', '929', '3448', '854', '669_m,d,g,b', '592_d,g,b', '412_d,v,b', '967', '2979', '892', '3554', '2804', '178_d,v,b', '3054', '1082', '3014', '1641', '254_d,g', '2910', '634_v,b', '214_d,v,g', '3434', '951', '3295', '641_m,d,v,g,b', '901', '3580', '1097', '2319', '3029', '2965', '276_m,d,v,g,b', '1057', '351_v,b', '943', '718', '1616', '3449', '2786', '414_m,d,g,b', '31_d,v', '3135', '3000', '2845', '104_d,g', '2805', '212_d,g,b', '3015', '264_v', '377', '1043', '129_m,b', '259_m,d,g,b', '1602', '3435', '3300', '466_d,v,b,g', '36_d,g,b', '1598', '2867', '110_d,v,g', '1938', '849', '2327', '580_d,v,b', '2973', '2006', '2787', '274_m,d,v,g,b', '3362', '3041', '34_d,v,g', '561_d,v,b', '3001', '475_d,g', '311_m,g,b', '724', '373_d,g,b', '2806', '3282', '3516', '2853', '3377', '879', '1964', '381_m,b', '3016', '3063', '347_d,g,b', '867', '2828', '3217', '863', '1804', '2927', '1066', '3363', '2934', '2795', '329_m,d,v,g,b', '76_d,v,g,b', '63_d,v,g,b', '41_d,g,b', '324_d,v,g,b', '3623', '181_v', '3484', '956', '1052', '2920', '693_d,v,b', '2781', '431_v,b', '1987', '3503', '2840', '3550', '944', '2800', '2749', '536_v,b', '940', '3145', '2161', '317_m,d,g,b', '2855', '3430', '2815', '284_m,d,g,b', '3386', '267_m,d,g,b', '1926', '169_v', '314_d,g,b', '970', '2961', '3485', '578_d,v,g,b', '2921', '3445', '567_m,b', '2001', '2782', '2877', '3266', '3591', '1988', '173_m,d,g,b', '1068', '650_d,v,b', '2936', '2801', '2983', '1_d,v,g', '3372', '3146', '206_m,d,v,g,b', '3431', '2257', '393_m,b', '328_m,d,g,b', '3292', '2998', '3205', '345_d,v,g,b', '560_d,v,g,b', '16_d,v,g', '3026', '1934', '2922', '1014', '2002', '1569', '685_m,d,g,b', '1069', '2937', '1628', '1675', '2758', '1960', '694_m,b', '98_d,v,g,b', '2444', '2817', '221_d,v,g,b', '364_m,d,v,g,b', '2864', '894', '3388', '2824', '890', '919', '3348', '3027', '1241', '915', '140_m,d,g,b', '2923', '1015', '2970', '1661', '1062', '2879', '882', '705_d,v,b', '3454', '2791', '3133', '2839', '3180', '903', '2850', '3560', '3374', '945', '57_d,v,g', '2759', '647_d,g,b', '3013', '3195', '27_d,v,g,b', '687', '3389', '395_m,b', '637', '471_m,d,b,g', '552_m,d,g,b', '2372', '3408', '2745', '3141', '738', '541_v,b', '662_d,g,b', '702_d,g,b', '2946', '53_d,g', '496_d,b,g', '35_m,b', '428_m,d,v,g,b', '116_v,b', '726', '535_d,b,g', '2866', '483_m,d,b,g', '409_v,b', '2873', '1699', '3215', '668_v,b', '60_d,g,b', '710', '2147', '756', '2888', '587_d,g,b', '1999', '1079', '790', '819', '122_v', '334_d,g,b', '3062', '519', '740', '3022', '690_d,v,g,b', '928', '1050', '899', '611_m,b', '3402', '2874', '3216', '323_d,v,g,b', '310_v,b', '152_m,d,v,g,b', '2334', '841', '297_d,v,g,b', '908', '996', '136_d,v', '38_d,v', '112_m,d,v,g,b', '2948', '183_d,v,g,b', '427_d,v,b', '2860', '2908', '3384', '2820', '352_d,g,b', '1325', '257_m,b', '1051', '1011', '652_m,b', '3490', '2229', '366_m,d,v,g,b', '2875', '3450', '934', '543_d,v,b,g', '930', '3038', '976', '369_m,d,g,b', '972', '271_v', '1073', '3370', '1632', '2755', '633_v,b', '391_m,d,g,b', '2715', '21_d,v,g,b', '3524', '960', '1088', '2956', '197_m,d,g,b', '2821', '638_m,d,b,g', '62_v,b', '3071', '1012', '346_d,v,b', '208_d,g,b', '727', '3539', '3404', '2876', '69_m,d,g,b', '3451', '3130', '2836', '2982', '1074', '3419', '244_d,v,g,b', '1589', '1827', '795', '651_v,b', '703', '295_v', '2957', '1049', '2917', '3393', '299_v,b', '858', '1707', '542_d,v,b,g', '3405', '3587', '808', '2238', '572_m,d,v,b,g', '2884', '70_d,v,g,b', '600_v,b', '55_v', '488_d,v,b', '1075', '917', '2943', '90_d,v,g', '888', '2903', '3193', '2764', '884', '2578', '3153', '517_v,b', '909', '515_m,d,v,b,g', '2958', '3347', '2870', '2038', '876', '2918', '3394', '1609', '872', '156_d,v,g,b', '3168', '198_m,b', '472_m,d,b,g', '620_v,b', '2885', '1576', '3369', '860', '689_m,d,v,g,b', '931', '59_m,d,g', '1076', '660_m,d,v,g,b', '3008', '973', '1682', '2904', '514', '3340', '3289', '245_v', '478_d,v,b', '3249', '355_d,g,b', '2586', '2265', '507_m,b', '659_m,d,g,b', '961', '594', '263_v', '2919', '176_d,v,g', '3395', '385_d,g,b', '3260', '547_v,b', '11_m,d,g,b', '1522', '18_m,b', '92_m,b', '3220', '3034', '1022', '728', '356_m,d,g,b', '518_d,b,g', '3414', '363_d,g,b', '3100', '1822', '3520', '3056', '231_m,b', '3381', '2032', '570_m,b', '2726', '2587', '708', '89_d,v,b', '2872', '51_d,v,g', '2832', '3221', '171_m,d,g,b', '2788', '143_m,d,g,b', '658_d,v,g,b', '3042', '272_v,b', '859', '358_m,d,v,g,b', '3415', '365_d,g,b', '308_d,g,b', '278_m,d,g,b', '2894', '3101', '2807', '1264', '3283', '2212', '2993', '563_m,b', '549_d,g,b', '2953', '889', '376_d,v,g,b', '516_d,v,b,g', '300_m,d,g,b', '2774', '885', '565_d,v,b', '655_d,v,g,b', '326_v,b', '1524', '407_m,b', '222_d,v,g,b', '2928', '2789', '700_v,b', '524_v,b', '10_m,b', '3423', '2669', '3018', '2954', '1645', '1046', '2914', '551_m,d,g,b', '96_m,b', '67_v,b', '3299', '924', '966', '37_d,g', '2969', '2834', '962', '2929', '450_d,v,b', '912', '188_d,v,g,b', '7_m,b', '3599', '1032', '531_d,v,b,g', '2900', '950', '415_m,d,g,b', '2809', '1547', '534_m,b', '900', '2955', '2_m,d,v,g,b', '2915', '361_v', '2962', '2776', '2641', '3486', '445_m,d,g,b', '2236', '239_m,d,g,b']
    # vals = ['3505', '2238_d,v,b', '2327_d,g,b', '315_v,b', '1960_m,d,g,b', '554_d,g,b', '118_d,g', '3_m,b', '3362_m,d,g,b', '3295_v,b', '2897', '930_d,g,b', '1076_m,d,g,b', '597_d,v,b', '550_v,b', '884_d,g,b', '1699_v,b', '2850_v,b', '667_v,b', '3063_d,v,g,b', '3305', '44_v,b', '2962_m,d,g,b', '1707_m,d,g,b', '588_d,v,g,b', '979_m,d,g,b', '841_v,b', '2874_d,g', '3604_v,b', '2836_d,v,b', '1628_m,d,g,b', '1954', '3426_v,b', '3195_d,g,b', '2236_v,b', '1588_v,b', '225_d,g', '1602_d,v,g,b', '1075_d,g,b', '695_d,v,g,b', '333_v,b', '915_d,g,b', '3045_v,b', '3539_m,d,g,b', '3029_d,v,g,b', '3283_d,v,b', '3370_v,b', '1014_m,d,g,b', '1052_m,d,g,b', '33_d,g', '699_d,v,g,b', '3093_v,b', '1088_v,b', '2993_d,v,g,b', '860_m,b', '2786_v,b', '726_d,g,b', '3414_v,b', '669_m,d,g,b', '592_d,g,b', '412_d,v,b', '1675_v,b', '2979', '892', '3216_v,b', '178_d,v,b', '2921_d,v,b', '3451_d,g,b', '2910_d,g,b', '254_d,g', '634_v,b', '214_d,v,g', '641_m,d,v,g,b', '901', '3292_d,v,b', '3402_v,b', '2853_v,b', '276_m,d,v,g,b', '738_m,d,g,b', '351_v,b', '414_m,d,g,b', '31_d,v', '3100_d,v,b', '961_m,d,g,b', '956_d,g,b', '104_d,g', '3449_v,b', '212_d,g,b', '2828_d,g,b', '264_v', '377', '129_m,b', '934_m,b', '259_m,d,g,b', '889_d,g,b', '466_d,v,b,g', '36_d,g,b', '863_m,b', '110_d,v,g', '3393_v,b', '3472_v,b', '2801_v,b', '1938', '580_d,v,b', '2957_d,v,g,b', '274_m,d,v,g,b', '34_d,v,g', '2888_v,b', '561_d,v,b', '3282_d,v,b', '3516_v,b', '475_d,g', '311_m,g,b', '373_d,g,b', '874_v', '1524_d,g,b', '2212_d,v,g,b', '3056_v,b', '2934_d,g,b', '2749_v,b', '381_m,b', '3435_d,v,g,b', '3016', '3215_d,g,b', '879_m,d,g,b', '347_d,g,b', '3015_d,g,b', '2669_m,d,g,b', '928_d,g,b', '3524_v,b', '2265_d,v,g,b', '2929_m,d,g,b', '2845_d,g,b', '3042_d,v,b', '1074_d,g,b', '2777_v,b', '872_m,b', '2840_m,d,v,g,b', '3299_d,g,b', '740_d,v,b', '329_m,d,v,g,b', '1547_v,b', '944_d,v,g,b', '76_d,v,g,b', '3289_v,b', '63_d,v,g,b', '41_d,g,b', '324_d,v,g,b', '3623', '181_v', '693_d,v,b', '819_v,b', '3168_m,d,g,b', '2745_v,b', '431_v,b', '900_v,b', '2961_d,g,b', '2821_m,d,g,b', '2303_m,d,g,b', '536_v,b', '3485_d,g,b', '2946_d,v,g,b', '2804_m,d,g,b', '2759_d,v,g,b', '2875_d,g', '317_m,d,g,b', '2855', '2860_d,g,b', '859_m,b', '284_m,d,g,b', '2832_v,b', '267_m,d,g,b', '1926', '169_v', '314_d,g,b', '3388_d,v,g,b', '3384_v,b', '578_d,v,g,b', '2948_d,v,b', '3000_v,b', '567_m,b', '912_d,v,b', '2824_d,g,b', '173_m,d,g,b', '2958_v,b', '650_d,v,b', '2936', '2983', '1_d,v,g', '3146_v,b', '2820_v,b', '2587_d,g,b', '1641_m,b', '206_m,d,v,g,b', '2800_v,b', '3550_v,b', '393_m,b', '328_m,d,g,b', '3372_v,b', '345_d,v,g,b', '560_d,v,g,b', '2943_d,g,b', '16_d,v,g', '1682_v,d,b', '718_d,v,g,b', '2788_v,b', '899_m,b', '1999_d,v,g,b', '836_m,b', '2870_m,d,g,b', '2908_d,v,b', '972_d,g,b', '685_m,d,g,b', '3405_d,v,g,b', '3180_m,d,v,g,b', '3052_d,v,g,b', '3381_m,d,g,b', '894_m,d,g,b', '694_m,b', '940_d,g,b', '2791_v,b', '1609_d,v,b', '867_v,b', '98_d,v,g,b', '2954_v,b', '2937_d,v,g,b', '221_d,v,g,b', '364_m,d,v,g,b', '3221_v,b', '3591_d,v,g,b', '3488_d,g,b', '1645_v,b', '140_m,d,g,b', '885_d,g,b', '2965_d,v,g,b', '705_d,v,b', '2904_d,g,b', '2877_m,d,v,g,b', '3260_m,d,g,b', '3484_d,g,b', '862_m,d,g,b', '1066_v,b', '57_d,v,g', '647_d,g,b', '27_d,v,g,b', '687', '2038_v,b', '2923_d,v,g,b', '3395_v,b', '950_m,d,g,b', '395_m,b', '637', '471_m,d,b,g', '3454_v,b', '1046_m,d,v,g,b', '552_m,d,g,b', '1062_d,v,b', '3554_m,d,g,b', '3419_v,b', '3490_d,v,b', '2914_m,d,g,b', '917_d,v,g,b', '1079_d,g,b', '541_v,b', '662_d,g,b', '2809_d,g', '702_d,g,b', '53_d,g', '3294_d,v,b', '496_d,b,g', '2876_m,b', '35_m,b', '428_m,d,v,g,b', '116_v,b', '724_d,g,b', '535_d,b,g', '483_m,d,b,g', '409_v,b', '2927_d,v,g,b', '2586_d,g,b', '3266_d,v,b', '3145_v,b', '3217_m,d,g,b', '668_v,b', '1668_v,b', '60_d,g,b', '3026_v,b', '2818_v,b', '2834_d,g,b', '587_d,g,b', '1827_d,g,b', '3486_d,g,b', '1595_m,b', '1598_m,d,g,b', '2916_d,v,g,b', '1988_m,b', '3153_v,b', '122_v', '334_d,g,b', '519', '3133_v,b', '3101_d,v,g,b', '3022', '690_d,v,g,b', '2872_d,v,b', '1057_d,g,b', '2032_d,v,b', '611_m,b', '323_d,v,g,b', '310_v,b', '152_m,d,v,g,b', '1275_v,b', '297_d,v,g,b', '2002_m,b', '2815_d,g,b', '3377_d,v,g,b', '136_d,v', '38_d,v', '112_m,d,v,g,b', '.DS_Store', '183_d,v,g,b', '3404_d,v,g,b', '427_d,v,b', '2715_v,b', '2884_d,v,g,b', '352_d,g,b', '1325', '2795_d,g,b', '2807_d,v,b', '257_m,b', '3034_d,v,g,b', '652_m,b', '937_d,v,b', '1015_d,g,b', '366_m,d,v,g,b', '3141_m,d,g,b', '2864_d,g,b', '543_d,v,b,g', '3038', '976', '369_m,d,g,b', '3247_d,v,g,b', '271_v', '3473_v,b', '1049_m,b', '1632', '633_v,b', '391_m,d,g,b', '3503_m,d,g,b', '3300_m,d,g,b', '2922_d,v,g,b', '21_d,v,g,b', '2319_d,g,b', '2799_d,v,g,b', '2781_d,g,b', '197_m,d,g,b', '638_m,d,b,g', '2764_d,g,b', '62_v,b', '3580_v,b', '1069_m,d,g,b', '1033_v,b', '3599_v,b', '2842_d,g,b', '346_d,v,b', '208_d,g,b', '924_v,b', '996_m,d,g,b', '882_m,b', '3027_d,g,b', '890_v,b', '69_m,d,g,b', '2372_m,d,g,b', '3001_d,v,b', '3520_v,b', '1964_v,b', '1640_m,d,g,b', '2956_v,b', '973_d,v,b', '244_d,v,g,b', '3053_m,d,g,b', '3363_m,d,g,b', '2973_m,d,g,b', '2782_d,v,g,b', '1822_d,g,b', '3408_d,g,b', '3018_m,d,g,b', '3587_v,b', '651_v,b', '703', '295_v', '3369_m,d,g,b', '2817_v,b', '2879_d,v,g,b', '2917', '3389_v,b', '1589_m,b', '299_v,b', '542_d,v,b,g', '854_m,d,g,b', '710_m,b', '3448_v,b', '3445_d,v,g,b', '572_m,d,v,b,g', '70_d,v,g,b', '600_v,b', '55_v', '488_d,v,b', '2444_m,d,g,b', '1051_d,v,g,b', '790_m,d,g,b', '90_d,v,g', '2903', '2726_v,b', '3013_v,b', '2805_v,b', '908_d,g,b', '517_v,b', '909', '3054_d,v,g,b', '515_m,d,v,b,g', '2806_d,g,b', '2334_d,g,b', '3347', '876', '1241_m,d,g,b', '156_d,v,g,b', '2147_v,b', '198_m,b', '472_m,d,b,g', '919_d,v,g,b', '620_v,b', '1081_d,v,g,b', '962_v,b', '929_d,v,g,b', '689_m,d,v,g,b', '59_m,d,g', '1068_m,d,g,b', '660_m,d,v,g,b', '2001_m,v,g,b', '1073_d,v,b', '514', '3014_d,g,b', '3394_d,v,g,b', '3340', '960_v,b', '245_v', '967_d,g,b', '478_d,v,b', '3249', '355_d,g,b', '507_m,b', '659_m,d,g,b', '594', '263_v', '2578_v,b', '176_d,v,g', '385_d,g,b', '1616_m,d,g,b', '547_v,b', '11_m,d,g,b', '1522', '18_m,b', '92_m,b', '2867_d,v,b', '970_v,b', '3008_v,b', '728', '356_m,d,g,b', '518_d,b,g', '2883_v,b', '1576_d,v,b', '2982_d,g,b', '363_d,g,b', '3071_v,b', '3450_d,v,g,b', '1022_d,g,b', '3560_v,d,b', '2964_d,g,b', '3193_d,v,b', '231_m,b', '1012_v,b', '2758_d,g,b', '570_m,b', '808_m,b', '3374_m,d,g,b', '89_d,v,b', '2970_v,b', '51_d,v,g', '2873_m,d,g,b', '931_m,d,g,b', '171_m,d,g,b', '743_m,d,g,b', '143_m,d,g,b', '658_d,v,g,b', '1264_m,b', '272_v,b', '1661_v,b', '358_m,d,v,g,b', '2229_v,b', '365_d,g,b', '2915_v,b', '308_d,g,b', '1043_m,d,g,b', '278_m,d,g,b', '2894', '727_d,g,b', '563_m,b', '549_d,g,b', '2953_d,g,b', '903_m,d,g,b', '849_m,b', '2257_v,b', '376_d,v,g,b', '516_d,v,b,g', '300_m,d,g,b', '565_d,v,b', '3430_d,g,b', '943_m,d,g,b', '655_d,v,g,b', '2920_d,g,b', '326_v,b', '407_m,b', '222_d,v,g,b', '2641_v,b', '1569_v,b', '2774_d,g,b', '3135_d,v,b', '700_v,b', '524_v,b', '2928_d,v,g,b', '10_m,b', '2006_m,d,g,b', '3415_m,d,g,b', '3130_d,v,g,b', '1050_v,b', '3348_d,g,b', '951_m,b', '2998_d,g,b', '1082_d,v,b', '888_d,v,g,b', '551_m,d,g,b', '96_m,b', '67_v,b', '708_m,b', '2866_m,d,g,b', '3386_m,d,g,b', '795_d,v,g,b', '1011_m,d,g,b', '858_m,b', '966', '37_d,g', '2969', '3367_m,d,v,g,b', '3431_d,v,b', '1097_v,b', '945_m,d,g,b', '2885_d,g,b', '450_d,v,b', '3423_v,b', '188_d,v,g,b', '1804_d,g,b', '7_m,b', '3062_v,b', '2787_d,v,g,b', '1032', '531_d,v,b,g', '1934_d,v,g,b', '2755_v,b', '2839_v,b', '415_m,d,g,b', '878_d,v,b', '2918_v,b', '3205_v,b', '534_m,b', '2900_d,g,b', '2789_d,g,b', '2955', '2_m,d,v,g,b', '2161_d,g,b', '3220_d,v,g,b', '361_v', '1987_m,d,g,b', '2776', '756_m,d,g,b', '3041_m,d,g,b', '3411_v,b', '2919_d,b,g', '445_m,d,g,b', '3434_d,g,b', '239_m,d,g,b']
    
    # valDict = txtToDict(imgDir + "labels.txt", typeID = int, typeV = str)[0]

    # select only the values which have been labelled
    # valsC = np.array(vals)[np.where([v.find("_")>0 for v in vals])[0]]

    # split the values into their id and features present
    # valsS = [v.split("_") for v in valsC]

    # vals = list(valDict.keys())

    # create a sparse matrix of the features present
    valKs = []
    for v in vals:
        try:
            vi, vk = v.split("_")
        except: 
            continue

        # vk = valDict[vi]
        if vi == ".DS":
            continue
        vi = int(vi)
        vk = vk.split(",")

        # by default set all features to being unobserved
        m = vl = b = d = 0

        # identify which features are observed for each sample
        for v in vk:
            # myometrium
            if v == "m":
                m = 1
            # villous tree
            elif v == "v":
                vl = 1
            # blood vessel
            elif v == "b":
                b = 1
            # decidua
            elif v == "d":
                d = 1
        
        # create np array for features
        valKs.append(np.array([vi, b, d, m, vl]))

    # create the data frame
    valK = pd.DataFrame(valKs, columns=["spec", "ve", "d", "m", "vi"]).sort_values(by = ["spec"]).set_index("spec")

    valK.to_csv(infoDir + "H653AClasses.txt")

    return(valK)

def dataOrganiser(imgDirs, destDir, l, ratios = [0.8, 0.1, 0.1]):

    '''
    This organises the raw data into the apporpriate directories for training

        Inputs:\n
    imgDirs, directory containing the sectioned images\n
    destDir, directory to save the newly arranged data
    l, pandas df of the labels
    ratios, training validation testing ratios

        Outputs:\n
    , information seperated into training, validation and testing for each class
    '''

    # create destination dirs
    testDir = destDir + "test/"
    trainDir = destDir + "train/"
    valDir = destDir + "val/"

    trainR, valR, testR = ratios
    if np.sum(ratios) != 1:
        print("!!! Ensure ratios sum to 1 !!!")
        return

    # ensure each directory is empty 
    label = ["decidua", "myometrium", "villous"]
    for lb in label:
        dirMaker(testDir + lb + "/", True)
        dirMaker(trainDir + lb + "/", True)
        dirMaker(valDir + lb + "/", True)

    for n, lb in enumerate(label):
        imgStore = []
        # get the spec ID for images which are only a single class (excluding vessesl)
        classID = l[(l["d"] == (n==0)*1) & (l["m"] == (n==1)*1) & (l["vi"] == (n==2)*1)]
        classSpec = np.array(classID.index)

        # get the image path for all these images
        for c in classSpec:
            c = str(c)
            imgAll = glob(imgDirs + str(c) + "/H*")
            for i in imgAll:
                imgStore.append(i)

        # randomally move images to either the train, val or test directories
        for n, i in enumerate(imgStore):
            r = random()
            if r <= trainR:
                # move images to traindir
                copyfile(i, trainDir + lb + "/" + str(n) + ".png")
            elif r <= trainR + valR:
                # move images to valdir
                copyfile(i, valDir + lb + "/" + str(n) + ".png")
            else:
                # move images ot testdir
                copyfile(i, testDir + lb + "/" + str(n) + ".png")

def modelTrainer(src, modelName, gpu = 0, layerNo = 0, epochs = 100, batch_size = 64, name = ""):
    
    # this function takes the compiled model and trains it on the imagenet data
    # Inputs:   (src), the data organised as test, train, val directories with each 
    #                   class organised as a directory
    #           (modelName), the specific model being used
    #           (gpu), the gpu being used. If no GPU is being used (ie on a CPU) then 
    #                   this variable is not important
    #           (epochs), number of training rounds for the model
    #           (batch_size), number of images processed for each round of parameter updates
    # Outputs:  (), atm no output but will eventually save the model to be used for 
    #                   evaluation

    # some basic info
    print("start, MODEL = " + modelName + " training on GPU " + str(gpu) + ", PID = " + str(os.getpid()))

    # set the GPU to be used
    CUDA_VISIBLE_DEVICES = gpu

    # info paths
    testDir = src + "test/"
    trainDir = src + "train/"
    valDir = src + "val/"

    # the path of the full imagenet catalogue info
    classPath = src + "words.txt"

    # create a dictionary which will contain the classes and their codes being used
    classes = os.listdir(trainDir)

    # get the image size (assumes all images are the same size)
    trainImages = glob(valDir + "*/*")
    IMAGE_SIZE = list(cv2.imread(trainImages[0]).shape)

    # create the model topology, make the final convolutional layers retrainable
    model, preproFunc = makeModel(modelName, IMAGE_SIZE, len(classes), layerNo)

    # create the data augmentation 
    # NOTE I don't think this is augmenting the data beyond the number of samples
    # that are present.... 
    gen_Aug = ImageDataGenerator(                    
        preprocessing_function=preproFunc,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        rotation_range = 360, 
        zoom_range = [0.5, 2], 
        horizontal_flip=True,
        vertical_flip = True
        )

    # create the data generator WITH augmentation
    train_generator = gen_Aug.flow_from_directory(
        trainDir,
        target_size=IMAGE_SIZE[:2],     # only use the first two dimensions of the images
        batch_size=batch_size,
        class_mode='binary',
        save_to_dir = None
    )

    # create the validating data generator (NO augmentation)
    gen_noAug = ImageDataGenerator(preprocessing_function=preproFunc)

    valid_generator = gen_noAug.flow_from_directory(
        valDir,
        target_size=IMAGE_SIZE[:2],
        batch_size=batch_size,
        class_mode='binary',
    )

    # create checkpoints, save every epoch
    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq=int(len(trainImages) / batch_size))

    # train the model 
    r = model.fit(x = train_generator, 
        validation_data=valid_generator, 
        epochs=epochs
        )

    # save the entire model
    model.save(name + 'saved_model')

    print("done\n\n")

def makeModel(modelType, IMAGE_SIZE, noClasses, layerNo):

    # create the model topology
    # Inputs:   (modelType), text variable which leads to a different model which has been made below
    #           (IMAGE_SIZE), size of the images which are inputted
    #           (noClasses), number of classes (ie number of neurons to use)
    # Outputs:  (model), compiled model as constructd per modelType choice

    if modelType == "VGG19":

        model, preproFunc = VGG19Maker(IMAGE_SIZE, noClasses, layerNo)

    elif modelType == "VGG16":

        model, preproFunc = VGG16Maker(IMAGE_SIZE, noClasses, layerNo)

    elif modelType == "ResNet50":

        model, preproFunc = ResNet50Maker(IMAGE_SIZE, noClasses, layerNo)

    elif modelType == "ResNet101":

        model, preproFunc = ResNet101Maker(IMAGE_SIZE, noClasses, layerNo)

    elif modelType == "EfficientNetB7":

        model, preproFunc = EfficientNetB7Maker(IMAGE_SIZE, noClasses, layerNo)
    
    elif modelType == "unet":
        
        model, preproFunc = UNETMaker(IMAGE_SIZE, noClasses)

    else:

        raise("No valid model selected")

    # print the model toplogy 
    # model.summary()

    return(model, preproFunc)

def fineTuning(ptm, layerNo, structure = 'block'):

    '''
    Specify the blocks to be trainable
        Inputs:\n
    (ptm), model
    (layerNo), the number of layers to modify. Starts with the blocks the closest to the output.
        if the input is False then nothing will be trained. If True then the WHOLE model will 
        be trainiable. If an integer number is specified then than number structures will be trained
    (structure), what structure is being specified, defaults block

        Outputs:\n
    (ptm), same network but with the trainiable parameter modified as necessary
    '''

    # if the number of layers is 0 then set the whole thing to being untrainable
    if layerNo == 0:
        ptm.trainable = False
        return ptm

    ptm.trainable = layerNo
    # if a specificed number of layers selected then state their trainability
    if type(layerNo)==int:
        blocks = sorted(list(np.unique(np.array([p.name.split("_")[0] for p in ptm.layers]))))
        blocksCopy = blocks.copy()
        for b in blocksCopy:
            if b.find(structure)==-1:
                blocks.remove(b)
                
        # select the highest block numbers first
        tarB = blocks[-layerNo:]
        for p in ptm.layers:
            # if the selected layers aren't matched then it is not being trained
            for t in tarB:
                if p.name.find(t)!=-1:
                    print(p.name + " trainable")
                    p.trainable = True

    return(ptm)

def VGG16Maker(IMAGE_SIZE, noClasses, layerNo = 0, Weights = 'imagenet', Top = False):

    # create a model with the VGG19 topology
    # Inputs:   (IMAGE_SIZE), size of the inputs
    #           (noClasses), number of classes the network will identify
    #           (Trainable), boolean whether the CNN component will be trainiable, defaults False
    #           (Weights), the pre-loaded weights that can be used, defaults to imagenet
    #           (Top), the dense layer which comes with the model, defaults to not being included
    # Outputs:  (model), compiled model

    # load the VGG16 and necessary layers from keras 
    from tensorflow.keras.applications import VGG16 as PretrainedModel
    from tensorflow.keras.applications.vgg16 import preprocess_input

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)     

    # ---- Fine tuning ---
    # get the name of all the blocks
    ptm = fineTuning(ptm, layerNo)
                            
    # create the dense layer. This is always trainable
    x = denseLayer(ptm.output, noClasses)
            
    # combine the convolutional imagenet pretrained model with the denselayer
    model = Model(inputs=ptm.input, outputs=x)  
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)

def VGG19Maker(IMAGE_SIZE, noClasses, layerNo = 0, Weights = 'imagenet', Top = False):

    # create a model with the VGG19 topology
    # Inputs:   (IMAGE_SIZE), size of the inputs
    #           (noClasses), number of classes the network will identify
    #           (Trainable), boolean whether the CNN component will be trainiable, defaults False
    #           (Weights), the pre-loaded weights that can be used, defaults to imagenet
    #           (Top), the dense layer which comes with the model, defaults to not being included
    # Outputs:  (model), compiled model

    # load the VGG19 and necessary layers from keras 
    from tensorflow.keras.applications import VGG19 as PretrainedModel
    from tensorflow.keras.applications.vgg19 import preprocess_input
    from tensorflow.keras.layers import Flatten, Dropout, Dense
    from tensorflow.keras.models import Model

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)         
            
    # boolean to fine-tune the CNN layers
    ptm = fineTuning(ptm, layerNo)

    # create the dense layer
    x = denseLayer(ptm.output, noClasses)
            
    model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)

def ResNet50Maker(IMAGE_SIZE, noClasses, layerNo = 0, Weights = 'imagenet', Top = False):

    # create a model with the VGG19 topology
    # Inputs:   (IMAGE_SIZE), size of the inputs
    #           (noClasses), number of classes the network will identify
    #           (Trainable), boolean whether the CNN component will be trainiable, defaults False
    #           (Weights), the pre-loaded weights that can be used, defaults to imagenet
    #           (Top), the dense layer which comes with the model, defaults to not being included
    # Outputs:  (model), compiled model

    # load the ResNet and necessary layers from keras 
    from tensorflow.keras.applications import ResNet50 as PretrainedModel
    from tensorflow.keras.applications.resnet50 import preprocess_input
    from tensorflow.keras.layers import Flatten, Dropout, Dense
    from tensorflow.keras.models import Model

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)         
            
    # boolean to fine-tune the CNN layers
    ptm = fineTuning(ptm, layerNo)

    # create the dense layer
    x = denseLayer(ptm.output, noClasses)
            
    model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)

def ResNet101Maker(IMAGE_SIZE, noClasses, layerNo = 0, Weights = 'imagenet', Top = False):

    # create a model with the ResNet50 topology
    # Inputs:   (IMAGE_SIZE), size of the inputs
    #           (noClasses), number of classes the network will identify
    #           (Trainable), boolean whether the CNN component will be trainiable, defaults False
    #           (Weights), the pre-loaded weights that can be used, defaults to imagenet
    #           (Top), the dense layer which comes with the model, defaults to not being included
    # Outputs:  (model), compiled model

    # load the ResNet and necessary layers from keras 
    from tensorflow.keras.applications import ResNet101 as PretrainedModel
    from tensorflow.keras.applications.resnet import preprocess_input

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)         
            
    # boolean whether to fine-tune the CNN layers
    ptm = fineTuning(ptm, layerNo)

    # create the dense layer
    x = denseLayer(ptm.output, noClasses)

    model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)

def EfficientNetB7Maker(IMAGE_SIZE, noClasses, layerNo = 0, Weights = 'imagenet', Top = False):

    # create a model with the ResNet50 topology
    # Inputs:   (IMAGE_SIZE), size of the inputs
    #           (noClasses), number of classes the network will identify
    #           (Trainable), boolean whether the CNN component will be trainiable, defaults False
    #           (Weights), the pre-loaded weights that can be used, defaults to imagenet
    #           (Top), the dense layer which comes with the model, defaults to not being included
    # Outputs:  (model), compiled model

    # load the ResNet and necessary layers from keras 
    from tensorflow.keras.applications import EfficientNetB7 as PretrainedModel
    from tensorflow.keras.applications.efficientnet import preprocess_input

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)         
            
    # boolean whether to fine-tune the CNN layers
    ptm = fineTuning(ptm, layerNo)

    # create the dense layer
    x = denseLayer(ptm.output, noClasses)

    model = Model(inputs=ptm.input, outputs=x)  # substitute the 4D CNN output for a 1D shape for the dense network input
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)

def UNETMaker(IMAGE_SIZE, noClasses, Trainable = False, Weights = 'imagenet', Top = False):

    from tensorflow.keras.applications import MobileNetV2 as PretrainedModel
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    
    def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
        """Upsamples an input.
        Conv2DTranspose => Batchnorm => Dropout => Relu
        Args:
            filters: number of filters
            size: filter size
            norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
            apply_dropout: If True, adds the dropout layer
        Returns:
            Upsample Sequential Model
        """

        class InstanceNormalization(tf.keras.layers.Layer):
            """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

            def __init__(self, epsilon=1e-5):
                super(InstanceNormalization, self).__init__()
                self.epsilon = epsilon

            def build(self, input_shape):
                self.scale = self.add_weight(
                    name='scale',
                    shape=input_shape[-1:],
                    initializer=tf.random_normal_initializer(1., 0.02),
                    trainable=True)

                self.offset = self.add_weight(
                    name='offset',
                    shape=input_shape[-1:],
                    initializer='zeros',
                    trainable=True)

            def call(self, x):
                mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
                inv = tf.math.rsqrt(variance + self.epsilon)
                normalized = (x - mean) * inv
                return self.scale * normalized + self.offset

        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    # creat model
    def unet_model(output_channels):

        # load the pretrained model and specify the weights being used
        ptm = PretrainedModel(
            input_shape=IMAGE_SIZE,  
            weights=Weights,         
            include_top=Top)     

        # --- create encoder ---
        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]

        layers = [ptm.get_layer(name).output for name in layer_names]

        # Create the feature extraction model
        down_stack = Model(inputs=ptm.input, outputs=layers)
        down_stack.trainable = False  

        # --- create decoder ---
        # upscaling function
        up_stack = [
        upsample(576, 3),  # 4x4 -> 8x8
        upsample(272, 3),  # 8x8 -> 16x16
        upsample(136, 3),  # 16x16 -> 32x32
        upsample(68, 3),   # 32x32 -> 64x64
        ]

        # Downsampling through the model
        x = ptm.input
        skips = down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])

        # This is the last layer of the model
        last = tf.keras.layers.Conv2DTranspose(
            output_channels, 3, strides=2,
            padding='same')  #64x64 -> 128x128

        x = last(x)

        return Model(inputs=inputs, outputs=x)

    model = unet_model(noClasses)
    
    # bolt the whole thing together, aka compile it
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # set the pre-processing function as the inbuilt vgg19 function
    preprocessingFunc = preprocess_input

    return(model, preprocessingFunc)

def denseLayer(ptmoutput, noClasses):

    # creates the dense layer
    # Inputs:   (ptmoutput), takes the CNN output layer
    #           (noClasses), number of classifiers to train
    # Outputs:  (x), constructed dense layer

    # create the dense layer
    x = Flatten()(ptmoutput)   
    x = Dropout(0.2)(x)        
    x = Dense(noClasses, activation='softmax')(x)

    return(x)

def dataOrganiser(imgDirs, destDir, l, ratios = [0.8, 0.1, 0.1]):

    '''
    This organises the raw data into the apporpriate directories for training

        Inputs:\n
    imgDirs, directory containing the sectioned images\n
    destDir, directory to save the newly arranged data
    l, pandas df of the labels
    ratios, training validation testing ratios

        Outputs:\n
    , information seperated into training, validation and testing for each class
    '''

    # create destination dirs
    testDir = destDir + "test/"
    trainDir = destDir + "train/"
    valDir = destDir + "val/"

    trainR, valR, testR = ratios
    if np.sum(ratios) != 1:
        print("!!! Ensure ratios sum to 1 !!!")
        return

    # ensure each directory is empty 
    label = ["decidua", "myometrium", "villous"]
    for lb in label:
        dirMaker(testDir + lb + "/", True)
        dirMaker(trainDir + lb + "/", True)
        dirMaker(valDir + lb + "/", True)

    for n, lb in enumerate(label):
        imgStore = []
        # get the spec ID for images which are only a single class (excluding vessesl)
        classID = l[(l["d"] == (n==0)*1) & (l["m"] == (n==1)*1) & (l["vi"] == (n==2)*1)]
        classSpec = np.array(classID.index)

        # get the image path for all these images
        for c in classSpec:
            c = str(c)
            imgAll = glob(imgDirs + "/" + str(c) + "/H*")
            for i in imgAll:
                imgStore.append(i)

        # randomally move images to either the train, val or test directories
        for n, i in enumerate(imgStore):
            r = random()
            if r <= trainR:
                # move images to traindir
                copyfile(i, trainDir + lb + "/" + str(n) + ".png")
            elif r <= trainR + valR:
                # move images to valdir
                copyfile(i, valDir + lb + "/" + str(n) + ".png")
            else:
                # move images ot testdir
                copyfile(i, testDir + lb + "/" + str(n) + ".png")

def segment(imgPath, model, sectL, sc):

    '''
    Perform segmentation with the model generated

        Inputs:\n
    imgPath, image path to segment
    modelPath, model directory to use for segmentation
    sectL, length of the section tile
    sc, window size

        Outputs:\n
    (), creates a segmented mask

    '''

    # load the image and get info
    name = nameFromPath(imgPath)
    img = cv2.imread(imgPath)
    h, w, c = img.shape
    hrange = int(np.floor(h/sc))
    wrange = int(np.floor(w/sc))

    # create a segmentation mask
    predict = np.zeros([hrange, wrange])
    for x in range(wrange):
        print(name + " column: " + str(x))
        for y in range(hrange):
            # printProgressBar(y + x * hrange, hrange * wrange, name + " processed", length=20)
            sect = img[y*sc:y*sc+sectL, x*sc:x*sc+sectL]
            
            if np.count_nonzero(sect) < sect.size * 0.5:
                # if less than 50% of the sect is information then 
                # assign to 0
                predict[y, x] = 0

            else:
                # if there is sufficient information to process
                # then categorise. NOTE that background is assigned 0
                # and the labels are l + 1
                predict[y, x] = np.argmax(model.predict(np.expand_dims(sect, 0), use_multiprocessing = False)) + 1

    cv2.imwrite(name + ".png", predict * 255/np.max(predict))
    print("     " + name + " DONE")

if __name__ == "__main__":


    # directory which contains the sections
    imgDirs = '/Volumes/USB/H653A_11.3/3/FeatureSections/NLAlignedSamplesSmallpng_False2/'
    destDir = '/Volumes/USB/H653A_11.3/3/FeatureSections/'
    infoDir = ""
    modelPath = "saved_model_3"

    '''
    imgDirs = '/eresearch/uterine/jres129/BoydCollection/SpecimenSections/H653A_11.3/linearSect/'
    destDir = '/people/jres129/Masters/Segmentation/'
    infoDir = '/people/jres129/Masters/Segmentation/'
    '''

    # labelData(imgDirs)
    
    # from all of the images get their corresponding labels
    # labels = getLabels(imgDirs, infoDir)

    # oragnise the data for TF usage
    # dataOrganiser(imgDirs, destDir, labels)

    # train the model
    for m in ["ResNet101"]: # "VGG16", "VGG19", 
        print("---- " + m + " ----")
        modelTrainer(destDir, m, epochs=50, name = m, layerNo = 1)

    # get the labels
    vals = os.listdir(destDir + 'test/')
    
    # from the testing data enable evaluation
    # NOTE labels dont' seem to match???
    evalStore = []
    evalID = []
    for v, va in enumerate(vals):
        testImgs = glob(destDir + 'test/' + va + '/*')[:20]
        for t in testImgs:
            evalStore.append(cv2.imread(t))
            evalID.append(v)

    evalStore = np.array(evalStore)
    evalID = np.array(evalID)

    imgssrc = sorted(glob('sampleImgs/*.png'))
    imgssrc = sorted(glob('/Volumes/Storage/H653A_11.3/3/NLAlignedSamplesFinal/*'))
    # segment the image
    # NOTE this can be massively parallelised (on HPC)

    cpuNo = 4
    sectL = 136
    sc = 100

    model = tf.keras.models.load_model(modelPath)


    for i in imgssrc:
        segment(i, model, sectL, sc)

    '''
    with Pool(processes=cpuNo) as pool:
            pool.starmap(segment, zip(imgssrc, repeat(modelPath), repeat(sectL), repeat(sc)))
    '''

'''
    @tf.function
    def load_image_train(datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask

    def load_image_test(datapoint):
        input_image = tf.image.resize(datapoint['image'], (128, 128))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

        input_image, input_mask = normalize(input_image, input_mask)

        return input_image, input_mask
'''