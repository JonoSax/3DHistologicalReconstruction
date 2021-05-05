'''
This script contains all the plotting funcitons for the diagrams used in 
my thesis
'''

import cv2
import numpy as np
from glob import glob
from random import uniform
from time import time
import tifffile as tifi
import pandas as pd
import plotly.express as px
from itertools import repeat

if __name__ != "HelperFunctions.plottingFunctions":
    from Utilities import *
    from SP_AlignSamples import align
    from SP_FeatureFinder import featFind
else:
    from HelperFunctions.Utilities import *
    from HelperFunctions.SP_AlignSamples import align
    from HelperFunctions.SP_FeatureFinder import featFind

def plottingFeaturesPerRes(IMGREF, name, matchedInfo, scales, circlesz = 1):

    '''
    this plotting funciton gets the features that have been produced per resolution 
    and combines them into a single diagram

        Inputs:\n
    IMGREF, image to plot the features on\n
    name, name of the sample\n
    matchedInfo, list of all the feature objects\n
    sclaes, the image scales used in the search\n
    circlesz, circle sz (put to 0 to remove circles)\n

    '''

    imgRefS = []
    sclInfoAll = []
    for n, scl in enumerate(scales):

        # get the position 
        sclInfo = [matchedInfo[i] for i in list(np.where([m.res == n for m in matchedInfo])[0])]

        if len(sclInfo) == 0: 
            continue
        else:
            sclInfoAll += sclInfo

        # for each resolution plot the points found
        imgRefM = cv2.resize(IMGREF.copy(), (int(IMGREF.shape[1] * scl), int(IMGREF.shape[0] * scl)))
        # downscale then upscale just so that the image looks like the downsample version but can 
        # be stacked
        imgRefM = cv2.resize(imgRefM, (int(IMGREF.shape[1]), int(IMGREF.shape[0])))
        imgRefM, _ = nameFeatures(imgRefM.copy(), imgRefM.copy(), sclInfoAll, circlesz=circlesz, combine = False, txtsz=0)
        
        imgRefM = cv2.putText(imgRefM, "Scale = " + str(scl), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, [255, 255, 255], thickness = 14)
        imgRefM = cv2.putText(imgRefM, "Scale = " + str(scl), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 0],  thickness = 6)
        
        cv2.imwrite("/Users/jonathanreshef/Downloads/" + name + "_" + str(scl) + ".png", imgRefM)
        
        imgRefS.append(imgRefM)

    imgRefS = np.hstack(imgRefS)
    cv2.imwrite("/Users/jonathanreshef/Downloads/" + name + ".png", imgRefS)
    # plt.imshow(imgRefS); plt.show()

def colourDistributionHistos(imgtarSmallorig, imgref, imgtarmod):

    '''
    Compares the distributions of the colour channels before and after
    colour normalisations compared to the reference image
    Run this at the end of the specimenID.imgNormColour
    '''

    colours = ['b', 'g', 'r']
    z = [np.arange(10), np.zeros(10)]
    ylim = 0.2

    # ----- Plot the origianl vs reference histogram plots -----

    blkDsh, = plt.plot(z[0], z[1], "k:")    # targetO
    blkDot, = plt.plot(z[0], z[1], "k--")    # targetMod
    blkLn, = plt.plot(z[0], z[1], "k")    # reference
    
    for c, co in enumerate(colours):
        o = np.histogram(imgtarSmallorig[:, :, c], 32, (0, 256))   # original
        r = np.histogram(imgref[:, :, c], 32, (0, 256)) # reference
        v = np.histogram(imgtarmod[:, :, c], 32, (0, 256))   # modified
        v = np.ma.masked_where(v == 0, v)

        maxV = np.sum(r[0][1:])        # get the sum of all the points
        plt.plot(o[1][2:], o[0][1:]/maxV, co + ":", linewidth = 2)
        plt.plot(v[1][2:], v[0][1:]/maxV, co + "--", linewidth = 2)
        plt.plot(r[1][2:], r[0][1:]/maxV, co, linewidth = 1)

    plt.legend([blkDsh, blkDot, blkLn], ["TargetOrig", "TargetMod", "Reference"])
    plt.ylim([0, ylim])
    plt.xlabel("Pixel value", fontsize = 14)
    plt.ylabel("Pixel distribution",  fontsize = 14)
    plt.title("Histogram of colour profiles",  fontsize = 14)
    plt.show()

    # ----- Plot the modified vs reference histogram plots -----
    
    for c, co in enumerate(colours):
        v = np.histogram(imgtarmod[:, :, c], 32, (0, 256))[0][1:]   # modified
        v = np.ma.masked_where(v == 0, v)
        plt.plot(v/maxV, co + "--", linewidth = 2)
        r = np.histogram(imgref[:, :, c], 32, (0, 256))[0][1:] # reference
        plt.plot(r/maxV, co, linewidth = 1)

    z = np.arange(10)

    plt.legend([blkDsh, blkLn], ["TargetMod", "Reference"])
    plt.ylim([0, ylim])
    plt.xlabel("Pixel value", fontsize = 14)
    plt.ylabel("Pixel distribution",  fontsize = 14)
    plt.title("Histogram of colour profiles\n modified vs reference",  fontsize = 18)
    plt.show()

def imageModGenerator():

    '''
    This function generates images with a random translation and rotation
    from an original image. Used to feed into the featurefind and align process
    to validate they work.
    '''

    def imgTransform(img, maxSize, maxRot = 90):
    
        '''
        From an original image make a bunch of new images with random 
        rotation and translation within a maximum sized area
        '''
        
        ip = img.shape[:2]
        angle = uniform(-maxRot, maxRot)/2

        xM, yM = (maxSize - img.shape[:2])/2
        translate = np.array([uniform(-xM, xM), uniform(-yM, yM)])
        
        rot = cv2.getRotationMatrix2D(tuple(np.array(ip)/2), float(angle), 1)
        warpedImg = cv2.warpAffine(img, rot, (ip[1], ip[0]))

        # get the position of the image at the middle of the image
        p = np.array((maxSize - ip)/2 + translate).astype(int)

        plate = np.zeros(np.insert(maxSize, 2, 3)).astype(np.uint8)
        plate[p[0]:p[0]+ ip[0], p[1]:p[1]+ ip[1]] = warpedImg

        return(plate) 

    dataHome = "/Users/jonathanreshef/Documents/2020/Masters/Thesis/Diagrams/alignerDemonstration/"

    imgSrc = dataHome + "3/BirdFace.png"
    img = cv2.imread(imgSrc)

    dirMaker(dataHome + "3/maskedSamples/")

    # get the reference image in the centre of the maxSize
    x, y = img.shape[:2]
    xM, yM = (np.array([x, y]) * 1.2).astype(int)
    xMi = int((xM-x)/2); yMi = int((yM-y)/2)
    imgN = np.zeros([xM, yM, 3]).astype(np.uint8)
    imgN[xMi:xMi+x, yMi:yMi+y, :] = img

    # create the centred image to align all samples to
    # cv2.imwrite(dataHome + "3/maskedSamples/modImg000.png", imgN)

    for i in range(20):
        imgN = imgTransform(img, np.array([xM, yM]), maxRot=360)
        name = str(i + 1)
        while len(name) < 3:
            name = "0" + name
        # cv2.imwrite(dataHome + "3/maskedSamples/modImg" + name + ".png", imgN)

    featFind(dataHome, "3", 1, 10, 1, 1)
    align(dataHome, "3", 1)

def triangulatorPlot(img, matchedInfo):

    '''
    Plots the features of samples and visualises the triangulation 
    calculation
    '''
                
    def triangulator(img, featurePoints, anchorPoints, feats = 5, crcsz = 5):

        '''
        Plot the triangulation of the features in the sample 
        from the anchor feautres

            Inputs:\n
        (img), image\n
        (featurePoints), list of ALL feature points\n
        (anchorPoints), number of points in featurePoints which are used for 
        the anchor\n
        (feats), maximum number of non-anchor points to plot

            Output:\n
        (imgAngles), image with the feature and its points annotated
        '''

        # get the anchor points and the points found from these
        anchors = featurePoints[:int(anchorPoints)]
        points = featurePoints[int(anchorPoints):]

        imgAngles = img.copy()
        col = [[255, 0, 0], [0, 255, 255], [255, 0, 255], [0, 255, 0], [255, 255, 0]]

        for n, p in enumerate(points[:feats]):
            # draw the point of interest
            cv2.circle(imgAngles, tuple(p.astype(int)), crcsz, [0, 0, 255], crcsz*2)
            for n1, a1 in enumerate(anchors):

                # draw the anchor points
                cv2.circle(imgAngles, tuple(a1.astype(int)), crcsz, [255, 0, 0], crcsz*2)
                for n2, a2 in enumerate(anchors):
                    if n1 == n2: 
                        continue
                    # draw the lines between the features
                    imgAngles = drawLine(imgAngles, p, a1, colour=col[n])
                    imgAngles = drawLine(imgAngles, p, a2, colour=col[n])
        
        return(imgAngles)
                    
    # get the reference and target points 
    refPts = [m.refP for m in matchedInfo]

    # annotate the image with the feature info
    for i in range(5):
        refAngles = triangulator(img, refPts, 5, i, crcsz = 12)

        cv2.imshow('angledImages', refAngles); cv2.waitKey(0)
    cv2.destroyWindow('angledImages')

def siftTimer():

    '''
    Timing the SIFT operator for various octave numbers and images
    resolutions 
    '''

    imgPath = '/Volumes/USB/H653A_11.3/3/maskedSamples/H653A_002_0.png'
    imgOrig = cv2.imread(imgPath)

    # perform sift search on multiple different resolutions of sift
    for n in range(1, 10):

        img = cv2.resize(imgOrig.copy(), (int(imgOrig.shape[1]/n), int(imgOrig.shape[0]/n)))

        for i in range(10, 11):
            sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = i) 

            a = time()
            for _ in range(10):
                s = sift.detectAndCompute(img, None)[0]
            fin = time()-a

            print("ImgSz = " + str(np.round(1/n, 2)) + " Octaves = " + str(i) + " Time: " + str(fin) + " feats = " + str(len(s)))

def vesselPositionsOnMaskedImgs():

    '''
    This is getting the masked positions as manually annotated from 
    the samples after they have been seperated by specimen ID. 
    Saved as pixel positions on the full resolution image
    '''

    src = '/Volumes/USB/H653A_11.3/2.5/'
    imgsrc = src + 'NLAlignedSamplesSmall/'
    destImgs = imgsrc + "targetVessels/"
    dirMaker(destImgs)

    downsampleImgs = sorted(glob(imgsrc + "*.png"))
    fullImgs = sorted(glob(imgsrc + "*.tif"))

    for n, d in enumerate(downsampleImgs):

        name = nameFromPath(d, 3)

        print("Processing " + name)

        # penalise non-green colours
        img = np.mean(cv2.imread(d) * np.array([-1, 1, -1]), axis = 2)

        maskPos = np.where(img > 50); maskPos = np.c_[maskPos[1], maskPos[0]]

        # ensure the dense reconstructions are the original size
        maskPos = np.insert(maskPos, 0, np.array([0, 0]), axis = 0)
        maskPos = np.insert(maskPos, 0, np.flip(img.shape[:2]), axis = 0)

        vessels = denseMatrixViewer([maskPos], plot = False, point = True)[0]
        cv2.imwrite(destImgs + name + "_vessels.png", vessels)
        
        listToTxt([maskPos], destImgs + name + "vessels.txt")

def bruteForceMatcherTimes():

    bf = cv2.BFMatcher_create()   
    sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers = 3)    

    # just get two random images 
    img1 = cv2.imread('/Volumes/USB/H653A_11.3/3/maskedSamples/H653A_002_0.png')
    img2 = cv2.imread('/Volumes/USB/H653A_11.3/3/maskedSamples/H653A_003_0.png')

    # get sift info
    _, des_ref = sift.detectAndCompute(img1,None)
    _, des_tar = sift.detectAndCompute(img2,None)

    des_ref_accum = []
    des_tar_accum = []

    timeStore = []
    tRange = np.arange(1, 20, 2)
    for i in tRange:
        print("Processing " + str(i))

        # make the lists i long
        des_ref_accum = []
        des_tar_accum = []
        for n in range(i):
            des_ref_accum.append(des_ref)
            des_tar_accum.append(des_tar)
        start = time()
        matches = bf.match(np.hstack(des_ref_accum), np.hstack(des_tar_accum))
        end = time()-start
        timeStore.append(end)

    plt.plot(tRange, timeStore); plt.show()

    print("")

def compareSmoothvsRaw():

    '''
    This function takes a raw and smoothed data frame and plots both of 
    them showing their differences

    NOTE only run this while featShaper() is running so the right variables
    are loaded
    '''

    # insert different types so that dashed and full lines can be plotted 
    dfSelectR.insert(4, "Type", 0)
    dfSelectSMFix2.insert(4, "Type", 1)

    # rename the raw feature Z axis
    dfSelectRNew = dfSelectR.rename(columns={'Zs': 'Z'})

    a = pd.concat([dfSelectRNew, dfSelectSMFix2])
    dfSelectRNew
    dfSelectSMFix2
    px.line_3d(a, x="X", y="Y", z="Z", color="ID", line_dash="Type", title = "Smooth vs Raw").show()    

def getLongFeats(n):

    '''
    This script identifies the longest n features for each specimen 
    '''

    src = "/Volumes/USB/"
    specs = glob(src + "H*")

    specAll = []

    for s in specs:

        print("\n---- " + s.split("/")[-1] + " ----")

        featsrc = s + "/4/FeatureSections/NLAlignedSamplesSmallpng_True2/"

        feats = glob(featsrc + "*")

        featIDOrder = np.argsort([-len(glob(f + "/*.png")) for f in feats])
        featIDLen = -np.sort([-len(glob(f + "/*.png")) for f in feats])

        featOrdered = []
        for f in featIDOrder:
            featOrdered.append(feats[f].split("/")[-1])

        specAll.append(np.c_[featIDLen.astype(int), featOrdered, ((s.split("/")[-1]+",")*(len(featIDLen))).split(",")[:-1]])

        for f, l in zip(featIDOrder[:n], featIDLen[:n]):
            specID = feats[f].split("/")[-1]
            print("ID: " + str(specID) + " len: " + str(l))

    specInfo = np.vstack(specAll)
    order = np.argsort(-np.array(np.vstack(specAll))[:, 0].astype(int))[:100]

    topSpecs = []
    print("Longest 100 features and their associated specimens")
    print("Length, featureID, specimen")
    for o in order:
        print(specInfo[o,:])

def getAllTrajectories():

    '''
    Get all the 3D trajectories from the feature tracking of all specimens
    '''

    src = "/Volumes/USB/"
    src = '/Volumes/USB/ANHIR/TargetTesting/'
    size = 2.5
    specs = sorted(glob(src + "C*"))

    for s in specs:

        specName = s.split("/")[-1]
        print("\n---- " + specName + " ----")

        # get the raw features
        rawfeatsrc = s + "/" + str(size) + "/FeatureSections/rawSelectedFixFeatures.csv"
        smfeatsrc = s + "/" + str(size) + "/FeatureSections/smoothSelectedFixFeatures.csv"

        try:
            rawfeats = pd.read_csv(rawfeatsrc)
            smfeats = pd.read_csv(smfeatsrc)
            
            px.line_3d(rawfeats, "X", "Y", "Z", "ID", title=specName + " raw").show()
            px.line_3d(smfeats, "X", "Y", "Z", "ID", title=specName + " smooth").show()
        except:
            print("     " + specName + " failed")

def trainingSampleProporptional():

    '''
    This is counting the number of feature for each type of tissue labelled
    '''

    vals = ['3505', '2842', '1668', '3045', '315_v,b', '3604', '554_d,g,b', '118_d,g', '743', '3_m,b', '1033', '3052', '2897', '1588', '3472', '597_d,v,b', '550_v,b', '1595', '667_v,b', '2916', '3305', '2777', '44_v,b', '588_d,v,g,b', '3411', '2883', '836', '3367', '878', '3093', '1954', '2799', '874', '3053', '1081', '2303', '3426', '1640', '3473', '225_d,g', '695_d,v,g,b', '333_v,b', '3247', '2818', '1275', '3294', '937', '862', '979', '33_d,g', '2964', '3488', '699_d,v,g,b', '929', '3448', '854', '669_m,d,g,b', '592_d,g,b', '412_d,v,b', '967', '2979', '892', '3554', '2804', '178_d,v,b', '3054', '1082', '3014', '1641', '254_d,g', '2910', '634_v,b', '214_d,v,g', '3434', '951', '3295', '641_m,d,v,g,b', '901', '3580', '1097', '2319', '3029', '2965', '276_m,d,v,g,b', '1057', '351_v,b', '943', '718', '1616', '3449', '2786', '414_m,d,g,b', '31_d,v', '3135', '3000', '2845', '104_d,g', '2805', '212_d,g,b', '3015', '264_v', '377', '1043', '129_m,b', '259_m,d,g,b', '1602', '3435', '3300', '466_d,v,b,g', '36_d,g,b', '1598', '2867', '110_d,v,g', '1938', '849', '2327', '580_d,v,b', '2973', '2006', '2787', '274_m,d,v,g,b', '3362', '3041', '34_d,v,g', '561_d,v,b', '3001', '475_d,g', '311_m,g,b', '724', '373_d,g,b', '2806', '3282', '3516', '2853', '3377', '879', '1964', '381_m,b', '3016', '3063', '347_d,g,b', '867', '2828', '3217', '863', '1804', '2927', '1066', '3363', '2934', '2795', '329_m,d,v,g,b', '76_d,v,g,b', '63_d,v,g,b', '41_d,g,b', '324_d,v,g,b', '3623', '181_v', '3484', '956', '1052', '2920', '693_d,v,b', '2781', '431_v,b', '1987', '3503', '2840', '3550', '944', '2800', '2749', '536_v,b', '940', '3145', '2161', '317_m,d,g,b', '2855', '3430', '2815', '284_m,d,g,b', '3386', '267_m,d,g,b', '1926', '169_v', '314_d,g,b', '970', '2961', '3485', '578_d,v,g,b', '2921', '3445', '567_m,b', '2001', '2782', '2877', '3266', '3591', '1988', '173_m,d,g,b', '1068', '650_d,v,b', '2936', '2801', '2983', '1_d,v,g', '3372', '3146', '206_m,d,v,g,b', '3431', '2257', '393_m,b', '328_m,d,g,b', '3292', '2998', '3205', '345_d,v,g,b', '560_d,v,g,b', '16_d,v,g', '3026', '1934', '2922', '1014', '2002', '1569', '685_m,d,g,b', '1069', '2937', '1628', '1675', '2758', '1960', '694_m,b', '98_d,v,g,b', '2444', '2817', '221_d,v,g,b', '364_m,d,v,g,b', '2864', '894', '3388', '2824', '890', '919', '3348', '3027', '1241', '915', '140_m,d,g,b', '2923', '1015', '2970', '1661', '1062', '2879', '882', '705_d,v,b', '3454', '2791', '3133', '2839', '3180', '903', '2850', '3560', '3374', '945', '57_d,v,g', '2759', '647_d,g,b', '3013', '3195', '27_d,v,g,b', '687', '3389', '395_m,b', '637', '471_m,d,b,g', '552_m,d,g,b', '2372', '3408', '2745', '3141', '738', '541_v,b', '662_d,g,b', '702_d,g,b', '2946', '53_d,g', '496_d,b,g', '35_m,b', '428_m,d,v,g,b', '116_v,b', '726', '535_d,b,g', '2866', '483_m,d,b,g', '409_v,b', '2873', '1699', '3215', '668_v,b', '60_d,g,b', '710', '2147', '756', '2888', '587_d,g,b', '1999', '1079', '790', '819', '122_v', '334_d,g,b', '3062', '519', '740', '3022', '690_d,v,g,b', '928', '1050', '899', '611_m,b', '3402', '2874', '3216', '323_d,v,g,b', '310_v,b', '152_m,d,v,g,b', '2334', '841', '297_d,v,g,b', '908', '996', '136_d,v', '38_d,v', '112_m,d,v,g,b', '2948', '183_d,v,g,b', '427_d,v,b', '2860', '2908', '3384', '2820', '352_d,g,b', '1325', '257_m,b', '1051', '1011', '652_m,b', '3490', '2229', '366_m,d,v,g,b', '2875', '3450', '934', '543_d,v,b,g', '930', '3038', '976', '369_m,d,g,b', '972', '271_v', '1073', '3370', '1632', '2755', '633_v,b', '391_m,d,g,b', '2715', '21_d,v,g,b', '3524', '960', '1088', '2956', '197_m,d,g,b', '2821', '638_m,d,b,g', '62_v,b', '3071', '1012', '346_d,v,b', '208_d,g,b', '727', '3539', '3404', '2876', '69_m,d,g,b', '3451', '3130', '2836', '2982', '1074', '3419', '244_d,v,g,b', '1589', '1827', '795', '651_v,b', '703', '295_v', '2957', '1049', '2917', '3393', '299_v,b', '858', '1707', '542_d,v,b,g', '3405', '3587', '808', '2238', '572_m,d,v,b,g', '2884', '70_d,v,g,b', '600_v,b', '55_v', '488_d,v,b', '1075', '917', '2943', '90_d,v,g', '888', '2903', '3193', '2764', '884', '2578', '3153', '517_v,b', '909', '515_m,d,v,b,g', '2958', '3347', '2870', '2038', '876', '2918', '3394', '1609', '872', '156_d,v,g,b', '3168', '198_m,b', '472_m,d,b,g', '620_v,b', '2885', '1576', '3369', '860', '689_m,d,v,g,b', '931', '59_m,d,g', '1076', '660_m,d,v,g,b', '3008', '973', '1682', '2904', '514', '3340', '3289', '245_v', '478_d,v,b', '3249', '355_d,g,b', '2586', '2265', '507_m,b', '659_m,d,g,b', '961', '594', '263_v', '2919', '176_d,v,g', '3395', '385_d,g,b', '3260', '547_v,b', '11_m,d,g,b', '1522', '18_m,b', '92_m,b', '3220', '3034', '1022', '728', '356_m,d,g,b', '518_d,b,g', '3414', '363_d,g,b', '3100', '1822', '3520', '3056', '231_m,b', '3381', '2032', '570_m,b', '2726', '2587', '708', '89_d,v,b', '2872', '51_d,v,g', '2832', '3221', '171_m,d,g,b', '2788', '143_m,d,g,b', '658_d,v,g,b', '3042', '272_v,b', '859', '358_m,d,v,g,b', '3415', '365_d,g,b', '308_d,g,b', '278_m,d,g,b', '2894', '3101', '2807', '1264', '3283', '2212', '2993', '563_m,b', '549_d,g,b', '2953', '889', '376_d,v,g,b', '516_d,v,b,g', '300_m,d,g,b', '2774', '885', '565_d,v,b', '655_d,v,g,b', '326_v,b', '1524', '407_m,b', '222_d,v,g,b', '2928', '2789', '700_v,b', '524_v,b', '10_m,b', '3423', '2669', '3018', '2954', '1645', '1046', '2914', '551_m,d,g,b', '96_m,b', '67_v,b', '3299', '924', '966', '37_d,g', '2969', '2834', '962', '2929', '450_d,v,b', '912', '188_d,v,g,b', '7_m,b', '3599', '1032', '531_d,v,b,g', '2900', '950', '415_m,d,g,b', '2809', '1547', '534_m,b', '900', '2955', '2_m,d,v,g,b', '2915', '361_v', '2962', '2776', '2641', '3486', '445_m,d,g,b', '2236', '239_m,d,g,b']

    valLabel = []
    for v in vals:
        if len(v.split("_"))>1:
            valLabel.append(v)

    src = '/Volumes/USB/H653A_11.3/3/FeatureSections/linearSect2/'

    classesLen = {}
    classesNum = {}
    for v in valLabel:
        id, cl = v.split("_")
        cl = "".join(sorted(cl.split(","))).replace("b", "").replace("g", "")    # remove blood vessesl from classes
        if classesLen.get(cl, None) is None:
            classesLen[cl] = []
            classesNum[cl] = 0
        feat = src + id + "/"
        classesLen[cl].append(len(glob(feat + "*.png")))
        classesNum[cl] += 1

    for c in classesLen:
        ratio = np.median(classesLen[c])
        print("Average len of " + c + " " + str(np.round(ratio, 2)))

    print(classes)

def smallTilesOfSections():

    '''
    Get an image and divide it into tiles. Highlight the part of the time 
    of interest
    '''

    img = cv2.imread("sect.png").astype(float)

    x, y, c = img.shape

    xs = x - 40
    ys = y - 40

    s = 20

    xr = x-xs + s
    yr = y-ys + s
    img *= 0.5
    imgAll = []
    for xn in np.arange(0, xr, s, int):
        imgHor = []
        for yn in np.arange(0, yr, s, int):
            plate = (np.ones([x + 10, y + 10, c])*255).astype(np.uint8)

            # highlight the selected area
            imgShow = img.copy()
            imgShow[xn:xn+xs, yn:yn+ys, :] *= 2
            imgShow = imgShow.astype(np.uint8)
            plate[:x, :y, :] = imgShow
            imgHor.append(plate)

        imgAll.append(np.hstack(imgHor))

    imgFinal = np.vstack(imgAll)
    plt.imshow(imgFinal); plt.show()
    cv2.imwrite("sectSections.png", imgFinal)

def getSpecimenDimensions():

    '''
    Gets the size of the x, y and z of the specimen based on slide thickness
    and the assumption of a 18µm pixel size (for a 2.5 full scale, 0.2x baseline image)
    '''

    src = "/Volumes/USB/"
    specs = sorted(glob(src + "H*"))
    path = "/4/NLAlignedSamplesSmall/"

    depths = {
    "H710B": 5,
    "H710C": 5,
    "H750A": 6,
    "H673A": 10,
    "H1029A": 10,
    "H653A": 15,
    "H671A": 10,
    "H671B": 10
    }

    for s in specs:
        name = s.split("/")[-1].split("_")[0]
        imgs = sorted(glob(s + path + "*.png"))
        imgpath = imgs[0]
        img = cv2.imread(imgpath)
        x, y, _ = img.shape
        xdim = np.round(x * 18 / 1000, 2)        # dims in mm
        ydim = np.round(y * 18 / 1000, 2)
        d = np.round(depths[name] * len(imgs) / 1000, 2)

        # print(name + ": " + str(xdim) + "x" + str(ydim) + "x" + str(d))

        print(name + " has a slide thickness of " + str(depths[name]) + " µm with " + str(len(imgs)) + " samples so the dimensions of the final reconstruction are " + str(xdim) + "x" + str(ydim) + "x" + str(d) + " (H x W x D) mm.\n\n")


if __name__ == "__main__":

    # imageModGenerator()

    # siftTimer()

    getLongFeats(10)

    # trainingSampleProporptional()