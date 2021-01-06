import numpy as np
import pandas as pd
from HelperFunctions.Utilities import dirMaker
from glob import glob
from random import random
from shutil import copyfile
import tensorflow as tf
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.models import Model

# this script takes the names classified by Hanna Allerkamp and turns them into
# categories

def VGG16Maker(IMAGE_SIZE, noClasses, Trainable = False, Weights = 'imagenet', Top = False):

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
    from tensorflow.keras.layers import Flatten, Dropout, Dense
    from tensorflow.keras.models import Model

    # load the pretrained model and specify the weights being used
    ptm = PretrainedModel(
        input_shape=IMAGE_SIZE,  
        weights=Weights,         
        include_top=Top)     
            
    # boolean to fine-tune the CNN layers
    ptm.trainable = Trainable

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

def modelTrainer(src, modelName, gpu = 0, epochs = 100, batch_size = 64):
    
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
    validImages = glob(valDir + "*/*")
    IMAGE_SIZE = list(cv2.imread(validImages[0]).shape)

    # create the model topology
    model, preproFunc = makeModel(modelName, IMAGE_SIZE, len(classes))

    # create the data augmentation 
    # NOTE I don't think this is augmenting the data beyond the number of samples
    # that are present.... 
    gen_Aug = ImageDataGenerator(                    
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preproFunc
    )

    # create the data generator WITH augmentation
    train_generator = gen_Aug.flow_from_directory(
        trainDir,
        target_size=IMAGE_SIZE[:2],     # only use the first two dimensions of the images
        batch_size=batch_size,
        class_mode='binary',
    )

    # create the validating data generator (NO augmentation)
    gen_noAug = ImageDataGenerator(preprocessing_function=preproFunc)
    valid_generator = gen_Aug.flow_from_directory(
        valDir,
        target_size=IMAGE_SIZE[:2],
        batch_size=batch_size,
        class_mode='binary',
    )

    # train the model 
    r = model.fit(train_generator, validation_data=valid_generator, epochs=epochs)

    print("done\n\n")

def makeModel(modelType, IMAGE_SIZE, noClasses):

    # create the model topology
    # Inputs:   (modelType), text variable which leads to a different model which has been made below
    #           (IMAGE_SIZE), size of the images which are inputted
    #           (noClasses), number of classes (ie number of neurons to use)
    # Outputs:  (model), compiled model as constructd per modelType choice

    if modelType == "VGG19":

        model, preproFunc = VGG19Maker(IMAGE_SIZE, noClasses)

    elif modelType == "VGG16":

        model, preproFunc = VGG16Maker(IMAGE_SIZE, noClasses)

    elif modelType == "ResNet50":

        model, preproFunc = ResNet50Maker(IMAGE_SIZE, noClasses)

    elif modelType == "ResNet101":

        model, preproFunc = ResNet101Maker(IMAGE_SIZE, noClasses)

    elif modelType == "EfficientNetB7":

        model, preproFunc = EfficientNetB7Maker(IMAGE_SIZE, noClasses)

    # print the model toplogy 
    # model.summary()

    return(model, preproFunc)

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

    # select only the values which have been labelled
    valsC = np.array(vals)[np.where([v.find("_")>0 for v in vals])[0]]

    # split the values into their id and features present
    valsS = [v.split("_") for v in valsC]

    # create a sparse matrix of the features present
    valKs = []
    for vi, vk in valsS:
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
    testDir = destDir + "test/"; dirMaker(testDir)
    trainDir = destDir + "train/"; dirMaker(trainDir)
    valDir = destDir + "val/"; dirMaker(valDir)

    trainR, valR, testR = ratios
    if np.sum(ratios) != 1:
        print("!!! Ensure ratios sum to 1 !!!")
        return

    label = ["decidua", "myometrium", "villous"]
    for lb in label:
        dirMaker(testDir + lb + "/")
        dirMaker(trainDir + lb + "/")
        dirMaker(valDir + lb + "/")

    for n, lb in enumerate(label):
        imgStore = []
        # get the spec ID for images which are only a single class (excluding vessesl)
        classID = l[(l["d"] == (n==0)*1) & (l["m"] == (n==1)*1) & (l["vi"] == (n==2)*1)]
        classSpec = np.array(classID.index)

        # get the image path for all these images
        for c in classSpec:
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




if __name__ == "__main__":


    # directory which contains the sections
    imgDirs = '/Volumes/Storage/H653A_11.3/3/FeatureSectionsFinal/linearSect/'
    destDir = '/Volumes/Storage/H653A_11.3/3/FeatureSectionsFinal/'
    infoDir = ""

    labels = getLabels(imgDirs, infoDir)

    # dataOrganiser(imgDirs, destDir, labels)

    modelTrainer(destDir, "VGG16")
    
    