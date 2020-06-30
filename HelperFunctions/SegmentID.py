'''

This script reads from the ndpa file the pins which describe annotatfeatures and 

'''

from Utilities import *
import numpy as np
import tifffile as tifi
import cv2
from scipy.optimize import minimize

tifLevels = [20, 10, 5, 2.5, 0.625, 0.3125, 0.15625]

def align(data, name = '', size = 0, extracting = True):

    # This function will take a whole slide tif file at the set resolution and 
    # extract the target image/s from the slide and create a txt file which contains the 
    # co-ordinates of the key features on that image for alignment
    # Inputs:   (data), directories of the tif files of interest
    #           (featDir), directory of the features
    # Outputs:  (), extracts the tissue sample from the slide

    # get the file of the features information 
    dataFeat = sorted(glob(data + name + 'featFiles/*.feat'))[0:2]
    dataTif = sorted(glob(data + name + 'tifFiles/*' + str(size) + '.tif'))[0:2]

    # create the dictionary of the directories
    featDirs = dictOfDirs(feat = dataFeat, tif = dataTif)

    shapes = {}
    tifShapes = {}
    for spec in featDirs.keys():

        # extract the single segment
        corners, tifShape = segmentExtract(data, featDirs[spec], size, False)
        
        for t in tifShape.keys():
            # NOTE this assumes that appending the features occurs in sequential order... CONFIRM
            tifShapes[t] = tifShape[t]

        # get the feature specific positions
        # NOTE that the list order is in descending order (eg 1a, 1b, 2a, 2b...)
        shape = featAdapt(data, featDirs[spec], corners, size)

        for s in shape.keys():
            # NOTE this assumes that appending the features occurs in sequential order... CONFIRM
            shapes[s] = shape[s]

    # get the segmented images
    dataSegment = dictOfDirs(segments = glob(data + name + 'segmentedSamples/*' + str(size) + '.tif'))

    shiftStore = {}

    segments = list(shapes.keys())
    shiftStore[segments[0]] = (np.array([0, 0]))
    ref = shapes[segments[0]]
    for i in segments[1:]:

        tar = shapes[i]

        # get all the features which both slices have
        tarL = list(tar.keys())
        refL = list(ref.keys())
        feat, c = np.unique(refL + tarL, return_counts=True)
        commonFeat = feat[np.where(c == 2)]

        # create the numpy array with ONLY the common features and their positions
        tarP = list()
        refP = list()
        for cf in commonFeat:
            tarP.append(tar[cf])
            refP.append(ref[cf])
        tarP = np.array(tarP)
        refP = np.array(refP)

        # get the shift needed and store
        res = minimize(objectiveCartesian, (0, 0), args=(tarP, refP), method = 'Nelder-Mead', tol = 1e-6)
        shift = np.round(res.x).astype(int)
        shiftStore[i] = shift

        # ensure the shift of frame is passed onto the next frame
        ref = {}
        for i in tar.keys():
            ref[i] = tar[i] - shift

    # get the measure of the amount of shift
    ss = dictToArray(shiftStore)

    maxSx = np.max(ss[:, 0])
    maxSy = np.max(ss[:, 1])
    minSx = np.min(ss[:, 0])
    minSy = np.min(ss[:, 1])

    tsa = dictToArray(tifShapes)

    # NOTE use tsathe tifShapes list to do this part
    # shapes = [img0, img1, img2, img3, img4] --> is now feats
    # standardise the size of all the images by fitting them into the largest possible dimensions
    fx, fy, fc = np.max(tsa, axis = 0)
    

    # create the new area for all images stored
    newFieldO = np.zeros([fx + maxSx - minSx, fy + maxSy - minSy, 3]).astype(np.uint8)
    fieldResizeO = np.zeros([fx, fy, fc]).astype(np.uint8)   # create an empty matrix to store the image

    # for n in range(len(shiftStore)):
    for n in dataSegment:

        field = cv2.cvtColor(cv2.imread(dataSegment[n]['segments']), cv2.COLOR_BGR2RGB)

        # put the original image into a standardised shape image
        fieldR = fieldResizeO.copy()
        w, h, c = tifShapes[n]                         
        fieldR[:w, :h, :] += field
        # plt.imshow(fieldR); plt.show()

        newField = newFieldO.copy()

        xp = -shiftStore[n][0] + maxSx
        yp = -shiftStore[n][1] + maxSy

        newField[xp:(xp+fx), yp:(yp+fy), :] = fieldR
        
        # save newField as the img

        # plt.imshow(newField); plt.show()


        for s in shapes[n].keys():       # use the target keys in case there are features not common to the previous original 
            v = shapes[n][s]
            pos = tuple(np.round(v - shiftStore[n]).astype(int))
            newField = cv2.circle(newField, pos, 100, (255, 0, 0), 100) 

        x, y, c = newField.shape
        # cv2.imwrite(nameFromPath(dataSegment[n]) + '.jpg', cv2.cvtColor(cv2.resize(adjustImg, (1000, int(1000 * x/y))), cv2.COLOR_BGR2RGB))
        img = cv2.cvtColor(cv2.resize(newField, (1000, int(1000 * x/y))), cv2.COLOR_BGR2RGB) 
        # plt.imshow(img); plt.show()
        cv2.imwrite(data + name + 'segmentedSamples/' + n + "_" + str(size) + 'annotated.jpg', img)


        '''
        # perform the fitting of the slices
        refO = feats[0]         # this is the feature being fitted for, initial call
        im = cv2.cvtColor(cv2.imread(dataSegment[0]), cv2.COLOR_BGR2RGB)
        adjustImg = np.zeros([im.shape[0], im.shape[1], 3]).astype(np.uint8)
        x, y, c = adjustImg.shape
        for i in refO.keys():       # use the target keys in case there are features
            adjustImg = cv2.circle(im, tuple(refO[i]), 100, (255, 0, 0), 100)
        ''' 


    # plt.imshow(newField); plt.show()

    print('done')




    pass

        # NOTE TODO perform rotational optimisation fit




def objectiveCartesian(pos, *args):

    # this function is working out the x and y translation to minimise error between co-ordinates
    # Inputs:   (pos), translational vector to optimise
    #           (args), the reference and target co-ordinates to fit for
    # Outputs:  (err), the squarred error of vectors given the shift pos

    ref = args[0]   # the first argument is ALWAYS the reference
    tar = args[1]   # the second argument is ALWAYS the target

    # error calcuation
    err = np.sum((tar + pos - ref)**2)

    return(err)      

def objectivePolar(w, *args):

    # this function is working out the rotational translation to minimise error between co-ordinates
    # Inputs:   (w), angular translation to optimise
    #           (args), the reference and target co-ordinates to fit for
    # Outputs:  (err), the squarred error of vectors given the shift pos

    ref = args[0]   # the first argument is ALWAYS the reference
    tar = args[1]   # the second argument is ALWAYS the target

    # error calculation
    # err = something

    return(err)      

def featAdapt(data, featDir, corners, size):

    # this function take the features of the annotated features and converts
    # them into the local co-ordinates which are used to align the tissues
    # Inputs:   (data), home directory for all the info
    #           (featDir), the dictionary containing the sample specific dirs
    #           (corners), the global co-ordinate of the top left edge of the img extracted
    #           (size), the size of the image being processed
    # Outputs:  (), save a dictionary of the positions of the annotations relative to
    #               the segmented tissue, saved in the segmented sample folder with 
    #               corresponding name
    #           (specFeatOrder), 

    featInfo = txtToDict(featDir['feat'])[0]
    scale = tifLevels[size] / max(tifLevels)
    featName = list(featInfo.keys())
    nameSpec = nameFromPath(featDir['feat'])

    # remove all the positional arguments
    locations = ["top", "bottom", "right", "left"]
    for n in range(len(featName)):  # NOTE you do need to do this rather than dict.keys() because dictionaries don't like changing size...
        f = featName[n]
        for l in locations:
            m = f.find(l)
            if m >= 0:
                featInfo.pop(f)

    featKey = list()
    for f in sorted(featInfo.keys()):
        key = f.split("_")
        featKey.append(key)
    
    # create a dictionary per identified sample 
    specFeatInfo = {}
    specFeatOrder = {}
    for v in np.unique(np.array(featKey)[:, 1]):
        specFeatInfo[v] = {}

    # allocate the scaled and normalised sample to the dictionary PER specimen
    for f, p in featKey:
        specFeatInfo[p][f] = (featInfo[f + "_" + p] * scale).astype(int)  - corners[p]
        
    # save the dictionary
    for p in specFeatInfo.keys():
        name = nameFromPath(featDir['tif']) + p + "_" + str(size) + ".feat"
        specFeatOrder[nameSpec+p] = specFeatInfo[p]
        dictToTxt(specFeatInfo[p], data + "segmentedSamples/" + name)

    return(specFeatOrder)

def segmentExtract(data, featDir, size, extracting = True):

    # this funciton extracts the individaul sample from the slice
    # Inputs:   (data), home directory for all the info
    #           (featDir), the dictionary containing the sample specific dirs 
    #           (size), the size of the image being processed
    #           (extracting), boolean whether to load in image and save new one
    # Outputs:  (), saves the segmented image 

    featInfo = txtToDict(featDir['feat'])[0]
    locations = ["top", "bottom", "right", "left"]
    scale = tifLevels[size] / max(tifLevels)

    # create the directory to save the samples
    segSamples = data + 'segmentedSamples/'
    try:
        os.mkdir(segSamples)
    except:
        pass
    tifShape = {}
    keys = list(featInfo.keys())
    bound = {}

    for l in locations:

        # find out what samples are in each
        pos = [k.lower() for k in keys if l in k.lower()]

        # store each array of positions and re-scale them to the size image
        # read in chronological order
        store = {}
        for p in sorted(pos):
            val = p.split("_")[-1]
            store[val] = (featInfo[p]*scale).astype(int)

        # save each co-ordinate point in a dictionary
        bound[l] = store

    # read in the tif image
    tif = tifi.imread(featDir['tif'])
    name = nameFromPath(featDir['tif'])

    areas = {}

    # for each entry extract info --> ASSUMES that there will be the same number of bounding points found
    for p in bound['bottom'].keys():

        # get the y position
        t = bound['top'][p][1]
        b = bound['bottom'][p][1]

        # get the x position
        r = bound['right'][p][0]
        l = bound['left'][p][0]

        yBuff = int((b - t) * 0.05)
        xBuff = int((r - l) * 0.05)

        # create padding
        tb = t - yBuff; bb = b + yBuff
        lb = l - xBuff; rb = r + xBuff

        # ensure bounding box is within the shape of the tif image
        if tb < 0:
            tb = 0
        if bb > tif.shape[0]:
            bb = tif.shape[0]
        if lb < 0:
            lb = 0
        if rb > tif.shape[1]:
            rb = tif.shape[1]
        
        areas[p] = np.array([lb, tb])

        # extract the portion of the tif image defined by the boundary
        tifSeg = tif[tb:bb, lb:rb, :]
        tifShape[name+p] = tifSeg.shape

        if extracting:
            # save the segmented tissue
            name = nameFromPath(featDir['tif']) + p + "_" + str(size) + '.tif'
            tifi.imwrite(segSamples + name, tifSeg)

    return(areas, tifShape)

        # plt.imshow(tifSeg); plt.show()

# dataHome is where all the directories created for information are stored 
dataHome = '/Users/jonathanreshef/Documents/2020/Masters/TestingStuff/Segmentation/Data.nosync/'
# dataHome = '/Volumes/Storage/'

# dataTrain is where the ndpi and ndpa files are stored 
dataTrain = dataHome + 'HistologicalTraining2/'
# dataTrain = dataHome + 'FeatureID/'
name = ''
size = 3

align(dataTrain, name, size)




'''
    # plt.imshow(adjustImg); plt.show()
    cv2.imwrite(dataSegment[0] + '.jpg', cv2.cvtColor(cv2.resize(adjustImg, (1000, int(1000 * x/y))), cv2.COLOR_BGR2RGB))

    resAdj = list()
    for n in range(1, len(feats)):

        # get the reference and target slices to optimise fitting for
        tarO = feats[n]       # these are the features being adjusted

        # get all the features which both slices have
        tarL = list(tarO.keys())
        refL = list(refO.keys())
        feat, c = np.unique(refL + tarL, return_counts=True)
        commonFeat = feat[np.where(c == 2)]

        # create the numpy array with ONLY the common features and their positions
        tarP = np.zeros([len(commonFeat), 2])
        refP = np.zeros([len(commonFeat), 2])
        for i in range(len(commonFeat)):
            tarP[i, :] = tarO[commonFeat[i]]
            refP[i, :] = refO[commonFeat[i]]

        # perform x,y optimisation fit
        res = minimize(objectiveCartesian, (0, 0), args=(refP, tarP), method = 'Nelder-Mead', tol = 1e-6)
        resAdj.append(res.x)

        # move the target co-ordinates and use this for the new reference
        im = cv2.cvtColor(cv2.imread(dataSegment[n]), cv2.COLOR_BGR2RGB)
        # plt.imshow(im); plt.show()

        # get the new size of the image
        x = int(abs(res.x[1]) + im.shape[0])
        y = int(abs(res.x[0]) + im.shape[1])
        adjustImg = np.zeros([x, y, 3]).astype(np.uint8)
        xi, yi, c = im.shape

        # adjust the position of the image, NOTE these should all be based on the im shape, NOT on res
        if (res.x[0] <= 0) & (res.x[1] <= 0):
            # bottom right shift
            adjustImg[-xi:, -yi:, :] = im
        elif (res.x[0] <= 0) & (res.x[1] > 0):
            # top right shift
            adjustImg[:xi, -yi:, :] = im
        elif (res.x[0] > 0) & (res.x[1] <= 0):
            # bottom left shift
            adjustImg[-xi:, :yi, :] = im
        elif (res.x[0] > 0) & (res.x[1] > 0):
            # top left shift
            adjustImg[:xi, :yi, :] = im

        
        # NOTE ATM it is drawing incorrectly....
        refO = {}                   # re-initialise the ref dict
        for i in tarO.keys():       # use the target keys in case there are features not common to the previous original 
            pos = tuple((tarO[i] - res.x).astype(int))
            adjustImg = cv2.circle(adjustImg, pos, 100, (255, 0, 0), 100) 
            refO[i] = pos

        
        cv2.imwrite(dataSegment[n] + '.jpg', cv2.cvtColor(cv2.resize(adjustImg, (1000, int(1000 * x/y))), cv2.COLOR_BGR2RGB))

        # plt.imshow(adjustImg); plt.show()
'''