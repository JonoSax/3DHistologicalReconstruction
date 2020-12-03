'''
This contains functions which are used by multipe different scripts
'''

import numpy as np
import matplotlib.pyplot as plt
from shutil import copy
import os
import cv2
import tifffile as tifi
from glob import glob
from PIL import Image
import multiprocessing
from itertools import repeat
import pandas as pd

# magnification levels of the tif files available
tifLevels = [20, 10, 5, 2.5, 0.625, 0.3125, 0.15625]

def listToTxt(data, dir, **kwargs):

    # Converts a list of information into a txt folder with the inputted name
    # Inputs:   (data), the single list to be saved
    #           (dir), the exact name and path which this list will be saved as
    #           (*args), inputs that appear at the top of the saved file
    # Outputs:  (), txt file saved in directory 

    # txt layout
    '''
    ArgumentNo_[number of addition arguments to read in]
    Arg_1_[some data]
    Arg_2_[some data]
    ...
    Rows_[X number of rows]
    Cols_[Y number of columns]
    [x0y0],[x0y1],[x0y2],[x0yY],...
    [x1y0],[x1y1],[x1y2],[x1yY],...
    [xXyY],...
    EndData
    '''
    # ensure that the exact directory being specified exists, if not create it
    dirMaker(dir)
    f = open(dir, 'w')

    # declar
    f.write("ArgNo_" + str(len(kwargs)) + "\n")

    argK = list()
    argV = list()


    # get optional arguments
    for k in kwargs.keys():
        argK.append(k)

    for v in kwargs.values():
        argV.append(v)

    # write the arguments at the top of the file
    for i in range(len(kwargs)):
        f.write(argK[i] + "_" + argV[i] + "\n")        
    
    f.write("ListEntries_" + str(len(data)) + "\n")
    for i in range(len(data)):
        sample = data[i]
        X, Y = sample.shape
        f.write("Rows_" + str(X) + "\n")
        f.write("Cols_" + str(Y) + "\n")

        for x in range(X):
            for y in range(Y):
                f.write(str(sample[x, y]))
                if (y+1)%Y:
                    f.write(",")
                else:
                    f.write("\n")

    f.write("EndData")

    f.close()

def dirMaker(dir):

    '''
    creates directories (including sub-directories)

    Input:    \n
    (dir), path to be made

    Output:   \n
    (), all sub-directories necessary are created\n
    (made), boolean whether this is the first time the directory has been made
    '''

    # ensure that the exact directory being specified exists, if not create it
    made = False     # boolean if the directory was successfully made for the first time
    dirSplit = dir.split("/")
    dirToMake = ""
    for d in range(dir.count("/")):
        dirToMake += str(dirSplit[d] + "/")
        try:
            os.mkdir(dirToMake)
            made = True
        except:
            pass
    return(made)

def txtToList(dir):

    # Reads in a text file which was saved with the listToTxt function
    # Inputs:   (dir), the name of a single file
    # Outputs:  (dataMain), a list containing the data
    #           (dataArgs), a dictionary containing the argument data

    f = open(dir, 'r')

    # argument numbers
    argNo = int(f.readline().replace("ArgNo_", ""))

    # store the arguments in a dictionary
    args = {}
    for i in range(argNo):
        arg = f.readline().split("_")
        args[arg[0]] = arg[1].replace("\n", "")

    allList = list()

    # use to verify all the information has been collected

    sampleList = list()
    listNo = int(f.readline().replace("ListEntries_", ""))
    for n in range(listNo):
        rows = int(f.readline().replace("Rows_", ""))
        cols = int(f.readline().replace("Cols_", ""))
        storedData = np.zeros([rows, cols])
        for r in range(rows):
            values = f.readline().split(",")
            for c in range(cols):
                storedData[r, c] = int(values[c].replace("\n", ""))

        sampleList.append(storedData)

    return(sampleList, args)

def dictToTxt(data, path, **kwargs):
    # This function saves a dictionary as a txt file
    # Converts a dictinoary into a txt file with the inputted name
    # Inputs:   (data), the single dictionary to be saved
    #           (dir), the exact name and path which this list will be saved as
    #           (*args), inputs that appear at the top of the saved file
    # Outputs:  (), txt file saved in directory 
    
    # ensure that the exact directory being specified exists, if not create it
    dirMaker(path)

    f = open(path, 'w')

    # declar
    f.write("ArgNo_" + str(len(kwargs)) + "\n")

    argK = list()
    argV = list()

    # get optional arguments
    for k in kwargs.keys():
        argK.append(k)

    for v in kwargs.values():
        argV.append(str(v))

    # write the arguments at the top of the file
    for i in range(len(kwargs)):
        f.write(argK[i] + "_" + argV[i] + "\n")        

    # this is just boilerplate code so allow for the previous implementations
    # of dictToTxt without specifying the type to still function
    try: kwargs["classType"]
    except: kwargs["classType"] = None 
    
    # set the number of entries
    f.write("Entries:" + str(len(data.keys())) + "\n")

    # write the informations
    for n in data.keys():
        if type(data[n]) == kwargs["classType"]:

            da = data[n].__dict__
            f.write("Sample:" + str(n) + "\n")
            for d in da:
                f.write(str(d) + ":" + str(type(da[d])) + ":" + str(da[d]) + "\n")
            
        else:
            f.write(str(n) + ":")
            for v in data[n]:
                f.write(str(v) + " ")
            f.write("\n")
    
    f.close()

def txtToDict(path, typeV = int, typeID = str):

    # Reads in a text file which was saved with the dictToTxt function
    # Inputs:   (dir), the name of a single file
    #           (typeV), value type to load into dictionary, defaults int
    #           (typeID), ID type to load into dictionary, defaults string
    # Outputs:  (sampleDict), either a list of dictionary of the information: 
    #           if paths is a list then the output is a dictinary named by the samples of the info
    #           if the paths is a string then the output is a list 


    def extract(p):
        pathinfo = {}
        f = open(p, 'r')
        # argument numbers
        argNo = int(f.readline().replace("ArgNo_", ""))
        # store the arguments in a dictionary
        args = {}
        for i in range(argNo):
            line = f.readline()
            arg = line.split("_")
            args[arg[0]] = arg[1].replace("\n", "")

            # create conditions to read in the info correctly
            # if logical True operator
            if args[arg[0]].lower().find("true") > -1:
                args[arg[0]] = True
            # if logical False operator
            elif args[arg[0]].lower().find("false") > -1:
                args[arg[0]] = False
            # if a tuple
            elif args[arg[0]].find("(") < args[arg[0]].find(")"):
                args[arg[0]] = tuple(np.array(args[arg[0]].replace("(", "").replace(")", "").split(",")).astype(int))

        # use to verify all the information has been collected
        dictNo = int(f.readline().replace("Entries:", ""))
        for n in range(dictNo):
            info = f.readline().split(":")
            key = typeID(info[0])
            data = info[1].replace("\n", "")
            # save data as a dictionary
            pathinfo[key] = np.array(data.split(" ")[0:-1]).astype(typeV)

        return([pathinfo, args])

    def typeReader(type, info):

        # this function converts string into the data type specified
        # Input:    (type), string of the type
        #           (info), info as a string
        # Output:   (data), the info converted into that data type

        # if numpy array
        if type == "<class 'numpy.ndarray'>":

            # get the nunumbers
            num = info.split("[")[-1].split("]")[0].split(" ")
            data = []

            # add the number into the array
            for n in num:
                try: data.append(float(n))
                except: pass
            data = np.array(data)

        # if int
        elif type == "<class 'int'>":
            data = int(info)

        # if float
        elif type == "<class 'numpy.float64'>":
            data = float(info)
        
        return(data)


    # if a list of paths is provided then create a dictionary containing dictionaries
    # of all the dictionaries of info
    if type(path) == list:
        sampleDict = {}
        for p in path:
            name = nameFromPath(p)
            sampleDict[name] = extract(p)

    # if the intput is a feature object
    elif str(typeV).find("feature") > -1:
        fi = open(path, 'r').read().split("\n")

        sampleDict = {}
        # ignore the first 3 lines
        for f in fi[3:]:
            if f == "":
                continue
            elif f.find("Sample")>-1:
                obj = typeV()
                samp = int(f.split("Sample:")[-1])
            else:
                i = f.split(":")
                exec(f"obj.{i[0]} = typeReader(i[1], i[2])")

                '''if i[0] == "refP":
                    obj.refP = 
                elif i[0] == "tarP":
                    obj.tarP = typeReader(i[1], i[2])
                elif i[0] == "dist":
                    obj.dist = typeReader(i[1], i[2])
                elif i[0] == "ID":
                    obj.ID = typeReader(i[1], i[2])
                '''
            sampleDict[samp] = obj
            
    elif type(path) == str:
        sampleDict = extract(path)

    else:
        sampleDict = []

    return(sampleDict)

def denseMatrixViewer(coords, plot = True, unique = False):

    # This function takes in a numpy array of co-ordinates in a global space and turns it into a local sparse matrix 
    # which can be view with matplotlib
    # Inputs:   (coords), a list of co-ordinates
    #           (plot), boolean to control plotting
    #           assume it is a vertical stack and colour them differently
    # Outputs:  (), produces a plot to view
    #           (area), the array 

    # if unique, then only show the features which are unique
    if unique:
        coordsmatch = []
        for c in coords:
            if type(c) is dict:
                coordsmatch.append(c)

        samekeys, _ = uniqueKeys(coordsmatch)
        p = 0
        for n, c in enumerate(coords):
            if type(c) is dict:
                coords[n] = samekeys[p]
                p += 1

    # if in the list of coords there is a dictionary, convert to an array
    for n, c in enumerate(coords): 
        if type(c) is dict:
            coords[n] = dictToArray(c)

    # enusre numpy array is int
    coordsMax = np.vstack(coords).astype(int)

    # get max and min size to bound the box of the whole image
    Xmax = int(coordsMax[:, 0].max())
    Xmin = int(coordsMax[:, 0].min())
    Ymax = int(coordsMax[:, 1].max())
    Ymin = int(coordsMax[:, 1].min())

    # add padding to the view
    pad = int(np.mean(np.array([Ymax - Ymin + 1, Xmax - Xmin + 1])) * 0.05)

    area = np.zeros([Ymax - Ymin + 1 + 2*pad, Xmax - Xmin + 1 + 2*pad, 3]).astype(np.uint8)

    cols = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
    sizes = [20, 16, 12, 8]

    for coord, col, s in zip(coords, cols, sizes):
        coord = (np.array(coord) - [Xmin, Ymin] + pad).astype(int)
        coord = list(coord)
        if type(coord[0]) is np.int64:
            coord = [coord]
        for xp, yp in coord:
            cv2.circle(area, (xp, yp), s, col, 4)

    if plot:
        plt.imshow(area)
        plt.show()

    shift = (Xmin, Ymin)

    return(area, shift)

def maskCover(dir, dirTarget, masks, small = True):

    # This function adds add the mask outline to the image
    # Inputs:   (dir), the SPECIFIC name of the tif image the mask was made on
    #           (dirTarget), the location to save the image
    #           (masks), list of each array of co-ordinates for all the annotations
    #           (small), boolean whether to add the mask to a smaller file version (ie jpeg) or to a full version (ie tif)
    # Outputs:  (), re-saves the image with mask of the vessels drawn over it

    imgR = tifi.imread(dir)
    hO, wO, cO = imgR.shape

    # if the image is more than 70 megapixels downsample 
    if (hO * wO >= 100 * 10 ** 6) & small:
        size = 2000
        aspectRatio = hO/wO
        imgR = cv2.resize(imgO, (size, int(size*aspectRatio)))

    h, w, c = imgR.shape

    # scale the kernel to the downsampled image
    scale = h/hO
    for mask in masks:
        maskN = np.unique((mask * scale).astype(int), axis = 0)
        for x, y in maskN:
            # inverse colours of mask areas
            imgR[y, x, :] = 255 - imgR[y, x, :]

    imgR = Image.fromarray(imgR)
    if small:
        imgR.save(dirTarget + ".jpeg")
    else:
        imgR.save(dirTarget + ".tif")

    # cv2.imwrite(newImg, imgR, [cv2.IMWRITE_JPEG_QUALITY, 80])
    # cv2.imshow('kernel = ' + str(kernel), imgR); cv2.waitKey(0)

def nameFromPath(paths, n = 2):
    # this function extracts the names from a path/s
    # Inputs:   (paths), either a list or string of paths 
    #           (n), number of 
    # Outputs:  (names), elist of the names from the paths

    if paths == None:
        return(None)

    # if it is a string input, output a list
    pathStr = False
    if type(paths) is str:
        paths = [paths]
        pathStr = True

    # if it is a dictionary input, convert to a list  
    if type(paths) is dict:
        pathsDict = paths
        paths = []
        for p in list(pathsDict.values()):
            paths.append(p)

    names = list()
    for path in paths:
        # choose the last part of the path and the suffix
        name = path.split("/")[-1].split(".")[0]

        # each '_' indicates a new piece of information in the name
        # if there is at least one underscore then seperate the information
        # defaults to saving only the first piece of underscored information 
        # (this is often the specific sample within the specimen, the next 
        # _ will often be the size of the sample used)
        if len(name.split("_")) > 1:
            name = "_".join(name.split("_")[0:n])
            name = name.replace(" ", "")
        names.append(name)

    # if the path input is a string, it will expect an output of a string as well
    if pathStr:
        names = names[0]

    return names

def regionOfPath(paths, n = 1):
    # this function extracts the user defined position of the (ie what is the path 1 directory up, 2....)
    # Inputs:   (paths), either a list or string of paths 
    #           (n), how many levels of directories up you want to move, defaults to moving up one level
    # Outputs:  (names), elist of the names from the paths

    pathStr = False
    if type(paths) is str:
        paths = [paths]
        pathStr = True

    names = list()
    for path in paths:

        # select the region of the path you are interested in
        name = "/".join(path.split("/")[:-n]) + "/"
        
        names.append(name)

    # if the path input is a string, it will expect an output of a string as well
    if pathStr:
        names = names[0]

    return names

def dictOfDirs(**kwargs):

    # this function takes a list of directories and a key name and puts them 
    # into a dictionary which is named after the sample name
    # Inputs:   (kwargs), needs the file type name and then a list of directories
    # Outputs:  (dictToWrite), a dictinoary which contains dictionaries of the samples
    #               and each of the files associated with it

    dictToWrite = {}

    # get all the names
    names = list()
    for k in kwargs:
        if type(kwargs[k]) == list:
            names += nameFromPath(kwargs[k], 3)
        else:
            names += [nameFromPath(kwargs[k]), 3]

    # get all the unique names
    names = np.unique(np.array(names))


    for n in names:
        dictToWrite[n] = {}

    for k in kwargs:
        path = kwargs[k]
        if type(path) != list:  path = [path]

        spec, no = np.unique(nameFromPath(path, 3), return_counts = True)

        for s, n in zip(spec, no):
            if n > 1:
                dictToWrite[s][k] = list()

        if len(spec) == 0:
            dictToWrite[s][k] = {}  # if there is nothing there just leave empty

        # if there are multiple files under the label for that specimen, append ot a list
        for p in path:
            if no[np.where(spec == nameFromPath(p))] > 1:
                dictToWrite[nameFromPath(p, 3)][k].append(p)
            else:
                dictToWrite[nameFromPath(p, 3)][k] = [p]

    return(dictToWrite)

def dictToArray(d, type = float):

    # converts a dictionary into a nparray. This works for only 1 layered dictionaries into a 2D matrix
    # Input:    (d), dictionary
    #           (type), number type to use
    # Output:   (l), array

    l = np.array(list(d.values())).astype(type)

    return(l)

def hist_match(source, template):
    """
    Courtesy of https://stackoverflow.com/questions/31490167/how-can-i-transform-the-histograms-of-grayscale-images-to-enforce-a-particular-r/31493356#31493356
    Adjust the pixel values of a layer of image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # remove the effect of black (it is working on a masked image)
    s_counts[0] = 0
    t_counts[0] = 0

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    normChannel = (interp_t_values[bin_idx].reshape(oldshape)).astype(np.uint8)

    # return the image with normalised distribtution of pixel values and 
    # in the same data type as np.uin8
    return normChannel

def findangle(point1, point2, point3 = None):

        # this function finds the anti-clockwise angle between three points, where point2 
        # is connected to both point1 and point2
        # Inputs:   (point1, 2), the positions of the points of interet
        #           (point3), an optional point, if not input then defaults to the horizontal 
        #           along the x-axis

        # create copies to prevent modifications to dictionaries
        p1 = point1.copy()
        p2 = point2.copy()

        def dotpro(vector1, vector2 = [0, 1]):

            # find the dot product of vectors
            # vector2 if not inputted defaults to the horizontal 

            unit_vector_1 = vector1 / np.linalg.norm(vector1)
            unit_vector_2 = vector2 / np.linalg.norm(vector2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            if (dot_product * 1 != dot_product).all():
                return(np.inf)      # if there is any kind of error, just return a silly number
            return dot_product

        # makes point2 the origin
        p1 -= p2
        if point3 is None:
            p3 = [0, 1]
        else:
            p3 = point3.copy()
            p3 -= p2

        # get the dot product of the two vectors created or defaulted 
        dp1 = dotpro(p1)
        dp3 = dotpro(p3)

        # if there is an error in the dotproduct calculation, return a silly angle
        if dp1 == np.inf or dp3 == np.inf:
            return(np.inf)

        # find the angle of the vectors relative to each other
        angle3 = np.arccos(dp3)
        angle1 = np.arccos(dp1)

        # adjust the angles relative to their position on the local grids
        if p1[0] < 0:
            angle1 = 2*np.pi - angle1

        # adjust the angles 
        if p3[0] < 0:
            angle3 = 2*np.pi - angle3
        
        angle = abs(angle1 - angle3)
        
        return(angle)

def uniqueKeys(dictL):

    # return dictionaries which contain only the featues which all
    # the dictionaries have
    # Inptus:   (dictL), list of dictionaries  
    # Outputs:  (dictMod), list of dictionaries with common values
    #           (commonKeys), list of the common keys found

    # collect all the keys used
    keys = []
    for d in dictL:
        keys += d.keys()

    # count the occurence of each unique key and get only the ones whch 
    # occur in all the dictionaries
    feat, c = np.unique(keys, return_counts=True)
    commonKeys = sorted(feat[np.where(c == len(dictL))])
    
    # create the dictionary with ONLY the common features 
    dictMod = []
    for d in dictL:
        dN = {}
        for cf in commonKeys:
            dN[cf] = d[cf]

        dictMod.append(dN)

    return(dictMod, commonKeys)

def matchMaker(resInfo, matchedInfo = [], manual = False, dist = 50, cpuNo = False, tol = 0.05, spawnPoints = 10, anchorPoints = 5, distCheck = True):

    '''
    ---------- NOTE the theory underpinning this function ----------
    The combination of individual features which produces the most matched features 
    due to being the most spatially coherent with each other, most likely contain 
    the features that represent actual biological structures becasue biological 
    structures between individual samples which are physically close to each other 
    are also spatailly coherent
    this takes lists of all the information from the SIFT feature identification 
    and bf matching and returns only n number of points which match the criteria

    Inputs:   \n
    (resInfo), all the information returned by the sift operator after being
    brute forced matched for that specific resolutionn\n
    (matchInfo), points that have already been found\n
    (manual), boolean as to whether the features that have been inputted 
    include manually annotated features. If True then treat the manual
    annotations as "ground truth" and don't overwrite\n
    (dist), sets the minimium distance between each feature. A larger distance 
    reduces the number of features that can be found but will also ensure 
    that features are  better spread across the sample, rather than being pack 
    arond a single strong feature\n
    (featNo), minimum number of features to find in the images. depreceated\n
    (cpuNo), number of CPUs to use, allows for parallelisation of the maximum
    number of features loop. NOTE this should be set to FALSE if the 
    processing of samples is parallelised --> you can't spawn processes from spawns\n
    (tol), the tolerance of the % of the matched features which are not included in the 
    the matching before breaking (ie 0.05 means that if there are no successful 
    matches for 5% of all the features found then break)\n
    (spawnPoints), the number of the points which are found by findgoodfeatures to be used to help
    find other good featues (the higher the number the more accurate the spatially aware 
    feature finding will be but will also become slower)\n
    (anchorPoints), the number of different feature sets to try before returning
    the infostore\n
    (distCheck), boolean whether to take into account the distance of objects for 
    spatial cohesion\n

    Outputs:  \n
    (infoStore), the set of features and corresponding information that have been 
    found to have the most spatial coherence\n
    '''

    # if there is a repeated feature, delete it (this comes from adding manual features)
    for n1, mi1 in enumerate(matchedInfo):
        for n2, mi2 in enumerate(matchedInfo):

            # if the same feature don't use it
            if n1 == n2:
                continue

            # if there are repeated features don't use
            if (mi1.tarP == mi2.tarP).all() or (mi1.refP == mi2.refP).all():
                del matchedInfo[n2]

    infoStore = matchedInfo

    # create a list containing all the confirmed matched points and current ponts of interest
    allInfo = matchedInfo.copy() + resInfo.copy()

    # if there aren't enough features to even perform this feat find process
    # just end the script
    if len(allInfo) < 2:
        return(allInfo)

    # sort the information based on the distance
    allInfo = sorted(allInfo, key=lambda allInfo: allInfo.dist)

    # append the next n number of best fit features to the matches but 
    # ONLY if their angle from the two reference features is within a tolerance 
    # range --> this heavily assumes that the two best fits found are actually 
    # good features...
    # try up to n times (spawnPoints): NOTE this is important because it reduces the reliance on the assumption 
    # that feature with lowest distance score is in fact and actual feature. This instead allows
    # for the feature finding process to rely more on the coherence of all the other 
    # features relative to each other

    # create the spawning points for the searches
    if len(allInfo) < spawnPoints:
        spawnPoints = len(allInfo) - 1
    elif cpuNo > spawnPoints:
        spawnPoints = cpuNo

    matchInfos = []
    # get the two best features from within a select range of the data
    for fits in range(spawnPoints):
        matchInfos.append(findbestfeatures(allInfo[fits:fits+spawnPoints], dist))

    # if the feature matching has to be performed sequentially (ie for non-rigid defomation)
    # then this step can definitely be parallelised

    # NOTE for some reason this is slower ot paraellise (both by pool and process) than 
    # being calculated sequentially.... WHY???
    if cpuNo != False:
        # Using Pool
        '''
        with multiprocessing.Pool(cpuNo) as pool:
            matchInfoNs = pool.starmap(findgoodfeatures, zip(matchInfos, repeat(allInfo), repeat(dist), repeat(tol)), repeat(r))
        '''

        # Using Process
        job = {}
        qs = {}
        for n, m in enumerate(matchInfos):
            qs[n] = multiprocessing.Queue()
            job[n] = multiprocessing.Process(target=findgoodfeatures, args = (m, allInfo, dist, tol, anchorPoints, distCheck, qs[n], ))
            job[n].start()
        matchInfoNs = []
        for n in job:
            matchInfoNs.append(qs[n].get())
            job[n].join()

        infoStore = matchInfoNs[0]
        for m in matchInfoNs[1:]:
            if len(m) > len(infoStore):
                infoStore = m

    # Serialised
    else:
        for matchInfo in matchInfos:

            # find features which are spatially coherent relative to the best feature for both 
            # the referenc and target image and with the other constrains
            matchInfoN = findgoodfeatures(matchInfo, allInfo, dist, tol, anchorPoints, distCheck)

            # Store the features found and if more features are found with a different combination
            # of features then save that instead
            if len(matchInfoN) > len(infoStore): 
                # re-initialise the matches found
                infoStore = matchInfoN

            # if there are no more features to search through, break
            if (len(allInfo) < 3): # or (len(infoStore) > 200): 
                break
                
            # print(str(fits) + " = " + str(len(matchInfoN)) + "/" + str(len(resInfo)))
                
        # denseMatrixViewer([infoStore.refP, infoStore.tarP], True)

    return(infoStore)

def findbestfeatures(allInfo, dist):

    # find the two best features ensuring that they are located 
    # a reasonable distane away
    # Inputs:   (allInfo), the list of features which contain the matches
    #           (dist), the sqaured distance between each feature to maintain

    # get the best features and remove from the list of sorted features
    matchInfo = []
    matchInfo.append(allInfo[0])
    
    # get the second best feature, ensuring that it is an acceptable distance away from 
    # the best feature (sometimes two really good matches are found pretty much next
    # to each other which isn't useful for fitting)
    for n, i in enumerate(allInfo):

        # if the distance between the next best feature is less than 100 
        # pixels, don't use it
        if (np.sqrt(np.sum((matchInfo[0].refP - i.refP)**2)) < dist) or (np.sqrt(np.sum((matchInfo[0].tarP - i.tarP)**2)) < dist):
            continue
        
        # if the 2nd best feature found meets the criteria append and move on
        else:
            matchInfo.append(i)
            break

    return(matchInfo)

def findgoodfeatures(matchInfo, allInfo, dist, tol, r = 5, distCheck = True, q = None):

    # find new features in the ref and target tissue which are positioned 
    # in approximately the same location RELATIVE to the best features found already

    matchInfoN = []

    # append the already found matching info
    matchInfoN += matchInfo

    # from all the remaining features, find the ones that meet the characteristics:
    #   - The new feature is a distance away from all previous features
    #   - The new features on each sample are within a threshold angle and distance
    #   difference relative to the best features found 
    noFeatFind = 0  # keep track of the number of times a match has not been found
    for an, i in enumerate(allInfo):

        # if the difference between the any of the already found feature is less than dist 
        # pixels, don't use it
        repeated = False

        for mi in matchInfoN:
            if (np.sqrt(np.sum((mi.refP - i.refP)**2)) < dist) or (np.sqrt(np.sum((mi.tarP - i.tarP)**2)) < dist):
                repeated = True
                break           

        if repeated:
            continue

        # get the relative angles of the new features compared to the best features 
        # found. Essentially triangulate the new feature relative to all the previously 
        # found features. This is particularly important as the further down the features
        # list used, the worse the sift match is so being in a position relative to all 
        # the other features found becomes a more important metric of fit
        # print("FEAT BEING PROCESSED")
        angdist = []
        ratiodist = []
        # use, up to, the top n best features: this is more useful/only used if finding
        # LOTS of features as this fitting procedure has a O(n^2) time complexity 
        # so limiting the search to this sacrifices limited accuracy for significant 
        # speed ups
        for n1, mi1 in enumerate(matchInfoN[:r]):
            for n2, mi2 in enumerate(matchInfoN[:r]):

                # if the features are repeated, don't use it
                if n1 == n2:
                    continue

                # find the angle for the new point and all the previously found point
                if (mi1.refP).any() == np.inf:
                    print("TO INFINITY AND BEYOND")
                newrefang = findangle(mi1.refP, mi2.refP, i.refP)
                newtarang = findangle(mi1.tarP, mi2.tarP, i.tarP)

                # store the difference of this new point relative to all the ponts
                # previously found 
                # NOTE this works on the assumption that the scale of the images
                # remains the same
                angdist.append(abs(newrefang - newtarang))

            # get the distances of the new points to the best feature
            newrefdist = np.sqrt(np.sum((mi1.refP - i.refP)**2))
            newtardist = np.sqrt(np.sum((mi1.tarP - i.tarP)**2))

            if newrefdist - newrefdist == 0 and newtardist - newtardist == 0:
                # finds how much larger the largest distance is compared to the smallest distance
                ratiodist.append((newtardist/newrefdist)**(1-((newrefdist>newtardist)*2)) - 1)
            else:
                ratiodist.append(np.inf)

        # if the new feature is than 5 degress off and within 5% distance each other 
        # from all the previously found features then append 
        # NOTE using distance is a thresholding criteria which doesn't work when there 
        # is deformation
        angConfirm = (np.array(angdist) < 10/180*np.pi).all()
        distConfirm = (np.array(ratiodist) < 0.05).all() or not distCheck       # inverse distCheck boolean so 
                                                                                # when False, distCheck is always true
                                                                                # and when True, only works when condition met
        if angConfirm and distConfirm:
        # NOTE using median is a more "gentle" thresholding method. allows more features
        # but the standard of these new features is not as high
        # if np.median(angdist) < 180/180*np.pi and np.median(ratiodist) < 1:
            # add the features
            matchInfoN.append(i)
            noFeatFind = 0
        else:
            # if more than 5% of all the features are investigated and there are no
            # new features found, break the matching process (unlikely to find anymore
            # good features)
            noFeatFind += 1
            if noFeatFind > int(len(allInfo) * tol):
                break

    if type(q) is type(None):
        return(matchInfoN)
    else:
        q.put(matchInfoN)

def nameFeatures(imgref, imgtar, matchedInfo, txtsz = 0.5, combine = False, width = 3):

    # this function takes a list of the feature objects and adds them
    # as annotations to both the ref and target images
    # Inputs:   (imgref, tar), images to annotate with the reference and target points
    #           (matchedInfo), list of the feature objects
    #           (txtsz), annotation size
    #           (combine), boolean whether to annotate individaul iamges or combine
    #               and draw lines between the features
    # Outputs:  (images), either the individual images with their features 
    #               annotated on with the two combined and lines indicating the
    #               locations of corresponding features
    # different colour for each resolution used to find features
    colours = [(255, 0, 0), (255, 0, 255), (255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 255, 255)]
    
    for n, nF in enumerate(matchedInfo):

        # if there is no match info just assign it to 0 (ie was a manual annotaiton)
        try: md = int(nF.dist); 
        except: md = np.inf; 

        try: ms = np.round(nF.size, 2)
        except: ms = np.inf

        # try the name otherwise none
        try: name = nF.ID
        except: names = "None"
        
        # mark the feature
        newref = nF.refP.astype(int)
        newtar = nF.tarP.astype(int)

        # draw the location of the feature
        cv2.circle(imgref, tuple(newref.astype(int)), int(txtsz*10), colours[nF.res], int(txtsz*6))
        cv2.circle(imgtar, tuple(newtar.astype(int)), int(txtsz*10), colours[nF.res], int(txtsz*6))

        # add the info about the feature
        text = str(name )#+ ", d: " + str(md) + ", s: " + str(ms))
        
        # add the feature number onto the reference image
        cv2.putText(img = imgref, text = str(name), 
        org = tuple(newref + np.array([-10, 15])),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = txtsz, color = (255, 255, 255), thickness = int(txtsz*10))

        cv2.putText(img = imgref, text = str(text), 
        org = tuple(newref + np.array([-10, 15])),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = txtsz, color = (0, 0, 0), thickness = int(txtsz*4))

        # add the feature number onto the reference image
        cv2.putText(img = imgtar, text = str(name), 
        org = tuple(newtar + np.array([-10, 15])),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = txtsz, color = (255, 255, 255), thickness = int(txtsz*10))
        
        cv2.putText(img = imgtar, text = str(text), 
        org = tuple(newtar + np.array([-10, 15])),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = txtsz, color = (0, 0, 0), thickness = int(txtsz*4))

    # if the combine flag is true, combine the images and draw the lines
    # between the features to connect them
    if combine:
        
        # assumes they are both the same size 
        x, y, c = imgref.shape

        # if the images aren't the same size, ensure they are both put into
        # a shape which fits both
        if (imgref.shape != imgtar.shape):
            x, y, c  = np.max(np.array([imgref.shape, imgtar.shape]), axis = 0)
            plate = np.zeros([x, y, c]).astype(np.uint8)

            xr, yr, c = imgref.shape
            xt, yt, c = imgtar.shape

            plateref = plate.copy(); plateref[:xr, :yr, :] = imgref; imgref = plateref
            platetar = plate.copy(); platetar[:xt, :yt, :] = imgtar; imgtar = platetar


        # combine the images
        imgCompare = np.hstack([imgref, imgtar])

        for m in matchedInfo:

            point0 = m.refP
            point1 = m.tarP + [y, 0]

            imgCompare = drawLine(imgCompare, point0, point1)

        return(imgCompare)
    else:
        return(imgref, imgtar)

def drawLine(img, point0, point1, blur = 2, colour = [0, 0, 255]):

    # draw a line between two points
    # Inputs:   (img), image to draw on
    #           (point#), the points to draw between, doesn't matter which 
    #               order they are specified
    #           (blur), the thickness of the line
    #           (colour), colour of the line
    # Outputs:  (img), image with the line drawn

    # get the distance between the points
    dist = np.ceil(np.sqrt(np.sum(abs(point1 - point0)**2)))

    # interpolate for the correct number of pixels between points
    xp = np.linspace(int(point0[1]), int(point1[1]), int(dist)).astype(int)
    yp = np.linspace(int(point0[0]), int(point1[0]), int(dist)).astype(int)


    # change the colour of these pixels which indicate the line
    for vx in range(-blur, blur, 1):
        for vy in range(-blur, blur, 1):
            img[xp+vx, yp+vy, :] = colour

    # NOTE this may do the interpolation between the points properly!
    # pos = np.linspace(point0.astype(int), point1.astype(int), int(dist)).astype(int)
    
    
    return(img)

def dictToDF(info, title, min = 3, feats = None):

    # create a pandas data frame from a dictionary of feature objects
    # Inputs:   (info), dictionary
    #           (title), list of the the column names
    #           (min), minimum number of times which a feature has to appear 
    #               for it to be used
    #           (feats), max number of features to include (if a feat has less
    #               # than the min then it will be excluded)
    #           (scl), scale to resize the points
    # Outputs:  (df), pandas data frame
    #     
    c = 0
    df = pd.DataFrame(columns=title)

    # get the name of the dict keys in order of the lengths
    if feats is None:
        keys = list(info.keys())
    else:
        keys = np.array(list(info.keys()))[np.argsort([-len(info[i]) for i in info])][:feats]

    for m in keys:
        if len(info[m]) < min:
            continue
        for nm, v in enumerate(info[m]):
            i = info[m][v]
            # only for the first iteration append the reference position
            if nm == 0:
                df.loc[c] = [i.refP[0], i.refP[1], int(v), int(m)]
                c += 1
                
            df.loc[c] = [i.tarP[0], i.tarP[1], int(v + 1), int(m)]
            c += 1

    return(df)

def tile(sz, x, y):

    '''
    gets the side length of an area proporptional to the image

    Inputs:   \n
    (sz), tile proporption of the image\n
    (x, y), img dims\n

    Outputs:  
    (s), the square length needed
    '''

    # if the size is 0 then don't create a tile
    if sz == 0:
        return(0)

    s = int(np.round(np.sqrt(x*y/sz)))   # border lenght of a tile to use

    return(s)

def moveImg(ref, tar, shift):

    '''
    Moves the target image by the shift amount and returns the error between the ref and target 
    image

    Inputs:     (ref), reference image (doesn't move)
                (tar), target image (moves by shift amount)
                (shift), pixel values to shift image by

    Outputs:    (refM, tarM), both images resized and the tar re-positioned 
    '''

    imgshape = ref.shape

    if len(ref.shape) == 3:
        x, y, c = imgshape
        xs, ys, c = shift.astype(int)
    elif len(ref.shape) == 2:
        x, y = imgshape
        xs, ys = shift.astype(int)

    # create the field which will contain both images. assumes they are the same size
    fieldN = np.zeros((ref.shape) + abs(shift.astype(int))).astype(np.uint8)
    field = np.zeros(ref.shape).astype(np.uint8)
    # shift the images
    xSc = int(np.clip(xs, 0, x))
    ySc = int(np.clip(ys, 0, y))
    xEc = int(np.clip(xs + x, 0, x))
    yEc = int(np.clip(ys + y, 0, y))

    fieldN[xSc:x+xSc, ySc:y+ySc] = tar
    tarM = fieldN[-x:, -y:]

    return(tarM)

def getSect(img, mpos, l, bw = True):

    # get the section of the image based off the feature object and the tile size
    # Inputs:   (img), numpy array of image
    #           (mpos), position
    #           (s), tile size 
    # Outputs:  (imgSect), black and white image section

    x, y, c = img.shape      

    # get target position from the previous match, use this as the 
    # position of a possible reference feature in the next image
    yp, xp = np.round(mpos).astype(int)
    xs = int(np.clip(xp-l, 0, x)); xe = int(np.clip(xp+l, 0, x))
    ys = int(np.clip(yp-l, 0, y)); ye = int(np.clip(yp+l, 0, y))
    sect = img[xs:xe, ys:ye]

    # NOTE turn into black and white to minimise the effect of colour
    if bw:
        sectImg = np.mean(sect, axis = 2).astype(np.uint8)
    else:
        sectImg = sect

    return(sectImg)

