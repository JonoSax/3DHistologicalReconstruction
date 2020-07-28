'''

this function will extract segSections from the aligned tissue

'''

from Utilities import *
import tifffile as tifi
from glob import glob
import cv2
import multiprocessing
from multiprocessing import Process

def extract(dataTrain, name, size):

    # This function reads in the .segSect files and aligned tif files
    # and extracts the feature

    # get the file of the features information 
    dirAligned = dataTrain + str(size) + '/alignedSamples/' + name
    dirSection = dataTrain + str(size) + '/sections/'
    dataAlignedTif = sorted(glob(dirAligned + '*.tif'))

    # collect all the seg sections and if there is more than one then get the largest
    # dimensions of all of them
    # NOTE this is pretty stupid but the origin for these co-ordinates is th
    # left (ie visual origin) instead of the actualy numpy origin (top left)
    segSectionsALL = {}
    segSectionsALL['seg0'] = (sorted(glob(dirAligned + name + '*.segsection')))
    segSectionsALL['seg1'] = (sorted(glob(dirAligned + name + '*.segsection1')))
    

    # this is really stupid but i'm processing this per sample
    for s in segSectionsALL:

        dirSection_s = dirSection + s + "/"
        dirMaker(dirSection_s)

        print('Processing ' + s)
        SegSections = segSectionsALL[s]
        dirs = dictOfDirs(
        tif = dataAlignedTif, 
        SegSections = SegSections)

        pos = list()
        info = txtToDict(SegSections)

        for n in info:
            pos.append(abs(info[n][0]['tr'] - info[n][0]['bl']))

        # get the maximum size of the selected section
        xMax, yMax = np.array(np.max(pos, axis = 0))    

        # ensure that there is a segSection availabe for all the images
        for n in dirs:
            try: 
                dirs[n]['SegSections']
                storeN = n      # store the last known instance where thre was a section annotated
            except:
                dirs[n]['SegSections'] = dirs[storeN]['SegSections']
        
        # parallelise jobs
        jobs = {}
        for n in dirs:
            # segExtract(n, dirs[n], dirSection, yMax, xMax, s)
            jobs[n] = (Process(target=segExtract, args=(n, dirs[n], dirSection_s, yMax, xMax, s)))     
            jobs[n].start()
        
        for n in dirs:
            jobs[n].join()

def segExtract(n, dirs, dirSection, yMax, xMax, s):

    # this function takes a single specified samples and the segmented feature of interest
    # and extracts it from the image
    # Inputs:   (n), specimen of interest
    #           (dirs), dictionary containing the directories of the specimen, NOTE because 
    #           the point of this is to propogate a selected feature through the entire sample, 
    #           if there is no specimen specicifc feature then it will use the last know selected feature
    #           (dirSection), the destiantion path to save the section
    #           (yMax, xMax), the max size of the features being drawn to standardise the section size
    #           (s), segment name
    # Outputs:  (), extracts from the aligned image the section selected and named after 
    #           the specimen name (n) and section name (s)           

        
    # create field to store all images in uniform shape
    field = np.zeros([yMax + 1, xMax + 1, 3]).astype(np.uint8)

    img = tifi.imread(dirs['tif'])

    # use the selected image size
    try:
        pos = txtToDict(dirs['SegSections'])[0]
        y0, x1 = pos['bl']
        y1, x0 = pos['tr']
    
    # if no image size is available then use the previously found position
    except:
        pass

    imgSection = img[x0:x1, y0:y1, :]

    # ensure all the saved images are the same size
    x, y, c = imgSection.shape
    field[:x, :y, :c] = imgSection
    tifi.imwrite(dirSection + n + "_" + s + ".tif", field)

    print("     " + n + " done")
        


dataTrain = '/Volumes/Storage/H653A_11.3new/'
name = ''
size = 3

extract(dataTrain, name, size)
