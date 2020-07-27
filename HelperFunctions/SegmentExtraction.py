'''

this function will extract segSections from the aligned tissue

'''

from Utilities import *
import tifffile as tifi
from glob import glob
import cv2

def extract(dataTrain, name, size):

    # This function reads in the .segSect files and aligned tif files
    # and extract the feature

    # get the file of the features information 
    dirAligned = dataTrain + str(size) + '/alignedSamples/' + name
    dirSection = dataTrain + str(size) + '/sections/'
    dirMaker(dirSection)
    dataAlignedTif = sorted(glob(dirAligned + '*.tif'))

    # collect all the seg sections and if there is more than one then get the largest
    # dimensions of all of them
    # NOTE this is pretty stupid but the origin for these co-ordinates is th
    # left (ie visual origin) instead of the actualy numpy origin (top left)
    SegSections = sorted(glob(dirAligned + '*.segSect'))

    dirs = dictOfDirs(tif = dataAlignedTif, seg = SegSections)

    
    pos = list()
    info = txtToDict(SegSections)
    for n in info:
        pos.append(abs(info[n][0]['tr'] - info[n][0]['bl']))

    xMax, yMax = np.array(np.max(pos, axis = 0))    

    for n in dirs:
        
        # create field to store all images in uniform shape
        field = np.zeros([yMax + 1, xMax + 1, 3]).astype(np.uint8)

        img = tifi.imread(dirs[n]['tif'])

        # use the selected image size
        try:
            pos = txtToDict(dirs[n]['seg'])[0]
            y0, x1 = pos['bl']
            y1, x0 = pos['tr']
        
        # if no image size is available then use the previously found position
        except:
            pass

        imgSection = img[x0:x1, y0:y1, :]

        # ensure all the saved images are the same size
        x, y, c = imgSection.shape
        field[:x, :y, :c] = imgSection
        tifi.imwrite(dirSection + n + "_segment.tif", field)

        print(n + " done")
    


dataTrain = '/Volumes/Storage/H653A_11.3new/'
name = ''
size = 3

extract(dataTrain, name, size)
