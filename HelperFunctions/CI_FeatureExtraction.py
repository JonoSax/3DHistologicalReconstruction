'''

this function will extract segSections from the aligned tissue to create a 
small 3D stack of a section of the entire tissue

'''

import tifffile as tifi
from glob import glob
import cv2
import multiprocessing
from multiprocessing import Process
if __name__ == "__main__":
    from Utilities import *
else:
    from HelperFunctions.Utilities import *
    
def featExtract(dataTrain, name, size):

    # This function propogates any identifies features through all the samples
    # and then extracts them

    # get the file of the features information 
    dirAligned = dataTrain + str(size) + '/alignedSamples/' + name
    dirSection = dataTrain + str(size) + '/sections/'
    dataAlignedTif = sorted(glob(dirAligned + '*.tif'))

    # NOTE this is pretty stupid but the origin for these co-ordinates is th
    # left (ie visual origin) instead of the actualy numpy origin (top left)

    segmentedSamples = dictOfDirs(
        segments = glob(dirAligned + "*.tif"), 
        segsection = glob(dirAligned + "*.segsection*"))
        
    # if there is a missing segesction from the segmented samples, thenn propogate it  
    # through the entire dictionary 
    segmentedSamples, fields = standardiseSections(segmentedSamples, dirSection)

    # parallelise the extraction of the images
    jobs = {}
    for s in segmentedSamples:
        jobs[s] = Process(target=segExtract, args = (segmentedSamples[s], dirSection, fields))
        jobs[s].start()

    for s in segmentedSamples:
        jobs[s].join()


def standardiseSections(segmentedSamples, dirSection):

    # This function ensures that the features are propogated for all the samples
    # Inputs:   (segmentedSamples), directory of directories which contains all the info
    #           (dirSections), directory to create teh 

    # identify all the sections types to find
    segSectAll = list()
    propogate = {}
    for spec in np.flip(list(segmentedSamples.keys())):
        for seg in segmentedSamples[spec]['segsection']:
            if type(segmentedSamples[spec]['segsection']) is str:   # ensure processing the whole path
                seg = segmentedSamples[spec]['segsection']
            sample = seg.split(".")[-1]
            propogate[sample] = seg
            segSectAll.append(sample)
    annotations = list(np.unique(np.array(segSectAll)))

    # copy the necessary directories to the ones which are missing
    for spec in segmentedSamples:
        annotationsC = annotations.copy()
        moveOn = False
        for seg in segmentedSamples[spec]['segsection']:
            if (type(segmentedSamples[spec]['segsection']) is str) & (moveOn == False):   # ensure processing the whole path
                seg = segmentedSamples[spec]['segsection']
                moveOn = True
            elif moveOn:
                break
            sample = seg.split(".")[-1]
            propogate[sample] = seg

            # remove 
            annotationsC.remove(seg.split(".")[-1])

        # for the missing samples, propogate the last known one
        for annoM in annotationsC:
            if moveOn:
                segmentedSamples[spec]['segsection'] = [segmentedSamples[spec]['segsection']]
                moveOn = False
            segmentedSamples[spec]['segsection'].append(propogate[annoM])

    # create
    segSections = {}
    for a in annotations:
        segSections[a] = {}
        segSections[a]['bl'] = []
        segSections[a]['tr'] = []

    for spec in segmentedSamples:
        for seg in segmentedSamples[spec]['segsection']:
            # doesn't matter if its tr, tl etc will be made absolute... as long as its a diagnoal this will work
            segSections[seg.split(".")[-1]]['bl'].append(txtToDict(seg)[0]['bl'])
            segSections[seg.split(".")[-1]]['tr'].append(txtToDict(seg)[0]['tr'])

    # create the uniform images sizes to copy each image to
    fields = {}
    for seg in segSections:
        dirMaker(dirSection + seg + '/')    # make the directories for the images
        xMax, yMax = np.array(np.max(abs(np.array(segSections[seg]['tr']) - np.array(segSections[seg]['bl'])), axis = 0))    
        fields[seg] = np.zeros([yMax + 1, xMax + 1, 3]).astype(np.uint8)

    return(segmentedSamples, fields)

def segExtract(segment, dirSection, fields):

    # this function takes a single specified samples and the segmented feature of interest
    # and extracts it from the image
    # Inputs:   (segment), dictionary of the specific sample
    #           (dirSection), the destiantion path to save the section
    #           (fields), the arrays of standard size to save the info in
    # Outputs:  (), extracts from the aligned image the section selected and named after 
    #           the specimen name (n) and section name (s)           

    # load the image
    img = tifi.imread(segment['segments'])
    
    # get the specimen name
    name = nameFromPath(segment['segments'])

    # for each segmentation annotation, extract and save it 
    for segDir in segment['segsection']:

        # get the sample name from the specific directory
        sample = segDir.split(".")[-1]

        # get the array for saving
        field = fields[sample].copy()

        # extract the section from the image
        pos = txtToDict(segDir)[0]
        y0, x1 = pos['bl']
        y1, x0 = pos['tr']
        imgSection = img[x0:x1, y0:y1, :]

        # save the section 
        x, y, c = imgSection.shape
        field[:x, :y, :c] = imgSection
        tifi.imwrite(dirSection + sample + '/' + name + "_" + sample + ".tif", field)

        print(name + " " + sample + " extracted")
    

if __name__ == "__main__":

    dataTrain = '/Volumes/Storage/H653A_11.3new/'
    name = ''
    size = 3

    extract(dataTrain, name, size)