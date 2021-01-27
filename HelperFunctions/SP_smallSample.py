import tifffile as tifi
import cv2 
from glob import glob
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
if __name__ != "HelperFunctions.SP_AlignSamples":
    from HelperFunctions.Utilities import nameFromPath, dirMaker
else:
    from HelperFunctions.Utilities import nameFromPath, dirMaker

def downsize(dataHome, size, scale = 0.3, cpuNo = False):

    # downsize the tif images, move them into the images directory and 
    # rename as appropriate

    files = sorted(glob(dataHome + str(size) + "/tifFiles/*.tif"))

    # create an error if there arent enough images
    if len(files) < 2:
        raise NameError("There are not enough image to perform any processing")

    targetDir = dataHome + str(size) + "/images/"
    dirMaker(targetDir)

    if cpuNo is False:
        for f in files:
            ds(f, scale, targetDir)
    else:
        with Pool(processes=cpuNo) as pool:
            pool.starmap(ds, zip(files, repeat(scale), repeat(targetDir)))

def ds(f, scale, targetDir):

    sampName = nameFromPath(f)
    tarPath = targetDir + sampName + ".png"

    print(sampName + " being resized")
    try:
        # read in the image 
        img = cv2.cvtColor(tifi.imread(f), cv2.COLOR_BGR2RGB)

        # resize the image
        img = cv2.resize(img, (int(img.shape[1] * scale),  int(img.shape[0] * scale)))

        # save the image as a png
        cv2.imwrite(tarPath, img)
    except:
        print("     " + sampName + " FAILED")

if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")

    dataSource = '/Volumes/USB/H673A_7.6/'
    dataSource = '/Volumes/USB/H671A_18.5/'
    dataSource = '/Volumes/Storage/H653A_11.3/'

    dataSource = '/Volumes/USB/Test/'


    size = 3
    name = ''
    scale = 0.2
    cpuNo = 6

    downsize(dataSource, size, scale, cpuNo)