import tifffile as tifi
import cv2 
from glob import glob
import multiprocessing
from multiprocessing import Pool
from itertools import repeat
import numpy as np
from tifffile.tifffile import imagej_description

if __name__ != 'HelperFunctions.SP_smallSample':
    from Utilities import nameFromPath, dirMaker
else:
    from HelperFunctions.Utilities import nameFromPath, dirMaker

def downsize(dataHome, size, scale = 0.3, cpuNo = False):

    # downsize the tif images, move them into the images directory and 
    # rename as appropriate

    tifFiles = dataHome + str(size) + "/tifFiles/"

    files = sorted(glob(tifFiles + "*.tif"))

    # create an error if there arent enough images
    if len(files) < 2:
        raise NameError("There are not enough image to perform any processing")

    targetDir = dataHome + str(size) + "/images/"
    dirMaker(targetDir)

    unrotated = len(glob(tifFiles + "_rotatedImgs*")) != 1
    if unrotated:
        print("     ROTATING")
    else:
        print("     NOT rotating")

    imgsRotated = []
    if cpuNo == 1:
        for f in files:
            imgsRotated.append(ds(f, scale, targetDir, unrotated))
    else:
        with Pool(processes=cpuNo) as pool:
            imgsRotated = pool.starmap(ds, zip(files, repeat(scale), repeat(targetDir), repeat(unrotated)))
            
    f = open(tifFiles + "_rotatedImgs.txt", "w")
    for i in imgsRotated:
        f.write(i + "\n")
    f.close()

def ds(f, scale, targetDir, unrotated = False):

    sampName = nameFromPath(f)
    tarPath = targetDir + sampName + ".png"

    name, id = sampName.split("_")

    # if samples with c are present, rotate
    if name.find("H710C") > -1 and id.find("C") > -1: 
        rotate = np.array([unrotated, True]).all()
    else:   rotate = False

    print(sampName + " being resized")
    try:

        try:
            imgF = tifi.imread(f)
            convert = True
        except:
            imgF = cv2.imread(f)
            convert = False

        # resize the image
        img = cv2.resize(imgF, (int(imgF.shape[1] * scale),  int(imgF.shape[0] * scale)))

        # convert the colour for proper saving 
        if convert: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # rotate if true
        if rotate:  
            print("     ROTATING " + sampName)
            img = cv2.rotate(img, cv2.ROTATE_180)
            imgF = cv2.rotate(imgF, cv2.ROTATE_180)
            tifi.imwrite(f, imgF)

        # save the image as a png
        cv2.imwrite(tarPath, img)

        return(sampName + "_" + str(rotate))
    except:
        print("     " + sampName + " FAILED")
        return(sampName + "_FAILED")

if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")

    dataSource = '/Volumes/USB/H673A_7.6/'
    dataSource = '/Volumes/USB/H671A_18.5/'

    dataSource = '/Volumes/USB/Test/'
    dataSource = '/Volumes/USB/H653A_11.3/'
    dataSource = '/Volumes/USB/H710C_6.1/'



    size = 3
    name = ''
    scale = 0.2
    cpuNo = 2

    downsize(dataSource, size, scale, cpuNo)