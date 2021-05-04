'''

Little script which pulls files from HPC

'''

from glob import glob
from HelperFunctions.Utilities import nameFromPath, dirMaker
import os
import multiprocessing
from itertools import repeat
import time
import numpy as np

def getFiles(s, size, dataHome):
    # copy the tif files from HPC

    print("Starting " + s)
    path = '/Volumes/USB/' + s + str(size) + '/tifFiles/' 
    dirMaker(path)
    imgs = sorted(glob(dataHome + s + '/3/tifFiles/*.tif'))
    for i in imgs:
        print("     Copying " + nameFromPath(i))
        os.system('scp -r ' + i + ' ' + path)
    print("Finished " + s)

def pushFiles(s, size, dataHome):
    # send the featExtracted files to 

    name = s.split("/")[-2]
    
    print("Starting " + name)

    dataDest = dataHome + "SpecimenSections/" + name + "/"
    dirMaker(dataDest)

    feats = s + str(size) + "/FeatureSectionsFinal/linearSect/"

    os.system('nohup scp -r ' + feats + '* ' + dataDest + " &")
    while True:
        dirsMade = len(os.listdir(dataDest))
        dirsThere = len(os.listdir(feats))
        print(name + ": " + str(dirsMade) + "/" + str(dirsThere))
        time.sleep(20)

        if dirsMade == dirsThere:
            break

    print("Finished " + s)


if __name__ == "__main__":

    dataHome = '/Volumes/resabi201900003-uterine-vasculature-marsden135/BoydCollection/'
    # dataHome = 'jres129@hpc2.bioeng.auckland.ac.nz:/people/jres129/eresearch/uterine/jres129/BoydCollection/'
    samples = ['H1029A_8.4', 'H671A_18.5', 'H671B_18.5', 'H673A_7.6', 'H710B_6.1', 'H710C_6.1', 'H750A_7.0' ]

    '''
    samples = [
    '/Volumes/Storage/H710C_6.1/',
    '/Volumes/USB/H671A_18.5/',
    '/Volumes/Storage/H653A_11.3/',
    '/Volumes/USB/H750A_7.0/',
    '/Volumes/USB/H710B_6.1/',
    '/Volumes/USB/H671B_18.5/']
    '''

    size = 3

    multiprocessing.set_start_method('spawn')

    cpuNo = 4

    action = "pull"

    # pull files from HPC
    if action == "pull":
        if cpuNo != 1:

            with multiprocessing.Pool(processes = cpuNo) as Pool:
                Pool.starmap(getFiles, zip(samples, repeat(size), repeat(dataHome)))

        else:
            for s in samples:
                getFiles(s, size, dataHome)

        "PID = 47606"

    elif action == "push":

        # push files to HPC
        if cpuNo != 1:
            jobs = {}
            for s in samples:
                jobs[s] = Process(target = pushFiles, args = (s, size, dataHome))
                jobs[s].start()

            for s in samples:
                jobs[s].join()

        else:
            for s in samples:
                pushFiles(s, size, dataHome)