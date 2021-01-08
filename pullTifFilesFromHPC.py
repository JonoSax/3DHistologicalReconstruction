from glob import glob
from HelperFunctions.Utilities import nameFromPath, dirMaker
import os
import multiprocessing
from multiprocessing import Process
import time
import numpy as np

def getFiles(s, size, dataHome):
    # copy the tif files from HPC

    print("Starting " + s)
    dirMaker('/Volumes/USB/tifFiles' + s + '/')
    os.system('scp -r ' + dataHome + s + '/3/tifFiles/* /Volumes/USB/tifFiles' + s + '/')
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
    samples = ['H653A_11.3']#['H1029A_8.4', 'H653A_11.3', 'H671A_18.5', 'H671B_18.5', 'H673A_7.6', 'H710B_6.1', 'H710C_6.1', 'H750A_7.0' ]

    samples = [
    '/Volumes/Storage/H710C_6.1/',
    '/Volumes/USB/H671A_18.5/',
    '/Volumes/Storage/H653A_11.3/',
    '/Volumes/USB/H750A_7.0/',
    '/Volumes/USB/H710B_6.1/',
    '/Volumes/USB/H671B_18.5/']

    size = 3

    multiprocessing.set_start_method("fork")

    cpuNo = 6

    action = "push"

    # pull files from HPC
    if action == "pull":
        if cpuNo != 1:
            jobs = {}
            for s in samples:
                jobs[s] = Process(target = getFiles, args = (s, dataHome))
                jobs[s].start()

            for s in samples:
                jobs[s].join()

        else:
            for s in samples:
                getFiles(s, dataHome)

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