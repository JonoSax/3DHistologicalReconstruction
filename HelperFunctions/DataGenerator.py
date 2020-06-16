'''
This script is extracting the manually segmented sections of the image 
from the WSI and the other information and categorising them into classes
appropriate for the ModelTrainer
'''

from glob import glob
from random import random
import os
from shutil import copy, rmtree

def main(dataSRC, portion, *args):
    # splits up the data acquired into testing and training as per tf requirements
    # Inputs:   (dataSRC), location of data
    #           (portion), the fraction of data to be used in testing
    #           (*args), the classes to be used, ATM not really utilised because this is a one class problem but could be enhanced
    # Outputs:  (), copies the associated files of the segmentations into the testing/training folder

    # NOTE: if this script is run more than once, because of the random number usage, will almost inevitbly cause images to appear in 
    # test and train folders...... ENSURE that the folder is cleared before running this script

    dataSource = dataSRC + 'segmentedTissue/'
    dataTF = dataSRC + 'segmentedTissueSorted/'
    trainDir = dataTF + 'train/'
    testDir = dataTF + 'test/'

    # create the directories which store the testing and training data
    cond0 = False
    cond1 = False
    while (cond0 == False) & (cond1 == False):
        try:
            os.mkdir(dataTF) 
            print("Created " + str(dataTF))
        except:
            try:
                os.mkdir(trainDir)
                os.mkdir(testDir)
                print("     Created " + str(trainDir.replace(dataTF, "")))
                print("     Created " + str(testDir.replace(dataTF, "")))
            except:
                for a in args:
                    try:
                        # ensure that the directory is empty by removing it then re-creating it
                        os.mkdir(trainDir + a)
                        print("         Created " + str(a))
                    except:
                        cond0 = True
                    try:
                        os.mkdir(testDir + a)
                        print("         Created " + str(a))
                    except:
                        cond1 = True

    # search for all the data segments
    imgs = glob(dataSource + "*.tif")

    for img in imgs:

        # find which class the image is 
        for arg in args:
            if img.find(arg) > 0:
                break

        
        # get a random number
        r = random()

        # if the random number is less than the portion, move that particular file into the testing folder
        if r < portion:
            copy(img, testDir + arg)
            
        # if the random number if more than the portion, move that particular file into the training folder
        else:
            copy(img, trainDir + arg)
