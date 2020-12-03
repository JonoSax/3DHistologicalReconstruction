import numpy as np
from glob import glob
from HelperFunctions.Utilities import nameFromPath
import pandas as pd

def find(dirHome, size, names):

    '''
    This takes the names of the samples and works out from the naming convention
    how many samples are missing. This is NOT full proof as it relies on the 
    robustness of the naming convention. 
    
    NOTE would recommend manually checking over the results.

        Inputs:\n
    (dirHome), specimen home directory\n
    (size), sample resolution\n
    (names), the naming convention to indicate a different sample. NOTE the format is
    in a dictionary where each name entry is added in as a list. If there are multiple
    different naming conventions for the same order then add them in sequentially. 
    eg: \n
    names = {0: ["A+B_0"], 1:["A+B_1"], 2:["C_0", "C_1"]}   \n
    There are 4 ways which the samples have been name which refer to 3 individual 
    samples (ie the 3rd image of a sample has 2 possible ways to be named)

        Outputs:\n
    (), saves the output as a csv file in the info folder of that specimen
    '''

    home = dirHome + str(size)
    imgsrc = home + "/alignedSamples/"

    imgNames = nameFromPath(sorted(glob(imgsrc + "*.png")), 3)
    spec = nameFromPath(imgNames[0], 1) + "_"

    imgIDs = [i.split(spec)[-1] for i in imgNames]

    missingSamplesPath = home + "/info/missingSamples.csv"

    missing = []
    for r, t in zip(imgIDs[:-1], imgIDs[1:]):

        # get the info from the name
        for p in names:
            name = names[p]
            for n in name:
                # store the position in the names sequence and the sample number
                if r.find(n) > -1: refInfo = [int(r.split(n)[0]), p]
                if t.find(n) > -1: tarInfo = [int(t.split(n)[0]), p]

        # get name info
        interSamp = tarInfo[0] - refInfo[0]
        intraSamp = tarInfo[1] - refInfo[1]
        IDs = len(names)

        # calculate the number of missing samples
        missing.append(interSamp * IDs + intraSamp - 1)

    # create a data frame of the information and save it as a csv file
    missingdf = pd.DataFrame(data = np.c_[imgIDs[:-1], imgIDs[1:], missing], columns = ["Ref", "Tar", "Missing"])
    missingdf.to_csv(missingSamplesPath)

if __name__ == "__main__":


    dirHome = '/Volumes/Storage/H710C_6.1/'
    dirHome = '/Volumes/Storage/H653A_11.3/'

    size = 3

    names = {0: ["A+B_0"], 1:["A+B_1"], 2:["C_0", "C_1"]}       # for H710C
    names = {0: ["_0"], 1: ["_1", "_2"]}                        # for H753A

    find(dirHome, size, names)

