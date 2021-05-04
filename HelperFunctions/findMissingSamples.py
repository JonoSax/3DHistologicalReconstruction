import numpy as np
from glob import glob
import pandas as pd

if __name__ != "HelperFunctions.findMissingSamples":
    from Utilities import nameFromPath
else:
    from HelperFunctions.Utilities import nameFromPath



def findMissingSamples(dirHome, size):

    '''
    This takes the names of the samples and works out from the naming convention
    how many samples are missing. This is NOT full proof as it relies on the 
    robustness of the naming convention. 
    
    NOTE would recommend manually checking over the results.

        Inputs:\n
    (dirHome), specimen home directory\n
    (size), sample resolution\n

        Outputs:\n
    (), saves the output as a csv file in the info folder of that specimen
    '''

    names = sampleCategories(dirHome)
    if names is None:
        return

    home = dirHome + str(size)
    imgsrc = home + "/alignedSamples/"

    imgNames = nameFromPath(sorted(glob(imgsrc + "*.png")), 3)
    spec = nameFromPath(imgNames[0], 1) + "_"

    imgIDs = [i.split(spec)[-1] for i in imgNames]

    missingSamplesPath = home + "/info/missingSamples.csv"

    try:
        # check if there is already a missing sample file
        open(missingSamplesPath)
        print("Missing samples found already. Are you sure you want to overwrite?")
        i = input("Yes or No: ")
        # if you don't want to overwrite then skip
        if i.lower().find("n") > -1:
            return
    except:
        pass

    missing = []
    for r, t in zip(imgIDs[:-1], imgIDs[1:]):

        # get the info from the name
        for p in names:
            name = names[p]
            for n in name:
                # store the position in the names sequence and the sample number
                try:
                    if r.find(n) > -1: refInfo = [int(r.split(n)[0]), p]
                except:
                    pass
                try:
                    if t.find(n) > -1: tarInfo = [int(t.split(n)[0]), p]
                except:
                    pass

        # get name info
        interSamp = tarInfo[0] - refInfo[0]
        intraSamp = tarInfo[1] - refInfo[1]
        IDs = len(names)
        
        # calculate the number of missing samples and store
        sampsMissing = interSamp * IDs + intraSamp - 1
        missing.append(sampsMissing)

    print(str(np.sum(missing)) + " samples missing")
    # create a data frame of the information and save it as a csv file
    missingdf = pd.DataFrame(data = np.c_[imgIDs[:-1], imgIDs[1:], missing], columns = ["Ref", "Tar", "Missing"])
    missingdf.to_csv(missingSamplesPath)

def sampleCategories(dirHome):
    '''
    this contains the hard coded dictionaries indicating the sample numbers 
    The dictionaries are organised as follows:
    names = {0: ["A+B_0"], 1:["A+B_1"], 2:["C_0", "C_1"]} 
    The dictionary input is the string from the sample of what the index number SHOULD be
    In this case: 
        samples with that end with A+B_0 refer to the first sample
        samples that end with A+B_1 refer to the second sample
        sample may end with either C_0 or C_1 but they both refer to the third sample

    This process is NOT full proof as the naming convention was done very manually
    so it is RECOMMENDED you manually check afterwards

        Outputs:\n
    (names): the appropriate dictionary of the names

    '''

    # get the specimen name 
    spec = dirHome.split("/")[-2]

    if spec == "H653A_11.3":
        names = {0: ["_0"], 1: ["_1"]}
    elif spec == "H671A_18.5":
        names = {0: ["_0"], 1: ["_1"]}
    elif spec == "H671B_18.5":
        names = {0: ["A+B_0", "_0"], 1:["A+B_1", "_1"], 2:["C_0", "_2"]}   
    elif spec == "H673A_7.6":
        names = {0: ["_0"]}
    elif spec == "H710B_6.1":
        names = {0: ["A+B+C_0", "A+B_0"], 1: ["A+B+C_1", "A+B_1"], 2: ["A+B+C_2", "C+D_0"], 3: ["D_0", "C+D_1"]}
    elif spec == "H710C_6.1":
        names = {0: ["A+B_0"], 1:["A+B_1"], 2:["C_0", "C_1"]}       
    elif spec == "H753A":
        names = {0: ["_0"], 1: ["_1", "_2"]}                        
    elif spec == "H750A_7.0":
        names = {0: ["A+B_0"], 1:["A+B_1"], 2:["C_0", "C_1"]}   
    elif spec == "H1029A_8.4":
        names = {0: ["_0"]}
    else:
        # if there is no match then manually enter the names
        print("No matching specimen naming convention. Manually create the dictionary.", end = " ")
        print("If there are multiple entries seperate with a comma. When finished press enter twice")
        n = 0
        names = {}
        while True:
            label = input("Label for index " + str(n) + ": ")
            if label.lower() == "":
                if len(names) == 0:
                    names = None
                break
            names[n] = []
            for l in label.split(","):
                names[n].append(l.replace(" ", ""))
            n += 1

    return(names)

if __name__ == "__main__":


    dirHome = '/Volumes/Storage/H710C_6.1/'
    dirHome = '/Volumes/USB/H671A_18.5/'
    dirHome = '/Volumes/USB/H653A_11.3/'

    size = 3

    findMissingSamples(dirHome, size)

