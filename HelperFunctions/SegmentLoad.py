'''
This script loads in the NDPA files which have the manually segmented 
sections and correlates these positions with array position on the WSI 
that has just been loaded
'''

import numpy as np
from glob import glob

def readndpa(annotationsSRC, annotationName):

    # This function reads a NDPA file and extracts the co-ordinate of the hand drawn points
    # Input:    Directory for a single *.ndpa file
    # Output:   A list containing numpy arrays which have the recorded positions of the drawn points on the slide. Each
    #           entry to the list refers to a section drawn

    names = glob(annotationsSRC + annotationName + "*.ndpa")

    posAll = list()

    for name in names:

        file = open(name)

        # find the number of sections identified in slice
        sections = open(name).read().count("ndpviewstate id=")

        # extract all info from text file line by line
        doco = list(file.readlines())

        # declare list and array
        posA = list()
        pos = np.empty([0, 2])
        l = 0

        # NOTE the skips in this loop are hard coded because as far as i can tell
        # they are enitrely predictable 
        while l < len(doco):

                if "<point>" in doco[l]:

                    x = doco[l+1]
                    y = doco[l+2]

                    x = x.replace("<x>", ""); x = int(x.replace("</x>\n", ""))
                    y = y.replace("<y>", ""); y = int(y.replace("</y>\n", ""))

                    pos = np.vstack((pos, [x, y]))

                    l += 4  # jump 4 line to the next set of co-ordinates

                elif "</pointlist>" in doco[l]:
                    posA.append(pos.astype(int))
                    pos = np.empty([0, 2])

                    l+=18   # jump 18 lines to the next section of co-ordinates

                else:
                    l+=1    # if no info found, just iterate through

        print(str(len(posA)/sections*100)+"% of section " + name + " found")
        posAll.append(posA)

    return(posAll)

