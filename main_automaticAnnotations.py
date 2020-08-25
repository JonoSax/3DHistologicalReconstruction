'''

This script performs normal tissue extraction but is specifically designed to 
align tissues which have no annotations

Processing includes:
    - Extracting the sample from the ndpi file at the given zoom
    - Creating features for each tissue sample 
    - Aligning the tissue
    - Extracting any particular features as defined by segSection (or )

'''

from HelperFunctions import *

# dataHome is where all the directories created for information are stored 
dataHome = '/Volumes/USB/H1029a/'
dataHome = '/Volumes/USB/H653A_11.3/'
dataHome = '/Volumes/USB/H673A_7.6/'
dataHome = '/Volumes/USB/H710C_6.1/'
dataHome = '/Volumes/USB/H710B_6.1/'

# research drive access via VPN
# dataHome = '/Volumes/resabi201900003-uterine-vasculature-marsden135/Boyd collection/H1029A_8.4/'


size = 3
kernel = 50
name = ''
portion = 0.2

print("\n----------- WSILoad -----------")
# extract the tif file of the specified size
# WSILoad(dataHome, name, size)

print("\n----------- smallerTif -----------")
# create jpeg images of all the tifs and a single collated pdf
# smallerTif(dataHome, name, size, 0.2)

print("\n----------- specID -----------")
# identifies the sample within the slide from the jpegs created
specID(dataHome, name, size)

print("\n----------- featFind -----------")
# identifies corresponding features per samples 
featFind(dataHome, name, size)

print("\n----------- AignSegments -----------") # --> NOTE add size of all the shapes and make these functions adjust the size of the files for this image size
# align all the specimens 
align(dataHome, name, size, True)

fd
print("\n----------- FixPoint -----------")
for imgref, imgtar in zip(imgrefs, imgtars):
    pass

# if there are any images which weren't annotated correct change their features manually
featChangePoint(imgref, imgtar, matchRef, matchTar, ts = 4)

print("\n----------- FeatureExtraction -----------")
# propogate an annotated feature through the aligned tissue and extract
# featSelectArea(dataHome, size, 3)

# NOTE run the results of the feat extraction THROUGH the align function again to 
# provide the fine scale alignment for the tissue segment
'''

How this will work is for the first slice where the feature is chosen, in the next slice
a slightly larger area is selected to perform feature mapping from the original tissue. 

    NOTE I think it would be ideal for the section selected if you can go and essentailly 
    outline EXACTLY what the issue is, rather than just using a block of tissue, so that 
    subsequent feature matching is on the actual target tissue rather than just noise as well

Once this new image is found the amount it has shift from the ORIGINAL image is used to expaned
the search area and this new feature is identifed. 

In the next slice the NEW feature is used to identify the NEXT translation of the slice to find 
the next expanded search area. 

This will involve performing a sift operation on each area and performing a translation (probably 
not rotation.... ) of the tissue and using this translation to expand the search area AND align 
the current tissue sample

THIS ASSUMES THAT THE Z-SHIFT IS REASONABLY CONSISTENT BETWEEN SLICES

'''