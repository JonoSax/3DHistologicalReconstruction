'''

Apply the masks over the images

'''


from HelperFunctions.Utilities import nameFromPath
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

src = '/Volumes/USB/H653A_11.3/3/'

imgsrc = src + "SegmentationTest/"
segsrc = src + "Segmentations/"

imgDest = src + "SegmentationEvals/"

imgs = sorted(glob(imgsrc + "*.png"))
masks = sorted(glob(segsrc + "*.png"))

maskToUse=[]
for i in imgs:
    try:
        m = nameFromPath(masks).index(nameFromPath(i))
        maskToUse.append(masks[m])
    except:
        maskToUse.append(None)



for n, (m, i) in enumerate(zip(maskToUse, imgs)):

    if m is None:
        continue

    name = nameFromPath(i, 3)

    print(name + " Processing")

    img = cv2.imread(i)
    mask = cv2.imread(m)

    xi, yi, _ = img.shape
    xm, ym, _ = mask.shape

    xr, yr = tuple(np.round((xi / xm) * np.array([xm, ym])).astype(int))
    imgStored = []
    for ni, i in enumerate(np.unique(mask)[1:]):
        
        # get the mask of only the particular tissue type
        maskFull = cv2.resize(((mask==i)*1).astype(np.uint8), tuple([yr, xr]))
        xf, yf, _ = maskFull.shape

        maskCover = np.zeros([xi, yi, 3]); maskCover[:xr, :yr, :] = maskFull

        imgMasked = (img * maskCover).astype(np.uint8)

        imgStored.append(imgMasked)

        cv2.imwrite(imgDest + str(name) + "_" + str(ni) + ".png", imgMasked)# np.hstack(imgStored))