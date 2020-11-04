import numpy as np
from tensorflow_addons.image import sparse_image_warp
import cv2
from HelperFunctions.Utilities import matchMaker, nameFeatures, drawLine
from HelperFunctions.SP_SampleAnnotator import roiselector
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy import ndimage

# for each fitted pair, create an object storing their key information
class feature:
    def __init__(self, refP = None, tarP = None, dist = None, trainIdx = None, descr = None, ID = None, res = 0):
        # the position of the match on the reference image
        self.refP = refP

        # the position of the match on the target image
        self.tarP = tarP

        # eucledian error of the difference in gradient fields
        self.dist = dist

        # the index value of this feature from the sift search
        self.trainIdx = trainIdx

        # gradient descriptors
        self.descr = descr

        # the feature number 
        self.ID = ID

        # resolution of the image used
        self.res = res

def imgManualFeatSelect(imgPath):

    img = cv2.imread(imgPath)

    y, x, c = img.shape

    imgAnno = img.copy()
  
    ref = []
    tar = []

    n = 10
    points = []
    for xr in range(1, n):
        for yr in range(1, n):
            points.append(np.array([y/n*yr, x/n*xr]))

    i = 0
    # allow for manual annotations of the images
    # points = [[100, 100], [200, 200], [300, 300]]

    points.append([0, 0])
    while True:
        # point = np.median(roiselector(imgAnno, "Select the points"), axis = 1)
        point = np.array(points[i])
        orig = tuple(point.astype(int))

        # if no point is added then that's the manual end of point selection
        if np.sum(point) == 0:
            break

        
        # make the target point something randomally 10 points away from the reference
        shift = np.random.rand(2)*20-10
        tar.append(point + shift)
        ref.append(point)
        cv2.circle(imgAnno, orig, 7, [255, 0, 0], 5)
        cv2.circle(imgAnno, tuple(orig + shift.astype(int)), 5, [0, 0, 255], 5)
        id = str(i)

        '''
        if i%2 == 0:
            ref.append(point)
            cv2.circle(imgAnno, orig, 7, [255, 0, 0], 5)
            id = str(int(np.floor(i/2)))
        else:
            tar.append(point)
            cv2.circle(imgAnno, orig, 5, [0, 0, 255], 5)
            id = str(int(np.floor(i/2)))
        '''
        cv2.putText(imgAnno, id, orig, cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 4)
        cv2.putText(imgAnno, id, orig, cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 0], 2)

        i += 1

    ref = np.array(ref)
    tar = np.array(tar)

    if len(ref.shape) < 2:
        ref = np.expand_dims(ref, axis = 0)
        tar = np.expand_dims(tar, axis = 0)

    # don't modify the image (just annotate it)
    imgNoMod, flow = warper(img, ref, tar, anno = True, warp = False)

    # modify the image with a border 
    imgMod1, flow1 = warper(img, ref, tar, anno = True, order = 2)

    # modify the image with boundaries
    imgMod2, flow2 = warper(img, ref, tar, anno = True, order = 1)

    # modify the image with boundaries
    # imgMod3, flow3 = warper(img, ref, tar, anno = True, order = 5, smoother=1000)

    imgMods = np.hstack([imgNoMod, imgMod1, imgMod2])
    flowFields1 = np.hstack([flow[0], flow1[0], flow2[0]]).astype(np.uint8)
    flowFields2 = np.hstack([flow[1], flow1[1], flow2[1]]).astype(np.uint8)

    cv2.imshow("imgs", np.vstack([imgMods, flowFields1, flowFields2]))
    cv2.waitKey(0)  
    cv2.destroyAllWindows()

    imgMod1_2 = np.abs(imgMod1 - imgMod2).astype(np.uint8)
    # imgMod1_3 = np.abs(imgMod1 - imgMod3).astype(np.uint8)
    # imgMod2_3 = np.abs(imgMod2 - imgMod3).astype(np.uint8)

    # cv2.imshow("imgs", np.hstack([imgMod1_2, imgMod1_3, imgMod2_3])); cv2.waitKey(0)

    print("w")

def warper(img, ref, tar, anno = False, warp = True, border = 10, smoother = 0, order = 2): 

    # takes the ref and target numpy arrays and the image and processed them for the 
    # correct inputs into the sparse_image_warp
    # Inputs:   (img), numpy array of image
    #           (ref, tar), numpy array of points
    #           (anno), if true will add the ref and target points onto the image
    #           (warp), if true will warp based on the ref and tar positions
    # Outputs:  (imgMod), modified image
    #           (flow), flow field of the transformation

    # intialise the flow to save 
    Flow1 = np.zeros(img.shape)
    Flow2 = np.zeros(img.shape)

    # flip the column order of the features for the sparse matrix calculation
    if warp:
        reff = np.fliplr(np.unique(np.array(ref.copy()), axis = 0))
        tarf = np.fliplr(np.unique(np.array(tar.copy()), axis = 0))
        tfrefPoints = np.expand_dims(reff.astype(float), 0)
        tftarPoints = np.expand_dims(tarf.astype(float), 0)
        tftarImg = np.expand_dims(img.copy(), 0).astype(float)

        # perform non-rigid deformation on the original sized image
        imgMod, Flow = sparse_image_warp(tftarImg, tfrefPoints, tftarPoints, num_boundary_points=border, regularization_weight=smoother, interpolation_order=order)

        # convert the image and flow field into something useful
        imgMod = np.array(imgMod[0]).astype(np.uint8)
        Flow = np.array(Flow[0])

        Flow = ((Flow - Flow.min()) / (Flow.max() - Flow.min())*255)

        for i in range(3):
            Flow1[:, :, i] = Flow[:, :, 0]
            Flow2[:, :, i] = Flow[:, :, 1]

        '''
        # make the two layered field red and blue
        imgFlow[:, :, 0] = Flow[:, :, 0] 
        imgFlow[:, :, 2] = Flow[:, :, 1] 

        # get the positions of the contributions of the origianl 0 layer
        w = np.where(imgFlow == 0)

        # normalise the flow field
        imgFlow = ((imgFlow - imgFlow.min()) / (imgFlow.max() - imgFlow.min())*255)

        # remove the contribution of the original 0 layer
        imgFlow[w[0], w[1], w[2]] = 0

        imgFlow = imgFlow.astype(np.uint8)
        '''
        
    else:
        imgMod = img.copy()
        '''
        imgFlow = imgFlow.astype(np.uint8)
        '''

    Flow1.astype(np.uint8)
    Flow2.astype(np.uint8)

    if anno:
        # add the positions of the features on the original image
        for r, t in zip(ref, tar):
            cv2.circle(imgMod, tuple(r.astype(int)), 7, [0, 0, 255], 5)
            cv2.circle(imgMod, tuple(t.astype(int)), 5, [255, 0, 0], 5)

    return(imgMod, [Flow1, Flow2])

def imgAutoFeatSelect(imgPath, opt):
    # ---- modifiying the images
    # load the image, convert to RGB for plt to read
    img = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)

    '''
    img = np.zeros([500, 500, 3]).astype(np.uint8)
    cv2.circle(img, (100, 100), 50, [255, 0, 0], 100)
    cv2.circle(img, (400, 100), 50, [0, 255, 0], 100)
    cv2.circle(img, (100, 400), 50, [0, 0, 255], 100)
    cv2.circle(img, (400, 400), 50, [255, 255, 0], 100)
    '''

    img0, img1 = imgModifier(img, opt)
    x, y, c = img0.shape

    # ---- identifying the locations of features
    sift = cv2.xfeatures2d.SIFT_create() 
    bf = cv2.BFMatcher()
    kp_ref, des_ref = sift.detectAndCompute(img0, None)
    kp_tar, des_tar = sift.detectAndCompute(img1, None)
    matches = bf.match(des_ref, des_tar)

    
    plt.imshow(cv2.drawKeypoints(img0, kp_ref, None)); plt.show()
    plt.imshow(cv2.drawKeypoints(img1, kp_tar, None)); plt.show()
    plt.imshow(cv2.drawMatches(img0, kp_ref, img1, kp_tar, matches, None)); plt.show()
    

    resInfo = []
    refInfo = []
    tarInfo = []
    for m in matches:
        featureInfo = feature()
        # store the feature information as it appears on the original sized image
        featureInfo.refP = np.array(kp_ref[m.queryIdx].pt) 
        featureInfo.tarP = np.array(kp_tar[m.trainIdx].pt) 
        featureInfo.dist = np.array(m.distance)
        featureInfo.trainIdx = m.trainIdx
        featureInfo.descr = np.array(des_tar[m.trainIdx])
        resInfo.append(featureInfo)


    # ---- identify spatially cohesive features
    matchedInfo = matchMaker(resInfo, dist=50, tol = 0.2)
    
    # create the bounds on the image edges
    bound = []
    bound.append(np.array([0, 0]))
    bound.append(np.array([0, x]))
    bound.append(np.array([y, 0]))
    bound.append(np.array([y, x]))
    for b in bound:
        refInfo.append(b)
        tarInfo.append(b)
    
    for m in matchedInfo:
        refInfo.append(m.refP)
        tarInfo.append(m.tarP)
    
    combfeats = nameFeatures(img0.copy(), img1.copy(), matchedInfo, combine=True)
    plt.imshow(combfeats); plt.show()

    # ---- NL fit the target image
    tfrefPoints = np.expand_dims(np.array(refInfo[:100]), 0)
    tftarPoints = np.expand_dims(np.array(tarInfo[:100]), 0)
    tfImg = np.expand_dims(img1, 0).astype(float)
    imgMod, imgFlow = sparse_image_warp(tfImg, tfrefPoints, tftarPoints)

    # ---- display the results
    imgMod = np.array(imgMod[0]).astype(np.uint8)
    plt.title("Target, Mod, UnMod")
    plt.imshow(np.hstack([img0, img1, imgMod])); plt.show()
    plt.title("Tar-Mod, Tar-Unmod, Mod-UnMod")
    plt.imshow(np.hstack([abs(np.mean(img0 - img1, axis = 2)), abs(np.mean(img0 - imgMod, axis = 2)), abs(np.mean(img1 - imgMod, axis = 2))]), cmap = 'gray'); plt.show()

    print("WAIT")

def imgModifier(img, opt):

    # function to modify the images based on the chosen flag
    # Inputs:   (img), numpy array of the image
    #           (opt), option to modify
    # Output:   (img0, img1), the modified reference and target image

    def makeGrid(img, w, n):

        # create a grid of points
        # Input:    (img), image adding grid to
        #           (r), the width of the square to make
        #           (n), number of points in the grid (has to be 
        #               a squared number)

        w = int(w/2)
        yr, xr, c = np.array(img.shape)/2
        xs = int(xr - w); xe = int(xr + w)
        ys = int(yr - w); ye = int(yr + w)

        xp = np.linspace(xs, xe, n)
        yp = np.linspace(ys, ye, n)

        pos = []
        for x in xp:
            for y in yp:
                pos.append(np.array([x, y]))

        pos.append(np.array([0, 0]))
        pos.append(np.array([0, int(yr*2)]))
        pos.append(np.array([int(xr*2), 0]))
        pos.append(np.array([int(xr*2), int(yr*2)]))
        
        pos = np.array(pos)

        return(pos)

    def makeCurve(img, w, cu, n):

        # create a grid of points
        # Input:    (img), image adding grid to
        #           (w), length of the line
        #           (cu), the curvature
        #           (n), number of points in the grid (has to be 
        #               a squared number)

        yr, xr, c = np.array(img.shape)/2
        xs = int(xr - w/2)
        ys = int(yr - w/2)

        pos = np.linspace(0, np.array([w, w**(1/cu)]), n)**[1, cu] + np.array([xs, ys])
        
        posA = []
        posA.append(np.array([0, 0]))
        posA.append(np.array([0, int(yr*2)]))
        posA.append(np.array([int(xr*2), 0]))
        posA.append(np.array([int(xr*2), int(yr*2)]))
        
        
        for p in posA:
            pos = np.insert(pos, 0, p, axis = 0)
        
        return(pos)

    img0 = img.copy()
    img1 = img.copy()

    x, y, c = img.shape

    if opt == 0:
        # pure translation
        # place the target image at the original scale in the opposite corner
        img0 = np.zeros([x + 100, y + 100, c]).astype(np.uint8)
        img1 = img0.copy()
        img0[:x, :y, :] = img
        img1[-x:, -y:, :] = img

    elif opt == 1:
        # scale transformation
        # rescale the target image and place in the centre
        imgrs = cv2.resize(img1, tuple([int(y*0.8), int(x*0.8)]))
        xr, yr, c = imgrs.shape
        xs = int((x-xr)/2); xe = xs + xr
        ys = int((y-yr)/2); ye = ys + yr
        img1 = np.zeros(img.shape).astype(np.uint8)
        img1[xs:xe, ys:ye, :] = imgrs
    
    elif opt == 2:
        # pure rotation
        # rotate the target image around the centre
        img1 = ndimage.rotate(img1, 10)
        xr, yr, c = np.array(img1.shape)/2
        xs = int(xr - x/2); xe = int(xr + x/2)
        ys = int(yr - y/2); ye = int(yr + y/2)
        img1 = img1[xs:xe, ys:ye, :]

    elif opt == 3:

        # distort the image with a curve
        refpos = makeCurve(img, 300, 1.3, 100)
        tarpos = makeCurve(img, 300, 0.7, 100)

        imganno = img0.copy()
        for r, t in zip(refpos, tarpos):
            cv2.putText(imganno, "ref", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, [255, 0, 0], thickness=10)
            cv2.circle(imganno, tuple(r.astype(int)), 10, [255, 0, 0], 4)
                        
            cv2.putText(imganno, "tar", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, [0, 0, 255], thickness=10)
            cv2.circle(imganno, tuple(t.astype(int)), 12, [0, 0, 255], 6)

        tfrefPoints = np.expand_dims(refpos, 0)
        tftarPoints = np.expand_dims(tarpos, 0)
        tfImg = np.expand_dims(img0, 0).astype(float)

        imgMod, imgFlow = sparse_image_warp(tfImg, tfrefPoints, tftarPoints)
        img1 = np.array(imgMod[0]).astype(np.uint8)

        plt.imshow(np.hstack([imganno, img1, img0-img1])); plt.show()

    elif opt == 4:
        
        # distort the image by shrinking the centre of the image
        refpos = makeGrid(img, 600, 10)
        tarpos = makeGrid(img, 800, 10)

        imganno = img0.copy()
        for r, t in zip(refpos, tarpos):
            cv2.putText(imganno, "ref", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, [255, 0, 0], thickness=10)
            cv2.circle(imganno, tuple(r.astype(int)), 10, [255, 0, 0], 4)
            
            cv2.putText(imganno, "tar", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, [0, 0, 255], thickness=10)
            cv2.circle(imganno, tuple(t.astype(int)), 12, [0, 0, 255], 6)
            

        tfrefPoints = np.expand_dims(refpos, 0)
        tftarPoints = np.expand_dims(tarpos, 0)
        tfImg = np.expand_dims(img0, 0).astype(float)

        imgMod, imgFlow = sparse_image_warp(tfImg, tfrefPoints, tftarPoints)

        img1 = np.array(imgMod[0]).astype(np.uint8)

        plt.imshow(np.hstack([imganno, img1, img0-img1])); plt.show()
    

    return(img0, img1)


if __name__ == "__main__":

    imgPath = "checkerboard.png"
    imgPath = "dog.png"
    opt = 3

    # imgAutoFeatSelect(imgPath, opt)
    imgManualFeatSelect(imgPath)