import math
from scipy.spatial.transform import Rotation
from scipy import ndimage
import numpy as np
import cv2
from matplotlib import pyplot as plt
import MRCode


# 1205: fix order of height and width
class MRCodeSim:
    # codeRes = (height, width)
    # codeSize: scalars
    # camRes = (height, width)
    def __init__(self, codeRes, codeSize, camRes, camFov, camPrincipal=None):
        self.codeRes = codeRes
        self.camRes = camRes
        self.camFov = camFov
        self.codeSize = codeSize
        self.camPrincipal = camPrincipal
        self.K = calcIntrinsics(camRes, camFov, camPrincipal)
        self.setNoiseParam()
        self.setDistance()
        self.pertSigma = 0
        self.airySigma = 0.5 # Gaussian approximation of Airy disk
        self.defocusSigma = 0
        self.linearBlurAngle = 0
        self.linearBlurLength = 0
        self.gamma = 1


    def setNoiseParam(self, numPhotons=1000, bbir=0.2, readNoise=1, fwc=10000, bitDepth=8, pixScale=0.2):
        self.numPhotons = numPhotons
        self.bbir = bbir
        self.readNoise = readNoise
        self.fwc = fwc
        self.bitDepth = bitDepth
        self.pixScale = pixScale


    def setDistance(self, distance=3, transDistRatio=0.1, rng=None):
        self.R = Rotation.identity()
        if rng is None:
            self.t = np.array([(np.random.rand()-0.5)*2*transDistRatio,
                               (np.random.rand()-0.5)*2*transDistRatio,1]) * distance
        else:
            self.t = np.array([(rng.random()-0.5)*2*transDistRatio,
                               (rng.random()-0.5)*2*transDistRatio,1]) * distance
    
    
    # distances: numpy array of distances
    def calcNumPixels(self, codeShape, distances):
        numPixels = self.codeSize/math.tan(math.radians(self.camFov/2))*self.camRes[0]/2/codeShape[0]\
                /distances
        return numPixels


    # code: 0...255
    def genImages(self, code, rng=None):
        upsampleFactor = 10
        # codeImg: a higher resolution verison of code, only for visualization
        codeImg = cv2.resize(code.astype(float), self.codeRes[::-1], interpolation=cv2.INTER_NEAREST)
        # compute homography H from simulator settings
        H = calcHomography(code.shape, self.codeSize, self.K, self.R, self.t)
        assert H[2,2] != 0
        H = H / H[2,2]
        # compute the corner correspondences between code space and image space
        srcCorners = np.array([[-0.5, -0.5], [-0.5, code.shape[0]-0.5], [code.shape[1]-0.5, -0.5],\
                [code.shape[1]-0.5, code.shape[0]-0.5]])
        srcCorners = np.transpose(srcCorners) # x: first row, y: second row
        dstCorners = H @ np.vstack((srcCorners, np.ones((1,4))))
        dstCorners = dstCorners[:2,:] / dstCorners[2:3,:]
        # compute a bounding box in the image space in which the code is placed
        # only super-sample (10x) within the bounding box
        blurSigma = self.airySigma + self.defocusSigma
        blurMargin = blurSigma * 3
        if self.linearBlurLength > 0:
            linearBlurKernel = createLinearBlurKernel(self.linearBlurAngle, 
                                                      self.linearBlurLength*upsampleFactor)
            blurMargin += linearBlurKernel.shape[0] / 2
        umin, vmin = np.floor(np.amin(dstCorners, axis=1)-blurMargin).astype(int)
        umax, vmax = np.ceil(np.amax(dstCorners, axis=1)+blurMargin).astype(int)
        u = np.arange(umin*upsampleFactor, umax*upsampleFactor+1, dtype=float) / upsampleFactor
        v = np.arange(vmin*upsampleFactor, vmax*upsampleFactor+1, dtype=float) / upsampleFactor
        uv, vv = np.meshgrid(u, v)
        dstPoints = np.vstack((uv.flatten(), vv.flatten(), np.ones((1,uv.size))))
        srcPoints = np.linalg.inv(H) @ dstPoints
        srcPoints = srcPoints[:2,:] / srcPoints[2:3,:]
        # warp the code to the super-sampled bounding box
        # TODO: adaptive to code resolution
        # currently it should work fine for binary codes, as we are using nearest interp
        codeImgInterp = code.astype(float)*(1-self.bbir)+self.bbir*255
        dstGrid = ndimage.map_coordinates(codeImgInterp, np.flipud(srcPoints), order=0, \
                mode='nearest')
        dstGrid[srcPoints[0,:]<-0.5] = 0
        dstGrid[srcPoints[0,:]>code.shape[1]-0.5] = 0
        dstGrid[srcPoints[1,:]<-0.5] = 0
        dstGrid[srcPoints[1,:]>code.shape[0]-0.5] = 0
        dstGrid = np.reshape(dstGrid, uv.shape)
        # simulate Airy disk and defocus blur by a Gaussian blur
        if blurSigma > 0:
            dstGrid = cv2.GaussianBlur(dstGrid, (0,0), blurSigma*upsampleFactor)
        # simulate motion blur
        if self.linearBlurLength > 0:
            dstGrid = cv2.filter2D(dstGrid, -1, linearBlurKernel)
        # paste the bounding box to image space
        # first apply a box filter and then downsample to original image resolution
        kernel = np.ones((upsampleFactor,upsampleFactor)) / upsampleFactor**2
        dstGridFiltered = cv2.filter2D(dstGrid, -1, kernel)
        dstGridFiltered = dstGridFiltered[::upsampleFactor,::upsampleFactor]
        camImg = np.zeros(self.camRes)
        # check boundary
        vDstStart = 0-min(0,vmin)
        vDstEnd = dstGridFiltered.shape[0]+min(0, camImg.shape[0]-1-vmax)
        uDstStart = 0-min(0,umin)
        uDstEnd = dstGridFiltered.shape[1]+min(0, camImg.shape[1]-1-umax)
        #camImg[vmin:vmax+1,umin:umax+1] = dstGridFiltered[::upsampleFactor,::upsampleFactor]
        camImg[max(0,vmin):min(camImg.shape[0],vmax+1),
               max(0,umin):min(camImg.shape[1],umax+1)] = dstGridFiltered[vDstStart:vDstEnd,
                                                                          uDstStart:uDstEnd]
        camImg = camImg*(1-self.bbir)+self.bbir*255
        # add camera noise
        camImg = addCameraNoise(camImg, self.numPhotons, self.readNoise, self.fwc, self.bitDepth, clipNegative=True, pixScale=self.pixScale, rng=rng)
        # add gamma correction
        camImg = addGamma(camImg, self.gamma)
        # perturbate homography
        if self.pertSigma > 0:
            Hp = perturbHomography(H, code.shape, self.pertSigma, rng)
        else:
            Hp = H
        return (codeImg, camImg, Hp)
        


    # code: 0...255
    def genImages2(self, code, rng=None):
        codeImg = cv2.resize(code.astype(float), self.codeRes[::-1], interpolation=cv2.INTER_NEAREST)
        H = calcHomography(self.codeRes, self.codeSize, self.K, self.R, self.t)
        Hb = calcHomography(code.shape, self.codeSize, self.K, self.R, self.t)
        if self.pertSigma > 0:
            Hp = perturbHomography(Hb, code.shape, self.pertSigma, rng)
        else:
            Hp = Hb/Hb[2,2]
        Htemp = calcHomography((code.shape[0]/2,code.shape[1]/2), self.codeSize, self.K, self.R, self.t)
        # Hp = perturbHomography(H, self.codeRes, self.pertSigma, rng)
        camImg = warpCode(codeImg*(1-self.bbir)+self.bbir*255, self.camRes, H)
        camImg = addCameraNoise(camImg, self.numPhotons, self.readNoise, self.fwc, self.bitDepth, pixScale=self.pixScale, rng=rng)
        # camImgAligned = warpCode(camImg, code.shape, Hp, True, cv2.INTER_AREA)
        # camImgAligned = warpCode(camImg, self.codeRes, Hp, True)
        return (codeImg, camImg, Hp)


# Calculate the intrinsic matrix from resolution, horizontal FOV and principal point (in px)
# Assume square pixels, pixel starts from 0, i-th pixel expands from i-0.5 to i+0.5
# res: [height, width]
# hfov: in degrees
# principalPoint: [cx, cy]
def calcIntrinsics(res, hfov, principalPoint=None):
    if principalPoint is None:
        principalPoint = ((0+res[1]-1)/2, (0+res[0]-1)/2)
    return np.array([[res[1]*0.5/math.tan(math.radians(hfov/2)), 0, principalPoint[0]],
        [0, res[1]*0.5/math.tan(math.radians(hfov/2)), principalPoint[1]],
        [0, 0, 1]])


# Calculate the homography used to warp the code image to camera frame
# codeShape: shape of the code image (height, width)
# codeHLen: scalar, length of the horizontal side of the code (in meters)
# K: 3x3 intrinsic matrix
# R: scipy.spatial.transform.Rotation object
# t: 3-vector
def calcHomography(codeShape, codeHLen, K, R, t):
    # S: matrix convert codeImg pixel coordinates to world coordinates (centered at (0,0,0))
    S = np.array([[codeHLen/codeShape[1], 0, -(codeShape[1]/2-0.5)*codeHLen/codeShape[1]],
        [0, codeHLen/codeShape[1], -(codeShape[0]/2-0.5)*codeHLen/codeShape[1]],
        [0, 0, 0],
        [0, 0, 1]])
    return K @ np.hstack((R.as_matrix(), t[:,None])) @ S

# Perturb homography by perturb four corners by sub-pixel shifts
# H: homography to perturb
# codeShape: shape of the code image (height, width)
# sigma: sigma of the gaussian noise to perturb the four corners
def perturbHomography(H, codeShape, sigma, rng=None):
    srcCorners = np.array([[0, 0], [0, codeShape[0]-1], [codeShape[1]-1, 0],
            [codeShape[1]-1, codeShape[0]-1]])
    dstCorners = np.hstack((srcCorners, np.ones((4,1)))) @ np.transpose(H)
    dstCorners = dstCorners[:,:2] / dstCorners[:,2:3]
    if rng is None:
        dstCorners = dstCorners + np.random.normal(0, sigma, dstCorners.shape)
    else:
        dstCorners = dstCorners + rng.normal(0, sigma, dstCorners.shape)
    Hp, mask = cv2.findHomography(srcCorners, dstCorners)
    return Hp


# Perturb homography by perturb four corners by sub-pixel shifts
# Specify the corner location within the code image, 
# which is helpful for code image with white borders
# H: homography to perturb
# codeShape: shape of the code image (height, width)
# codeCorners: corners in code image coordinates, 4x2 array
# sigma: sigma of the gaussian noise to perturb the four corners
def perturbHomographyFromCorners(H, codeShape, codeCorners, sigma, rng=None):
    srcCorners = np.array([[0, 0], [0, codeShape[0]-1], [codeShape[1]-1, 0],
            [codeShape[1]-1, codeShape[0]-1]])
    dstCorners = np.hstack((codeCorners, np.ones((4,1)))) @ np.transpose(H)
    dstCorners = dstCorners[:,:2] / dstCorners[:,2:3]
    if rng is None:
        dstCorners = dstCorners + np.random.normal(0, sigma, dstCorners.shape)
    else:
        dstCorners = dstCorners + rng.normal(0, sigma, dstCorners.shape)
    Hp, mask = cv2.findHomography(srcCorners, dstCorners)
    return Hp
    
    
# Warp a "printed" code as if it is captured by a camera
# codeImg: code image at assumed resolution
# H: 3x3 homography matrix
def warpCode(codeImg, outSize, H, inv=False, flags=None):
    if flags is None:
        if inv:
            flags = cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP
        else:
            flags = cv2.INTER_AREA
    else:
        if inv:
            flags = flags | cv2.WARP_INVERSE_MAP
    return cv2.warpPerspective(codeImg, H, (outSize[1], outSize[0]), flags=flags)


# Add simulated camera noise to an image
# im: input image, np.array of floats, in [0...255]
# maxPhotons: max image intensity (255) is mapped to maxPhotons (assuming QE=1)
# readNoise: std of read noise in electrons
# fullWell: full well capacity in electrons
# bitDepth: bit depth of output image, to simulate quantization error, if 0 then return floats (0 to 1)
# clipNegative: whether or not to clip negative values
# pixScale: scale the intensity, if none then scale max of all pixels to 1, if 0 then scale fullWell to 1
def addCameraNoise(im, maxPhotons, readNoise, fullWell, bitDepth=8, clipNegative=False, pixScale=1, rng=None):
    ims = im / 255 * maxPhotons
    if rng is None:
        imn = np.random.poisson(ims) + np.random.normal(0, readNoise, ims.shape)
    else:
        imn = rng.poisson(ims) + rng.normal(0, readNoise, ims.shape)
    np.clip(imn, 0 if clipNegative else None, fullWell, out=imn)
    if pixScale is None:
        imn = (imn-np.amin(imn)) / (np.amax(imn)-np.amin(imn))
        if bitDepth > 0:
            imn = np.around(imn * (2**bitDepth-1))
    elif pixScale == 0:
        imn = imn / fullWell
    else:
        imn = imn * pixScale
    return imn


# Add gamma correction
# im: input image, np.array of floats, in [0...255]
# gamma: gamma exponent
def addGamma(im, gamma):
    return (im / 255) ** gamma * 255


# Compute motion blur kernel
# reference: https://docs.opencv.org/master/d1/dfd/tutorial_motion_deblur_filter.html
def createLinearBlurKernel(angle, length):
    point = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]) * length / 2
    bound = np.amax(np.ceil(np.abs(point)))
    size = np.array([bound, bound]).astype(int) * 2 + 1
    h = np.zeros(size)
    h = cv2.ellipse(h, (int(bound), int(bound)),
                    (0, int(np.round(length/2))), angle, 0, 360, 255)
    h = h / np.sum(h)
    return h


# Compute motion blur kernel
# V1: use cv2.line, not used
def createLinearBlurKernelV1(angle, length):
    point = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]) * length / 2
    bound = np.amax(np.ceil(np.abs(point)))
    size = np.array([bound, bound]).astype(int) * 2 + 1

    ssize = size * 100 + 1
    img = np.zeros(ssize, np.uint8)
    shift = (ssize.astype(float) - 1) / 2
    scale = ssize[0] / size[0]
    startPoint = np.round(-point * scale + shift)
    startPoint = tuple(startPoint.astype(int))
    endPoint = np.round(point * scale + shift)
    endPoint = tuple(endPoint.astype(int))
    img2 = cv2.line(img, startPoint, endPoint, 255, 1,
                    cv2.LINE_AA)
    img3 = img2.astype(float) / 255
    img3 = cv2.resize(img3, tuple(size), interpolation=cv2.INTER_AREA)
    img3 = img3 / np.sum(img3)

#    cv2.imshow('test', img2)
#    cv2.waitKey(0)
#    cv2.imshow('test', cv2.resize(img3, (500,500), interpolation=cv2.INTER_NEAREST))
#    cv2.waitKey(0)

    return img3


# Normalize image for visualization
def normVis(im):
    return (im - np.amin(im)) / (np.amax(im) - np.amin(im)) * 255


# test code:
if __name__ == "__main__":
    print("==================== Test calcIntrinsics ====================")
    K1 = calcIntrinsics([1000,1000], 90)
    print(K1)
    print(K1 @ np.array([1,1,1]))
    K2 = calcIntrinsics([480, 480], 60, [240, 239])
    print(K2)
    print(K2 @ np.array([math.sqrt(3)/3,math.sqrt(3)/3,1]))
    
    print("==================== Test calcHomography  ====================")
    codeShape1 = (1000,1000)
    codeHLen1 = 2
    R1 = Rotation.identity()
    t1 = np.array((0,0,1))
    H1 = calcHomography(codeShape1, codeHLen1, K1, R1, t1)
    print(H1)

    print("==================== Test warpCode ====================")
    codeImg = cv2.imread("code5.png", cv2.IMREAD_COLOR)
    codeImg = codeImg.astype(float) * 0.8 + 255 * 0.2
    codeShape2 = codeImg.shape
    H2 = calcHomography(codeShape2, codeHLen1, K1, R1, t1)
    camImg2 = warpCode(codeImg, (1000,1000), H2)
    cv2.imwrite("code5-w2.png", camImg2)
    print("Test 1: front view, written to code5-w2.png")

    t3 = np.array((1,0,3))
    H3 = calcHomography(codeShape2, codeHLen1, K1, R1, t3)
    camImg3 = warpCode(codeImg, (1000,1000), H3)
    cv2.imwrite("code5-w3.png", camImg3)
    print("Test 2: translated, written to code5-w3.png")

    R4 = Rotation.from_rotvec(np.pi/3 * np.array([0,0,1]))
    H4 = calcHomography(codeShape2, codeHLen1, K1, R4, t3)
    camImg4 = warpCode(codeImg, (1000,1000), H4)
    cv2.imwrite("code5-w4.png", camImg4)
    print("Test 3: in-plane rotation, written to code5-w4.png")

    R5 = Rotation.from_rotvec(np.pi/4 * np.array([1,0,0]))
    t5 = np.array([0,-1,3])
    H5 = calcHomography(codeShape2, codeHLen1, K1, R5, t5)
    camImg5 = warpCode(codeImg, (1000,1000), H5)
    cv2.imwrite("code5-w5.png", camImg5)
    print("Test 4: x-rotation, written to code5-w5.png")

    K2 = calcIntrinsics([1000,1000], 60)
    t6 = np.array([0,0,3])
    H6 = calcHomography(codeShape2, codeHLen1, K2, R5, t6)
    camImg6 = warpCode(codeImg, (1000,1000), H6)
    cv2.imwrite("code5-w6.png", camImg6)
    print("Test 5: smaller fov, written to code5-w6.png")

    H7 = calcHomography(codeShape2, 1, K2, R5, t6)
    camImg7 = warpCode(codeImg, (1000,1000), H7)
    cv2.imwrite("code5-w7.png", camImg7)
    print("Test 6: smaller code, written to code5-w7.png")

    print("==================== Test addCameraNoise ====================")
    camImg6n1 = addCameraNoise(camImg6, 1000, 5, 10000)
    cv2.imwrite("code5-w6n1.png", camImg6n1)
    print("Test 1: 1000 photons, written to code5-w6n1.png")

    camImg6n2 = addCameraNoise(camImg6, 100, 5, 10000)
    cv2.imwrite("code5-w6n2.png", camImg6n2)
    print("Test 2: 100 photons, written to code5-w6n2.png")
    
    camImg6n3 = addCameraNoise(camImg6, 100, 10, 10000)
    cv2.imwrite("code5-w6n3.png", camImg6n3)
    print("Test 3: 100 photons, read noise = 10, written to code5-w6n3.png")
    
    camImg6n4 = addCameraNoise(camImg6, 10, 5, 10000)
    cv2.imwrite("code5-w6n4.png", camImg6n4)
    print("Test 4: 20 photons, read noise = 5, written to code5-w6n4.png")

    print("==================== Test warpCode(inv)  ====================")
    camImg6n2b = warpCode(camImg6n2, codeImg.shape, H6, True)
    cv2.imwrite("code5-w6n2-b.png", camImg6n2b)
    print("Test: warp back, written to code5-w6n2-b.png")

    print("==================== Test decode warped code  ====================")
    coder = MRCode.MRCoder((120,120), 4, (0.8,0.7,0.6), (5,3,2))
    code, data = coder.genRandomCode()
    K = calcIntrinsics([1000,1000], 60)
    R = Rotation.identity()
    t = np.array([0,0,3])
    H = calcHomography((120,120), 2, K, R, t)
    camImg = warpCode(code*0.8+0.2*255, (1000,1000), H)
    camImg = addCameraNoise(camImg, 100, 5, 10000)
    camImgAligned = warpCode(camImg, (120,120), H, True)
    decodedData = coder.decode(camImgAligned)
    print(data)
    print(decodedData)
    print(str(data) == str(decodedData))

    print("==================== Test calculating error ratio ====================")
    print(coder.calcRawErrorRatio(code, camImgAligned))
    print(coder.calcDecodedErrorRatio(data, camImgAligned))
