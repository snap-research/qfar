import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import ndimage

class LBPCode:
    def __init__(self, code0, numLevels=2, threshs=(0.7,)):
        if len(threshs) < numLevels - 1:
            threshs = threshs + (threshs[-1],)*(numLevels-1-len(threshs))
        self.numLevels = numLevels
        self.threshs = threshs
        self.code0 = code0.astype(np.uint8)
        self.shape = code0.shape
        self.genLevelCode()
        self.mask = None


    @staticmethod
    def genRandomCode(shape, numLevels=2, threshs=(0.7,), rng=None):
        if rng is None:
            return LBPCode(np.random.randint(0, 2, shape), numLevels, threshs)
        else:
            return LBPCode(rng.integers(0, 2, shape), numLevels, threshs)


    def setQRMask(self, version=1):
        mask0 = np.ones_like(self.code0, dtype=bool)
        pmin = 0
        pmax = self.shape[0]
        # Position marker
        mask0[pmin:pmin+8,pmin:pmin+8] = False
        mask0[pmin:pmin+8,pmax-8:pmax] = False
        mask0[pmax-8:pmax,pmin:pmin+8] = False
        # Alignment
        if version == 2:
            mask0[pmax-9:pmax-4,pmax-9:pmax-4] = False
        # Timing
        mask0[pmin+6,:] = False
        mask0[:,pmin+6] = False
        self.mask = [mask0]

        maski = mask0.astype(float)
        for l in range(self.numLevels-1):
            ratio = l + 2
            kernel = np.ones((ratio, ratio))
            maskCur = cv2.filter2D(maski, -1, kernel, None, (0,0))
            maskCur = maskCur[:self.shape[0]-self.shape[0]%ratio:ratio,
                    :self.shape[1]-self.shape[1]%ratio:ratio]
            self.mask.append(maskCur > 0)


    def getNumAvailableBits(self):
        availableBits = []
        for l in range(self.numLevels):
            if self.mask is None:
                availableBits.append(self.levelCode[k].size)
            else:
                availableBits.append(np.count_nonzero(self.mask[l]))
        return availableBits

    
    def genLevelCode(self):
        code0 = self.code0.astype(float)
        self.levelCode = [code0]
        self.levelCodeBin = [self.code0]
        self.levelCodeTer = [self.code0]
        self.numBits = code0.size

        # Start from lowest level
        for l in range(self.numLevels-1):
            ratio = l + 2
            kernel = np.ones((ratio, ratio)) / ratio**2
            codeCur = cv2.filter2D(code0, -1, kernel, None, (0,0))
            # Discard the last block if not divisible
            codeCur = codeCur[:self.shape[0]-self.shape[0]%ratio:ratio,\
                    :self.shape[1]-self.shape[1]%ratio:ratio]
            codeCurBin = (codeCur > 0.5).astype(np.uint8)
            codeCurTer = np.ones(codeCur.shape, np.uint8) * -1
            codeCurTer[codeCur>self.threshs[l]] = 1
            codeCurTer[codeCur<1-self.threshs[l]] = 0
            self.levelCode.append(codeCur)
            self.levelCodeBin.append(codeCurBin)
            self.levelCodeTer.append(codeCurTer)
            self.numBits += codeCur.size


class LBPCodePartial:
    def __init__(self, codeImg, H, shape=None, numLevels=2, threshs=(0.8,0.7,), normRange=None,
            normPercentile=None):
        self.codeImg = codeImg
        self.H = H
        self.numLevels = numLevels
        if len(threshs) < numLevels:
            self.threshs = threshs + (threshs[-1],)*(numLevels-len(threshs))
        else:
            self.threshs = threshs
        if shape is None:
            self.shape = codeImg.shape[0:2]
        else:
            self.shape = shape
        self.normRange = normRange
        if normPercentile is None:
            normPercentile = [0, 100]
        self.normPercentile = normPercentile

        self.localGamma = None

        #self.extractCode()


    def extractCode(self, correctGamma=True, linearInterp=False):
        codeImg = self.codeImg
        H = self.H
        self.levelCode = []
        self.levelCodeTer = []
        self.numBits = []
        self.numCertainBits = []
#        plt.hist(codeImg.flatten())
#        plt.show()

        intermediateSize = 251
        ratio = self.shape[0] / intermediateSize
        S = np.array([
            [ratio, 0, 0.5*(ratio-1)],
            [0, ratio, 0.5*(ratio-1)],
            [0, 0, 1]])
        Hs = H @ S
        codeIntermediate = cv2.warpPerspective(codeImg, Hs,
                                               (intermediateSize, intermediateSize), 
                                               flags=cv2.INTER_NEAREST |\
                                               cv2.WARP_INVERSE_MAP
                                              )
        codeCur = cv2.resize(codeIntermediate, self.shape, interpolation=cv2.INTER_AREA)
        
        codeNorm = self.normalizeIntensity(codeCur, correctGamma)
        
        for l in range(self.numLevels):
            ratio = l + 1
            kernel = np.ones((ratio, ratio)) / ratio**2
            codeCur = cv2.filter2D(codeNorm, -1, kernel, None, (0,0))
            # Discard the last block if not divisible
            codeCur = codeCur[:codeCur.shape[0]-codeCur.shape[0]%ratio:ratio,\
                    :codeCur.shape[1]-codeCur.shape[1]%ratio:ratio]
#            plt.hist(codeCur.flatten())
#            plt.show()
            codeCurTer = np.ones(codeCur.shape, int) * -1
            codeCurTer[codeCur>self.threshs[l]] = 1
            codeCurTer[codeCur<1-self.threshs[l]] = 0
            self.levelCode.append(codeCur)
            self.levelCodeTer.append(codeCurTer)
            self.numBits.append(codeCur.size)
            self.numCertainBits.append(np.count_nonzero(codeCurTer >= 0))


    def extractCode5(self, correctGamma=True, linearInterp=False):
        codeImg = self.codeImg
        H = self.H
        self.levelCode = []
        self.levelCodeTer = []
        self.numBits = []
        self.numCertainBits = []
#        plt.hist(codeImg.flatten())
#        plt.show()

        Hs = H
        # TODO: deal with spatial-varying depth
        det = (Hs[0,0]*Hs[1,1]-Hs[0,1]*Hs[1,0]) / Hs[2,2]**2
        if det > 1:
            codeCur = cv2.warpPerspective(codeImg, Hs, self.shape, \
                    flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)
        else:
            if linearInterp:
                codeCur = cv2.warpPerspective(codeImg, Hs, self.shape, \
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
            else:
                codeCur = cv2.warpPerspective(codeImg, Hs, self.shape, \
                        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
        
        codeNorm = self.normalizeIntensity(codeCur, correctGamma)
            
        
        for l in range(self.numLevels):
            ratio = l + 1
            kernel = np.ones((ratio, ratio)) / ratio**2
            codeCur = cv2.filter2D(codeNorm, -1, kernel, None, (0,0))
            # Discard the last block if not divisible
            codeCur = codeCur[:codeCur.shape[0]-codeCur.shape[0]%ratio:ratio,\
                    :codeCur.shape[1]-codeCur.shape[1]%ratio:ratio]
#            plt.hist(codeCur.flatten())
#            plt.show()
            codeCurTer = np.ones(codeCur.shape, int) * -1
            codeCurTer[codeCur>self.threshs[l]] = 1
            codeCurTer[codeCur<1-self.threshs[l]] = 0
            self.levelCode.append(codeCur)
            self.levelCodeTer.append(codeCurTer)
            self.numBits.append(codeCur.size)
            self.numCertainBits.append(np.count_nonzero(codeCurTer >= 0))


    def extractCode2(self):
        codeImg = self.codeImg
        H = self.H
        self.levelCode = []
        self.levelCodeTer = []
        self.numBits = []
        self.numCertainBits = []
#        plt.hist(codeNorm.flatten())
#        plt.show()

        for l in range(self.numLevels):
            ratio = l + 1
            curShape = (math.floor(self.shape[0]/ratio), math.floor(self.shape[1]/ratio))
            # TODO: fix non-divisible blocks
            S = np.array([
                [ratio, 0, (ratio-1)/2],
                [0, ratio, (ratio-1)/2],
                [0, 0, 1]])
            Hs = H @ S
            # TODO: deal with spatial-varying depth
            det = (Hs[0,0]*Hs[1,1]-Hs[0,1]*Hs[1,0]) / Hs[2,2]**2
            if det > 1:
                codeCur = cv2.warpPerspective(codeImg, Hs, curShape, \
                        flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)
            else:
                codeCur = cv2.warpPerspective(codeImg, Hs, curShape, \
                        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP)
            if l == 0:
                normRange = self.normRange
                if normRange is None:
                    normRange = np.percentile(codeCur, (1, 99))
            if normRange[0] < normRange[1]:
                codeCur = (codeCur - normRange[0]) / (normRange[1] - normRange[0])
                codeCur[codeCur<0] = 0
                codeCur[codeCur>1] = 1
#            plt.hist(codeCur.flatten())
#            plt.show()
            codeCurTer = np.ones(codeCur.shape, int) * -1
            codeCurTer[codeCur>self.threshs[l]] = 1
            codeCurTer[codeCur<1-self.threshs[l]] = 0
            self.levelCode.append(codeCur)
            self.levelCodeTer.append(codeCurTer)
            self.numBits.append(codeCur.size)
            self.numCertainBits.append(np.count_nonzero(codeCurTer >= 0))


    def extractCode3(self):
        codeImg = self.codeImg
        H = self.H
        self.levelCode = []
        self.levelCodeTer = []
        self.numBits = []
        self.numCertainBits = []
#        plt.hist(codeImg.flatten())
#        plt.show()

        ratio0 = round(codeImg.shape[0] / self.shape[0])
        shape0 = (self.shape[0]*ratio0, self.shape[1]*ratio0)
        S = np.array([
            [1/ratio0, 0, (1/ratio0-1)/2],
            [0, 1/ratio0, (1/ratio0-1)/2],
            [0, 0, 1]])
        Hs = H @ S
        codeCur = cv2.warpPerspective(codeImg, Hs, shape0, \
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        normRange = self.normRange
        if normRange is None:
            normRange = np.percentile(codeCur, (0, 100))
        if normRange[0] < normRange[1]:
            codeCur = (codeCur - normRange[0]) / (normRange[1] - normRange[0])
            codeCur[codeCur<0] = 0
            codeCur[codeCur>1] = 1
        codeNorm = codeCur
        
        for l in range(self.numLevels):
            ratio = ratio0 * (l + 1)
            kernel = np.ones((ratio, ratio)) / ratio**2
            codeCur = cv2.filter2D(codeNorm, -1, kernel, None, (0,0))
            # Discard the last block if not divisible
            codeCur = codeCur[:codeCur.shape[0]-codeCur.shape[0]%ratio:ratio,\
                    :codeCur.shape[1]-codeCur.shape[1]%ratio:ratio]
#            plt.hist(codeCur.flatten())
#            plt.show()
            codeCurTer = np.ones(codeCur.shape, int) * -1
            codeCurTer[codeCur>self.threshs[l]] = 1
            codeCurTer[codeCur<1-self.threshs[l]] = 0
            self.levelCode.append(codeCur)
            self.levelCodeTer.append(codeCurTer)
            self.numBits.append(codeCur.size)
            self.numCertainBits.append(np.count_nonzero(codeCurTer >= 0))


    def extractCode4(self):
        codeImg = self.codeImg
        H = self.H
        self.levelCode = []
        self.levelCodeTer = []
        self.numBits = []
        self.numCertainBits = []
#        plt.hist(codeImg.flatten())
#        plt.show()

        # First do inverse perspective transform to a square image with similar size
        srcCorners = np.array([[-0.5, -0.5], [-0.5, self.shape[0]-0.5], [self.shape[1]-0.5, -0.5],\
                [self.shape[1]-0.5, self.shape[0]-0.5]])
        srcCorners = np.transpose(srcCorners) # x: first row, y: second row
        dstCorners = H @ np.vstack((srcCorners, np.ones((1,4))))
        dstCorners = dstCorners[:2,:] / dstCorners[2:3,:]
        umin, vmin = np.floor(np.amin(dstCorners, axis=1)).astype(int)
        umax, vmax = np.ceil(np.amax(dstCorners, axis=1)).astype(int)
        rectSize = max((umax-umin, vmax-vmin))
        # TODO: here assuming square code
        rectStep = self.shape[0] / rectSize
        u = np.arange(-0.5+rectStep/2, self.shape[1]-0.5, rectStep)
        v = np.arange(-0.5+rectStep/2, self.shape[0]-0.5, rectStep)
        uv, vv = np.meshgrid(u, v)
        rectPoints = np.vstack((uv.flatten(), vv.flatten(), np.ones((1,uv.size))))
        dstPoints = H @ rectPoints
        dstPoints = dstPoints[:2,:] / dstPoints[2:3,:]
        # TODO: adaptive to code resolution
        # this one should be fine for binary codes
        rectGrid = ndimage.map_coordinates(codeImg, np.flipud(dstPoints), order=1, \
                mode='nearest')
        rectGrid = np.reshape(rectGrid, uv.shape)

        # Then low-pass filter and downsample
        ratioP = math.ceil(1/rectStep)
        if ratioP > 1:
            kernel = np.ones((ratioP,ratioP)) / ratioP**2
            rectGridFiltered = cv2.filter2D(rectGrid, -1, kernel)
        else:
            rectGridFiltered = rectGrid
        codeCur = cv2.resize(rectGridFiltered, self.shape, interpolation=cv2.INTER_LINEAR)
        #plt.imshow(rectGrid, cmap='gray')
        #plt.figure()
        #plt.imshow(codeCur, cmap='gray')
        #plt.show()

        # Normalize the intensities
        normRange = self.normRange
        if normRange is None:
            normRange = np.percentile(codeCur, (0, 100))
        if normRange[0] < normRange[1]:
            codeCur = (codeCur - normRange[0]) / (normRange[1] - normRange[0])
            codeCur[codeCur<0] = 0
            codeCur[codeCur>1] = 1
        codeNorm = codeCur
        
        # Threhold the bits
        for l in range(self.numLevels):
            ratio = l + 1
            kernel = np.ones((ratio, ratio)) / ratio**2
            codeCur = cv2.filter2D(codeNorm, -1, kernel, None, (0,0))
            # Discard the last block if not divisible
            codeCur = codeCur[:codeCur.shape[0]-codeCur.shape[0]%ratio:ratio,\
                    :codeCur.shape[0]-codeCur.shape[0]%ratio:ratio]
#            plt.hist(codeCur.flatten())
#            plt.show()
            codeCurTer = np.ones(codeCur.shape, int) * -1
            codeCurTer[codeCur>self.threshs[l]] = 1
            codeCurTer[codeCur<1-self.threshs[l]] = 0
            self.levelCode.append(codeCur)
            self.levelCodeTer.append(codeCurTer)
            self.numBits.append(codeCur.size)
            self.numCertainBits.append(np.count_nonzero(codeCurTer >= 0))


    def normalizeIntensity(self, codeImg, correctGamma):
        perc = np.percentile(codeImg, [5, 50, 95])
        if perc[2] - perc[1] < 0.1 or perc[1] - perc[0] < 0.1:
            self.localGamma = None
            return codeImg
        codeNorm = (codeImg - perc[0]) / (perc[2] - perc[0])
        codeNorm[codeNorm<0] = 0
        codeNorm[codeNorm>1] = 1
        if correctGamma:
            gamma = math.log(0.5, (perc[1] - perc[0]) / (perc[2] - perc[0]))
            codeNorm **= gamma
            self.localGamma = gamma
        return codeNorm


    def isCompatibleWith(self, fullCode):
        if self.numLevels != fullCode.numLevels or self.shape != fullCode.shape:
            return False
        
        for l in range(self.numLevels):
            diffMap = (self.levelCodeTer[l] != fullCode.levelCodeBin[l]) \
                    & (self.levelCodeTer[l] != -1)
            if not fullCode.mask is None:
                diffMap &= fullCode.mask[l]
            if True in diffMap:
                return False
        return True

    
    def countWrongBits(self, fullCode):
        wrongBits = []
        for l in range(self.numLevels):
            diffMap = (self.levelCodeTer[l] != fullCode.levelCodeBin[l]) \
                    & (self.levelCodeTer[l] != -1)
            if not fullCode.mask is None:
                diffMap &= fullCode.mask[l]
            wrongBits.append(np.count_nonzero(diffMap))
        return wrongBits
    
    
    def countWrongBitsWeighted(self, fullCode):
        wrongBits = []
        for l in range(self.numLevels):
            diffMap = (self.levelCodeTer[l] != fullCode.levelCodeBin[l]) \
                    & (self.levelCodeTer[l] != -1)
            if not fullCode.mask is None:
                diffMap &= fullCode.mask[l]
            wrongBits.append(np.count_nonzero(diffMap)*(l+1)**2)
        return wrongBits


    def countWrongBitsBatch(self, level, fullCodeBatch, mask=None):
        levelCodeTer = self.levelCodeTer[level].reshape((1, -1))
        diffMap = (levelCodeTer != fullCodeBatch) & (levelCodeTer != -1)
        if not mask is None:
            diffMap &= mask
        wrongBits = np.count_nonzero(diffMap, 1)
        return wrongBits
    

    def countWrongBitsWeightedBatch(self, level, fullCodeBatch, mask=None):
        levelCodeTer = self.levelCodeTer[level].reshape((1, -1))
        diffMap = (levelCodeTer != fullCodeBatch) & (levelCodeTer != -1)
        if not mask is None:
            diffMap &= mask
        wrongBits = np.count_nonzero(diffMap, 1)*(level+1)**2
        return wrongBits
    
    
    def getNumAvailableBits(self, fullCode):
        if fullCode is None or fullCode.mask is None:
            return self.numCertainBits
        availableBits = []
        for l in range(self.numLevels):
            abMap = (self.levelCodeTer[l] >= 0) & fullCode.mask[l]
            availableBits.append(np.count_nonzero(abMap))
        return availableBits


    def getAvailableBitsVis(self, fullCode):
        vis = []
        for l in range(self.numLevels):
            visCur = np.tile(self.levelCode[l][:,:,np.newaxis], (1,1,3))
            abMap = (self.levelCodeTer[l] >= 0)
            if not fullCode.mask is None:
                abMap &= fullCode.mask[l]
            abMapR = np.dstack((abMap, np.zeros_like(abMap,bool), np.zeros_like(abMap,bool)))
            abMapG = np.dstack((np.zeros_like(abMap,bool), abMap, np.zeros_like(abMap,bool)))
            abMapB = np.dstack((np.zeros_like(abMap,bool), np.zeros_like(abMap,bool), abMap))
            visCur[abMapR] = visCur[abMapR] * 0.5
            visCur[abMapG] = visCur[abMapG] * 0.5 + 0.5
            visCur[abMapB] = visCur[abMapB] * 0.5
            vis.append(visCur)
        return vis


    def getWrongBitsVis(self, fullCode):
        vis = []
        for l in range(self.numLevels):
            visCur = np.tile(self.levelCode[l][:,:,np.newaxis], (1,1,3))
            diffMap = (self.levelCodeTer[l] != fullCode.levelCodeBin[l]) \
                    & (self.levelCodeTer[l] != -1)
            if not fullCode.mask is None:
                diffMap &= fullCode.mask[l]
            diffMapR = np.dstack((diffMap, np.zeros_like(diffMap,bool), np.zeros_like(diffMap,bool)))
            diffMapG = np.dstack((np.zeros_like(diffMap,bool), diffMap, np.zeros_like(diffMap,bool)))
            diffMapB = np.dstack((np.zeros_like(diffMap,bool), np.zeros_like(diffMap,bool), diffMap))
            visCur[diffMapR] = visCur[diffMapR] * 0.5 + 0.5
            visCur[diffMapG] = visCur[diffMapG] * 0.5
            visCur[diffMapB] = visCur[diffMapB] * 0.5
            vis.append(visCur)
        return vis


    def getLevelCodeTerVis(self):
        vis = []
        for l in range(self.numLevels):
            code = self.levelCodeTer[l].astype(float)
            code[code<0] = 0.5
            vis.append(code)
        return vis


    def getIntensityDistance(self, fullCode):
        l1Dist = []
        l2Dist = []
        for i in range(self.numLevels):
            diff = self.levelCode[i] - fullCode.levelCode[i]
            l1Dist.append(np.sum(np.abs(diff)))
            l2Dist.append(np.sum(diff ** 2))
        return (l1Dist, l2Dist)


    def getIntensityDistanceWeighted(self, fullCode):
        l1Dist = []
        l2Dist = []
        for i in range(self.numLevels):
            diff = self.levelCode[i] - fullCode.levelCode[i]
            l1Dist.append(np.mean(np.abs(diff))*(i+1)**2)
            l2Dist.append(np.mean(diff ** 2)*(i+1)**2)
        return (l1Dist, l2Dist)


    def getIntensityDistanceWeightedBatch(self, level, fullCodeBatch, norm='l2'):
        levelCode = self.levelCode[level].reshape((1, -1))
        diff = levelCode - fullCodeBatch
        if norm == 'l1':
            dist = np.mean(np.abs(diff), 1)*(level+1)**2
        elif norm == 'l2':
            dist = np.mean(diff ** 2, 1)*(level+1)**2
        else:
            raise Exception('Unsupported norm')
        return dist.flatten()


    def matchDatabase(self, level, fullCodeBatch, norm='l2'):
        dist = self.getIntensityDistanceWeightedBatch(level, fullCodeBatch, norm)
        part = np.partition(dist, (0, 1))
        return (True, np.argmin(dist), 1-part[0]/part[1])


    def matchDatabaseBin(self, level, fullCodeBatch):
        dist = self.countWrongBitsBatch(level, fullCodeBatch)
        part = np.partition(dist, (0, 1))
        return (True, np.argmin(dist), 1-part[0]/part[1])


    def debugCompatible(self, fullCode):
        l = 1
        codeCurTerVis = self.levelCodeTer[l].astype(float)
        codeCurTerVis[codeCurTerVis==-1] = 0.5
        plt.imshow(codeCurTerVis*255, interpolation='nearest', cmap='gray')
        plt.figure()
        plt.imshow(fullCode.levelCodeBin[l]*255, interpolation='nearest', cmap='gray')
        plt.figure()
        diffMap = (self.levelCodeTer[l] == fullCode.levelCodeBin[l]) \
                | (self.levelCodeTer[l] == -1)
        plt.imshow(diffMap)
        # plot code histograms
        plt.figure()
        plt.hist(fullCode.levelCode[l].flatten())
        plt.figure()
        plt.hist(self.levelCode[l].flatten())
        # plot detected codes
        plt.figure()
        plt.imshow(fullCode.levelCode[l], cmap='gray')
        plt.figure()
        plt.imshow(self.levelCode[l], cmap='gray')
        plt.figure()
        plt.imshow(fullCode.levelCode[0], cmap='gray')
        plt.figure()
        plt.imshow(self.levelCode[0], cmap='gray')
        plt.show()

# corners: 4x2 array, order: UL, UR, LL, LR
# versionSize: 21 for V1
def estimateHomography(corners, versionSize, numPoints=4):
    pmin = 0 - 0.5
    pmax = versionSize - 0.5
    srcCorners = np.array([
        [pmin, pmin],
        [pmax, pmin],
        [pmin, pmax],
        [pmax, pmax],
        [(pmin+pmax)/2, (pmin+pmax)/2]
    ])

    if numPoints == 3:
        corners[3,:] = corners[2,:] + corners[1,:] - corners[0,:]
        H, mask = cv2.findHomography(srcCorners[:4,:], corners)
    elif numPoints == 4:
        H, mask = cv2.findHomography(srcCorners[:4,:], corners)
    elif numPoints == 5:
        divisor = (corners[0,0] - corners[3,0]) * (corners[1,1] - corners[2,1])\
                - (corners[0,1] - corners[3,1]) * (corners[1,0] - corners[2,0])
        isX = ((corners[0,0] * corners[3,1] - corners[0,1] * corners[3,0]) \
               * (corners[1,0] - corners[2,0]) -\
               (corners[1,0] * corners[2,1] - corners[1,1] * corners[2,0]) \
               * (corners[0,0] - corners[3,0])) \
                / divisor
        isY = ((corners[0,0] * corners[3,1] - corners[0,1] * corners[3,0]) \
               * (corners[1,1] - corners[2,1]) -\
               (corners[1,0] * corners[2,1] - corners[1,1] * corners[2,0]) \
               * (corners[0,1] - corners[3,1])) \
                / divisor
        dstCorners = np.vstack((corners, np.array((isX, isY))))
        H, mask = cv2.findHomography(srcCorners, dstCorners)

    return H



if __name__ == '__main__':
    print('==================== Test genFullCode ====================')
    for i in range(3):
        code = LBPCode.genRandomCode(shape=(4,4))
        print(code.code0)
        print(code.levelCode)
        print(code.levelCodeBin)
        print(code.levelCodeTer)
        print(code.numBits)
