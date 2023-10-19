import math, bisect, functools, random, itertools, operator
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import factorial
import utils

class MRCoder:
    def __init__(self, shape=(16,16), numLevels=2, ps=(0.5,), ks=(4,)):
        self.updateParams(shape, numLevels, ps, ks)

    # shape: 2-tuple, shape of lowest level code
    # numLevels:
    # ps: list of p at each level (length=numLevels-1), which is the majority threshold provided by the user
    #   the actual maximum number of minority bits is computed as self.qs
    # ks: list of k at each level (length=numLevels-1), which is the side of code blocks
    # kms: square of k, # of bits in each block
    # es: effective bits at each level
    def updateParams(self, shape, numLevels, ps, ks):
        self.shape = shape
        self.m = self.shape[0] * self.shape[1]
        self.numLevels = numLevels
        self.ps = ps
        self.ks = ks
        self.kms = [x ** 2 for x in ks]
        self.shapes = [shape]
        for i in range(1, numLevels):
            self.shapes.append((self.shapes[-1][0] // ks[i-1], self.shapes[-1][1] // ks[i-1]))
        self.es, self.qs = MRCoder.calcEffectiveBits(shape, ps, ks)
        self.buildBinomTable()
        self.buildMapTable()

    # binomTable: size kmMax+1 x kmMax+1
    # binomTable[i][j] = j choose i, for more efficient search in encode
    def buildBinomTable(self):
        # TODO: compute once, save this into a file
        kmMax = max(self.ks) ** 2
        self.binomTable = [[0] * (kmMax+1) for i in range(kmMax+1)]
        self.binomTable[0][0] = 1
        for j in range(1, kmMax+1):
            for i in range(j+1):
                self.binomTable[i][j] = self.binomial(j-1, i) + self.binomial(j-1, i-1)

    # mapTable: size numLevels-1 x qs[i]+1
    # mapTable[i][j] = the index of last code of Level i, choose up to j minority bits
    def buildMapTable(self):
        self.mapTable = []
        for i in range(self.numLevels-1):
            self.mapTable.append([0] * (self.qs[i]+1))
            for j in range(1, len(self.mapTable[i])):
                self.mapTable[i][j] = self.mapTable[i][j-1] + self.binomial(self.kms[i],j)

    # n choose k
    def binomial(self, n, k):
        if n < 0 or k < 0 or k > n:
            return 0
        else:
            return self.binomTable[k][n]

    def cbn2arr(self, cbn, level, bit):
        code = np.ones(self.kms[level])
        if len(cbn) > 0:
            code[np.array(cbn).T] = 0.0
        if bit == 0: # if this is a black/0 bit, invert 0s and 1s
            code = 1 - code
        code = code.reshape((self.ks[level], self.ks[level]))
        return code

    def arr2cbn(self, arr, bit):
        if bit == 0:
            codeCbn = np.flatnonzero(arr > 0)
        else:
            codeCbn = np.flatnonzero(arr == 0)
        return np.flip(codeCbn)

    def str2arr(self, s, level):
        return np.array([float(x) for x in s]).reshape(self.shapes[level])


    # Encode a lower-level block
    # level: 0-based, 0...numLevels-2
    # n: natural number (>=0) to encode
    # return: code in combination format: (c_b, c_{b-1}, ..., c_1)
    def encodeLower(self, level, n):
        # First find the value of b, # of minority bits chosen
        b = bisect.bisect_left(self.mapTable[level], n)
        # TODO: better Exception?
        if b > self.qs[level]:
            raise Exception("Integer to encode (%d) greater than max value (%d)" % (n, self.mapTable[level][-1]))

        # Then find the corresponding b-bit blocked code
        code = []
        if b > 0:
            nr = n - self.mapTable[level][b-1] - 1
            for i in range(b, 1, -1):
                ci = bisect.bisect(self.binomTable[i], nr)
                code.append(ci-1)
                nr = nr - self.binomTable[i][ci-1]
            code.append(nr)
        return code

    # Top level: 
    # return: code in string format 
    def encodeTop(self, n):
        if n > 2**self.es[-1]-1:
            raise Exception("Integer to encode (%d) greater than max value (%d)" % (n, 2**self.es[-1]-1))
        return format(n, "0%db" % self.es[-1])


    # Decode a lower-level block
    # level: 0-based, 0...numLevels-2
    # upperBit: the bit interpretation of this block in the upper level
    # codeImgBlock: intensities of a single block, np.array
    # thresh: threshold to separate black/white bits
    # return: (array, int), arr and int representation of the code
    def decodeLower(self, level, upperBit, codeImgBlock, thresh):
        # thinking in the space of upperBit = 1(white)
        normInten = codeImgBlock if upperBit else -codeImgBlock
        normThresh = thresh if upperBit else -thresh
        normIntenFlat = normInten.flatten()
        codeArrFlat = normIntenFlat > normThresh
        codeStatus = 0

        # if majority count < km - q, set the q bits with lowest intensity to be 0 (error correction)
        if np.count_nonzero(codeArrFlat) < self.kms[level] - self.qs[level]:
            idx = np.argpartition(normIntenFlat, self.qs[level])
            codeArrFlat = np.ones(codeArrFlat.shape)
            codeArrFlat[idx[:self.qs[level]]] = 0
            codeStatus = 1

        # convert to combination representation, and then to integer
        codeCbn = self.arr2cbn(codeArrFlat, 1)
        b = codeCbn.shape[0]
        codeInt = sum([self.binomial(codeCbn[i], b-i) for i in range(b)])
        if b > 0:
            codeInt += self.mapTable[level][b-1] + 1
        
        # convert back to upperBit = 0 if needed
        if not upperBit:
            codeArrFlat = 1 - codeArrFlat
        codeArr = codeArrFlat.reshape(codeImgBlock.shape)
        return (codeArr, codeInt, codeStatus)


    # Top level:
    # codeImgTop: intensities at top level, np.array
    # thresh: threshold to separate black/white bits
    # return: (array, int), arr and int representation of the code
    def decodeTop(self, codeImgTop, thresh):
        bits = codeImgTop > thresh
        codeStr = ''.join(['1' if b else '0' for b in bits.flatten()])
        return (bits, int(codeStr, 2))


    # Decode entire code
    # codeImg: aligned code image
    # thresh: threshold to separate black/white bits, automatically chosen if None
    # verbose: print debug info
    # return: list of lists (ints in raster scan order for each level)
    def decode(self, codeImg, thresh=None, verbose=False):
        # TODO: automatically determine thresh
        if thresh is None:
            thresh = (np.amin(codeImg) + np.amax(codeImg)) / 2

        inten = self.extractInten(codeImg)
        
        codeInts = []
        codeArrCurr, codeIntCurr = self.decodeTop(inten[-1], thresh)
        codeInts.append([codeIntCurr])

        for l in range(self.numLevels-2, -1, -1):
            codeIntsLevel = []
            codeArrPrev = codeArrCurr
            codeArrCurr = np.zeros(self.shapes[l])
            k = self.ks[l]
            for idx, bit in np.ndenumerate(codeArrPrev):
                intenBlock = inten[l][idx[0]*k:idx[0]*k+k, idx[1]*k:idx[1]*k+k]
                codeArrCurrBlock, codeIntCurr, codeStatus = self.decodeLower(l, bit, intenBlock, thresh)
                if codeStatus and verbose:
                    print('Warning: lower-than-threshold block detected and corrected: Level %d, (%d, %d)' % (l, idx[0], idx[1]))
                codeArrCurr[idx[0]*k:idx[0]*k+k, idx[1]*k:idx[1]*k+k] = codeArrCurrBlock
                codeIntsLevel.append(codeIntCurr)
            codeInts.append(codeIntsLevel)

        return codeInts


    # Decode a lower-level block
    # (This version is based on normalized intensity, not used)
    # level: 0-based, 0...numLevels-2
    # upperBit: the bit interpretation of this block in the upper level
    # notmIntenBlock: normalized intensities of a single block, np.array
    # return: (array, int), arr and int representation of the code
    def decodeLowerNormInten(self, level, upperBit, normIntenBlock):
        # thinking in the space of upperBit = 1(white)
        normInten = normIntenBlock if upperBit else 1-normIntenBlock
        normIntenFlat = normInten.flatten()
        codeArrFlat = normIntenFlat > 0.5
        codeStatus = 0

        # if minority count > q, choose the q bits with lowest intensity (error correction)
        if np.count_nonzero(codeArrFlat) < self.kms[level] - self.qs[level]:
            idx = np.argpartition(normIntenFlat, self.qs[level])
            codeArrFlat = np.ones(codeArrFlat.shape)
            codeArrFlat[idx[:self.qs[level]]] = 0
            codeStatus = 1

        # convert to combination representation, and then to integer
        codeCbn = self.arr2cbn(codeArrFlat, 1)
        b = codeCbn.shape[0]
        codeInt = sum([self.binomial(codeCbn[i], b-i) for i in range(b)])
        if b > 0:
            codeInt += self.mapTable[level][b-1] + 1
        
        # convert back to upperBit = 0 if needed
        if not upperBit:
            codeArrFlat = 1 - codeArrFlat
        codeArr = codeArrFlat.reshape(normIntenBlock.shape)
        return (codeArr, codeInt, codeStatus)


    # Top level:
    # (This version is based on normalized intensity, not used)
    # notmIntenTop: normalized intensities at top level, np.array
    # return: (array, int), arr and int representation of the code
    def decodeTopNormInten(self, normIntenTop):
        bits = normIntenTop > 0.5
        codeStr = ''.join(['1' if b else '0' for b in bits.flatten()])
        return (bits, int(codeStr, 2))


    # Decode entire code
    # (This version is based on normalized intensity, not used)
    # codeImg: aligned code image
    # intenRange: intensity range (may come from fiducial?), set to the range of codeImg if None
    # return: list of lists (ints in raster scan order for each level)
    def decodeNormInten(self, codeImg, intenRange=None, verbose=False):
        normInten = self.extractNormInten(codeImg, intenRange)
        
        codeInts = []
        codeArrCurr, codeIntCurr = self.decodeTopNormInten(normInten[-1])
        codeInts.append([codeIntCurr])

        for l in range(self.numLevels-2, -1, -1):
            codeIntsLevel = []
            codeArrPrev = codeArrCurr
            codeArrCurr = np.zeros(self.shapes[l])
            k = self.ks[l]
            for idx, bit in np.ndenumerate(codeArrPrev):
                normIntenBlock = normInten[l][idx[0]*k:idx[0]*k+k, idx[1]*k:idx[1]*k+k]
                codeArrCurrBlock, codeIntCurr, codeStatus = self.decodeLowerNormInten(l, bit, normIntenBlock)
                if codeStatus and verbose:
                    print('Warning: lower-than-threshold block detected and corrected: Level %d, (%d, %d)' % (l, idx[0], idx[1]))
                codeArrCurr[idx[0]*k:idx[0]*k+k, idx[1]*k:idx[1]*k+k] = codeArrCurrBlock
                codeIntsLevel.append(codeIntCurr)
            codeInts.append(codeIntsLevel)

        return codeInts


    # Generate random code
    # return #1: numpy array, each element represents a bit, 0 or 255
    # return #2: list of lists (ints in raster scan order for each level)
    def genRandomCode(self):
        # Top level
        codeInts = []
        n = random.randrange(2**self.es[-1])
        codeStr = self.encodeTop(n)
        codeCur = self.str2arr(codeStr, -1)
        codeInts.append([n])

        # Lower levels
        for l in range(self.numLevels-2, -1, -1):
            codeIntsLevel = []
            codePrev = codeCur
            codeCur = np.zeros(self.shapes[l])
            k = self.ks[l]
            for idx, bit in np.ndenumerate(codePrev):
                n = random.randrange(self.mapTable[l][-1]+1)
                codeCbn = self.encodeLower(l, n)
                codeBlock = self.cbn2arr(codeCbn, l, bit)
                codeCur[idx[0]*k:idx[0]*k+k, idx[1]*k:idx[1]*k+k] = codeBlock
                codeIntsLevel.append(n)
            codeInts.append(codeIntsLevel)

        return (codeCur*255., codeInts)

    # Compute the effective bits at each level
    @staticmethod
    def calcEffectiveBits(shape, ps, ks, r=0.01):
        numLevels = len(ps) + 1
        kms = [k**2 for k in ks]
        m = shape[0] * shape[1]
        ms = list(itertools.accumulate([m] + kms, operator.floordiv))
        qs = [0] * (numLevels-1) # min # of majority bits
        qs[0] = math.floor(ps[0]*kms[0]+1)
        es = [0] * numLevels

        # Level 0
        numCodes = factorial(kms[0]) / (factorial(np.arange(qs[0],kms[0]+1)) * factorial(kms[0]-np.arange(qs[0],kms[0]+1)))
        es[0] = math.log(sum(numCodes), 2) * ms[1]
        # probability that Level l+1 block intensity > p
        pnp = 1

        # Level 1 to numLevels-2
        # white/black pixel intensity distribution at Level 1
        pw = np.concatenate((np.zeros(qs[0]), numCodes/sum(numCodes))) 
        pb = np.flip(pw)
        for l in range(1, numLevels-1):
            # find the least q > km*p and satisfy p
            for q in range(math.floor(kms[l]*ps[l]+1), kms[l]+1):
                # white/black pixel intensity distribution at Level l+1
                pwNext = 1;
                for i in range(q):
                    pwNext = np.convolve(pwNext, pw)
                for i in range(q, kms[l]):
                    pwNext = np.convolve(pwNext, pb)
                # probability that Level l+1 block intensity > p
                n = len(pwNext)
                pnp = sum(pwNext[math.floor((n-1)*ps[l]+1):])
                if pnp > 1-r:
                    break
            qs[l] = q
            numCodes = factorial(kms[l]) / (factorial(np.arange(q,kms[l]+1)) * factorial(kms[l]-np.arange(q,kms[l]+1)))
            es[l] = math.log(sum(numCodes), 2) * ms[l+1]

        # Top level (Level numLevels-1)
        es[-1] = ms[-1] if pnp > 1-r else 0

        # Convert qs to max # of minority bits which is the one used in the MRCoder class
        qs = [kms[i] - qs[i] for i in range(len(kms))]

        return es, qs

    # Extract bits (no decoding, no error correction) from an aligned code image, whose resolution is a multiple of original code resolution
    # codeImg: aligned code image
    # intenRange: intensity range (may come from fiducial?), set to the range of codeImg if None
    def extractBits(self, codeImg, intenRange=None):
        normIntens = self.extractNormInten(codeImg, intenRange)
        return [a > 0.5 for a in normIntens]

    # Extract intensity for each level from an aligned code image
    # codeImg: aligned code image
    # return: list of numLevels normalized intensity maps, of shapes[l]
    def extractInten(self, codeImg):
        imgShape = codeImg.shape
        ratio = imgShape[0] / self.shape[0]
        assert not ratio % 1 and ratio == imgShape[1] / self.shape[1]
        ratio = int(ratio)

        # Start from lowest level
        imgDs = codeImg
        intens = []
        for l in range(self.numLevels):
            kernel = np.ones((ratio, ratio)) / ratio**2
            imgDs = cv2.filter2D(imgDs, -1, kernel, None, (0,0))
            imgDs = imgDs[::ratio,::ratio]
            intens.append(imgDs)
            if l < self.numLevels-1:
                ratio = self.ks[l]
        return intens


    # Extract normalized intensity in [0,1] from an aligned code image
    # codeImg: aligned code image
    # intenRange: intensity range (may come from fiducial?), set to the range of codeImg if None
    # return: list of numLevels normalized intensity maps, of shapes[l]
    def extractNormInten(self, codeImg, intenRange=None):
        if intenRange is None:
            intenRange = (np.amin(codeImg), np.amax(codeImg))
        intenThresh = (intenRange[0] + intenRange[1]) / 2
        imgShape = codeImg.shape
        ratio = imgShape[0] / self.shape[0]
        assert not ratio % 1 and ratio == imgShape[1] / self.shape[1]
        ratio = int(ratio)

        # Start from lowest level
        imgDs = codeImg
        normIntens = []
        for l in range(self.numLevels):
            kernel = np.ones((ratio, ratio)) / ratio**2
            imgDs = cv2.filter2D(imgDs, -1, kernel, None, (0,0))
            imgDs = imgDs[::ratio,::ratio]
            ni = (imgDs - intenRange[0]) / (intenRange[1] - intenRange[0])
            np.clip(ni, 0, 1)
            normIntens.append(ni)
            if l < self.numLevels-1:
                ratio = self.ks[l]
        return normIntens


    # Calculare raw bits error ratio
    # codeGt: ground truth code
    # codeImg: aligned code image
    # intenRange: intensity range
    # return: ber at each level (top-down) + total
    def calcRawErrorRatio(self, codeGt, codeImg, intenRange=None):
        bitsGt = self.extractBits(codeGt, intenRange)
        bitsImg = self.extractBits(codeImg, intenRange)
        errList = [np.count_nonzero(a != b) for a,b in zip(bitsGt, bitsImg)]
        capList = [s.size for s in bitsGt]
        return [a/b for a,b in zip(reversed(errList), reversed(capList))] + [sum(errList) / sum(capList)]


    # Calculate decoded bits error ratio
    # dataInt: ground truth data in ints
    # decodedInt: decoded data in ints
    # return: ber at each level (top-down) + total
    def calcDecodedErrorRatio(self, dataInt, decodedInt):
        # Top level
        dataTop = format(dataInt[0][0], "0%db" % self.es[-1])
        decodeTop = format(decodedInt[0][0], "0%db" % self.es[-1])
        countDiff = lambda list1, list2: sum([a != b for a, b in zip(list1, list2)])
        errTop = countDiff(dataTop, decodeTop)
        # Lower levels
        # bits per symbol at each level
        countBitError = lambda list1, list2, bitLength: sum([utils.calcBitError(a, b, math.ceil(bitLength)) for a, b in zip(list1, list2)])
        bitLengthList = [a/len(b) for a,b in zip(reversed(self.es[:-1]), dataInt[1:])]
        errList = [errTop] + [countBitError(a, b, c) for a,b,c in zip(dataInt[1:], decodedInt[1:], bitLengthList)]
        capList = list(reversed(self.es))
        return [a/b for a,b in zip(errList, capList)] + [sum(errList) / sum(capList)]


    # Save extracted bits as images
    def saveExtractedBits(self, bits, filePrefix, display=False):
        targetRatio = round(1000 / max(self.shape))
        targetSize = (self.shape[0]*targetRatio, self.shape[1]*targetRatio)
        print([b.shape for b in bits])
        bitImgs = [b*255. for b in bits]
        for i in range(self.numLevels):
            print(bitImgs[i].shape)
            cv2.imwrite(filePrefix+"-d%d.png" % i, cv2.resize(bitImgs[i], targetSize, interpolation=cv2.INTER_NEAREST))
        
        if display:
            [showCode(b) for b in bitImgs]
            plt.show()


    # Save code and block averages for each level
    def saveCodeLevels(self, code, filePrefix, display=False):
        targetRatio = round(1000 / max(self.shape))
        targetSize = (self.shape[0]*targetRatio, self.shape[1]*targetRatio)
        cv2.imwrite(filePrefix+".png", cv2.resize(code, targetSize, interpolation=cv2.INTER_NEAREST))
        codes = []
        for i in range(1, self.numLevels):
            codes.append(cv2.resize(code, self.shapes[i], interpolation=cv2.INTER_AREA))
            cv2.imwrite(filePrefix+"d%d.png" % i, cv2.resize(codes[i-1], targetSize, interpolation=cv2.INTER_NEAREST))
        
        if display:
            showCode(code)
            [showCode(c) for c in codes]
            plt.show()


    # Save code and block averages for each level
    def saveCodeLevelsBinary(self, code, filePrefix, display=False):
        targetRatio = round(1000 / max(self.shape))
        targetSize = (self.shape[0]*targetRatio, self.shape[1]*targetRatio)
        imtemp = cv2.resize(code, targetSize, interpolation=cv2.INTER_NEAREST)
        imtemp = (imtemp > 127) * 255
        cv2.imwrite(filePrefix+"-binary.png", imtemp)
        codes = []
        for i in range(1, self.numLevels):
            imtemp = cv2.resize(code, self.shapes[i], interpolation=cv2.INTER_AREA)
            imtemp = (imtemp > 127) * 255
            imtemp = cv2.resize(imtemp, targetSize, interpolation=cv2.INTER_NEAREST)
            codes.append(imtemp)
            cv2.imwrite(filePrefix+"-binary-d%d.png" % i, imtemp)
        
        if display:
            showCode(code)
            [showCode(c) for c in codes]
            plt.show()


# Helper function that resize a code using nearest neighbors
def resizeCode(code, targetSize):
    return cv2.resize(code, targetSize, interpolation=cv2.INTER_NEAREST)


# Helper function that displays a code on screen
def showCode(code):
    plt.figure()
    plt.imshow(code, interpolation="nearest", cmap="gray", vmin=0, vmax=255)
    # plt.show()


# Tests
if __name__ == "__main__":
    print("\n\n\n====================== Test calcEffectiveBits =======================")
    print(MRCoder.calcEffectiveBits((60,60), (0.8,0.7), (5, 3)))
    print()
    print(MRCoder.calcEffectiveBits((60,60), (0.75,0.6), (5, 3)))
    print()
    print(MRCoder.calcEffectiveBits((60,60), (0.8,0.7), (10, 3)))
    print()
    print(MRCoder.calcEffectiveBits((120,120), (0.8,0.7,0.6), (5, 3, 2)))
    print()

    print("\n\n\n====================== Test 2-level code  =======================")
    coder = MRCoder()
    print(coder.binomTable)
    print(coder.qs)
    print(coder.mapTable)
    testEncodeLower = (0, 1, 3, 16, 17, 100, 136, 137, 26331, 26332, 26333)
    for n in testEncodeLower:
        try:
            print("%d -> %s" % (n, str(coder.encodeLower(0, n))))
        except Exception as err:
            print("Error: %s" % err)
    testEncodeTop = (0, 10000, 65535, 65536)
    for n in testEncodeTop:
        try:
            print("%d -> %s" % (n, coder.encodeTop(n)))
        except Exception as err:
            print("Error: %s" % err)
    
    code1, data1 = coder.genRandomCode()
    print(data1)
    coder.saveCodeLevels(code1, "code1")
    print()

    print("\n\n\n====================== Test 3-level code  =======================")
    coder2 = MRCoder((60,60), 3, (0.8,0.7), (5,3))
    print(coder2.binomTable)
    print(coder2.qs)
    print(coder2.mapTable)
    code2, data2 = coder2.genRandomCode()
    coder2.saveCodeLevels(code2, "code2")
    print()

    coder3 = MRCoder((60,60), 3, (0.75,0.6), (5,3))
    code3, data3 = coder3.genRandomCode()
    coder3.saveCodeLevels(code3, "code3")
    coder3.saveCodeLevelsBinary(code3, "code3")
    print()

    coder4 = MRCoder((60,60), 3, (0.8,0.7), (10,3))
    code4, data4 = coder4.genRandomCode()
    coder4.saveCodeLevels(code4, "code4", False)
    print()

    print("\n\n\n====================== Test 4-level code  =======================")
    coder5 = MRCoder((120,120), 4, (0.8,0.7,0.6), (5,3,2))
    code5, data5 = coder5.genRandomCode()
    coder5.saveCodeLevels(code5, "code5", False)
    print()

    print("\n\n\n====================== Test extractBits  =======================")
    bits = coder5.extractBits(cv2.resize(code5, (600,600), interpolation=cv2.INTER_NEAREST))
    coder5.saveExtractedBits(bits, "code5-extracted", False)

    print("\n\n\n====================== Test decode  =======================")
    print(coder.binomTable)
    print(coder.qs)
    print(coder.mapTable)
    for n in testEncodeLower:
        try:
            codeCbn = coder.encodeLower(0, n)
            codeArr = coder.cbn2arr(codeCbn, 0, 1)
            decodeArr, decodeInt, decodeStatus = coder.decodeLower(0, 1, codeArr, 0.5)
            print("white: %d -> %s -> %d" % (n, str(codeCbn), decodeInt))
            codeArr = coder.cbn2arr(codeCbn, 0, 0)
            decodeArr, decodeInt, decodeStatus = coder.decodeLower(0, 0, codeArr, 0.5)
            print("black: %d -> %s -> %d" % (n, str(codeCbn), decodeInt))
        except Exception as err:
            print("Error: %s" % err)
    for n in testEncodeTop:
        try:
            codeStr = coder.encodeTop(n)
            codeArr = coder.str2arr(codeStr, -1)
            decodeArr, decodeInt = coder.decodeTop(codeArr, 0.5)
            print("%d -> %s -> %d" % (n, codeStr, decodeInt))
        except Exception as err:
            print("Error: %s" % err)
    decodeInt = coder5.decode(code5)
    # print("Data:", data5)
    # print("Decoded:", decodeInt)
    print(str(data5) == str(decodeInt))

