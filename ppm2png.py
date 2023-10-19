# Convert PPM to PNG
from pathlib import Path
from time import perf_counter
import json
import subprocess
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import qrcode
import LBPCode
import MRCodeSim
import utils
from utils import print_log

workDir = Path('D:/python/MRCode/qrImages_1026/')
srcDir = workDir.joinpath('ppm')
dstDir = workDir.joinpath('png')
dstDir.mkdir(parents=True, exist_ok=True)
print('Created dir: ' + str(dstDir))

imgSuffix = '.ppm'
outSuffix = '.png'
imgPattern = '*' + imgSuffix
imgIDList = [p.with_suffix('').name for p in srcDir.glob(imgPattern)]
#imgIDList = ['20200804_151604', '20200804_151613', '20200804_151622']
#imgIDList = ['20200810_171856']
imgIDList.sort()
print(imgIDList)

toRotate = True


with open(str(workDir.joinpath('results.txt')), 'a') as f:
    for imgID in imgIDList:
        t0 = perf_counter()
        print_log('IMG: ' + imgID, f)
        im = cv2.imread(str(srcDir.joinpath(imgID+imgSuffix)), cv2.IMREAD_UNCHANGED)
        if toRotate:
            im = np.transpose(im, (1, 0, 2))
            im = np.flipud(im)
        cv2.imwrite(str(dstDir.joinpath(imgID+outSuffix)), im)

        t1 = perf_counter()
        print_log('Time for processing image: %f' % (t1-t0), f)
        print_log('', f)

