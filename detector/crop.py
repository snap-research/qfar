from pathlib import Path
import sys
import shutil
import math
import cv2
import glob
import numpy as np

# imgDir = Path('inference_input')
imgDir = Path(sys.argv[1])
infDir = Path('inference_output')
infPathList = list(infDir.glob('*'))
imgIDList = [p.with_suffix('').name for p in infPathList]
imgSuffix = infPathList[0].suffix

cropDir = Path('crop')
if cropDir.exists():
    shutil.rmtree(cropDir)
cropDir.mkdir(exist_ok=True)

#rot90 = 3
#borderRatio = 6
borderRatio = 10

for imgID in imgIDList:
    imgPath = imgDir.joinpath('%s%s' % (imgID, imgSuffix))
    resultPattern = infDir.joinpath('%s%s/%s_*.txt' % (imgID, imgSuffix, imgID))
    print(imgPath)
    print(resultPattern)
    
    im = cv2.imread(str(imgPath), cv2.IMREAD_COLOR)
    if im.shape[0] > im.shape[1]:
        im = np.rot90(im, 1)
        rot90 = 3
    else:
        rot90 = 0
    # rot90 = 2
    
    cropCount = 0
    infoPath = cropDir.joinpath('%s.txt' % imgID)
    with open(infoPath, 'w') as infoFile:
        for resultPath in glob.glob(str(resultPattern)):
            boxes = []
            with open(resultPath, 'r') as f:
                for line in f:
                    box = [int(s) for s in line.strip().split(' ')[:4]]
                    boxes.append(box)

            for box in boxes:
                w = box[2] - box[0]
                h = box[3] - box[1]
                box0 = max(0, box[0] - math.ceil(w / borderRatio))
                box2 = min(im.shape[1], box[2] + math.ceil(w / borderRatio))
                box1 = max(0, box[1] - math.ceil(h / borderRatio))
                box3 = min(im.shape[0], box[3] + math.ceil(h / borderRatio))
                crop = im[box1:box3,box0:box2,:]
                if rot90 > 0:
                    crop = np.rot90(crop, rot90)
                outputPath = cropDir.joinpath('%s_%d.png' % (imgID, cropCount))
                print(outputPath)
                print(box0, box1, box2, box3)
                cv2.imwrite(str(outputPath), crop)
                cropCount += 1
                if rot90 == 3:
                    print('%d %d %d %d' % (im.shape[0]-box[3], box[0], im.shape[0]-box[1], box[2]), file=infoFile)
                else:
                    print('%d %d %d %d' % (box[0], box[1], box[2], box[3]), file=infoFile)
            
