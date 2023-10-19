import os
from pathlib import Path
import sys

import numpy as np
import cv2
import glob
import natsort

import torch
import torch.nn as nn
import torch.optim as optim
from model_detect import Net


isclas = False
model = Net(isclas=isclas)

svdir = 'model'
#model.load_state_dict(torch.load('%s/1000.pth' % svdir, map_location='cpu'))
model.load_state_dict(torch.load('%s/2000_possibleSmallCodRandBkg.pth' % svdir, map_location='cpu'))

# input data
# folder_data = '1218-ctrl-se/'
# filenames = glob.glob(folder_data + '*.png')
folder_data = Path(sys.argv[1]).resolve()
filenames = glob.glob(str(folder_data.joinpath('crop/*.png')))
filenames = natsort.natsorted(filenames)
print(filenames)
count = 0

folder_result = folder_data.joinpath('corner')
folder_result.mkdir(exist_ok=True)

for filename in filenames:

    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imgrz_gray = cv2.resize(im, (224, 224), interpolation=cv2.INTER_NEAREST)
    imgrz_gray = imgrz_gray.astype(np.float32)
    imgrz_gray = imgrz_gray / 255.0

    h, w = imgrz_gray.shape
    data = torch.from_numpy(imgrz_gray[np.newaxis, :]).unsqueeze(1)

    # predict
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    data = data.to(device)
    # model.eval()
    kppred, probmap = model(data, verts=1)

    kppred = kppred.detach().cpu().numpy()
    kppred = kppred.squeeze()
    kppred = (kppred + 1) / 2
    kppred[:, 0] *= w
    kppred[:, 1] *= h
    kppred -= 0.5

    imgpoint2 = imgrz_gray.copy()
    imgpoint2 = (imgpoint2 * 255).astype(np.uint8)
    imgpoint2 = cv2.cvtColor(imgpoint2, cv2.COLOR_GRAY2RGB)
    for ii, p in enumerate(kppred):
        x = p[0]
        y = p[1]
        color = [0, 0, 0]
        color[2 - ii] = 255
        cv2.circle(imgpoint2, (int(x), int(y)), 1, color, -1)

    #cv2.imshow('prediction', imgpoint2)
    #cv2.waitKey(0)

    cv2.imwrite(str(folder_result.joinpath(Path(filename).stem+'_result.png')), imgpoint2)
    
    # Sizhuo: save corner coordinates in original size
    with open(folder_result.joinpath(Path(filename).stem+'_result.txt'), 'w') as f:
        for idx in [0, 3, 1, 2]:
            x = (kppred[idx, 0] + 0.5) / 224 * im.shape[1] - 0.5
            y = (kppred[idx, 1] + 0.5) / 224 * im.shape[0] - 0.5
            print('%g %g' % (x, y), file=f)
