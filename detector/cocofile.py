

import glob
import os
import numpy as np
np.random.seed(123456)

# fold = '/u6/a/wenzheng/remote2/datasets/qr-detect-yolo-0.05-0.50'
fold = 'detection_lowres'
imfolder = '%s/images' % fold

a = glob.glob('%s/*.jpg'%imfolder)
a = sorted(a)
# a = a[:1000]

alen = len(a)
idxrandom = np.random.permutation(alen)
a = [a[i] for i in idxrandom]

atrain = a[:alen * 7 // 8]
atest = a[alen * 7 // 8 :]

savefold = 'detection_lowres/data'

fid = open('%s/coco_qr_train.txt' % savefold, 'w')
for line in atrain:
#     print(line)
    fid.write(line)
    fid.write('\n')
fid.close()

fid = open('%s/coco_qr_test.txt' % savefold, 'w')
for line in atest:
    fid.write(line)
    fid.write('\n')
fid.close()

fid = open('%s/coco_qr.names' % savefold, 'w')
fid.write('code')
fid.write('\n')
fid.close()

fid = open('%s/coco_qr.data' % savefold, 'w')
fid.write('classes=1')
fid.write('\n')
fid.write('train=%s/coco_qr_train.txt'%savefold)
fid.write('\n')
fid.write('valid=%s/coco_qr_test.txt'%savefold)
fid.write('\n')
fid.write('names=%s/coco_qr.names'%savefold)
fid.write('\n')
fid.write('backup=backup/')
fid.write('\n')
fid.write('eval=coco')
fid.write('\n')
fid.close()
print('done')
