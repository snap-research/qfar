

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn

import numpy as np
from model import Im2score, Im2heat
from utils import heatmap2points, \
perspectivepoints, transgird


########################################################
class Net(nn.Module):

    def __init__(self, transmode='perspective', kpmode='point', detach=0, \
                 imdim=224, heatdim=56, sampledim=240, isclas=False):
        super(Net, self).__init__()
        
        self.transmode = transmode
        self.kpmode = kpmode
        assert kpmode == 'point'
        self.detach = detach
        
        self.imdim = imdim
        self.heatdim = heatdim
        self.sampledim = sampledim

        ########################################################
        # Spatial transformer localization-network
        self.Im2stn = Im2heat(transmode, kpmode)
        
        # regress
        self.clas = isclas
        self.Im2score = Im2score(isclas=isclas)
        
        ################################################################
        imx = np.arange(sampledim, dtype=np.float32).reshape(1, -1)
        imx = (imx + 0.5) / sampledim * 2 - 1
        imx = np.tile(imx, [sampledim, 1]).reshape(1, sampledim, sampledim, 1)
        
        imy = np.arange(sampledim, dtype=np.float32).reshape(-1, 1)
        imy = (imy + 0.5) / sampledim * 2 - 1
        imy = np.tile(imy, [1, sampledim]).reshape(1, sampledim, sampledim, 1)
        
        im1 = np.ones_like(imx)
        imxy1_1xhxwx3 = np.concatenate([imx, imy, im1], axis=3)
        self.imxy1_1xhxwx3 = torch.from_numpy(imxy1_1xhxwx3)
        
        ############################################
        # coord
        dim = heatdim
        x = (np.arange(dim, dtype=np.float32) + 0.5) / dim * 2 - 1
        y = x.copy()
        x = x.reshape(1, -1)
        y = y.reshape(-1, 1)
        x = np.tile(x, [dim, 1])
        y = np.tile(y, [1, dim])
        self.xcoord = torch.from_numpy(x).view(1, 1, dim, dim)
        self.ycoord = torch.from_numpy(y).view(1, 1, dim, dim)
        self.coord_xy1 = torch.cat((self.xcoord, self.ycoord, torch.ones_like(self.xcoord)), dim=1)

        #########################################################
        assert transmode == 'perspective'
        srcpoints = np.array([[0, 0], [0, sampledim - 1], [sampledim - 1, sampledim - 1], [sampledim - 1, 0]], dtype=np.float32)
        
        srcpoints = (srcpoints + 0.5) / sampledim * 2 - 1
        srcpoints = srcpoints * 0.85
        srcpoints = np.concatenate([srcpoints, np.ones_like(srcpoints[:, :1])], axis=1)
        self.srcpoints = torch.from_numpy(srcpoints)
        
        self.norm = nn.BatchNorm2d(1, affine=False)
        
    def stn(self, x):
        # transform the input
        heatmap = self.Im2stn(x)
        
        if self.kpmode == 'point':
            re = heatmap2points(heatmap, self.xcoord, self.ycoord)
            
        # perspective
        mtx = perspectivepoints(self.srcpoints, re)
            
        # new image
        x_align = transgird(x, mtx, self.imxy1_1xhxwx3)

#         x_norm = self.norm(x_align)

        return x

    def forward(self, x, croparea=None, debugkp=None, verts=None):
        
        # transform the input
        heatmap = self.Im2stn(x)
        
        if self.kpmode == 'point':
            re = heatmap2points(heatmap, self.xcoord, self.ycoord)
            
        print("\tIn Model: device is", x.device)
        
        return re, heatmap
                
#         # perspective
#         mtx = perspectivepoints(self.srcpoints, re)
            
#         # new image
#         x_align = transgird(x, mtx, self.imxy1_1xhxwx3)
        
#         # nromalize
#         x_norm = self.norm(x_align)

#         # Perform the usual forward pass
#         if self.clas:
#             x, clas = self.Im2score(x_norm, verts)
#         else:
#             x = self.Im2score(x_norm, verts)
#             clas = None
        
#         # add square constraint
#         mtx2 = None
#         if not (croparea is None):
#             dst_be_bx1x2 = croparea[:, 0:1, :]
#             dst_en_bx1x2 = croparea[:, 1:2, :]
#             dst_cropsz_bx1x2 = dst_en_bx1x2 - dst_be_bx1x2
            
#             dst_fullsz_bx1x1 = croparea[:, 2:3, 0:1]
#             dst_fi_bx1x1 = croparea[:, 2:3, 1:2]
#             dst_uv_bx1x2 = croparea[:, 3:4, :]
            
#             if debugkp is None:
#                 dstpoints_bx4x2 = (re + 1) / 2
#             else:
#                 dstpoints_bx4x2 = (debugkp + 1) / 2
            
#             dstpoints_bx4x2 = dstpoints_bx4x2 * dst_cropsz_bx1x2 - 0.5
#             dstpoints_bx4x2 = dstpoints_bx4x2 + dst_be_bx1x2
            
#             dstpoints_bx4x2 = (dstpoints_bx4x2 + 0.5) / dst_fullsz_bx1x1
#             dstpoints_bx4x2 = dstpoints_bx4x2 - dst_uv_bx1x2
#             dstpoints_bx4x2 = dstpoints_bx4x2 / dst_fi_bx1x1
            
#             mtx2 = perspectivepoints(self.srcpoints, dstpoints_bx4x2)
        
#         if self.kpmode == 'point':
#             bs, kpnum, vh, vw = heatmap.shape
#             # initialize
#             prob_bxkxhxw = heatmap
#             prob_bxkxhw = prob_bxkxhxw.view(bs, kpnum, vh * vw)
#             prob_bxkxhw = nn.Softmax(dim=2)(prob_bxkxhw)
#             heatmap = prob_bxkxhw.view(bs, kpnum, vh, vw)
            
#             return (x, clas, x_align), re, heatmap, mtx, mtx2

