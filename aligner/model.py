

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


############################################################################
class ResBlock2D(nn.Module):

    def __init__(self, feat_dim, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ResBlock2D, self).__init__()

        self.linear1 = nn.Conv2d(feat_dim, feat_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(feat_dim)
        self.linear2 = nn.Conv2d(feat_dim, feat_dim, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(feat_dim)

    def forward(self, x):

        x1_fc = self.linear1(x)
        x1_bn = self.bn1(x1_fc)
        x1_relu = nn.ReLU()(x1_bn)

        x2_fc = self.linear2(x1_relu)
        x2_bn = self.bn2(x2_fc)
        x2_res = x2_bn + x
        x2_relu = nn.ReLU()(x2_res)

        return x2_relu


########################################################################
from scipy import signal


def gkern(kernlen=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = signal.gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


class GaussianSmoothingDownsampling(nn.Module):

    def __init__(self, channels):
        super(GaussianSmoothingDownsampling, self).__init__()
        
        '''
        kernelnp = np.array([[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]], dtype=np.float32)
        kernel = torch.from_numpy(kernelnp)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        self.register_buffer('weight', kernel)
        '''
        self.groups = channels
        self.conv = F.conv2d
    
    def setweights(self, ks, std):
        self.ks = ks
        '''
        dist = np.arange(ks, dtype=np.float32) - ks // 2
        xs = np.tile(dist.reshape(1, -1), [ks, 1])
        ys = np.tile(dist.reshape(-1, 1), [1, ks])
        dist2 = xs ** 2 + ys ** 2
        std2 = (ks / 2) ** 2
        gaussian = np.exp(-dist2 / std2)
        '''
        gaussian = gkern(ks, std).astype(np.float32)
        gaussian = gaussian / np.sum(gaussian)
        kernel = torch.from_numpy(gaussian)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(self.groups, *[1] * (kernel.dim() - 1))
        
        self.weights = kernel

    def forward(self, x, sz):
        blur = self.conv(x, weight=self.weights.to(x.device), groups=self.groups, padding=self.ks // 2)
        downsample = F.interpolate(blur, size=(sz[1], sz[0]), mode='bilinear', align_corners=True)
        return downsample


class Downsampling(nn.Module):

    def __init__(self):
        super(Downsampling, self).__init__()

    def forward(self, x, sz):
        downsample = F.interpolate(x, size=(sz[1], sz[0]), mode='bilinear', align_corners=True)
        return downsample


##########################################################################3
# 128 2 64
class Im2heat(nn.Module):

    def __init__(self, transmode, kpmode):
        super(Im2heat, self).__init__()
        
        self.transmode = transmode
        self.kpmode = kpmode
        
        assert self.transmode == 'perspective'
        self.dim = 4
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(# 224 2 112
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            # nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),)
        
        # 112 2 112
        # dia=1
        self.res1 = ResBlock2D(32, kernel_size=7, stride=1, padding=3, dilation=1)
        
        self.localization2 = nn.Sequential(# 112 2 56
            nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=6, dilation=2),
            # nn.MaxPool2d(2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),)
        
        # 56 2 56
        self.res2 = ResBlock2D(64, kernel_size=7, stride=1, padding=6, dilation=2)
        
        # final heatmap
        self.convkp = nn.Conv2d(64, self.dim, kernel_size=7, stride=1, padding=3)
        
    def forward(self, x):
        
        # 128 2 64
        x = self.localization(x)
        x = self.res1(x)
        
        # 64 2 32
        x = self.localization2(x)
        x = self.res2(x)
        
        # heat map
        x = self.convkp(x)
        
        return x

        
##########################################################################3
# 128 2 29
# 128 2 29
class Reshape(nn.Module):

    def __init__(self):
        super(Reshape, self).__init__()
    
    def forward(self, x):
        bs = x.shape[0]
        return x.view(bs, -1)


class Im2score(nn.Module):

    def __init__(self, isclas=False):
        super(Im2score, self).__init__()
        
        # resize
        self.resize = GaussianSmoothingDownsampling(1)
        self.resize.setweights(3, 240 / ((21 + 2) * 8))
        
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            # 240 2 120
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            ResBlock2D(32, kernel_size=3, stride=1, padding=1, dilation=1),
            # ResBlock2D(16, kernel_size=3, stride=1, padding=1, dilation=1),
            
            # 120 2 60
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            ResBlock2D(64, kernel_size=3, stride=1, padding=1, dilation=1),
            # ResBlock2D(32, kernel_size=3, stride=1, padding=1, dilation=1),
            
            # 60 2 30
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            ResBlock2D(128, kernel_size=3, stride=1, padding=1, dilation=1),
            # ResBlock2D(64, kernel_size=3, stride=1, padding=1, dilation=1),
            
            # final conv
            nn.Conv2d(128, 1, kernel_size=3, padding=1)
            )

    def forward(self, x, verts):
        
        for i, vert in enumerate(verts):
            vertsz = (17 + vert * 4 + 2) * 8
            assert vertsz == 23 * 8
        
        x_resize = self.resize(x, (vertsz, vertsz))
        x_deblur = self.localization(x_resize)
        
        return x_deblur

