

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from numpy.linalg import svd
from numpy.linalg import solve


##################################################
def transgird(x, mtx, gridxy1_1xhxwx3):
    # grid
    gridxy1_1xhxwx3 = gridxy1_1xhxwx3.to(x.device)
    
    _, h, w, _ = gridxy1_1xhxwx3.shape
    gridxy1_1x3xhxw = gridxy1_1xhxwx3.permute(0, 3, 1, 2)
    grid_1x3xk = gridxy1_1x3xhxw.reshape(1, 3, -1)
    gridnew = torch.matmul(mtx, grid_1x3xk.repeat(x.shape[0], 1, 1))
    
    # mtx could be bx2x3 or bx3x3
    # bx2x3 is affine
    # bx3x3 is perspective
    dim = gridnew.shape[1]
    if dim == 3:
        gridnew = gridnew[:, :2, :] / (1e-8 + gridnew[:, 2:, :])
    
    gridnew_bx2xhxw = gridnew.view(-1, 2, h, w)
    gridnew_bxhxwx2 = gridnew_bx2xhxw.permute(0, 2, 3, 1)
    
    # interpolate
    return F.grid_sample(x, gridnew_bxhxwx2)


##########################################################################
def perspectivepointsold(srcpoints_4x3, dstpoints_bx4x2):
    
    # src, [-1, -1], [-1, 1], [1, 1], [1, -1]
    # dst, kp in image
    # mtx * src = dst
    srcpoints_bx4x3 = srcpoints_4x3.unsqueeze(0).repeat(dstpoints_bx4x2.shape[0], 1, 1).to(dstpoints_bx4x2.device)

    A_xy1_bx4x3 = srcpoints_bx4x3
    zeros_bx4x3 = torch.zeros_like(A_xy1_bx4x3)
    A_xy1000_bx4x6 = torch.cat([A_xy1_bx4x3, zeros_bx4x3], dim=2)
    A_000xy1_bx4x6 = torch.cat([zeros_bx4x3, A_xy1_bx4x3], dim=2)
    
    srcpoints_bx4x2 = srcpoints_bx4x3[:, :, :2]
    axay = dstpoints_bx4x2[:, :, :1] * srcpoints_bx4x2
    bxby = dstpoints_bx4x2[:, :, 1:2] * srcpoints_bx4x2
    
    # build mtx
    row1357_bx4x8 = torch.cat([A_xy1000_bx4x6, -axay], dim=2)
    row2468_bx4x8 = torch.cat([A_000xy1_bx4x6, -bxby], dim=2)
    A_bx8x8 = torch.cat([row1357_bx4x8, row2468_bx4x8], dim=1)
    B_bx8x1 = torch.cat([dstpoints_bx4x2[:, :, :1], dstpoints_bx4x2[:, :, 1:2]], dim=1)
    
    mtx_bx8x1 = torch.solve(B_bx8x1, A_bx8x8)[0]
    mtx_bx9x1 = torch.cat([mtx_bx8x1, torch.ones_like(mtx_bx8x1[:, :1, :])], dim=1)
    mtx_bx3x3 = mtx_bx9x1.view(-1, 3, 3)
    
    return mtx_bx3x3


def perspectivepoints(srcpoints_4x3, dstpoints_bx4x2):
    
    # src, [-1, -1], [-1, 1], [1, 1], [1, -1]
    # dst, kp in image
    # mtx * src = dst
    srcpoints_bx4x3 = srcpoints_4x3.unsqueeze(0).repeat(dstpoints_bx4x2.shape[0], 1, 1).to(dstpoints_bx4x2.device)

    X_bx4x1 = srcpoints_bx4x3[:, :, :1]
    Y_bx4x1 = srcpoints_bx4x3[:, :, 1:2]
    
    u_bx4x1 = dstpoints_bx4x2[:, :, :1]
    v_bx4x1 = dstpoints_bx4x2[:, :, 1:2]
    
    rows0_bx4x3 = torch.zeros_like(srcpoints_bx4x3)
    rowsXY_bx4x3 = -torch.cat((X_bx4x1, Y_bx4x1, torch.ones_like(X_bx4x1)), dim=2)
    
    hx_bx4x9 = torch.cat((rowsXY_bx4x3, rows0_bx4x3, \
                    u_bx4x1 * X_bx4x1, u_bx4x1 * Y_bx4x1, u_bx4x1), dim=2)
    hy_bx4x9 = torch.cat((rows0_bx4x3, rowsXY_bx4x3, \
                    v_bx4x1 * X_bx4x1, v_bx4x1 * Y_bx4x1, v_bx4x1), dim=2)
    
    h_bx8x9 = torch.cat((hx_bx4x9, hy_bx4x9), dim=1)
    h_bx9x8 = h_bx8x9.permute(0, 2, 1)

    '''
    mtx_b = []
    for i, h in enumerate(h_bx9x8):
        print(dstpoints_bx4x2[i])
        print(h)
        u, s, v = torch.svd(h, some=False)
        mtx_b.append(u[:, 8].reshape(3, 3))
    
    mtx_bx3x3 = torch.stack(mtx_b)
    '''
    
    # now it supports batch
    u, _, _ = torch.svd(h_bx9x8, some=False)
    mtx_bx9 = u[:, :, 8]
    mtx_bx3x3 = mtx_bx9.reshape(-1, 3, 3)
    
    return mtx_bx3x3


###############################################################################################
def heatmap2points(x, xcoord, ycoord):
    
    bs, kpnum, vh, vw = x.shape
    # initialize
    prob_bxkxhxw = x
    prob_bxkxhw = prob_bxkxhxw.view(bs, kpnum, vh * vw)
    prob_bxkxhw = nn.Softmax(dim=2)(prob_bxkxhw)

    # coordinate
    prob_bxkxhxw = prob_bxkxhw.view(bs, kpnum, vh, vw)
    sx = prob_bxkxhxw * xcoord.to(x.device)
    sy = prob_bxkxhxw * ycoord.to(x.device)
    sx = sx.view(bs, kpnum, vh * vw).sum(2)
    sy = sy.view(bs, kpnum, vh * vw).sum(2)
    return torch.stack([sx, sy], dim=2)


def heatmap2lines(x, coord_xy1):
    
    bs, kpnum, vh, vw = x.shape
    prob_bxkxhxw = nn.Sigmoid()(x)

    # 4 lines
    xy1_1x3xhxw = coord_xy1.to(x.device)
    lines = []
    for i in range(kpnum):
        prob = prob_bxkxhxw[:, i:i + 1, :, :]
        lineeq = xy1_1x3xhxw * prob
        AT_bx3xk = lineeq.view(bs, 3, -1)
        A_bxkx3 = AT_bx3xk.permute(0, 2, 1)
        ATA_bx3x3 = torch.matmul(AT_bx3xk, A_bxkx3)
        
        '''
        eqs = []
        for j in range(bs):
            ATA = ATA_bx3x3[j]
            # svd is unstable
            # u, s, v = torch.svd(ATA)
            # eq = u[:, 2:3]
            _, vec = torch.symeig(ATA, eigenvectors=True, upper=True)
            eq = vec[:, :1]
            eqs.append(eq)
        eqs_3xb = torch.cat(eqs, dim=1)
        eqs_bx3 = eqs_3xb.t()
        '''
        # now it supports batch
        u, _, _ = torch.svd(ATA_bx3x3)
        eqs_bx3 = u[:, :, 2]
        lines.append(eqs_bx3)
    
    return lines


def lines2points(lines_4xbx3):    
    # intersections
    points = []
    idices = [[0, 3], [1, 0], [2, 1], [3, 2]]
    for idx in idices:
        be = lines_4xbx3[idx[0]]
        en = lines_4xbx3[idx[1]]
        po = torch.cross(be, en)
        po = po[:, :2] / (1e-8 + po[:, 2:3])
        points.append(po)

    points_bx4x2 = torch.stack(points, dim=1)
    points_bx4x2 = torch.clamp(points_bx4x2, -1.0, 1.0)
    return points_bx4x2


##############################################################################
def heatmap2linesnp(x, coord_xy1):
    
    bs, kpnum, vh, vw = x.shape
    prob_bxkxhxw = nn.Sigmoid()(x)

    # 4 lines
    xy1_1x3xhxw = coord_xy1.cpu().numpy()
    prob_bxkxhxw = prob_bxkxhxw.detach().cpu().numpy()
    
    lines = []
    for i in range(kpnum):
        prob_bx1xhxw = prob_bxkxhxw[:, i:i + 1, :, :]
        lineeq_bx3xhxw = xy1_1x3xhxw * prob_bx1xhxw
        eqs_bx3 = robustheatmap2lines(lineeq_bx3xhxw.reshape(bs, 3, -1))
        lines.append(eqs_bx3)
    
    lines = [da.to(x.device) for da in lines]
    return lines


def robustheatmap2lines(AT_bx3xk):
    # z = At
    # z = ax + by + d
    # A = [x, y, ones(size(x))];
    # t = (A'*A)\(A'*z);
    # disp(t);
    
    A_bxkx3 = np.transpose(AT_bx3xk, [0, 2, 1])
    ATA_bx3x3 = np.matmul(AT_bx3xk, A_bxkx3)
    
    # now it supports batch
    u, _, _ = svd(ATA_bx3x3)
    t_bx3x1 = u[:, :, 2:3]
    tstop_b = t_bx3x1[:, 2, 0]
    
    # robust estimation
    for i in range(100):
        e_bxkx1 = np.matmul(A_bxkx3, t_bx3x1);
        sigmaGM2_bx1x1 = np.var(e_bxkx1, axis=1, keepdims=True);
        w_bxkx1 = 2 * sigmaGM2_bx1x1 / ((e_bxkx1 ** 2 + sigmaGM2_bx1x1) ** 2);
        
        A2 = A_bxkx3 * w_bxkx1;
        AT = np.transpose(A2, [0, 2, 1])
        ATA = np.matmul(AT, A2)
        u, _, _ = svd(ATA)
        t_bx3x1 = u[:, :, 2:3]
        err = np.mean(np.abs(t_bx3x1[:, 2, 0] - tstop_b))
        # print('iter %d err %.5f' % (i, err))
        if err < 1e-5:
            break;
        tstop_b = t_bx3x1[:, 2, 0]
    
    return torch.from_numpy(t_bx3x1[:, :, 0])


#####################################################################
def heatmap2planenp(x, coord_xy1):
    
    bs, kpnum, vh, vw = x.shape
    prob_bxkxhxw = nn.Sigmoid()(x)

    # 4 lines
    xy1_1x3xhxw = coord_xy1.cpu().numpy()
    prob_bxkxhxw = prob_bxkxhxw.detach().cpu().numpy()
    
    xy1_1x3xk = xy1_1x3xhxw.reshape(1, 3, -1)
    xy1_bx3xk = np.tile(xy1_1x3xk, reps=(bs, 1, 1))
    prob_bx4xk = prob_bxkxhxw.reshape(bs, kpnum, -1)
    
    lines = []
    for i in range(kpnum):
        prob_bx1xk = prob_bx4xk[:, i:i + 1, :]
        eqs_bx3 = robustheatmap2planes(xy1_bx3xk, prob_bx1xk, i)
        lines.append(eqs_bx3)
    
    lines = [da.to(x.device) for da in lines]
    return lines


def robustheatmap2planes(xy1_bx3xk, z_bx1xk, kpidx):
    
    # A_bxkx3 * t_bx3x1 = b_bxkx1
    A_bxkx3 = np.transpose(xy1_bx3xk, [0, 2, 1])
    b_bxkx1 = np.transpose(z_bx1xk, axes=[0, 2, 1])
    W = (b_bxkx1 > 0.05).astype(np.float32)
    A_bxkx3 = A_bxkx3 * W
    b_bxkx1 = b_bxkx1 * W
    
    AT = np.transpose(A_bxkx3, [0, 2, 1])
    ATA = np.matmul(AT, A_bxkx3)
    ATb = np.matmul(AT, b_bxkx1)

    t_bx3x1 = solve(ATA, ATb)
    tstop_b = t_bx3x1[:, 2, 0]
    
    # robust estimation
    for i in range(100):
        e_bxkx1 = np.matmul(A_bxkx3, t_bx3x1) - b_bxkx1;
        sigmaGM2_bx1x1 = np.var(e_bxkx1, axis=1, keepdims=True);
        w_bxkx1 = 2 * sigmaGM2_bx1x1 / ((e_bxkx1 ** 2 + sigmaGM2_bx1x1) ** 2);
        
        A2 = A_bxkx3 * w_bxkx1
        b2 = b_bxkx1 * w_bxkx1
        AT = np.transpose(A2, [0, 2, 1])
        ATA = np.matmul(AT, A2)
        ATb = np.matmul(AT, b2)
        # t_bx3x1 = solve(ATA, ATb)
        try:
            t_bx3x1 = solve(ATA, ATb)
        except:
            if kpidx == 0:
                tmp = [1, 0, 1]
            elif kpidx == 1:
                tmp = [0, 1, -1]
            elif kpidx == 2:
                tmp = [1, 0, -1]
            elif kpidx == 3:
                tmp = [0, 1, 1]
            t_bx3x1 = np.tile(np.array([tmp], dtype=np.float32), (xy1_bx3xk.shape[0], 1)).reshape(-1, 3, 1)
        
        err = np.mean(np.abs(t_bx3x1[:, 2, 0] - tstop_b))
        print('iter %d err %.5f' % (i, err))
        if err < 1e-5:
            break;
        tstop_b = t_bx3x1[:, 2, 0]
    
    return torch.from_numpy(t_bx3x1[:, :, 0])
    
