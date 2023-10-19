# Simulate code localization
from enum import Enum
import math
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import numpy as np
import cv2
from matplotlib import pyplot as plt
import MRCodeSim

rng = np.random.default_rng()
DEBUG = False

class CamParam:
    def __init__(self, hfov, res=(4000,3000), pp=None):
        self.hfov = hfov
        self.res = res
        if pp is None:
            self.pp = np.array(((0+res[1]-1)/2, (0+res[0]-1)/2))
        else:
            self.pp = np.array(pp)
        self.f = res[1]*0.5/np.tan(np.radians(hfov/2))
        self.K = np.array([
            [self.f, 0, self.pp[0]],
            [0, self.f, self.pp[1]],
            [0, 0, 1]
        ])
        # transform cam space to code space
        self.codeR = R.identity()
        self.codeT = np.zeros(3)
        # transform cam space to world space
        self.worldR = R.identity()
        self.worldT = np.zeros(3)
        self.worldGpsT = np.zeros(3)

    def getCodePose(self, codeSize=1):
        gtCodeAngles = self.codeR.inv().as_euler('xyz')
        gtCodeT = self.codeR.inv().apply(-self.codeT)
        gtCodeT *= codeSize
        gtCodeT[2] /= self.f
        return np.hstack((gtCodeT, gtCodeAngles))


class CodeParam:
    def __init__(self, codeSize):
        self.codeSize = codeSize
        # transoform code space to world space
        self.worldR = R.identity()
        self.worldT = np.zeros(3)

    def getRvec(self):
        return self.worldR.inv().as_rotvec().reshape([3,1])

    def getTvec(self):
        return -self.worldR.inv().apply(self.worldT).reshape([3,1])


LOCALIZE_CENTER = 0
LOCALIZE_P3P = 1
LOCALIZE_EPNP = 2
LOCALIZE_RANSAC = 3
LOCALIZE_ITERATIVE = 4
LOCALIZE_ITERATIVE_2D = 5

# randomly generate cameras and code poses
# TODO: add principal point and lens distortion?
def randWorld(N, radiusRange, maxAngle, gpsSigma=0, yMinRange=(0,0), yDiffMax=0,
                  codeSizeRange=(1,1), codeXTiltRange=(0,0), codeZTiltRange=(0,0),
                  camFovRange=(47,47), lookAtRange=(0,0), camTiltRange=(0,0)):
    # randomly sample camera locations in code space
    r = np.sqrt((radiusRange[1]**2-radiusRange[0]**2) * rng.random(N) + radiusRange[0]**2)
    theta = np.radians(rng.random(N) * 2 * maxAngle - maxAngle)
    camGtZ = -r * np.cos(theta)
    camGtX = r * np.sin(theta)
    camGtY = rng.uniform(yMinRange[0], yMinRange[1]) + rng.uniform(-yDiffMax, yDiffMax, N)

    # randomly generate code param
    codeSize = rng.uniform(codeSizeRange[0], codeSizeRange[1])
    codeParam = CodeParam(codeSize)
    
    # randomly generate cam param and orientations
    camFov = rng.uniform(camFovRange[0], camFovRange[1], N)
    lookAtRadius = rng.uniform(lookAtRange[0], lookAtRange[1], N)
    lookAtTheta = rng.uniform(-np.pi, np.pi, N)
    camTilt = np.radians(rng.uniform(camTiltRange[0], camTiltRange[1], N))
    camParams = []
    for i in range(N):
        camCur = CamParam(camFov[i])
        camCur.codeT = np.array([camGtX[i], camGtY[i], camGtZ[i]])

        # map the "lookat" ray to (0,0,0) in the code space
        lookAtX = lookAtRadius[i] * np.cos(lookAtTheta[i])
        lookAtY = lookAtRadius[i] * np.sin(lookAtTheta[i])
        lookAt = np.array([lookAtX, lookAtY, camCur.f])
        lookAt /= np.linalg.norm(lookAt)
        target = np.array([0, 0, 0]) - camCur.codeT
        targetNorm = np.linalg.norm(target)
        assert targetNorm > 0
        target /= targetNorm
        rot1 = np.arctan2(target[2], target[0]) - np.arctan2(lookAt[2], lookAt[0])
        rot1R = R.from_euler('y', rot1)
        lookAtTemp = rot1R.apply(lookAt)
        lookAtTemp /= np.linalg.norm(lookAtTemp)
        crossProd = np.cross(lookAtTemp, target)
        crossProdNorm = np.linalg.norm(crossProd)
        if crossProdNorm == 0:
            rot2 = 0
            rot2R = R.identity()
        else:
            dotProd = np.dot(lookAtTemp, target)
            rot2 = np.arctan2(crossProdNorm, dotProd)
            rot2R = R.from_rotvec(crossProd / crossProdNorm * rot2)
        rot0R = R.from_euler('z', camTilt[i])
        camCur.codeR = rot2R * rot1R * rot0R
        camParams.append(camCur)

        # verify lookAt works
#        temp3d = camCur.codeR.inv().apply(-camCur.codeT)
#        temp2d = temp3d @ camCur.K.T
#        temp2d = temp2d[:2] / temp2d[2:] - camCur.pp
#        print_debug('LookAt GT: [%g %g]' % (lookAtX, lookAtY) )
#        print_debug('LookAt warped: ', temp2d)
#        print_debug('LookAt norm error: ', np.linalg.norm([lookAtX, lookAtY]) \
#              - np.linalg.norm(temp2d))
#        print_debug()
    
    # y-rotate and translate
    codeWorldT = rng.uniform(0, 10, 3)
    codeWorldR = R.from_euler('y', rng.uniform(0, 2*np.pi))
    for i in range(N):
        camParams[i].worldR = codeWorldR * camParams[i].codeR
        camParams[i].worldT = codeWorldR.apply(camParams[i].codeT) + codeWorldT
        camParams[i].worldGpsT = rng.normal(camParams[i].worldT, gpsSigma)

    # x-rotate and z-rotate the code
    codeParam.worldT = codeWorldT
    codeXTilt = np.radians(rng.uniform(codeXTiltRange[0], codeXTiltRange[1]))
    codeZTilt = np.radians(rng.uniform(codeZTiltRange[0], codeZTiltRange[1]))
    codeParam.worldR = codeWorldR * R.from_euler('XZ', [codeXTilt, codeZTilt])
    for i in range(N):
        camParams[i].codeR = codeWorldR.inv() * camParams[i].worldR
        camParams[i].codeT = codeWorldR.inv().apply(camParams[i].worldT - codeWorldT)

    # project code corners to camera
    camCorners = []
    for i in range(N):
        camCorners.append(project2Cam(codeParam, camParams[i]))

    return (codeParam, camParams, camCorners)



# forward process: given camera parameters, compute image coordinates of the 3 corners
# eulerAngles: xyz angles, transform of the code
# t: 3D positon of the code
def project2Cam(codeParam, camParam, weakPersp=False):
    r = camParam.codeR.inv()
    t = -r.apply(camParam.codeT)
    K = camParam.K
    L = codeParam.codeSize
    cornersCode = np.array([[-L/2, -L/2, 0], [L/2, -L/2, 0], [-L/2, L/2, 0], 
                            [L/2, L/2, 0]])
    cornersCam3D = r.apply(cornersCode) + t[None,:]
#    print_debug('cornersCam3D')
#    print_debug(cornersCam3D)
    if not weakPersp:
        cornersCam = cornersCam3D @ K.T
        cornersCam = cornersCam[:,:2] / cornersCam[:,2:]
    else:
        cornersCam = cornersCam3D[:,:2] * np.array([K[0,0],K[1,1]]) / t[2] \
                + np.array([K[0,2], K[1,2]])
    return cornersCam


# backward process: given corners in the image, compute code pose constraint
def estimateCodePose(cornersCam, res):
    # move to the center
    #ccc = cornersCam - np.array(((0+res[1]-1)/2, (0+res[0]-1)/2))
    q = np.array(((0+res[1]-1)/2, (0+res[0]-1)/2)) - cornersCam[0,:] 
    ccc = cornersCam - cornersCam[0,:]
    # estimate z-rotation
    if ccc[0,0] == ccc[1,0]:
        gamma = np.pi / 2
    else:
        gamma = np.arctan((ccc[1,1]-ccc[0,1]) / (ccc[0,0]-ccc[1,0]))
    
    # undo z-rotation
    rGamma = np.array([[np.cos(gamma), np.sin(gamma)], [-np.sin(gamma), np.cos(gamma)]])
    ccr = ccc @ rGamma
    if ccr[0,0] > ccr[1,0]:
        gamma += np.pi
        ccr = -ccr
    gamma = -gamma
    if gamma > np.pi:
        gamma -= np.pi * 2

    # variables for convenience
    u1 = ccr[0,0]
    u2 = ccr[1,0]
    u3 = ccr[2,0]
    v1 = ccr[0,1]
    v2 = ccr[1,1]
    v3 = ccr[2,1]

    a = u2 ** 2
    b = -(v3 ** 2 + u3 ** 2 + u2 ** 2)
    c = v3 ** 2
    sqrtDelta = np.sqrt(b**2 - 4*a*c)
    root = (-b-sqrtDelta)/2/a
    cosAlpha = np.sqrt(root) # only one root for cosAlpha
    dzf = cosAlpha / v3
    dx = -dzf * q[0] + 0.5
    dy = -dzf * q[1] + 0.5
    # always return the positive alpha, up to the caller to take the opposite
    if u3 > 0:
        alpha1 = np.arccos(cosAlpha)
        beta1 = np.arccos(cosAlpha / v3 * u2)
    else:
        alpha1 = np.arccos(cosAlpha)
        beta1 = -np.arccos(cosAlpha / v3 * u2)
    return np.array((dx, dy, dzf, alpha1, beta1, gamma))


# old version, not using RANSAC
def localizeCodeOld(estCodePoses, camParams, method, initRvec=None, initTvec=None):
    if method == LOCALIZE_P3P:
        N = 4
    else:
        N = len(estCodePoses)
    camCodeProj = np.zeros((N, 1, 2))
    camLoc = np.zeros((N,1,3))
    z = np.array([0.0, 0.0, -1.0])
    ry = R.from_euler('y', np.pi) # rotate the virtual camera by pi
    for i in range(N):
        oppPose = np.array([-estCodePoses[i][3], -estCodePoses[i][4], estCodePoses[i][5]])
        proj = R.from_euler('xyz', oppPose).inv().apply(z)
        proj = ry.apply(proj)
        proj = proj[:2] / proj[2]
        gtProj = camParams[i].codeT
        gtProj = ry.apply(gtProj)
        gtProj = gtProj[:2] / gtProj[2]
#        print('estCodePoses:', estCodePoses[i][3:6])
#        print('gtProj:', gtProj)
#        print('estProj:', proj)
        camCodeProj[i,0,:] = proj
        #camCodeProj[i,0,:] = gtProj
        camLoc[i,0,:] = camParams[i].worldGpsT
        #camLoc[i,0,:] = camParams[i].codeT
        #camLoc[i,0,1] = 0

    camK = np.eye(3)
    useExtrinsicGuess = not (initRvec is None or initTvec is None)

    if method == LOCALIZE_CENTER:
        retval = True
        rvec = np.zeros(3)
        tvec = -np.mean(camLoc[:,0,:], 0)

    elif method == LOCALIZE_P3P:
        retval, rvec, tvec = cv2.solvePnP(camLoc, camCodeProj, camK, None, 
                                          flags=cv2.SOLVEPNP_P3P)
    elif method == LOCALIZE_EPNP:
        retval, rvec, tvec = cv2.solvePnP(camLoc, camCodeProj, camK, None, 
                                          flags=cv2.SOLVEPNP_EPNP)
    elif method == LOCALIZE_ITERATIVE:
        retval, rvec, tvec = cv2.solvePnP(camLoc, camCodeProj, camK, None,
                                          initRvec, initTvec, useExtrinsicGuess,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
    elif method == LOCALIZE_RANSAC:
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(camLoc, camCodeProj, camK, None) 
    elif method == LOCALIZE_ITERATIVE_2D:
        camCodeProj[i,0,1] = 0
        retval, rvec, tvec = cv2.solvePnP(camLoc, camCodeProj, camK, None,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        retval = False

    if not retval:
        estWorldR = None
        estWorldT = None
    else:
        estWorldR = R.from_rotvec(rvec.flatten()).inv()
        estWorldT = -estWorldR.apply(tvec.flatten())
        estWorldR = ry * estWorldR
    return (retval, estWorldR, estWorldT)


def localizeCode(estCodePoses, camParams, method, numIters=100, thresh=1, ignoreY=False):
    N = len(estCodePoses)
    camCodeProj = np.zeros((N, 2))
    camLoc = np.zeros((N, 3))
    z = np.array([0.0, 0.0, -1.0])
    ry = R.from_euler('y', np.pi) # rotate the virtual camera by pi
    for i in range(N):
        oppPose = np.array([-estCodePoses[i][3], -estCodePoses[i][4], estCodePoses[i][5]])
        proj = R.from_euler('xyz', oppPose).inv().apply(z)
        proj = ry.apply(proj)
        proj = proj[:2] / proj[2]
        gtProj = camParams[i].codeT
        gtProj = ry.apply(gtProj)
        gtProj = gtProj[:2] / gtProj[2]
#        print('estCodePoses:', estCodePoses[i][3:6])
#        print('gtProj:', gtProj)
#        print('estProj:', proj)
        camCodeProj[i,:] = proj
        #camCodeProj[i,:] = gtProj
        camLoc[i,:] = camParams[i].worldGpsT
        #camLoc[i,:] = camParams[i].codeT
    if ignoreY:
        camLoc[:,1] = 0

    if method == LOCALIZE_CENTER:
        retval = True
        rvec = np.zeros(3)
        tvec = -np.mean(camLoc[:,:], 0)
    elif method == LOCALIZE_RANSAC:
        retval, rvec, tvec = localizeCodeRansac(camLoc, camCodeProj, numIters, thresh)
    else:
        retval = False

    if not retval:
        estWorldR = None
        estWorldT = None
    else:
        estWorldR = R.from_rotvec(rvec.flatten()).inv()
        estWorldT = -estWorldR.apply(tvec.flatten())
        estWorldR = ry * estWorldR
    return (retval, estWorldR, estWorldT)


def localizeCodeRansac(camLoc, camCodeProj, numIters, thresh):
    N = camLoc.shape[0]
    camK = np.eye(3)
    camCodeProjAlt = -camCodeProj
    bestRvec = None
    bestTvec = None
    bestChoice = None
    bestErr = math.inf
    bestNumInliers = 0
    bestInliers = None
    rng = np.random.default_rng()
    # repeat numIters times
    for i in range(numIters):
        # randomly choose points
        randIndices = rng.choice(N, 4, False)
        randCamLoc = np.ascontiguousarray(camLoc[randIndices]).reshape((4,1,3))
        randCamCodeProj = np.ascontiguousarray(camCodeProj[randIndices]).reshape((4,1,2))
        # randomly choose between the two possible directions
        dirChoice = rng.integers(0, 2, 4)
        for j in range(4):
            if dirChoice[j] == 1:
                randCamCodeProj[j,:,:] = -randCamCodeProj[j,:,:]

        # solve P3P
        retval, rvec, tvec = cv2.solvePnP(randCamLoc, randCamCodeProj, camK, None, 
                                          flags=cv2.SOLVEPNP_P3P)
        if not retval:
            continue

        # compute the reprojection error
        rmat = R.from_rotvec(rvec.flatten()).as_matrix().T
        reproj = camLoc @ rmat + tvec.reshape((1, 3))
        reproj = reproj[:,:2] / reproj[:,2:]
        err = np.sum((camCodeProj-reproj) ** 2, 1)
        errAlt = np.sum((camCodeProjAlt-reproj) ** 2, 1)
        errComb = np.minimum(err, errAlt)
        mse = np.mean(errComb)
        inliers = errComb < thresh
        numInliers = np.count_nonzero(inliers)

        # update best estimate
        #if mse < bestErr:
        if numInliers > bestNumInliers or numInliers == bestNumInliers and mse < bestErr:
            bestRvec = rvec
            bestTvec = tvec
            bestNumInliers = numInliers
            bestInliers = inliers
            bestErr = mse
            bestChoice = err < errAlt

    # refine pose
    camCodeProjInliers = camCodeProjAlt
    camCodeProjInliers[bestChoice] = camCodeProj[bestChoice]
    camCodeProjInliers = np.ascontiguousarray(camCodeProjInliers[inliers])\
            .reshape((-1,1,2))
    camLocInliers = np.ascontiguousarray(camLoc[inliers]).reshape((-1,1,3))
    retval, rvec, tvec = cv2.solvePnP(camLocInliers, camCodeProjInliers, camK, None,
                                      bestRvec, bestTvec, True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    return (retval, rvec, tvec)



def plotCorners(cornersCam, camParam):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(cornersCam[[0,1,3,2,0],0],cornersCam[[0,1,3,2,0],1])
    axs[0].plot(cornersCam[0,0], cornersCam[0,1], 'ro')
    axs[0].plot(cornersCam[1,0], cornersCam[1,1], 'go')
    axs[0].plot(cornersCam[2,0], cornersCam[2,1], 'bo')
    axs[0].set_xlim(0, camParam.res[1])
    axs[0].set_ylim(camParam.res[0], 0)
    axs[0].set_aspect('equal')

    axs[1].plot(cornersCam[[0,1,3,2,0],0],cornersCam[[0,1,3,2,0],1])
    axs[1].plot(cornersCam[0,0], cornersCam[0,1], 'ro')
    axs[1].plot(cornersCam[1,0], cornersCam[1,1], 'go')
    axs[1].plot(cornersCam[2,0], cornersCam[2,1], 'bo')
    #axs[1].xlim(0, res[1])
    #axs[1].ylim(res[0], 0)
    axs[1].invert_yaxis()
    axs[1].set_aspect('equal')
    

def print_debug(*objects, **options):
    if DEBUG:
        print(*objects, **options)


def print_error(mse):
    successMask = mse >= 0
    successCount = np.count_nonzero(successMask)
    successRatio = successCount / mse.size
    rmse = np.sqrt(np.mean(mse[successMask]))
    print('Success Ratio:', successRatio)
    print('RMSE:', rmse)
    

if __name__ == '__main__':
    N = 20 # number of cameras
    M = 100 # number of tests
    errorRansac = np.zeros(M)
    
    for j in range(M):
        # first generate poses of cameras in the (unnormalized) code space
        codeParam, camParams, camCorners = randWorld(N, (10,20), 60, 
                                                     gpsSigma=5,
                                                     yMinRange=(1,1),
                                                     yDiffMax=0,
                                                     lookAtRange=(0,0),
                                                     codeXTiltRange=(0,0),
                                                     codeZTiltRange=(0,0),
                                                     camTiltRange=(0,0),
                                                     codeSizeRange=(1,1),
                                                     camFovRange=(47,47))

        # verify corners
#        for i in range(N):
#            plotCorners(camCorners[i], camParams[i])
#            plt.show()

        # compute relative code poses
        codePoses = []
        for i in range(N):
            estCodePose = estimateCodePose(camCorners[i], camParams[i].res)
            codePoses.append(estCodePose)

            # verify pose 
            gtCodePose = camParams[i].getCodePose()
            #print_debug('Estimated pose:', estCodePose)
            #print_debug('GT pose:', gtCodePose)
            #print_debug('Error pose:', estCodePose - gtCodePose)

        # estimate code location
        retval, estWorldR, estWorldT = localizeCode(codePoses, camParams,
                                                    LOCALIZE_ITERATIVE,
                                                    codeParam.getRvec(),
                                                    codeParam.getTvec())
        if retval:
            mse = (estWorldT[0]-codeParam.worldT[0])**2\
                    + (estWorldT[2]-codeParam.worldT[2])**2
            errorRansac[j] = mse
            if mse > 10 and False:
                print('gtWorldT:', codeParam.worldT)
                print('estWorldT:', estWorldT)
                for i in range(N):
                    print()
                    print('GT pose:', camParams[i].getCodePose(codeParam.codeSize))
                    print('Estimated pose:', codePoses[i])
                input()

        else:
            errorRansac[j] = -1


    print('RANSAC:')
    print_error(errorRansac)
