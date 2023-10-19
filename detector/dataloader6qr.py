

from __future__ import print_function
from __future__ import division

import os
import glob

import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from cv2 import imwrite


#######################################################
class DataProvider(Dataset):
    """
    Class for the data provider
    """

    def __init__(self, folder_code, folder_face, folder_back, \
                 imdim=224, \
                 versions=[1, 1], \
                 border=4, \
                 borders=[1, 3], \
                 background_sizes=[0.7, 1.0], \
                 depth_ranges=[500, 10000], \
                 focal_depth_ranges=[500, 10000], \
                 code_sizes=[210, 210], \
                 rotxy=30, \
                 rotz=45, \
                 phone='iphonex', \
                 albedo=[0.5, 1.0], \
                 blacklevel=[0.0, 0.3], \
                 noiseg=0.02, \
                 noisep=0.03, \
                 jpegquality=[80, 95], \
                 mode='train', \
                 datadebug=False):
        
        assert phone == 'iphonex'
        
        # now we simulate a iphone 8 plus
        # every parameter has physical meaning
        # https://www.imaging-resource.com/PRODS/apple-iphone-8-plus/apple-iphone-8-plusA.HTM
        self.h = 3.5
        self.w = 4.8
        self.hpix = 3024
        self.wpix = 4032
        self.unitpix = max(self.hpix / self.h, self.wpix / self.w)
        
        # midofy w
        self.w = self.wpix / self.unitpix
        
        # https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length
        # "Converted focal length into 35mm camera" = 
        # (Diagonal distance of image area in the 35mm camera (43.27mm) / 
        # Diagonal distance of image area on the image sensor of the DSC) * 
        # focal length of the lens of the DSC
        
        # 29 = 43.3 / iphone_diag_length * iphone_focal
        self.focal_eq = 28
        self.diag = np.sqrt(self.h ** 2 + self.w ** 2)
        self.focal = self.focal_eq / 43.3 * self.diag
        
        # f / 1.8
        self.apperature = self.focal / 1.8
        
        # in a word, we fix focal length, apperature
        # change different depth(different blur kernel)
        # different code size(different resolution)
        # different rotation
#         self.testphysicalparam()
        
        """
        split: 'train', 'train_val' or 'test'
        """
        self.mode = mode
        self.datadebug = datadebug
        
        self.imdim = imdim
        self.imszs = background_sizes
        
        self.code_sizes = code_sizes
        self.depth_ranges = depth_ranges
        self.focal_depth_ranges = focal_depth_ranges
        
        self.rotxy = rotxy
        self.rotz = rotz
        
        self.albedo = albedo
        self.blacklevel = blacklevel
        
        self.noiseg = noiseg
        self.noisep = noisep
        self.jpegquality = jpegquality
        
        ##########################
        """
        folder1 = 'C:\\projects\\dataset\\img_align_celeba_png.7z\\img_align_celeba_png.7z\\snapcode';
        folder2 = 'C:\\projects\\dataset\\img_align_celeba_png.7z\\img_align_celeba_png.7z\\img_align_celeba_png';
        folder3 = 'C:\\Users\\wangj\\Downloads\\SUN2012\\SUN2012\\SUN2012\\Images';
        """

        folder1 = folder_code
        folder2 = folder_face
        folder3 = folder_back

        backs = []
        for i1, fol in enumerate(glob.glob('%s/*' % folder3)):
            for i2, fol2 in enumerate(glob.glob('%s/*' % fol)):
#                 print('{} {}'.format(i1, i2))
                backs.extend(glob.glob('%s/*.jpg' % fol2))
        
        backs = sorted(backs)
        ranidx = np.random.permutation(len(backs))
        backs = [backs[i] for i in ranidx]
        
        # Old
        imnum = 10000
        if mode == 'train':
            imnum = 8000
        else:
            imnum = 1000

        # Jian's
        imnum = 10
        if mode == 'train':
            imnum = 100
        else:
            imnum = 1

        self.imnum = imnum
        print('imnum {}'.format(imnum))
        
        self.backs = backs
        print(self.backs[0])
        print(self.backs[-1])
        print('backnum {}'.format(len(self.backs)))
        
        ###############################################################
        import qrcode
        
        self.textidx = '0123456789abcdefghijklmnopqrstuvwxyz'
        
        # similar to snapcode
        self.code_totalsize = 720
        self.border = border
        self.versions = versions
        self.borders = borders
        
        cod_generate = []
        for vert in range(versions[0], versions[1] + 1):
            grid_size = 17 + vert * 4
            grid_len = self.code_totalsize // grid_size
            qr = qrcode.QRCode(
                version=vert,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=grid_len,
                border=self.border,
                )
            cod_generate.append(qr)
        
        self.code_generate = cod_generate
    
    def cal_focal_i(self, depth):
        return 1 / (1 / self.focal - 1 / depth)
    
    def testphysicalparam(self):
        
        re = []
        
        # object size in mm
        for size in [210]:
            # focal depth in m
            for focal_depth in [1.0, 3.0, 5.0, 7.0, 9.0]:
                # depth in m
                for depth in [1.0, 3.0, 5.0, 7.0, 9.0]:
                    # unit mm
                    # focal_depth = 1 # focal infinity
                    f_i = self.cal_focal_i(focal_depth * 1000)
                    
                    # depth = 5 # 5m
                    f_d = 1 / (1 / self.focal - 1 / (depth * 1000))
                    
                    if f_d >= f_i:
                        blur_bernel = self.apperature * (f_d - f_i) / f_d
                    else:
                        blur_bernel = self.apperature * (f_i - f_d) / f_d
                        
                    blur_kernel_pix = self.unitpix * blur_bernel
                    print('focal_depth %d, depth %d, blur kernel %.2f' % (focal_depth * 1000, depth * 1000, blur_kernel_pix))
                    
                    # A4 210 ������������������������������������������������������ 297
                    # assume it is in the center
                    # size = 210
                    # sensor_sz / size = focal / (depth * 1000)
                    sensor_h = size * f_i / (depth * 1000)
                    sensor_pix = self.unitpix * sensor_h
                    print('depth %d, size %d, pix %.2f, relative blur %.2f\n\n' % (depth * 1000, size, sensor_pix, blur_kernel_pix / sensor_pix * 128))
                    
                    re.append([focal_depth, depth, blur_kernel_pix, sensor_pix])
        
        for i in range(len(re)):
            focal_depth, depth, blur_kernel_pix, sensor_pix = re[i]
            print('focal {}, depth {}, blur {}, res {}'.format(focal_depth, depth, blur_kernel_pix, sensor_pix))
    
        print('done')

    def __len__(self):
        return self.imnum

    def __getitem__(self, idx):
        return self.prepare_instance(idx)
    
    def prepare_instance(self, idx):
        """
        Prepare a single instance
        """

        re = {}
        re['valid'] = True
        '''
        imgt, imori, im, code, kp, kpcrop, flag, mask = self.load_im_cam(idx)
        if flag:
            re['imgt'] = imgt
            re['imori'] = imori
            re['im'] = im
            re['code'] = code            
            re['kp'] = kp
            re['kpcrop'] = kpcrop
            re['mask'] = mask
        else:
            re['valid'] = False
        '''
        try:
            imgt, imori, im, code, kp, kpcrop, flag, mask = self.load_im_cam(idx)
            if flag:
                re['imgt'] = imgt
                re['imori'] = imori
                re['im'] = im
                re['code'] = code            
                re['kp'] = kp
                re['kpcrop'] = kpcrop
                re['mask'] = mask
            else:
                re['valid'] = False
        except:
            re['valid'] = False
            return re

        return re

    def load_im_cam(self, i):
        
        ######################################################
        # code generate
        randomtxtidx = np.random.randint(36, size=(3,))
        text = [self.textidx[j] for j in randomtxtidx]
        
        vert = np.random.randint(self.versions[0], self.versions[1] + 1)
        qr = self.code_generate[vert - self.versions[0]]
        qr.clear()
        qr.add_data(text)
        qr.make(fit=True)
        qrcode = qr.make_image(fill_color="black", back_color="white")
        qrcode = np.asarray(qrcode, dtype=np.float32)
        
        if self.datadebug:
            print('max val {} min val {}'.format(np.max(qrcode), np.min(qrcode)))
        
        ###################################################
        # ground truth 18x18 matrix
        
        # grid_size : (17 + vert * 4)
        # grid_len:  self.code_totalsize // grid_size
        # border: 4
        
        boxsz = 17 + vert * 4
        boxlen = self.code_totalsize // boxsz
        codelen = boxsz * boxlen
        border = self.border * boxlen
        h, w = qrcode.shape
        assert h == codelen + border + border
        
        qrcodecontent = qrcode[border:-border, border:-border]
        hc = 17 + vert * 4
        wc = 17 + vert * 4
        code = np.zeros((hc, wc), dtype=np.float32)
        
        for i in range(hc):
            for j in range(wc):
                idxi = i * boxlen + boxlen // 2
                idxj = j * boxlen + boxlen // 2
                if qrcodecontent[idxi, idxj] >= 0.5:
                    code[i, j] = 1
        
        ###############################################
        # 720x720
        imrgb = np.tile(np.expand_dims(qrcodecontent, 2), [1, 1, 3])
        imalpha = np.ones_like(imrgb[:, :, :1])
        immask = imalpha.copy()
        
        h, w, c = imrgb.shape
        assert h == boxsz * boxlen
        
        # add border
        # two borders, first is white, second is black
        bormin = 1 * boxlen
        bormax = 4 * boxlen
        bor = np.random.randint(bormin, bormax + 1)
        
        imrgbbig = np.ones((h + 2 * bor, w + 2 * bor, 3), dtype=np.float32)
        imrgbbig[bor:h + bor, bor:w + bor, :] = imrgb
        imalphabig = np.ones((h + 2 * bor, w + 2 * bor, 1), dtype=np.float32)
        imalphabig[bor:h + bor, bor:w + bor, :] = imalpha
        immaskbig = np.zeros((h + 2 * bor, w + 2 * bor, 1), dtype=np.float32)
        immaskbig[bor:h + bor, bor:w + bor, :] = immask
        
        imrgb = imrgbbig
        imalpha = imalphabig
        immask = immaskbig
        h, w, _ = imrgb.shape
        
        borall = bor
        
        #############################################
        # enlarge boundary by 100 pixels
        # 720 + 2 * bor
        bor = 50
        imrgbbig = np.zeros((h + 2 * bor, w + 2 * bor, 3), dtype=np.float32)
        imrgbbig[bor:h + bor, bor:w + bor, :] = imrgb
        imalphabig = np.zeros((h + 2 * bor, w + 2 * bor, 1), dtype=np.float32)
        imalphabig[bor:h + bor, bor:w + bor, :] = imalpha
        immaskbig = np.zeros((h + 2 * bor, w + 2 * bor, 1), dtype=np.float32)
        immaskbig[bor:h + bor, bor:w + bor, :] = immask
        
        imrgb = imrgbbig
        imalpha = imalphabig
        immask = immaskbig
        h, w, _ = imrgb.shape
        
        borall += bor
        bor = borall
        
        ##############################################
        # ground turth image
        imgt = imrgb * imalpha + 1 - imalpha
        blurkernel = 2 * int(h / self.imdim / 2) + 1
        if blurkernel > 1:
            imgt = cv2.boxFilter(imgt, -1, (blurkernel, blurkernel))
        imgt = cv2.resize(imgt, (self.imdim, self.imdim))
        
        #############################################################
        # 750x750 is too big
        # so we resize it to add face
        a = np.random.rand() + 2
        h = int(h / a)
        w = int(w / a)
        bor = bor / a  # float
        
        # always first box filter then resize
        imrgb = cv2.boxFilter(imrgb, -1, (3, 3))
        imrgb = cv2.resize(imrgb, (w, h))
        
        imalpha = cv2.boxFilter(imalpha, -1, (3, 3))
        imalpha = cv2.resize(imalpha, (w, h))
        
        immask = cv2.boxFilter(immask, -1, (3, 3))
        immask = cv2.resize(immask, (w, h))
        
        if self.datadebug:
            cv2.imshow("code", imgt)
            # cv2.waitKey()
        
        #######################################################
        # add light variation here
        imrgb = imrgb / np.max(imrgb)
        albedo = np.random.rand() * (self.albedo[1] - self.albedo[0]) + self.albedo[0]
        blacklevel = np.random.rand() * (self.blacklevel[1] - self.blacklevel[0]) + self.blacklevel[0]
        imrgb = imrgb * albedo + blacklevel
        
        if np.max(imrgb) > 1.0:
            imrgb = imrgb / np.max(imrgb)
        
        #########################################################
        # delete face adding
        
        ############################################################
        # simulate physical process
        i = -1
        suc = False
        while i < 20:
            i += 1
            
            code_size = np.random.randint(low=self.code_sizes[0], high=self.code_sizes[1] + 1)
            focal_depth = np.random.randint(low=self.focal_depth_ranges[0], high=self.focal_depth_ranges[1] + 1)
            depth = np.random.randint(low=self.depth_ranges[0], high=self.depth_ranges[1] + 1)
            
            # unit: mm
            # f_i is where the camera plane
            f_i = self.cal_focal_i(focal_depth)
            
            # sensor_sz / code_size = focal / (depth * 1000)
            sensor_h = code_size / depth * f_i
            sensor_pix = self.unitpix * sensor_h
            
            # blur size
            f_d = 1 / (1 / self.focal - 1 / depth)
            
            if f_d >= f_i:
                blur_bernel = self.apperature * (f_d - f_i) / f_d
            else:
                blur_bernel = self.apperature * (f_i - f_d) / f_d
            
            blur_kernel_pix = self.unitpix * blur_bernel
            
            relative_blur = blur_kernel_pix / sensor_pix * 128
            if relative_blur <= 13:  # and (sensor_pix > 1.3 * code.shape[0]):
                suc = True
                break
        
        if self.datadebug:
            print('dep {}, focal {}, code_size {} blur size {}, projected size {}'\
                  .format(depth, focal_depth, code_size, relative_blur, sensor_pix))
        
        if not suc:
            print('bad res')
            return None, None, None, None, None, None, False, None
        
        ##################################################################
        # formulate it in OpenGL
        # perspective
        srcTri = np.array([[0, 0], [0, code_size], [code_size, code_size], [code_size, 0]], dtype=np.float32)
        srcTriCenter = srcTri - code_size / 2
        srcTriCenter = np.concatenate((srcTriCenter, np.zeros_like(srcTriCenter[:, :1])), axis=1)
        
        # perspective
        '''
        if self.mode == 'test':
            basedeg = 10
        else:
            basedeg = self.rotxy
        rotxdeg = (np.random.rand() * 2 - 1) * basedeg / 180 * np.pi
        if rotxdeg >= 0:
            rotxdeg += (self.rotxy - basedeg) / 180 * np.pi
        else:
            rotxdeg -= (self.rotxy - basedeg) / 180 * np.pi
        rotydeg = (np.random.rand() * 2 - 1) * basedeg / 180 * np.pi
        if rotydeg >= 0:
            rotydeg += (self.rotxy - basedeg) / 180 * np.pi
        else:
            rotydeg -= (self.rotxy - basedeg) / 180 * np.pi
            '''
        
        rotxdeg = (np.random.rand() * 2 - 1) * self.rotxy / 180 * np.pi
        rotydeg = (np.random.rand() * 2 - 1) * self.rotxy / 180 * np.pi
        rotzdeg = (np.random.rand() * 2 - 1) * self.rotz / 180 * np.pi
        
        rotxmtx = np.array([[1, 0, 0],
                            [0, np.cos(rotxdeg), -np.sin(rotxdeg)],
                            [0, np.sin(rotxdeg), np.cos(rotxdeg)]], dtype=np.float32)
        rotymtx = np.array([[np.cos(rotydeg), 0, np.sin(rotydeg)],
                            [0, 1, 0],
                            [-np.sin(rotydeg), 0, np.cos(rotydeg)]], dtype=np.float32)
        rotzmtx = np.array([[np.cos(rotzdeg), -np.sin(rotzdeg), 0],
                            [np.sin(rotzdeg), np.cos(rotzdeg), 0],
                            [0, 0, 1]], dtype=np.float32)
        rotmtx = np.matmul(rotzmtx, np.matmul(rotymtx, rotxmtx))
        
        dstTriCenter0 = np.matmul(srcTriCenter, rotmtx.T)
        
        # put it far
        dstTriCenter0 = dstTriCenter0 + np.tile(np.array([[0, 0, depth]], dtype=np.float32), [4, 1])
        
        # we should make sure the 4 points are inside the image
        # what's the shift range?
        xmax = self.w / 2 / f_i * depth
        ymax = self.h / 2 / f_i * depth
        
        i = -0
        suc = False
        while i < 10:
            i += 1
            xshift = (np.random.rand() * 2 - 1) * xmax
            yshift = (np.random.rand() * 2 - 1) * ymax
            shift = np.array([xshift, yshift, 0], dtype=np.float32).reshape(1, 3)
            
            dstTriCenter = dstTriCenter0 + shift
            dstTriCenter = dstTriCenter[:, :2] / dstTriCenter[:, 2:3] * f_i
            if np.max(np.abs(dstTriCenter[:, 0])) < self.w / 2 \
            and np.max(np.abs(dstTriCenter[:, 1])) < self.h / 2:
                suc = True
                break
        
        if not suc:
            print('bad shift')
            return None, None, None, None, None, None, False, None
        
        ##########################################################################
        # now, we do it in pixel space
        dstTri = dstTriCenter.copy()
        dstTri[:, 0] += self.w / 2
        dstTri[:, 1] += self.h / 2
        dstTri = dstTri * self.unitpix - 0.5
        
        # calculate mtx
        srcTri = np.array([[bor, bor], [bor, h - 1 - bor], [w - 1 - bor, h - 1 - bor], [w - 1 - bor, bor]], dtype=np.float32)
        warp_mat = cv2.getPerspectiveTransform(srcTri, dstTri).astype(np.float32)
        
        ##########################################################################
        # box filter
        boxkernel = 2 * int(h / sensor_pix / 2) + 1
        if boxkernel > 1:
            imrgb = cv2.boxFilter(imrgb, ddepth=-1, ksize=(boxkernel, boxkernel))
            imalpha = cv2.boxFilter(imalpha, ddepth=-1, ksize=(boxkernel, boxkernel))
            immask = cv2.boxFilter(immask, ddepth=-1, ksize=(boxkernel, boxkernel))
            
        warp_dst = cv2.warpPerspective(imrgb, warp_mat, (self.wpix, self.hpix), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warp_alpha = cv2.warpPerspective(imalpha, warp_mat, (self.wpix, self.hpix), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warp_mask = cv2.warpPerspective(immask, warp_mat, (self.wpix, self.hpix), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        ##################################################################
        # backgroud might be too big
        dstborder = dstTri.copy()
        dstleftup = np.array([0, 0], dtype=np.float32)
        dstrightbot = np.array([self.wpix, self.hpix], dtype=np.float32)
        
        leftup = np.min(dstborder, axis=0)
        rightdown = np.max(dstborder, axis=0)
        if (rightdown[1] - leftup[1]) / self.hpix < self.imszs[0]:
            ratio = self.imszs[0] + (self.imszs[1] - self.imszs[0]) * np.random.rand()
            forelen = rightdown[1] - leftup[1]
            hlen = int(forelen / ratio) + 16
            hcenter = int((rightdown[1] + leftup[1]) / 2)
            hbe = np.random.randint(low=int(rightdown[1] - hlen) + 2, high=int(leftup[1]) - 1)
            if hbe < 0:
                hbe = 0
            hen = hbe + hlen
            if hen > self.hpix:
                hen = self.hpix
            # warp_dst[hbe - 3:hbe + 3, :, 2] = 1.0
            # warp_dst[hen - 3:hen + 3, :, 2] = 1.0
            warp_dst = warp_dst[hbe:hen, :, :]
            warp_alpha = warp_alpha[hbe:hen, :]
            warp_mask = warp_mask[hbe:hen, :]
            dstleftup[1] = hbe;
            dstrightbot[1] = hen;

        if (rightdown[0] - leftup[0]) / self.wpix < self.imszs[0]:
            forelen = rightdown[0] - leftup[0]
            ratio = self.imszs[0] + (self.imszs[1] - self.imszs[0]) * np.random.rand()
            wlen = int(forelen / ratio) + 16
            wcenter = int((rightdown[0] + leftup[0]) / 2)
            wbe = np.random.randint(low=int(rightdown[0] - wlen) + 2, high=int(leftup[0]) - 1)
            if wbe < 0:
                wbe = 0
            wen = wbe + wlen
            if wen > self.wpix:
                wen = self.wpix
            # warp_dst[:, wbe - 3:wbe + 3, 2] = 1.0
            # warp_dst[:, wen - 3:wen + 3, 2] = 1.0
            warp_dst = warp_dst[:, wbe:wen, :]
            warp_alpha = warp_alpha[:, wbe:wen]
            warp_mask = warp_mask[:, wbe:wen]
            dstleftup[0] = wbe;
            dstrightbot[0] = wen;
        
        # change size
        h, w, _ = warp_dst.shape
        
        ###############################################################
        # background
        backidx = np.random.randint(low=0, high=len(self.backs))
        back = cv2.imread(self.backs[backidx], cv2.IMREAD_COLOR)
        if back is None:
            return None, None, None, None, None, None, False, None
        back = back.astype(np.float32) / 255.0
        
        h3, w3, c3 = back.shape
        assert c3 == 3
        
        # background may be smaller
        if h3 < h or w3 < w:
            r1 = h / h3
            r2 = w / w3
            r = 1.1 * max(r1, r2)  # avoid numeric error
            h3 = int(r * h3)
            w3 = int(r * w3)
            back = cv2.resize(back, (w3, h3))
        
        # background is large than image
        hbe = np.random.randint(low=0, high=h3 - h + 1)
        wbe = np.random.randint(low=0, high=w3 - w + 1)
        back = back[hbe:hbe + h, wbe:wbe + w, :]
        
        imgbig = warp_dst * warp_alpha[:, :, None] + back * (1 - warp_alpha[:, :, None])
        
        #############################################################################
        # blur
        blurkernel = int(blur_kernel_pix / 2)
        blurkernel = blurkernel * 2 + 1
        
        if blurkernel > 1:
            imgblur = cv2.blur(imgbig, (blurkernel, blurkernel))
        else:
            imgblur = imgbig.copy()
        
        ########################################################################
        # noise, gaussian + poisson
        noisenp = self.noiseg * np.random.randn(h, w, 3).astype(np.float32)
        noise2np = self.noisep * np.random.randn(h, w, 3).astype(np.float32)
        imgblur = imgblur + imgblur * noise2np + noisenp
        
        # jpeg
        imgblur[imgblur < 0] = 0
        imgblur[imgblur > 1] = 1
        imgblur = (imgblur * 255).astype(np.uint8)
        jpgquality = np.random.randint(low=self.jpegquality[0], high=self.jpegquality[1] + 1)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpgquality]
        result, encimg = cv2.imencode('.jpg', imgblur, encode_param)
        imgblur = cv2.imdecode(encimg, 1)
        
        imgblur = imgblur.astype(np.float32) / 255.0
        
        # normalize
        if True:
            maxval = np.max(imgblur)
            minval = np.min(imgblur)
            imgblur = (imgblur - minval) / (maxval - minval)
        
        ############################################################
        # resize
        ratio = max(h, w) / self.imdim
        boxkernel = int(ratio / 2) * 2 + 1
        
        # boxkernel_h = int(h / self.imdim / 2) * 2 + 1
        # boxkernel_w = int(w / self.imdim / 2) * 2 + 1
        
        imgrz = imgblur.copy()
        if boxkernel > 1:
            imgrz = cv2.boxFilter(imgrz, -1, (boxkernel, boxkernel))
            immask = cv2.boxFilter(warp_mask, -1, (boxkernel, boxkernel))
        imgrz = cv2.resize(imgrz, (self.imdim, self.imdim))
        immask = cv2.resize(immask, (self.imdim, self.imdim))
        
        #################################################################
        if self.datadebug:
            dstbbox = np.max(dstborder, axis=0) - np.min(dstborder, axis=0)
            
            cv2.imshow('gt', imgt)
            cv2.imshow('face', imrgb)
            cv2.imshow('warp', warp_dst)
            cv2.imshow('warp_alpha', warp_alpha)
            cv2.imshow('warp_mask', warp_mask)
            
            imgpoint = imgbig.copy()
            imgpoint = (imgpoint * 255).astype(np.uint8)
            for i, p in enumerate(dstborder):
                x = int(p[0] - dstleftup[0])
                y = int(p[1] - dstleftup[1])
                color = [0, 0, 0]
                color[2 - i] = 255
                cv2.circle(imgpoint, (x, y), 3, color, -1)
            
            imgpoint2 = imgrz.copy()
            imgpoint2 = (imgpoint2 * 255).astype(np.uint8)
            for i, p in enumerate(dstborder):
                x = (p[0] - dstleftup[0] + 0.5) / (dstrightbot[0] - dstleftup[0]) * self.imdim - 0.5
                x = int(x)
                
                y = (p[1] - dstleftup[1] + 0.5) / (dstrightbot[1] - dstleftup[1]) * self.imdim - 0.5
                y = int(y)
                color = [0, 0, 0]
                color[2 - i] = 255
                cv2.circle(imgpoint2, (x, y), 3, color, -1)
            
            cv2.imshow('back', imgpoint)
            cv2.imshow('blur', imgblur)
            cv2.imshow('resize', imgpoint2)
            cv2.waitKey()
            
            # perspective mtx
            # dstbordernorm = (dstborder + 0.5) / self.imdim * 2 - 1
            dstbordernorm = (dstborder + 0.5) / self.unitpix
            dstbordernorm[:, 0] = dstbordernorm[:, 0] - self.w / 2
            dstbordernorm[:, 1] = dstbordernorm[:, 1] - self.h / 2
            
            srcnorm = np.array([[0, 0], [0, self.imdim - 1], \
                                [self.imdim - 1, self.imdim - 1], \
                                [self.imdim - 1, 0]], dtype=np.float32)
            srcnorm = (srcnorm + 0.5) / self.imdim * 2 - 1
            
            warp_matnorm = cv2.getPerspectiveTransform(srcnorm, dstbordernorm / f_i).astype(np.float32)

            print('warperror %.5f' % (np.mean(np.abs(dstbordernorm - dstTriCenter))))
            print('warperror %.6f' % (np.sum(warp_matnorm[:, 0] * warp_matnorm[:, 1])))
            print('warperror %.6f' % (np.sum(warp_matnorm[:, 0] ** 2) - np.sum(warp_matnorm[:, 1] ** 2)))
            
        dstsz = np.array([self.unitpix, f_i], dtype=np.float32)
        dstuv = np.array([self.w / 2, self.h / 2], dtype=np.float32)
        kpcrop = np.stack([dstleftup, dstrightbot, dstsz, dstuv], axis=0)
        return cv2.cvtColor(imgt, cv2.COLOR_BGR2GRAY), \
            imgblur, \
            cv2.cvtColor(imgrz, cv2.COLOR_BGR2GRAY), \
            code, \
            dstborder, kpcrop, True, vert


def collate_fn(batch_list):

    collated = {}
    batch_list = [data for data in batch_list if data['valid']]
    if len(batch_list) == 0:
        return None

    # keys = batch_list[0].keys()
    keys = []
    for key in keys:
        val = [item[key] for item in batch_list]
        collated[key] = val

    viewnum = 1
    keys = ['imgt', 'imori', 'im', 'code', 'kp', 'kpcrop', 'mask']
    for key in keys:
        val = [item[key] for item in batch_list]
        try:
            val = np.stack(val, axis=0)
        except:
            pass
        collated[key] = val

    return collated


def get_data_loaders(f1, f2, f3, \
                     imszs, \
                     depths, focal_depths, code_szs, \
                     rotxy, rotz, \
                     mode, bs, numworkers):

    print('Building dataloaders')

    dataset_train = DataProvider(f1, f2, f3, \
                                 background_sizes=imszs, \
                                 depth_ranges=depths, \
                                 focal_depth_ranges=focal_depths, \
                                 code_sizes=code_szs, \
                                 rotxy=rotxy, \
                                 rotz=rotz, \
                                 mode=mode, datadebug=False)
    
    # always true
    shuffle = True
    if mode == 'train_val' or mode == 'test':
        shuffle = False

    train_loader = DataLoader(dataset_train, batch_size=bs, \
                              shuffle=shuffle, num_workers=numworkers, collate_fn=collate_fn)

    print('train num {}'.format(len(dataset_train)))
    print('train iter'.format(len(train_loader)))

    return train_loader


##############################################
if __name__ == '__main__':
    
#     folder1 = '/home/w.chen/datasets/snapcode'
#     folder2 = '/home/w.chen/datasets/celeb7z'
#     folder3 = '/home/w.chen/datasets/SUN2012/Images'
    
#     folder1 = '/u6/a/wenzheng/remote/dataset/snap/snapcode'
#     folder2 = '/u6/a/wenzheng/remote/dataset/snap/celeb'
#     folder3 = '/u6/a/wenzheng/remote/dataset/snap/SUN2012/Images'
    
    folder1 = '/home/jupyter/snapcode'
    folder2 = '/home/jupyter/img_align_celeba_png'
    folder3 = '/home/jupyter/backgrounds'
    
    imszs = [0.05, 0.5]
#     rootfolder = '/u6/a/wenzheng/remote2/test-realqr'
#     if not os.path.isdir(rootfolder):
#         os.mkdir(rootfolder)
    
    # Jian's
    deps = [ 9000, 20000]
    focal_deps = [ 9000, 10000]
    code_sizes = [53, 55] #[53, 106]
    rotxy_range = 15
    rotz_range = 5
    
    train_loader = get_data_loaders(folder1, folder2, folder3, \
                                            imszs=imszs, \
                                            depths=deps, focal_depths=deps, code_szs=code_sizes, \
                                            rotxy=rotxy_range, rotz=rotz_range, \
                                            mode='train', \
                                            bs=32, numworkers=0)
    
    if True:
        
        fodl = '/disk3/detection_data_lowres'
        if not os.path.isdir(fodl):
            os.mkdir(fodl)
        
        imfodl = '%s/images' % fodl
        labelfodl = '%s/labels' % fodl
        if not os.path.isdir(imfodl):
            os.mkdir(imfodl)
        if not os.path.isdir(labelfodl):
            os.mkdir(labelfodl)
        
        idd = -1
        for _ in range(10):
            for i, data in enumerate(train_loader):
                if data is None:
                    continue
    
                # generate opencv positive object    
                imgtall = data['imgt']
                imoriall = data['imori']
                kpall = data['kp']
                kpcropall = data['kpcrop']
                
                for j, im in enumerate(imoriall):
                    
                    idd += 1
                    print(idd)
                    
                    im = (im * 255).astype(np.uint8)
                    imname = '%s/im-%d.jpg' % (imfodl, idd)
                    cv2.imwrite(imname, im)
                    
                    kp = kpall[j]
                    kpcrop = kpcropall[j]
                    
                    dstborder = kp
                    dstleftup = kpcrop[0]
                    
                    xmin, ymin = np.min(dstborder - dstleftup.reshape(1, 2), axis=0)
                    xmax, ymax = np.max(dstborder - dstleftup.reshape(1, 2), axis=0)
                    xcenter = (xmin + xmax) / 2
                    ycenter = (ymin + ymax) / 2
                    hei = ymax - ymin
                    wid = xmax - xmin
                    
                    '''
                    imdim = 64
                    if xmax >= imdim:
                        xmax = imdim - 1
                    if ymax >= imdim:
                        ymax = imdim - 1
                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0
                    '''
                    
                    labelname = '%s/im-%d.txt' % (labelfodl, idd)
                    fid = open(labelname, 'w')
                    line = '%s/im-%d.jpg 1 %d %d %d %d\n' % (imfodl, idd, xmin, ymin, xmax - xmin, ymax - ymin)
                    # class x_center y_center width height
                    h, w, _ = im.shape
                    xcenter = (xcenter + 0.5) / w
                    ycenter = (ycenter + 0.5) / h
                    hei = hei / h
                    wid = wid / w
                    line = '0 %.5f %.5f %.5f %.5f' % (xcenter, ycenter, wid, hei)
                    fid.write(line)
                    fid.close()
                    '''
                    dstborder = kp
                    dstleftup = kpcrop[0]
                    
                    cv2.imshow('gt', im)                
                    imgpoint = im.copy()
                    for i, p in enumerate(dstborder):
                        x = int(p[0] - dstleftup[0])
                        y = int(p[1] - dstleftup[1])
                        color = [0, 0, 0]
                        color[2 - i] = 255
                        cv2.circle(imgpoint, (x, y), 3, color, -1)
                    
                    cv2.imshow('point', imgpoint)
                    cv2.waitKey()
                    '''

