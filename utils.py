import os
import shutil
from pathlib import Path
import subprocess
import glob
from time import perf_counter
import numpy as np
import cv2
from natsort import natsorted
import LBPCode
import qrcode 

# get i-th bit
# pos: 0...7, 0 is the least significant bit
def getBit(byte, pos):
    return byte >> pos & 1


def calcBitError(byte0, byte1, bitLength=8):
    if byte0 is None or byte1 is None:
        return bitLength
    return sum(getBit(byte0, i) != getBit(byte1, i) for i in range(bitLength))


alpha_num = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:'
def genRandomQRCode(version, length, mode='byte', rng=None):
    qr = qrcode.QRCode(
            version=version,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=0
    )
    if mode == 'byte':
        if rng is None:
            data = np.random.bytes(length)
        else:
            data = rng.bytes(length)
    elif mode == 'alphanumeric':
        if rng is None:
            dataInd = np.random.randint(len(alpha_num), size=length)
        else:
            dataInd = rng.integers(len(alpha_num), size=length)
        data = ''.join([alpha_num[i] for i in dataInd])
    elif mode == 'numeric':
        raise Exception('Not implemented yet')
    elif mode == 'kanji':
        raise Exception('Not implemented yet')
    else:
        raise Exception('Not supported mode: ' + str(mode))
    qr.add_data(data)
    qr.make(fit=False)
    qrImg = qr.make_image(fill_color='black', back_color='white')
    qrArr = np.asarray(qrImg).astype(np.uint8)
    return qrArr


def print_log(message, outputFile=None, end='\n'):
    print(message, end=end)
    if not outputFile is None:
        print(message, end=end, file=outputFile)

def run_detect(src_dir, args=[], python_cmd='python'):
    # Record current working directory so we can come back later
    cwd = Path.cwd()
    # Prepare the directory structure
    src_dir = Path(src_dir).resolve()
    farqr_dir = Path(__file__).resolve().parent
    detector_dir = farqr_dir.joinpath('detector')
    input_dir = Path(detector_dir).joinpath('inference_input').resolve()
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir()
    for filename in glob.glob(str(src_dir.joinpath('data/*.*'))):
        shutil.copy(filename, detector_dir.joinpath('inference_input'))

    # Change to detector dir and run detector
    os.chdir(detector_dir)
    subprocess.run([python_cmd, 'yolo_detect_gpu.py'] + args)
    subprocess.run([python_cmd, 'crop.py', str(src_dir.joinpath('data'))])

    # Copy data back to source dir
    output_dir = src_dir.joinpath('inference_output')
    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(detector_dir.joinpath('inference_output'), output_dir)
    crop_dir = src_dir.joinpath('crop')
    if crop_dir.exists():
        shutil.rmtree(crop_dir)
    shutil.copytree(detector_dir.joinpath('crop'), crop_dir)
    
    # Return to cwd
    os.chdir(cwd)

def run_align(src_dir, python_cmd='python'):
    # Record current working directory so we can come back later
    cwd = Path.cwd()
    # Prepare the directory structure
    farqr_dir = Path(__file__).resolve().parent
    aligner_dir = farqr_dir.joinpath('aligner')
    src_dir_abs = str(Path(src_dir).absolute())

    # Change to aligner dir and run aligner
    os.chdir(aligner_dir)
    subprocess.run([python_cmd, 'aligner.py', src_dir_abs])

    # Return to cwd
    os.chdir(cwd)
    
def run_decode(src_dir, code_database, code_shape=(21,21)):
    # Set parameters
    num_levels = 2
    threshs = [0.5] * num_levels
    # Set paths
    decode_dir = Path(src_dir)
    img_dir = Path(src_dir).joinpath('data')
    align_dir = Path(src_dir).joinpath('corner')
    detect_dir = Path(src_dir).joinpath('crop')
    img_ids = [p.stem for p in img_dir.glob('?*.*')]
    img_ids = natsorted(img_ids)
    print(img_ids)
    # Decode
    with open(decode_dir.joinpath('results.txt'), 'w') as f:
        for img_id in img_ids:
            t1 = perf_counter()
            print_log('IMG ID: ' + img_id, f)
            box_filepath = detect_dir.joinpath(img_id + '.txt')
            if not box_filepath.exists():
                print_log('Detection file not found. Skip...', f)
                continue
            boxes = []
            with open(box_filepath, 'r') as box_file:
                for line in box_file:
                    box = [int(s) for s in line.strip().split(' ')]
                    boxes.append(box)
            # process each crop, record successful crops
            with open(decode_dir.joinpath(img_id + '.txt'), 'w') as img_result_file:
                for l in range(len(boxes)):
                    # resample partial code
                    crop_filepath = detect_dir.joinpath('%s_%d.png' % (img_id, l))
                    img = cv2.imread(str(crop_filepath), cv2.IMREAD_GRAYSCALE)
                    corner_filepath = align_dir.joinpath('%s_%d_result.txt' % (img_id, l))
                    corners = []
                    with open(corner_filepath, 'r') as corner_file:
                        for line in corner_file:
                            corner = [float(s) for s in line.strip().split(' ')]
                            corners.append(corner)
                    corners = np.array(corners)
                    H = LBPCode.estimateHomography(corners, code_shape[0], 3)
                    code_partial = LBPCode.LBPCodePartial(img, H, code_shape, num_levels,
                                                         threshs)
                    code_partial.extractCode()
                    # matching
                    match_success, match_idx, match_ratio = code_partial.matchDatabase(0, code_database, 'l2')
                    print_log('Matched code: %d' % match_idx, f)
                    print_log('Matched ratio: %f' % match_ratio, f)
                    print(' '.join([str(s) for s in boxes[l]]) + ' %d %f' % (match_idx, match_ratio), file=img_result_file)

            t2 = perf_counter()
            print_log('Time for processing image %s: %f\n' % (img_id,
                                                             t2-t1), f)

def run_decode_opencv(src_dir):
    # Set parameters
    num_levels = 2
    threshs = [0.5] * num_levels
    # Set paths
    decode_dir = Path(src_dir)
    img_dir = Path(src_dir).joinpath('data')
    align_dir = Path(src_dir).joinpath('corner')
    detect_dir = Path(src_dir).joinpath('crop')
    img_files = [p.name for p in img_dir.glob('?*.*')]
    img_files = natsorted(img_files)
    print(img_files)
    # Initialize detectors
    farqr_dir = Path(__file__).resolve().parent
    wechat_model_dir = farqr_dir.joinpath('wechat_models')
    qr_detector = cv2.QRCodeDetector()
    wc_detector = cv2.wechat_qrcode_WeChatQRCode(
        str(wechat_model_dir.joinpath('detect.prototxt')), 
        str(wechat_model_dir.joinpath('detect.caffemodel')), 
        str(wechat_model_dir.joinpath('sr.prototxt')),
        str(wechat_model_dir.joinpath('sr.caffemodel'))
    )

    with open(decode_dir.joinpath('cv_results.txt'), 'w') as f:
        for img_file in img_files:
            img_id = Path(img_file).stem
            t1 = perf_counter()
            print_log('IMG ID: ' + img_id, f)
            # Use OpenCV detect multi
            print_log('OpenCV Decode Multi: ', f)
            im = cv2.imread(str(img_dir.joinpath(img_file)), cv2.IMREAD_COLOR)
            # print_log(im.shape, f)
            # im = np.rot90(im, 3)
            try:
                retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(im)
                print_log(str(decoded_info), f)
            except BaseException as err:
                print_log('Error!', f)
                print_log(str(err), f)
            # Use WeChat to detect multi
            print_log('Wechat Decode Multi: ', f)
            messages, points = wc_detector.detectAndDecode(im)
            print_log(messages, f)
            # Use OpenCV detect on each FarQR detection
            print_log('OpenCV Decode on FarQR detection: ', f)
            box_filepath = detect_dir.joinpath(img_id + '.txt')
            if not box_filepath.exists():
                print_log('Detection file not found. Skip...', f)
                continue
            boxes = []
            with open(box_filepath, 'r') as box_file:
                for line in box_file:
                    box = [int(s) for s in line.strip().split(' ')]
                    boxes.append(box)
            # process each crop, record successful crops
            for l in range(len(boxes)):
                # resample partial code
                crop_filepath = detect_dir.joinpath('%s_%d.png' % (img_id, l))
                img = cv2.imread(str(crop_filepath), cv2.IMREAD_COLOR)
                success, corners = qr_detector.detect(img)
                if success:
                    try:
                        message, straight = qr_detector.decode(img, corners)
                        if len(message) > 0:
                            print_log('Success! %d = %s' % (l, message), f)
                    except BaseException as err:
                        print_log('Error!', f)
                        print_log(str(err), f)
                    

            t2 = perf_counter()
            print_log('Time for processing image %s: %f\n' % (img_id,
                                                             t2-t1), f)

# Tests
if __name__ == '__main__':
    print("==================== Test calcBitsError() ====================")  
    testBytes = [167, 179]
    for b in testBytes:
        print(bin(b))
        for i in range(8):
            print(getBit(b, i),)
        print()
    print("Bits Error: %d" % calcBitError(testBytes[0], testBytes[1]))
