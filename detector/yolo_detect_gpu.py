import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import cv2
import natsort
import shutil
from shutil import copyfile

print('ONNX_EXPORT:', ONNX_EXPORT)

def detect(save_txt=True, save_img=False):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img = opt.output, opt.source, opt.weights, opt.half, opt.view_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        tmp = torch.load(weights, map_location=device)
        model.load_state_dict(tmp['model'])
        #model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights_qr2/export.onnx', verbose=True)
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    t0 = time.time()
    for path, ims, im0s, vid_cap in dataset:
        t = time.time()
        
        preds = []
        print('im0s.shape', im0s.shape)
        # Get detections
        for img in ims:
            print('img.shape', img.shape)
            img = torch.from_numpy(img).to(device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred, _ = model(img)
            preds.append(pred)
            # print('preds', preds)
        
        res_nums = [pred.shape[1] for pred in preds]
        res_shapes = [(im.shape[1:], im0s.shape) for im in ims]
        pred = torch.cat(preds, dim=1)

        if opt.half:
            pred = pred.float()

        for i, det in enumerate(non_max_suppression_multires(pred, res_nums, res_shapes, opt.conf_thres, opt.nms_thres)):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s
                print('p', p)

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                print('plotting the predicted box')
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                # Write results
                for *xyxy, conf, _, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + '.txt', 'a') as file:
                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (classes[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.namedWindow("output", cv2.WINDOW_NORMAL)
                cv2.imshow("output", im0)
                # cv2.imshow(p, im0)
                cv2.waitKey(0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        # if platform == 'darwin':  # MacOS
        #     os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)\n\n' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3_qr.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco_qr.data', help='coco.data file path')
#     parser.add_argument('--weights', type=str, default='weightsqr/best.pt', help='path to weights file')
#     parser.add_argument('--source', type=str, default='/u6/a/wenzheng/remote2/Snap_project/QRrealcap/QR-code-video-data/rotation/rotation_res1280x720_focus@1maaaaa', help='source')  # input file/folder, 0 for webcam
#     parser.add_argument('--output', type=str, default='/u6/a/wenzheng/remote2/Snap_project/QRrealcap/QR-code-video-data/rotation/rotation_res1280x720_focus@1m-detectaaaaa', help='output folder')  # output folder

    parser.add_argument('--weights', type=str, default='weights_qr2/best_trainbysmallcode.pt', help='path to weights file')
    # parser.add_argument('--weights', type=str, default='weights_qr2/best_large_code.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='inference_input', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference_output', help='output folder')  # output folder
    
#    parser.add_argument('--img-size', type=int, default=[5760, 3840, 1920, 960, 480], help='inference size (pixels)')
#    parser.add_argument('--img-size', type=int, default=[3392], help='inference size (pixels)') # must be multipliers of 32
    parser.add_argument('--img-size', type=int, default=[3200, 1600, 800], help='inference size (pixels)') # must be multipliers of 32
#    parser.add_argument('--img-size', type=int, default=[2560, 1280, 640], help='inference size (pixels)')
    # parser.add_argument('--img-size', type=int, default=[256, 224, 192, 160, 128], help='inference size (pixels)')

    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='0', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_false', help='display results')
    opt = parser.parse_args()
    
    opt.view_img = False
    print(opt)

    with torch.no_grad():
        if False:
            # one round
            detect()
        else:
            # several rounds
            # remove the detected codes' area, and detect again
            if os.path.exists('inference_input_intermediate'):
                shutil.rmtree('inference_input_intermediate')
            os.makedirs('inference_input_intermediate')
            if os.path.exists('inference_output'):
                shutil.rmtree('inference_output')
            os.makedirs('inference_output')
            if os.path.exists('inference_output_intermediate'):
                shutil.rmtree('inference_output_intermediate')
            os.makedirs('inference_output_intermediate')

            k_round = 0
            # Sizhuo: quick fix
            #while True:
            while k_round < 1:
                if k_round == 0:
                    filenames = glob.glob('inference_input/' + '*.*')
                    opt.source = 'inference_input'
                    opt.output = 'inference_output_intermediate'
                    # Sizhuo: is this necessary? ignore orientation?
                    # for filename in filenames:
                        # im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                        # cv2.imwrite(filename, im)
                    for filename in filenames:
                        im = cv2.imread(filename, cv2.IMREAD_COLOR)
                        if im.shape[0] > im.shape[1]:
                            im = np.rot90(im)
                            cv2.imwrite(filename, im)
                else:
                    filenames = glob.glob('inference_input_intermediate/' + '*.*')
                    opt.source = 'inference_input_intermediate'
                    opt.output = 'inference_output_intermediate'
                filenames = natsort.natsorted(filenames)
                print(filenames)
                if len(filenames) == 0:
                    break

                detect()

                for filename in filenames:
                    im = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
                    if opt.source == 'inference_input_intermediate':
                        os.remove(filename)

                    figurename = os.path.basename(filename)

                    try:
                        f = open('inference_output_intermediate/' + figurename + '.txt', "r")
                    except:
                        continue
                    detectionResults = []
                    for line in f.readlines():
                        detectionResults.append(line.split(' '))
                    f.close()
                    # print(detectionResults)
                    # print(len(detectionResults))

                    detectionResults = np.array(detectionResults)
                    # print(detectionResults)
                    # print(detectionResults.shape)

                    if not os.path.exists('inference_output/' + figurename):
                        os.makedirs('inference_output/' + figurename)
                    imresult = cv2.imread('inference_output_intermediate/' + figurename, cv2.IMREAD_UNCHANGED)
                    cv2.imwrite('inference_output/' + figurename + '/' + os.path.splitext(figurename)[0] + '_' + str(
                        k_round) + '.png', imresult)
                    copyfile('inference_output_intermediate/' + figurename + '.txt',
                             'inference_output/' + figurename + '/' + os.path.splitext(figurename)[0] + '_' + str(
                                 k_round) + '.txt')

                    cnt = 0
                    for loc_code in detectionResults:
                        x0 = detectionResults[cnt][0].astype(int)
                        y0 = detectionResults[cnt][1].astype(int)
                        x1 = detectionResults[cnt][2].astype(int)
                        y1 = detectionResults[cnt][3].astype(int)
                        if len(im.shape) == 3:
                            print('3 channels')
                            im[y0:y1, x0:x1, :] = 0
                        else:
                            im[y0:y1, x0:x1] = 0
                        cnt += 1

                    cv2.imwrite('inference_input_intermediate/' + figurename, im)

                k_round += 1