# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

<<<<<<< HEAD
=======
import numpy as np
from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
>>>>>>> 6d72699 (stream)
import argparse
import time
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
<<<<<<< HEAD

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
=======
from flask import Flask, render_template, Response, send_from_directory
import subprocess


TRACK_ID = None
app = Flask(__name__)

camera = cv2.VideoCapture(0)
cnt = 0


def gen_frames():  # generate frame by frame from camera
    global cnt
    while True:
        # Capture frame-by-framefraf
        success, frame_ = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Convert
            # img = np.expand_dims(frame_, axis=0)
            img = frame_[:, :, ::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)

            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if False else False
            pred = model(img, augment=True, visualize=visualize)
            pred = non_max_suppression(pred, 0.4, 0.5, None, True, max_det=1000)

            for i, det in enumerate(pred):  # detections per image
                anno_list = []
                id_list = []

                # if webcam:  # batch_size >= 1
                #     im0 = frame_.copy()
                # else:
                im0 = frame_.copy()

                annotator = Annotator(im0, line_width=2, pil=not ascii)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string

                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # pass detections to deepsort
                    outputs = deepsort.update(
                        xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):

                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            anno_list.append(output[0:5])
                            id_list.append(id)

                            if (TRACK_ID is None):
                                c = int(cls)  # integer class
                                label = f'{id} {names[c]} {conf:.2f}'
                                annotator.box_label(
                                    bboxes, label, color=colors(c, True))
                            else:
                                if(TRACK_ID == id):
                                    c = int(cls)  # integer class
                                    label = f'{id} {names[c]} {conf:.2f}'
                                    annotator.box_label(
                                        bboxes, label, color=colors(c, True))

                else:
                    deepsort.increment_ages()

            # ret, buffer = cv2.imencode('.jpg', im0)
            # frame = buffer.tobytes()
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


def detect(opt, video):
    out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, half = \
        opt.output, video, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half
    # out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, half = \
    #     opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
    #     opt.save_txt, opt.imgsz, opt.evaluate, opt.half
    # Dataloader
    # if webcam:
    #     view_img = check_imshow()
    #     cudnn.benchmark = True  # set True to speed up constant image size inference
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    #     bs = len(dataset)  # batch_size
    # else:
    ##################
    camera = cv2.VideoCapture(video)
    while True:
        # Capture frame-by-frame
        success, frame_ = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Convert
            # img = np.expand_dims(frame_, axis=0)
            img = frame_[:, :, ::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)

            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
    ##################

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    dt, seen = [0.0, 0.0, 0.0], 0
>>>>>>> 6d72699 (stream)
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            anno_list = []
            id_list = []

            seen += 1
<<<<<<< HEAD
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
=======
            # if webcam:  # batch_size >= 1
            #     p, im0, frame = path[i], im0s[i].copy(), dataset.count
            #     s += f'{i}: '
            # else:
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
>>>>>>> 6d72699 (stream)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
<<<<<<< HEAD
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4
=======
                outputs = deepsort.update(
                    xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
>>>>>>> 6d72699 (stream)

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
<<<<<<< HEAD

                        c = int(cls)  # integer class
                        label = f'{id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                LOGGER.info('No detections')
=======
                        anno_list.append(output[0:5])
                        id_list.append(id)

                        if (TRACK_ID is None):
                            c = int(cls)  # integer class
                            label = f'{id} {names[c]} {conf:.2f}'
                            annotator.box_label(
                                bboxes, label, color=colors(c, True))
                        else:
                            if(TRACK_ID == id):
                                c = int(cls)  # integer class
                                label = f'{id} {names[c]} {conf:.2f}'
                                annotator.box_label(
                                    bboxes, label, color=colors(c, True))

                        # if save_txt:
                        #     # to MOT format
                        #     bbox_left = output[0]
                        #     bbox_top = output[1]
                        #     bbox_w = output[2] - output[0]
                        #     bbox_h = output[3] - output[1]
                        #     # Write MOT compliant results to file
                        #     # label format
                        #     with open(txt_path, 'a') as f:
                        #         f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,
                        #                                        bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

            else:
                deepsort.increment_ages()

            # Print time (inference-only)
            # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
>>>>>>> 6d72699 (stream)

            # Stream results
            im0 = annotator.result()
            if show_vid:
<<<<<<< HEAD
                cv2.imshow(str(p), im0)
=======
                cv2.imshow(p, im0)
                input = [anno_list, id_list]
                cv2.setMouseCallback(p, MouseHandler, input)
>>>>>>> 6d72699 (stream)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
<<<<<<< HEAD
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
=======
            # if save_vid:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'

                vid_writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)
    # @app.route('/video_feed')
    # def video_feed():
    #     """Video streaming route. Put this in the src attribute of an img tag."""
    #     return Response(gen_frames(),
    #                     mimetype='multipart/x-mixed-replace; boundary=frame')

    # @app.route('/')
    # def index():
    #     """Video streaming home page."""
    #     return render_template('index.html')


@app.route('/hls/<string:file_name>')
def stream(file_name):
    video_dir = './hls'
    return send_from_directory(directory=video_dir, path=file_name)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def MouseHandler(event, x, y, flags, list):
    global TRACK_ID

    if event == cv2.EVENT_LBUTTONDOWN:

        if(TRACK_ID is None):
            for anno in list[0]:
                if(x > anno[0] and x < anno[2] and y > anno[1] and y < anno[3]):
                    TRACK_ID = anno[4]
                    print(f"TRACK on ID:{TRACK_ID}")
                    # if track on anyone break the loop
                    break

        else:
            for anno in list[0]:
                if(x > anno[0] and x < anno[2] and y > anno[1] and y < anno[3] and TRACK_ID == anno[4]):
                    TRACK_ID = None
                    print(f"UNTRACK on ID:{TRACK_ID}")

    # unlock any track for right button
    # to prevent tracking object is out of screen
    if event == cv2.EVENT_RBUTTONDOWN:
        TRACK_ID = None
>>>>>>> 6d72699 (stream)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
<<<<<<< HEAD
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
=======
    parser.add_argument('--yolo_weights', nargs='+', type=str,
                        default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_weights', type=str,
                        default='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', help='ckpt.t7 path')
    # file/folder, 0 for webcam
    # parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true',
                        help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--evaluate', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true",
                        help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true',
                        help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true',
                        help='use OpenCV DNN for ONNX inference')
>>>>>>> 6d72699 (stream)
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    out, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, half = \
        opt.output, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half
    # out, source, yolo_weights, deep_sort_weights, show_vid, save_vid, save_txt, imgsz, evaluate, half = \
    #     opt.output, opt.source, opt.yolo_weights, opt.deep_sort_weights, opt.show_vid, opt.save_vid, \
    #     opt.save_txt, opt.imgsz, opt.evaluate, opt.half
    # webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    attempt_download(deep_sort_weights,
                     repo='mikel-brostrom/Yolov5_DeepSort_Pytorch')
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=True)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    # half precision only supported by PyTorch on CUDA
    half &= pt and device.type != 'cpu'
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    # txt_file_name = source.split('/')[-1].split('.')[0]
    # txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup

    video_path = './video'
    check_list = []
    while True:
        video_list = os.listdir(video_path)
        for video in video_list:
            name, ext = os.path.splitext(video)
            if ext == '.ts' and video not in check_list:
                time.sleep(2)
                ret = subprocess.Popen(f'ffmpeg -y -i {os.path.join(video_path, video)} -codec copy -f mp4 video/{name}.mp4')
                # name += '.mp4'
                # time.sleep(1)
                ret.wait()
                detect(opt, os.path.join(video_path, name + '.mp4'))
                subprocess.Popen(f"ffmpeg -y -i {os.path.join(out, (name+'.mp4'))} -codec copy video/{name}.ts")
                check_list.append(video)

                ####
                # app.run(host='0.0.0.0', port=5000)
                # with torch.no_grad():
                #     detect(opt)
