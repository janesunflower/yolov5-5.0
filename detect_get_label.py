import sys
sys.path.insert(0, './yolov5')

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh, increment_path, save_one_box
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import build_tracker
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np


def detect(opt):
    source, yolo_weights, show_vid, save_vid, save_txt, imgsz, save_crop, save_conf, save_img, view_img, save_oral_img = \
        opt.source, opt.yolo_weights, opt.show_vid, opt.save_vid, \
            opt.save_txt, opt.img_size, opt.save_crop, opt.save_conf, opt.save_img, opt.view_img, opt.save_oral_img
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    project = 'runs/detect'
    name = 'exp'
    exist_ok = False

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    labels_name = 'labels' + '_' + opt.source.split('/')[-1].split('.')[0]
    (save_dir / labels_name if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(opt.device)

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(yolo_weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride) # fly_05.mp4, 640

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()


    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device) # 3, 384,640 img为经过变换后的图，默认new_shape=（640, 640）, im0s为原始尺寸的图片
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) # 1，3，384，640

        ##################################################################################
        # Yolov5 检测 Inference
        t1 = time_sync()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms) # box，conf，cls
        t2 = time_sync()

        ##################################################################################
        # Process detections
        for i, det in enumerate(pred): # i表示n个目标 # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s #img0=(544,960)

            s += '%gx%g ' % img.shape[2:]  # print string
            p = Path(p)
            save_path = str( save_dir / p.name)
            label_name = 'labels' + '_' + p.stem
            txt_path = str(save_dir / label_name) + '/' + ('' if dataset.mode == 'image' else '%06d' % (frame_idx))  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imgoral = im0.copy() if save_oral_img else im0
            imc = im0.copy() if save_crop else im0  # for save_crop

            # draw
            annotator = Annotator(im0, line_width=1, pil=not ascii)

            if det is not None and len(det):
                print("frame_idx:", frame_idx)
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round() # rescale到544,960,3

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf and conf > 0.8 else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    hide_labels = False
                    hide_conf = False
                    if save_crop and conf > 0.8:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            crop_name = 'crops'+ '_' + p.stem
                            save_one_box(xyxy, imc, file=save_dir / crop_name / str(frame_idx) / f'{frame_idx}.jpg', BGR=True)


                ####################################################################
                # box_colors = [(255,0,0), (0,255,0), (0,255,255), (0,0,255), (255,255,0), (255,0,255)]
                for i in range(len(det)):
                    rect = det[i, 0:4]
                    conf = det[i, 4]
                    if conf > 0.8:
                        cv2.rectangle(im0, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])),(255,0,0), 1, cv2.LINE_AA)
                        cv2.putText(im0, str(i), (int(rect[0]), int(rect[1])+2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            # im0 = annotator.result()
            if show_vid:
                cv2.imshow('img', im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # 保存原始图片
            if save_oral_img:
                oral_name = 'oral_imgs'+ '_' + p.stem
                oral_path = str(save_dir / oral_name)
                if not os.path.exists(oral_path):
                    os.makedirs(oral_path)
                cv2.imwrite(oral_path + '/' + '%06d.jpg' % (frame_idx), imgoral)

            # 保存标注了检测框的图片
            if save_img:
                bbox_name = 'bbox_imgs' + '_' + p.stem
                p_path = str(save_dir / bbox_name)
                if not os.path.exists(p_path):
                    os.makedirs(p_path)
                cv2.imwrite(p_path + '/' + '%06d.jpg' % (frame_idx), im0)

            # Save results (image with detections)
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
                        save_path += '.mp4'


                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    print('Done. (%.3fs)' % (time.time() - t0))


'''
生成对应视频的imgs，修改视频名称即可
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, help='model.pt path', default = 'weights/crowdhuman_yolov5m.pt') # DJI_tracking.pt default='yolov5/weights/yolov5s.pt',
    parser.add_argument('--source', type=str, default='/home/jarvis/codes/Yolov5_DeepSort_Pytorch/videos/fly_01.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--save_crop', type = bool, default=True)
    parser.add_argument('--save_conf', type = bool, default=False)
    parser.add_argument('--save_img', type=bool, default=True)
    parser.add_argument('--save_oral_img', type=bool, default=True)
    parser.add_argument('--view_img', type=bool, default=True)
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results', default=False) # 可视化视频
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results', default=True)
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt', default=True)
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17', default='0')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    with torch.no_grad():
        detect(args)
