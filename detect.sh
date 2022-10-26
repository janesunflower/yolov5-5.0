#!/usr/bin/env bash

python detect.py                                                              \
--weights /home/jarvis/AProjects/02Detect/weights/yolov5-5.0/yolov5s.pt                                                  \
--source /home/jarvis/AProjects/data/tracking/DJI_tracking/1119/imgs/test8/000000.jpg    \
--conf-thres 0.7                                                                         \
--iou-thres 0.45 \
--classes 0 \
--view-img