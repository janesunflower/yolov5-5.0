#!/usr/bin/env bash

python train.py \
--weights weights/yolov5s.pt \
--cfg models/yolov5s.yaml \
--data data/dji_track.yaml\
