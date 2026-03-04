#!/bin/bash
# train/train_yolo.sh
# make sure you have ultralytics installed: pip install ultralytics

yolo detect train model=yolov8n.pt data=../data/merged/data.yaml imgsz=640 epochs=50 batch=16 name=crowd_merged
