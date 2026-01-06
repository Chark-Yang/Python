"""
D435iF + YOLOv5 简单示例

"""

import pyrealsense2 as rs
import cv2
import torch
import numpy as np

# YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    img = np.asanyarray(color_frame.get_data())
    results = model(img)

    cv2.imshow('YOLO + D435iF', results.render()[0])
    if cv2.waitKey(1) == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()
