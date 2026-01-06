"""
RealSense D435iF + YOLOv5 实时物体检测与 DexGraspNet 抓取文件匹配示例

"""



import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import os
import warnings
import random


warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*torch.cuda.amp.autocast.*"
)

# =============================
# YOLOv5 模型加载
# =============================
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True
)
model.conf = 0.5  # 置信度阈值

# =============================
# YOLO → DexGraspNet 映射表
# =============================
#keyboard chair mouse person tv banana cup
YOLO_TO_DEXGRASP = {
    "banana": "banana",
    "bottle": "bottle",
    "cup": "cup",
    "apple": "ddg-gd_apple",
}

DEXGRASP_ROOT = "/home/chark/DexGraspNet/data/dataset/dexgraspnet"  # 存 grasp npy 的目录

# -------------------------
# DexGraspNet 物体寻找函数
# -------------------------
            
def find_object_in_dexgrasp(class_name):
    if class_name in YOLO_TO_DEXGRASP:
        grasp_prefix = YOLO_TO_DEXGRASP[class_name]
        print(f"匹配 DexGraspNet 物体: {grasp_prefix}")

        # 示例：找一个现成 grasp 文件
        # grasp_file = os.path.join(
        #     DEXGRASP_ROOT,
        #     f"{grasp_prefix}_poisson_002.npy"
        # )

        files_with_name = [name for name in os.listdir(DEXGRASP_ROOT) if class_name in name]
        print("包含该类别的文件有：", len(files_with_name))
        index =random.randint(0,len(files_with_name)-1)

        if files_with_name:
            print("/n-----------------------------")
            print(f"✔ 找到抓取文件: {files_with_name[index]}")
        else:
            print("✘ 未找到对应抓取文件")
    else:
        print("DexGraspNet 中无该类别")


#只有新物体进入视野才打印抓取信息

POSITION_THRESHOLD = 0.50  # 5 dm
known_objects = []

def is_new_object(class_name, position, known_objects, threshold=0.05):
    for obj in known_objects:
        if obj["class"] == class_name:
            dist = np.linalg.norm(position - obj["position"])
            if dist < threshold:
                return False
    return True




# =============================
# 1.RealSense 初始化
# =============================
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline.start(config)

align = rs.align(rs.stream.color)

print(">>> D435iF + YOLOv5 启动成功，按 q 退出")

# =============================
# 2.主循环
# =============================
try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # -------------------------
        # YOLOv5 推理
        # -------------------------
        results = model(color_image)
        
        detections = results.pred[0]

        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        for *xyxy, conf, cls in detections:
            cls_id = int(cls)
            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(int, xyxy)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # -------------------------
            # 深度 → 3D 反投影
            # -------------------------
            depth = depth_frame.get_distance(cx, cy)
            if depth <= 0:
                continue

            X, Y, Z = rs.rs2_deproject_pixel_to_point(
                intrinsics, [cx, cy], depth
            )
            position = np.array([X, Y, Z])

            #其实不用区分第一次检测和后续检测，也能实现相同功能
            if is_new_object(class_name, position, known_objects):
                    print("\n==============================")
                    print(f"🆕 新物体进入视野")
                    print(f"类别: {class_name}")
                    print(f"置信度: {conf:.2f}")
                    print(f"3D位置: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")
                    find_object_in_dexgrasp(class_name)

                    known_objects.append({
                        "class": class_name,
                        "position": position
                    })
            
            # -------------------------
            # 可视化
            # -------------------------
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                color_image,
                class_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        

        cv2.imshow("YOLOv5 + D435iF", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
