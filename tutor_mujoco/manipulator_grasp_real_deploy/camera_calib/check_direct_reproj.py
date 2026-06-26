"""
直接对每帧图像 solvePnP 求重投影误差（不经过手眼链和机械臂位姿）
目的：验证相机内参和畸变系数在这些图像上的适用性
"""

import cv2
import numpy as np
import os

# ==================== 参数设置 ====================
data_dir = "merged_reproj_test"   # 存放 jpg 图片的文件夹（不需要 pose.npy）

# 棋盘格规格
pattern_size = (8, 11)   # 内角点数
square_size = 0.015      # 棋盘格格子边长，单位：米（15 mm）

# ==================== 加载内参 ====================
K = np.load("K.npy")
dist = np.load("dist.npy")

# ==================== 生成棋盘格物理点 ====================
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# ==================== 遍历所有图片 ====================
image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".jpg")])
print(f"共找到 {len(image_paths)} 张图片\n")

errors = []

for i, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        print(f"Frame {i}: 未检测到棋盘格")
        continue

    # 亚像素精化
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # 直接 solvePnP 求这一帧的相机外参（棋盘格 -> 相机）
    retval, rvec, tvec = cv2.solvePnP(objp, corners, K, dist)

    # 用求出的位姿重投影
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    corners = corners.reshape(-1, 2)

    # 计算均方根误差
    err = np.sqrt(np.mean(np.sum((corners - proj) ** 2, axis=1)))
    errors.append(err)
    print(f"Frame {i}: RMSE = {err:.4f} pixels")

# ==================== 汇总 ====================
if errors:
    print(f"\n平均 RMSE: {np.mean(errors):.4f} pixels")
    print(f"最大 RMSE: {np.max(errors):.4f} pixels")
    print(f"最小 RMSE: {np.min(errors):.4f} pixels")
else:
    print("无有效检测帧。")