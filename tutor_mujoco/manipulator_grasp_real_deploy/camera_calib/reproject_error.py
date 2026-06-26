"""
计算重投影误差,单帧得到T_board2end

流程：定义重投影图片文件夹路径 -> 定义参考帧序号,计算T_board2end -> 对文件夹内所有图片计算重投影误差

原理:用cv2图像检测一次,检测到棋盘格每个角点的位置,再通过标定的T_cam2base和T_board2end,计算出每个角点的理论位置；计算误差

注意修改3个地方: 重投影棋盘格图片路径,参考帧序号,手眼标定得到的T_cam2base

"""



import cv2
import numpy as np

from scipy.spatial.transform import Rotation as R

import os

# 将tcp长度为6的数组转换成4x4 齐次变换矩阵
def tcp_to_matrix(tcp):
    """
    tcp: [x, y, z, rx, ry, rz]  长度 6 的数组/列表
         单位：米，弧度
         旋转顺序:XYZ (rx, ry, rz)
    返回:4x4 齐次变换矩阵 T_end2base
    """
    x, y, z, rx, ry, rz = tcp
    # 注意 scipy 的 from_euler 参数顺序是 (seq, angles)，seq 是旋转轴顺序，angles 对应顺序
    # 对于 XYZ，要先绕 X 再绕 Y 再绕 Z，所以 seq='xyz'，角度列表为 [rx, ry, rz]
    # 判断方式，如果只旋转Rz0.5rad，rx和ry都没有变化，那么旋转顺序就是xyz,就是内旋，尝试过外旋，结果不对
    rot = R.from_euler('xyz', [rx, ry, rz], degrees=False).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = np.array([x, y, z]) / 1000.0
    return T



# data_dir = "260614_data2_reproj"   # 重投影照片的文件夹，记得修改
data_dir = "merged_reproj" 

image_paths = []
T_end2base_list = []

for fname in sorted(os.listdir(data_dir)):
    if fname.endswith(".jpg"):
        idx_str = fname.split(".")[0]        # "000"
        pose_file = f"{idx_str}_pose.npy"
        pose_path = os.path.join(data_dir, pose_file)
        img_path = os.path.join(data_dir, fname)
        
        if os.path.exists(pose_path):
            tcp = np.load(pose_path)          # 形状 (6,)
            T = tcp_to_matrix(tcp)            # 4x4
            image_paths.append(img_path)
            T_end2base_list.append(T)

print(f"Loaded {len(image_paths)} frames.")


# 已知量手眼标定矩阵，记得核对
# 260614_data数据集的手眼标定结果
# R_cam2base = np.array([[-0.93580742, -0.25455773, 0.24385412],
#                        [-0.35227523,  0.70064919, -0.62047794],
#                        [-0.01290873, -0.66655163, -0.7453471 ]])

# t_cam2base = np.array([[0.39877877],
#                         [0.0994587 ],
#                         [0.64403215]])

# 260626_data数据集的手眼标定结果，260614_data数据集内参标定
# R_cam2base = np.array([[-0.89300452, -0.27717414,  0.35456653],
#                        [-0.44937812,  0.59212252, -0.6689172 ],
#                        [-0.02454028, -0.75668053, -0.65332408]])

# t_cam2base = np.array([[0.32463971],
#                         [0.18871528],
#                         [0.68643789]])

# # 260626_data数据集的手眼标定结果，260626_data数据集内参标定
# R_cam2base = np.array([[-0.89345233, -0.27773958,  0.35299244],
#                         [-0.44837833,  0.59779942, -0.66452444],
#                         [-0.02645394, -0.75199507, -0.65863769]])

# t_cam2base = np.array([[0.325015],
#                         [0.18943109],
#                         [0.68771726]])

# merged数据集的手眼标定结果，merged数据集内参标定
R_cam2base = np.array([[-0.9377583,  -0.26024247,  0.22996355],
                       [-0.34524681,  0.62688479, -0.6984412 ],
                       [ 0.03760341, -0.73436321, -0.67771429]])

t_cam2base = np.array([[0.33575064],
                        [0.16561175],
                        [0.6566156 ]])

T_cam2base = np.eye(4)
T_cam2base[:3, :3] = R_cam2base
T_cam2base[:3, 3] = t_cam2base.flatten()   # 4x4 矩阵，相机在基坐标系下  


K = np.load("K.npy")   # 相机内参 3x3
dist = np.load("dist.npy")  # 畸变系数
 

# 标定板特征点（以棋盘格为例）
pattern_size = (8, 11)        # 内角点数
square_size = 0.015           # 格子大小 (m)
# objp存储所有角点 在标定板自身坐标系中的三维坐标
# np.mgrid[0:8, 0:6] 产生两个 8×6 的网格矩阵，分别记录每个点的列索引和行索引，最后乘以格子大小得到实际坐标
# objp 形状为 (48, 3)，第三列全为 0，表示标定板平面。
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size

# 1. 求 T_board_end（选用一帧参考帧）
ref_idx =2 

T_end2base_ref = T_end2base_list[ref_idx]   # 参考帧的末端位姿
img_ref = cv2.imread(image_paths[ref_idx])
gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
# ret：是否成功检测到
# corners_ref：检测到的角点图像坐标（精度为像素级）
ret, corners_ref = cv2.findChessboardCorners(gray_ref, pattern_size, None)
# 如果成功检测到
if ret:
    corners_ref = cv2.cornerSubPix(gray_ref, corners_ref, (11,11), (-1,-1), 
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    retval, rvec, tvec = cv2.solvePnP(objp, corners_ref, K, dist)
    # T_board2cam_ref 即 从标定板坐标系到相机坐标系的 4×4 齐次变换矩阵，以cam为基准
    T_board2cam_ref = np.eye(4)
    T_board2cam_ref[:3,:3] = cv2.Rodrigues(rvec)[0]
    T_board2cam_ref[:3, 3] = tvec.ravel()
    
    T_board2base = T_cam2base @ T_board2cam_ref
    T_board2end = np.linalg.inv(T_end2base_ref) @ T_board2base
# 否则检测失败
else:
    raise Exception("参考帧检测失败")

# 2. 对所有验证帧计算重投影误差
errors = []
for i, (img_path, T_end2base_i) in enumerate(zip(image_paths, T_end2base_list)):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners_detect = cv2.findChessboardCorners(gray, pattern_size, None)
    if not ret:
        continue
    # 检测棋盘格角点并亚像素化
    corners_detect = cv2.cornerSubPix(gray, corners_detect, (11,11), (-1,-1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    
    # 计算标定板在相机下的位姿
    # T_board2base有2种计算方式，一种是T_cam2base @ T_board2cam_i，另一种是T_end2base_i @ T_board2end
    T_board2cam_i = np.linalg.inv(T_cam2base) @ T_end2base_i @ T_board2end
    # 将旋转矩阵转换为旋转向量
    rvec_i, _ = cv2.Rodrigues(T_board2cam_i[:3,:3])
    # 直接取平移向量部分
    tvec_i = T_board2cam_i[:3, 3]
    
    # 投影
    # projectPoints：根据标定板坐标系下的三维点 objp、位姿 rvec_i, tvec_i、相机内参 K 和畸变 dist，计算每个点在该图像中的理想投影位置（像素坐标）。
    # proj_points 原始形状为 (N, 1, 2)，通过 reshape(-1, 2) 变成 (N, 2)，方便与检测角点对齐。
    proj_points, _ = cv2.projectPoints(objp, rvec_i, tvec_i, K, dist)
    proj_points = proj_points.reshape(-1, 2)
    # 将检测到的亚像素角点同样整形为 (N, 2)，与投影点形状一致。
    corners_detect = corners_detect.reshape(-1, 2)
    
    # 误差
    # 均方根误差（RMSE），先计算MSE均方差，再开根号得到RMSE
    # 用cv2图像检测一次，检测到棋盘格每个角点的位置；再通过标定的T_cam2base和T_board2end，计算出每个角点的理论位置；计算误差
    err = np.sqrt(np.mean(np.sum((corners_detect - proj_points) ** 2, axis=1)))
    errors.append(err)
    print(f"Frame {i}: RMSE = {err:.4f} pixels")

print(f"Average RMSE: {np.mean(errors):.4f} pixels")