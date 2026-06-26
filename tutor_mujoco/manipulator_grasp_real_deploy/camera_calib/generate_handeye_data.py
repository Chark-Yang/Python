"""
准备手眼标定(Hand-Eye Calibration)的输入数据,它读取事先采集的图像和对应的机械臂末端位姿，
通过棋盘格检测计算出标定板(target)在相机(cam)下的位姿，
同时将机械臂末端(gripper)在基座(base)下的位姿转换成标定所需的形式

原本从机械臂读取的末端位姿 (x,y,z,rx,ry,rz) 通常表示 
末端坐标系(gripper)的原点在基坐标系(base)下的位置，
也就是说，它描述的是 末端 → 基座 的变换（记为 T_base_gripper)。

R_target2cam 和 t_target2cam,标定板坐标系 → 相机坐标系 的旋转矩阵和平移向量
R_gripper2base 和 t_gripper2base,机械臂末端(gripper)坐标系 → 机械臂基座(base)坐标系 的旋转矩阵和平移向量

注意修改2个地方: 1. 数据集文件夹路径,2. 输出.npz文件名
"""

import os
import cv2
import numpy as np

from scipy.spatial.transform import Rotation


# DATA_DIR = "260614_data2"   # 你采集数据的文件夹
DATA_DIR = "merged"   

CHESSBOARD_SIZE = (8, 11)
SQUARE_SIZE = 0.015

K = np.load("K.npy")
dist = np.load("dist.npy")


objp = np.zeros(
    (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
    np.float32
)

objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE


R_gripper2base_list = []
t_gripper2base_list = []

R_board2cam_list = []
t_board2cam_list = []


for idx in range(1000):

    img_file = os.path.join(
        DATA_DIR,
        f"{idx:03d}.jpg"
    )

    pose_file = os.path.join(
        DATA_DIR,
        f"{idx:03d}_pose.npy"
    )

    if not os.path.exists(img_file):
        continue

    img = cv2.imread(img_file)

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE
    )

    if not ret:
        print(
            f"skip {idx}"
        )
        continue

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11,11),
        (-1,-1),
        (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
    )
    ok, rvec, tvec = cv2.solvePnP(
        objp,
        corners,
        K,
        dist
    )

    if not ok:
        continue

    R_board2cam, _ = cv2.Rodrigues(
        rvec
    )

    pose = np.load(
        pose_file
    )

    x, y, z, rx, ry, rz = pose
    # 第一次尝试使用xyz内旋，
    R_gripper2base = Rotation.from_euler(
        'xyz',
        [rx, ry, rz]
    ).as_matrix()

    t_gripper2base = np.array(
        [x, y, z]
    ) / 1000.0

    R_board2cam_list.append(
        R_board2cam
    )

    t_board2cam_list.append(
        tvec.reshape(3)
    )

    R_gripper2base_list.append(
        R_gripper2base
    )

    t_gripper2base_list.append(
        t_gripper2base
    )


print(
    "valid samples:",
    len(R_board2cam_list)
)

# 根据AX=XB的推导公式，保存相应的数据
np.savez(
    "calibrateHandEye_input_merged.npz",

    R_board2cam=np.array(
        R_board2cam_list
    ),

    t_board2cam=np.array(
        t_board2cam_list
    ),

    R_gripper2base=np.array(
        R_gripper2base_list
    ),

    t_gripper2base=np.array(
        t_gripper2base_list
    )
)