"""
这段代码是用于“眼在手外”场景的手眼标定
使用cv2.calibrateHandEye函数,该函数默认是按照“眼在手上”的逻辑设计的，
但通过对输入参数进行数学变换，就可以用于“眼在手外”。

通常可以使用机械臂API 读到“法兰相对于基座的位姿T_gripper2base”,
AX=XB公式中使用的也是T_gripper2base”,
但调用函数时需要输入T_base2gripper 即原本的T_gripper2base的逆矩阵。
相机位姿保持不变,仍然输入T_board2cam(公式和函数保持一致)

最后函数返回的时候
cv2.calibrateHandEye 返回的 R_cam2gripper 和 t_cam2gripper,
实际上就是相机坐标系相对于机械臂基座坐标系的变换矩阵 T_cam2base
"""

import cv2
import yaml
import numpy as np

data = np.load(
    "calibrateHandEye_input_260614.npz"
)

R_board2cam = data[
    "R_board2cam"
]

t_board2cam = data[
    "t_board2cam"
]

R_g2b_list = data[
    "R_gripper2base"
]

t_g2b_list = data[
    "t_gripper2base"
]

R_base2gripper_list = []
t_base2gripper_list = []


for R_g2b, t_g2b in zip(R_g2b_list, t_g2b_list):
    # 矩阵求逆（旋转矩阵的逆等于其转置）
    R_b2g = R_g2b.T
    # 平移向量求逆公式： t_inv = - R_inv * t
    t_b2g = -R_b2g @ t_g2b 
    
    R_base2gripper_list.append(R_b2g)
    t_base2gripper_list.append(t_b2g)




# 1. 换回正确的函数 cv2.calibrateHandEye
# 2. 注意参数顺序：先传机械臂姿态，再传相机姿态
# 3. 建议显式指定标定算法，cv2.CALIB_HAND_EYE_TSAI 是最经典的 Tsai-Lenz 算法
R_cam2base, t_cam2base = cv2.calibrateHandEye(
    R_base2gripper_list,
    t_base2gripper_list,
    R_board2cam,
    t_board2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

print("R_cam2base =")
print(R_cam2base)

print("t_cam2base =")
print(t_cam2base)

# 清理字典，只保存我们需要的 cam2base 结果
result = {
    "R_cam2base": R_cam2base.tolist(),
    "t_cam2base": t_cam2base.reshape(3).tolist()
}

with open("handeye_260614.yaml", "w") as f:
    yaml.dump(result, f, default_flow_style=False) # 加上 default_flow_style=False 让 yaml 更易读