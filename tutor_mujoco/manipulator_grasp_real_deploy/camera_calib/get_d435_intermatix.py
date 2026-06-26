"""
新版本的保存内参矩阵和畸变系数的脚本
先采集棋盘格的图片，然后使用 MATLAB 进行相机标定，得到内参矩阵和畸变系数。

"""

import numpy as np

# 1. 转置后的内参矩阵 K
# 使用260614_data2计算得到
# K = np.array([
#     [604.9577, 0.0,       321.3569],
#     [0.0,      605.7141,  243.0247],
#     [0.0,      0.0,       1.0     ]
# ], dtype=np.float64)

# 260626_data计算得到的K
# K = np.array([
#     [604.5563, 0.0,       320.1829],
#     [0.0,      605.5834,  245.6451],
#     [0.0,      0.0,       1.0     ]
# ], dtype=np.float64)

# merged计算得到的K
K = np.array([
    [600.8591, 0.0,       322.1869],
    [0.0,      601.8001,  246.7771],
    [0.0,      0.0,       1.0     ]
], dtype=np.float64)


# 2. 填入畸变系数 dist
#字母 k 系列 (k1, k2, k3)：专门指代 Radial Distortion（径向畸变）
#字母 p 系列 (p1, p2)：专门指代 Tangential Distortion（切向畸变）
# 顺序为: [k1, k2, p1, p2, k3]
# 使用260614_data2
# dist = np.array(
#     [0.1295, -0.2665, 0.0011, 0.0014, 0.0],
#     dtype=np.float64
# )

# # 260626_data计算得到的dist
# dist = np.array(
#     [0.1092, -0.1950, 0.0025, 0.0012, 0.0],
#     dtype=np.float64
# )

# merged计算得到的dist
dist = np.array(
    [0.1111, -0.2164, 0.0029, 0.0018, 0.0],
    dtype=np.float64
)



# 打印核对
print("=== 新的内参矩阵 K ===")
print(K)
print("\n=== 新的畸变系数 dist ===")
print(dist)

# 保存为 npy 文件
np.save("K.npy", K)
np.save("dist.npy", dist)

print("\n文件已成功保存: K.npy, dist.npy")