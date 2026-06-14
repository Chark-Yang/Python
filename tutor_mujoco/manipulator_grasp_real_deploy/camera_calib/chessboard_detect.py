"""
data检测成功率
"""

import cv2
import os

# （列，行），是内角点的数量

CHESSBOARD_SIZE = (8, 11)

success_num = 0

PATH = "260614_data2_reproj"   # 你采集数据的文件夹

for i in range(50):

    img = cv2.imread(
        f"{PATH}/{i:03d}.jpg"
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    ret, corners = cv2.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE
    )

    print(i, ret)

    if ret:
        success_num += 1

print(
    f"{success_num}/10 success"
)