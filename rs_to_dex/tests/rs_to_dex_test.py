"""
 运行程序，给定物体类别，随机选择一个抓取姿势，使用subProcess 并调用 visualize_result.py 进行可视化

"""


import os
import subprocess
import random
import numpy as np

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

        files_with_name = [name for name in os.listdir(DEXGRASP_ROOT) if class_name in name]
        print("包含该类别的文件有：", len(files_with_name))
        if files_with_name:
            index = random.randint(0, len(files_with_name)-1)
            selected_file = files_with_name[index]
            print("\n-----------------------------")
            print(f"✔ 找到抓取文件: {selected_file}")
            
            # 提取 object_code (去掉 .npy)
            object_code = selected_file[:-4]  # 假设文件名以 .npy 结尾
            
            # 在后台调用 visualize_result.py
            cmd = [
                "/home/chark/miniconda3/envs/dex_generation/bin/python",
                "/home/chark/DexGraspNet/grasp_generation/tests/visualize_result.py",
                "--object_code", object_code,
                "--result_path", DEXGRASP_ROOT
            ]
            print(f"调用可视化: {' '.join(cmd)}")
            subprocess.Popen(cmd)  # 在后台运行，不阻塞
            
        else:
            print("✘ 未找到对应抓取文件")
    else:
        print("DexGraspNet 中无该类别")


if __name__ == '__main__':

    find_object_in_dexgrasp("bottle")
