"""
功能：

连接 D435
实时显示 RGB 图像
    按 s 保存：
        RGB 图片
        当前 JAKA TCP 位姿
        时间戳
    按 q 退出
        自动编号

注意:cd 到camera_calib文件夹
   修改1个地方,SAVE_DIR
"""

import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs

import jkrc


def jaka_get_tcp_position():
    robot.login()#登录  
    robot.power_on() #上电  
    robot.enable_robot()  
    print("获取机械臂末端位置") 

    ret = robot.get_tcp_position()  
    if ret[0] == 0:  
        print("the tcp position is :",ret[1])  
    else:  
        print("some things happend,the errcode is: ",ret[0])  
    
    robot.logout()
    
    # print(ret[1])
    return ret[1]


# =========================
# JAKA 配置
# =========================

# 机械臂初始化
robot = jkrc.RC("192.168.2.155")#返回机器人对象  


# =========================
# 保存目录
# =========================

SAVE_DIR = "merged_reproj_test"  

os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# D435 初始化
# =========================

pipeline = rs.pipeline()

config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

profile = pipeline.start(config)

print("D435 started.")


# =========================
# 自动寻找下一个编号
# =========================

existing = []

for file in os.listdir(SAVE_DIR):

    if file.endswith(".jpg"):

        try:
            idx = int(file.split(".")[0])
            existing.append(idx)

        except:
            pass

if len(existing) == 0:
    sample_idx = 0
else:
    sample_idx = max(existing) + 1

print(f"Start index = {sample_idx}")


# =========================
# 主循环
# =========================

try:

    while True:

        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        color_image = np.asanyarray(
            color_frame.get_data()
        )

        vis = color_image.copy()

        cv2.putText(
            vis,
            f"sample: {sample_idx}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        cv2.putText(
            vis,
            "s: save   q: quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("D435 RGB", vis)

        key = cv2.waitKey(1) & 0xFF

        # ------------------
        # 保存
        # ------------------
        if key == ord("s"):

            tcp_ret = jaka_get_tcp_position()

            # 根据你之前的输出：
            #
            # tcp_ret:
            # [x,y,z,rx,ry,rz]
            #

            pose = np.array(
                tcp_ret,
                dtype=np.float64
            )

            img_path = os.path.join(
                SAVE_DIR,
                f"{sample_idx:03d}.jpg"
            )

            pose_path = os.path.join(
                SAVE_DIR,
                f"{sample_idx:03d}_pose.npy"
            )

            cv2.imwrite(
                img_path,
                color_image
            )

            np.save(
                pose_path,
                pose
            )

            print("\n====================")
            print(f"Saved sample {sample_idx}")
            print(f"Image : {img_path}")
            print(f"Pose  : {pose_path}")
            print("TCP pose:")
            print(pose)
            print("====================\n")

            sample_idx += 1

        # ------------------
        # 退出
        # ------------------
        elif key == ord("q"):

            break

finally:

    pipeline.stop()

    cv2.destroyAllWindows()

    try:
        robot.logout()
    except:
        pass

    print("Exit.")