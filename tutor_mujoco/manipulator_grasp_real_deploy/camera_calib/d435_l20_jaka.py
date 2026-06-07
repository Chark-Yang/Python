"""
购买了一个新的拓展坞,线长2m,测试D435和l20同时插上拓展坞后,测试是否可以同时使用
同时加入jaka,测试jaka、d435和l20能否同时使用
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(ROOT_DIR, 'linkerhand-python-sdk'))

from LinkerHand.linker_hand_api import LinkerHandApi

import time
import jkrc  

import pyrealsense2 as rs       #用于控制realsense摄像头
import numpy as np              #处理图像数据
import cv2                          #用于显示图像


# 封装jaka的函数
def jaka_joint_move(joint_pose):
    
    robot.login()#登录  
    robot.power_on() #上电  
    robot.enable_robot()  
    # 运动模式，绝对运动是0，相对运动是1;阻塞True,非阻塞False
    robot.joint_move(joint_pose,0,True,0.1) 

    time.sleep(3)
    robot.logout()

def jaka_linear_move_z(distance):
    robot.login()#登录  
    robot.power_on() #上电  
    robot.enable_robot()  
    print("沿Z轴向上移动") 

    tcp_pos=[0,0,distance,0,0,0]  
    robot.linear_move(tcp_pos,1,True,15)  

    time.sleep(3)  
    robot.logout()

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


# 手掌姿势预定义
pose_fist=[40, 0, 0, 0, 0, 131, 10, 100, 180, 240, 19, 255, 255, 255, 255, 135, 0, 0, 0, 0]
pose_open=[255, 255, 255, 255, 255, 255, 10, 100, 180, 240, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255]
pose_OK=[191, 95, 255, 255, 255, 136, 107, 100, 180, 240, 72, 255, 255, 255, 255, 116, 99, 255, 255, 255]
pose_like=[255, 0, 0, 0, 0, 127, 10, 100, 180, 240, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0]

pose_grasp0=[255, 255, 255, 255, 255, 150, 10, 100, 180, 240, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255]
pose_grasp1=[168, 138, 137, 255, 255, 150, 10, 100, 180, 240, 0, 255, 255, 255, 255, 63, 64, 66, 255, 255]
pose_grasp2=[202, 255, 255, 255, 255, 150, 10, 100, 180, 240, 0, 255, 255, 255, 255, 1, 255, 255, 255, 255]





if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)      #初始化RealSenseSense摄像头，并配置为捕获640x480的深度和颜色图像，每秒30帧
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 机械臂初始化
    robot = jkrc.RC("192.168.2.155")#返回机器人对象      
    
    # 灵巧手初始化API hand_type:left or right   hand_joint:L7 or L10 or L20 or L25
    linker_hand = LinkerHandApi(hand_type="right", hand_joint="L20")
    linker_hand.set_speed(speed=[120,200,200,200,200])
    


    # 每次移动机械臂之前，首先确保手处于张开位置，防止碰到其他物体
    linker_hand.finger_move(pose=pose_open)
    time.sleep(3)


    jaka_get_tcp_position()
    linker_hand.finger_move(pose=pose_fist)
    time.sleep(3)

    # Start streaming
    pipeline.start(config)
    try:
        while True:                                 #使用while循环不断捕获图像数据，直到用户关闭窗口。
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()            # 使用wait_for_frames()函数等待捕获到一组深度和颜色图像帧。
            depth_frame = frames.get_depth_frame()          #从帧中获取深度图像。
            color_frame = frames.get_color_frame()          #从帧中获取颜色图像。
            if not depth_frame or not color_frame:          #如果捕获到的帧中没有深度或颜色图像，则跳过当前循环，等待下一帧。
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())        # 将捕获到的深度图像转换为NumPy数组，以便进行后续处理。
            color_image = np.asanyarray(color_frame.get_data())        #将捕获到的颜色图像...

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)      #使用OpenCV库的applyColorMap()函数将深度图像转换为彩色图像，并使用cv2.hstack()函数将颜色和深度图像水平堆叠。
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)           #使用OpenCV库的namedWindow()函数创建一个窗口，并使用下面的imshow()函数将图像显示在窗口中。
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)                    #使用cv2.waitKey()函数等待用户按下键盘上的某个键，并返回按键的ASCII码。
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:             #如果用户按下'q'键或按ESC键，则使用cv2.destroyAllWindows()函数关闭窗口，并使用pipeline.stop()函数停止摄像头的流。
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()
