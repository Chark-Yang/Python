"""
只移动jaka机械臂,看看能否正常通讯移动，测试求逆解

260627
成功测试求逆解
"""

import sys
sys.path.append('/home/chark/linkerhand-py-sdk-real-deploy')
# print(sys.path)
from LinkerHand.linker_hand_api import LinkerHandApi

import time
import jkrc  

import numpy as np
import spatialmath as sm

import math


def jaka_get_tcp_position():
    robot.login()#登录  
    robot.power_on() #上电  
    robot.enable_robot()  

    ret = robot.get_tcp_position()  
    if ret[0] == 0:  
        print("机械臂末端位置 :",ret[1])  
    else:  
        print("some things happend,the errcode is: ",ret[0])  
    robot.logout()

    return ret[1]


def jaka_get_joint_position():

    # 获取当前关节角度
    robot.login()#登录  
    ret = robot.get_joint_position()  

    if ret[0] == 0:  
        print("当前关节角度 :",ret[1])  
    else:  
        print("some things happend,the errcode is: ",ret[0])  
    
    robot.logout()  #登出
    return ret[1]
      

def jaka_kine_inverse(current_joint_pos, target_tcp_pose):
    
    # 先登录
    robot.login()
    robot.power_on()
    robot.enable_robot()

    # 求逆解
    ret_ik_test = robot.kine_inverse(current_joint_pos, target_tcp_pose)
    if ret_ik_test[0] == 0:
        print("测试逆解成功，目标关节角度: ", ret_ik_test[1])
    else:
        print("测试逆解失败，错误码: ", ret_ik_test[0])


    # 完成后登出
    robot.logout()
    return ret_ik_test[1]  

def jaka_joint_move(joint_pose):
    
    robot.login()#登录  
    robot.power_on() #上电  
    robot.enable_robot()  

    # 0 绝对 1相对； True 阻塞，False 非阻塞； 0.1rad/s 速度
    robot.joint_move(joint_pose,0,True,0.1) 

    time.sleep(3)
    robot.logout()




def extract_ik_EEpose(T_se3):
    """提取平移和欧拉角((RPY)"""

    t = T_se3.t  # [x, y, z]，单位：米
    rpy = T_se3.rpy()  # [roll, pitch, yaw]，默认单位：弧度
    
    # 【⚠️ 核心修改】JAKA SDK 笛卡尔坐标系的 xyz 通常要求是毫米 (mm)！
    # 将米乘以 1000 转换为毫米
    x_mm = t[0] * 1000.0
    y_mm = t[1] * 1000.0
    z_mm = t[2] * 1000.0
    
    # 拼接成 JAKA 需要的 6D 数组: [x(mm), y(mm), z(mm), rx(rad), ry(rad), rz(rad)]
    return [x_mm, y_mm, z_mm, rpy[0], rpy[1], rpy[2]]



# 机械臂初始化
robot = jkrc.RC("192.168.2.155")#返回机器人对象  

# 获取当前关节角度
current_joint_pos = jaka_get_joint_position()  

# [x,y,z, rx,ry,rz] 单位 mm, rad
current_cartesian = jaka_get_tcp_position()   


# target_tcp_pose = [
#     373.160,
#     -533.370,
#     405.838,
#     math.radians(66.522),
#     math.radians(-15.726),
#     math.radians(38.163)
# ]

target_tcp_pose = [470.758264, -554.955517, 163.065974, 1.5843905567123402, 0.023747211602022648, 0.5736128782835708]



ikine_joint_pos = jaka_kine_inverse(current_joint_pos, target_tcp_pose)

jaka_joint_move(ikine_joint_pos)

print("机械臂移动完成！")









   
