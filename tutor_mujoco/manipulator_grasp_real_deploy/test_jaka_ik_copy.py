"""
只移动jaka机械臂,看看能否正常通讯移动，测试求逆解
"""

import sys
sys.path.append('/home/chark/linkerhand-py-sdk-real-deploy')
# print(sys.path)
from LinkerHand.linker_hand_api import LinkerHandApi

import time
import jkrc  

import numpy as np
import spatialmath as sm




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

    return ret[1]


def jaka_get_joint_position():

    # 获取当前关节角度
    robot.login()#登录  
    ret = robot.get_joint_position()  

    if ret[0] == 0:  
        print("the current joint position is :",ret[1])  
    else:  
        print("some things happend,the errcode is: ",ret[0])  
    
    robot.logout()  #登出
    return ret[1]
      


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

current_joint_pos = jaka_get_joint_position()  # 获取当前关节角度
# # print("当前关节角度(rad)  current_joint_pos：", current_joint_pos)


# ================= 验证测试 =================
# 1. 先获取当前真实的 TCP 笛卡尔位姿 [x, y, z, rx, ry, rz]
ret_tcp = jaka_get_tcp_position()

current_cartesian = list(ret_tcp)
print("当前真实 TCP 位姿: ", current_cartesian)

# 2. 模拟一个目标点：保持姿态不变，仅把 Z 轴向上抬高 100 mm
test_cartesian = current_cartesian.copy()
test_cartesian[2] += 100.0  
print("测试目标 TCP 位姿 (Z抬高100mm): ", test_cartesian)

# 3. 对这个绝对可达的目标点求逆解
ret_ik_test = robot.kine_inverse(current_joint_pos, test_cartesian)
if ret_ik_test[0] == 0:
    print("测试逆解成功，目标关节角度: ", ret_ik_test[1])
else:
    print("测试逆解失败，错误码: ", ret_ik_test[0])
















   
