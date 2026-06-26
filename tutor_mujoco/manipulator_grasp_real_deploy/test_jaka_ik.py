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

# 封装jaka的函数
def jaka_joint_move(move_mode=0,is_block=True,joint_pose=None):
    
    robot.login()#登录  
    robot.power_on() #上电  
    robot.enable_robot()  
    # 运动模式，绝对运动是0，相对运动是1;阻塞True,非阻塞False
    robot.joint_move(joint_pose,move_mode,is_block,0.1) 

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

def jaka_linear_move_x(distance):
    robot.login()#登录  
    robot.power_on() #上电  
    robot.enable_robot()  
    print("沿X轴向前移动") 

    tcp_pos=[distance,0,0,0,0,0]  
    robot.linear_move(tcp_pos,1,True,15)  

    time.sleep(3)  
    robot.logout()

def jaka_linear_move_y(distance):
    robot.login()#登录  
    robot.power_on() #上电  
    robot.enable_robot()  
    print("沿Y轴向前移动") 

    tcp_pos=[0,distance,0,0,0,0]  
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




# 注意修改成绝对运动
joint_shakeHand = [1.7788700470813605, 1.5160035649775419, 1.7058096395683742, 3.6527536195264347, 1.447604617737367, 0.8140719507856603]
joint_transfer = [1.8685457970871147, 1.0379887926373477, 0.9810034260405898, 3.6153792610972757, 1.507176579738873, 0.8142748802177897]
joint_object = [1.7812167469804219, 2.0933435938787466, 2.052321218060239, 2.102622532127219, 1.6272105867538167, 0.8141790092819777]

# 机械臂竖直位置，类似于手臂竖直伸向天空
joint_shuzhi = [2.366358502357796, 1.5838702391557358, 0.001664555414212032, 1.5655614909562199, 1.5149303794737827, 0.8142628898058285]
# 机械臂初始位置,类似于伸出右手
joint_init = [0.8960019819309415, 2.1849749853113605, 1.3874603971807173, 2.661231233710415, 1.6272465405364076, 0.8142389264351986]

# 机械臂位置2
joint_pos2 = [2.1365610004843165, 2.13749347754378, 1.2112323733051624, 2.947678178659521, 1.3684471991885114, 0.8142509168471599]


# 关节目标角度（单位：弧度）
# joint_pos = [math.pi, math.pi/2, 0, math.pi/4, 0, 0]

# 机械臂初始化
robot = jkrc.RC("192.168.2.155")#返回机器人对象  

current_joint_pos = jaka_get_joint_position()  # 获取当前关节角度
# # print("当前关节角度(rad)  current_joint_pos：", current_joint_pos)

# # 2. 定义 4x4 齐次变换矩阵
# T_matrix = np.array([
#     [-0.1875, -0.7926,  0.5803,  0.4512 ],
#     [-0.9256,  0.3403,  0.1657, -0.5615 ],
#     [-0.3288, -0.5060, -0.7974,  0.05712],
#     [ 0.0,     0.0,     0.0,     1.0    ]
# ])

# # 3. 转换为 spatialmath 的 SE3 对象
# T_se3 = sm.SE3(T_matrix, check=False)

# # # 4. 获取 JAKA 格式的笛卡尔位姿 [x, y, z, rx, ry, rz]
# cartesian_pose = extract_ik_EEpose(T_se3)
# print("目标笛卡尔位姿(mm, rad)：", cartesian_pose)


# ================= 验证测试 =================
# 1. 先获取当前真实的 TCP 笛卡尔位姿 [x, y, z, rx, ry, rz]
ret_tcp = jaka_get_tcp_position()
if ret_tcp != 0:
    current_cartesian = list(ret_tcp)
    print("当前真实 TCP 位姿: ", current_cartesian)
    
    # 2. 模拟一个目标点：保持姿态不变，仅把 Z 轴向上抬高 100 mm
    test_cartesian = current_cartesian.copy()
    test_cartesian[2] += 100.0  
    print("测试目标 TCP 位姿 (Z抬高100mm): ", test_cartesian)
    
    # 3. 对这个绝对可达的目标点求逆解
    ret_ik_test = robot.kine_inverse(current_joint_pos, test_cartesian)
    if ret_ik_test[0] == 0:
        print("✅ 测试逆解成功，目标关节角度: ", ret_ik_test[1])
    else:
        print("❌ 测试逆解失败，错误码: ", ret_ik_test[0])
else:
    print("获取 TCP 位置失败，错误码: ", ret_tcp[0])



# robot.login()                      
# ret_ik = robot.kine_inverse(current_joint_pos, cartesian_pose)  
# if ret_ik[0] == 0:  
#     print("ikine_inverse success, result = :",ret_ik[1])  
# else:  
#     print("some things happend,the errcode is: ",ret_ik[0])               
# robot.logout()












   
