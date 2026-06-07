"""
只移动jaka机械臂,看看能否正常通讯移动
"""

import sys
sys.path.append('/home/chark/linkerhand-py-sdk-real-deploy')
# print(sys.path)
from LinkerHand.linker_hand_api import LinkerHandApi

import time
import jkrc  




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

# 机械臂初始化
robot = jkrc.RC("192.168.2.155")#返回机器人对象  
print("末端绕x旋转10°")
jaka_get_tcp_position()

# print("沿X轴向前移动100mm")
# jaka_linear_move_x(100)
# jaka_get_tcp_position()


# # TCP目标位置（单位：mm）
# print("沿Y轴向前移动100mm")
# jaka_linear_move_y(100)
# jaka_get_tcp_position()











   
