"""
特点: 感知到触觉,停止夹紧,2个线程
运动流程:手掌张开->大拇指横摆就位->大拇指根部弯曲到位->其余四指预备抓取到位 ->5根手指同时步进弯曲,直到触觉检测到接触,哪根手指触觉力到达阈值就停止哪根手指的弯曲

分别打印出各个手指的压力值的变化
"""

import sys
sys.path.append('/home/chark/linkerhand-python-sdk')
# print(sys.path)
from LinkerHand.linker_hand_api import LinkerHandApi

import time
import threading


# 获取法向压力
def get_normal_force():
    force = linker_hand.get_force()
    # print(f"force:{force}")
    return force[0]

# 获取切向压力
def get_tangential_force():
    force = linker_hand.get_force()
    print(f"force:{force}")
    return force[1]

# 获取切向压力方向
def get_tangential_force_dir():
    force = linker_hand.get_force()
    print(f"force:{force}")
    return force[2]

# 获取接近感觉
def get_approach_inc():
    force = linker_hand.get_force()
    print(f"force:{force}")
    return force[3]


# value_mcp = 255
# # 大拇指尖端步进弯曲
# def finger_move_thumb_mcp(step):
#         global value_mcp, stop_thumb
        
#         if(stop_thumb):
#             return

#         value_mcp -= step
        
#         # 大拇指就位时对应的拇指根部是0，保证拇指根部值不会小于0
#         if(value_mcp > 0):
#             linker_hand.finger_move([202, 255, 255, 255, 255, 150, 10, 100, 180, 240, 0, 255, 255, 255, 255, value_mcp, 255, 255, 255, 255])
#         else:
#             value_mcp = 0



# 小指、无名指、中指、食指,拇指步进弯曲
def finger_bend_5_fingers(mcp_step = 5, pip_step = 5):
        global value_pinky_mcp, value_pinky_pip, value_ring_mcp, value_ring_pip, value_middle_mcp, value_middle_pip, value_index_mcp, value_index_pip,value_thumb_mcp
        global stop_pinky, stop_ring, stop_middle, stop_index,stop_thumb
        
        if(stop_pinky and stop_ring and stop_middle and stop_index and stop_thumb):
            return

        if(not stop_pinky):
            value_pinky_mcp -= mcp_step
            value_pinky_pip -= pip_step

        if(not stop_ring):
            value_ring_mcp -= mcp_step
            value_ring_pip -= pip_step

        if(not stop_middle):
            value_middle_mcp -= mcp_step
            value_middle_pip -= pip_step

        if(not stop_index):
            value_index_mcp -= mcp_step
            value_index_pip -= pip_step
        
        if(not stop_thumb):
            value_thumb_mcp -= mcp_step+3

        if value_pinky_mcp < 0:
            value_pinky_mcp = 0
        if value_pinky_pip < 0:
            value_pinky_pip = 0

        if value_ring_mcp < 0:
            value_ring_mcp = 0
        if value_ring_pip < 0:
            value_ring_pip = 0
        
        if value_middle_mcp < 0:
            value_middle_mcp = 0
        if value_middle_pip < 0:
            value_middle_pip = 0
            
        if value_index_mcp < 0:
            value_index_mcp = 0
        if value_index_pip < 0:
            value_index_pip = 0

        if value_thumb_mcp < 0:
            value_thumb_mcp = 0

        if(value_pinky_mcp != 0 or value_pinky_pip != 0 or value_ring_mcp != 0 or value_ring_pip != 0 or value_middle_mcp != 0 or value_middle_pip != 0 or value_index_mcp != 0 or value_index_pip != 0 or value_thumb_mcp != 0):
            linker_hand.finger_move([202, value_index_mcp, value_middle_mcp, value_ring_mcp, value_pinky_mcp, 150, 37, 100, 180, 240, 0, 255, 255, 255, 255, value_thumb_mcp, value_index_pip, value_middle_pip, value_ring_pip, value_pinky_pip])
        else:
            print(linker_hand.get_state())
            stop_index = True
            stop_middle = True
            stop_ring = True        
            stop_pinky = True
            stop_thumb = True



# stop_thumb = False
# force_threshold = 20
# normal_force = []
# thumb_normal_force = []
# 触觉监听线程，只干一件事：设标志位
# def tactile_monitor():
#     global stop_thumb,normal_force,thumb_normal_force,value_mcp

#     while not stop_thumb:
#         normal_force = get_normal_force()   
#         thumb_normal_force = normal_force[0]
        
#         if thumb_normal_force > force_threshold:
#             print("force_threshold:", force_threshold)
#             print("thumb_normal_force:", thumb_normal_force)
#             print("value_mcp:",value_mcp)
#             print("触觉检测到接触，发出停止信号")
#             stop_thumb = True
#             break

#         time.sleep(0.01)  # 10ms，别太快

stop_pinky = False
stop_ring = False
stop_middle = False
stop_index = False
stop_thumb = False

force_threshold = 15
normal_force = []

pinky_normal_force = []
ring_normal_force = []
middle_normal_force = []
index_normal_force = []
thumb_normal_force = []

# 检测5根手指的触觉是否达到阈值
def tactile_pinky_ring_monitor():
    global stop_pinky,stop_ring,stop_middle,stop_index,stop_thumb
    global normal_force,pinky_normal_force,ring_normal_force,middle_normal_force,index_normal_force,thumb_normal_force

    while not (stop_pinky and stop_ring and stop_middle and stop_index and stop_thumb):
        normal_force = get_normal_force()   
        pinky_normal_force = normal_force[4]
        ring_normal_force = normal_force[3]
        middle_normal_force = normal_force[2]
        index_normal_force = normal_force[1]
        thumb_normal_force = normal_force[0]

        if pinky_normal_force > force_threshold and not stop_pinky:
            print("="*20)
            print("force_threshold:", force_threshold)
            print("pinky_normal_force:", pinky_normal_force)
            print("pinky触觉检测到接触，发出停止信号")
            stop_pinky = True
            
        
        if ring_normal_force > force_threshold and not stop_ring:
            print("="*20)
            print("force_threshold:", force_threshold)
            print("pinky_normal_force:", ring_normal_force)
            print("ring触觉检测到接触，发出停止信号")
            stop_ring = True

        if middle_normal_force > force_threshold and not stop_middle:
            print("="*20)
            print("force_threshold:", force_threshold)
            print("middle_normal_force:", middle_normal_force)
            print("middle触觉检测到接触，发出停止信号")
            stop_middle = True

        if index_normal_force > force_threshold and not stop_index:
            print("="*20)
            print("force_threshold:", force_threshold)
            print("index_normal_force:", index_normal_force)
            print("index触觉检测到接触，发出停止信号")
            stop_index = True

        if ring_normal_force > force_threshold and not stop_ring:
            print("="*20)
            print("force_threshold:", force_threshold)
            print("pinky_normal_force:", ring_normal_force)
            print("ring触觉检测到接触，发出停止信号")
            stop_ring = True  

        if thumb_normal_force > force_threshold and not stop_thumb:
            print("="*20)
            print("force_threshold:", force_threshold)
            print("thumb_normal_force:", thumb_normal_force)
            print("thumb触觉检测到接触，发出停止信号")
            stop_thumb = True  

        time.sleep(0.01)  # 10ms，别太快


# 手掌姿势预定义
pose_fist=[40, 0, 0, 0, 0, 131, 10, 100, 180, 240, 19, 255, 255, 255, 255, 135, 0, 0, 0, 0]
pose_open=[255, 255, 255, 255, 255, 255, 10, 100, 180, 240, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255]
pose_OK=[191, 95, 255, 255, 255, 136, 107, 100, 180, 240, 72, 255, 255, 255, 255, 116, 99, 255, 255, 255]
pose_like=[255, 0, 0, 0, 0, 127, 10, 100, 180, 240, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0]

# pose_grasp0是拇指侧摆到位,pose_grasp1拇指根部弯曲到位,pose_grasp2拇指末端弯曲到位
pose_grasp0=[255, 255, 255, 255, 255, 150, 10, 100, 180, 240, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255]
pose_grasp1=[202, 255, 255, 255, 255, 150, 10, 100, 180, 240, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255]
pose_grasp2=[202, 255, 255, 255, 255, 150, 10, 100, 180, 240, 0, 255, 255, 255, 255, 1, 255, 255, 255, 255]

# pinky_grasp2是小指根部微微弯曲，PIP微微弯曲，呈现预备抓握状态
pose_pinky_grasp2=[255, 255, 255, 255, 199, 255, 37, 100, 180, 240, 245, 255, 255, 255, 255, 255, 255, 255, 255, 148]

# 其余四指抓取预备姿势
pose_pinky_ring_grasp2=[255, 199, 199, 199, 199, 255, 37, 100, 180, 240, 245, 255, 255, 255, 255, 255, 148, 148, 148, 148]

# 拇指及其余四指抓取预备姿势
pose_5_fingers_grasp2=[202, 199, 199, 199, 199, 150, 37, 100, 180, 240, 0, 255, 255, 255, 255, 255, 148, 148, 148, 148]

# 灵巧手初始化API hand_type:left or right   hand_joint:L7 or L10 or L20 or L25
linker_hand = LinkerHandApi(hand_type="right", hand_joint="L20")
linker_hand.set_speed(speed=[120,200,200,200,200])


# 手掌先张开，避免移动过程中与其他物体接触
linker_hand.finger_move(pose=pose_open)
time.sleep(2)


# 启动触觉监听线程
tactile_thread = threading.Thread(target=tactile_pinky_ring_monitor)
tactile_thread.start()

# 大拇指就位，先横摆
linker_hand.finger_move(pose=pose_grasp0)
print("大拇指横摆就位")
time.sleep(2)

# 拇指根部弯曲到位，稍微弯曲
linker_hand.finger_move(pose=pose_grasp1)
print("大拇指根部弯曲到位")
time.sleep(2)


# 拇指末端弯曲，直到触觉检测到接触
# while not stop_thumb:
#     finger_move_thumb_mcp(5)
#     time.sleep(0.05)


# 获取预备抓取姿势对应位置的值
value_pinky_mcp = pose_5_fingers_grasp2[4]
value_pinky_pip = pose_5_fingers_grasp2[19]

value_ring_mcp = pose_5_fingers_grasp2[3]
value_ring_pip = pose_5_fingers_grasp2[18]

value_middle_mcp = pose_5_fingers_grasp2[2]
value_middle_pip = pose_5_fingers_grasp2[17]

value_index_mcp = pose_5_fingers_grasp2[1]
value_index_pip = pose_5_fingers_grasp2[16]

value_thumb_mcp = pose_5_fingers_grasp2[15]


# 其余4指就位，预备抓取哪根手指触觉力到达阈值就停止哪根手指的弯曲
linker_hand.finger_move(pose=pose_5_fingers_grasp2)
time.sleep(2)


while not (stop_pinky and stop_ring and stop_middle and stop_index and stop_thumb):
    finger_bend_5_fingers(2, 2)
    time.sleep(0.05)

