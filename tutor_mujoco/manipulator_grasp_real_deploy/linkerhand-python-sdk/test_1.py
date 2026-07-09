"""
特点: 感知到触觉,停止夹紧,2个线程
运动流程:手掌张开->大拇指横摆就位->大拇指根部弯曲到位->其余四指预备抓取到位 ->5根手指同时步进弯曲,直到触觉检测到接触,哪根手指触觉力到达阈值就停止哪根手指的弯曲

抓取采集数据,保存为npz格式
"""

import sys
sys.path.append('/home/chark/linkerhand-python-sdk')
# print(sys.path)
from LinkerHand.linker_hand_api import LinkerHandApi

import time
import threading
import os
import numpy as np


import termios
import tty
import select


# 如果 linker_hand.get_force() 底层走串口/通信接口，两个线程同时读可能不稳定。建议加一个锁。
force_lock = threading.Lock()
# 获取法向压力
def get_normal_force():
    with force_lock:
        force = linker_hand.get_force()
    # print(f"force:{force}")
    return force[0]

# 获取切向压力
def get_tangential_force():
    force = linker_hand.get_force()
    # print(f"force:{force}")
    return force[1]

# 获取切向压力方向
def get_tangential_force_dir():
    force = linker_hand.get_force()
    # print(f"force:{force}")
    return force[2]

# 获取接近感觉
def get_approach_inc():
    force = linker_hand.get_force()
    # print(f"force:{force}")
    return force[3]



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

        # 只要有1个手指关节不为0，就继续发送指令
        if(value_pinky_mcp != 0 or value_pinky_pip != 0 or value_ring_mcp != 0 or value_ring_pip != 0 or value_middle_mcp != 0 or value_middle_pip != 0 or value_index_mcp != 0 or value_index_pip != 0 or value_thumb_mcp != 0):
            linker_hand.finger_move([202, value_index_mcp, value_middle_mcp, value_ring_mcp, value_pinky_mcp, 150, 37, 100, 180, 240, 0, 255, 255, 255, 255, value_thumb_mcp, value_index_pip, value_middle_pip, value_ring_pip, value_pinky_pip])
        else:
            print(linker_hand.get_state())
            stop_index = True
            stop_middle = True
            stop_ring = True        
            stop_pinky = True
            stop_thumb = True


stop_pinky = False
stop_ring = False
stop_middle = False
stop_index = False
stop_thumb = False

# 达到接触阈值的标志位
contact_pinky = False
contact_ring = False
contact_middle = False
contact_index = False
contact_thumb = False

# 达到接触阈值时的mcp位置
pinky_contact_mcp = None
ring_contact_mcp = None
middle_contact_mcp = None
index_contact_mcp = None
thumb_contact_mcp = None

contact_threshold = 2        # 判断首次接触
force_stop_threshold = 15
max_press_delta = 100          # 首次接触后最多继续弯曲的位移量

normal_force = []
pinky_normal_force = []
ring_normal_force = []
middle_normal_force = []
index_normal_force = []
thumb_normal_force = []

# 检测5根手指的触觉是否达到阈值
def tactile_pinky_ring_monitor():

    global stop_pinky, stop_ring, stop_middle, stop_index, stop_thumb
    global contact_pinky, contact_ring, contact_middle, contact_index, contact_thumb
    global pinky_contact_mcp, ring_contact_mcp, middle_contact_mcp, index_contact_mcp, thumb_contact_mcp
    global normal_force, pinky_normal_force, ring_normal_force, middle_normal_force, index_normal_force, thumb_normal_force


    while not (stop_pinky and stop_ring and stop_middle and stop_index and stop_thumb):
        normal_force = get_normal_force()   
        pinky_normal_force = normal_force[4]
        ring_normal_force = normal_force[3]
        middle_normal_force = normal_force[2]
        index_normal_force = normal_force[1]
        thumb_normal_force = normal_force[0]
        
         # 小指
        if pinky_normal_force > contact_threshold and not contact_pinky:
            contact_pinky = True
            pinky_contact_mcp = value_pinky_mcp
            print("pinky首次接触, force:", pinky_normal_force, "mcp:", pinky_contact_mcp)

        # 接触但是还没有停止时
        if contact_pinky and not stop_pinky:
            press_delta = pinky_contact_mcp - value_pinky_mcp
            # 停止条件：触觉力大于阈值或者位移量大于最大允许位移量
            if pinky_normal_force > force_stop_threshold or press_delta >= max_press_delta + 20:
                stop_pinky = True
                print("pinky停止, force:", pinky_normal_force, "press_delta:", press_delta)


        # 无名指
        if ring_normal_force > contact_threshold and not contact_ring:
            contact_ring = True
            ring_contact_mcp = value_ring_mcp
            print("ring首次接触, force:", ring_normal_force, "mcp:", ring_contact_mcp)

        if contact_ring and not stop_ring:
            press_delta = ring_contact_mcp - value_ring_mcp
            if ring_normal_force > force_stop_threshold or press_delta >= max_press_delta + 10:
                stop_ring = True
                print("ring停止, force:", ring_normal_force, "press_delta:", press_delta)

        # 中指
        if middle_normal_force > contact_threshold and not contact_middle:
            contact_middle = True
            middle_contact_mcp = value_middle_mcp
            print("middle首次接触, force:", middle_normal_force, "mcp:", middle_contact_mcp)

        if contact_middle and not stop_middle:
            press_delta = middle_contact_mcp - value_middle_mcp
            if middle_normal_force > force_stop_threshold or press_delta >= max_press_delta:
                stop_middle = True
                print("middle停止, force:", middle_normal_force, "press_delta:", press_delta)

        # 食指
        if index_normal_force > contact_threshold and not contact_index:
            contact_index = True
            index_contact_mcp = value_index_mcp
            print("index首次接触, force:", index_normal_force, "mcp:", index_contact_mcp)

        if contact_index and not stop_index:
            press_delta = index_contact_mcp - value_index_mcp
            if index_normal_force > force_stop_threshold or press_delta >= max_press_delta:
                stop_index = True
                print("index停止, force:", index_normal_force, "press_delta:", press_delta)

        # 拇指
        if thumb_normal_force > contact_threshold and not contact_thumb:
            contact_thumb = True
            thumb_contact_mcp = value_thumb_mcp
            print("thumb首次接触, force:", thumb_normal_force, "mcp:", thumb_contact_mcp)

        if contact_thumb and not stop_thumb:
            press_delta = thumb_contact_mcp - value_thumb_mcp
            if thumb_normal_force > force_stop_threshold or press_delta >= max_press_delta:
                stop_thumb = True
                print("thumb停止, force:", thumb_normal_force, "press_delta:", press_delta)

        time.sleep(0.01)  # 10ms，别太快




step_id = 0

step_records = []
time_records = []
position_records = []
normal_force_records = []
contact_records = []
stop_records = []



def record_sample():
    global step_id

    force = get_normal_force()
    current_time = time.perf_counter() - start_time

    position = [
        value_thumb_mcp,
        value_index_mcp, value_index_pip,
        value_middle_mcp, value_middle_pip,
        value_ring_mcp, value_ring_pip,
        value_pinky_mcp, value_pinky_pip,
    ]

    force_value = [
        force[0],  # thumb
        force[1],  # index
        force[2],  # middle
        force[3],  # ring
        force[4],  # pinky
    ]

    contact_mcp = [
        -1 if thumb_contact_mcp is None else thumb_contact_mcp,
        -1 if index_contact_mcp is None else index_contact_mcp,
        -1 if middle_contact_mcp is None else middle_contact_mcp,
        -1 if ring_contact_mcp is None else ring_contact_mcp,
        -1 if pinky_contact_mcp is None else pinky_contact_mcp,
    ]

    stop_state = [
        stop_thumb,
        stop_index,
        stop_middle,
        stop_ring,
        stop_pinky,
    ]

    step_records.append(step_id)
    time_records.append(current_time)
    position_records.append(position)
    normal_force_records.append(force_value)
    contact_records.append(contact_mcp)
    stop_records.append(stop_state)

    step_id += 1

def is_q_pressed():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        key = sys.stdin.read(1)
        return key.lower() == "q"
    return False


def stop_all_fingers():
    global stop_pinky, stop_ring, stop_middle, stop_index, stop_thumb

    stop_pinky = True
    stop_ring = True
    stop_middle = True
    stop_index = True
    stop_thumb = True

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
pose_5_fingers_grasp2=[202, 199, 199, 199, 199, 
                       150, 37, 100, 180, 240, 
                       0, 255, 255, 255, 255, 
                       255, 148, 148, 148, 148]

# 灵巧手初始化API hand_type:left or right   hand_joint:L7 or L10 or L20 or L25
linker_hand = LinkerHandApi(hand_type="right", hand_joint="L20")




def main():

    global value_pinky_mcp, value_pinky_pip
    global value_ring_mcp, value_ring_pip
    global value_middle_mcp, value_middle_pip
    global value_index_mcp, value_index_pip
    global value_thumb_mcp
    global stop_pinky, stop_ring, stop_middle, stop_index, stop_thumb

    save_dir = "tactile_dataset"
    os.makedirs(save_dir, exist_ok=True)
    trial_id = time.strftime("%Y%m%d_%H%M%S")


    label = input("请输入标签 hard/soft: ")
    object_name = input("请输入物体名称: ")

    
    # linker_hand.set_speed(speed=[120,200,200,200,200])
    linker_hand.set_speed(speed=[80,120,120,120,120])

    # 手掌先张开，避免移动过程中与其他物体接触
    linker_hand.finger_move(pose=pose_open)
    time.sleep(2)



    # 大拇指就位，先横摆
    linker_hand.finger_move(pose=pose_grasp0)
    print("大拇指横摆就位")
    time.sleep(2)

    # 拇指根部弯曲到位，稍微弯曲
    linker_hand.finger_move(pose=pose_grasp1)
    print("大拇指根部弯曲到位")
    time.sleep(2)



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


    # 启动触觉监听线程
    tactile_thread = threading.Thread(target=tactile_pinky_ring_monitor)
    tactile_thread.start()

    

    old_settings = termios.tcgetattr(sys.stdin)

    max_steps = 160
    # 允许少于 5 指结束的最小步数
    min_steps_before_partial_stop = 120

    try:
        tty.setcbreak(sys.stdin.fileno())

        global start_time
        start_time = time.perf_counter()

        print("开始采集，按 q 可提前停止")


        # 5根手指没有全部停止，继续循环
        # while not (stop_pinky and stop_ring and stop_middle and stop_index and stop_thumb):
        while True:
            stopped_count = sum([
                stop_thumb,
                stop_index,
                stop_middle,
                stop_ring,
                stop_pinky,
            ])
            if stopped_count == 5:
                print("已有5根手指停止，结束本次采集")
                stop_all_fingers()
                end_reason = "all_stopped"
                break
            
            # 超过一定时间，还是没有5根手指停止，就看有没有4根手指接触，如果有，就结束采集
            # 4 根有效手指可以提前结束
            if stopped_count == 4 and step_id >= min_steps_before_partial_stop:
                print("已有4根手指停止，结束本次采集")
                stop_all_fingers()
                end_reason = "four_fingers_stopped"
                break
            
            # 3 根有效手指要等到最大步数再结束,勉强接受
            if stopped_count == 3 and step_id >= max_steps:
                print("已有3根手指停止，结束本次采集")
                stop_all_fingers()
                end_reason = "three_fingers_max_steps"
                break

            if step_id >= max_steps:
                print("达到最大采集步数，结束本次采集")
                stop_all_fingers()
                end_reason = "max_steps"
                break

            if is_q_pressed():
                print("检测到 q,提前结束采集")
                stop_all_fingers()
                end_reason = "user_q"
                break

            finger_bend_5_fingers(2, 2)
            record_sample()
            time.sleep(0.05)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    tactile_thread.join()

    save_path = os.path.join(save_dir, f"{trial_id}_{label}_{object_name}.npz")

    np.savez(
        save_path,
        step=np.array(step_records),
        time=np.array(time_records),
        position=np.array(position_records),
        force=np.array(normal_force_records),
        contact=np.array(contact_records),
        stop=np.array(stop_records),
        label=np.array(label),
        object_name=np.array(object_name),
        trial_id=np.array(trial_id),
        position_names=np.array([
            "thumb_mcp",
            "index_mcp", "index_pip",
            "middle_mcp", "middle_pip",
            "ring_mcp", "ring_pip",
            "pinky_mcp", "pinky_pip",
        ]),
        force_names=np.array([
            "thumb_force",
            "index_force",
            "middle_force",
            "ring_force",
            "pinky_force",
        ]),

        settings=np.array([
            contact_threshold,
            force_stop_threshold,
            max_press_delta,
            max_steps,
        ]),
        settings_names=np.array([
            "contact_threshold",
            "force_stop_threshold",
            "max_press_delta",
            "max_steps",
        ]),

        end_reason=np.array(end_reason)
    )

    print("数据已保存:", save_path)


if __name__ == "__main__":
    main()