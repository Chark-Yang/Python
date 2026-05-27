import mujoco.viewer
import os
import time
import numpy as np

# URDF_PATH = os.path.join(os.path.dirname(__file__), 'right', 'linkerhand_l20_right.urdf')

XML_PATH = os.path.join(os.path.dirname(__file__), 'scene.xml')

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# 预先找到这 5 个由 RL 控制的关节在 21 个 actuator 中的索引编号
# 假设我们控制的是每个手指的 mcp_pitch (负责主要弯曲的根部关节)
rl_action_indices = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thumb_cmc_pitch_pos"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "index_mcp_pitch_pos"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "middle_mcp_pitch_pos"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ring_mcp_pitch_pos"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pinky_mcp_pitch_pos")
]

print("RL 控制的 actuator 索引:", rl_action_indices)

# 定义一个全长 的基础指令数组 (预设姿态 Pre-shape)
base_ctrl = np.zeros(model.nu)
print(len(base_ctrl), model.nu)  

# 为 PIP 和 DIP 关节设置一个微弯曲的固定角度 (例如 0.5 弧度，约 30 度)
# 你可以通过打印 model.actuator_names 来查看所有名字并赋值
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if "pip_pos" in name:
        base_ctrl[i] = 0.6  # 食指等4指pip
    elif "dip_pos" in name and "thumb" not in name:
        base_ctrl[i] = 1.01  # 食指等4指dip
    elif "mcp_roll_pos" in name:
        base_ctrl[i] = 0    # 食指等4指mcp_roll
    elif "thumb_mcp_pos" in name:
        base_ctrl[i] = 0.41   # 拇指mcp_pos
    elif "thumb_dip_pos" in name:
        base_ctrl[i] = 1.2  # 拇指dip_pos
    elif "cmc_yaw_pos" in name:
        base_ctrl[i] = 0.343  # 拇指cmc_yaw_pos
    elif "cmc_roll_pos" in name:
        base_ctrl[i] = 1.22   # 拇指cmc_roll_pos
    else:
        base_ctrl[i] = 0.0  # 强化学习控制的5个关节在0

print("预设姿态 base_ctrl:", base_ctrl)


# 手腕根部预抓取 6D 位姿
pre_grasp_pose = [0.01, 0.0, 0, 1.57, 1.57, 0.0] 

# 6个手腕电机的名字
wrist_actuators = [
    "wrist_x_pos", "wrist_y_pos", "wrist_z_pos", 
    "wrist_roll_pos", "wrist_pitch_pos", "wrist_yaw_pos"
]

# 阶段一：高空姿态对齐 (在初始的 Z=1.0 高度，偏移 X=0.01，并完成翻转)
# 注意：这里的 Z 是 0，因为它是相对于 hand_root 当前高度的偏移量
phase1_pose = [0.05, 0.0, 0.0, 1.57, 1.57, 0.0]

# 阶段二：垂直下降抓取 (保持姿态不变，Z轴向下移动 0.2 米去靠近杯子)
phase2_pose = [0.05, 0.0, -0.55, 1.57, 1.57, 0.0] 


# 定义我们要监测的 5 个触觉传感器名字
sensor_names = ["index_touch", "middle_touch", "ring_touch", "pinky_touch", "thumb_touch"]

# 全局计数器
step_counter = 0

data.ctrl[:] = base_ctrl
# 终极回调函数：到点释放控制权，专心做监测！
def master_controller(model, data):
    global step_counter
    
    # ================= 1. 轨迹控制 (带释放机制) =================
    # 核心修改：只在仿真的前 3 秒内接管电机，3 秒后彻底“撒手”！
    if data.time < 3.0:
        if data.time < 2.0:
            current_target_pose = phase1_pose
        else:
            current_target_pose = phase2_pose
            
        # 核心修改：精准赋值！只修改手腕的 6 个电机，绝对不碰整个 data.ctrl[:] 数组！
        for i, act_name in enumerate(wrist_actuators):
            act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id != -1:
                data.ctrl[act_id] = current_target_pose[i]
                
    # ================= 2. 实时触觉监测 (永远运行) =================
    step_counter += 1
    if step_counter % 50 == 0:  # 每 0.1 秒检查一次
        touched = False
        msg = f"⏱️ {data.time:.1f}s | 💥 接触力: "
        
        for i, s_name in enumerate(sensor_names):
            force = data.sensordata[i]
            if force > 0.001:  # 过滤底噪
                msg += f"[{s_name}: {force:.3f}] "
                touched = True
                
        if touched:
            print(msg)

# 注册回调函数
mujoco.set_mjcb_control(master_controller)

# 启动阻塞式渲染器
mujoco.viewer.launch(model, data)

# 退出时，清理回调函数
mujoco.set_mjcb_control(None)

# for i, act_name in enumerate(wrist_actuators):
#         act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
#         if act_id != -1:
#             base_ctrl[act_id] = phase1_pose[i]

# data.ctrl[:] = base_ctrl

# time.sleep(1.0)  # 等待 1 秒，让手腕先移动到阶段一的预抓取姿态

# for i, act_name in enumerate(wrist_actuators):
#         act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
#         if act_id != -1:
#             base_ctrl[act_id] = phase2_pose[i]

# data.ctrl[:] = base_ctrl

# time.sleep(1.0)




# 定义一个回调函数，MuJoCo 的物理引擎在每走一步(0.002秒)之前，都会自动调用这个函数！
# def trajectory_controller(model, data):
#     if data.time < 2.0:
#         current_target_pose = phase1_pose
#     else:
#         current_target_pose = phase2_pose
        
#     # 动态把当前的 6D 目标位姿，塞进 base_ctrl 数组里
#     for i, act_name in enumerate(wrist_actuators):
#         act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
#         if act_id != -1:
#             base_ctrl[act_id] = current_target_pose[i]
            
#     # 把更新后的指令发送给底层电机
#     data.ctrl[:] = base_ctrl

# data.ctrl[:] = base_ctrl

# 注册回调函数！这是 MuJoCo 操控真机的灵魂接口
# mujoco.set_mjcb_control(trajectory_controller)

# mujoco.viewer.launch(model, data)

# 退出时，清理回调函数，防止影响下一次运行
# mujoco.set_mjcb_control(None)
   

# if len(data.sensordata) > 0:
#     index_force = data.sensordata[0]
#     print(f"食指触觉传感器值: {index_force:.4f}")   


