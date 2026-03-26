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

# 定义一个全长 21 的基础指令数组 (预设姿态 Pre-shape)
base_ctrl = np.zeros(model.nu)
print(len(base_ctrl), model.nu)  # 确认长度是21

# 为 PIP 和 DIP 关节设置一个微弯曲的固定角度 (例如 0.5 弧度，约 30 度)
# 你可以通过打印 model.actuator_names 来查看所有名字并赋值
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if "pip_pos" in name:
        base_ctrl[i] = 0.6  # 食指等4指pip
    elif "dip_pos" in name and "thumb" not in name:
        base_ctrl[i] = 0.5  # 食指等4指dip
    elif "mcp_roll_pos" in name:
        base_ctrl[i] = 0    # 食指等4指mcp_roll
    elif "thumb_mcp_pos" in name:
        base_ctrl[i] = 0.41   # 拇指mcp_pos
    elif "thumb_dip_pos" in name:
        base_ctrl[i] = 0.549  # 拇指dip_pos
    elif "cmc_yaw_pos" in name:
        base_ctrl[i] = 0.343  # 拇指cmc_yaw_pos
    elif "cmc_roll_pos" in name:
        base_ctrl[i] = 1.22   # 拇指cmc_roll_pos
    else:
        base_ctrl[i] = 0.0  # 强化学习控制的5个关节在0

print("预设姿态 base_ctrl:", base_ctrl)


# 【瞬移大法】强行覆盖位置，速度清零防爆炸
# data.qpos[:] = base_ctrl
# data.qvel[:] = 0.0

# 【极其关键】必须告诉电机也瞄准这个位置！不给电机发新指令，它就会和你抢夺控制权
# 因为你没有给 ctrl 赋新值，电机的目标位置依然停留在旧的地方（比如起始点 0）
# 于是上帝把机械臂瞬移到新位置，但是电机接受到的命令还是旧位置，于是一百万的Kp就会爆发很大的力量，进行抵抗，所以一直有很大的震荡
data.ctrl[:] = base_ctrl  
for _ in range(100):
    mujoco.mj_step(model, data)




mujoco.viewer.launch(model, data)
# try:
#     while viewer.is_running():
#         data.ctrl[:] = base_ctrl  
#         mujoco.mj_step(model, data)
#         viewer.sync()
#         time.sleep(0.01)
# finally:
#     viewer.close()



