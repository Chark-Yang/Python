import mujoco
import mujoco.viewer
import pinocchio as pin
import numpy as np
import time
import os

# ==========================================
# 1. 参数与路径配置
# ==========================================
URDF_PATH = os.path.join(os.path.dirname(__file__), 'manipulator_grasp','assets', 'jaka_description', 'jaka_s5.urdf')
MUJOCO_XML_PATH = os.path.join(os.path.dirname(__file__), 'manipulator_grasp','assets', 'jaka_description', 'scene_init.xml')


# ==========================================
# 2. 加载 Pinocchio 和 MuJoCo 模型
# ==========================================
# 加载 Pinocchio 模型 (用于运动学和动力学计算)
pin_model = pin.buildModelFromUrdf(URDF_PATH)
pin_data = pin_model.createData()
nq = pin_model.nq  # 关节位置维度
nv = pin_model.nv  # 关节速度维度
print(f"Pinocchio model loaded: nq={nq}, nv={nv}")

# 加载 MuJoCo 模型 (用于物理仿真)
mj_model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
mj_data = mujoco.MjData(mj_model)

# ==========================================
# 3. 五次多项式轨迹规划器
# ==========================================
def quintic_polynomial(t, t_0, t_f, q_0, q_f):
    """
    生成平滑的五次多项式轨迹，确保起止点的速度和加速度均为 0
    """
    if t < t_0:
        return q_0, np.zeros_like(q_0), np.zeros_like(q_0)
    elif t > t_f:
        return q_f, np.zeros_like(q_f), np.zeros_like(q_f)
    
    time_ratio = (t - t_0) / (t_f - t_0)
    
    # 五次多项式系数项
    c3 = 10
    c4 = -15
    c5 = 6
    
    s = c3 * time_ratio**3 + c4 * time_ratio**4 + c5 * time_ratio**5
    s_dot = (3*c3 * time_ratio**2 + 4*c4 * time_ratio**3 + 5*c5 * time_ratio**4) / (t_f - t_0)
    s_ddot = (6*c3 * time_ratio + 12*c4 * time_ratio**2 + 20*c5 * time_ratio**3) / (t_f - t_0)**2
    
    q_des = q_0 + (q_f - q_0) * s
    v_des = (q_f - q_0) * s_dot
    a_des = (q_f - q_0) * s_ddot
    
    return q_des, v_des, a_des

# ==========================================
# 4. 控制器与主循环
# ==========================================
# PD 控制器增益
Kp = np.eye(nv) * 20000.0  # 比例增益 (根据你的机械臂量级微调)
Kd = np.eye(nv) * 20.0   # 微分增益

# 定义运动起止状态
q_start = np.zeros(nq)
q_goal = np.array([0, 1.57, 0, 1.57, 0, 0.0]) # Jaka通常是6自由度，填入目标关节角
t_0 = 1.0  # 1秒时开始运动
t_f = 4.0  # 4秒时到达目标

# 初始化 MuJoCo 状态
mj_data.qpos[0:nq] = q_start
mujoco.mj_step(mj_model, mj_data)

# 启动 MuJoCo 交互式可视化窗口
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    
    # 获取仿真的物理时间步长
    dt = mj_model.opt.timestep
    print(f"dt:{dt}")
    
    while viewer.is_running():
        step_start = time.time()
        
        # 1. 从 MuJoCo 读取当前状态
        q_curr = mj_data.qpos[:nq].copy()
        v_curr = mj_data.qvel[:nv].copy()
        t_curr = mj_data.time
        # print(f"t_curr:{t_curr}")
        
        # 2. 获取当前时刻的期望轨迹 (位置、速度、加速度)
        q_des, v_des, a_des = quintic_polynomial(t_curr, t_0, t_f, q_start, q_goal)
        
        # 3. 计算力矩 (Computed Torque Control)
        # 闭环控制法则: a_cmd = a_des + Kp(q_des - q_curr) + Kd(v_des - v_curr)
        a_cmd = a_des + Kp @ (q_des - q_curr) + Kd @ (v_des - v_curr)
        
        # 使用 Pinocchio 的 RNEA 算法计算所需的前馈与补偿力矩
        # 内部相当于求解了: \tau = M(q) * a_cmd + C(q, v)*v + G(q)
        tau = pin.rnea(pin_model, pin_data, q_curr, v_curr, a_cmd)
        
        # 4. 将力矩发送给 MuJoCo 的驱动器
        mj_data.ctrl[:nq] = tau
        
        # 5. 步进物理仿真
        mujoco.mj_step(mj_model, mj_data)
        print(mj_data.qpos[0:nq])
        
        # 同步可视化界面
        viewer.sync()
        
        # 保持实时运行节拍
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)