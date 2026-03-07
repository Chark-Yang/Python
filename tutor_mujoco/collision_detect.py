import mujoco
import numpy as np
import glfw
import mujoco.viewer
from scipy.optimize import minimize

# 全局变量保存模型数据供 IK 使用
global_model = None
global_data = None
global_ee_id = None

def compute_end_effector_pos(q):
    """使用 mujoco 正向运动学计算末端位置"""
    global global_model, global_data, global_ee_id
    if global_model is None or global_data is None or global_ee_id is None:
        return np.zeros(3)
    global_data.qpos[:7] = q
    mujoco.mj_forward(global_model, global_data)
    return global_data.body(global_ee_id).xpos.copy()

def inverse_kinematics(current_q, target_dir, target_pos, model, data, ee_id):
    """数值优化逆运动学，不依赖 pinocchio"""
    global global_model, global_data, global_ee_id
    global_model = model
    global_data = data
    global_ee_id = ee_id

    target_pos = np.array(target_pos)

    # 定义目标函数：计算末端位置与目标位置的距离
    def objective(q):
        ee_pos = compute_end_effector_pos(q)
        return np.linalg.norm(ee_pos - target_pos)

    # 使用 SLSQP 优化算法最小化误差，找到最优关节角度
    res = minimize(objective, current_q, method='SLSQP', options={'ftol':1e-6, 'maxiter':200})
    # 获取优化后的关节角度和最终误差
    final_q = res.x
    final_err = objective(final_q)


    if final_err < 1e-2:
        print("Convergence achieved!")
    else:
        print("Warning: IK did not reach desired precision")
    print(f"result: {final_q.tolist()}")
    print(f"final error: {final_err}")

    return final_q.tolist()

def limit_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

model = mujoco.MjModel.from_xml_path('model/mujoco_learning/franka_emika_panda/scene.xml')
data = mujoco.MjData(model)

class CustomViewer:
    def __init__(self, model, data):
        self.handle = mujoco.viewer.launch_passive(model, data)
        self.pos = 0.0001

        # 找到末端执行器的 body id
        self.end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'left_finger')
        print(f"End effector ID: {self.end_effector_id}")
        if self.end_effector_id == -1:
            print("Warning: Could not find the end effector with the given name.")

        # 初始关节角度，是从模型中的初始状态读取的
        self.initial_q = data.qpos[:7].copy()
        
        print(f"Initial joint positions: {self.initial_q}")
        theta = np.pi
        #  标准的旋转矩阵，绕 x 轴旋转 theta 角度
        self.R_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

        self.x = 0.5
        self.new_q = self.initial_q

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport

    def run_loop(self):
        status = 0
        while self.is_running():
            mujoco.mj_forward(model, data)
            # 运动阶段：沿x轴移动末端执行器，从0.3-0.5
            if status == 0:
                if self.x <= 0.5: 
                    # self.x += 0.01
                    new_q = inverse_kinematics(self.initial_q, self.R_x, [self.x, 0.0, 0.28], model, data, self.end_effector_id)
                    print(f"new_q: {new_q}")                
                else:
                    status = 1 
            # 抓取阶段：当末端执行器接近目标位置时，进行抓取
            elif status == 1:
                data.ctrl[7] = 10.0
            data.qpos[:7] = new_q
            # 遍历所有接触点
            for i in range(data.ncon):
                contact = data.contact[i]
                # 获取几何体对应的body_id
                body1_id = model.geom_bodyid[contact.geom1]
                body2_id = model.geom_bodyid[contact.geom2]

                # 通过mj_id2name转换body_id为名称
                body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
                body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

                # print(f"接触点 {i}: 几何体 {contact.geom1} 名字 {body1_name} 和 {contact.geom2} 名字 {body2_name} 在位置 {contact.pos} 发生接触")
            mujoco.mj_step(model, data)
            self.sync()

viewer = CustomViewer(model, data)
# 调整摄像机距离，使其能够完整地看到机器人和桌面，3代表3米
viewer.cam.distance = 3
# 调整摄像机的方位角，使其从侧面观察机器人，90度是正侧面，0度是正前方，180度是背面
viewer.cam.azimuth = 90
# 调整摄像机的仰角，使其更好地观察到机器人和桌面，负数是俯视，正数是仰视，0是水平视角    
viewer.cam.elevation = -30
viewer.run_loop()