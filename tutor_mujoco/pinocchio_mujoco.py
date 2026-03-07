"""
检测碰撞，但是pinocchio的碰撞检测功能不太好用，后面agent自动修改的程序，使用数值优化的方式实现了逆运动学，效果不好
"""

import mujoco
import numpy as np
import glfw
from scipy.optimize import minimize
from numpy.linalg import norm

# 全局变量，存储 mujoco 模型和数据
global_model = None
global_data = None
end_effector_id = None

def compute_end_effector_pos(q):
    """计算给定关节角度下的末端执行器位置"""
    global global_model, global_data, end_effector_id
    if global_model is None or global_data is None:
        return np.array([0, 0, 0])
    
    # 设置关节角度
    global_data.qpos[:7] = q
    # 进行正向运动学计算
    mujoco.mj_forward(global_model, global_data)
    # 获取末端执行器位置
    return global_data.body(end_effector_id).xpos.copy()

def inverse_kinematics(current_q, target_dir, target_pos, model, data, ee_id):
    """
    使用数值优化方法实现逆运动学
    Args:
        current_q: 当前关节角度
        target_dir: 目标方向（旋转矩阵，这里简化使用）
        target_pos: 目标位置
        model: mujoco 模型
        data: mujoco 数据
        ee_id: 末端执行器 body ID
    """
    global global_model, global_data, end_effector_id
    global_model = model
    global_data = data
    end_effector_id = ee_id
    
    # 定义目标位置
    target_pos = np.array(target_pos)
    
    # 定义目标函数：末端执行器位置与目标位置的距离
    def objective(q):
        try:
            ee_pos = compute_end_effector_pos(q)
            pos_error = np.linalg.norm(ee_pos - target_pos)
            return pos_error
        except:
            return 1e10
    
    # 使用 scipy 优化器求解逆运动学
    result = minimize(
        objective, 
        current_q,
        method='SLSQP',
        options={'ftol': 1e-6, 'maxiter': 200}
    )
    
    final_q = result.x
    final_error = objective(final_q)
    
    # 打印结果信息
    if final_error < 0.01:
        print("Convergence achieved!")
    else:
        print(
            "Warning: the iterative algorithm has not reached convergence "
            "to the desired precision"
        )
    
    print(f"result: {final_q.tolist()}")
    print(f"final error: {final_error}")
    
    return final_q.tolist()

def scroll_callback(window, xoffset, yoffset):
    global cam
    # 调整相机的缩放比例
    cam.distance *= 1 - 0.1 * yoffset

def limit_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def main():
    global cam
    # 加载模型
    model = mujoco.MjModel.from_xml_path('model/mujoco_learning/franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)

    # 打印所有 body 的 ID 和名称
    print("All bodies in the model:")
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"ID: {i}, Name: {body_name}")

    # 初始化 GLFW
    if not glfw.init():
        return

    window = glfw.create_window(1200, 900, 'Panda Arm Control', None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # 设置鼠标滚轮回调函数
    glfw.set_scroll_callback(window, scroll_callback)

    # 初始化渲染器
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    pert = mujoco.MjvPerturb()
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

    scene = mujoco.MjvScene(model, maxgeom=10000)

    # 找到末端执行器的 body id
    end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link7')
    print(f"End effector ID: {end_effector_id}")
    if end_effector_id == -1:
        print("Warning: Could not find the end effector with the given name.")
        glfw.terminate()
        return

    # 初始关节角度
    initial_q = data.qpos[:7].copy()
    print(f"Initial joint positions: {initial_q}")
    theta = np.pi
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    x = 0.3
    new_q = initial_q
    while not glfw.window_should_close(window):
        # 获取当前末端执行器位置
        mujoco.mj_forward(model, data)
        end_effector_pos = data.body(end_effector_id).xpos
        print(f"End effector position: {end_effector_pos}")

        if x < 0.6: 
            x += 0.004
            new_q = inverse_kinematics(initial_q, R_x, [x, 0.2, 0.7], model, data, end_effector_id)
        data.qpos[:7] = new_q

        # 模拟一步
        mujoco.mj_step(model, data)

        # 更新渲染场景
        viewport = mujoco.MjrRect(0, 0, 1200, 900)
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, con)

        # 交换前后缓冲区
        glfw.swap_buffers(window)
        glfw.poll_events()

    # 清理资源
    glfw.terminate()


if __name__ == "__main__":
    main()