"""
测试ikpy的正向运动学
"""
import mujoco.viewer
import os
import ikpy.chain
import transforms3d as tf
import numpy as np
import time
import spatialmath as sm

def viewer_init(viewer):
    """渲染器的摄像头视角初始化"""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = [0, 0.5, 0.5]
    viewer.cam.distance = 2.5
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -30

class JointSpaceTrajectory:
    """关节空间坐标系下的线性插值轨迹"""
    def __init__(self, start_joints, end_joints, steps):
        self.start_joints = np.array(start_joints)
        self.end_joints = np.array(end_joints)
        self.steps = steps
        self.step = (self.end_joints - self.start_joints) / self.steps
        self.trajectory = self._generate_trajectory()
        self.waypoint = self.start_joints
 
    def _generate_trajectory(self):
        for i in range(self.steps + 1):
            yield self.start_joints + self.step * i
        # 确保最后精确到达目标关节值
        yield self.end_joints
 
    def get_next_waypoint(self, qpos):
        # 检查当前的关节值是否已经接近目标路径点。若是，则更新下一个目标路径点；若否，则保持当前目标路径点不变。
        
        try:
            self.waypoint = next(self.trajectory)
            return self.waypoint
        except StopIteration:
            pass
        return self.waypoint

def main():
    URDF_PATH = os.path.join(os.path.dirname(__file__), 'manipulator_grasp','assets', 'jaka_description', 'jaka_s5.urdf')
    MUJOCO_XML_PATH = os.path.join(os.path.dirname(__file__), 'manipulator_grasp','assets', 'jaka_description', 'scene_init.xml')
    
    model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file(URDF_PATH, base_elements=["Link_00"], active_links_mask=[False, True, True, True, True, True, True])
    
    # print(f"IKPy 链包含 {len(my_chain.links)} 个连杆。详情如下：")
    # for link in my_chain.links:
    #     print(f" - {link.name}: 偏移量(Translation) = {link.translation_vector}")
    
    # start_joints = np.array([0, 0, 0, 1.57, 0, 0])

    start_joints = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/2, 0])

    test_joints = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/2, 0])

    data.qpos[:6] = test_joints

    mujoco.mj_forward(model, data)
    

    end_joints = np.array([np.pi/2, np.pi/2, 0, np.pi/2, np.pi/2, 0])
    # end_joints = np.array([np.pi/2, 0, 0, 1.57, 1.57, 0])
    

    #  定义关节角度 (弧度制)
    # 注意：数组长度必须与 my_chain.links 的数量一致（包含基座和固定连杆）
    ikpy_joints = [0,np.pi/2, np.pi/2, 0, np.pi/2, np.pi/2, 0]

    # 计算正向运动学
    # 返回的是一个 4x4 的齐次变换矩阵 (Homogeneous Transformation Matrix)
    transformation_matrix = my_chain.forward_kinematics(ikpy_joints)
    print(f"Transformation Matrix from IKPy:{transformation_matrix}")

    # 获取末端 Link (假设名字是 Link_06 或 dummy_tcp，请根据你的 URDF 修改)
    end_link_name = "Link_06" # 如果你有法兰，填法兰的名字
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_link_name)
    
    # MuJoCo 中末端的全局位置和旋转矩阵
    mj_pos = data.xpos[body_id]
    mj_rot = data.xmat[body_id].reshape(3, 3)

    # 获取基座 Link_00 的全局位置
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Link_00")
    print(f"Base ID: {base_id}")
    
    # 兼容处理：如果 base_id == -1（基座被融合进世界原点了），则基座坐标为 [0,0,0]
    if base_id == -1:
        base_pos = np.array([0.8, 0.6, 0.745])
    else:
        base_pos = data.xpos[base_id]

    # IKPy 算出来的是相对基座的局部坐标,要转换成全局坐标需要加上基座坐标
    ikpy_local_pos = transformation_matrix[:3, 3]

    # 【神级转换】局部坐标 + 基座全局坐标 = IKPy的全局坐标
    ikpy_global_pos = ikpy_local_pos + base_pos

    # n_wc = np.array([0.0, -1.0, 0.0])
    # o_wc = np.array([-1.0, 0.0, -0.5])
    # t_wc = np.array([1.0, 0.6, 2.0])
    # T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    # 末端的位姿矩阵
    T = sm.SE3.Trans(base_pos) * sm.SE3(transformation_matrix)

    print("【1】IKPY 算出的末端相对于基座(Link_00)的坐标 (x,y,z):")
    print(transformation_matrix[:3, 3])
    print(f"IKPY 全局坐标: {ikpy_global_pos}")
    print(f"末端全局位姿矩阵 T:\n{T}")
    
    print("【2】MuJoCo 算出的末端全局坐标 (x,y,z):")
    print(mj_pos)

    print(f"base_pos: {base_pos}")
    
    
   

    

    joint_trajectory = JointSpaceTrajectory(start_joints, end_joints, steps=2000)


    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_init(viewer)
        while viewer.is_running():
            step_start = time.time()

            waypoint = joint_trajectory.get_next_waypoint(data.qpos[:6])
            
            # 【瞬移大法】强行覆盖位置，速度清零防爆炸
            data.qpos[:6] = waypoint
            data.qvel[:6] = 0.0

            # 【极其关键】必须告诉电机也瞄准这个位置！不给电机发新指令，它就会和你抢夺控制权
            # 因为你没有给 ctrl 赋新值，电机的目标位置依然停留在旧的地方（比如起始点 0）
            # 于是上帝把机械臂瞬移到新位置，但是电机接受到的命令还是旧位置，于是一百万的Kp就会爆发很大的力量，进行抵抗，所以一直有很大的震荡
            data.ctrl[:6] = waypoint  


            # 【神级补丁】直接喂给物理引擎重力补偿，抵消那 2 毫秒物理步长里重力造成的微小下坠
            data.qfrc_applied[:] = data.qfrc_bias


            mujoco.mj_step(model, data)
            viewer.sync()

            # 【关键】让代码稍微等一等，保持与真实时间同步，否则动画一闪而过
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            

    

    


if __name__ == "__main__":
    main()


