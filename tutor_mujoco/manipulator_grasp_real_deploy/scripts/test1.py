""""
测试jaka机械臂能否在mujoco中正确加载，并且执行一个简单的关节空间轨迹。这个脚本不涉及夹爪和抓取，仅仅是验证机械臂模型和基本运动。
"""

import mujoco.viewer
import os
import ikpy.chain
import transforms3d as tf
import numpy as np
import time

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
    print(len(my_chain.links))
    
    start_joints = np.array([0, 0, 0, 1.57, 0, 0])
    data.qpos[:6] = start_joints 

    mujoco.mj_forward(model, data)
    
    # 设置目标点
    # ee_pos = [0.13, 0.6, 0.6]
    # ee_euler = [0, 0, 0]
    # ref_pos = [0, -1.57, -1.34, 2.65, -1.3, 1.55, 0]
    # ee_orientation = tf.euler.euler2mat(*ee_euler)
 
    # joint_angles = my_chain.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position=ref_pos)
    # print(f"Calculated joint angles: {joint_angles}")
    # end_joints = joint_angles[1:7]

    end_joints = np.array([np.pi/2, 1.57, 0, 1.57, 1.57, 0])
    # end_joints = np.array([np.pi/2, 0, 0, 1.57, 1.57, 0])

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