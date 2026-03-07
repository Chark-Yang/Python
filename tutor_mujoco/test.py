"""
使用ikpy库计算末端执行器的逆运动学，实现碰撞检测（对比pinocchio_mujoco）
"""

import mujoco.viewer
import os
import ikpy.chain
import transforms3d as tf
import numpy as np


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
        if np.allclose(qpos, self.waypoint, atol=0.02):
            try:
                self.waypoint = next(self.trajectory)
                return self.waypoint
            except StopIteration:
                pass
        return self.waypoint



def main():

    
    model = mujoco.MjModel.from_xml_path('model/mujoco_learning/franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file("model/panda_arm.urdf",
                                               active_links_mask=[False]*1 + [True]*7 + [False]*2)
    # print(len(my_chain.links))
    # print("Number of actuators:", model.nu)
    # print("Ctrl shape:", data.ctrl.shape)
    # for i in range(model.nu):
    #     print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i))

    start_joints = [0.34109947, 0.33980988, -0.07398333, -2.40874181, 2.74551278, 3.75068942, -1.42579633]
    data.ctrl[:7]= start_joints
    
    ref_pos = [0, 0.348, 0.459, -0.11, -2.02, 2.73, 3.75, 0.145, 0, 0 ]
    ref_pos[7] = np.clip(ref_pos[7]-np.pi/2,-2.89,2.89)

    ee_pos = [0.5, 0, 0.1]
    ee_euler = [0, 0, 0]
    ee_orientation = tf.euler.euler2mat(*ee_euler)


    joint_angles = my_chain.inverse_kinematics(ee_pos, ee_orientation, initial_position = ref_pos)

    
    end_joints = joint_angles[1:-2]
    # end_joints = [0.29620221, 0.52427826, -0.12506813, -1.979432, 2.6702319, 3.60257978, -1.42579633]
    print("End joints:")
    print(end_joints)

    

    # data.ctrl[7] = 0.04

    joint_trajectory = JointSpaceTrajectory(start_joints, end_joints, steps=100)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            waypoint = joint_trajectory.get_next_waypoint(data.qpos[:7])
            data.ctrl[:7] = waypoint
            print(data.ctrl[:7])

            mujoco.mj_step(model, data)
            viewer.sync()
 
if __name__ == "__main__":
    main()