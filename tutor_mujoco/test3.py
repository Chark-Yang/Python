"""
显示一个UR5e机械臂在Mujoco中执行关节空间轨迹
"""

import mujoco.viewer
import ikpy.chain
import transforms3d as tf
import numpy as np
 
 
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
        if np.allclose(qpos, self.waypoint, atol=0.02):
            try:
                self.waypoint = next(self.trajectory)
                return self.waypoint
            except StopIteration:
                pass
        return self.waypoint
 
 
def main():
    model = mujoco.MjModel.from_xml_path('model/mujoco_menagerie/universal_robots_ur5e/scene.xml')
    data = mujoco.MjData(model)
    my_chain = ikpy.chain.Chain.from_urdf_file("model/ur5e.urdf",
                                               active_links_mask=[False, False] + [True] * 6 + [False])
 
    # ref_pos和initial_position 必须和URDF的link数量一致
    print(len(my_chain.links))
    start_joints = np.array([-1.57, -1.34, 2.65, -1.3, 1.55, 0])  # 对应机械臂末端位姿[-0.14, 0.3, 0.1, 3.14, 0, 1.57]，前3个是xyz坐标，后3个是末端姿态
    
    data.qpos[:6] = start_joints # 确保渲染一开始机械臂便处于起始位置，而非MJCF中的默认位置
 
    # 设置目标点
    ee_pos = [-0.13, 0.6, 0.1]
    ee_euler = [3.14, 0, 1.57]
    ref_pos = [0, 0, -1.57, -1.34, 2.65, -1.3, 1.55, 0, 0]
    ee_orientation = tf.euler.euler2mat(*ee_euler)
 
    joint_angles = my_chain.inverse_kinematics(ee_pos, ee_orientation, "all", initial_position=ref_pos)
    print(f"Calculated joint angles: {joint_angles}")
    end_joints = joint_angles[2:-1]
 
    joint_trajectory = JointSpaceTrajectory(start_joints, end_joints, steps=100)
 
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_init(viewer)
        while viewer.is_running():
            waypoint = joint_trajectory.get_next_waypoint(data.qpos[:6])
            data.ctrl[:6] = waypoint
 
            mujoco.mj_step(model, data)
            viewer.sync()
 
 
if __name__ == "__main__":
    main()