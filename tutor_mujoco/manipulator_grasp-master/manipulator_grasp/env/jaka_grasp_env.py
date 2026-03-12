import os.path
import sys

sys.path.append('../../manipulator_grasp')

import time
import numpy as np
import spatialmath as sm
import mujoco
import mujoco.viewer

from manipulator_grasp.arm.robot import Robot, UR5e, jaka
from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.utils import mj
import ikpy.chain


class jakaGraspEnv:

    def __init__(self):
        self.sim_hz = 500

        self.mj_model: mujoco.MjModel = None
        self.mj_data: mujoco.MjData = None
        self.robot: Robot = None
        self.joint_names = []
        self.robot_q = np.zeros(6)
        self.robot_T = sm.SE3()
        self.T0 = sm.SE3()

        self.mj_renderer: mujoco.Renderer = None
        self.mj_depth_renderer: mujoco.Renderer = None
        self.mj_viewer: mujoco.viewer.Handle = None
        self.height = 256
        self.width = 256
        self.fovy = np.pi / 4
        self.camera_matrix = np.eye(3)
        self.camera_matrix_inv = np.eye(3)
        self.num_points = 4096

    def reset(self):

        URDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'jaka_description', 'jaka_s5.urdf')
        MUJOCO_XML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'scenes', 'scene_jaka.xml')

        self.mj_model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.robot = jaka.JakaRobot(URDF_PATH, "Link_05")
        self.robot.set_base(mj.get_body_pose(self.mj_model, self.mj_data, "jaka_base").t)
        # print("robot base:", self.robot.base)
        
        # 设置起始位置
        # self.robot_q = np.array([0, 0, 0, 0, 0, 0])
        # 使用上帝之手摆好姿势之后，记得也给电机赋值self.mj_data.ctrl[:6]，要不全是0，机械臂会直接带着一百万的增益，疯狂发力，导致你看不到正确的位姿。
        self.robot_q = np.array([1.57, 1.57, 0, 1.57, 1.57, 0])
        self.robot.set_joint(self.robot_q)
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        [mj.set_joint_q(self.mj_model, self.mj_data, jn, self.robot_q[i]) for i, jn in enumerate(self.joint_names)]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # 告诉电机：你现在的目标就是保持在 1.57，不要动！
        self.mj_data.ctrl[:6] = self.robot_q

        # 1. 计算夹爪末端当前的绝对位姿 (此时还没有设置 Tool)
        T_flange = self.robot.fkine(self.robot_q)
        print("起始状态当前位姿：\n", T_flange)
        
        
        # 2. 计算夹爪底座应该去的真实位置
        T_gripper_base = sm.SE3(T_flange) 
        # print(sm.SE3(T_gripper_base))

        # 挂载夹爪，正向运动学求解，计算末端位姿
        mj.attach(self.mj_model, self.mj_data, "attach", "2f85", sm.SE3(T_gripper_base))

        # 设置偏移，正向运动学求解夹爪末端位姿；夹爪只是显示作用，运动学计算仍然以机械臂末端为准
        robot_tool = sm.SE3.Trans(0.0, 0.13, 0.0)
        self.robot.set_tool(robot_tool)
        self.robot_T = self.robot.fkine(self.robot_q)
        self.T0 = self.robot_T.copy()
        print(f"self.T0：{self.T0}")


        self.mj_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_depth_renderer = mujoco.renderer.Renderer(self.mj_model, height=self.height, width=self.width)
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.enable_depth_rendering()
        # self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

        self.camera_matrix = np.array([
            [self.height / (2.0 * np.tan(self.fovy / 2.0)), 0.0, self.width / 2.0],
            [0.0, self.height / (2.0 * np.tan(self.fovy / 2.0)), self.height / 2.0],
            [0.0, 0.0, 1.0]
        ])
        self.camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        self.step_num = 0
        # observation = self._get_obs()
        observation = None
        return observation

    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
        if self.mj_renderer is not None:
            self.mj_renderer.close()
        if self.mj_depth_renderer is not None:
            self.mj_depth_renderer.close()

    def step(self, action=None):
        if action is not None:
            self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data)

        # self.mj_viewer.sync()

    def render(self):
        self.mj_renderer.update_scene(self.mj_data, 0)
        self.mj_depth_renderer.update_scene(self.mj_data, 0)
        return {
            'img': self.mj_renderer.render(),
            'depth': self.mj_depth_renderer.render()
        }



if __name__ == '__main__':
    env = jakaGraspEnv()
    env.reset()
    for i in range(10000):
        env.step()
    imgs = env.render()
    env.close()
