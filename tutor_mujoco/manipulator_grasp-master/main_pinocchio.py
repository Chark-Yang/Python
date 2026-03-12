"""
和源码main.py并列，互相独立
在jakaGraspEnv使用pinocchio库实现运动规划

"""

import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
from PIL import Image
import spatialmath as sm

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
from manipulator_grasp.env.jaka_grasp_env import jakaGraspEnv

import mujoco.viewer
import transforms3d as tf
import time
import ikpy.chain


def get_net():
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    checkpoint_path = 'logs/log_rs/checkpoint-rs.tar'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    
    # if os.path.exists(checkpoint_path):
    #     print("Loading checkpoint...")
    #     checkpoint = torch.load(checkpoint_path)
    #     net.load_state_dict(checkpoint['model_state_dict'])
    # else:
    #     print("⚠ No checkpoint found. Using random weights.")

    net.eval()
    return net


def get_and_process_data(imgs):
    num_point = 20000

    # imgs = np.load(os.path.join(data_dir, 'imgs.npz'))
    color = imgs['img'] / 255.0
    depth = imgs['depth']

    height = 256
    width = 256
    fovy = np.pi / 4
    intrinsic = np.array([
        [height / (2.0 * np.tan(fovy / 2.0)), 0.0, width / 2.0],
        [0.0, height / (2.0 * np.tan(fovy / 2.0)), height / 2.0],
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0

    camera = CameraInfo(height, width, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    mask = depth < 2.0
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    if len(cloud_masked) >= num_point:
        idxs = np.random.choice(len(cloud_masked), num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_point - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud


def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg


def collision_detection(gg, cloud):
    voxel_size = 0.01
    collision_thresh = 0.01
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
    gg = gg[~collision_mask]

    return gg


def vis_grasps(gg, cloud):
    # gg.nms()
    # gg.sort_by_score()
    # gg = gg[:1]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


def generate_grasps(net, imgs, visual=False):
    end_points, cloud = get_and_process_data(imgs)
    gg = get_grasps(net, end_points)
    gg = collision_detection(gg, np.array(cloud.points))
    gg.nms()
    gg.sort_by_score()
    gg = gg[:1]
    if visual:
        vis_grasps(gg, cloud)
    return gg


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

if __name__ == '__main__':
    # 初始化网络
    net = get_net()

    # 初始化仿真环境
    env = jakaGraspEnv()
    env.reset()

    env.mj_viewer = mujoco.viewer.launch_passive(env.mj_model, env.mj_data)
    for i in range(1000):
        env.step()
        env.mj_viewer.sync()
    imgs = env.render()

    gg = generate_grasps(net, imgs, True)

    # 抓取位置
    robot = env.robot
    T_wb = robot.base
    n_wc = np.array([0.0, -1.0, 0.0])
    o_wc = np.array([-1.0, 0.0, -0.5])
    t_wc = np.array([1.0, 0.6, 2.0])
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    # print("T_wc:\n", T_wc)
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(
        sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))
  
    T_wo = T_wc * T_co
    print("T_wo:\n", T_wo)


    # 在画面中渲染计算出来的T_flange_world和 T_wo
    # T_flange_des是机械臂末端相对于机械臂基座的偏移,必须先求一次逆解才能获取到正常T_flange_des
    # T_flange_world是机械臂末端在全局坐标系中位姿
    # env.robot.ikine(T_wo)

    T_flange_des = env.robot.get_T_flange_des(T_wo)
    T_flange_world = T_wb * sm.SE3(T_flange_des)
    print(f"T_flange_world:\n{T_flange_world}")
    
    #  提取位置和旋转
    target_pos = T_flange_world.t
    target_quat = np.zeros(4)

    gg_pos = T_wo.t
    gg_quat = np.zeros(4)

    # 将旋转矩阵 (3x3) 转换为四元数 (w, x, y, z)，MuJoCo 专用的转换函数
    mujoco.mju_mat2Quat(target_quat, T_flange_world.R.flatten())
    mujoco.mju_mat2Quat(gg_quat, T_wo.R.flatten())


    # 把位姿写给 target_sphere
    # 获取 target_sphere 在 mujoco 模型中的 ID
    target_body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_sphere")
    gg_body_id = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gg_grasp_sphere")

    # 动态修改模型中的位置和姿态
    env.mj_model.body_pos[target_body_id] = target_pos
    env.mj_model.body_quat[target_body_id] = target_quat

    env.mj_model.body_pos[gg_body_id] = gg_pos
    env.mj_model.body_quat[gg_body_id] = gg_quat

    # 刷新物理状态和渲染数据
    mujoco.mj_forward(env.mj_model, env.mj_data)

    #  画面定格，让你慢慢看！
    # print("【Debug】目标法兰盘位姿已渲染（紫色小球+红绿蓝坐标轴）。")
    # print("请在仿真窗口中检查位置和方向是否符合预期。")
    # print("看完后关闭窗口，或者在终端按 Ctrl+C 退出...")
    
    env.mj_viewer.sync()
    time.sleep(3)
    
    
    # start_joints = np.array([0, 0, 0, 1.57, 0, 0])
    # 竖直伸直状态np.array([1.57, 1.57, 0, 1.57, 1.57, 0])
    # 水平伸直状态np.array([0, 0, 0, 0, 0, 0])
    # start_joints = np.array([1.57, 1.57, 0, 1.57, 1.57, 0])
    # end_joints = np.array([1.57, 1.57, 0, 1.57, 1.57, 0])
    # 求逆解参考关节角度 np.array([0.3, 0.7, -1, 2.0, 1.57, 0])
    # 抓取小物块的关节角度

    action = np.zeros(7)

    # T0（机械臂起始位置）

    start_joints = np.array([1.57, 1.57, 0, 1.57, 1.57, 0])
    env.mj_data.qpos[:6] = start_joints
    mujoco.mj_forward(env.mj_model, env.mj_data)
    env.mj_viewer.sync()


    # T0->T2(抓取前10cm，上方10cm)
    T2 = sm.SE3.Trans(-0.0, 0.0, 0.2) * T_flange_world
    print(f"抓取点前10cm  T2:{T2}")

    # 计算逆运动学
    ikine_joint_angels = env.robot.ikine(T2)
    ikine_joint_list_2 = ikine_joint_angels.tolist()

    start_joints_2 = np.array([1.57, 1.57, 0, 1.57, 1.57, 0])
    end_joints_2 = np.array(ikine_joint_list_2)
    print(f"end_joints_2:{end_joints_2}")

    env.mj_data.qpos[:6] = start_joints_2
    mujoco.mj_forward(env.mj_model, env.mj_data)

    joint_trajectory_2 = JointSpaceTrajectory(start_joints_2, end_joints_2, steps=2000)


    while env.mj_viewer.is_running():
            step_start = time.time()

            waypoint = joint_trajectory_2.get_next_waypoint(env.mj_data.qpos[:6])
            # print("当前目标路径点：", waypoint)
            # 【瞬移大法】强行覆盖位置，速度清零防爆炸
            env.mj_data.qpos[:6] = waypoint
            env.mj_data.qvel[:6] = 0.0
            action[:6] = waypoint

            # 【极其关键】必须告诉电机也瞄准这个位置！不给电机发新指令，它就会和你抢夺控制权
            # 因为你没有给 ctrl 赋新值，电机的目标位置依然停留在旧的地方（比如起始点 0）
            # 于是上帝把机械臂瞬移到新位置，但是电机接受到的命令还是旧位置，于是一百万的Kp就会爆发很大的力量，进行抵抗，所以一直有很大的震荡
            env.mj_data.ctrl[:6] = waypoint

            # 【神级补丁】直接喂给物理引擎重力补偿，抵消那 2 毫秒物理步长里重力造成的微小下坠
            env.mj_data.qfrc_applied[:] = env.mj_data.qfrc_bias

            mujoco.mj_step(env.mj_model, env.mj_data)
            env.mj_viewer.sync()

            # 【关键】让代码稍微等一等，保持与真实时间同步，否则动画一闪而过
            time_until_next_step = env.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            # 允许有0.001弧度（0.05度）的极微小误差
            if np.allclose(env.mj_data.qpos[:6], end_joints_2, atol=1e-3):
                print("机械臂已成功到达T2抓取前10cm")
                break
                
    # T2->T3(抓取点)

    T3 = T_flange_world
    print(f"抓取点T3:{T3}")

    # 计算逆运动学
    ikine_joint_angels = env.robot.ikine(T3)
    ikine_joint_list_3 = ikine_joint_angels.tolist()


    start_joints_3 = end_joints_2

    end_joints_3 = np.array(ikine_joint_list_3)
    print(f"end_joints_3: {end_joints_3}")
  

    env.mj_data.qpos[:6] = start_joints_3
    mujoco.mj_forward(env.mj_model, env.mj_data)

    joint_trajectory_3 = JointSpaceTrajectory(start_joints_3, end_joints_3, steps=2000)


    while env.mj_viewer.is_running():
            step_start = time.time()

            waypoint = joint_trajectory_3.get_next_waypoint(env.mj_data.qpos[:6])
            # print("当前目标路径点：", waypoint)
            # 【瞬移大法】强行覆盖位置，速度清零防爆炸
            env.mj_data.qpos[:6] = waypoint
            env.mj_data.qvel[:6] = 0.0
            action[:6] = waypoint

            # 【极其关键】必须告诉电机也瞄准这个位置！不给电机发新指令，它就会和你抢夺控制权
            # 因为你没有给 ctrl 赋新值，电机的目标位置依然停留在旧的地方（比如起始点 0）
            # 于是上帝把机械臂瞬移到新位置，但是电机接受到的命令还是旧位置，于是一百万的Kp就会爆发很大的力量，进行抵抗，所以一直有很大的震荡
            env.mj_data.ctrl[:6] = waypoint

            # 【神级补丁】直接喂给物理引擎重力补偿，抵消那 2 毫秒物理步长里重力造成的微小下坠
            env.mj_data.qfrc_applied[:] = env.mj_data.qfrc_bias

            mujoco.mj_step(env.mj_model, env.mj_data)
            env.mj_viewer.sync()

            # 【关键】让代码稍微等一等，保持与真实时间同步，否则动画一闪而过
            time_until_next_step = env.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            # 允许有0.001弧度（0.05度）的极微小误差
            if np.allclose(env.mj_data.qpos[:6], end_joints_3, atol=1e-3):
                print("机械臂已成功到达T3点")
                break

    # 关闭夹爪
    for i in range(1500):
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255])
        env.mj_data.ctrl[:] = action
        mujoco.mj_step(env.mj_model, env.mj_data)
        env.mj_viewer.sync()

    print("关闭夹爪已完成") 


    # T3 -> T4(抬起物体30cm)

    T4 = sm.SE3.Trans(0.0, 0.0, 0.2) * T3
    print(f"抬起物体30cm  T4:{T4}")

    # 计算逆运动学
    ikine_joint_angels = env.robot.ikine(T4)
    ikine_joint_list_4 = ikine_joint_angels.tolist()


    start_joints_4 = end_joints_3

    end_joints_4 = np.array(ikine_joint_list_4)
    print(f"end_joints_4: {end_joints_4}")
  

    env.mj_data.qpos[:6] = start_joints_4
    mujoco.mj_forward(env.mj_model, env.mj_data)

    joint_trajectory_4 = JointSpaceTrajectory(start_joints_4, end_joints_4, steps=2000)


    while env.mj_viewer.is_running():
            step_start = time.time()

            waypoint = joint_trajectory_4.get_next_waypoint(env.mj_data.qpos[:6])
            # print("当前目标路径点：", waypoint)
            # 【瞬移大法】强行覆盖位置，速度清零防爆炸
            env.mj_data.qpos[:6] = waypoint
            env.mj_data.qvel[:6] = 0.0
            action[:6] = waypoint

            # 【极其关键】必须告诉电机也瞄准这个位置！不给电机发新指令，它就会和你抢夺控制权
            # 因为你没有给 ctrl 赋新值，电机的目标位置依然停留在旧的地方（比如起始点 0）
            # 于是上帝把机械臂瞬移到新位置，但是电机接受到的命令还是旧位置，于是一百万的Kp就会爆发很大的力量，进行抵抗，所以一直有很大的震荡
            env.mj_data.ctrl[:6] = waypoint

            # 【神级补丁】直接喂给物理引擎重力补偿，抵消那 2 毫秒物理步长里重力造成的微小下坠
            env.mj_data.qfrc_applied[:] = env.mj_data.qfrc_bias

            mujoco.mj_step(env.mj_model, env.mj_data)
            env.mj_viewer.sync()

            # 【关键】让代码稍微等一等，保持与真实时间同步，否则动画一闪而过
            time_until_next_step = env.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            # 允许有0.001弧度（0.05度）的极微小误差
            if np.allclose(env.mj_data.qpos[:6], end_joints_4, atol=1e-3):
                print("机械臂已成功到达T4点")
                break
    
    # T4 -> T5(移动物体到（1.4,0.2,z轴位置不动）)
    T5 = sm.SE3.Rt(sm.SO3(T4.R, check=False).norm(), [1.2, 0.2, T4.t[2]])

    # T5 = sm.SE3.Trans(1.4, 0.2, T4.t[2]) * sm.SE3(T4.R)              

    # print(f"T4.t: {T4.t}")
    # print(f"T4.R: {T4.R}")

    print(f"T5:{T5}")

    # 计算逆运动学
    ikine_joint_angels = env.robot.ikine(T5)
    ikine_joint_list_5 = ikine_joint_angels.tolist()


    start_joints_5 = end_joints_4

    end_joints_5 = np.array(ikine_joint_list_5)
    print(f"end_joints_5: {end_joints_5}")
  

    env.mj_data.qpos[:6] = start_joints_5
    mujoco.mj_forward(env.mj_model, env.mj_data)

    joint_trajectory_5 = JointSpaceTrajectory(start_joints_5, end_joints_5, steps=2000)


    while env.mj_viewer.is_running():
            step_start = time.time()

            waypoint = joint_trajectory_5.get_next_waypoint(env.mj_data.qpos[:6])
            # print("当前目标路径点：", waypoint)
            # 【瞬移大法】强行覆盖位置，速度清零防爆炸
            env.mj_data.qpos[:6] = waypoint
            env.mj_data.qvel[:6] = 0.0
            action[:6] = waypoint

            # 【极其关键】必须告诉电机也瞄准这个位置！不给电机发新指令，它就会和你抢夺控制权
            # 因为你没有给 ctrl 赋新值，电机的目标位置依然停留在旧的地方（比如起始点 0）
            # 于是上帝把机械臂瞬移到新位置，但是电机接受到的命令还是旧位置，于是一百万的Kp就会爆发很大的力量，进行抵抗，所以一直有很大的震荡
            env.mj_data.ctrl[:6] = waypoint

            # 【神级补丁】直接喂给物理引擎重力补偿，抵消那 2 毫秒物理步长里重力造成的微小下坠
            env.mj_data.qfrc_applied[:] = env.mj_data.qfrc_bias

            mujoco.mj_step(env.mj_model, env.mj_data)
            env.mj_viewer.sync()

            # 【关键】让代码稍微等一等，保持与真实时间同步，否则动画一闪而过
            time_until_next_step = env.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            # 允许有0.001弧度（0.05度）的极微小误差
            if np.allclose(env.mj_data.qpos[:6], end_joints_5, atol=1e-3):
                print("机械臂已成功到达T5点")
                break

    # T5 -> T6(目标点[0.2, 0.2, T5.t[2]] )

    T6 = sm.SE3.Rt(sm.SO3.Rz(-np.pi / 2) * sm.SO3(T5.R,check=False).norm(), [0.2, 0.2, T5.t[2]])

    print(f"T6:{T6}")

    print("type(end_joints_5):",type(end_joints_5))
    # 计算逆运动学
    ikine_joint_angels = env.robot.ikine(T6, q_guess=end_joints_5)
    ikine_joint_list_6 = ikine_joint_angels.tolist()


    start_joints_6 = end_joints_5

    end_joints_6 = np.array(ikine_joint_list_6)
    print(f"end_joints_6: {end_joints_6}")
  

    env.mj_data.qpos[:6] = start_joints_6
    mujoco.mj_forward(env.mj_model, env.mj_data)

    joint_trajectory_6 = JointSpaceTrajectory(start_joints_6, end_joints_6, steps=2000)


    while env.mj_viewer.is_running():
            step_start = time.time()

            waypoint = joint_trajectory_6.get_next_waypoint(env.mj_data.qpos[:6])
            # print("当前目标路径点：", waypoint)
            # 【瞬移大法】强行覆盖位置，速度清零防爆炸
            env.mj_data.qpos[:6] = waypoint
            env.mj_data.qvel[:6] = 0.0
            action[:6] = waypoint

            # 【极其关键】必须告诉电机也瞄准这个位置！不给电机发新指令，它就会和你抢夺控制权
            # 因为你没有给 ctrl 赋新值，电机的目标位置依然停留在旧的地方（比如起始点 0）
            # 于是上帝把机械臂瞬移到新位置，但是电机接受到的命令还是旧位置，于是一百万的Kp就会爆发很大的力量，进行抵抗，所以一直有很大的震荡
            env.mj_data.ctrl[:6] = waypoint

            # 【神级补丁】直接喂给物理引擎重力补偿，抵消那 2 毫秒物理步长里重力造成的微小下坠
            env.mj_data.qfrc_applied[:] = env.mj_data.qfrc_bias

            mujoco.mj_step(env.mj_model, env.mj_data)
            env.mj_viewer.sync()

            # 【关键】让代码稍微等一等，保持与真实时间同步，否则动画一闪而过
            time_until_next_step = env.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            # 允许有0.001弧度（0.05度）的极微小误差
            if np.allclose(env.mj_data.qpos[:6], end_joints_6, atol=1e-3):
                print("机械臂已成功到达T6点")
                break


    # T6->T7（下降10cm）
    time.sleep(2)
    T7 = sm.SE3.Trans(0.0, 0.0, -0.2) * T6

    print(f"T7:{T7}")

    # 计算逆运动学
    ikine_joint_angels = env.robot.ikine(T7, q_guess=end_joints_6)
    ikine_joint_list_7 = ikine_joint_angels.tolist()


    start_joints_7 = end_joints_6

    end_joints_7 = np.array(ikine_joint_list_7)
    print(f"end_joints_7: {end_joints_7}")
  

    env.mj_data.qpos[:6] = start_joints_7
    mujoco.mj_forward(env.mj_model, env.mj_data)

    joint_trajectory_7 = JointSpaceTrajectory(start_joints_7, end_joints_7, steps=2000)


    while env.mj_viewer.is_running():
            step_start = time.time()

            waypoint = joint_trajectory_7.get_next_waypoint(env.mj_data.qpos[:6])
            # print("当前目标路径点：", waypoint)
            # 【瞬移大法】强行覆盖位置，速度清零防爆炸
            env.mj_data.qpos[:6] = waypoint
            env.mj_data.qvel[:6] = 0.0
            action[:6] = waypoint

            # 【极其关键】必须告诉电机也瞄准这个位置！不给电机发新指令，它就会和你抢夺控制权
            # 因为你没有给 ctrl 赋新值，电机的目标位置依然停留在旧的地方（比如起始点 0）
            # 于是上帝把机械臂瞬移到新位置，但是电机接受到的命令还是旧位置，于是一百万的Kp就会爆发很大的力量，进行抵抗，所以一直有很大的震荡
            env.mj_data.ctrl[:6] = waypoint

            # 【神级补丁修正版】只给机械臂的6个电机做重力补偿
            env.mj_data.qfrc_applied[:6] = env.mj_data.qfrc_bias[:6]
            # 确保其他物体（物块、夹爪）不受残余外力影响，正常受重力下落
            env.mj_data.qfrc_applied[6:] = 0.0

            mujoco.mj_step(env.mj_model, env.mj_data)
            env.mj_viewer.sync()

            # 【关键】让代码稍微等一等，保持与真实时间同步，否则动画一闪而过
            time_until_next_step = env.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            # 允许有0.001弧度（0.05度）的极微小误差
            if np.allclose(env.mj_data.qpos[:6], end_joints_7, atol=1e-3):
                print("机械臂已成功到达T7点")
                break
    
    # 打开夹爪
    # 释放前，确保清除所有额外的重力补偿，让物理法则接管！
    env.mj_data.qfrc_applied[:] = 0.0

    for i in range(1500):
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.mj_data.ctrl[:] = action
        mujoco.mj_step(env.mj_model, env.mj_data)
        env.mj_viewer.sync()
        # 加上微小的睡眠时间，让画面保持与真实物理时间同步，你才能看到物体掉落
        time.sleep(0.002)

    print("打开夹爪已完成") 

    # T7 -> T8(避免碰撞)
    T8 = sm.SE3.Trans(0.0, 0.0, 0.2) * T7

    print(f"T8:{T8}")

    # 计算逆运动学
    ikine_joint_angels = env.robot.ikine(T8, q_guess=end_joints_7)
    ikine_joint_list_8 = ikine_joint_angels.tolist()


    start_joints_8 = end_joints_7

    end_joints_8 = np.array(ikine_joint_list_8)
    print(f"end_joints_8: {end_joints_8}")
  

    env.mj_data.qpos[:6] = start_joints_8
    mujoco.mj_forward(env.mj_model, env.mj_data)

    joint_trajectory_8 = JointSpaceTrajectory(start_joints_8, end_joints_8, steps=2000)


    while env.mj_viewer.is_running():
            step_start = time.time()

            waypoint = joint_trajectory_8.get_next_waypoint(env.mj_data.qpos[:6])
            # print("当前目标路径点：", waypoint)
            # 【瞬移大法】强行覆盖位置，速度清零防爆炸
            env.mj_data.qpos[:6] = waypoint
            env.mj_data.qvel[:6] = 0.0
            action[:6] = waypoint

            # 【极其关键】必须告诉电机也瞄准这个位置！不给电机发新指令，它就会和你抢夺控制权
            # 因为你没有给 ctrl 赋新值，电机的目标位置依然停留在旧的地方（比如起始点 0）
            # 于是上帝把机械臂瞬移到新位置，但是电机接受到的命令还是旧位置，于是一百万的Kp就会爆发很大的力量，进行抵抗，所以一直有很大的震荡
            env.mj_data.ctrl[:6] = waypoint

            # 【神级补丁修正版】只给机械臂的6个电机做重力补偿
            env.mj_data.qfrc_applied[:6] = env.mj_data.qfrc_bias[:6]
            # 确保其他物体（物块、夹爪）不受残余外力影响，正常受重力下落
            env.mj_data.qfrc_applied[6:] = 0.0

            mujoco.mj_step(env.mj_model, env.mj_data)
            env.mj_viewer.sync()

            # 【关键】让代码稍微等一等，保持与真实时间同步，否则动画一闪而过
            time_until_next_step = env.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            # 允许有0.001弧度（0.05度）的极微小误差
            if np.allclose(env.mj_data.qpos[:6], end_joints_8, atol=1e-3):
                print("机械臂已成功到达T8点")
                break
    
    # T8 -> T9(回到初始值)
    

    # 计算逆运动学
    # ikine_joint_angels = env.robot.ikine(T8, q_guess=end_joints_7)
    # ikine_joint_list_8 = ikine_joint_angels.tolist()


    start_joints_9 = end_joints_8

    end_joints_9 = np.array([1.57, 1.57, 0, 1.57, 1.57, 0])
    print(f"end_joints_9: {end_joints_9}")
  

    env.mj_data.qpos[:6] = start_joints_9
    mujoco.mj_forward(env.mj_model, env.mj_data)

    joint_trajectory_9 = JointSpaceTrajectory(start_joints_9, end_joints_9, steps=2000)


    while env.mj_viewer.is_running():
            step_start = time.time()

            waypoint = joint_trajectory_9.get_next_waypoint(env.mj_data.qpos[:6])
            # print("当前目标路径点：", waypoint)
            # 【瞬移大法】强行覆盖位置，速度清零防爆炸
            env.mj_data.qpos[:6] = waypoint
            env.mj_data.qvel[:6] = 0.0
            action[:6] = waypoint

            # 【极其关键】必须告诉电机也瞄准这个位置！不给电机发新指令，它就会和你抢夺控制权
            # 因为你没有给 ctrl 赋新值，电机的目标位置依然停留在旧的地方（比如起始点 0）
            # 于是上帝把机械臂瞬移到新位置，但是电机接受到的命令还是旧位置，于是一百万的Kp就会爆发很大的力量，进行抵抗，所以一直有很大的震荡
            env.mj_data.ctrl[:6] = waypoint

            # 【神级补丁修正版】只给机械臂的6个电机做重力补偿，这样电机不会在你瞬移的时候爆发巨大的反作用力，不会发生震荡
            env.mj_data.qfrc_applied[:6] = env.mj_data.qfrc_bias[:6]
            # 确保其他物体（物块、夹爪）不受残余外力影响，正常受重力下落
            env.mj_data.qfrc_applied[6:] = 0.0

            mujoco.mj_step(env.mj_model, env.mj_data)
            env.mj_viewer.sync()

            # 【关键】让代码稍微等一等，保持与真实时间同步，否则动画一闪而过
            time_until_next_step = env.mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
            # 允许有0.001弧度（0.05度）的极微小误差
            if np.allclose(env.mj_data.qpos[:6], end_joints_9, atol=1e-3):
                print("机械臂已成功到达T9点")
                break
    

    time.sleep(2)

    # 关闭环境
    env.close()
