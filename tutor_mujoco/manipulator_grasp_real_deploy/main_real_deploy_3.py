"""
和源码main.py并列,互相独立;

实机抓取：
相机realsense D435IF获取深度图像和RGB图像,输入graspnet网络,生成抓取位姿；
jaka机械臂内置函数逆运动学,计算末端位姿
控制灵巧手进行抓取

目前260531
仅实现读取相机图像,yolo检测物体生成mask,调用graspnet生成抓取位姿并可视化

260614
手眼标定得到相机外参, gg抓取位姿转换到基座标系下

260627
手眼标定得到相机外参, gg抓取位姿转换到基座标系下, 变换gg得到末端位姿,求机械臂逆解,让机械臂臂末端移动到对应位置

点云可视化抓取位姿和坐标系，方便进行判断


注意：核对并修改相机外参,和内参(搜索intrinsics)

Run command example:
conda activate mujoco_graspnet
    cd 到当前目录下，   
    python main_real_deploy.py
"""

import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
from PIL import Image
import spatialmath as sm
from spatialmath import SE3

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

import cv2
import pyrealsense2 as rs

import jkrc  


def get_net():
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    checkpoint_path = 'logs/log_rs/checkpoint-rs.tar'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()
    return net


def load_yolo_model(
    weights='yolov5s',
    conf=0.35,
    iou=0.45,
    device=None,
):  
    """
    加载 YOLOv5 模型。

    weights='yolov5s'：使用 torch.hub 的 COCO 预训练模型。
    weights='/path/to/best.pt'：使用你自己训练的模型。
    """
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # YOLOv5 源码本地目录（torch hub 缓存位置）
    hub_dir = os.path.expanduser('~/.cache/torch/hub/ultralytics_yolov5_master')
    if not os.path.isdir(hub_dir):
        raise FileNotFoundError(
            f"未找到本地 YOLOv5 仓库: {hub_dir}\n"
            "请先运行一次网络版的 torch.hub.load 或手动克隆:\n"
            "git clone https://github.com/ultralytics/yolov5 ~/.cache/torch/hub/ultralytics_yolov5_master"
        )

    # 如果 weights 是本地 .pt 文件，保持；否则作为模型名自动拼接
    if not weights.endswith('.pt'):
        weights = os.path.join(hub_dir, f'{weights}.pt')  # 如 yolov5s.pt

    # 使用本地源码加载，完全不联网
    model = torch.hub.load(
        hub_dir,
        'custom',
        path=weights,
        source='local',
        trust_repo=True
    )

    model.to(device)
    model.conf = conf
    model.iou = iou
    model.eval()

    # print(f"[YOLO] loaded weights: {weights}")
    # print(f"[YOLO] device: {device}, conf={conf}, iou={iou}")
    return model

def detect_yolo_bbox(yolo_model, color_rgb, target_class=None, min_conf=0.35):
    """
    输入：
        color_rgb: RGB 图像, H x W x 3, uint8
        target_class: 目标类别名，例如 'bottle'/'cup'。如果为 None,则取置信度最高的检测框。

    返回：
        bbox: [x1, y1, x2, y2]
        det_info: dict
    """
    
    results = yolo_model(color_rgb)

    det = results.xyxy[0]
    if det is None or len(det) == 0:
        return None, None

    det = det.detach().cpu().numpy()
    names = yolo_model.names

    candidates = []
    for row in det:
        x1, y1, x2, y2, conf, cls_id = row
        cls_id = int(cls_id)
        cls_name = names[cls_id]

        if conf < min_conf:
            continue

        if target_class is not None and cls_name != target_class:
            continue

        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        candidates.append({
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'conf': float(conf),
            'cls_id': cls_id,
            'cls_name': cls_name,
            'area': float(area),
        })

    if len(candidates) == 0:
        return None, None

    # 自动选择置信度最高的物体;key=func用于指定排序的参数
    # best即det_info，其中包含了bbox、conf、cls_id、cls_name、area等信息
    best = max(candidates, key=lambda d: d['conf'])
    return best['bbox'], best

# yolo的矩形框转成掩码，稍微扩展一点区域
def bbox_to_mask(depth, bbox, pad=8):
    h, w = depth.shape
    x1, y1, x2, y2 = bbox

    x1 = int(max(0, x1 - pad))
    y1 = int(max(0, y1 - pad))
    x2 = int(min(w, x2 + pad))
    y2 = int(min(h, y2 + pad))

    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask

# 深度过滤，同时根据物体深度中值过滤掉错误深度的点云
def refine_mask_by_depth(depth, object_mask, min_depth=0.15, max_depth=1.5, depth_margin=0.08):
    valid_depth = (depth > min_depth) & (depth < max_depth)
    candidate = object_mask & valid_depth

    if np.sum(candidate) < 50:
        return candidate

    z_values = depth[candidate]
    z_med = np.median(z_values)

    depth_refined = valid_depth & (depth > z_med - depth_margin) & (depth < z_med + depth_margin)
    refined_mask = object_mask & depth_refined
    return refined_mask

def clean_mask(mask, kernel_size=5, erode_iter=1):
    """形态学处理：填洞 + 去边缘毛刺。"""
    mask_uint8 = mask.astype(np.uint8) * 255
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    if erode_iter > 0:
        mask_uint8 = cv2.erode(mask_uint8, kernel, iterations=erode_iter)

    return mask_uint8 > 0

def draw_detection_overlay(color_bgr, bbox=None, det_info=None, object_mask=None):
    """在 RGB 窗口上显示 bbox 和 mask 叠加效果。"""
    vis = color_bgr.copy()

    if object_mask is not None:
        overlay = vis.copy()
        overlay[object_mask] = (0, 255, 0)
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if det_info is not None:
            label = f"{det_info['cls_name']} {det_info['conf']:.2f}"
            cv2.putText(vis, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(vis, "Press s: YOLO+mask+GraspNet | q/ESC: quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    return vis



def start_realsense(width=640, height=480, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    profile = pipeline.start(config)

    # 获取深度单位：RealSense 的 z16 深度不是米，需要乘 depth_scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"[RealSense] depth_scale = {depth_scale}")

    # 对齐 depth 到 color，保证 RGB 和 Depth 像素对应
    align = rs.align(rs.stream.color)

    return pipeline, align, depth_scale

def get_realsense_imgs(pipeline,
    align,
    depth_scale,
    yolo_model=None,
    target_class=None,
    warmup=30,
    show=True,
    min_depth=0.15,
    max_depth=2.0,
):
    """
    返回：
        imgs['img']         : RGB 图像, uint8, H x W x 3
        imgs['depth']       : 深度图, float32, 单位 m, H x W
        imgs['intrinsic']   : RealSense 内参
        imgs['object_mask'] : YOLO bbox + depth refined mask, bool, H x W
        imgs['bbox']        : YOLO 检测框
        imgs['det_info']    : YOLO 检测类别、置信度等信息
    """

    # 前几帧不稳定，先跳过
    for _ in range(warmup):
        pipeline.wait_for_frames()

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # BGR, uint8
        color_bgr = np.asanyarray(color_frame.get_data())

        # OpenCV 是 BGR，但 Open3D/一般网络可视化习惯 RGB
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        # RealSense 原始 depth 是 uint16，需要转成米
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_m = depth_raw.astype(np.float32) * depth_scale


        # 读取相机真实内参
        # intr = color_frame.profile.as_video_stream_profile().intrinsics
        # intrinsic = np.array([
        #     [intr.fx, 0.0, intr.ppx],
        #     [0.0, intr.fy, intr.ppy],
        #     [0.0, 0.0, 1.0]
        # ], dtype=np.float32)

        # merged计算得到的K
        intrinsic = np.array([
            [600.8591, 0.0,       322.1869],
            [0.0,      601.8001,  246.7771],
            [0.0,      0.0,       1.0     ]
        ], dtype=np.float64)

        bbox = None
        det_info = None
        object_mask = None

        if yolo_model is not None:
            bbox, det_info = detect_yolo_bbox(
                yolo_model=yolo_model,
                color_rgb=color_rgb,
                target_class=target_class,
                min_conf=getattr(yolo_model, 'conf', 0.35),
            )

            if bbox is not None:
                bbox_mask = bbox_to_mask(depth_m, bbox, pad=8)
                object_mask = refine_mask_by_depth(
                    depth=depth_m,
                    object_mask=bbox_mask,
                    min_depth=min_depth,
                    max_depth=max_depth
                )
                object_mask = clean_mask(object_mask, kernel_size=5, erode_iter=0)

        if show:
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_raw, alpha=0.03),
                cv2.COLORMAP_JET
            )
            # 带检测框和mask的det_vis算出来
            det_vis = draw_detection_overlay(
                color_bgr=color_bgr,
                bbox=bbox,
                det_info=det_info,
                object_mask=object_mask,
            )
            images = np.hstack((det_vis, depth_colormap))
            cv2.imshow("RealSense + YOLO bbox/mask", images)

            key = cv2.waitKey(1)
            # 按 q 或 ESC 退出程序
            if key & 0xFF == ord('q') or key == 27:
                raise KeyboardInterrupt

            # 按 s 拍一帧送入 GraspNet
            if key & 0xFF != ord('s'):
                continue
        
        if object_mask is None or bbox is None:
            print("[YOLO] 当前帧没有检测到目标，未送入 GraspNet。请调整物体位置/target_class/conf 后再按 s。")
            continue
    
        imgs = {
            "img": color_rgb,
            "depth": depth_m,
            "intrinsic": intrinsic,
            'object_mask': object_mask,
            'bbox': bbox,
            'det_info': det_info,
        }

        print(f"[YOLO] selected: {det_info}")
        print(f"[Mask] object_mask pixels: {np.sum(object_mask)}")
        return imgs


def build_o3d_cloud(points, colors):
    cloud_o3d = o3d.geometry.PointCloud()
    cloud_o3d.points = o3d.utility.Vector3dVector(points.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))
    return cloud_o3d

def get_and_process_data(imgs):
    num_point = 20000

    color = imgs['img'].astype(np.float32) / 255.0
    depth = imgs['depth'].astype(np.float32)   # 单位：米
    height, width = depth.shape

    # 优先使用 RealSense 的真实内参
    if 'intrinsic' in imgs:
        intrinsic = imgs['intrinsic']
    else:
        # 兼容原来的 MuJoCo 仿真图像
        fovy = np.pi / 4
        intrinsic = np.array([
            [height / (2.0 * np.tan(fovy / 2.0)), 0.0, width / 2.0],
            [0.0, height / (2.0 * np.tan(fovy / 2.0)), height / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]
    factor_depth = 1.0

    camera = CameraInfo(width, height, fx, fy, cx, cy, factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # 完整场景点云：用于碰撞检测
    scene_mask = (depth > 0.15) & (depth < 2.0)
    scene_points = cloud[scene_mask]
    scene_colors = color[scene_mask]
    scene_cloud_o3d = build_o3d_cloud(scene_points, scene_colors)

    # 目标物体点云：用于 GraspNet 输入
    if 'object_mask' not in imgs or imgs['object_mask'] is None:
        raise RuntimeError("imgs 中没有 object_mask，无法只针对目标物体生成抓取。")

    object_mask = imgs['object_mask'].astype(bool)
    if object_mask.shape != depth.shape:
        raise RuntimeError(f"object_mask.shape={object_mask.shape}, depth.shape={depth.shape}，二者必须一致。")
    

    object_mask = object_mask & scene_mask
    object_points = cloud[object_mask]
    object_colors = color[object_mask]

    if len(object_points) < 100:
        raise RuntimeError("目标物体有效点太少，可能是 YOLO 框不准、深度缺失、距离阈值不合适或 mask 被腐蚀过度。")
    if len(object_points) >= num_point:
        idxs = np.random.choice(len(object_points), num_point, replace=False)
    else:
        idxs1 = np.arange(len(object_points))
        idxs2 = np.random.choice(len(object_points), num_point - len(object_points), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
        
    cloud_sampled = object_points[idxs]
    color_sampled = object_colors[idxs]

    object_cloud_o3d = build_o3d_cloud(object_points, object_colors)

    end_points = {}
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cloud_sampled = cloud_sampled.to(device)

    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, object_cloud_o3d, scene_cloud_o3d


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

def vis_grasps(gg, T_base_cam, T_base_flange, object_cloud, scene_cloud=None):
    
    # -------- 可视化准备 --------
    # 获取物体点云和场景点云（generate_grasps 内部已生成，generate_grasps 返回点云)
    
    # 创建可视化几何体列表
    vis_geoms = []
    # 获取物体点云和场景点云（generate_grasps 已经返回）
    # 注意：object_cloud 和 scene_cloud 当前都在相机坐标系下
    # 1. 物体点云：变换到基座系，并降采样
    object_cloud_base = object_cloud.voxel_down_sample(voxel_size=0.005)
    object_cloud_base.transform(T_base_cam)      
    vis_geoms.append(object_cloud_base)

    # 2. GraspNet 夹爪模型：同样变换到基座系
    grippers = gg.to_open3d_geometry_list()
    for g in grippers:
        g.transform(T_base_cam)                     # 变换到基座系
        g.paint_uniform_color([0.5, 0.5, 0.5])      # 灰色
    vis_geoms += grippers

    # 3. 抓取坐标系（相机系下构建，然后变换到基座系）
    T_gg_cam = np.eye(4)
    T_gg_cam[:3, :3] = gg.rotation_matrices[0]
    T_gg_cam[:3, 3]  = gg.translations[0]
    T_gg_base = T_base_cam @ T_gg_cam               # 变换到基座系
    vis_geoms.append(create_coordinate_frame(T_gg_base, size=0.06))
    
    # 5. 法兰目标坐标系（已经在基座系下，直接使用）
    frame_flange = create_coordinate_frame(T_base_flange.A, size=0.06)
    vis_geoms.append(frame_flange)
    
    # 6. 添加基座系原点，用于参考,非常有必要
    frame_base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis_geoms.append(frame_base)

    # 启动可视化
    print("\n显示可视化窗口，请检查抓取姿态（灰色）与目标法兰（彩色轴）。关闭open3d窗口后继续执行程序。\n")
    o3d.visualization.draw_geometries(vis_geoms)
    



def generate_grasps(net, imgs):
    end_points, object_cloud, scene_cloud = get_and_process_data(imgs)
    # graspnet 只看目标物体点云，碰撞检测需要看完整场景点云
    gg = get_grasps(net, end_points)
    gg = collision_detection(gg, np.array(scene_cloud.points))
    gg.nms()
    gg.sort_by_score()
    gg = gg[:1]
    return gg , object_cloud, scene_cloud


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

def build_camera_to_base(R, t):
    """
    根据手眼标定结果构造 T_cam2base (相机系 → 基座系)
    R : 3x3 旋转矩阵
    t : 3x1 平移向量
    返回 4x4 numpy 矩阵
    """
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    return T

def grasp_in_base_frame(gg, T_cam2base, index = 0):
    """
    将 GraspGroup 中第 index 个抓取位姿从相机系转到基座系
    gg : GraspGroup 对象
    index : int
    T_cam2base : 4x4 数组
    返回 sm.SE3 对象（便于后面取平移、旋转）
    """
    # 抓取在相机系下的位姿 (sm.SE3)
    T_gg2cam = sm.SE3.Trans(gg.translations[index]) * sm.SE3(
        sm.SO3.TwoVectors(
            x=gg.rotation_matrices[index][:, 0],
            y=gg.rotation_matrices[index][:, 1]
        )
    )
    
    # 转换
    T_gg2base = T_cam2base @ T_gg2cam.A   # 4x4 矩阵相乘
    
    # # 拆成 R 和 t，用专门的方法构造 SE3
    # R = T_gg2base[:3, :3]
    # t = T_gg2base[:3, 3]
    
    # return sm.SE3.Trans(t) * sm.SE3(sm.SO3(R))                # 转为 SE3 对象，方便提取 xyz/rpy
    # 【修改部分】：直接用 4x4 齐次变换矩阵构造 SE3
    # 加入 check=False 防止由于手眼标定浮点数精度导致的“非正交矩阵”报错
    return sm.SE3(T_gg2base, check=False)


# 求抓取点对应的机械臂末端位置（相对于机械臂基座）
def get_T_base_flange(T_gg2base:SE3)-> SE3:
    """
    
    # T_wo: gg传过来的抓取位姿T_wo
    # 得到T_wo_modi,计算出来T_flange_des
    """
    # ikpy求逆解需要的是机械臂末端相对于基座的位姿矩阵
    # T_wo是代表的夹爪末端中心点对应的位姿，我们实际需要末端法兰的位姿，求逆解，机械臂移动到对应位置，控制夹爪抓取（夹爪渲染的时候是一个显示）
    # T_base_flange是经过调整后的位姿，才是机械臂末端应该去的位姿；
    # T_base_flange如何得到：传入的T_gg2base实际要沿自身T_gg2base的x轴移动0.13m,再绕自身y轴旋转90度，再绕自身z轴旋转90度，才是机械臂末端正确的位姿
    # T_base_flange世界坐标系下的机械臂末端位姿，需要转换的机械臂基坐标系下，这样才能求逆解
    # 经过实际观察，夹爪在绕再绕自身y轴旋转90度之后，应该再绕自身z轴旋转90度，这样夹取姿势比较合理

    # # 定义沿自身 X 轴平移 0.13m 的变换矩阵
    T_trans_x = sm.SE3.Tx(-0.13)

    # # 定义沿自身 z 轴平移 0.2m 的变换矩阵
    T_trans_z = sm.SE3.Tz(-0.3)

    #  定义绕自身 Y 轴旋转 90 度 (π/2) 的变换矩阵
    T_rot_y = sm.SE3.Ry(np.pi / 2)
    # 定义绕自身 Z 轴旋转 90 度 (π/2) 的变换矩阵
    T_rot_z = sm.SE3.Rz(np.pi / 2)
    # 组合变换：严格按照发生的顺序【右乘】;
    # T_base_flange = T_gg2base * T_rot_y * T_rot_z
    T_base_flange = T_gg2base * T_trans_x * T_trans_z * T_rot_y * T_rot_z
    

    print(f"机械臂末端位姿 T_base_flange:\n{T_base_flange}")
    
    return T_base_flange

def se3_to_jaka_pose(T: sm.SE3):
    """
    将 spatialmath SE3 转换为 JAKA kine_inverse 的笛卡尔位姿
    返回 [x, y, z, rx, ry, rz]
        x, y, z : 毫米 (mm)
        rx, ry, rz : 弧度 (rad)  —— 固定角 RPY（绕基坐标系 X, Y, Z 轴）
    """
    # 平移，单位：米 -> 毫米
    t = T.t                     # shape (3,) 单位 m
    x_mm = t[0] * 1000.0
    y_mm = t[1] * 1000.0
    z_mm = t[2] * 1000.0

    # 姿态：RPY (弧度)
    rpy = T.rpy()               # [roll, pitch, yaw] 单位 rad

    return [x_mm, y_mm, z_mm, rpy[0], rpy[1], rpy[2]]

def jaka_kine_inverse(current_joint_pos, target_tcp_pose):
    
    # # 先登录
    # robot.login()
    # robot.power_on()
    # robot.enable_robot()

    # 求逆解
    ret_ik_test = robot.kine_inverse(current_joint_pos, target_tcp_pose)
    if ret_ik_test[0] == 0:
        print("测试逆解成功，目标关节角度: ", ret_ik_test[1])
    else:
        print("测试逆解失败，错误码: ", ret_ik_test[0])

    # 完成后登出
    # robot.logout()
    return ret_ik_test[1] if ret_ik_test[0]==0 else None

def jaka_get_joint_position():

    # 获取当前关节角度
    # robot.login()#登录  
    ret = robot.get_joint_position()  

    if ret[0] == 0:  
        print("the current joint position is :",ret[1])  
    else:  
        print("some things happend,the errcode is: ",ret[0])  
    
    # robot.logout()  #登出
    return ret[1] if ret[0]==0 else None

def jaka_joint_move(joint_pose):
    
    # robot.login()#登录  
    # robot.power_on() #上电  
    # robot.enable_robot()  

    # 0 绝对 1相对； True 阻塞，False 非阻塞； 0.1rad/s 速度
    robot.joint_move(joint_pose,0,True,0.1) 

    time.sleep(3)
    # robot.logout()

def create_coordinate_frame(T: np.ndarray, size=0.05):
    """
    用 Open3D 创建一个坐标轴（红-X，绿-Y，蓝-Z）
    T : 4x4 齐次变换矩阵
    size : 轴长度
    """
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(T)
    return frame


if __name__ == '__main__':

    R_cam2base = np.array([[-0.9377583,  -0.26024247,  0.22996355],
                            [-0.34524681,  0.62688479,-0.6984412 ],
                            [ 0.03760341, -0.73436321, -0.67771429]])
    
    t_cam2base = np.array([[0.33575064],
                            [0.16561175],
                            [0.6566156 ]])
    T_base_cam = build_camera_to_base(R_cam2base, t_cam2base)

    # 机械臂初始化
    robot = jkrc.RC("192.168.2.155")#返回机器人对象  

    robot.login()
    robot.power_on()
    robot.enable_robot()

    # 如果你只想抓 COCO 里的某一类，写 target_class='bottle' 或 'cup'
    # 如果设为 None，则取 YOLO 检测置信度最高的物体
    TARGET_CLASS = 'bottle'

    # 如果你有自己的训练权重，改成例如：
    YOLO_WEIGHTS = '/home/chark/Python/tutor_mujoco/manipulator_grasp_real_deploy/yolov5s.pt'

    # 初始化网络
    net = get_net()
    yolo_model = load_yolo_model(weights=YOLO_WEIGHTS, conf=0.35, iou=0.45)

    pipeline, align, depth_scale = start_realsense()

    try:
        while True:
            print("[提示] 点击 RealSense 窗口，按 s 采集当前帧并送入 GraspNet。")
            print("[提示] 按 q 或 ESC 退出。")
            print(f"[提示] 当前 TARGET_CLASS = {TARGET_CLASS}，None 表示取置信度最高的检测框。")

            imgs = get_realsense_imgs(
                pipeline=pipeline,
                align=align,
                depth_scale=depth_scale,
                yolo_model=yolo_model,
                target_class=TARGET_CLASS,
                warmup=30,
                show=True,
                min_depth=0.15,
                max_depth=2.0,
            )

            print("[GraspNet] 正在基于 YOLO mask 后的目标点云生成抓取位姿...")

            gg, object_cloud, scene_cloud  = generate_grasps(net, imgs)

            # print("========== GraspNet 输出 ==========")
            # print("抓取位置 translation，单位 m，相机坐标系下:")
            # print(gg.translations[0])

            # print("抓取姿态 rotation matrix，相机坐标系下:")
            # print(gg.rotation_matrices[0])

            # 转换到基座系
            T_base_gg = grasp_in_base_frame(gg, T_base_cam, 0)
            print("\n基座系下抓取位姿:")
            print(T_base_gg)

            T_base_flange = get_T_base_flange(T_base_gg)
            target_tcp_pose = se3_to_jaka_pose(T_base_flange)
            print("目标 TCP 位姿 (JAKA 格式):", target_tcp_pose)

            vis_grasps(gg, T_base_cam, T_base_flange, object_cloud, scene_cloud=None)

        

            # 等待用户按键确认
            print("\n确认此抓取？在终端输入 c 继续，其他键放弃，q 退出程序。")
            key = input(">>> ").strip().lower()
            if key == 'q':
                break
            if key != 'c':
                print("放弃该抓取，重新采集。")
                continue

            

            
            # 获取当前关节角度
            current_joint_pos = jaka_get_joint_position()  

            # 调用逆解（确保 robot 已经登录并使能）
            ikine_joint_pos = jaka_kine_inverse(current_joint_pos, target_tcp_pose)  
            
            jaka_joint_move(ikine_joint_pos)

            print("机械臂移动完成！")

            robot.logout()

           
            break   

        
    except KeyboardInterrupt:
        print("用户退出。")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
