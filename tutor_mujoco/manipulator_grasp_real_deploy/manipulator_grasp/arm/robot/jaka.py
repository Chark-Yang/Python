"""
自己重新写的jaka类，使用pinocchio进行运动学求解，切断与rtb的联系，保持接口兼容上层代码

仍然继承自 Robot 基类，
为什么还要继承？
父类中定义了很多接口和属性（如 set_base, set_tool, fkine, ikine 等），这些都是上层代码（如 env）直接调用的。
如果你不继承，那么你就得在 env 里重写一大堆。
继承的话，jaka 就戴上了 Robot 的“面具”，外面依然调用函数，但是不会报错，我们只是把需要重写的函数覆盖掉了，其他不相关的函数和属性都保留了下来。

需要注意：
这个基类是深度绑定了 roboticstoolbox-python (RTB) 和传统的 DH 参数建模的，
fkine 里直接调用了 self.robot.fkine(q)（这里的 self.robot 期望是一个 RTB 对象，
Pinocchio 走的是现代的 URDF (基于拓扑树) 路线。如果你直接继承但不做覆盖，
当你调用 self.set_base() 或 self.fkine() 时，代码会因为找不到 RTB 对象的属性而直接崩溃。

如何操作
重写（Override）**那些会调用 RTB 的方法，把它们的底层运算全部悄悄替换成 Pinocchio。
"""


import numpy as np
import pinocchio as pin
import spatialmath as sm
from spatialmath import SE3
from typing import List

from .robot import Robot
import ikpy.chain


class JakaRobot(Robot):
    def __init__(self, urdf_path: str, ee_frame_name: str):
        # 1. 调用父类构造函数，生成那些兼容上层代码的属性 (如 q0, _base, _tool)
        super().__init__()
        
        # # 2. 初始化 Pinocchio
        # self.pin_model = pin.buildModelFromUrdf(urdf_path)
        # self.pin_data = self.pin_model.createData()
        
        # self._dof = self.pin_model.nq  # 覆盖父类的 dof
        # self.q0 = pin.neutral(self.pin_model)
        
        # # 3. 找到末端执行器
        # if not self.pin_model.existFrame(ee_frame_name):
        #     raise ValueError(f"找不到 Frame: {ee_frame_name}")
        # self.ee_frame_id = self.pin_model.getFrameId(ee_frame_name)
        
        
        # 使用ikpy解析urdf , active_links_mask=[False, True, True, True, True, True, True, False, False]

        self.ikpy_chain = ikpy.chain.Chain.from_urdf_file(urdf_path, base_elements=["Link_00"],active_links_mask=[False, True, True, True, True, True, True])
        print(f"Links in ikpy_chain: {len(self.ikpy_chain.links)}")


    # ==========================================
    # 覆盖设置方法：切断与 RTB 的联系，只保留 SE3 属性
    # ==========================================
    def set_base(self, base: np.ndarray):
        self._base = sm.SE3.Trans(base)
        print(f"JakaRobot: base set to {self._base}")

    def set_tool(self, tool: SE3):
        self._tool = tool
        # 删掉父类里的 self.robot.tool = self._tool

    def disable_tool(self):
        self._tool = sm.SE3()

    def disable_base(self):
        self._base = sm.SE3()

    # ==========================================
    # 覆盖运动学方法：用 Pinocchio 替换 RTB
    # ==========================================
    def fkine(self, q: np.ndarray) -> SE3:
        # """正运动学：Base * Pinocchio_FK * Tool"""
        # pin.forwardKinematics(self.pin_model, self.pin_data, q)
        # pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        # # 获取法兰盘位姿
        # pin_T = self.pin_data.oMf[self.ee_frame_id]
        # T_flange = SE3(pin_T.homogeneous)
        
        q1 = [0] + q.tolist()  
        ikpy_T = self.ikpy_chain.forward_kinematics(q1)

        return self._base * ikpy_T * self._tool

    # 求抓取点对应的机械臂末端位置（相对于机械臂基座）
    def get_T_flange_des(self,T_wo:SE3)-> SE3:
        """
        
        # T_wo: gg传过来的抓取位姿T_wo
        # 得到T_wo_modi,计算出来T_flange_des

        """
        # ikpy求逆解需要的是机械臂末端相对于基座的位姿矩阵
        # T_wo是代表的夹爪末端中心点对应的位姿，我们实际需要末端法兰的位姿，求逆解，机械臂移动到对应位置，控制夹爪抓取（夹爪渲染的时候是一个显示）
        # T_wo_modi是经过调整后的位姿，才是机械臂末端应该去的位姿；
        # T_wo_modi如何得到：传入的T_wo实际要沿自身T_wo 的x轴移动0.13m,再绕自身y轴旋转90度，才是机械臂末端正确的位姿
        # T_wo_modi世界坐标系下的机械臂末端位姿，需要转换的机械臂基坐标系下，这样才能求逆解
        
        # 定义沿自身 X 轴平移 0.13m 的变换矩阵
        T_trans_x = sm.SE3.Tx(-0.13)

        #  定义绕自身 Y 轴旋转 90 度 (π/2) 的变换矩阵
        T_rot_y = sm.SE3.Ry(np.pi / 2)

        # 组合变换：严格按照发生的顺序【右乘】;
        T_wo_modi = T_wo * T_trans_x * T_rot_y 

        t_base = self._base

        # t_base.inv()是T_world_base的逆，即T_base_world，T_wo_modi是T_world_flange
        # 2者相乘得到T_base_flange
        # 夹爪应该绕自身z轴旋转90度，这样夹取姿势比较合理
        T_flange_des = t_base.inv() * T_wo_modi * sm.SE3.Rz(np.pi / 2)

        self._T_wo_modi = T_wo_modi
        self._T_flange_des = T_flange_des

        # print(f"T_flange_des:\n{self._T_flange_des}")
        return self._T_flange_des


    def ikine(self, Tep: SE3, q_guess: np.ndarray = None) -> np.ndarray:
        """
        
        # Tep: 求逆解时期望的机械臂末端世界坐标系下的位姿
        #返回机械臂求完逆解之后的关节角度
        """
        # ikpy求逆解需要的是机械臂末端相对于基座的位姿矩阵--T_b_flange(b代表base基座标系,flange代表末端法兰) 
        # 传入的是世界坐标系下的机械系臂末端位姿，函数内部先转换成基座标系的末端位姿，再求逆解
        # Tep = T_wb * T_b_flange --->  t_base.inv() * Tep = T_b_flange 
        
        t_base = self._base
        T_b_flange = t_base.inv() * Tep 

        # print("T_b_flange:\n", T_b_flange)

        # 动态设置初始猜测值
        if q_guess is None:
            ref_pos = [0, 0.3, 0.7, -1, 2.0, 1.57, 0]
        else:
            # ikpy_chain 包含了 base (0)，所以要在前面补一个 0
            ref_pos = np.concatenate(([0], q_guess))


        # 显式指定 orientation_mode='all'，强制要求解算器兼顾姿态
        joint_angles = self.ikpy_chain.inverse_kinematics(
            target_position = T_b_flange.t, 
            target_orientation = T_b_flange.R,
            orientation_mode = 'all', 
            initial_position = ref_pos
        )
        
        return joint_angles[1:]

    # def get_T_flange_des(self):






    #     print(f"T_flange_des:\n{self._T_flange_des}")
    #     return self._T_flange_des

        # # 转换回 Pinocchio 格式
        # oMdes = pin.SE3(T_flange_des.A)
        
        # # 2. CLIK 数值迭代求解参数
        # q = self.q0.copy() if self.q0 is not None else pin.neutral(self.pin_model)
        # eps = 1e-4  # 允许的误差精度
        # IT_MAX = 1000 # 最大迭代次数
        # DT = 1e-1   # 迭代步长
        # damp = 1e-12 # 阻尼系数(防奇异点爆炸)
        
        # # 3. 迭代循环
        # for i in range(IT_MAX):
        #     pin.forwardKinematics(self.pin_model, self.pin_data, q)
        #     pin.updateFramePlacements(self.pin_model, self.pin_data)
            
        #     # 当前法兰盘位姿
        #     oMcurr = self.pin_data.oMf[self.ee_frame_id]
            
        #     # 计算位姿误差 (SE3 空间里的对数映射)
        #     err = pin.log(oMcurr.inverse() * oMdes).vector
            
        #     if np.linalg.norm(err) < eps:
        #         # 成功收敛，返回结果
        #         return q
                
        #     # 计算雅可比矩阵
        #     J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            
        #     # 伪逆求解关节速度: dq = J^+ * err (带阻尼最小二乘法)
        #     v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            
        #     # 积分更新关节角
        #     q = pin.integrate(self.pin_model, q, v * DT)
            
        # # 如果循环结束还没收敛，通常返回一个空数组或者报错 (参考 UR5e 的写法)
        # print("Warning: IK 求解未收敛！")
        # return np.array([])

    # ==========================================
    # 覆盖动力学方法 (如果你在抓取时用到了这些)
    # ==========================================
    def get_gravity(self, q: np.ndarray):
        return pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
        
    def inv_dynamics(self, qs, dqs, ddqs) -> np.ndarray:
        return pin.rnea(self.pin_model, self.pin_data, qs, dqs, ddqs)