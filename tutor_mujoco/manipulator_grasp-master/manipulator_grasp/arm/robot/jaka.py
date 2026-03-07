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
from spatialmath import SE3
from typing import List

from .robot import Robot


class JakaRobot(Robot):
    def __init__(self, urdf_path: str, ee_frame_name: str):
        # 1. 调用父类构造函数，生成那些兼容上层代码的属性 (如 q0, _base, _tool)
        super().__init__()
        
        # 2. 初始化 Pinocchio
        self.pin_model = pin.buildModelFromUrdf(urdf_path)
        self.pin_data = self.pin_model.createData()
        
        self._dof = self.pin_model.nq  # 覆盖父类的 dof
        self.q0 = pin.neutral(self.pin_model)
        
        # 3. 找到末端执行器
        if not self.pin_model.existFrame(ee_frame_name):
            raise ValueError(f"找不到 Frame: {ee_frame_name}")
        self.ee_frame_id = self.pin_model.getFrameId(ee_frame_name)
        
        # 注意：我们故意不去给 self.robot 赋值 RTB 对象，让它保持 None
        # 因为我们接下来会覆盖掉所有用到 self.robot 的方法！

    # ==========================================
    # 覆盖设置方法：切断与 RTB 的联系，只保留 SE3 属性
    # ==========================================
    def set_base(self, base: np.ndarray):
        self._base = SE3.Trans(base)
        # 删掉父类里的 self.robot.base = self._base，防止报错

    def set_tool(self, tool: SE3):
        self._tool = tool
        # 删掉父类里的 self.robot.tool = self._tool

    def disable_tool(self):
        self._tool = SE3()

    def disable_base(self):
        self._base = SE3()

    # ==========================================
    # 覆盖运动学方法：用 Pinocchio 替换 RTB
    # ==========================================
    def fkine(self, q: np.ndarray) -> SE3:
        """正运动学：Base * Pinocchio_FK * Tool"""
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        # 获取法兰盘位姿
        pin_T = self.pin_data.oMf[self.ee_frame_id]
        T_flange = SE3(pin_T.homogeneous)
        
        return self._base * T_flange * self._tool

    def ikine(self, Tep: SE3) -> np.ndarray:
        """
        逆运动学：Pinocchio 数值解法 (CLIK)
        Tep: 期望的末端世界位姿 (包含了 base 和 tool 的影响)
        """
        # 1. 把目标位姿“剥离” base 和 tool，还原出法兰盘在机器人基坐标系下的期望位姿
        # Tep = base * T_flange * tool  =>  T_flange = base^-1 * Tep * tool^-1
        T_flange_des = self._base.inv() * Tep * self._tool.inv()
        
        # 转换回 Pinocchio 格式
        oMdes = pin.SE3(T_flange_des.A)
        
        # 2. CLIK 数值迭代求解参数
        q = self.q0.copy() if self.q0 is not None else pin.neutral(self.pin_model)
        eps = 1e-4  # 允许的误差精度
        IT_MAX = 1000 # 最大迭代次数
        DT = 1e-1   # 迭代步长
        damp = 1e-12 # 阻尼系数(防奇异点爆炸)
        
        # 3. 迭代循环
        for i in range(IT_MAX):
            pin.forwardKinematics(self.pin_model, self.pin_data, q)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            
            # 当前法兰盘位姿
            oMcurr = self.pin_data.oMf[self.ee_frame_id]
            
            # 计算位姿误差 (SE3 空间里的对数映射)
            err = pin.log(oMcurr.inverse() * oMdes).vector
            
            if np.linalg.norm(err) < eps:
                # 成功收敛，返回结果
                return q
                
            # 计算雅可比矩阵
            J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            
            # 伪逆求解关节速度: dq = J^+ * err (带阻尼最小二乘法)
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            
            # 积分更新关节角
            q = pin.integrate(self.pin_model, q, v * DT)
            
        # 如果循环结束还没收敛，通常返回一个空数组或者报错 (参考 UR5e 的写法)
        print("Warning: IK 求解未收敛！")
        return np.array([])

    # ==========================================
    # 覆盖动力学方法 (如果你在抓取时用到了这些)
    # ==========================================
    def get_gravity(self, q: np.ndarray):
        return pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
        
    def inv_dynamics(self, qs, dqs, ddqs) -> np.ndarray:
        return pin.rnea(self.pin_model, self.pin_data, qs, dqs, ddqs)