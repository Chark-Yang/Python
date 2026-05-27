import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import os


# 预抓取姿势，对应手部关节角度，在reset()里进行设置，
# hand_root的3个平移 + pinky4个关节 + ring4个关节 + middle4个关节 + index4个关节 + thumb5个关节 = 24维
# pinky4个关节分别是：mcp_roll, mcp_pitch, pip, dip;其中mcp_roll固定，pip和dip给定值（微微弯曲），mcp_pitch是强化学习的主要控制目标
# thumb5个关节分别是：thumb_cmc_yaw, thumb_cmc_roll, thumb_cmc_pitch, thumb_mcp, thumb_dip;除thumb_cmc_pitch之外，其他4个关节都固定（微微弯曲），thumb_cmc_pitch是强化学习的主要控制目标
# pinky给值：0 0# 0.6 0.5；thumb给值：0.343 1.22 0# 0.41 0.549
#  手给值= [0 0 0  0 0# 0.6 0.5  0 0# 0.6 0.5  0 0# 0.6 0.5  0 0# 0.6 0.5   0.343 1.22 0# 0.41 0.549]
# 手给值 = [0 0 0  0 0 0.6 0.5  0 0 0.6 0.5  0 0 0.6 0.5  0 0 0.6 0.5   0.343 1.22 0 0.41 0.549]
class LinkerHandEnv(gym.Env):
    """
    Linker L20 灵巧手残差强化学习环境 (基于 Gymnasium)
    """
    def __init__(self, xml_path=None, max_steps=500, render_mode=None):
        super().__init__()
        
        # --- 增加这行，Gymnasium 录制视频需要的 metadata， metadata相当于与gym沟通的协议---
        self.metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
        self.render_mode = render_mode

        # 1. 加载 MuJoCo 模型与数据
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # --- 新增渲染器初始化 ---
        if self.render_mode == "rgb_array":
            # 这里的宽高可以自己调，640x480 是比较兼顾速度和清晰度的
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        
        self.max_steps = max_steps
        self.current_step = 0
        
        # 2. 定义动作空间 (Action Space)
        # 共有 24个位置驱动器。RL 算法喜欢对称且归一化的动作 [-1, 1]
        # self.num_actions = self.model.nu 
        # print(f"Number of actions: {self.num_actions}")

        self.num_actions = 5
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        # 获取底层驱动器的真实物理限位 (ctrlrange)，用于后续的动作反归一化
        self.ctrl_ranges = self.model.actuator_ctrlrange


        # 预先找到这 5 个由 RL 控制的关节在 24 个 actuator 中的索引编号
        # 假设我们控制的是每个手指的 mcp_pitch (负责主要弯曲的根部关节)
        self.rl_action_indices = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "thumb_cmc_pitch_pos"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "index_mcp_pitch_pos"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "middle_mcp_pitch_pos"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "ring_mcp_pitch_pos"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pinky_mcp_pitch_pos")
        ]

        # print("RL 控制的 actuator 索引:", self.rl_action_indices)

        # 定义一个全长 的基础指令数组 (预设姿态 Pre-shape)
        self.base_ctrl = np.zeros(self.model.nu)

        # 为 PIP 和 DIP 关节设置一个微弯曲的固定角度 (例如 0.5 弧度，约 30 度)
        # 你可以通过打印 model.actuator_names 来查看所有名字并赋值
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if "pip_pos" in name:
                self.base_ctrl[i] = 0.6  # 食指等4指pip
            elif "dip_pos" in name and "thumb" not in name:
                self.base_ctrl[i] = 0.5  # 食指等4指dip
            elif "mcp_roll_pos" in name:
                self.base_ctrl[i] = 0    # 食指等4指mcp_roll
            elif "thumb_mcp_pos" in name:
                self.base_ctrl[i] = 0.41   # 拇指mcp_pos
            elif "thumb_dip_pos" in name:
                self.base_ctrl[i] = 0.549  # 拇指dip_pos
            elif "cmc_yaw_pos" in name:
                self.base_ctrl[i] = 0.343  # 拇指cmc_yaw_pos
            elif "cmc_roll_pos" in name:
                self.base_ctrl[i] = 1.22   # 拇指cmc_roll_pos
            else:
                self.base_ctrl[i] = 0.0  # 强化学习控制的5个关节在0

        # print("预设姿态 base_ctrl:", self.base_ctrl)

        # 获取 6 个手腕电机的索引
        self.wrist_actuators = [
            "wrist_x_pos", "wrist_y_pos", "wrist_z_pos", 
            "wrist_roll_pos", "wrist_pitch_pos", "wrist_yaw_pos"
        ]
        self.wrist_act_indices = []
        for act_name in self.wrist_actuators:
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            if act_id != -1:
                self.wrist_act_indices.append(act_id)



        # 3. 定义状态空间 (Observation Space)
        # 状态包括：所有关节角度(qpos) + 所有关节速度(qvel) + 所有的传感器数据(触觉/力觉)
        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

    def _get_obs(self):
        """
        获取当前状态观测值 (Observation)
        """
        # 关节角度 (21维)
        qpos = self.data.qpos.copy()
        # 关节角速度 (21维)
        qvel = self.data.qvel.copy()
        # 传感器数据 (自动读取你在 XML 中定义的所有触觉/力觉传感器)
        sensor_data = self.data.sensordata.copy()

        # 获取水杯在世界坐标系中的 3D 绝对坐标 (X, Y, Z)
        # 注意："Cylinder" 是你在 scene.xml 里给杯子 body 命的名
        # obj_pos = self.data.body("Cylinder").xpos.copy()
        
        # 获取手掌根部在世界坐标系中的 3D 绝对坐标
        # hand_pos = self.data.body("hand_root").xpos.copy()
        
        # 计算相对位置向量 (引导网络理解方向)
        # rel_pos = obj_pos - hand_pos
        
        # 将它们拼接成一个一维 numpy 数组送给神经网络，关节状态 + 触觉传感器数据 +目标位置 + 当前手位置 + 相对距离
        # obs = np.concatenate([qpos, qvel, sensor_data, obj_pos, hand_pos, rel_pos])
        obs = np.concatenate([qpos, qvel, sensor_data])
        return obs.astype(np.float32)

    def render(self):
        """返回当前帧的 RGB 图像矩阵供 Gym 录像"""
        if self.render_mode == "rgb_array":
            # 摄像机捕捉最新的物理位置
            self.renderer.update_scene(self.data, camera="main_cam")
            # 按下快门，输出numpy矩阵图片
            return self.renderer.render()


    def reset(self, seed=None, options=None):
        """
        环境重置，每个 Episode 开始时调用
        """
        super().reset(seed=seed)
        self.current_step = 0
        
        # 重置物理状态为初始零位
        mujoco.mj_resetData(self.model, self.data)
        
        # （可选）在这里可以给关节加入一点随机的初始噪声，增加鲁棒性
        # self.data.qpos[:] += self.np_random.uniform(-0.05, 0.05, self.model.nq)

        mujoco.mj_forward(self.model, self.data) # 更新物理状态

        # 核心逻辑：RL 训练前的“预抓取轨迹”执行
        # 阶段一：高空姿态对齐 (假设用 2秒完成， timestep=0.002, 需要 1000 步)
        phase1_pose = [0.05, 0.0, 0.0, 1.57, 1.57, 0.0]
        for idx, val in zip(self.wrist_act_indices, phase1_pose):
            self.base_ctrl[idx] = val
            
        self.data.ctrl[:] = self.base_ctrl
        for _ in range(1000): 
            mujoco.mj_step(self.model, self.data)

        # 阶段二：垂直下降抓取 (假设用 1秒完成， 需要 500 步)
        phase2_pose = [0.05, 0.0, -0.55, 1.57, 1.57, 0.0] 
        for idx, val in zip(self.wrist_act_indices, phase2_pose):
            self.base_ctrl[idx] = val
            
        self.data.ctrl[:] = self.base_ctrl
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data) # 更新物理状态

        
        obs = self._get_obs()
        info = {} # 可以用来传递额外的调试信息
        
        return obs, info

    def step(self, action):
        """
        与环境交互一步
        action: 来自神经网络的动作指令，范围 [-1, 1]
        """
        self.current_step += 1

        # 1. 初始化发送给 MuJoCo 的 21 维指令数组（先用预设姿态打底）
        final_ctrl = self.base_ctrl.copy()
        
        # 2. 将神经网络输出的 5 维 action 映射到真实物理范围
        for idx, act_val in zip(self.rl_action_indices, action):
            min_ctrl = self.ctrl_ranges[idx, 0]
            max_ctrl = self.ctrl_ranges[idx, 1]
            # 把 [-1, 1] 映射到 [min_ctrl, max_ctrl]
            scaled_act = min_ctrl + (act_val + 1.0) * 0.5 * (max_ctrl - min_ctrl)
            
            # 将 RL 的决策覆盖到基础指令对应的索引上
            final_ctrl[idx] = scaled_act
        
        # 将计算好的真实目标角度发给 MuJoCo 控制器
        self.data.ctrl[:] = final_ctrl
        
        # 2. 步进物理引擎 (可以多次 step 模拟更高的控制频率)
        # 假设 XML 的 timestep 是 0.002，这里 step 5次就是 0.01秒 (100Hz 的控制频率)
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
            
        # 3. 获取新状态
        obs = self._get_obs()
        
        # 4. 计算奖励 (Reward) - 目前是占位符，我们需要根据任务设计
        reward = self._compute_reward(obs, action)
        
        # 5. 判断是否结束 (Termination/Truncation)
        terminated = False # 任务是否成功完成/失败
        truncated = self.current_step >= self.max_steps # 是否达到最大步数超时
        
        info = {}
        
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, obs, action):
        """
        奖励函数设计 (Reward Engineering) - 论文的创新点通常在这里！
        """
        # reward = 0.0

        # # 1. 获取最新坐标
        # obj_pos = self.data.body("Cylinder").xpos.copy()
        # hand_pos = self.data.body("hand_root").xpos.copy()

        # 2. 计算手和杯子的三维空间直线距离
        # dist = np.linalg.norm(obj_pos - hand_pos)

        # 3. 距离奖励核心公式：使用负指数映射。
        # 距离越小，这个值越接近 1；距离越远，越接近 0。这能给网络一个非常平滑的梯度引导。
        # reward_dist = np.exp(-5.0 * dist)

        # 4. 动作平滑惩罚：微微惩罚剧烈运动，防止手抽风
        # action_penalty = -0.01 * np.sum(np.square(action))

        # # 5. （可选）终极奖励：如果手真的靠近到杯子 5 厘米以内，给一个暴击分！
        # reward_reach = 0.0
        # if dist < 0.05:
        #     reward_reach = 5.0
        
        reward_sensor = 0.0
        # 示例 2: 触觉奖励 (Touch Reward) - 如果传感器检测到力，给予巨大奖励
        # 假设 sensordata 里的第一个值是食指的力传感器

        reward_sensor = 0.0
        
        # 读取全部 5 个传感器的受力数据 (假设你的 sensor 标签里依次定义了食指到大拇指)
        # 传感器顺序: [index, middle, ring, pinky, thumb]
        touch_forces = np.array(self.data.sensordata[:5])
        
        # 过滤掉物理引擎的数值底噪 (受力大于 0.01 才算真正碰到)
        active_touches = touch_forces > 0.01 
        num_touching_fingers = np.sum(active_touches) # 统计有几根手指碰到了

        if num_touching_fingers > 0:
            # A. 接触数量奖励：每多一根手指碰到，给 1.0 分基础分
            reward_sensor += num_touching_fingers * 1.0
            
            # B. 连续力觉奖励：鼓励出力，但防止“死里捏”
            # 使用 np.log1p(x) 即 ln(1+x)，把巨大的物理力映射到平滑的小数值区间
            # clip 限制最大受力，防止网络学会把杯子捏爆来刷分
            force_reward = np.sum(np.log1p(np.clip(touch_forces, 0, 10.0))) * 0.1
            reward_sensor += force_reward

            # C. 协同抓取暴击分 (Synergy / Force Closure Bonus) 【论文核心创新点】
            # 在机器人学中，最稳固的抓取必须是“大拇指”和“其他手指”形成对立夹击
            thumb_touching = active_touches[4] # 大拇指是最后一个传感器
            other_fingers_touching = np.sum(active_touches[:4])
            
            if thumb_touching and other_fingers_touching > 0:
                reward_sensor += 5.0  # 形成对立抓取，给大分！
                
            if num_touching_fingers == 5:
                reward_sensor += 10.0 # 五指全包裹 (Power Grasp)，终极大满贯！

        # ================= 3. 组合总奖励 =================
        # 必须把动作惩罚加进去，否则网络只会疯狂输出极大值
        
        total_reward = reward_sensor
                        
        return total_reward

# ==========================================
# 本地测试代码 (仅在此文件直接运行时执行)
# ==========================================
if __name__ == "__main__":
    print("初始化环境...")
    XML_PATH = os.path.join(os.path.dirname(__file__), 'scene.xml')
    env = LinkerHandEnv(xml_path=XML_PATH) # 确保 XML 名字和你的匹配
    
    obs, info = env.reset()
    print(f"状态空间维度: {env.observation_space.shape}")
    print(f"动作空间维度: {env.action_space.shape}")
    
    print("\n开始随机动作测试...")
    for i in range(10):
        # 随机采样一个 [-1, 1] 的 21维 动作
        random_action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        print(f"Step {i+1} | Reward: {reward:.4f} | 传感器总和: {np.sum(obs[-len(env.data.sensordata):]):.4f}")
        
    print("环境测试成功！可以无缝接入 CleanRL 了。")