import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class LinkerHandEnv(gym.Env):
    """
    Linker L20 灵巧手残差强化学习环境 (基于 Gymnasium)
    """
    def __init__(self, xml_path="linker_l20.xml", max_steps=500):
        super().__init__()
        
        # 1. 加载 MuJoCo 模型与数据
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # 2. 定义动作空间 (Action Space)
        # 共有 21 个位置驱动器。RL 算法喜欢对称且归一化的动作 [-1, 1]
        self.num_actions = self.model.nu 
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        
        # 获取底层驱动器的真实物理限位 (ctrlrange)，用于后续的动作反归一化
        self.ctrl_ranges = self.model.actuator_ctrlrange
        
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
        
        # 将它们拼接成一个一维 numpy 数组送给神经网络
        obs = np.concatenate([qpos, qvel, sensor_data])
        return obs.astype(np.float32)

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
        
        obs = self._get_obs()
        info = {} # 可以用来传递额外的调试信息
        
        return obs, info

    def step(self, action):
        """
        与环境交互一步
        action: 来自神经网络的动作指令，范围 [-1, 1]
        """
        self.current_step += 1
        
        # 1. 动作映射 (Action Mapping): 将 [-1, 1] 映射回真实的 ctrlrange
        # 公式: u_real = min + (a + 1) * 0.5 * (max - min)
        min_ctrl = self.ctrl_ranges[:, 0]
        max_ctrl = self.ctrl_ranges[:, 1]
        scaled_action = min_ctrl + (action + 1.0) * 0.5 * (max_ctrl - min_ctrl)
        
        # 将计算好的真实目标角度发给 MuJoCo 控制器
        self.data.ctrl[:] = scaled_action
        
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
        reward = 0.0
        
        # 示例 1: 动作惩罚 (Action Penalty) - 鼓励残差动作尽可能小，节省能耗且安全
        action_penalty = -0.01 * np.sum(np.square(action))
        reward += action_penalty
        
        # 示例 2: 触觉奖励 (Touch Reward) - 如果传感器检测到力，给予巨大奖励
        # 假设 sensordata 里的第一个值是食指的力传感器
        if len(self.data.sensordata) > 0:
            index_force = self.data.sensordata[0]
            if index_force > 0.01:
                reward += 1.0 # 摸到东西了！
                
        return reward

# ==========================================
# 本地测试代码 (仅在此文件直接运行时执行)
# ==========================================
if __name__ == "__main__":
    print("初始化环境...")
    env = LinkerHandEnv(xml_path="linker_l20.xml") # 确保 XML 名字和你的匹配
    
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