import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 搬运原来的网络结构 (只需要 Actor)
# ==========================================
# TD3 和 DDPG 的 Actor 结构完全相同，且本身就是确定性输出，不需要去噪音
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # 动作缩放 (Action Rescaling)
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

# ==========================================
# 2. 核心参数与环境配置
# ==========================================
if __name__ == "__main__":
    env_id = "Hopper-v4"
    
    # 【注意】在这里填入你刚刚跑出的 TD3 或 DDPG 的文件夹名字
    # run_name = "Hopper-v4__ddpg_continuous_action__1__1773825178" 
    run_name = "Hopper-v4__td3_continuous_action__1__1773827299"
    
    model_path = f"runs/{run_name}/best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🚀 准备加载巅峰模型: {model_path}")

    # 使用 human 模式实时弹出 3D 渲染窗口
    env = gym.make(env_id, render_mode="human")
    
    # 离线强化学习 (Off-Policy) 不需要 NormalizeObservation！环境极其纯净！
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # 伪装 single_space 用于初始化网络维度
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space

    actor = Actor(env).to(device)

    # ==========================================
    # 3. 加载权重
    # ==========================================
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint[0]  # 提取 Actor 的字典
        
        # 【核心神级补丁】：动态修正 CleanRL 源码里 DDPG 和 TD3 的历史遗留维度差异！
        # 如果存档里的 action_scale 维度 (如 [1, 3]) 和当前网络 (如 [3]) 匹配不上
        if state_dict["action_scale"].shape != actor.action_scale.shape:
            # 直接用 view_as 强行把存档里的 tensor 捏成当前网络需要的形状
            state_dict["action_scale"] = state_dict["action_scale"].view_as(actor.action_scale)
            state_dict["action_bias"] = state_dict["action_bias"].view_as(actor.action_bias)
            print("🔧 检测到 DDPG/TD3 维度差异，已自动触发维度压缩修复！")
            
        actor.load_state_dict(state_dict)
        print("✅ 神经网络脑部权重加载完毕！环境无需归一化，直接起飞！")
    else:
        raise FileNotFoundError(f"找不到模型文件，请检查路径: {model_path}")

    # 锁死网络推理模式
    actor.eval()

    # ==========================================
    # 4. 视觉盛宴：开始推理
    # ==========================================
    print("\n🎬 演出开始！请观看弹出的 MuJoCo 窗口...")
    test_episodes = 3  # 看 3 局
    
    for ep in range(test_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0
        ep_length = 0
        
        while not done:
            # numpy 转 Tensor
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 直接调用 Actor，它天然输出的就是最优均值，不带噪声！
                action = actor(obs_tensor)
            
            # 执行动作并渲染
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            
            ep_return += reward
            ep_length += 1
            
        print(f"🏆 第 {ep + 1} 局结束: 存活步数 = {ep_length}, 最终得分 = {ep_return:.2f}")
        
    env.close()
    print("========== 推理结束 ==========\n")