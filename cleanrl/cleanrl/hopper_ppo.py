import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# ==========================================
# 1. 搬运原来的网络结构 (必须和训练时一模一样)
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(env.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(env.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(env.single_action_space.shape)))

    def get_action_mean(self, x):
        # 推理专属：直接输出均值，彻底剥离标准差 (logstd) 带来的探索噪声
        return self.actor_mean(x)

# ==========================================
# 2. 核心参数与环境配置
# ==========================================
if __name__ == "__main__":
    env_id = "Hopper-v4"
    # 【注意】这里直接写死你刚跑出的那个完美的文件夹时间戳！
    run_name = "Hopper-v4__ppo_continuous_action__1__1773804269"
    model_path = f"runs/{run_name}/best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n🚀 准备加载巅峰模型: {model_path}")

    # 使用 human 模式，会直接弹出一个渲染窗口让你实时观看！
    env = gym.make(env_id, render_mode="human")
    # 为了保持单环境和多环境的 API 兼容性，套一个 Dummy 向量环境的单体壳
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # 【极其关键】：穿上和训练时一模一样的三件“归一化”外套
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))

    # 初始化伪装的 single_observation_space 用于实例化网络
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space

    agent = Agent(env).to(device)

    # ==========================================
    # 3. 加载权重与状态 (神装合体)
    # ==========================================
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        agent.load_state_dict(checkpoint["model_state_dict"])
        
        # 穿透 Wrapper，把眼睛（环境统计量）装回去
        if checkpoint.get("obs_rms") is not None:
            temp_env = env
            while hasattr(temp_env, "env"):
                # 【核心修复】：绝不能用 hasattr，必须用 isinstance 看衣服牌子！
                if isinstance(temp_env, gym.wrappers.NormalizeObservation):
                    # 只有真正找到了归一化这层，才把统计量塞进去
                    temp_env.obs_rms = checkpoint["obs_rms"]
                    print("✅ 成功精准定位 NormalizeObservation，环境归一化状态已完美注入！")
                    break
                temp_env = temp_env.env
        print("✅ 神经网络脑部权重加载完毕！")
    else:
        raise FileNotFoundError(f"找不到模型文件，请检查路径: {model_path}")

    # 锁死网络，关闭任何 Dropout 或 Batch Norm 的动态更新
    agent.eval()

    # ==========================================
    # 4. 视觉盛宴：开始推理
    # ==========================================
    print("\n🎬 演出开始！请观看弹出的 MuJoCo 窗口...")
    test_episodes = 3  # 看 3 局就够了
    
    for ep in range(test_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0
        ep_length = 0
        
        while not done:
            # 将 numpy 数组转成 Tensor 喂给网络
            obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 直接调用新写的 get_action_mean，拿到最纯粹的确定性动作
                action = agent.get_action_mean(obs_tensor)
            
            # 执行动作并渲染
            obs, reward, terminated, truncated, info = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            
            ep_return += reward
            ep_length += 1
            
        print(f"🏆 第 {ep + 1} 局结束: 存活步数 = {ep_length}, 最终得分 = {ep_return:.2f}")
        
    env.close()
    print("========== 推理结束 ==========\n")