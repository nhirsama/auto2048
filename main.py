import os
import sys

# 必须在 import torch 之前设置！
# 10.3.0 是 RDNA 2 架构(gfx1032/1030)的通用兼容版本
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import multiprocessing
import torch
from time import strftime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# 导入自定义模块
from src.env import Game2048Env

# ==========================================
# 1. 统一超参数配置中心
# ==========================================
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 2e-4,  # 稍微调低，保证稳定性
    "n_steps": 2048,
    "batch_size": 1024,
    "n_epochs": 10,
    "ent_coef": 0.02,  # 重要：从 0.1 降到 0.02，减少无效乱走
    "device": "cuda",
    "policy_kwargs": dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # 规模适中
    ),
    "tensorboard_log": "./logs/",
    "verbose": 1
}

MODEL_PATH = "ppo_2048_master.zip"
CHECKPOINT_DIR = "./checkpoints/"


# ==========================================
# 2. 环境创建逻辑
# ==========================================
def make_env(rank, seed=0):
    def _init():
        env = Game2048Env()
        env = Monitor(env)  # 必须用 Monitor 包装环境
        return env

    set_random_seed(seed)
    return _init


def train():
    timestamp = strftime("%Y%m%d_%H%M")
    RUN_NAME = f"PPO_{timestamp}_gpu_ent0.02"
    # 获取 CPU 核心数进行环境采样
    num_cpu = multiprocessing.cpu_count()
    print(f">>> [硬件加速] 检测到 GPU: {torch.cuda.get_device_name(0)}")
    print(f">>> [并行采样] 使用 {num_cpu} 个 CPU 核心同步采集数据...")

    # 创建矢量化环境
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # ==========================================
    # 3. 加载或创建模型 (统一参数管理)
    # ==========================================
    if os.path.exists(MODEL_PATH):
        print(f">>> [继续训练] 正在加载模型并应用统一参数: {MODEL_PATH}")
        # PPO.load 的 **kwargs 会覆盖掉 zip 文件中保存的旧参数
        model = PPO.load(
            MODEL_PATH,
            env=env,
            **{k: v for k, v in PPO_CONFIG.items() if k != "policy"}
        )
    else:
        print(">>> [新建模型] 正在使用统一参数初始化神经网络...")
        model = PPO(env=env, **PPO_CONFIG)

    # 自动保存回调
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // num_cpu, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix='rl_model'
    )

    print(">>> 训练开始！使用 TensorBoard 监控: tensorboard --logdir ./logs/")
    try:
        model.learn(
            total_timesteps=400000000,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            tb_log_name=RUN_NAME
        )
    except KeyboardInterrupt:
        print("\n>>> 收到中断信号，正在保存进度...")
    finally:
        model.save(MODEL_PATH)
        print(f">>> 训练成果已保存至: {MODEL_PATH}")


if __name__ == "__main__":
    # Arch Linux 并行保护
    multiprocessing.freeze_support()
    train()
