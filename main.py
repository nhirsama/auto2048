import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
# 导入自定义模块
from src.env import Game2048Env


def make_env(rank, seed=0):
    def _init():
        env = Game2048Env()
        env = Monitor(env)  # 必须用 Monitor 包装环境
        return env

    set_random_seed(seed)
    return _init


def train():
    # 配置参数
    MODEL_PATH = "ppo_2048_master.zip"
    LOG_DIR = "./logs/"
    CHECKPOINT_DIR = "./checkpoints/"

    # 获取 CPU 核心数，实现全核并行训练
    num_cpu = multiprocessing.cpu_count()
    print(f">>> [并行加速] 检测到 {num_cpu} 个 CPU 核心，正在创建训练集群...")

    # 矢量化环境
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # 增量训练逻辑：检查是否可以从上次结果继续
    if os.path.exists(MODEL_PATH):
        print(f">>> [断点续传] 发现已存模型 {MODEL_PATH}，正在加载训练进度...")
        model = PPO.load(MODEL_PATH, env=env,
                         verbose=1,
                         tensorboard_log=LOG_DIR,
                         learning_rate=1e-4,
                         batch_size=1024,
                         n_steps=2048,  # 每个核心单次采集的步数
                         ent_coef=0.1,  # 从原来的 0.01 调高到 0.05，强制 AI 重新变乱，去探索
                         )
    else:
        print(">>> [全新训练] 未发现旧模型，正在初始化神经网络...")
        model = PPO(
            "MlpPolicy",
            env,
            # device="cpu",  # 针对你的 MX250 兼容性问题强制 CPU
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=1e-4,
            batch_size=128,
            n_steps=1024,  # 每个核心单次采集的步数
            ent_coef=0.05,  # 从原来的 0.01 调高到 0.05，强制 AI 重新变乱，去探索
        )

    # 自动保存回调
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50000 // num_cpu, 1),
        save_path=CHECKPOINT_DIR,
        name_prefix='rl_model'
    )

    print(">>> 训练已开始。你可以通过 'tensorboard --logdir ./logs/' 查看进度曲线。")
    try:
        # 建议总步数：1,000,000 起步
        model.learn(
            total_timesteps=10000000,
            callback=checkpoint_callback,
            reset_num_timesteps=False  # 保持 TensorBoard 曲线连续
        )
    except KeyboardInterrupt:
        print("\n>>> 检测到中断信号，正在安全保存模型...")
    finally:
        model.save(MODEL_PATH)
        print(f">>> 训练完成，模型保存在: {MODEL_PATH}")


if __name__ == "__main__":
    # 必要的并行保护
    multiprocessing.freeze_support()
    train()
