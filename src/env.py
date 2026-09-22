import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.game import Game2048Core


class Game2048Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = Game2048Core(n=4)
        # 动作空间：0,1,2,3 (上下左右)
        self.action_space = spaces.Discrete(4)
        # 状态空间：4x4 矩阵，值通常不会超过 2^16
        self.observation_space = spaces.Box(low=0, high=16, shape=(4, 4), dtype=np.float32)

    def _get_obs(self):
        board = np.array(self.game.get_board(), dtype=np.float32)
        # 对非零元素取 log2，让特征处于较小区间
        board[board > 0] = np.log2(board[board > 0])
        return board

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action):
        # 修正：game.step 只返回两个值：原始奖励和是否结束
        board_raw, reward_raw, done = self.game.step(int(action))

        # --- 重新设计的奖励逻辑 ---
        reward = 0.0

        if reward_raw == -1:
            # 撞墙罚分：必须确保罚分绝对大于“空格奖励”带来的收益
            # 否则 AI 依然会选择撞墙刷分
            reward = -999.0
        else:
            # 有效移动：给合并奖励 + 适量的空格奖励
            reward = float(reward_raw)*2
            empty_count = np.sum(board_raw == 0)
            reward += empty_count * 0.5  # 降低权重，防止刷分

        return self._get_obs(), reward, done, False, {}
