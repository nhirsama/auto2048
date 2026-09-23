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
        # 状态空间：4x4 矩阵，值通过 log2 处理
        self.observation_space = spaces.Box(low=0, high=16, shape=(4, 4), dtype=np.float32)

    def _get_obs(self):
        board = np.array(self.game.get_board(), dtype=np.float32)
        # 对非零元素取 log2
        board[board > 0] = np.log2(board[board > 0])
        return board

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}

    def step(self, action):
        # 保持与 game 接口一致
        board_raw, reward_raw, done = self.game.step(int(action))

        reward = 0.0

        if reward_raw == -1:
            # 1. 撞墙重罚：防止 AI 陷入无效循环
            reward = -10.0
        else:
            # 2. 基础合并奖励：保持原有的得分逻辑
            reward = float(reward_raw) * 2.0

            # 3. 空格奖励：保持棋盘开阔（权重略微调低，防止过度刷分）
            empty_count = np.sum(board_raw == 0)
            reward += empty_count * 0.8

            # --- 4. 蛇形布局与单调性奖励 ---
            # 我们定义一条从左上到右下的蛇形路径，目标是让数字沿路径递增
            # 路径索引: (0,0) -> (0,1) -> (0,2) -> (0,3) -> (1,3) -> (1,2) ...
            snake_path = [
                (0, 0), (0, 1), (0, 2), (0, 3),
                (1, 3), (1, 2), (1, 1), (1, 0),
                (2, 0), (2, 1), (2, 2), (2, 3),
                (3, 3), (3, 2), (3, 1), (3, 0)
            ]

            # 获取当前棋盘的数值（用于计算单调性）
            # 注意：这里直接用原始值或 log2 值均可，log2 值更平滑
            log_board = self._get_obs()

            mono_reward = 0
            for i in range(len(snake_path) - 1):
                prev_val = log_board[snake_path[i]]
                next_val = log_board[snake_path[i + 1]]

                # 如果后一个格子比前一个大（符合向末端递增的蛇形趋势）
                if next_val >= prev_val and next_val > 0:
                    # 奖励与数值大小成正比
                    mono_reward += next_val * 0.5
                elif next_val < prev_val:
                    # 违背蛇形排列则给予小惩罚
                    mono_reward -= prev_val * 0.2

            reward += mono_reward

            # 5. 角落大数奖励：如果最大值在蛇形路径的终点 (3,0) 或起点 (0,0)
            max_tile_log = np.max(log_board)
            if log_board[3, 0] == max_tile_log or log_board[0, 0] == max_tile_log:
                reward += max_tile_log * 2.0

        return self._get_obs(), reward, done, False, {}