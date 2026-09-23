import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.game import Game2048Core


class Game2048Env(gym.Env):
    def __init__(self):
        super().__init__()
        self.none_cnt = 0
        self.game = Game2048Core(n=4)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=16, shape=(4, 4), dtype=np.float32)

        # --- 优化1: 预定义 Corner Building 权重矩阵 ---
        # 参考 qpwoeirut 的 Corner Building 策略。
        # 这是一个"蛇形"梯度矩阵，引导大数去左上角 (0,0)，并保持通过相邻格子的连贯性。
        # 相比你之前的路径判断，矩阵点积运算更快且梯度更平滑。
        self.corner_weights = np.array([
            [16, 15, 14, 13],
            [9, 10, 11, 12],
            [8, 7, 6, 5],
            [1, 2, 3, 4]
        ], dtype=np.float32)
        # 归一化权重，避免奖励过大
        self.corner_weights /= np.max(self.corner_weights)

    def _get_obs(self):
        board = np.array(self.game.get_board(), dtype=np.float32)
        board[board > 0] = np.log2(board[board > 0])
        return board

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.none_cnt = 0
        return self._get_obs(), {}

    def step(self, action):
        board_raw, reward_raw, done = self.game.step(int(action))

        # 获取 Log2 处理后的状态（用于计算 Observation 和 启发式奖励）
        # 注意：这里我们手动处理一次用于计算，避免多次调用 _get_obs
        board_log = np.zeros_like(board_raw, dtype=np.float32)
        mask = board_raw > 0
        board_log[mask] = np.log2(board_raw[mask])

        if done:
            return board_log, 0, done, False, {}

        reward = 0.0

        if reward_raw == -1:
            # 1. 撞墙/无效移动惩罚
            self.none_cnt += 1
            # 动态惩罚：连续无效移动惩罚加倍
            reward = -2.0 * self.none_cnt
            if self.none_cnt >= 10:
                return board_log, reward, True, False, {}
        else:
            self.none_cnt = 0

            # 2. 基础合并奖励 (Merge Score)
            # 保持原有的逻辑，这是最基础的目标
            reward += float(reward_raw) * 1.0

            # 3. 优化后的 Corner Building 奖励
            # 替代原有的 "蛇形路径" 循环判断。
            # 直接计算当前盘面与权重矩阵的点积。
            # 这鼓励大数占据高权重位置（左上角），并按权重梯度排列。
            heuristic_score = np.sum(board_log * self.corner_weights)

            # 系数 0.1 需要根据你的训练稳定性调整。
            # 如果 Agent 过于关注摆阵而不合并，可以调低此系数。
            reward += heuristic_score * 0.2

            # 4. 单调性奖励 (Monotonicity) - 简化版
            # qpwoeirut 强调单调性的重要性。
            # 我们奖励每一行/列相邻元素差值较小或有序的情况。
            # 这里简单实现：计算行和列的"逆序度"作为惩罚。
            # (可选：为了训练速度，上面的 corner_weights 其实已经隐含了单调性引导，
            #  如果计算资源有限，可以省略下面这一段)
            penalty = 0
            # 行单调性惩罚 (左边应该 >= 右边)
            diff_row = board_log[:, :-1] - board_log[:, 1:]
            penalty += np.sum(diff_row < 0) * 0.5  # 每一个逆序对罚 0.5

            # 列单调性惩罚 (上边应该 >= 下边)
            diff_col = board_log[:-1, :] - board_log[1:, :]
            penalty += np.sum(diff_col < 0) * 0.5

            reward -= penalty * 0.06

            # 5. 空格奖励 (Empty Tile)
            # 保持棋盘流动性的辅助奖励
            empty_count = np.sum(board_raw == 0)
            reward += empty_count * 0.6

        return board_log, reward, done, False, {}
