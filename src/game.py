import ctypes
import os
import numpy as np


class StepGameReturn(ctypes.Structure):
    _fields_ = [("r0", ctypes.c_int), ("r1", ctypes.c_int)]


class Game2048Core:
    def __init__(self, n=4, lib_path='./lib/lib2048.so'):
        self.n = n
        full_path = os.path.abspath(lib_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"未找到动态库: {full_path}")

        self.lib = ctypes.CDLL(full_path)

        # 签名配置
        self.lib.InitGame.argtypes = [ctypes.c_int]
        self.lib.StepGame.argtypes = [ctypes.c_int]
        self.lib.StepGame.restype = StepGameReturn
        self.lib.GetBoard.argtypes = [ctypes.POINTER(ctypes.c_int)]

        self.lib.InitGame(n)
        # 预分配好 buffer，避免重复创建
        self.board_buffer = (ctypes.c_int * (n * n))()

    def reset(self):
        """重置游戏"""
        self.lib.ResetGame()
        return self.get_board()  # 重置后顺便返回棋盘

    def step(self, action):
        """
        核心修改：执行动作后立即获取新棋盘并返回
        返回: (新棋盘, 奖励, 是否结束)
        """
        res = self.lib.StepGame(action)
        reward = res.r0
        done = bool(res.r1)

        # 动作执行完，立即拉取最新的棋盘数据
        new_board = self.get_board()

        return new_board, reward, done

    def get_board(self):
        """拉取当前棋盘数据并转为 numpy 矩阵"""
        self.lib.GetBoard(self.board_buffer)
        # 使用 copy() 确保数据安全，或者直接返回新数组
        return np.array(self.board_buffer).reshape((self.n, self.n)).copy()