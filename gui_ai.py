import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, QTimer
from stable_baselines3 import PPO

# 确保导入路径正确
from src.game import Game2048Core


class AI2048Gui(QWidget):
    def __init__(self, model_path="ppo_2048_master.zip"):
        super().__init__()
        self.setWindowTitle("2048 AI 自动游玩展示")
        self.setFixedSize(400, 500)

        # 1. 游戏状态变量
        self.total_score = 0  # 核心：新增总分累加器
        self.game = Game2048Core(n=4)

        # 2. 加载模型
        try:
            self.model = PPO.load(model_path, device="cpu")
            print(f">>> 成功加载模型: {model_path}")
        except Exception as e:
            print(f">>> 错误: 无法加载模型 {model_path}。原因: {e}")
            sys.exit(1)

        # 3. 界面配色
        self.colors = {
            0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
            16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
            256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e"
        }

        # 4. 初始化 UI
        self.cells = []
        self.setup_ui()
        self.update_board()  # 初始刷新一次

        # 5. 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.ai_step)
        self.timer.start(50)  # 50ms 一步，速度非常快

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # 分数显示标签
        self.score_label = QLabel("Score: 0")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setStyleSheet("""
            font-size: 26px; 
            font-weight: bold; 
            color: #776e65;
            background-color: #bbada0;
            border-radius: 10px;
            margin: 10px;
            padding: 5px;
        """)
        main_layout.addWidget(self.score_label)

        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        for i in range(4):
            row = []
            for j in range(4):
                label = QLabel("")
                label.setFixedSize(80, 80)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet(self.get_style(0))
                grid_layout.addWidget(label, i, j)
                row.append(label)
            self.cells.append(row)
        main_layout.addLayout(grid_layout)

    def get_style(self, val):
        color = self.colors.get(val, "#3c3a32")
        text_color = "#776e65" if val <= 4 else "#f9f6f2"
        size = 24 if val < 1000 else 20
        return f"background-color: {color}; color: {text_color}; font-size: {size}px; font-weight: bold; border-radius: 5px;"

    def get_ai_obs(self):
        """获取棋盘并进行特征预处理 (log2)"""
        # 修正：显式转为 numpy 数组以支持 astype 和数值运算
        board = np.array(self.game.get_board(), dtype=np.float32)
        board[board > 0] = np.log2(board[board > 0])
        return board

    def ai_step(self):
        obs = self.get_ai_obs()

        # AI 决策
        action, _ = self.model.predict(obs, deterministic=True)
        print(action)
        # 执行动作
        # 注意：这里的 reward_raw 应该是你在 Go 后端定义的 mergeScore
        board_raw, reward_raw, done = self.game.step(int(action))

        # 核心：更新总分
        # 只有在 reward_raw > 0（即发生了合并）时才加分，排除撞墙的 -1 情况
        if reward_raw > 0:
            self.total_score += reward_raw
            self.score_label.setText(f"Score: {self.total_score}")

        # 更新界面展示
        self.update_board()

        if done:
            self.timer.stop()
            self.score_label.setText(f"GAME OVER! Final Score: {self.total_score}")
            self.score_label.setStyleSheet(self.score_label.styleSheet() + "color: #f65e3b;")

    def update_board(self):
        # 获取最新棋盘状态
        board = np.array(self.game.get_board())
        for i in range(4):
            for j in range(4):
                val = board[i, j]
                self.cells[i][j].setText(str(val) if val > 0 else "")
                self.cells[i][j].setStyleSheet(self.get_style(val))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AI2048Gui()
    window.show()
    sys.exit(app.exec())