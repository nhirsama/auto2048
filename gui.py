import sys
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel
from PySide6.QtCore import Qt
from src.game import Game2048Core

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Python-Go 2048 (Qt)")
        self.game = Game2048Core(n=4)
        self.cells = []
        self.setup_ui()
        self.update_ui()

    def setup_ui(self):
        self.layout = QGridLayout(self)
        for i in range(4):
            row = []
            for j in range(4):
                label = QLabel("")
                label.setFixedSize(80, 80)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet("background-color: #cdc1b4; font-size: 24px; font-weight: bold; border-radius: 5px;")
                self.layout.addWidget(label, i, j)
                row.append(label)
            self.cells.append(row)

    def keyPressEvent(self, event):
        mapping = {Qt.Key_Up: 0, Qt.Key_Down: 1, Qt.Key_Left: 2, Qt.Key_Right: 3}
        if event.key() in mapping:
            board,ok,  done = self.game.step(mapping[event.key()])
            self.update_ui()

    def update_ui(self):
        board = self.game.get_board()
        for i in range(4):
            for j in range(4):
                val = board[i][j]
                self.cells[i][j].setText(str(val) if val != 0 else "")
                # 这里可以根据 val 设置不同的颜色

if __name__ == "__main__":
    # 安装依赖: uv add pyside6
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())