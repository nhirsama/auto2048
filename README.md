# auto2048

用 PPO 训练玩 2048。游戏核是 Go（cgo）位棋盘实现，Python 侧用 Gymnasium + Stable-Baselines3。

需要 Go（cgo）、C 编译器、Python 3.13+、[uv](https://docs.astral.sh/uv/)。训练默认走 ROCm GPU（`HSA_OVERRIDE_GFX_VERSION=10.3.0`）。

```bash
mkdir -p lib
go build -buildmode=c-shared -o lib/lib2048.so ./game2048/game.go
uv sync
```

- 训练：`uv run python main.py`
- 手动玩：`uv run python gui.py`
- 看 AI：先有 `ppo_2048_master.zip`，再 `uv run python gui_ai.py`
