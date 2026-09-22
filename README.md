# auto2048

需要 Go（cgo）、C 编译器、Python 3.14+、[uv](https://docs.astral.sh/uv/)。

```bash
mkdir -p lib
go build -buildmode=c-shared -o lib/lib2048.so ./game2048/game.go
uv sync
```

训练：`uv run python main.py`  
手动玩：`uv run python gui.py`  
看 AI：先训练出 `ppo_2048_master.zip`，再 `uv run python gui_ai.py`
