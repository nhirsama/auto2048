package main

import "C"
import (
	"fmt"
	"unsafe"
)

var globalGame *Game

//export InitGame
func InitGame(n C.int) {
	globalGame = NewGame(int(n))
}

//export ResetGame
func ResetGame() {
	globalGame.Reset()
}

//export StepGame
func StepGame(action C.int) (C.int, C.int) {
	reward, done := globalGame.Step(int(action))
	d := 0
	if done {
		d = 1
	}
	return C.int(reward), C.int(d)
}

//export GetBoard
func GetBoard(ptr *C.int) {
	// 将 Go 的棋盘数据拷贝到 C 传过来的指针内存中
	board := globalGame.Board
	for i, v := range board {
		// 使用 unsafe 指针操作
		*(*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + uintptr(i)*4)) = C.int(v)
	}
}

func main() {
	game := NewGame(4)
	for {
		var mv int
		fmt.Scanf("%d", &mv)
		reward, done := game.Step(mv) // 尝试向左滑
		if done {
			fmt.Println("done")
			break
		} else {
			fmt.Println("reward", reward)
			for i := 0; i < 4; i++ {
				for j := 0; j < 4; j++ {
					fmt.Printf("%d", game.Board[i*4+j])
				}
				fmt.Println()
			}
		}
	}
}

type Game struct {
	N          int   // 棋盘边长
	Board      []int // 一维数组表示的棋盘，大小为 N*N
	TotalScore int   // 累计总分
}

// NewGame 创建并初始化一个 N*N 的游戏
func NewGame(n int) *Game {
	g := &Game{
		N:     n,
		Board: make([]int, n*n),
	}
	g.Reset()
	return g
}

// Reset 重置游戏状态并生成第一个方块
func (g *Game) Reset() {
	for i := range g.Board {
		g.Board[i] = 0
	}
	g.TotalScore = 0
	g.SpawnFixed()
}

// SpawnFixed 核心逻辑：按固定顺序（从左到右，从上到下）在第一个空格生成 2
func (g *Game) SpawnFixed() bool {
	for i := 0; i < len(g.Board); i++ {
		if g.Board[i] == 0 {
			g.Board[i] = 2
			return true
		}
	}
	return false
}

// getTile 坐标转换：根据动作方向，将 (i, j) 映射到一维 Board 的索引
// i 是行索引，j 是列内部偏移（0到N-1）
func (g *Game) getTile(action, i, j int) int {
	switch action {
	case 0:
		return g.Board[j*g.N+i] // Up: i为列，j为行
	case 1:
		return g.Board[(g.N-1-j)*g.N+i] // Down
	case 2:
		return g.Board[i*g.N+j] // Left: i为行，j为列
	case 3:
		return g.Board[i*g.N+(g.N-1-j)] // Right
	default:
		return 0
	}
}

// setTile 坐标转换：将合并后的值写回一维数组
func (g *Game) setTile(action, i, j, val int) {
	switch action {
	case 0:
		g.Board[j*g.N+i] = val
	case 1:
		g.Board[(g.N-1-j)*g.N+i] = val
	case 2:
		g.Board[i*g.N+j] = val
	case 3:
		g.Board[i*g.N+(g.N-1-j)] = val
	}
}

// mergeLine 核心合并算法：处理一行/一列的压缩与合并
func (g *Game) mergeLine(line []int) ([]int, int) {
	n := len(line)
	next := make([]int, n)
	score := 0

	// 1. 挤压：去掉所有 0
	p := 0
	for _, v := range line {
		if v != 0 {
			next[p] = v
			p++
		}
	}

	// 2. 合并：相邻相等则翻倍
	for i := 0; i < n-1; i++ {
		if next[i] != 0 && next[i] == next[i+1] {
			next[i] *= 2
			score += next[i]
			// 后面元素前移
			for j := i + 1; j < n-1; j++ {
				next[j] = next[j+1]
			}
			next[n-1] = 0
		}
	}
	return next, score
}

// Step 执行一步动作 (0:Up, 1:Down, 2:Left, 3:Right)
func (g *Game) Step(action int) (reward int, done bool) {
	changed := false
	moveScore := 0

	for i := 0; i < g.N; i++ {
		// 提取当前方向的“线”
		line := make([]int, g.N)
		for j := 0; j < g.N; j++ {
			line[j] = g.getTile(action, i, j)
		}

		// 合并
		merged, score := g.mergeLine(line)
		moveScore += score

		// 写回并检查是否有变化
		for j := 0; j < g.N; j++ {
			if g.getTile(action, i, j) != merged[j] {
				g.setTile(action, i, j, merged[j])
				changed = true
			}
		}
	}

	// 如果没有方块移动或合并，判定为无效动作
	if !changed {
		return -1, g.IsGameOver()
	}

	g.TotalScore += moveScore
	g.SpawnFixed() // 动作有效才生成新方块
	return moveScore, g.IsGameOver()
}

// IsGameOver 检查是否无法再移动
func (g *Game) IsGameOver() bool {
	// 检查是否有空位
	for _, v := range g.Board {
		if v == 0 {
			return false
		}
	}
	// 检查相邻是否可合并
	for i := 0; i < g.N; i++ {
		for j := 0; j < g.N-1; j++ {
			// 水平相邻
			if g.Board[i*g.N+j] == g.Board[i*g.N+j+1] {
				return false
			}
			// 垂直相邻
			if g.Board[j*g.N+i] == g.Board[(j+1)*g.N+i] {
				return false
			}
		}
	}
	return true
}
