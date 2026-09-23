package main

import "C"
import (
	"sync"
	"unsafe"
)

var (
	rowMoveTable  [65536]uint16
	rowScoreTable [65536]uint32
	initOnce      sync.Once
)

type Game struct {
	Board uint64 // 4x4 棋盘，每个格子 4-bit (存储 log2 值)
}

var globalGame *Game

// ==========================================
// 1. 查找表初始化 (修复了合并时的分数计算逻辑)
// ==========================================
func initLUT() {
	for row := 0; row < 65536; row++ {
		// 解析行：row 的位布局为 [T3][T2][T1][T0] (从低位到高位)
		line := [4]int{
			(row >> 0) & 0xF,
			(row >> 4) & 0xF,
			(row >> 8) & 0xF,
			(row >> 12) & 0xF,
		}

		score := 0
		next := [4]int{0, 0, 0, 0}
		p := 0
		// 压缩空格
		for _, v := range line {
			if v != 0 {
				next[p] = v
				p++
			}
		}

		// 合并
		for i := 0; i < 3; i++ {
			if next[i] != 0 && next[i] == next[i+1] {
				next[i]++             // 数值翻倍 (log2+1)
				score += 1 << next[i] // 增加合并后的真实分数
				for j := i + 1; j < 3; j++ {
					next[j] = next[j+1]
				}
				next[3] = 0
			}
		}

		var resRow uint16
		for i := 0; i < 4; i++ {
			resRow |= uint16(next[i]) << (i * 4)
		}
		rowMoveTable[row] = resRow
		rowScoreTable[row] = uint32(score)
	}
}

// ==========================================
// 2. 导出 API (保持接口不变)
// ==========================================

//export InitGame
func InitGame(n C.int) {
	initOnce.Do(initLUT)
	globalGame = &Game{}
	globalGame.Reset()
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
	b := globalGame.Board
	// 解压逻辑：确保 4-bit 块正确映射到 4x4 数组
	for i := 0; i < 16; i++ {
		val := (b >> (i * 4)) & 0xF
		realVal := 0
		if val > 0 {
			realVal = 1 << val
		}
		// 安全地写入 C 数组
		target := (*C.int)(unsafe.Pointer(uintptr(unsafe.Pointer(ptr)) + uintptr(i)*unsafe.Sizeof(C.int(0))))
		*target = C.int(realVal)
	}
}

// ==========================================
// 3. 高性能位运算逻辑 (核心 Bug 修复)
// ==========================================

// 修复后的转置函数：针对 4x4 的 4-bit 块矩阵
func transpose(x uint64) uint64 {
	a1 := x & 0xF0F00F0FF0F00F0F
	a2 := x & 0x0000F0F00000F0F0
	a3 := x & 0x0F0F00000F0F0000
	x = a1 | (a2 << 12) | (a3 >> 12)
	a1 = x & 0xFF00FF0000FF00FF
	a2 = x & 0x00000000FF00FF00
	a3 = x & 0x00FF00FF00000000
	x = a1 | (a2 << 24) | (a3 >> 24)
	return x
}

// 快速水平翻转：将 [T3][T2][T1][T0] 变为 [T0][T1][T2][T3]
func reverseRow(row uint16) uint16 {
	return (row >> 12) | ((row >> 4) & 0x00F0) | ((row << 4) & 0x0F00) | (row << 12)
}

func (g *Game) Step(action int) (int, bool) {
	oldBoard := g.Board
	var totalScore uint32
	tmpBoard := g.Board

	// 统一转换逻辑
	if action == 0 || action == 1 { // Up, Down
		tmpBoard = transpose(tmpBoard)
	}

	var resBoard uint64
	for i := 0; i < 4; i++ {
		row := uint16(tmpBoard >> (i * 16))
		// Down (1) 和 Right (3) 需要水平翻转后再查表
		if action == 1 || action == 3 {
			row = reverseRow(row)
		}

		resRow := rowMoveTable[row]
		totalScore += rowScoreTable[row]

		if action == 1 || action == 3 {
			resRow = reverseRow(resRow)
		}
		resBoard |= uint64(resRow) << (i * 16)
	}

	if action == 0 || action == 1 {
		resBoard = transpose(resBoard)
	}

	if resBoard == oldBoard {
		return -1, g.IsGameOver()
	}

	g.Board = resBoard
	g.SpawnFixed()
	return int(totalScore), g.IsGameOver()
}

func (g *Game) SpawnFixed() {
	// 找到第一个空位生成一个 2 (log2 值为 1)
	for i := 0; i < 16; i++ {
		if (g.Board >> (i * 4) & 0xF) == 0 {
			g.Board |= uint64(1) << (i * 4)
			break
		}
	}
}

func (g *Game) IsGameOver() bool {
	// 1. 检查是否有空格
	for i := 0; i < 16; i++ {
		if (g.Board >> (i * 4) & 0xF) == 0 {
			return false
		}
	}
	// 2. 检查水平和垂直方向是否还能合并
	for i := 0; i < 4; i++ {
		row := (g.Board >> (i * 16)) & 0xFFFF
		for j := 0; j < 3; j++ {
			if (row>>(j*4))&0xF == (row>>((j+1)*4))&0xF {
				return false
			}
		}
	}
	// 利用转置检查垂直方向
	tBoard := transpose(g.Board)
	for i := 0; i < 4; i++ {
		row := (tBoard >> (i * 16)) & 0xFFFF
		for j := 0; j < 3; j++ {
			if (row>>(j*4))&0xF == (row>>((j+1)*4))&0xF {
				return false
			}
		}
	}
	return true
}

func (g *Game) Reset() {
	g.Board = 0
	g.SpawnFixed()
}

func main() {}
