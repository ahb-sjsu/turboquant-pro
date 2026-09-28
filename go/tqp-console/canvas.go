// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

package main

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// A Canvas is a W x H grid of cells, each a rune and a colour role; the roles are
// the engine's (and the Python panels'): "cyan", "dim", "yellow_bold", "sel", ...
type Cell struct {
	R    rune
	Role string
}

type Canvas struct {
	W, H int
	C    []Cell
}

func NewCanvas(w, h int) *Canvas {
	if w < 0 {
		w = 0
	}
	if h < 0 {
		h = 0
	}
	c := &Canvas{W: w, H: h, C: make([]Cell, w*h)}
	for i := range c.C {
		c.C[i] = Cell{R: ' '}
	}
	return c
}

// Put writes s at (y, x) in role, clipped to the canvas.
func (c *Canvas) Put(y, x int, s string, role string) {
	if y < 0 || y >= c.H {
		return
	}
	for _, r := range s {
		if x >= c.W {
			return
		}
		if x >= 0 {
			c.C[y*c.W+x] = Cell{R: r, Role: role}
		}
		x++
	}
}

func (c *Canvas) Get(y, x int) Cell {
	if y < 0 || y >= c.H || x < 0 || x >= c.W {
		return Cell{R: ' '}
	}
	return c.C[y*c.W+x]
}

// PutSpans writes a row of spans at (y, x) and returns the column after them.
func (c *Canvas) PutSpans(y, x int, spans []Span, maxw int) int {
	end := x + maxw
	for _, s := range spans {
		if x >= end {
			break
		}
		c.Put(y, x, clip(s.Text, end-x), s.Role)
		x += runeLen(s.Text)
	}
	return x
}

// Blit copies src onto c with its top-left at (y, x).
func (c *Canvas) Blit(src *Canvas, y, x int) {
	for r := 0; r < src.H; r++ {
		for k := 0; k < src.W; k++ {
			yy, xx := y+r, x+k
			if yy >= 0 && yy < c.H && xx >= 0 && xx < c.W {
				c.C[yy*c.W+xx] = src.C[r*src.W+k]
			}
		}
	}
}

// Box draws a frame with a title on its top edge; focus draws it in cyan.
func (c *Canvas) Box(y, x, h, w int, title, role string, focus bool) {
	if h < 2 || w < 4 {
		return
	}
	if focus {
		role = "cyan"
	}
	c.Put(y, x, "┌"+strings.Repeat("─", w-2)+"┐", role)
	for r := 1; r < h-1; r++ {
		c.Put(y+r, x, "│", role)
		c.Put(y+r, x+w-1, "│", role)
	}
	c.Put(y+h-1, x, "└"+strings.Repeat("─", w-2)+"┘", role)
	if title != "" {
		tr := "bold"
		if focus {
			tr = "cyan"
		}
		c.Put(y, x+2, clip(" "+title+" ", w-4), tr)
	}
}

// Lines is the canvas as plain text, one string per row (the P snapshot).
func (c *Canvas) Lines() []string {
	out := make([]string, c.H)
	var b strings.Builder
	for y := 0; y < c.H; y++ {
		b.Reset()
		for x := 0; x < c.W; x++ {
			b.WriteRune(c.C[y*c.W+x].R)
		}
		out[y] = strings.TrimRight(b.String(), " ")
	}
	return out
}

func runeLen(s string) int { return utf8.RuneCountInString(s) }

// clip keeps at most n runes of s.
func clip(s string, n int) string {
	if n <= 0 {
		return ""
	}
	if runeLen(s) <= n {
		return s
	}
	i := 0
	for k := range s {
		if i == n {
			return s[:k]
		}
		i++
	}
	return s
}

func rjust(s string, n int) string {
	if l := runeLen(s); l < n {
		return strings.Repeat(" ", n-l) + s
	}
	return s
}

func ljust(s string, n int) string {
	if l := runeLen(s); l < n {
		return s + strings.Repeat(" ", n-l)
	}
	return s
}

// ---------------------------------------------------------------- output

var baseSGR = map[string]string{
	"":        "0",
	"cyan":    "0;36",
	"teal":    "0;36",
	"purple":  "0;35",
	"magenta": "0;35",
	"green":   "0;32",
	"amber":   "0;33",
	"yellow":  "0;33",
	"red":     "0;31",
	"dim":     "0;90",
	"bold":    "0;1;97",
	"grid":    "0;2;34",
	"sel":     "0;30;46",
}

var sgrCache = map[string]string{}

// SGR is the escape sequence for a role, including the _dim / _bold variants.
func SGR(role string) string {
	if s, ok := sgrCache[role]; ok {
		return s
	}
	s, ok := baseSGR[role]
	if !ok {
		s = "0"
		if i := strings.LastIndex(role, "_"); i > 0 {
			if b, ok2 := baseSGR[role[:i]]; ok2 {
				switch role[i+1:] {
				case "dim":
					s = b + ";2"
				case "bold":
					s = b + ";1"
				default:
					s = b
				}
			}
		}
	}
	s = "\x1b[" + s + "m"
	sgrCache[role] = s
	return s
}

// Renderer sends only the rows that changed since the last frame.
type Renderer struct {
	prev *Canvas
}

func (r *Renderer) Invalidate() { r.prev = nil }

func (r *Renderer) Frame(c *Canvas) []byte {
	var b strings.Builder
	full := r.prev == nil || r.prev.W != c.W || r.prev.H != c.H
	if full {
		b.WriteString("\x1b[0m\x1b[H\x1b[2J")
	}
	for y := 0; y < c.H; y++ {
		row := c.C[y*c.W : (y+1)*c.W]
		if !full {
			same := true
			prow := r.prev.C[y*c.W : (y+1)*c.W]
			for i := range row {
				if row[i] != prow[i] {
					same = false
					break
				}
			}
			if same {
				continue
			}
		}
		fmt.Fprintf(&b, "\x1b[%d;1H", y+1)
		role := "\x00"
		for _, cell := range row {
			if cell.Role != role {
				b.WriteString(SGR(cell.Role))
				role = cell.Role
			}
			b.WriteRune(cell.R)
		}
	}
	b.WriteString("\x1b[0m")
	cp := &Canvas{W: c.W, H: c.H, C: append([]Cell(nil), c.C...)}
	r.prev = cp
	return []byte(b.String())
}
