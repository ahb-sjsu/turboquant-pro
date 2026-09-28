// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

package main

// The grid's panels, drawn from the view model. Layout and wording follow the
// Python panels in turboquant_pro/console/tui.py, which remain the reference
// (and draw the web and vector views).

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// pyFmt is tui.fmt: whole numbers from 1000 up, else d decimals.
func pyFmt(v *float64, d int) string {
	if v == nil {
		return "-"
	}
	if math.Abs(*v) >= 1000 {
		return strconv.FormatFloat(*v, 'f', 0, 64)
	}
	return strconv.FormatFloat(*v, 'f', d, 64)
}

const sparkRamp = " ▁▂▃▄▅▆▇█"

// Spark is tui.spark: the last width values as blocks scaled to their maximum.
func Spark(values []*float64, width int) string {
	if width <= 0 {
		return ""
	}
	vals := values
	if len(vals) > width {
		vals = vals[len(vals)-width:]
	}
	var real []float64
	for _, v := range vals {
		if v != nil {
			real = append(real, *v)
		}
	}
	if len(real) < 2 {
		return clip(ljust("collecting", width), width)
	}
	top := real[0]
	for _, v := range real {
		top = math.Max(top, v)
	}
	if top == 0 {
		top = 1
	}
	ramp := []rune(sparkRamp)
	var b strings.Builder
	for _, v := range vals {
		if v == nil {
			b.WriteRune(' ')
			continue
		}
		i := int(math.RoundToEven(*v / top * 8))
		if i < 1 {
			i = 1
		}
		if i > len(ramp)-1 {
			i = len(ramp) - 1
		}
		b.WriteRune(ramp[i])
	}
	return rjust(b.String(), width)
}

func bar(frac float64, width int) string {
	n := int(math.RoundToEven(frac * float64(width)))
	if n < 0 {
		n = 0
	}
	if n > width {
		n = width
	}
	return strings.Repeat("█", n) + strings.Repeat("░", width-n)
}

func maxReal(values []*float64, last int) (float64, int) {
	if len(values) > last {
		values = values[len(values)-last:]
	}
	top, n := 0.0, 0
	for _, v := range values {
		if v != nil {
			if n == 0 || *v > top {
				top = *v
			}
			n++
		}
	}
	return top, n
}

// ---------------------------------------------------------------- header

func (a *App) drawHeader(c *Canvas) {
	v := a.view
	w := c.W
	x := 0
	c.Put(0, x, " TurboQuant Pro ", "bold")
	x += 16
	c.Put(0, x, "console ", "cyan")
	x += 8
	stamp := "--:--:--Z"
	if v != nil && v.Header.Stamp != "" {
		stamp = v.Header.Stamp
	}
	hint := stamp + "  q quit  ? keys  z zoom  i notes  P snap"
	if w-runeLen(hint) < 24 {
		hint = stamp + "  q quit  ? keys  z zoom"
	}
	room := w - runeLen(hint) - 3
	var labels []Span
	if st := a.engineState(); st.Text != "" {
		labels = append(labels, st)
	}
	if v != nil {
		labels = append(labels, v.Header.Labels...)
	}
	for _, l := range labels {
		if x+runeLen(l.Text) <= room {
			c.Put(0, x, l.Text, l.Role)
			x += runeLen(l.Text) + 1
		}
	}
	c.Put(0, w-runeLen(hint)-1, hint, "dim")
}

// ---------------------------------------------------------------- panels

func (a *App) title(n int) string { return a.hello.Titles[strconv.Itoa(n)] }

func (a *App) drawSystem(c *Canvas, y, x, hh, ww int) {
	c.Box(y, x, hh, ww, a.title(1), "dim", a.focus == 1)
	v := a.view
	inner := ww - 2
	cols := 1
	if inner >= 50 {
		cols = 2
	}
	rowsAvail := hh - 2
	cw := inner / cols
	var items []SysItem
	if v != nil && v.P1 != nil {
		items = v.P1.Items
	}
	nrows := (len(items) + cols - 1) / cols
	if nrows > rowsAvail {
		nrows = rowsAvail
	}
	for i, it := range items {
		if nrows == 0 || i >= nrows*cols {
			break
		}
		yy, xx := y+1+i%nrows, x+1+(i/nrows)*cw
		right := xx + cw - 2
		c.Put(yy, xx+1, it.Label, "dim")
		c.Put(yy, right-runeLen(it.Kind), it.Kind, "dim")
		role := ""
		if it.Dim {
			role = "dim"
		}
		c.Put(yy, right-runeLen(it.Kind)-1-runeLen(it.Value), it.Value, role)
	}
	// the two processes, measured: this client and the engine it draws for
	if rowsAvail-nrows >= 2 {
		eng := "engine: no reply"
		if v != nil {
			cpu := "-"
			if v.Engine.Cpu != nil {
				cpu = fmt.Sprintf("%.1f%%", *v.Engine.Cpu)
			}
			eng = fmt.Sprintf("engine pid %d  cpu %s  %s", v.Engine.Pid, cpu, v.Engine.Blas)
		}
		c.Put(y+hh-3, x+2, clip(eng, ww-4), "dim")
		c.Put(y+hh-2, x+2, clip(fmt.Sprintf("client pid %d  cpu %.1f%%", a.pid, a.selfCPU), ww-4), "dim")
	}
}

func (a *App) drawThroughput(c *Canvas, y, x, hh, ww int) {
	c.Box(y, x, hh, ww, a.title(2), "dim", a.focus == 2)
	v := a.view
	if v == nil || v.P2 == nil {
		return
	}
	labW := 9
	sw := ww - 4 - labW
	if sw < 4 {
		sw = 4
	}
	rows := 1
	if hh >= 9 {
		rows = 2
	}
	yy := y + 1
	n := 0
	for _, s := range v.P2.Series {
		c.Put(yy, x+2, fmt.Sprintf("%s %s %s", s.Name, s.Now, s.Unit), s.Role)
		line := Spark(s.Values, sw)
		for k := 0; k < rows; k++ {
			c.Put(yy+1+k, x+2, line, s.Role)
		}
		if top, cnt := maxReal(s.Values, sw); cnt >= 2 {
			c.Put(yy+1, x+3+sw, clip(pyFmt(&top, s.Digits), labW-1), "dim")
			c.Put(yy+rows, x+3+sw, clip("0 "+s.Unit, labW-1), "dim")
		}
		if s.Name == "QPS" {
			n = len(s.Values)
		}
		yy += rows + 1
	}
	if n > sw {
		n = sw
	}
	if yy < y+hh-1 {
		left := "-"
		if n > 0 {
			left = fmt.Sprintf("-%d s", n)
		}
		c.Put(yy, x+2, left, "dim")
		c.Put(yy, x+2+sw-3, "now", "dim")
		if a.annotate {
			mid := " " + v.P2.Note + " "
			off := (sw - runeLen(mid)) / 2
			if off < runeLen(left)+1 {
				off = runeLen(left) + 1
			}
			c.Put(yy, x+2+off, mid, "dim")
		}
	}
}

func (a *App) drawPipeline(c *Canvas, y, x, hh, ww int) {
	c.Box(y, x, hh, ww, a.title(3), "dim", a.focus == 3)
	v := a.view
	if v == nil || v.P3 == nil {
		return
	}
	mx := 1e-9
	first := true
	for _, s := range v.P3.Stages {
		if s.Ms != nil && (first || *s.Ms > mx) {
			mx, first = *s.Ms, false
		}
	}
	bw := ww - 20
	if bw < 4 {
		bw = 4
	}
	for i, s := range v.P3.Stages {
		step := 1
		if hh >= 9 {
			step = 2
		}
		yy := y + 1 + i*step
		c.Put(yy, x+2, ljust(s.Name, 7), "")
		frac := 0.0
		if s.Ms != nil {
			frac = *s.Ms / mx
		}
		c.Put(yy, x+9, bar(frac, bw), "teal")
		c.Put(yy, x+10+bw, rjust(s.Text, 7), "")
	}
	c.Put(y+hh-2, x+2, clip(v.P3.Note, ww-4), "dim")
}

func (a *App) drawReadscope(c *Canvas, y, x, hh, ww int) {
	c.Box(y, x, hh, ww, a.title(4), "purple", a.focus == 4)
	v := a.view
	if v == nil || v.P4 == nil {
		return
	}
	for i, r := range v.P4.Rows {
		if i >= hh-2 {
			break
		}
		c.Put(y+1+i, x+2, ljust(r.K, 12), "purple")
		c.Put(y+1+i, x+15, clip(r.V, ww-17), r.Role)
	}
}

func (a *App) drawIndex(c *Canvas, y, x, hh, ww int) {
	c.Box(y, x, hh, ww, a.title(5), "dim", a.focus == 5)
	v := a.view
	if v == nil || v.P5 == nil {
		return
	}
	if v.P5.Empty != "" {
		c.Put(y+1, x+2, clip(v.P5.Empty, ww-4), "dim")
		return
	}
	for i, r := range v.P5.Rows {
		if i >= hh-2 {
			break
		}
		c.Put(y+1+i, x+2, ljust(r.K, 10), "dim")
		c.Put(y+1+i, x+13, clip(r.V, ww-15), "")
	}
}

func (a *App) drawQueries(c *Canvas, y, x, hh, ww int) {
	c.Box(y, x, hh, ww, a.title(6), "dim", a.focus == 6)
	v := a.view
	if v == nil || v.P6 == nil {
		return
	}
	type col struct {
		QCol
		x int
	}
	var cols []col
	xx := x + 2
	for _, q := range v.P6.Cols {
		if xx+q.Width > x+ww-2 {
			break
		}
		cols = append(cols, col{q, xx})
		xx += q.Width + 1
	}
	for _, q := range cols {
		name := q.Name
		if q.Right {
			name = rjust(name, q.Width)
		}
		c.Put(y+1, q.x, name, "dim")
	}
	rows := v.P6.Rows
	if max := hh - 3; len(rows) > max {
		if max < 0 {
			max = 0
		}
		rows = rows[:max]
	}
	for i, r := range rows {
		sel := i == a.sel
		role := ""
		if sel {
			role = "sel"
		}
		for k, q := range cols {
			if k >= len(r) {
				break
			}
			val := r[k]
			if q.Right {
				val = rjust(val, q.Width)
			} else {
				val = ljust(val, q.Width)
			}
			c.Put(y+2+i, q.x, clip(val, q.Width), role)
		}
		if sel {
			c.Put(y+2+i, x+1, ">", "cyan")
		}
	}
}

func (a *App) drawNats(c *Canvas, y, x, hh, ww int) {
	v := a.view
	title := a.title(9)
	var nv *NatsView
	if v != nil {
		nv = v.P9
	}
	if nv != nil && nv.TitleExtra != "" {
		title += "  " + nv.TitleExtra
	}
	c.Box(y, x, hh, ww, title, "cyan", a.focus == 9)
	if nv == nil {
		return
	}
	switch nv.State {
	case "none":
		c.Put(y+1, x+2, clip(nv.Message, ww-4), "dim")
		return
	case "down":
		c.Put(y+1, x+2, clip(nv.Message, ww-4), "red")
		return
	}
	summary := nv.Summary.Text
	if runeLen(summary) > ww-4 {
		if i := strings.Index(summary, ", JS "); i > 0 {
			summary = summary[:i]
		}
	}
	c.Put(y+1, x+2, clip(summary, ww-4), nv.Summary.Role)
	valW, maxW := 27, 11
	sw := ww - 4 - valW - maxW
	if sw < 0 {
		sw = 0
	}
	for i, r := range nv.Rows {
		if i >= hh-3 {
			break
		}
		yy := y + 2 + i
		c.Put(yy, x+2, ljust(r.Label, 10), "dim")
		role := ""
		if len(r.Values) == 0 || r.Values[len(r.Values)-1] == nil {
			role = "dim"
		}
		c.Put(yy, x+12, clip(rjust(r.Value, 11), 11), role)
		c.Put(yy, x+23, " "+r.Kind, "dim")
		if sw >= 4 {
			c.Put(yy, x+2+valW, Spark(r.Values, sw), "cyan")
			if _, cnt := maxReal(r.Values, sw); cnt >= 2 && r.HasMax {
				c.Put(yy, x+3+valW+sw, clip("max "+r.Max, maxW-1), "dim")
			}
		}
	}
}

// ---------------------------------------------------------------- overlays

func (a *App) drawHelp(c *Canvas) {
	keys := a.hello.Keys
	switch a.instrument() {
	case "scope":
		keys = a.hello.KeysScope
	case "spectrum":
		keys = a.hello.KeysSpectrum
	}
	y, x, hh, ww := sheet(c, "keys", len(keys)+6, 76)
	for i, k := range keys {
		if 1+i >= hh-1 {
			break
		}
		c.Put(y+1+i, x+2, ljust(k[0], 14), "cyan")
		c.Put(y+1+i, x+17, clip(k[1], ww-19), "")
	}
	c.Put(y+len(keys)+2, x+2, clip(a.hello.HelpNote, ww-4), "dim")
}

// sheet clears a centred rectangle and frames it, as tui._sheet does.
func sheet(c *Canvas, title string, hh, ww int) (int, int, int, int) {
	if hh > c.H-2 {
		hh = c.H - 2
	}
	if ww > c.W-4 {
		ww = c.W - 4
	}
	y, x := (c.H-hh)/2, (c.W-ww)/2
	if y < 1 {
		y = 1
	}
	if x < 2 {
		x = 2
	}
	for r := 0; r < hh; r++ {
		c.Put(y+r, x, strings.Repeat(" ", ww), "")
	}
	c.Box(y, x, hh, ww, title, "purple", false)
	return y, x, hh, ww
}
