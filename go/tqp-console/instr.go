// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

package main

// The scope and the spectrum analyzer, drawn from the view model onto a canvas
// of their own (a grid panel's interior, or the whole screen when zoomed). The
// geometry and the braille plotting follow turboquant_pro/console/scope_view.py
// and spectrum_view.py; the numbers and every label come from the engine.

import (
	"strconv"
	"strings"
)

const (
	yl      = 7 // columns left of the graticule for the y scale
	braille = 0x2800
	sideW   = 26 // the zoomed scope's channel/trigger panel
)

// braille dot for sub-column dx (0|1) and sub-row dy (0 = top .. 3 = bottom)
var dotBit = [2][4]int{{0x01, 0x02, 0x04, 0x40}, {0x08, 0x10, 0x20, 0x80}}

// ScopeGeometry is the graticule interior (gw, gh) for a scope canvas w x h.
func ScopeGeometry(w, h int, zoom bool) (int, int) {
	side := 0
	if zoom && w >= 110 {
		side = sideW
	}
	extra := 4
	if zoom {
		extra = 8
	}
	return w - side - 2 - yl, h - extra
}

// SpectrumGeometry is (gw, gh, waterfall rows) for a spectrum canvas w x h.
func SpectrumGeometry(w, h int, zoom bool) (int, int, int) {
	wf := 0
	if zoom && h >= 30 {
		wf = (h - 6) / 4
		if wf < 3 {
			wf = 3
		}
	}
	gw := w - 2 - yl
	if !zoom {
		return gw, h - 5, 0
	}
	gh := h - 7
	if wf > 0 {
		gh -= wf + 1
	}
	return gw, gh, wf
}

func graticule(c *Canvas, y0, x0, gw, gh, hdiv, vdiv int) {
	c.Box(y0, x0, gh+2, gw+2, "", "dim", false)
	for j := 0; j < gh; j++ {
		for i := 0; i < gw; i++ {
			onV := (i*hdiv)%gw < hdiv
			onH := (j*vdiv)%gh < vdiv
			if onV && onH {
				c.Put(y0+1+j, x0+1+i, "┼", "grid")
			} else if onV && j%2 == 0 || onH && i%2 == 0 {
				c.Put(y0+1+j, x0+1+i, "·", "grid")
			}
		}
	}
}

type cellKey struct{ y, x int }

func plotCells(c *Canvas, y0, x0 int, cells map[cellKey]int, role string) {
	for k, bits := range cells {
		c.Put(y0+1+k.y, x0+1+k.x, string(rune(braille+bits)), role)
	}
}

func softkeys(c *Canvas, keys []string) {
	x := 0
	for _, k := range keys {
		if x+runeLen(k)+3 > c.W {
			break
		}
		c.Put(c.H-1, x, "["+k+"]", "dim")
		x += runeLen(k) + 3
	}
}

func notes(c *Canvas, lines []Span, y0, x0, gw, maxLines int) {
	for i, l := range lines {
		if i >= maxLines {
			break
		}
		c.Put(y0+1+i, x0+2, clip(" "+l.Text+" ", gw-2), l.Role)
	}
}

func xAxis(c *Canvas, ticks []Tick, y, x0, gw, hdiv int) {
	every := 1
	if gw < 8*hdiv {
		every = 2
	}
	for i, t := range ticks {
		if i%every != 0 && i != len(ticks)-1 { // the right end (now) always
			continue
		}
		w := runeLen(t.Text)
		x := x0 + 1 + int(t.At*float64(gw))
		if i == len(ticks)-1 {
			x -= w
		} else {
			x -= w / 2
		}
		if x < x0 {
			x = x0
		}
		c.Put(y, x, t.Text, "dim")
	}
}

func legend(c *Canvas, items []Span, y0, x0, gw int) {
	x := x0 + 2
	for _, it := range items {
		if x+runeLen(it.Text) > x0+gw {
			break
		}
		c.Put(y0, x, it.Text, it.Role)
		x += runeLen(it.Text) + 1
	}
}

// ---------------------------------------------------------------- scope

func DrawScope(c *Canvas, v *ScopeView, zoom, annotate bool) {
	if v == nil {
		c.Put(1, 1, "instrument not running", "dim")
		return
	}
	w, h := c.W, c.H
	gw, gh := v.Gw, v.Gh
	y0, x0 := 1, yl
	vdiv, hdiv := v.Vdiv, v.Hdiv
	if vdiv == 0 {
		vdiv, hdiv = 8, 10
	}
	x := 0
	for _, s := range v.Status {
		c.Put(0, x, s.Text, s.Role)
		x += runeLen(s.Text)
	}
	if gw < 2 || gh < 2 {
		return
	}
	graticule(c, y0, x0, gw, gh, hdiv, vdiv)
	if v.Fft != nil {
		drawFFT(c, v, y0, x0, gw, gh)
		if zoom {
			softkeys(c, v.Softkeys)
		}
		return
	}
	// calibration: the selected channel's scale, time below, legend on top
	if v.Yunit != nil {
		step := 1
		if gh < 2*vdiv {
			step = 2
		}
		for k := 0; k <= vdiv; k += step {
			row := 0
			if k < vdiv {
				row = gh - 1 - int(float64(k)*float64(gh)/float64(vdiv))
			}
			if k < len(v.Yticks) {
				c.Put(y0+1+row, x0-yl, rjust(v.Yticks[k].Text, yl-1), v.Yunit.Role)
			}
		}
		c.Put(y0, 0, rjust(clip(v.Yunit.Text, yl-1), yl-1), v.Yunit.Role)
	}
	xAxis(c, v.Xticks, y0+gh+2, x0, gw, hdiv)
	legend(c, v.Legend, y0, x0, gw)

	rowOf := func(div float64, sub int) int { return int(div / float64(vdiv) * float64(gh*sub)) }
	for _, ch := range v.Channels {
		if !ch.On {
			continue
		}
		for _, p := range ch.Persist {
			role := ch.Role
			if p[2] == 1 {
				role += "_dim"
			}
			c.Put(y0+1+p[0], x0+1+p[1], "░", role)
		}
		cells := map[cellKey]int{}
		subH := gh * 4
		for sx, cc := range ch.Cols {
			if len(cc) < 2 {
				continue
			}
			lo, hi := rowOf(cc[0], 4), rowOf(cc[1], 4)
			if lo < 0 {
				lo = 0
			}
			if hi > subH-1 {
				hi = subH - 1
			}
			for sy := lo; sy <= hi; sy++ {
				k := cellKey{gh - 1 - sy/4, sx / 2}
				cells[k] |= dotBit[sx%2][3-sy%4]
			}
			if hi < lo { // off the graticule: mark the edge it left by
				edge := gh - 1
				if lo >= subH {
					edge = 0
				}
				cells[cellKey{edge, sx / 2}] |= 0x09
			}
		}
		role := ch.Role
		if ch.N == v.Sel {
			role += "_bold"
		}
		plotCells(c, y0, x0, cells, role)
		if gy := gh - 1 - rowOf(ch.GroundDiv, 1); gy >= 0 && gy < gh {
			c.Put(y0+1+gy, x0, strconv.Itoa(ch.N), ch.Role)
		}
	}
	for _, ch := range v.Channels {
		if !ch.On || ch.Ref == nil {
			continue
		}
		ry := gh - 1 - rowOf(ch.Ref.At, 1)
		if ry >= 0 && ry < gh {
			for i := 1; i < gw; i += 3 {
				c.Put(y0+1+ry, x0+1+i, "╌", "purple")
			}
			lx := gw - runeLen(ch.Ref.Text)
			if lx < 1 {
				lx = 1
			}
			c.Put(y0+1+ry, x0+lx, ch.Ref.Text, "purple")
		}
	}
	if v.TriggerLevel != nil {
		if ty := gh - 1 - rowOf(v.TriggerLevel.Div, 1); ty >= 0 && ty < gh {
			c.Put(y0+1+ty, x0+gw+1, "◀", v.TriggerLevel.Role)
		}
	}
	if v.TriggerX != nil {
		tx := int(*v.TriggerX * float64(gw))
		if tx < 0 {
			tx = 0
		}
		if tx > gw-1 {
			tx = gw - 1
		}
		c.Put(y0, x0+1+tx, "▼", "cyan")
	}
	if zoom && w >= 110 && len(v.Side) > 0 {
		sx := w - sideW
		c.Box(y0, sx, gh+2, sideW, "channels", "dim", false)
		for i, l := range v.Side {
			if i >= gh {
				break
			}
			c.Put(y0+1+i, sx+1, clip(l.Text, sideW-2), l.Role)
		}
	}
	if !zoom {
		if annotate {
			notes(c, v.Notes, y0, x0, gw, gh-1)
		}
		return
	}
	for i, m := range v.Meas {
		c.Put(y0+gh+3+i, 1, clip(m.Text, w-2), m.Role)
	}
	if annotate {
		notes(c, v.Notes, y0, x0, gw, gh-1)
	}
	softkeys(c, v.Softkeys)
	_ = h
}

func drawFFT(c *Canvas, v *ScopeView, y0, x0, gw, gh int) {
	f := v.Fft
	c.Put(0, 0, strings.Repeat(" ", c.W), "")
	c.Put(0, 0, clip(f.Head.Text, c.W), f.Head.Role)
	if f.Message != nil {
		c.Put(y0+2, 3, f.Message.Text, f.Message.Role)
		return
	}
	cells := map[cellKey]int{}
	sub := gh * 4
	prev := -1
	for sx, fr := range f.Fracs {
		ry := int(fr * float64(sub))
		lo, hi := ry, ry
		if prev >= 0 {
			lo, hi = min(ry, prev), max(ry, prev)
		}
		prev = ry
		for sy := lo; sy <= hi; sy++ {
			cells[cellKey{gh - 1 - sy/4, sx / 2}] |= dotBit[sx%2][3-sy%4]
		}
	}
	plotCells(c, y0, x0, cells, f.Role)
	c.Put(y0+gh+2, 1, clip(f.Caption.Text, c.W-2), f.Caption.Role)
}

// ---------------------------------------------------------------- spectrum

func DrawSpectrum(c *Canvas, v *SpecView, zoom, annotate bool) {
	if v == nil {
		c.Put(1, 1, "instrument not running", "dim")
		return
	}
	w := c.W
	gw, gh := v.Gw, v.Gh
	y0, x0 := 1, yl
	vdiv, hdiv := v.Vdiv, v.Hdiv
	if vdiv == 0 {
		vdiv, hdiv = 8, 10
	}
	x := 0
	for _, s := range v.Status {
		c.Put(0, x, s.Text, s.Role)
		x += runeLen(s.Text)
	}
	if !v.Ok {
		if v.Reason != nil {
			c.Put(2, 2, v.Reason.Text, v.Reason.Role)
		}
		return
	}
	if gw < 2 || gh < 2 {
		return
	}
	graticule(c, y0, x0, gw, gh, hdiv, vdiv)
	span := v.RefDb - v.Bottom
	if span == 0 {
		span = 1
	}
	rowOf := func(db float64, sub int, delta bool) int {
		if delta {
			return int((db/(v.DbDiv*float64(vdiv)) + 0.5) * float64(gh*sub))
		}
		return int((db - v.Bottom) / span * float64(gh*sub))
	}
	if v.Delta {
		cy := gh - 1 - rowOf(0, 1, true)
		c.Put(y0+1+cy, x0+2, " Δ 0 dB ", "purple")
	}
	if v.LimitDb != nil {
		if r := gh - 1 - rowOf(*v.LimitDb, 1, false); r >= 0 && r < gh {
			for i := 0; i < gw; i += 2 {
				c.Put(y0+1+r, x0+1+i, "─", "red")
			}
			c.Put(y0+1+r, x0+gw-11, " water lvl ", "red")
		}
	}
	for _, tr := range v.Traces {
		dl := tr.Mode == "delta"
		cells := map[cellKey]int{}
		prev, have := 0, false
		for sx, val := range tr.Cols {
			if val == nil {
				have = false
				continue
			}
			ry := rowOf(*val, 4, dl)
			lo, hi := ry, ry
			if have {
				lo, hi = min(ry, prev), max(ry, prev)
			}
			prev, have = ry, true
			if lo < 0 {
				lo = 0
			}
			if hi > gh*4-1 {
				hi = gh*4 - 1
			}
			for sy := lo; sy <= hi; sy++ {
				cells[cellKey{gh - 1 - sy/4, sx / 2}] |= dotBit[sx%2][3-sy%4]
			}
		}
		role := tr.Role
		if tr.N == v.Sel {
			role += "_bold"
		}
		plotCells(c, y0, x0, cells, role)
	}
	for _, m := range v.Markers {
		cx := int(m.Frac * float64(gw))
		if r := gh - 1 - rowOf(m.Db, 1, false); r >= 0 && r < gh {
			c.Put(y0+1+r, x0+1+cx, m.Glyph, "bold")
		}
	}
	// calibration
	step := 1
	if gh < 2*vdiv {
		step = 2
	}
	for k := 0; k <= vdiv && k < len(v.Yticks); k += step {
		row := gh - 1 - rowOf(v.Yticks[k].At, 1, false)
		if row < 0 {
			row = 0
		}
		if row > gh-1 {
			row = gh - 1
		}
		c.Put(y0+1+row, x0-yl, rjust(v.Yticks[k].Text, yl-1), "dim")
	}
	c.Put(y0, 0, rjust("dB", yl-1), "dim")
	xAxis(c, v.Xticks, y0+gh+2, x0, gw, hdiv)
	c.Put(y0+gh+2, x0+gw+2-4, " dir", "dim")
	legend(c, v.Legend, y0, x0, gw)

	y := y0 + gh + 3
	if zoom && v.Waterfall != nil {
		ramp := []rune(" ░▒▓█")
		c.Put(y, 1, v.Waterfall.Label, "dim")
		for k, row := range v.Waterfall.Levels {
			for i, lv := range row {
				if lv < 0 || lv >= len(ramp) {
					continue
				}
				c.Put(y+1+k, x0+1+i, string(ramp[lv]), "cyan")
			}
		}
		y += len(v.Waterfall.Levels) + 1
	}
	if zoom {
		c.Put(y, 1, clip(v.Names, w-2), "")
	} else {
		y--
	}
	c.Put(y+1, 1, clip(v.Readout.Text, w-2), v.Readout.Role)
	if annotate {
		for k, l := range v.Notes {
			c.Put(2+k, yl+2, clip(" "+l.Text+" ", gw-2), l.Role)
		}
	}
	if zoom {
		softkeys(c, v.Softkeys)
	}
}
