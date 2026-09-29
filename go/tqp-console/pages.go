// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

package main

// Pages. The engine names the pages a session has, from the sources attached
// (the index grid, the machine, NATS alone), and the panels on each; nothing is
// drawn for a source that is not attached. < and > move between pages, and each
// page keeps its own focus.
//
// The machine page's six panels share one shape (PanelView), drawn by one
// renderer: a summary line, calibrated rows (value, kind, a sparkline, a note),
// strips of per-CPU cells, and a table, each optional.

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"
)

// protocolVersion is viewmodel.PROTOCOL: a client and an engine from different
// releases refuse to pair rather than draw a view neither understands.
const protocolVersion = 2

const sparkMax = 30 // cells of history on a PanelView row

type Page struct {
	Name   string
	Panels []int
	Titles map[string]string
}

// PRow is [label, value, kind, series|null, note, role].
type PRow struct {
	Label, Value, Kind string
	Values             []*float64
	Note, Role         string
}

func (r *PRow) UnmarshalJSON(b []byte) error {
	a, err := tuple(b, 6)
	if err != nil {
		return err
	}
	r.Label, r.Value, r.Kind = str(a[0]), str(a[1]), str(a[2])
	json.Unmarshal(a[3], &r.Values)
	r.Note, r.Role = str(a[4]), str(a[5])
	return nil
}

// PStrip is [label, [fraction|null, ...]]: one cell per item (a logical CPU).
type PStrip struct {
	Label string
	Cells []*float64
}

func (s *PStrip) UnmarshalJSON(b []byte) error {
	a, err := tuple(b, 2)
	if err != nil {
		return err
	}
	s.Label = str(a[0])
	return json.Unmarshal(a[1], &s.Cells)
}

type PTable struct {
	Cols  []QCol
	Rows  [][]string
	Roles []string
}

type PanelView struct {
	State      string
	Message    string
	Summary    *Span
	TitleExtra string `json:"title_extra"`
	Rows       []PRow
	Strips     []PStrip
	Table      *PTable
	Note       string
}

// ---------------------------------------------------------------- navigation

func (a *App) page() Page {
	if len(a.hello.Pages) == 0 { // an engine that names no pages: the index grid
		return Page{Name: "index", Panels: []int{1, 2, 3, 4, 5, 6, 7, 8, 9}, Titles: a.hello.Titles}
	}
	return a.hello.Pages[a.pageIdx]
}

func (a *App) hasPanel(n int) bool {
	for _, p := range a.page().Panels {
		if p == n {
			return true
		}
	}
	return false
}

// stepFocus moves the focus by step (1 or -1) among the page's panels.
func (a *App) stepFocus(step int) {
	ps := a.page().Panels
	if len(ps) == 0 {
		return
	}
	i := 0
	for k, p := range ps {
		if p == a.focus {
			i = (k + step + len(ps)) % len(ps)
			break
		}
	}
	a.focus = ps[i]
}

// defaultFocus is where a page opens: the query stream on the index grid (the
// panel whose keys work at once), else the page's first panel.
func defaultFocus(p Page) int {
	for _, n := range p.Panels {
		if p.Name == "index" && n == 6 {
			return 6
		}
	}
	if len(p.Panels) > 0 {
		return p.Panels[0]
	}
	return 0
}

// setPage moves to page i (wrapping), keeping each page's focus.
func (a *App) setPage(i int) {
	n := len(a.hello.Pages)
	if n < 2 {
		a.setMessage("this session has one page")
		return
	}
	a.focusOf[a.page().Name] = a.focus
	a.pageIdx = ((i % n) + n) % n
	a.zoom, a.sel = "", 0
	p := a.page()
	if f, ok := a.focusOf[p.Name]; ok {
		a.focus = f
	} else {
		a.focus = defaultFocus(p)
	}
	a.rend.Invalidate()
	a.requestView()
}

// ---------------------------------------------------------------- the machine page

// drawMachine lays out the six machine panels in three rows of two: CPU and
// thermal (load beside the heat it makes), memory and GPU, disks and network.
func (a *App) drawMachine(c *Canvas) {
	w, h := c.W, c.H
	avail := h - 2
	top, mid := avail*3/10, avail*3/10
	if top < 8 {
		top = 8
	}
	if mid < 7 {
		mid = 7
	}
	if avail-top-mid < 5 { // a short terminal: three equal rows
		top, mid = avail/3, avail/3
	}
	bottom := avail - top - mid
	half := w / 2
	disksW := w * 46 / 100 // the network table is the wider one
	var pv map[string]*PanelView
	if a.view != nil {
		pv = a.view.Machine
	}
	y := 1
	cells := []struct{ n, y, x, h, w int }{
		{1, y, 0, top, half}, {2, y, half, top, w - half},
		{3, y + top, 0, mid, half}, {6, y + top, half, mid, w - half},
		{4, y + top + mid, 0, bottom, disksW}, {5, y + top + mid, disksW, bottom, w - disksW},
	}
	for _, p := range cells {
		drawPanelView(c, p.y, p.x, p.h, p.w, a.title(p.n), a.focus == p.n, pv[strconv.Itoa(p.n)])
	}
}

// drawPanelView draws one PanelView in a box. Rows that do not fit are dropped
// from the bottom (a table says how many), and the note takes the last line
// when there is room for it.
func drawPanelView(c *Canvas, y, x, hh, ww int, title string, focus bool, pv *PanelView) {
	if pv != nil && pv.TitleExtra != "" {
		title += "  " + pv.TitleExtra
	}
	c.Box(y, x, hh, ww, title, "dim", focus)
	if pv == nil || hh < 3 {
		return
	}
	inner := ww - 4
	yy, last := y+1, y+hh-2
	if pv.State != "ok" {
		c.Put(yy, x+2, clip(pv.Message, inner), "dim")
		return
	}
	if pv.Summary != nil && pv.Summary.Text != "" {
		c.Put(yy, x+2, clip(pv.Summary.Text, inner), pv.Summary.Role)
		yy++
	}
	noteRow := -1
	if pv.Note != "" && last-yy >= 2 {
		noteRow, last = last, last-1
	}
	yy = drawRows(c, yy, last, x+2, inner, pv.Rows)
	yy = drawStrips(c, yy, last, x+2, inner, pv.Strips)
	if pv.Table != nil {
		drawTable(c, yy, last, x+2, inner, pv.Table)
	}
	if noteRow >= 0 {
		c.Put(noteRow, x+2, clip(pv.Note, inner), "dim")
	}
}

func widest(n, lo, hi int, each func(i int) string) int {
	w := lo
	for i := 0; i < n; i++ {
		if l := runeLen(each(i)); l > w {
			w = l
		}
	}
	if w > hi {
		w = hi
	}
	return w
}

// drawRows: label, value (right-aligned, in the row's role), kind, a sparkline
// of the row's history, and a note, in columns sized to the rows.
func drawRows(c *Canvas, yy, last, x, inner int, rows []PRow) int {
	if len(rows) == 0 {
		return yy
	}
	labW := widest(len(rows), 4, 14, func(i int) string { return rows[i].Label })
	valW := widest(len(rows), 7, 12, func(i int) string { return rows[i].Value })
	fixed := labW + valW + 7 // two gaps and the 4-letter kind with its gap
	// the notes are read, the sparkline glanced at: notes get their width first,
	// and the sparkline at most sparkMax cells (a minute of history at 2 s polls)
	notes := widest(len(rows), 0, inner-fixed, func(i int) string { return rows[i].Note })
	sparkW := inner - fixed - notes - 1
	if sparkW > sparkMax {
		sparkW = sparkMax
	}
	if sparkW < 4 {
		sparkW = 0
	}
	noteW := inner - fixed
	if sparkW > 0 {
		noteW -= sparkW + 1
	}
	if noteW < 0 {
		noteW = 0
	}
	for _, r := range rows {
		if yy > last {
			break
		}
		cx := x
		c.Put(yy, cx, ljust(clip(r.Label, labW), labW), "dim")
		cx += labW + 1
		role := r.Role
		if r.Value == "-" {
			role = "dim"
		}
		c.Put(yy, cx, rjust(clip(r.Value, valW), valW), role)
		cx += valW + 1
		c.Put(yy, cx, ljust(r.Kind, 4), "dim")
		cx += 5
		if sparkW > 0 {
			if len(r.Values) > 0 {
				c.Put(yy, cx, Spark(r.Values, sparkW), "teal")
			}
			cx += sparkW + 1
		}
		if noteW > 0 && r.Note != "" {
			c.Put(yy, cx, clip(r.Note, noteW), "dim")
		}
		yy++
	}
	return yy
}

// cell is one fraction as a block height; an unknown one is a dot.
func cell(f *float64) rune {
	if f == nil {
		return '·'
	}
	ramp := []rune(sparkRamp)
	i := int(math.RoundToEven(*f * 8))
	if i < 1 {
		i = 1
	}
	if i > 8 {
		i = 8
	}
	return ramp[i]
}

func drawStrips(c *Canvas, yy, last, x, inner int, strips []PStrip) int {
	if len(strips) == 0 {
		return yy
	}
	labW := widest(len(strips), 4, 10, func(i int) string { return strips[i].Label })
	for _, s := range strips {
		if yy > last {
			break
		}
		c.Put(yy, x, ljust(clip(s.Label, labW), labW), "dim")
		var b strings.Builder
		for i, f := range s.Cells {
			if i >= inner-labW-1 {
				break
			}
			b.WriteRune(cell(f))
		}
		c.Put(yy, x+labW+1, b.String(), "teal")
		yy++
	}
	return yy
}

// drawTable: a header and rows, columns dropped from the right when they do
// not fit; rows that do not fit are counted on the last line.
func drawTable(c *Canvas, yy, last, x, inner int, t *PTable) {
	if yy > last {
		return
	}
	var cols []QCol
	used := 0
	for _, q := range t.Cols {
		if used+q.Width > inner {
			break
		}
		cols = append(cols, q)
		used += q.Width + 1
	}
	cx := x
	for _, q := range cols {
		name := ljust(q.Name, q.Width)
		if q.Right {
			name = rjust(q.Name, q.Width)
		}
		c.Put(yy, cx, name, "dim")
		cx += q.Width + 1
	}
	yy++
	room := last - yy + 1
	rows := t.Rows
	more := 0
	if len(rows) > room {
		if room < 1 {
			return
		}
		more = len(rows) - (room - 1)
		rows = rows[:room-1]
	}
	for i, r := range rows {
		role := ""
		if i < len(t.Roles) {
			role = t.Roles[i]
		}
		cx = x
		for k, q := range cols {
			if k >= len(r) {
				break
			}
			v := ljust(clip(r[k], q.Width), q.Width)
			if q.Right {
				v = rjust(clip(r[k], q.Width), q.Width)
			}
			c.Put(yy, cx, v, role)
			cx += q.Width + 1
		}
		yy++
	}
	if more > 0 {
		c.Put(yy, x, clip("+"+strconv.Itoa(more)+" more (a taller terminal shows them)", inner), "dim")
	}
}

// ---------------------------------------------------------------- the NATS page

// drawNatsPage: the scope over the fabric's signals (panel 7) on top, and
// panel 9 below it, the same calibrated rows as on the index grid (they fit any
// height and say so when the server is unreachable); z on either opens it full
// screen, 9 as the whole fabric instrument.
func (a *App) drawNatsPage(c *Canvas) {
	w, h := c.W, c.H
	g := a.geometry(w, h)
	y := 1
	c.Box(y, 0, g.inst, w, a.title(7), "dim", a.focus == 7)
	if a.view != nil && a.view.P7 != nil {
		sub := NewCanvas(w-2, g.inst-2)
		DrawScope(sub, a.view.P7, false, a.annotate)
		c.Blit(sub, y+1, 1)
	}
	y += g.inst
	a.drawNats(c, y, 0, h-1-y, w)
}
