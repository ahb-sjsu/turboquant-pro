// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

package main

// The client: layout, keys, and the terminal's life cycle.
//
// Job control, as a full-screen program should do it:
//   - Ctrl-Z (a byte in raw mode) or SIGTSTP: restore the terminal, then stop.
//   - SIGCONT in the foreground (fg, or continued after an outside SIGSTOP):
//     take the terminal again and redraw every cell.
//   - SIGCONT in the background (bg, or an outside SIGSTOP/SIGCONT such as a
//     thermal guard's): give the terminal back in full, draw and read nothing,
//     and wait to be brought to the foreground.
//   - SIGINT / SIGTERM / SIGHUP and every error: restore the terminal, exit.
//   - The terminal hung up (its window closed) with no SIGHUP delivered, as
//     happens behind a relay such as screen or sudo's pty: seen on the terminal
//     itself (a failed read, then POLLHUP), and handled as SIGHUP.
// SIGTTOU and SIGTTIN are ignored so that the terminal can always be restored,
// and so that the process never stops itself on a background read or write.

import (
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// refresh is how often the client asks the engine for a frame of data. The
// instruments update at 4 Hz inside the engine; 2 Hz on screen is what a person
// reads (btop's default is 0.5 Hz), and it halves both processes' work.
const refresh = 500 * time.Millisecond

type App struct {
	term  *Term
	eng   *Engine
	hello Hello

	view      *View
	viewAt    time.Time
	viewErr   error
	viewBusy  bool
	enginePid int

	focus, sel     int
	zoom, overlay  string
	sheet          *Sheet
	inspectID      string
	annotate       bool
	message        string
	messageAt      time.Time
	exportDir      string
	pid            int
	selfCPU        float64
	cpuMark        [2]float64 // wall seconds, cpu seconds
	waitForeground bool
	shownState     Span
	pageIdx        int            // into hello.Pages
	focusOf        map[string]int // each page's focus, kept across < and >

	rend  Renderer
	last  *Canvas
	jobs  chan func() func() // run on the engine worker; the result runs on main
	done  chan func()
	keysC chan []string
	hupC  chan struct{} // closed when the terminal has hung up
}

func NewApp(t *Term, e *Engine, enginePid int, exportDir string) *App {
	return &App{
		term: t, eng: e, enginePid: enginePid, exportDir: exportDir,
		focus: 6, annotate: true, pid: os.Getpid(),
		jobs:  make(chan func() func(), 16),
		done:  make(chan func(), 16),
		keysC: make(chan []string, 16),
		hupC:  make(chan struct{}),

		focusOf: map[string]int{},
	}
}

func (a *App) setMessage(m string) {
	a.message, a.messageAt = m, time.Now()
}

// ---------------------------------------------------------------- engine work

func (a *App) worker() {
	for job := range a.jobs {
		func() {
			defer func() {
				if r := recover(); r != nil {
					a.done <- func() { a.setMessage(fmt.Sprintf("engine call failed: %v", r)) }
				}
			}()
			if apply := job(); apply != nil {
				a.done <- apply
			}
		}()
	}
}

func (a *App) submit(job func() func()) {
	select {
	case a.jobs <- job:
	default:
		a.setMessage("engine busy; key dropped")
	}
}

type geometry struct {
	w, h                int
	top, inst, mid, q   int
	scope, spectrum     [3]int
	scopeZoom, specZoom bool
	fabricW, fabricH    int
}

func (a *App) geometry(w, h int) geometry {
	g := geometry{w: w, h: h}
	switch a.zoom {
	case "scope":
		gw, gh := ScopeGeometry(w, h-2, true, ScopeAxes(w, true, a.scopeChannelsOn()))
		g.scope, g.scopeZoom = [3]int{gw, gh, 0}, true
		return g
	case "spectrum":
		gw, gh, wf := SpectrumGeometry(w, h-2, true)
		g.spectrum, g.specZoom = [3]int{gw, gh, wf}, true
		return g
	case "fabric":
		g.fabricW, g.fabricH = w, h-2
		return g
	}
	switch a.page().Name {
	case "machine", "dht":
		return g
	case "nats": // the scope over the fabric's signals, the fabric below it
		// the fabric instrument (server, leaf links, clients, events) gets the
		// rows it says it needs to list every client, the scope the rest; on a
		// short terminal the fabric takes what the scope's minimum leaves, and
		// below the fabric's own minimum panel 9's rows stand in
		avail := h - 2
		need := fabricMinH
		if a.view != nil && a.view.Fabric != nil && a.view.Fabric.Need > need {
			need = a.view.Fabric.Need
		}
		if fab := avail - scopeMinH - 2; fab >= fabricMinH {
			if fab > need {
				fab = need
			}
			g.fabricW, g.fabricH = w-2, fab
			g.inst = avail - (fab + 2)
		} else {
			g.inst = avail * 55 / 100
			if g.inst < scopeMinH {
				g.inst = scopeMinH
			}
		}
		gw, gh := ScopeGeometry(w-2, g.inst-2, false, ScopeAxes(w-2, false, a.scopeChannelsOn()))
		g.scope = [3]int{gw, gh, 0}
		return g
	}
	g.top = 7
	if h >= 30 {
		g.top = 9
	}
	fab := a.hasPanel(9) && a.view != nil && a.view.P9 != nil && a.view.P9.State != "none"
	switch {
	case fab && h >= 44:
		g.mid = 13
	case h >= 44:
		g.mid = 10
	case h >= 30:
		g.mid = 8
	default:
		g.mid = 6
	}
	avail := h - 2 - g.top - g.mid
	g.q = avail / 3
	if g.q < 5 {
		g.q = 5
	}
	g.inst = avail - g.q
	if g.inst < 10 {
		g.inst, g.q = 1, avail-1
	}
	if g.inst > 1 {
		half := w / 2
		sgw, sgh := ScopeGeometry(half-2, g.inst-2, false, ScopeAxes(half-2, false, a.scopeChannelsOn()))
		pgw, pgh, _ := SpectrumGeometry(w-half-2, g.inst-2, false)
		g.scope = [3]int{sgw, sgh, 0}
		g.spectrum = [3]int{pgw, pgh, 0}
	}
	return g
}

func (a *App) requestView() {
	if a.viewBusy {
		return
	}
	a.viewBusy = true
	w, h := a.term.Size()
	g := a.geometry(w, h)
	req := map[string]any{"op": "view", "page": a.page().Name, "grid": a.zoom == ""}
	if g.scope[0] > 1 && g.scope[1] > 1 {
		req["scope"], req["scope_zoom"] = []int{g.scope[0], g.scope[1]}, g.scopeZoom
	}
	if g.spectrum[0] > 1 && g.spectrum[1] > 1 {
		req["spectrum"], req["spectrum_zoom"] = []int{g.spectrum[0], g.spectrum[1], g.spectrum[2]}, g.specZoom
	}
	if g.fabricW > 0 {
		req["fabric"] = []int{g.fabricW, g.fabricH}
	}
	a.submit(func() func() {
		var v View
		err := a.eng.Call(req, &v, 1500*time.Millisecond)
		return func() {
			a.viewBusy = false
			if err != nil {
				a.viewErr = err
				return
			}
			a.view, a.viewAt, a.viewErr = &v, time.Now(), nil
			if v.Message != "" {
				a.setMessage(v.Message)
			}
		}
	})
}

func (a *App) engineCall(req map[string]any, then func(KeyReply)) {
	a.submit(func() func() {
		var r KeyReply
		err := a.eng.Call(req, &r, 3*time.Second)
		return func() {
			if err != nil {
				a.setMessage("engine: " + err.Error())
				return
			}
			if r.Message != "" {
				a.setMessage(r.Message)
			}
			if then != nil {
				then(r)
			}
			a.requestView()
		}
	})
}

func (a *App) requestInspect(id string, replay bool) {
	if id == "" {
		return
	}
	w, h := a.term.Size()
	req := map[string]any{"op": "inspect", "id": id, "replay": replay, "w": w, "h": h}
	a.submit(func() func() {
		var s Sheet
		err := a.eng.Call(req, &s, 10*time.Second)
		return func() {
			if err != nil {
				a.setMessage("engine: " + err.Error())
				return
			}
			if s.Message != "" && len(s.Rect) < 4 {
				a.setMessage(s.Message)
				return
			}
			a.sheet, a.overlay, a.inspectID = &s, "inspect", id
		}
	})
}

// engineState is the header's warning when the engine is not answering.
func (a *App) engineState() Span {
	if a.viewErr == nil && time.Since(a.viewAt) < 3*time.Second {
		return Span{}
	}
	stat, err := os.ReadFile(fmt.Sprintf("/proc/%d/stat", a.enginePid))
	if err != nil {
		return Span{"[engine gone: q to quit]", "red"}
	}
	if i := strings.LastIndexByte(string(stat), ')'); i > 0 && i+2 < len(stat) && stat[i+2] == 'T' {
		return Span{"[engine STOPPED from outside: data held]", "red"}
	}
	return Span{fmt.Sprintf("[engine: no reply %.0fs]", time.Since(a.viewAt).Seconds()), "amber"}
}

func (a *App) measureSelf() {
	stat, err := os.ReadFile("/proc/self/stat")
	if err != nil {
		return
	}
	s := string(stat)
	i := strings.LastIndexByte(s, ')')
	f := strings.Fields(s[i+2:])
	if len(f) < 13 {
		return
	}
	ut, _ := strconv.ParseFloat(f[11], 64)
	stt, _ := strconv.ParseFloat(f[12], 64)
	cpu := (ut + stt) / 100.0 // clock ticks (USER_HZ = 100 on Linux)
	now := float64(time.Now().UnixNano()) / 1e9
	if a.cpuMark[0] > 0 && now-a.cpuMark[0] >= 1.5 {
		a.selfCPU = 100 * (cpu - a.cpuMark[1]) / (now - a.cpuMark[0])
		a.cpuMark = [2]float64{now, cpu}
	} else if a.cpuMark[0] == 0 {
		a.cpuMark = [2]float64{now, cpu}
	}
}

// ---------------------------------------------------------------- keys

func (a *App) onKey(k string) (code int, quit bool) {
	switch k {
	case "ctrl+c":
		return 130, true
	case "ctrl+z":
		a.suspend()
		return 0, false
	case "q":
		return 0, true
	case "escape":
		if a.overlay == "" && a.zoom != "" {
			a.zoom = ""
			a.requestView()
		}
		a.overlay, a.sheet = "", nil
		return 0, false
	case "?":
		if a.overlay == "help" {
			a.overlay = ""
		} else {
			a.overlay = "help"
		}
		return 0, false
	}
	if a.overlay == "inspect" {
		if k == "r" {
			a.requestInspect(a.inspectID, true)
		}
		return 0, false
	}
	if a.overlay != "" {
		return 0, false
	}
	switch k {
	case "<":
		a.setPage(a.pageIdx - 1)
		return 0, false
	case ">":
		a.setPage(a.pageIdx + 1)
		return 0, false
	case "z":
		if a.zoom != "" {
			a.zoom = ""
		} else if z, ok := a.hello.Zoomable[strconv.Itoa(a.focus)]; ok && a.hasPanel(a.focus) {
			a.zoom = z
		} else {
			a.setMessage(a.zoomHint())
		}
		a.requestView()
		return 0, false
	case "i":
		a.annotate = !a.annotate
		if a.annotate {
			a.setMessage("notes on")
		} else {
			a.setMessage("notes off")
		}
		return 0, false
	case "S":
		a.engineCall(map[string]any{"op": "action", "name": "setup", "zoom": a.zoom, "page": a.page().Name}, nil)
		return 0, false
	case "e":
		a.engineCall(map[string]any{"op": "action", "name": "export"}, nil)
		return 0, false
	case "P":
		a.snapshot()
		return 0, false
	}
	if a.zoom == "fabric" {
		return 0, false
	}
	// the focused instrument takes its keys, in the grid as when zoomed; Tab and
	// Shift-Tab always move the focus on
	if inst := a.instrument(); inst != "" && !(a.zoom == "" && (k == "tab" || k == "backtab")) {
		a.engineCall(map[string]any{"op": "key", "zoom": inst, "name": k, "page": a.page().Name}, func(r KeyReply) {
			if r.Inspect != "" {
				a.requestInspect(r.Inspect, false)
			}
		})
		return 0, false
	}
	var ids []string
	if a.view != nil && a.view.P6 != nil {
		ids = a.view.P6.Ids
	}
	switch k {
	case "p":
		a.engineCall(map[string]any{"op": "action", "name": "pause"}, nil)
	case "down", "j":
		if a.sel < len(ids)-1 {
			a.sel++
		}
	case "up", "k":
		if a.sel > 0 {
			a.sel--
		}
	case "enter":
		if a.sel < len(ids) {
			a.requestInspect(ids[a.sel], false)
		}
	case "r":
		if a.sel < len(ids) {
			a.requestInspect(ids[a.sel], true)
		}
	case "tab":
		a.stepFocus(1)
	case "backtab":
		a.stepFocus(-1)
	default:
		if len(k) == 1 && k[0] >= '1' && k[0] <= '9' && a.hasPanel(int(k[0]-'0')) {
			a.focus = int(k[0] - '0')
		}
	}
	return 0, false
}

// scopeChannelsOn is how many scope channels the last view had on (an axis each).
func (a *App) scopeChannelsOn() int {
	if a.view == nil || a.view.P7 == nil {
		return 1
	}
	return len(a.view.P7.Axes)
}

// instrument is the instrument that takes the keys: the zoomed one, else the
// focused panel if it is a scope (7, on the index or the NATS page) or the
// spectrum (8). The engine is told the page, so the right scope answers.
func (a *App) instrument() string {
	switch {
	case a.zoom == "scope" || a.zoom == "spectrum":
		return a.zoom
	case a.zoom != "":
		return ""
	case a.focus == 7 && a.hasPanel(7):
		return "scope"
	case a.focus == 8 && a.hasPanel(8):
		return "spectrum"
	}
	return ""
}

// zoomHint says which panels of this page z opens.
func (a *App) zoomHint() string {
	names := map[string]string{"scope": "scope", "spectrum": "spectrum", "fabric": "NATS"}
	var can []string
	for _, n := range a.page().Panels {
		if z, ok := a.hello.Zoomable[strconv.Itoa(n)]; ok {
			can = append(can, strconv.Itoa(n)+" ("+names[z]+")")
		}
	}
	if len(can) == 0 {
		return "no panel on this page zooms"
	}
	return "z opens " + strings.Join(can, ", ")
}

func (a *App) snapshot() {
	if a.last == nil {
		return
	}
	name := "tqp-console-" + time.Now().UTC().Format("20060102T150405Z") + ".txt"
	path := filepath.Join(a.exportDir, name)
	body := strings.Join(a.last.Lines(), "\n") + "\n"
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		a.setMessage("snapshot failed: " + err.Error())
		return
	}
	a.setMessage("snapshot written: " + path)
}

// ---------------------------------------------------------------- job control

func (a *App) suspend() {
	var r KeyReply
	a.eng.Call(map[string]any{"op": "action", "name": "hold"}, &r, 300*time.Millisecond)
	a.term.Leave()
	syscall.Kill(a.pid, syscall.SIGSTOP)
	// continued: SIGCONT arrives on the signal channel and onContinue runs
}

func (a *App) onContinue() {
	if a.term.Foreground() {
		a.term.Enter() // the shell may have changed the modes while we were stopped
		a.rend.Invalidate()
		a.waitForeground = false
	} else {
		a.term.Leave()
		a.waitForeground = true
	}
	a.submit(func() func() {
		var r KeyReply
		a.eng.Call(map[string]any{"op": "action", "name": "release"}, &r, 500*time.Millisecond)
		return nil
	})
}

// ---------------------------------------------------------------- frame

func (a *App) draw() {
	if !a.term.Entered() || !a.term.Foreground() {
		return
	}
	w, h := a.term.Size()
	c := NewCanvas(w, h)
	if w < 80 || h < 24 {
		c.Put(0, 0, clip(fmt.Sprintf("terminal %dx%d is too small: 80x24 at least (q quits)", w, h), w), "amber")
	} else {
		a.drawHeader(c)
		switch {
		case isPanelPage(a.page().Name):
			a.drawPanelPage(c)
		case a.zoom != "":
			a.drawZoom(c, a.zoom)
		case a.page().Name == "nats":
			a.drawNatsPage(c)
		default:
			a.drawGrid(c)
		}
		a.drawMessage(c)
		switch a.overlay {
		case "help":
			a.drawHelp(c)
		case "inspect":
			if a.sheet != nil && len(a.sheet.Rect) == 4 {
				y, x := a.sheet.Rect[0], a.sheet.Rect[1]
				for i, row := range a.sheet.Spans {
					c.Put(y+i, x, strings.Repeat(" ", a.sheet.Rect[3]), "")
					c.PutSpans(y+i, x, row, a.sheet.Rect[3])
				}
			}
		}
	}
	a.last = c
	a.term.Write(a.rend.Frame(c))
}

func (a *App) drawGrid(c *Canvas) {
	w, h := c.W, c.H
	g := a.geometry(w, h)
	third := w / 3
	y := 1
	a.drawSystem(c, y, 0, g.top, third)
	a.drawThroughput(c, y, third, g.top, third)
	a.drawPipeline(c, y, 2*third, g.top, w-2*third)
	y += g.top
	if g.inst > 1 {
		half := w / 2
		for _, p := range []struct {
			n, x, w int
		}{{7, 0, half}, {8, half, w - half}} {
			c.Box(y, p.x, g.inst, p.w, a.title(p.n), "dim", a.focus == p.n)
			sub := NewCanvas(p.w-2, g.inst-2)
			if a.view != nil {
				if p.n == 7 {
					DrawScope(sub, a.view.P7, false, a.annotate)
				} else {
					DrawSpectrum(sub, a.view.P8, false, a.annotate)
				}
			}
			c.Blit(sub, y+1, p.x+1)
		}
		y += g.inst
	} else {
		if a.view != nil {
			c.Put(y, 1, clip(a.view.Strip, w-2), "dim")
		}
		y++
	}
	if a.hasPanel(9) {
		a.drawReadscope(c, y, 0, g.mid, third)
		a.drawIndex(c, y, third, g.mid, third)
		a.drawNats(c, y, 2*third, g.mid, w-2*third)
	} else { // no NATS source: readscope and index share the row
		half := w / 2
		a.drawReadscope(c, y, 0, g.mid, half)
		a.drawIndex(c, y, half, g.mid, w-half)
	}
	y += g.mid
	a.drawQueries(c, y, 0, h-1-y, w)
}

func (a *App) drawZoom(c *Canvas, which string) {
	sub := NewCanvas(c.W, c.H-2)
	if a.view != nil {
		switch which {
		case "scope":
			DrawScope(sub, a.view.P7, true, a.annotate)
		case "spectrum":
			DrawSpectrum(sub, a.view.P8, true, a.annotate)
		case "fabric":
			if a.view.Fabric != nil {
				for i, row := range a.view.Fabric.Spans {
					sub.PutSpans(i, 0, row, sub.W)
				}
			}
		}
	}
	c.Blit(sub, 1, 0)
}

func (a *App) drawMessage(c *Canvas) {
	y := c.H - 1
	if a.message != "" && time.Since(a.messageAt) < 6*time.Second {
		c.Put(y, 1, clip(a.message, c.W-2), "amber")
		return
	}
	pages := ""
	if len(a.hello.Pages) > 1 {
		pages = "< > page   "
	}
	zoom := "z zoom 7 8 9"
	if !a.hasPanel(9) {
		zoom = "z zoom 7 8"
	}
	hint := "Tab / 1-9 focus   " + pages + zoom + "   Up/Down select   Enter inspect   P snapshot   ? keys   q quit"
	switch {
	case isPanelPage(a.page().Name):
		n := strconv.Itoa(len(a.page().Panels))
		hint = "Tab / 1-" + n + " focus   " + pages + "p pause   P snapshot   ? keys   q quit"
	case a.page().Name == "nats" && a.zoom == "" && a.focus != 7:
		hint = "Tab / 7 9 focus   " + pages + "z zoom   P snapshot   ? keys   q quit"
	case a.zoom != "":
		hint = "z or Esc: back to the grid   ? keys for this instrument   P snapshot   q quit"
	case a.focus == 7:
		hint = "keys go to the scope: 1-4 channel on/off  Shift+1-4 select  c signal  Up/Down scale  Left/Right time  a autoset   Tab: next panel  ? keys"
	case a.focus == 8:
		hint = "keys go to the spectrum: 1-4 trace  m mode  c source  Up/Down ref level  k peak   Tab / Shift-Tab: next panel   z zoom   ? keys"
	}
	c.Put(y, 1, clip(hint, c.W-2), "dim")
}

// ---------------------------------------------------------------- run

func (a *App) reader() {
	buf := make([]byte, 256)
	for {
		n, err := syscall.Read(a.term.fd, buf)
		if err != nil || n <= 0 {
			// the terminal is gone (read gives end of file, poll says hung up):
			// quit as on SIGHUP, which may never come
			if a.term.HungUp() {
				close(a.hupC)
				return
			}
			// EIO in the background (SIGTTIN ignored), EINTR, or no terminal
			time.Sleep(100 * time.Millisecond)
			continue
		}
		b := append([]byte(nil), buf[:n]...)
		keys, incomplete := ParseKeys(b)
		for incomplete && waitReadable(a.term.fd, 30*time.Millisecond) {
			m, err := syscall.Read(a.term.fd, buf)
			if err != nil || m <= 0 {
				break
			}
			b = append(b, buf[:m]...)
			keys, incomplete = ParseKeys(b)
		}
		if incomplete { // a lone ESC: the Escape key
			keys = append(keys, "escape")
		}
		if len(keys) > 0 {
			a.keysC <- keys
		}
	}
}

func waitReadable(fd int, d time.Duration) bool {
	var set syscall.FdSet
	set.Bits[fd/64] |= 1 << (uint(fd) % 64)
	tv := syscall.NsecToTimeval(d.Nanoseconds())
	n, err := syscall.Select(fd+1, &set, nil, nil, &tv)
	return err == nil && n > 0
}

// Run owns the terminal until the user quits; it returns the exit status.
func (a *App) Run() int {
	signal.Ignore(syscall.SIGTTOU, syscall.SIGTTIN)
	sig := make(chan os.Signal, 8)
	signal.Notify(sig, syscall.SIGWINCH, syscall.SIGINT, syscall.SIGTERM,
		syscall.SIGHUP, syscall.SIGQUIT, syscall.SIGCONT, syscall.SIGTSTP)
	defer a.term.Leave()

	if a.term.Foreground() {
		if err := a.term.Enter(); err != nil {
			fmt.Fprintln(os.Stderr, "tqp-console:", err)
			return 1
		}
	} else {
		a.waitForeground = true
	}
	if a.hello.Zoom != nil && *a.hello.Zoom != "" {
		a.zoom = *a.hello.Zoom
	}
	go a.worker()
	go a.reader()
	a.requestView()
	tick := time.NewTicker(refresh) // the data's pace; keys redraw at once
	defer tick.Stop()
	for {
		select {
		case keys := <-a.keysC:
			for _, k := range keys {
				if code, quit := a.onKey(k); quit {
					return code
				}
			}
			a.draw()
		case apply := <-a.done:
			apply()
			a.draw()
		case <-a.hupC:
			return 129 // as SIGHUP
		case s := <-sig:
			switch s {
			case syscall.SIGWINCH:
				a.rend.Invalidate()
				a.requestView()
			case syscall.SIGCONT:
				a.onContinue()
			case syscall.SIGTSTP:
				a.suspend()
			case syscall.SIGINT:
				return 130
			case syscall.SIGTERM:
				return 143
			case syscall.SIGHUP:
				return 129
			case syscall.SIGQUIT:
				return 131
			}
			a.draw()
		case <-tick.C:
			redraw := false
			if a.waitForeground && a.term.Foreground() {
				a.term.Enter()
				a.rend.Invalidate()
				a.waitForeground, redraw = false, true
			}
			// a message leaving the status line, or the engine falling silent
			// (the header says so), is a change worth a frame; nothing else is
			if a.message != "" && time.Since(a.messageAt) >= 6*time.Second {
				a.message, redraw = "", true
			}
			if st := a.engineState(); st != a.shownState {
				a.shownState, redraw = st, true
			}
			a.measureSelf()
			a.requestView()
			if redraw {
				a.draw()
			}
		}
	}
}
