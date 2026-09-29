// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

package main

import (
	"encoding/json"
	"strings"
	"testing"
)

func appWithPages(pages ...Page) *App {
	a := NewApp(nil, nil, 0, ".")
	a.hello.Pages = pages
	a.focus = defaultFocus(a.page())
	return a
}

var (
	indexPage   = Page{Name: "index", Panels: []int{1, 2, 3, 4, 5, 6, 7, 8}, Titles: map[string]string{"4": "4 readscope", "5": "5 index", "9": "9 NATS fabric"}}
	machinePage = Page{Name: "machine", Panels: []int{1, 2, 3, 4, 5, 6}, Titles: map[string]string{"1": "1 CPU", "4": "4 disks"}}
)

// Tab and the digits move only among the current page's panels, and each page
// keeps its own focus across < and >.
func TestFocusStaysOnTheCurrentPagesPanels(t *testing.T) {
	a := appWithPages(indexPage, machinePage)
	if a.focus != 6 {
		t.Fatalf("the index page opens on the query stream, got %d", a.focus)
	}
	a.onKey("9") // no NATS source: there is no panel 9
	if a.focus != 6 {
		t.Fatalf("a digit for an absent panel moved the focus to %d", a.focus)
	}
	a.focus = 8
	a.onKey("tab") // from 8 the next panel of this page is 1, not 9
	if a.focus != 1 {
		t.Fatalf("Tab from 8 on a page without 9 went to %d", a.focus)
	}
	a.onKey("backtab")
	if a.focus != 8 {
		t.Fatalf("Shift-Tab from 1 went to %d", a.focus)
	}
	a.focus = 5
	a.onKey(">")
	if a.page().Name != "machine" || a.focus != 1 {
		t.Fatalf("> opens the machine page on its first panel: %s %d", a.page().Name, a.focus)
	}
	a.onKey("4")
	a.onKey("<")
	if a.page().Name != "index" || a.focus != 5 {
		t.Fatalf("< returns to the index page and its focus: %s %d", a.page().Name, a.focus)
	}
	a.onKey(">")
	if a.focus != 4 {
		t.Fatalf("the machine page kept its focus: %d", a.focus)
	}
	a.onKey(">") // wraps
	if a.page().Name != "index" {
		t.Fatal("> wraps round the pages")
	}
}

func TestInstrumentsTakeKeysOnlyOnTheIndexPage(t *testing.T) {
	a := appWithPages(machinePage, indexPage)
	a.focus = 1
	if a.instrument() != "" {
		t.Fatal("no instrument on the machine page")
	}
	a.onKey("z")
	if a.zoom != "" || !strings.Contains(a.message, "no panel on this page zooms") {
		t.Fatalf("z on the machine page: zoom %q, message %q", a.zoom, a.message)
	}
	one := appWithPages(machinePage)
	one.onKey(">")
	if !strings.Contains(one.message, "one page") {
		t.Fatalf("> with one page says so: %q", one.message)
	}
}

// Without a NATS source the index grid draws no panel 9, and readscope and
// index share its row.
func TestTheGridDrawsOnlyTheAttachedPanels(t *testing.T) {
	a := appWithPages(indexPage)
	c := NewCanvas(160, 48)
	a.drawGrid(c)
	screen := strings.Join(c.Lines(), "\n")
	if strings.Contains(screen, "9 NATS") {
		t.Fatal("panel 9 drawn with no NATS source")
	}
	if !strings.Contains(screen, "4 readscope") || !strings.Contains(screen, "5 index") {
		t.Fatal("readscope and index missing")
	}
	withNats := appWithPages(Page{Name: "index", Panels: []int{1, 2, 3, 4, 5, 6, 7, 8, 9}, Titles: indexPage.Titles})
	c = NewCanvas(160, 48)
	withNats.drawGrid(c)
	if !strings.Contains(strings.Join(c.Lines(), "\n"), "9 NATS") {
		t.Fatal("panel 9 missing with a NATS source")
	}
}

func decodePanel(t *testing.T, raw string) *PanelView {
	t.Helper()
	var pv PanelView
	if err := json.Unmarshal([]byte(raw), &pv); err != nil {
		t.Fatal(err)
	}
	return &pv
}

// Every panel shape fits its box at every size: nothing is written outside it,
// and a table that does not fit says how many rows it left out.
func TestAPanelViewFitsItsBox(t *testing.T) {
	rows := `[["pkg 0","37.5 %","deri",[0.1,null,0.4],"iowait 0.2 %","amber"],["all","-","deri",null,"",""]]`
	pv := decodePanel(t, `{"state":"ok","summary":["48 CPUs in 2 packages","cyan"],"title_extra":"rates over 2.0 s polls",
	 "rows":`+rows+`,"strips":[["pkg 0",[0.1,0.9,null,0.5]]],
	 "table":{"cols":[["device",9,false],["read",10,true]],"rows":[["sda","1.2 MB/s"],["sdb","-"],["sdc","-"],["sdd","-"]],"roles":["amber","","",""]},
	 "note":"a note"}`)
	for _, sz := range [][2]int{{20, 3}, {30, 6}, {60, 9}, {100, 20}} {
		c := NewCanvas(sz[0]+4, sz[1]+2)
		drawPanelView(c, 1, 2, sz[1], sz[0], "1 CPU", false, pv)
		lines := c.Lines()
		for y, l := range lines {
			for x, r := range []rune(l) {
				outside := y < 1 || y > sz[1] || x < 2 || x >= 2+sz[0]
				if outside && r != ' ' {
					t.Fatalf("size %v: %q written outside the box at (%d, %d)", sz, r, y, x)
				}
			}
		}
	}
	c := NewCanvas(60, 9)
	drawPanelView(c, 0, 0, 9, 60, "4 disks", false, pv)
	screen := strings.Join(c.Lines(), "\n")
	if !strings.Contains(screen, "more (a taller terminal shows them)") {
		t.Fatalf("a table cut short says so:\n%s", screen)
	}
	if !strings.Contains(screen, "48 CPUs") || !strings.Contains(screen, "37.5 %") {
		t.Fatalf("summary and rows:\n%s", screen)
	}
	none := decodePanel(t, `{"state":"none","message":"unavailable: no NVML"}`)
	c = NewCanvas(40, 5)
	drawPanelView(c, 0, 0, 5, 40, "6 GPU", false, none)
	if !strings.Contains(strings.Join(c.Lines(), "\n"), "unavailable: no NVML") {
		t.Fatal("a panel with no data says why")
	}
}

func TestACellIsAHeightAndUnknownIsADot(t *testing.T) {
	zero, half, full := 0.0, 0.5, 1.0
	if cell(nil) != '·' || cell(&zero) != '▁' || cell(&half) != '▄' || cell(&full) != '█' {
		t.Fatalf("cells: %c %c %c %c", cell(nil), cell(&zero), cell(&half), cell(&full))
	}
}

func TestTheMachinePageDrawsAtEverySize(t *testing.T) {
	a := appWithPages(machinePage)
	a.view = &View{Machine: map[string]*PanelView{
		"1": decodePanel(t, `{"state":"ok","rows":[["all","12.0 %","deri",[0.1,0.2],"",""]]}`),
		"6": decodePanel(t, `{"state":"none","message":"no GPU"}`),
	}}
	for _, sz := range [][2]int{{80, 24}, {120, 40}, {160, 48}, {220, 60}} {
		c := NewCanvas(sz[0], sz[1])
		a.drawMachine(c)
		screen := strings.Join(c.Lines(), "\n")
		if !strings.Contains(screen, "1 CPU") || !strings.Contains(screen, "no GPU") {
			t.Fatalf("%v:\n%s", sz, screen)
		}
	}
}

var natsPage = Page{Name: "nats", Panels: []int{7, 9}, Titles: map[string]string{"7": "7 scope  NATS signals", "9": "9 NATS fabric"}}

// On the NATS page the scope (7) takes its keys and z opens it or the fabric
// (9); the engine is told the page, so the NATS scope answers, not the index's.
func TestTheNatsPageHasAScopeThatTakesItsKeys(t *testing.T) {
	a := appWithPages(indexPage, natsPage)
	a.hello.Zoomable = map[string]string{"7": "scope", "8": "spectrum", "9": "fabric"}
	a.onKey(">")
	if a.page().Name != "nats" || a.focus != 7 || a.instrument() != "scope" {
		t.Fatalf("the NATS page opens on its scope: %s %d %q", a.page().Name, a.focus, a.instrument())
	}
	before := len(a.jobs)
	a.onKey("2")
	job := len(a.jobs) - before
	if job != 1 {
		t.Fatalf("2 on the NATS scope is a channel key for the engine: %d calls", job)
	}
	a.onKey("tab")
	if a.focus != 9 || a.instrument() != "" {
		t.Fatalf("Tab moves to the fabric: %d %q", a.focus, a.instrument())
	}
	a.onKey("z")
	if a.zoom != "fabric" {
		t.Fatalf("z on 9 opens the fabric full screen: %q", a.zoom)
	}
	a.onKey("z")
	a.onKey("8") // there is no panel 8 on this page
	if a.focus != 9 {
		t.Fatalf("a digit for an absent panel moved the focus to %d", a.focus)
	}
	c := NewCanvas(160, 48)
	a.drawNatsPage(c)
	screen := strings.Join(c.Lines(), "\n")
	if !strings.Contains(screen, "7 scope") || !strings.Contains(screen, "9 NATS fabric") {
		t.Fatalf("both panels drawn:\n%s", screen)
	}
}
