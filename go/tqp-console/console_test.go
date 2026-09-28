// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

package main

import (
	"encoding/json"
	"strconv"
	"strings"
	"testing"
)

func TestTheTerminalModesAreOnlyTheThreeWeUndo(t *testing.T) {
	for _, seq := range []string{seqEnter, seqLeave} {
		for _, bad := range []string{"?1000", "?1002", "?1003", "?1006", "?1015", "?2004", "?1004", ">"} {
			if strings.Contains(seq, bad) {
				t.Fatalf("%q sets %s, which a killed client would leave behind", seq, bad)
			}
		}
	}
	for _, mode := range []string{"?1049", "?25", "?7"} {
		if !strings.Contains(seqEnter, mode) || !strings.Contains(seqLeave, mode) {
			t.Fatalf("mode %s is not both set and undone", mode)
		}
	}
}

func TestKeysFromRawBytes(t *testing.T) {
	cases := map[string][]string{
		"q":            {"q"},
		"\x1b[A\x1b[B": {"up", "down"},
		"\x1bOC":       {"right"},
		"\r\t ":        {"enter", "tab", "space"},
		"\x03\x1a":     {"ctrl+c", "ctrl+z"},
		"\x1bq":        {"escape", "q"},
		"P?7":          {"P", "?", "7"},
	}
	for in, want := range cases {
		got, inc := ParseKeys([]byte(in))
		if inc || strings.Join(got, ",") != strings.Join(want, ",") {
			t.Errorf("%q: got %v (incomplete %v), want %v", in, got, inc, want)
		}
	}
	for _, partial := range []string{"\x1b", "\x1b[", "\x1b[1;"} {
		if _, inc := ParseKeys([]byte(partial)); !inc {
			t.Errorf("%q should wait for the rest of the sequence", partial)
		}
	}
}

func TestFramesSendOnlyChangedRows(t *testing.T) {
	var r Renderer
	c := NewCanvas(10, 3)
	c.Put(0, 0, "hello", "cyan")
	first := string(r.Frame(c))
	if !strings.Contains(first, "\x1b[2J") || !strings.Contains(first, "hello") {
		t.Fatalf("the first frame is a full one: %q", first)
	}
	c2 := NewCanvas(10, 3)
	c2.Put(0, 0, "hello", "cyan")
	c2.Put(2, 0, "bye", "")
	second := string(r.Frame(c2))
	if strings.Contains(second, "hello") || !strings.Contains(second, "\x1b[3;1H") {
		t.Fatalf("only row 3 changed: %q", second)
	}
	if SGR("purple_dim") != "\x1b[0;35;2m" || SGR("nope") != "\x1b[0m" {
		t.Fatal("roles map to SGR")
	}
}

func TestSparkMatchesThePythonPanels(t *testing.T) {
	f := func(v float64) *float64 { return &v }
	if got := Spark([]*float64{f(1), f(2), f(4), f(8)}, 4); got != "▁▂▄█" {
		t.Fatalf("spark = %q", got)
	}
	if got := Spark([]*float64{f(1)}, 12); got != "collecting  " {
		t.Fatalf("spark with one point = %q", got)
	}
}

// A one-channel scope view as the engine sends it: a flat trace at 6 divisions
// of 8, on a 40 x 8 graticule. The braille must sit at the row that height maps
// to, and the calibration must be printed at the left.
func TestTheScopeDrawsTracesAndItsScaleWhereTheDivisionsSay(t *testing.T) {
	cols := make([][]float64, 80)
	for i := range cols {
		cols[i] = []float64{6, 6}
	}
	raw, _ := json.Marshal(map[string]any{
		"status": [][]any{{" RUN ", "green"}},
		"gw":     40, "gh": 8, "vdiv": 8, "hdiv": 10,
		"channels": []map[string]any{{
			"n": 1, "signal": "latency", "role": "yellow", "on": true,
			"cols": cols, "ground_div": 4.0, "persist": [][]int{},
		}},
		"sel": 1,
		"axes": []map[string]any{{"n": 1, "role": "yellow", "unit": "ms",
			"ticks": [][]any{{0, "-40"}, {1, "-30"}, {2, "-20"}, {3, "-10"}, {4, "0"}, {5, "10"}, {6, "20"}, {7, "30"}, {8, "40"}}}},
		"xticks": [][]any{{0.0, "-10"}, {1.0, "0 s"}},
		"legend": [][]any{{" 1 latency 10 ms/div ", "yellow"}},
		"notes":  [][]any{},
	})
	var v ScopeView
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatal(err)
	}
	c := NewCanvas(40+2+yl, 8+4)
	DrawScope(c, &v, false, true)
	lines := c.Lines()
	if !strings.Contains(lines[0], "RUN") || !strings.Contains(lines[1], "ms") {
		t.Fatalf("status and unit: %q / %q", lines[0], lines[1])
	}
	// 6 of 8 divisions on 8 rows: sub-row int(6/8*32) = 24 -> row 8-1-6 = 1
	row := lines[1+1+1]
	if !strings.ContainsRune(row, rune(braille|dotBit[0][3]|dotBit[1][3])) {
		t.Fatalf("trace not on the row its divisions map to: %q", row)
	}
	// rows: status, box top, 8 interior, box bottom, then the time axis
	if !strings.Contains(lines[len(lines)-1], "0 s") {
		t.Fatalf("time axis: %q", lines[len(lines)-1])
	}
	if !strings.Contains(strings.Join(lines, "\n"), "1 latency 10 ms/div") {
		t.Fatal("the legend names the channel with its scale")
	}
}

func TestTheViewDecodesTuples(t *testing.T) {
	var v View
	raw := `{"header":{"labels":[["[live]","green"]],"stamp":"2026-01-01 00:00:00Z"},
	 "p1":{"items":[["QPS","20.0 q/s","meas",false]]},
	 "p9":{"state":"ok","summary":["NATS 1 leaf","cyan"],"rows":[["leaf rtt","90.2 ms","samp",[1.0,null,2.0],"90.2"]]}}`
	if err := json.Unmarshal([]byte(raw), &v); err != nil {
		t.Fatal(err)
	}
	if v.P1.Items[0].Value != "20.0 q/s" || v.P9.Rows[0].Values[1] != nil || v.P9.Rows[0].Max != "90.2" {
		t.Fatalf("decoded %+v", v)
	}
}

// The focused instrument takes its keys in the grid; Tab still moves focus; on
// any other panel the digits choose a panel.
func TestKeysGoToTheFocusedInstrument(t *testing.T) {
	a := NewApp(nil, nil, 0, ".")
	queued := func() int { return len(a.jobs) }
	a.focus = 7
	a.onKey("2")
	if a.focus != 7 || queued() != 1 {
		t.Fatalf("on the scope, 2 is a channel: focus %d, %d engine calls", a.focus, queued())
	}
	a.onKey("tab")
	if a.focus != 8 || queued() != 1 {
		t.Fatalf("Tab moves on from an instrument: focus %d", a.focus)
	}
	a.onKey("up")
	if queued() != 2 {
		t.Fatal("on the spectrum, Up is its reference level")
	}
	a.onKey("tab")
	a.onKey("3")
	if a.focus != 3 || queued() != 2 {
		t.Fatalf("on panel 9, 3 chooses panel 3: focus %d", a.focus)
	}
	if a.instrument() != "" {
		t.Fatal("panel 3 is not an instrument")
	}
	a.zoom = "fabric"
	a.focus = 7
	if a.instrument() != "" {
		t.Fatal("zoomed on the fabric, no instrument takes keys")
	}
}

// The scope's channel checkboxes: all four stay visible, shortened if need be.
func TestChannelCheckboxesAllStayVisible(t *testing.T) {
	items := []Span{
		{" [x] 1 latency 20 ms/div ", "yellow"},
		{" [ ] 2 scan ", "dim"},
		{" [x] 3 agree 0.2 ratio/div ", "magenta"},
		{" [ ] 4 move ", "dim"},
	}
	for _, gw := range []int{90, 50, 26} {
		c := NewCanvas(gw+2+yl, 3)
		legend(c, items, 0, yl, gw)
		line := c.Lines()[0]
		for _, want := range []string{"[x]", "1", "[ ]", "2", "3", "4"} {
			if !strings.Contains(line, want) {
				t.Fatalf("gw %d: %q lacks %q", gw, line, want)
			}
		}
		if strings.Count(line, "[x]") != 2 || strings.Count(line, "[ ]") != 2 {
			t.Fatalf("gw %d: every channel keeps its box: %q", gw, line)
		}
	}
	if got := shorten(" [ ] 2 scan ", 2); got != "[ ]2" {
		t.Fatalf("shortest form %q", got)
	}
}

// Each enabled channel gets its own y axis, in its colour, with its unit on top;
// the selected channel's axis sits next to the graticule.
func TestEveryEnabledChannelHasItsOwnAxis(t *testing.T) {
	ticks := func(scale float64) [][]any {
		var out [][]any
		for k := 0; k <= 8; k++ {
			out = append(out, []any{k, strconv.FormatFloat((float64(k)-4)*scale, 'g', 4, 64)})
		}
		return out
	}
	gw, naxes := 40, 2
	raw, _ := json.Marshal(map[string]any{
		"gw": gw, "gh": 8, "vdiv": 8, "hdiv": 10, "sel": 1,
		"axes": []map[string]any{
			{"n": 1, "role": "yellow", "unit": "ms", "ticks": ticks(20)},
			{"n": 3, "role": "magenta", "unit": "ratio", "ticks": ticks(0.25)},
		},
	})
	var v ScopeView
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatal(err)
	}
	c := NewCanvas(gw+2+yl*naxes, 8+4)
	DrawScope(c, &v, false, true)
	head := c.Lines()[1]
	if !strings.Contains(head, "ratio") || !strings.Contains(head, "ms") {
		t.Fatalf("both units on the axis header row: %q", head)
	}
	if strings.Index(head, "ratio") > strings.Index(head, "ms") {
		t.Fatalf("the selected channel's axis is next to the graticule: %q", head)
	}
	if c.Get(2, yl*2-2).Role != "yellow" || c.Get(2, yl-2).Role != "magenta" {
		t.Fatalf("each axis in its channel's colour: %q", c.Lines()[2])
	}
	foot := c.Lines()[1+8+1]
	if strings.Index(foot, "CH3") > strings.Index(foot, "CH1") || strings.Index(foot, "CH3") < 0 {
		t.Fatalf("each axis names its channel below it: %q", foot)
	}
	if ScopeAxes(38, false, 4) != 2 || ScopeAxes(200, true, 4) != 4 || ScopeAxes(10, false, 3) != 1 {
		t.Fatal("as many axes as leave a 20-column graticule, at least one")
	}
}
