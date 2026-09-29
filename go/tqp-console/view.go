// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

package main

// The engine's view model (turboquant_pro/console/viewmodel.py), decoded.
// Tuples arrive as JSON arrays; each has a small decoder.

import (
	"encoding/json"
	"fmt"
)

func tuple(data []byte, n int) ([]json.RawMessage, error) {
	var a []json.RawMessage
	if err := json.Unmarshal(data, &a); err != nil {
		return nil, err
	}
	if len(a) < n {
		return nil, fmt.Errorf("tuple of %d, want %d", len(a), n)
	}
	return a, nil
}

func str(m json.RawMessage) string {
	var s *string
	json.Unmarshal(m, &s)
	if s == nil {
		return ""
	}
	return *s
}

// Span is [text, role|null].
type Span struct {
	Text, Role string
}

func (s *Span) UnmarshalJSON(b []byte) error {
	a, err := tuple(b, 2)
	if err != nil {
		return err
	}
	s.Text, s.Role = str(a[0]), str(a[1])
	return nil
}

// Tick is [position, text]: a division, a fraction or a dB value, and its label.
type Tick struct {
	At   float64
	Text string
}

func (t *Tick) UnmarshalJSON(b []byte) error {
	a, err := tuple(b, 2)
	if err != nil {
		return err
	}
	json.Unmarshal(a[0], &t.At)
	t.Text = str(a[1])
	return nil
}

type SysItem struct {
	Label, Value, Kind string
	Dim                bool
}

func (s *SysItem) UnmarshalJSON(b []byte) error {
	a, err := tuple(b, 4)
	if err != nil {
		return err
	}
	s.Label, s.Value, s.Kind = str(a[0]), str(a[1]), str(a[2])
	json.Unmarshal(a[3], &s.Dim)
	return nil
}

type Series struct {
	Name, Unit, Now, Role string
	Digits                int
	Values                []*float64
}

type Stage struct {
	Name string
	Ms   *float64
	Text string
}

func (s *Stage) UnmarshalJSON(b []byte) error {
	a, err := tuple(b, 3)
	if err != nil {
		return err
	}
	s.Name, s.Text = str(a[0]), str(a[2])
	json.Unmarshal(a[1], &s.Ms)
	return nil
}

type KV struct{ K, V, Role string }

func (r *KV) UnmarshalJSON(b []byte) error {
	a, err := tuple(b, 2)
	if err != nil {
		return err
	}
	r.K, r.V = str(a[0]), str(a[1])
	if len(a) > 2 {
		r.Role = str(a[2])
	}
	return nil
}

type QCol struct {
	Name  string
	Width int
	Right bool
}

func (q *QCol) UnmarshalJSON(b []byte) error {
	a, err := tuple(b, 3)
	if err != nil {
		return err
	}
	q.Name = str(a[0])
	json.Unmarshal(a[1], &q.Width)
	json.Unmarshal(a[2], &q.Right)
	return nil
}

type NatsRow struct {
	Label, Value, Kind string
	Values             []*float64
	Max                string
	HasMax             bool
}

func (r *NatsRow) UnmarshalJSON(b []byte) error {
	a, err := tuple(b, 5)
	if err != nil {
		return err
	}
	r.Label, r.Value, r.Kind = str(a[0]), str(a[1]), str(a[2])
	json.Unmarshal(a[3], &r.Values)
	var m *string
	json.Unmarshal(a[4], &m)
	if m != nil {
		r.Max, r.HasMax = *m, true
	}
	return nil
}

type Channel struct {
	N         int
	Signal    string
	Role      string
	On        bool
	Cols      [][]float64 // per sub-column [lo_div, hi_div], or null
	GroundDiv float64     `json:"ground_div"`
	Persist   [][3]int    // [row from top, col, level 1|2]
	Ref       *Tick       // [div, label]
}

// Axis is one channel's y scale: its value at each division 0..VDIV.
type Axis struct {
	N     int
	Role  string
	Unit  string
	Ticks []Tick
}

type Level struct {
	Div  float64
	Role string
}

type FFTView struct {
	Head    Span
	Role    string
	Fracs   []float64
	Caption Span
	Message *Span
}

type ScopeView struct {
	Status       []Span
	Gw, Gh       int
	Vdiv, Hdiv   int
	Channels     []Channel
	Sel          int
	TriggerLevel *Level   `json:"trigger_level"`
	TriggerX     *float64 `json:"trigger_x"`
	Axes         []Axis   // one y axis per enabled channel, the selected first
	Xticks       []Tick
	Legend       []Span
	Notes        []Span
	Side         []Span
	Meas         []Span
	Softkeys     []string
	Fft          *FFTView
}

type Trace struct {
	N    int
	Role string
	Mode string
	Cols []*float64
}

type Marker struct {
	Frac, Db float64
	Glyph    string
}

func (m *Marker) UnmarshalJSON(b []byte) error {
	a, err := tuple(b, 3)
	if err != nil {
		return err
	}
	json.Unmarshal(a[0], &m.Frac)
	json.Unmarshal(a[1], &m.Db)
	m.Glyph = str(a[2])
	return nil
}

type Waterfall struct {
	Label  string
	Levels [][]int
}

type SpecView struct {
	Status     []Span
	Gw, Gh     int
	Vdiv, Hdiv int
	Reason     *Span
	Ok         bool
	Bottom     float64
	RefDb      float64 `json:"ref_db"`
	DbDiv      float64 `json:"db_div"`
	Delta      bool
	LimitDb    *float64 `json:"limit_db"`
	Sel        int
	Traces     []Trace
	Markers    []Marker
	Yticks     []Tick
	Xticks     []Tick
	Legend     []Span
	Notes      []Span
	Readout    Span
	Names      string
	Softkeys   []string
	Waterfall  *Waterfall
}

type NatsView struct {
	State      string
	Message    string
	Summary    Span
	TitleExtra string `json:"title_extra"`
	Rows       []NatsRow
}

type View struct {
	Header struct {
		Labels []Span
		Stamp  string
	}
	Engine struct {
		Pid    int
		Cpu    *float64
		Blas   string
		Paused bool
	}
	Message string
	Strip   string
	P1      *struct{ Items []SysItem }
	P2      *struct {
		Series []Series
		Note   string
	}
	P3 *struct {
		Stages []Stage
		Note   string
	}
	P4 *struct{ Rows []KV }
	P5 *struct {
		Rows  []KV
		Empty string
	}
	P6 *struct {
		Cols []QCol
		Rows [][]string
		Ids  []string
	}
	P7     *ScopeView
	P8     *SpecView
	P9     *NatsView
	Fabric *struct {
		Spans [][]Span
		Need  int // rows the fabric needs to list every client
	}
	Panels map[string]*PanelView // a page of PanelViews: machine, dht
}

type Hello struct {
	Protocol     int
	Pages        []Page
	Keys         [][2]string
	KeysMachine  [][2]string `json:"keys_machine"`
	KeysScope    [][2]string `json:"keys_scope"`
	KeysSpectrum [][2]string `json:"keys_spectrum"`
	Zoomable     map[string]string
	Titles       map[string]string
	HelpNote     string `json:"help_note"`
	Zoom         *string
	Engine       struct{ Pid int }
}

type Sheet struct {
	Rect    []int
	Spans   [][]Span
	Message string
}

type KeyReply struct {
	Message string
	Inspect string
	Error   string
}
