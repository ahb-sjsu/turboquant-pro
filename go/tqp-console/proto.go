// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

package main

// The engine connection: one JSON object per line each way over a Unix socket.
// Calls have a deadline; a call that misses it (a paused or busy engine) drops
// the connection so that its late answer cannot be read as the next one's, and
// the next call reconnects.

import (
	"bufio"
	"encoding/json"
	"errors"
	"net"
	"sync"
	"time"
)

type Engine struct {
	path string
	mu   sync.Mutex
	conn net.Conn
	r    *bufio.Reader
}

func NewEngine(path string) *Engine { return &Engine{path: path} }

func (e *Engine) Call(req map[string]any, out any, timeout time.Duration) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.conn == nil {
		c, err := net.DialTimeout("unix", e.path, timeout)
		if err != nil {
			return err
		}
		e.conn, e.r = c, bufio.NewReaderSize(c, 1<<20)
	}
	e.conn.SetDeadline(time.Now().Add(timeout))
	b, _ := json.Marshal(req)
	if _, err := e.conn.Write(append(b, '\n')); err != nil {
		e.drop()
		return err
	}
	line, err := e.r.ReadBytes('\n')
	if err != nil {
		e.drop()
		return err
	}
	var probe struct{ Error string }
	if json.Unmarshal(line, &probe) == nil && probe.Error != "" {
		return errors.New(probe.Error)
	}
	return json.Unmarshal(line, out)
}

func (e *Engine) drop() {
	if e.conn != nil {
		e.conn.Close()
		e.conn = nil
	}
}

func (e *Engine) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.drop()
}
