// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

package main

// The terminal, handled the way btop handles it.
//
// Only three modes are ever set: the alternate screen (?1049), a hidden cursor
// (?25) and no autowrap (?7). No mouse reporting, no extended keyboard protocol,
// no bracketed paste, no focus events: nothing the shell would be left holding
// if this process were killed outright. Leave() undoes exactly what Enter() did
// and is safe to call any number of times.

import (
	"os"
	"sync"
	"syscall"
	"unsafe"
)

const (
	seqEnter = "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[H\x1b[2J"
	seqLeave = "\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l"
)

type Term struct {
	fd      int
	out     *os.File
	saved   syscall.Termios
	entered bool
	mu      sync.Mutex
}

func ioctl(fd int, req uintptr, arg unsafe.Pointer) error {
	_, _, e := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), req, uintptr(arg))
	if e != 0 {
		return e
	}
	return nil
}

// OpenTerm takes stdin/stdout, which must be a terminal, and records its
// settings so that Leave() can always put them back.
func OpenTerm() (*Term, error) {
	t := &Term{fd: int(os.Stdin.Fd()), out: os.Stdout}
	if err := ioctl(t.fd, syscall.TCGETS, unsafe.Pointer(&t.saved)); err != nil {
		return nil, err
	}
	return t, nil
}

// Size is the terminal's columns and rows (80x24 if it will not say).
func (t *Term) Size() (int, int) {
	var ws struct{ Row, Col, X, Y uint16 }
	if err := ioctl(int(t.out.Fd()), syscall.TIOCGWINSZ, unsafe.Pointer(&ws)); err != nil || ws.Col == 0 {
		return 80, 24
	}
	return int(ws.Col), int(ws.Row)
}

// Foreground is true when this process's group owns the terminal. Drawing and
// reading happen only then: a background job leaves the terminal to the shell.
func (t *Term) Foreground() bool {
	var pgrp int32
	if err := ioctl(t.fd, syscall.TIOCGPGRP, unsafe.Pointer(&pgrp)); err != nil {
		// ENOTTY: a terminal that is not our controlling terminal, so there is
		// no job control to defer to: draw. Any other error: we cannot tell,
		// so leave the terminal alone.
		return err == syscall.ENOTTY
	}
	return int(pgrp) == syscall.Getpgrp()
}

// HungUp is true when the terminal has gone away: its window closed, or the
// other end of its pseudo-terminal was closed. A hung-up terminal reports
// POLLHUP, and a live one never does, in the foreground or the background.
func (t *Term) HungUp() bool {
	pfd := struct {
		fd              int32
		events, revents int16
	}{fd: int32(t.fd)}
	var ts syscall.Timespec // zero: do not wait
	n, _, e := syscall.Syscall6(syscall.SYS_PPOLL, uintptr(unsafe.Pointer(&pfd)), 1,
		uintptr(unsafe.Pointer(&ts)), 0, 0, 0)
	return e == 0 && n == 1 && pfd.revents&pollHUP != 0
}

const pollHUP = 0x10 // POLLHUP, the same value on every Linux architecture

// Enter switches to raw input and the alternate screen.
func (t *Term) Enter() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	raw := t.saved
	raw.Iflag &^= syscall.IGNBRK | syscall.BRKINT | syscall.PARMRK | syscall.ISTRIP |
		syscall.INLCR | syscall.IGNCR | syscall.ICRNL | syscall.IXON
	raw.Oflag &^= syscall.OPOST
	raw.Lflag &^= syscall.ECHO | syscall.ECHONL | syscall.ICANON | syscall.ISIG | syscall.IEXTEN
	raw.Cflag &^= syscall.CSIZE | syscall.PARENB
	raw.Cflag |= syscall.CS8
	raw.Cc[syscall.VMIN] = 1
	raw.Cc[syscall.VTIME] = 0
	if err := ioctl(t.fd, syscall.TCSETS, unsafe.Pointer(&raw)); err != nil {
		return err
	}
	t.out.WriteString(seqEnter)
	t.entered = true
	return nil
}

// Leave restores the screen and the terminal settings recorded at start.
func (t *Term) Leave() {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.entered {
		return
	}
	t.out.WriteString(seqLeave)
	ioctl(t.fd, syscall.TCSETS, unsafe.Pointer(&t.saved))
	t.entered = false
}

// Entered reports whether the screen is currently ours.
func (t *Term) Entered() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.entered
}

// Write sends frame bytes, only while the screen is ours.
func (t *Term) Write(b []byte) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.entered {
		t.out.Write(b)
	}
}
