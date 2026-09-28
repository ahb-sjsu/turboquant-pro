// TurboQuant Pro console: terminal client.
// Copyright (c) 2026 Andrew H. Bond. MIT License.

// Command tqp-console draws the TurboQuant console in a terminal. `tqp console`
// starts it: the Python engine runs the session and serves the view model on a
// Unix socket; this client owns the terminal, draws, and reads keys.
//
//	tqp-console --socket PATH --engine-pid PID [--lifeline-fd FD] [--export-dir DIR]
//
// The lifeline descriptor is the write end of the engine's standard input. This
// process only holds it: when this process ends, however it ends, the engine
// sees end of file and exits too.
package main

import (
	"flag"
	"fmt"
	"os"
	"runtime/debug"
	"time"
)

func main() {
	sock := flag.String("socket", "", "the engine's Unix socket")
	enginePid := flag.Int("engine-pid", 0, "the engine's process id")
	flag.Int("lifeline-fd", -1, "held open for the engine's lifetime (not used otherwise)")
	exportDir := flag.String("export-dir", ".", "where P, e and S write files")
	flag.Parse()
	if *sock == "" {
		fmt.Fprintln(os.Stderr, "tqp-console: started by `tqp console`; --socket is required")
		os.Exit(2)
	}
	os.Exit(run(*sock, *enginePid, *exportDir))
}

func run(sock string, enginePid int, exportDir string) (code int) {
	t, err := OpenTerm()
	if err != nil {
		fmt.Fprintln(os.Stderr, "tqp-console: not a terminal:", err)
		return 2
	}
	defer func() { // any panic: the terminal first, then the report
		if r := recover(); r != nil {
			t.Leave()
			fmt.Fprintf(os.Stderr, "tqp-console: internal error: %v\n%s", r, debug.Stack())
			code = 70
		}
	}()
	eng := NewEngine(sock)
	defer eng.Close()
	app := NewApp(t, eng, enginePid, exportDir)
	deadline := time.Now().Add(10 * time.Second)
	for {
		if err := eng.Call(map[string]any{"op": "hello"}, &app.hello, 2*time.Second); err == nil {
			break
		} else if time.Now().After(deadline) {
			fmt.Fprintln(os.Stderr, "tqp-console: the engine does not answer:", err)
			return 2
		}
		time.Sleep(200 * time.Millisecond)
	}
	if app.hello.Engine.Pid != 0 {
		app.enginePid = app.hello.Engine.Pid
	}
	return app.Run()
}
