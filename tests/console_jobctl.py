"""End-to-end job-control checks of `tqp console` under an interactive bash.

Run as a script (on a Linux machine with the Go client built):

    python tests/console_jobctl.py

and imported by tests/test_console_terminal.py. Each case starts the console as
a job of `bash -i` in a pseudo-terminal, does something to it (keys, job
control, signals from outside), and then checks, from the bytes the programs
wrote and from the process table, that:

* the terminal is back to normal: alternate screen off, cursor visible,
  autowrap on;
* no mode was ever turned on that a shell would be left holding (mouse
  reporting, an extended keyboard protocol, bracketed paste, focus events);
* the shell reads typed input again (a command typed afterwards runs);
* no console client or engine process is left over.
"""

from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import signal
import struct
import sys
import termios
import time

FORBIDDEN = {
    b"?1000h": "mouse reporting",
    b"?1002h": "mouse reporting",
    b"?1003h": "mouse reporting",
    b"?1006h": "mouse reporting (SGR)",
    b"?1015h": "mouse reporting (urxvt)",
    b"?2004h": "bracketed paste",
    b"?1004h": "focus events",
}
KITTY = re.compile(rb"\x1b\[>[0-9;]*u")  # extended keyboard protocol push


def terminal_state(out: bytes) -> dict:
    """The terminal modes after ``out``, plus any forbidden mode ever set."""
    st = {"alt": False, "cursor_hidden": False, "autowrap_off": False, "bad": []}
    for m in re.finditer(rb"\x1b\[(\?[0-9;]+)([hl])", out):
        for code in m.group(1)[1:].split(b";"):
            on = m.group(2) == b"h"
            key = b"?" + code + m.group(2)
            if key in FORBIDDEN:
                st["bad"].append(FORBIDDEN[key])
            if code == b"1049":
                st["alt"] = on
            elif code == b"25":
                st["cursor_hidden"] = not on
            elif code == b"7":
                st["autowrap_off"] = not on
    if KITTY.search(out):
        st["bad"].append("extended keyboard protocol")
    return st


def _controlling_tty():
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


class Shell:
    """``bash -i`` on a new pseudo-terminal that is its controlling terminal.
    Started with subprocess (fork + exec in C), not pty.fork(), which is unsafe
    in a multi-threaded process such as a test run."""

    def __init__(self, cwd: str, env: dict):
        import subprocess

        self.fd, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 48, 160, 0, 0))
        self.proc = subprocess.Popen(
            ["bash", "--norc", "--noprofile", "-i"],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            cwd=cwd,
            env=dict(os.environ, **env, TERM="xterm-256color", PS1="PROMPT$ "),
            start_new_session=True,
            preexec_fn=_controlling_tty,
            close_fds=True,
        )
        os.close(slave)
        self.pid = self.proc.pid
        self.out = b""
        self.pump(1.0)
        # readline turns bracketed paste on at every prompt; off here, any such
        # mode in the output can only be the console's
        self.send(b"bind 'set enable-bracketed-paste off'\r", 0.5)

    def pump(self, t: float) -> bytes:
        got, end = b"", time.time() + t
        while time.time() < end:
            r, _, _ = select.select([self.fd], [], [], 0.05)
            if r:
                try:
                    got += os.read(self.fd, 65536)
                except OSError:
                    break
        self.out += got
        return got

    def send(self, b: bytes, t: float = 0.5) -> bytes:
        os.write(self.fd, b)
        return self.pump(t)

    def wait_for(self, needle: bytes, t: float) -> bool:
        end = time.time() + t
        while time.time() < end:
            if needle in _plain(self.out):
                return True
            self.pump(0.2)
        return needle in _plain(self.out)

    def close(self):
        if self.fd is not None:
            try:
                os.write(self.fd, b"\nexit\n")
                self.pump(0.5)
            except OSError:
                pass
        for s in (signal.SIGHUP, signal.SIGKILL):
            try:
                os.kill(self.pid, s)
            except ProcessLookupError:
                pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            pass
        if self.fd is not None:
            os.close(self.fd)


def _plain(b: bytes) -> bytes:
    return re.sub(rb"\x1b\[[0-9;?<>=$]*[A-Za-z~]|\x1b[()][0-9A-B]|\x1b[=>]", b"", b)


def consoles(before: set) -> dict:
    """Console client and engine processes not present in ``before``."""
    found = {}
    for p in os.listdir("/proc"):
        if not p.isdigit() or int(p) in before:
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline", "rb").read()
            st = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[0]
        except OSError:
            continue
        if cmd.startswith(b"tqp-console\0") or b"/tqp-console\0--socket" in cmd:
            found[int(p)] = ("client", st)
        elif b"turboquant_pro.console.engine" in cmd:
            found[int(p)] = ("engine", st)
    return found


def _pid_of(kind: str, before: set):
    for p, (k, _) in consoles(before).items():
        if k == kind:
            return p
    return None


def _client_exe(before: set):
    """The file the running client was started from, or None."""
    p = _pid_of("client", before)
    try:
        return os.readlink(f"/proc/{p}/exe") if p else None
    except OSError:
        return None


def run_case(name, action, cmd: bytes, cwd: str, env: dict) -> dict:
    before = set(consoles(set()))
    sh = Shell(cwd, env)
    try:
        start = len(sh.out)
        sh.send(cmd)
        up = sh.wait_for(b"1 system", 90)
        exe = _client_exe(before)
        action(sh, before)
        time.sleep(1.0)
        sh.pump(0.5)
        # does the shell read typed input again?
        sh.send(b"\x03", 0.3)  # cancel anything half-typed at the prompt
        token = f"ok_{name}".replace(" ", "_").replace(",", "").encode()
        sh.send(b"echo " + token + b"\r", 1.5)
        shell_ok = (b"\n" + token) in _plain(sh.out[start:]).replace(b"\r", b"")
        # a job left in the background is the shell's to show; bring it back
        # and quit it so the process check below is about leaks, not the case
        left = consoles(before)
        state = terminal_state(sh.out[start:])
        return {
            "case": name,
            "drew": up,
            "terminal_normal": not state["alt"]
            and not state["cursor_hidden"]
            and not state["autowrap_off"],
            "forbidden_modes": sorted(set(state["bad"])),
            "shell_reads_input": shell_ok,
            "left_processes": {str(k): v for k, v in left.items()},
            "client_exe": exe,
        }
    finally:
        for p in consoles(before):
            try:
                os.kill(p, signal.SIGCONT)
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        sh.close()


# ------------------------------------------------------------------ the cases
def q(sh, before):
    sh.send(b"q", 3)


def ctrl_c(sh, before):
    sh.send(b"\x03", 3)


def ctrl_z_fg_q(sh, before):
    sh.send(b"\x1a", 2)
    sh.send(b"fg\r", 3)
    sh.send(b"q", 3)


def ctrl_z_bg_fg_q(sh, before):
    sh.send(b"\x1a", 2)
    sh.send(b"bg\r", 2)
    sh.send(b"fg\r", 3)
    sh.send(b"q", 3)


def outside_stop_cont(sh, before):  # a thermal guard's pause of the client
    p = _pid_of("client", before)
    os.kill(p, signal.SIGSTOP)
    sh.pump(2)
    os.kill(p, signal.SIGCONT)
    sh.pump(2)
    sh.send(b"fg\r", 3)
    sh.send(b"q", 3)


def engine_paused(sh, before):  # the engine stopped from outside: screen lives
    p = _pid_of("engine", before)
    os.kill(p, signal.SIGSTOP)
    sh.pump(5)
    os.kill(p, signal.SIGCONT)
    sh.pump(2)
    sh.send(b"q", 3)


def sigterm(sh, before):
    os.kill(_pid_of("client", before), signal.SIGTERM)
    sh.pump(3)


def sighup(sh, before):
    os.kill(_pid_of("client", before), signal.SIGHUP)
    sh.pump(3)


def client_killed(sh, before):  # SIGKILL: nothing can restore; stty sane can
    os.kill(_pid_of("client", before), signal.SIGKILL)
    sh.pump(3)
    sh.send(b"reset\r", 3)


def run_hangup_case(name, cmd: bytes, cwd: str, env: dict, wait: float = 10.0) -> dict:
    """Close the terminal under a running console (a PuTTY window closed) and
    report whether any client or engine outlives it. There is no terminal left
    to check, only the process table."""
    before = set(consoles(set()))
    sh = Shell(cwd, env)
    try:
        sh.send(cmd)
        up = sh.wait_for(b"1 system", 90)
        exe = _client_exe(before)
        os.close(sh.fd)  # the terminal's other end goes away: a hang-up
        sh.fd = None
        end = time.time() + wait
        while consoles(before) and time.time() < end:
            time.sleep(0.2)
        left = consoles(before)
        return {
            "case": name,
            "drew": up,
            "left_processes": {str(k): v for k, v in left.items()},
            "client_exe": exe,
        }
    finally:
        for p in consoles(before):
            try:
                os.kill(p, signal.SIGCONT)
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
        sh.close()


# a hang-up reaches the console through the shell's SIGHUP, or, when something
# between the terminal and the console does not pass it on (a relay such as
# screen or sudo's pty, simulated here with setsid), only as the terminal
# itself going away
HANGUPS = [
    ("terminal closed", b""),
    ("terminal closed, no SIGHUP relayed", b"setsid -w "),
]


CASES = [
    ("q", q),
    ("Ctrl-C", ctrl_c),
    ("Ctrl-Z, fg, q", ctrl_z_fg_q),
    ("Ctrl-Z, bg, fg, q", ctrl_z_bg_fg_q),
    ("outside SIGSTOP, SIGCONT, fg, q", outside_stop_cont),
    ("engine paused from outside", engine_paused),
    ("SIGTERM", sigterm),
    ("SIGHUP", sighup),
]


def main(argv=None) -> int:
    """Run the cases (all, or those named) against this source tree, or with
    ``--installed``, against an installed package: ``--python`` is then that
    environment's interpreter, the shell runs outside the source tree, and the
    client must be the one installed with the package, not a source build."""
    import argparse
    import tempfile

    ap = argparse.ArgumentParser(description=main.__doc__.split(",")[0])
    ap.add_argument("cases", nargs="*", help="case names (default: all)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--installed", action="store_true")
    args = ap.parse_args(argv)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.installed:
        cwd, env = tempfile.mkdtemp(prefix="tqp-jobctl-"), {"PYTHONPATH": ""}
        prefix = os.path.dirname(os.path.dirname(os.path.abspath(args.python)))
        want = os.path.join(prefix, "lib")
    else:
        cwd, env, want = repo, {"PYTHONPATH": repo}, None
    cmd = (
        f"{args.python} -m turboquant_pro.cli console --demo "
        "--nats http://127.0.0.1:8222\r"
    ).encode()
    known = [n for n, _ in CASES] + [n for n, _ in HANGUPS]
    unknown = [c for c in args.cases if c not in known]
    if unknown:
        ap.error(f"unknown cases {unknown}; known: {known}")
    chosen = set(args.cases or known)

    def where_ok(r):  # installed: the client must come from the package
        exe = r.get("client_exe") or ""
        bundled = "/turboquant_pro/console/bin/tqp-console"
        return want is None or (exe.startswith(want) and exe.endswith(bundled))

    fails = 0
    for name, action in CASES:
        if name not in chosen:
            continue
        r = run_case(name, action, cmd, cwd, env)
        ok = (
            r["drew"]
            and r["terminal_normal"]
            and not r["forbidden_modes"]
            and r["shell_reads_input"]
            and not r["left_processes"]
            and where_ok(r)
        )
        fails += not ok
        print(("PASS " if ok else "FAIL ") + str(r))
    for name, prefix in HANGUPS:
        if name not in chosen:
            continue
        r = run_hangup_case(name, prefix + cmd, cwd, env)
        ok = r["drew"] and not r["left_processes"] and where_ok(r)
        fails += not ok
        print(("PASS " if ok else "FAIL ") + str(r))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
