#!/usr/bin/env python3
"""Run a GPU pod's work under in-pod telemetry, a hard deadline and a utilization floor.

    python pod_guard.py --deadline 7200 --heartbeat /data/kv5/runs/x/heartbeat.json -- cmd args...

The pod ends itself, so nothing outside ever deletes an active Job (in ssu-atlas-ai a
vanished active Job reads as an enforcement kill to a peer session's circuit breaker,
and the nats-bursting descriptor carries no activeDeadlineSeconds):

- every --interval seconds one line ``[gpu] t=... util=.. mem=.. elapsed=.. last=...`` goes
  to stdout (``kubectl logs``) and the same record to --heartbeat on the volume, with the
  child's latest output line, so a pod is never observed blind;
- past --deadline seconds the child is stopped and the guard exits 124;
- after --warmup seconds, --samples consecutive samples under --min-util percent stop the
  child and the guard exits 125: an idle GPU pod is a cluster violation, so it ends itself
  rather than wait for an outside watcher;
- otherwise the guard exits with the child's code.

The child's stdout and stderr pass through unchanged, prefixed by nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time


def gpu_sample() -> tuple:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip().splitlines()[0]
        util, mem = (int(x.strip()) for x in out.split(","))
        return util, mem
    except Exception:
        return None, None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline", type=int, required=True, help="seconds")
    ap.add_argument("--heartbeat", required=True)
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=600)
    ap.add_argument("--min-util", type=int, default=40)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        raise SystemExit("no command")
    os.makedirs(os.path.dirname(a.heartbeat), exist_ok=True)

    t0 = time.time()
    child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, start_new_session=True)
    last = {"line": ""}

    def pump():
        for line in child.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.strip():
                last["line"] = line.strip()[:240]

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()

    def stop(reason: str, code: int) -> int:
        print(f"[guard] STOP {reason}", flush=True)
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=60)
        except Exception:
            os.killpg(child.pid, signal.SIGKILL)
        beat(reason)
        return code

    low = 0

    def beat(state: str, util=None, mem=None) -> None:
        rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "state": state,
               "elapsed_s": int(time.time() - t0), "util": util, "mem_mib": mem,
               "low_samples": low, "last": last["line"]}
        tmp = a.heartbeat + ".tmp"
        with open(tmp, "w") as f:
            json.dump(rec, f)
        os.replace(tmp, a.heartbeat)

    while True:
        try:
            rc = child.wait(timeout=a.interval)
            reader.join(timeout=10)  # the child's last lines, before the final heartbeat
            print(f"[guard] child exited rc={rc} after {int(time.time() - t0)}s", flush=True)
            beat(f"exited rc={rc}")
            return rc
        except subprocess.TimeoutExpired:
            pass
        el = time.time() - t0
        util, mem = gpu_sample()
        if el > a.warmup and util is not None and util < a.min_util:
            low += 1
        else:
            low = 0
        print(f"[gpu] t={int(el)}s util={util} mem={mem}MiB low={low} last={last['line'][:120]}",
              flush=True)
        beat("running", util, mem)
        if el > a.deadline:
            return stop(f"deadline {a.deadline}s reached", 124)
        if low >= a.samples:
            return stop(f"GPU under {a.min_util}% for {low} samples after warm-up", 125)


if __name__ == "__main__":
    raise SystemExit(main())
