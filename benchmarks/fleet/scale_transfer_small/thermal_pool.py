"""Run a list of single-threaded commands on Atlas with concurrency set by batch-probe's
ThermalController (the house rule for CPU work on Atlas).

usage: python thermal_pool.py TASKFILE [--max 12] [--target 82]
TASKFILE holds one shell command per line. A new command starts only while fewer are running
than ThermalController.get_threads() allows; a running command is never stopped when the
allowance drops. Prints a line per start and finish and a summary at the end; exits non-zero
if any command failed.
"""

import argparse
import subprocess
import sys
import time

from batch_probe import ThermalController

ap = argparse.ArgumentParser()
ap.add_argument("taskfile")
ap.add_argument("--max", type=int, default=12)
ap.add_argument("--target", type=float, default=82.0)
a = ap.parse_args()

tasks = [ln.strip() for ln in open(a.taskfile) if ln.strip() and not ln.startswith("#")]
thermal = ThermalController(
    target_temp=a.target, max_threads=a.max, min_threads=1, verbose=False
)
thermal.start()
running, failed, done = {}, [], 0
t0 = time.time()
# Ramp: the controller has no reading at start, and starting every allowed worker at once
# spikes the package past the guardian's threshold before the controller can react. Grow the
# pool by at most one worker per RAMP_S seconds, starting from one.
RAMP_S = 20
cap, last_grow = 1, time.time()
try:
    while tasks or running:
        if time.time() - last_grow >= RAMP_S:
            cap, last_grow = min(a.max, cap + 1), time.time()
        for p in [p for p in running if p.poll() is not None]:
            cmd = running.pop(p)
            done += 1
            if p.returncode != 0:
                failed.append(cmd)
            print(
                f"[{time.time() - t0:7.0f}s] exit={p.returncode} {cmd[-80:]}",
                flush=True,
            )
        allowed = max(1, min(cap, thermal.get_threads()))
        while tasks and len(running) < allowed:
            cmd = tasks.pop(0)
            running[subprocess.Popen(cmd, shell=True)] = cmd
            print(
                f"[{time.time() - t0:7.0f}s] start (running {len(running)}/{allowed}) {cmd[-80:]}",
                flush=True,
            )
        time.sleep(2)
finally:
    thermal.stop()
print(
    f"POOL_DONE ok={done - len(failed)} failed={len(failed)} wall_s={time.time() - t0:.0f}",
    flush=True,
)
for c in failed:
    print("FAILED", c, flush=True)
sys.exit(1 if failed else 0)
