#!/usr/bin/env python3
"""Run one model's Part V waves in order on its volume set, one pod at a time. Run on Atlas,
in a screen session.

    python part5_chain.py --commit SHA --model qwen2.5-1.5b-instruct --waves tierb

NRP pods can be deleted at any time (preemption, the idle-GPU sweep, node trouble), so a
wave is a resumable unit: its cells resume per document and per chunk, and a Job that
fails or vanishes is resubmitted unchanged (the prereg's operational rerun), up to
MAX_TRIES times. Only a wave that keeps failing stops the chain, which is how a real
defect shows itself, or a pod that ends with an arm's outputs short (its INCOMPLETE line):
a cell that fails inside a pod fails the same way again. Each wave goes out through ``part5_nrp.py set-run`` (preflight,
guard heartbeat, set-busy check, fresh-creation check); a busy set or a stale guard
heartbeat holds the chain rather than stopping it. A restarted chain adopts the wave's
Job if it is still running this commit, and a wave completed for this commit and model
(the state file) is skipped. Nothing is ever deleted from here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import part5_grid as P  # noqa: E402
import part5_nrp as N  # noqa: E402

# Tier B runs every arm (prereg, "Tier B throughout"); the gate wave goes first because
# the baselines read its u3 and the g0 wave fits the calibration tables before any
# scored arm reads them. The fidelity check is Qwen2.5-7B's only.
ORDERS = {
    "tierb": ["gate", "g0", "verdict", "head", "baselines", "reported3",
              "reported2", "reported4", "head2", "head4"],
    # Tier A after its gate passes G1: the verdict waves and V5's baselines (the
    # registration guard in set-run holds them until the prereg is merged)
    "tiera": ["g0", "verdict", "head", "baselines"],
}
POLL_S = 60
MAX_TRIES = 8  # submissions of one wave; resumed cells make each retry cheaper
HOLD_S = 6 * 3600  # a busy set or stale guard heartbeat holds the chain this long at most


def say(log, msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def job_state(name: str, commit: str | None = None) -> str:
    """complete, failed, active, missing or unknown; with ``commit``, a Job running
    another commit's code reads as ``other``."""
    r = N.W.kubectl("get", "job", name, "-o", "json")
    if r.returncode != 0:
        return "missing" if "NotFound" in r.stderr else "unknown"
    job = json.loads(r.stdout)
    if commit:
        cmd = " ".join(job["spec"]["template"]["spec"]["containers"][0].get("command", []))
        if f"HARNESS_COMMIT={commit}" not in cmd:
            return "other"
    st = job.get("status", {})
    conds = {c["type"] for c in st.get("conditions", []) if c["status"] == "True"}
    if "Complete" in conds:
        return "complete"
    if "Failed" in conds:
        return "failed"
    return "active"


def incomplete(name: str) -> str:
    """The pod's INCOMPLETE line if the Job failed its outputs check, else ''. A cell
    that fails inside a pod fails the same way on resubmission, so this stops the chain."""
    r = N.W.kubectl("logs", f"job/{name}", "--tail=50")
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith(N.INCOMPLETE)]
    return lines[-1] if r.returncode == 0 and lines else ""


def submit(log, commit: str, model: str, wave: str) -> bool:
    """Submit one wave, holding while its set is busy or the guard is stale."""
    held = 0
    told = False
    while True:
        try:
            N.main(["set-run", "--commit", commit, "--model", model, "--wave", wave])
            return True
        except SystemExit as e:
            msg = str(e)
            if "waits for registration" in msg:  # held until the prereg is merged
                if not told:
                    say(log, f"holding {wave}: {msg}")
                    told = True
                time.sleep(5 * POLL_S)
                continue
            transient = ("not beating", "in use by", "timed out")  # kubectl's API stalls too
            if any(t in msg for t in transient) and held < HOLD_S:
                held += POLL_S
                time.sleep(POLL_S)
                continue
            say(log, f"STOP {wave}: not submitted: {msg}")
            return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--waves", default="tierb", help="an order name or a comma list")
    a = ap.parse_args(argv)
    waves = ORDERS.get(a.waves) or a.waves.split(",")
    bad = [w for w in waves if w not in P.WAVES
           or (w == "fidelity" and a.model != P.FIDELITY_MODEL)]
    if bad:
        raise SystemExit(f"not chainable here: {bad}")
    key = N.short(a.model)
    log = os.path.join(N.STATE, f"chain_{key}.log")
    state_path = os.path.join(N.STATE, f"chain_{key}.json")
    state = json.load(open(state_path)) if os.path.exists(state_path) else {}
    done = state.setdefault(a.commit, [])
    say(log, f"chain {a.model} commit={a.commit[:12]} waves={waves} done={done}")
    for wave in waves:
        if wave in done:
            continue
        name = f"kv5-run-{wave}-{key}"
        s = job_state(name, a.commit)
        if s == "complete":  # finished while no chain was watching
            tries = MAX_TRIES
        elif s == "active":
            say(log, f"adopted {name}")
            tries = 1
        else:
            if not submit(log, a.commit, a.model, wave):
                return 1
            say(log, f"submitted {name}")
            tries = 1
        while True:
            s = job_state(name, a.commit)
            if s in ("active", "unknown"):
                time.sleep(POLL_S)
                continue
            if s == "complete":
                done.append(wave)
                json.dump(state, open(state_path, "w"), indent=1)
                say(log, f"done {wave}")
                break
            short = incomplete(name) if s == "failed" else ""
            if short:
                say(log, f"STOP {wave}: {short}; read the cells' lb.log and ppl.log")
                return 1
            if tries >= MAX_TRIES:
                say(log, f"STOP {wave}: job {s} after {tries} submissions; read its log")
                return 1
            say(log, f"resubmit {wave}: job {s} (submission {tries + 1} of {MAX_TRIES})")
            if not submit(log, a.commit, a.model, wave):
                return 1
            tries += 1
            time.sleep(POLL_S)
    say(log, f"CHAIN_DONE {a.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
