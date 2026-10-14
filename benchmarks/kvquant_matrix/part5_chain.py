#!/usr/bin/env python3
"""Run one model's Part V waves in order on its volume set, one pod at a time. Run on Atlas,
in a screen session.

    python part5_chain.py --commit SHA --model qwen2.5-1.5b-instruct --waves tierb

Each wave goes out through ``part5_nrp.py set-run`` (preflight, guard heartbeat, set-busy
check, fresh-creation check), and the next is submitted only when its Job has completed.
A wave whose Job fails, or disappears, stops the chain and is reported: nothing is ever
deleted or retried from here (a cell is rerun unchanged only after its failure is read).
A wave already completed for this commit and model (the state file) is skipped, so a
stopped chain is resumed by running the same command again.
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
}
POLL_S = 60
WAIT_GUARD_S = 1800  # a stale guard heartbeat holds the chain this long, then stops it


def say(log, msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def job_state(name: str) -> str:
    """complete, failed, active, or missing."""
    r = N.W.kubectl("get", "job", name, "-o", "json")
    if r.returncode != 0:
        return "missing" if "NotFound" in r.stderr else "unknown"
    st = json.loads(r.stdout).get("status", {})
    conds = {c["type"] for c in st.get("conditions", []) if c["status"] == "True"}
    if "Complete" in conds:
        return "complete"
    if "Failed" in conds:
        return "failed"
    return "active"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--waves", default="tierb", help="an order name or a comma list")
    a = ap.parse_args(argv)
    waves = ORDERS.get(a.waves) or a.waves.split(",")
    bad = [w for w in waves if w not in P.WAVES or w == "fidelity"]
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
        held = 0
        while True:
            try:
                N.main(["set-run", "--commit", a.commit, "--model", a.model, "--wave", wave])
                break
            except SystemExit as e:
                msg = str(e)
                if "not beating" in msg and held < WAIT_GUARD_S:  # the guard restarts itself
                    held += POLL_S
                    time.sleep(POLL_S)
                    continue
                say(log, f"STOP {wave}: not submitted: {msg}")
                return 1
        say(log, f"submitted {name}")
        while True:
            time.sleep(POLL_S)
            s = job_state(name)
            if s in ("active", "unknown"):
                continue
            if s == "complete":
                done.append(wave)
                json.dump(state, open(state_path, "w"), indent=1)
                say(log, f"done {wave}")
                break
            say(log, f"STOP {wave}: job {s}; read its log before anything reruns")
            return 1
    say(log, f"CHAIN_DONE {a.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
