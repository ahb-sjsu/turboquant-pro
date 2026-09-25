"""Circuit breaker for NRP submissions: hold work while jobs in the namespace are being deleted.

    python nrp_breaker.py --queue Q.json --state S.json --log L [--dry-run]    # on Atlas

The signal is namespace-wide: a Job whose uid disappears while it was active (neither
succeeded nor failed) is an unexplained deletion, whoever owned it. States:

    CLOSED     queued items are submitted, one per tick
    OPEN       >= THRESHOLD deletions inside WINDOW: nothing is submitted
    HALF_OPEN  QUIET seconds without a deletion: ONE queued item goes out as the probe;
               it surviving PROBE_OK seconds (or completing) closes the breaker, any
               deletion reopens it with the quiet period doubled (capped at MAX_QUIET)

The queue is a JSON list of {"cmd": <bash command that submits>, "job": <Job name>}. A queued
job that is deleted goes back to the queue (at most MAX_TRIES submissions); a job that FAILS is
a real error and is not retried. A kubectl listing that errors, or comes back empty while jobs
were active a tick ago, is skipped, never read as a mass deletion. Starts OPEN: it has not seen
the namespace quiet yet. Runs on Atlas; the loop's pause is Atlas-side, no cluster job sleeps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

NS = "ssu-atlas-ai"
WINDOW, THRESHOLD = 1800, 2
QUIET, MAX_QUIET, PROBE_OK = 2700, 4 * 3600, 1200
MAX_TRIES, TICK = 3, 30


class Breaker:
    def __init__(self, now: float, quiet: float = QUIET):
        self.state = "OPEN"
        self.base_quiet = quiet
        self.quiet = quiet
        self.last_deletion = now  # unseen history counts as recent
        self.deletions = []  # (time, name)
        self.prev = {}  # uid -> (name, active)
        self.probe = None  # (job name, submitted at)

    def observe(self, jobs: dict, now: float) -> list:
        """``jobs``: uid -> (name, active, succeeded, failed). Returns the deleted names."""
        gone = [
            name
            for uid, (name, active) in self.prev.items()
            if active and uid not in jobs
        ]
        self.prev = {u: (j[0], j[1]) for u, j in jobs.items()}
        for name in gone:
            self.deletions.append((now, name))
            self.last_deletion = now
        self.deletions = [(t, n) for t, n in self.deletions if now - t <= WINDOW]
        return gone

    def step(self, now: float, gone: list, jobs_by_name: dict) -> str:
        """Advance the state; returns the new state. ``jobs_by_name``: name -> (active,
        succeeded, failed) for jobs present now."""
        if self.state == "CLOSED":
            if len(self.deletions) >= THRESHOLD:
                self.state = "OPEN"
        elif self.state == "OPEN":
            if now - self.last_deletion >= self.quiet:
                self.state = "HALF_OPEN"
        elif self.state == "HALF_OPEN":
            if gone:
                self.state = "OPEN"
                self.quiet = min(2 * self.quiet, MAX_QUIET)
                self.probe = None
            elif self.probe:
                name, t0 = self.probe
                st = jobs_by_name.get(name)
                if (st and st[1]) or (st and st[0] and now - t0 >= PROBE_OK):
                    self.state = "CLOSED"
                    self.quiet = self.base_quiet
                    self.probe = None
                elif st and st[2]:
                    self.probe = None  # a real failure says nothing about deletions
        return self.state

    def may_submit(self) -> bool:
        return self.state == "CLOSED" or (self.state == "HALF_OPEN" and not self.probe)


def list_jobs():
    r = subprocess.run(
        ["kubectl", "-n", NS, "--request-timeout=60s", "get", "jobs", "-o", "json"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if r.returncode != 0:
        return None
    out = {}
    for j in json.loads(r.stdout)["items"]:
        s = j.get("status", {})
        done = bool(s.get("succeeded")) or bool(s.get("failed"))
        out[j["metadata"]["uid"]] = (
            j["metadata"]["name"],
            not done,
            bool(s.get("succeeded")),
            bool(s.get("failed")),
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--cwd", default=".")
    ap.add_argument("--dry-run", action="store_true", help="never submit")
    a = ap.parse_args(argv)

    def log(msg):
        with open(a.log, "a") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%SZ ", time.gmtime()) + msg + "\n")

    queue = [dict(q, tries=q.get("tries", 0)) for q in json.load(open(a.queue))]
    inflight, done, failed = {}, [], []
    br = Breaker(time.time())
    log(f"start OPEN, quiet {QUIET}s, {len(queue)} queued")
    while True:
        now = time.time()
        jobs = list_jobs()
        if jobs is None or (not jobs and any(v[1] for v in br.prev.values())):
            log("listing failed or empty; tick skipped")
            time.sleep(TICK)
            continue
        gone = br.observe(jobs, now)
        by_name = {v[0]: v[1:] for v in jobs.values()}
        for n in gone:
            log(f"DELETED {n}")
        before = br.state
        state = br.step(now, gone, by_name)
        if state != before:
            log(
                f"{before} -> {state} (deletions in window {len(br.deletions)}, quiet {br.quiet}s)"
            )
        for name, item in list(inflight.items()):
            st = by_name.get(name)
            if name in gone or st is None:
                del inflight[name]
                if item["tries"] < MAX_TRIES:
                    queue.insert(0, item)
                    log(f"REQUEUE {name} after deletion (tries {item['tries']})")
                else:
                    failed.append(item)
                    log(f"GAVE UP {name} after {item['tries']} deleted tries")
            elif st[1]:
                done.append(item)
                del inflight[name]
                log(f"DONE {name}")
            elif st[2]:
                failed.append(item)
                del inflight[name]
                log(f"FAILED {name} (job failed; not retried)")
        if queue and br.may_submit() and not a.dry_run:
            item = queue.pop(0)
            item["tries"] += 1
            r = subprocess.run(
                ["bash", "-lc", item["cmd"]], cwd=a.cwd, capture_output=True, text=True
            )
            log(
                f"SUBMIT {item['job']} try {item['tries']} rc={r.returncode} {r.stdout[-200:]!r}"
            )
            if r.returncode == 0:
                inflight[item["job"]] = item
                if br.state == "HALF_OPEN":
                    br.probe = (item["job"], now)
                    log(f"PROBE {item['job']}")
            else:
                failed.append(item)
        tmp = a.state + ".tmp"
        json.dump(
            {
                "state": br.state,
                "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                "deletions_in_window": br.deletions,
                "last_deletion_age_s": round(now - br.last_deletion),
                "quiet_s": br.quiet,
                "probe": br.probe,
                "queued": [q["job"] for q in queue],
                "inflight": list(inflight),
                "done": [q["job"] for q in done],
                "failed": [q["job"] for q in failed],
            },
            open(tmp, "w"),
            indent=1,
        )
        os.replace(tmp, a.state)
        if not queue and not inflight:
            log("queue drained; exit")
            return 0
        time.sleep(TICK)


if __name__ == "__main__":
    raise SystemExit(main())
