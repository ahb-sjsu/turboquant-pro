#!/usr/bin/env python3
"""Report-only observer of Part V's NRP jobs (app=tqp-kv5). Run on Atlas; deletes nothing.

    python part5_observe.py [--interval 60] [--pending-alert 1200] [--stall-alert 900]

Prints one line whenever something about a job changes, so a Monitor over it is never blind:
the job's state; a Pending pod's scheduling reason, condensed from its latest
FailedScheduling event (e.g. ``no free A10: 23 nodes Insufficient nvidia.com/gpu``); a
waiting container's reason (image pull, crash loop); the newest ``[gpu]``/``[guard]``/
``[t]``/progress line of the pod's log. Alerts, printed once each:

- ``PENDING_LONG`` when a pod has been Pending longer than --pending-alert seconds;
- ``STALLED`` when a Running pod's log has not changed for --stall-alert seconds;
- ``FINISHED`` / ``FAILED`` with the log tail, when a job ends.

It exits when no tqp-kv5 job is active. Deciding what to do with a stuck job is left to a
person: in ssu-atlas-ai an active Job is never deleted from outside (feedback memory
nrp_no_active_job_deletes); pods bound themselves with pod_guard.py.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import subprocess
import time

NS = "ssu-atlas-ai"


def k(*args, timeout=60) -> str:
    r = subprocess.run(["kubectl", "-n", NS, "--request-timeout=40s", *args],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout


def jobs() -> list:
    d = json.loads(k("get", "jobs", "-l", "app=tqp-kv5", "-o", "json") or '{"items": []}')
    return d["items"]


def pod_of(job: str):
    d = json.loads(k("get", "pods", "-l", f"job-name={job}", "-o", "json") or '{"items": []}')
    return d["items"][0] if d["items"] else None


def sched_reason(pod: str) -> str:
    ev = json.loads(k("get", "events", "--field-selector",
                      f"involvedObject.name={pod},reason=FailedScheduling", "-o", "json")
                    or '{"items": []}')["items"]
    if not ev:
        return "scheduling"
    msg = sorted(ev, key=lambda e: e.get("lastTimestamp") or "")[-1].get("message", "")
    gpu = re.search(r"(\d+) Insufficient nvidia.com/gpu", msg)
    if gpu:
        return f"no free GPU of the requested product: {gpu.group(1)} nodes Insufficient nvidia.com/gpu"
    return msg[:160]


def signals(job: str) -> tuple:
    """(latest progress line, latest [gpu] line, its utilization) from the pod's log."""
    lines = [x for x in k("logs", f"job/{job}", "--tail=60").splitlines() if x.strip()]
    gpu = [x for x in lines if x.startswith("[gpu]")]
    prog = [x for x in lines if not x.startswith("[gpu]")]
    util = None
    if gpu:
        m = re.search(r"util=(\d+)", gpu[-1])
        util = int(m.group(1)) if m else None
    return (prog[-1][:220] if prog else ""), (gpu[-1][:160] if gpu else ""), util


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--pending-alert", type=int, default=1200)
    ap.add_argument("--stall-alert", type=int, default=900)
    ap.add_argument("--gpu-every", type=int, default=600, help="seconds between telemetry lines")
    a = ap.parse_args()
    seen, alerted, log_changed, gpu_shown = {}, set(), {}, {}
    while True:
        active = 0
        now = time.time()
        for j in jobs():
            name = j["metadata"]["name"]
            st = j.get("status", {})
            if st.get("succeeded") or st.get("failed"):
                key = (name, "end")
                if key not in alerted:
                    end = "FINISHED" if st.get("succeeded") else "FAILED"
                    if name in seen:  # only jobs that ended while observed
                        tail = " | ".join(k("logs", f"job/{name}", "--tail=4").splitlines())
                        print(f"{name}: {end} | {tail[:400]}", flush=True)
                    alerted.add(key)
                continue
            active += 1
            pod = pod_of(name)
            if pod is None:
                state = "no pod yet"
            else:
                phase = pod["status"].get("phase", "?")
                state = phase
                if phase == "Pending":
                    cs = pod["status"].get("containerStatuses") or []
                    waiting = cs and cs[0].get("state", {}).get("waiting")
                    state = (f"Pending: {waiting.get('reason')}" if waiting
                             else f"Pending: {sched_reason(pod['metadata']['name'])}")
                    created = pod["metadata"]["creationTimestamp"]
                    age = now - calendar.timegm(time.strptime(created, "%Y-%m-%dT%H:%M:%SZ"))
                    if age > a.pending_alert and (name, "pending") not in alerted:
                        print(f"{name}: PENDING_LONG {int(age // 60)} min | {state}", flush=True)
                        alerted.add((name, "pending"))
                elif phase == "Running":
                    prog, gpu, util = signals(name)
                    if prog != log_changed.get(name, (None,))[0]:
                        log_changed[name] = (prog, now)
                        alerted.discard((name, "stall"))
                        print(f"{name}: {prog}", flush=True)
                    elif (now - log_changed[name][1] > a.stall_alert and (util is None or util < 40)
                          and (name, "stall") not in alerted):
                        mins = int((now - log_changed[name][1]) // 60)
                        print(f"{name}: STALLED no progress for {mins} min and GPU {util}% | {gpu}",
                              flush=True)
                        alerted.add((name, "stall"))
                    if gpu and now - gpu_shown.get(name, 0) > a.gpu_every:
                        print(f"{name}: {gpu}", flush=True)
                        gpu_shown[name] = now
            if seen.get(name) != state:
                print(f"{name}: {state}", flush=True)
                seen[name] = state
        if not active:
            print("no active tqp-kv5 job", flush=True)
            return 0
        time.sleep(a.interval)  # Atlas-side loop; no cluster job sleeps


if __name__ == "__main__":
    raise SystemExit(main())
