"""Stop our own NRP pods that sit under the cluster's utilization floors.

    python utilization_guard.py --selector app=tqp-rbq --apply

Bootstrapping a campaign, where the first cell of each class runs before anyone knows what it
uses, wants a short fuse rather than the defaults, and a heartbeat the submitter can check:

    python utilization_guard.py --selector app=tqp-rbq --apply         --interval 30 --window 10 --grace 900 --heartbeat pool/utilization_guard.heartbeat

which judges a pod on its first twenty minutes. Do not tighten this much further: a fuse has
to be shorter than the cluster's own window but longer than honest I/O, and a six-minute fuse
killed cells whose legitimate opening act is reading 38 GiB from the volume.

The cluster requires 20-200% of requested CPU and 20-150% of requested memory, measured as a
time average; pods at or below 1 CPU and 2 GiB are exempt. A snapshot of `kubectl top` is not
evidence of compliance: on 2026-09-15 eight campaign pods were flagged at 1-4% CPU while a
snapshot showed several at ~3.7 of 4 cores, because long CephFS-bound phases sink the average.

This samples `kubectl top` every --interval seconds, keeps a rolling window per pod, and once a
pod is older than --grace reports (or with --apply deletes) the Job of any pod whose windowed
mean falls below --floor of its request. It never touches pods it did not match, and without
--apply it only prints, so it can run alongside a campaign as an alarm before the cluster's own.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import time

EXEMPT_CPU, EXEMPT_MEM = 1.0, 2.0 * 2**30
FLOOR_TOLERANCE = 0.05  # how far under the floor a pod must sit before it is stopped


def sh(*args, ns):
    cmd = ["kubectl", "-n", ns, "--request-timeout=40s", *args]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=90, stdin=subprocess.DEVNULL
    )


def parse_cpu(v):
    """Kubernetes CPU quantity -> cores."""
    v = v.strip()
    if v.endswith("m"):
        return float(v[:-1]) / 1000
    if v.endswith("n"):
        return float(v[:-1]) / 1e9
    if v.endswith("u"):
        return float(v[:-1]) / 1e6
    return float(v)


def parse_mem(v):
    """Kubernetes memory quantity -> bytes."""
    v = v.strip()
    units = {
        "Ki": 2**10,
        "Mi": 2**20,
        "Gi": 2**30,
        "Ti": 2**40,
        "K": 1e3,
        "M": 1e6,
        "G": 1e9,
    }
    for u, mul in units.items():
        if v.endswith(u):
            return float(v[: -len(u)]) * mul
    return float(v)


def requests(ns, selector):
    """{pod: (cpu_cores, mem_bytes, job_name, age_s)} for running pods matching the selector."""
    r = sh(
        "get",
        "pods",
        "-l",
        selector,
        "--field-selector=status.phase=Running",
        "-o",
        "json",
        ns=ns,
    )
    if r.returncode:
        return {}
    now = time.time()
    out = {}
    for p in json.loads(r.stdout).get("items", []):
        res = p["spec"]["containers"][0].get("resources", {}).get("requests", {})
        if not res.get("cpu") or not res.get("memory"):
            continue
        job = next(
            (
                o["name"]
                for o in p["metadata"].get("ownerReferences", [])
                if o["kind"] == "Job"
            ),
            None,
        )
        start = p["status"].get("startTime")
        age = (
            now - time.mktime(time.strptime(start, "%Y-%m-%dT%H:%M:%SZ"))
            if start
            else 0.0
        )
        out[p["metadata"]["name"]] = (
            parse_cpu(res["cpu"]),
            parse_mem(res["memory"]),
            job,
            age,
        )
    return out


def usage(ns):
    """{pod: (cpu_cores, mem_bytes)} from metrics-server."""
    r = sh("top", "pods", "--no-headers", ns=ns)
    out = {}
    for line in r.stdout.splitlines():
        f = line.split()
        if len(f) >= 3:
            out[f[0]] = (parse_cpu(f[1]), parse_mem(f[2]))
    return out


def record(path, job, mean_cpu, mean_mem, peak_mem, req_cpu, req_mem, samples):
    """Merge one Job's measured usage into the observations file the submitters read.

    This is the measurement half of the guard: a submitter that has never seen a class run
    has no honest way to size it, and a fabricated estimate is what let eight pods go out at
    4 CPUs they never used. Peaks only ever grow, so a later quiet window cannot erase one.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            all_obs = json.load(fh)
    except (OSError, ValueError):
        all_obs = {}
    prev = all_obs.get(job, {})
    all_obs[job] = dict(
        mean_cpu_cores=round(mean_cpu, 3),
        mean_mem_gib=round(mean_mem / 2**30, 3),
        peak_mem_gib=round(max(peak_mem / 2**30, prev.get("peak_mem_gib", 0)), 3),
        request_cpu=req_cpu,
        request_mem_gib=round(req_mem / 2**30, 3),
        samples=samples,
        updated=time.strftime("%FT%TZ", time.gmtime()),
    )
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(all_obs, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="ssu-atlas-ai")
    ap.add_argument("--selector", default="app=tqp-rbq")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--window", type=int, default=15, help="samples kept per pod")
    ap.add_argument(
        "--grace", type=float, default=900.0, help="seconds before a pod is judged"
    )
    ap.add_argument(
        "--floor",
        type=float,
        default=0.20,
        help="fraction of request to enforce; the cluster's own floor, and deliberately not "
        "stricter: the margin belongs in sizing (nrp/sizing.py), because stopping a pod the "
        "cluster would accept throws away good work",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="delete the offending Job (default: report only)",
    )
    ap.add_argument(
        "--log-dir", help="save a stopped Job's logs here before deleting it"
    )
    ap.add_argument(
        "--observations",
        help="JSON file of per-Job measured usage, merged and rewritten each cycle; the "
        "submitters size the next run from it",
    )
    ap.add_argument(
        "--heartbeat",
        help="touch this path every cycle; submit_pool.py refuses to send an unmeasured "
        "calibration cell unless this heartbeat is fresh",
    )
    a = ap.parse_args()

    hist = collections.defaultdict(lambda: collections.deque(maxlen=a.window))
    acted = set()
    while True:
        if a.heartbeat:
            with open(a.heartbeat, "w") as fh:
                fh.write(time.strftime("%FT%TZ", time.gmtime()))
        req, use = requests(a.namespace, a.selector), usage(a.namespace)
        for pod, (rc, rm, job, age) in sorted(req.items()):
            if pod in use:
                hist[pod].append(use[pod])
            h = hist[pod]
            if age < a.grace or len(h) < a.window or job in acted:
                continue
            mc = sum(x[0] for x in h) / len(h)
            mm = sum(x[1] for x in h) / len(h)
            if a.observations and job:
                record(
                    a.observations, job, mc, mm, max(x[1] for x in h), rc, rm, len(h)
                )
            # Stop only what is clearly below the floor. A pod sitting exactly at it is
            # compliant, and killing one there throws away good work for a rounding error:
            # the pruned-scan benchmark was stopped at "mem 1.8/9.0 GiB = 20%".
            low = []
            slack = a.floor * (1 - FLOOR_TOLERANCE)
            if rc > EXEMPT_CPU and mc < slack * rc:
                low.append(f"cpu {mc:.2f}/{rc:g} cores = {100 * mc / rc:.0f}%")
            if rm > EXEMPT_MEM and mm < slack * rm:
                low.append(
                    f"mem {mm / 2**30:.1f}/{rm / 2**30:.1f} GiB = {100 * mm / rm:.0f}%"
                )
            if not low:
                continue
            stamp = time.strftime("%FT%TZ", time.gmtime())
            print(f"{stamp} UNDER-USED {pod} job={job} " + "; ".join(low), flush=True)
            if a.apply and job:
                if (
                    a.log_dir
                ):  # keep the evidence: deleting the Job deletes its pod's logs
                    os.makedirs(a.log_dir, exist_ok=True)
                    logs = sh("logs", f"job/{job}", ns=a.namespace)
                    with open(
                        os.path.join(a.log_dir, f"{job}.log"), "w", encoding="utf-8"
                    ) as fh:
                        head = f"# stopped {stamp}: " + "; ".join(low)
                        fh.write(head + os.linesep + logs.stdout)
                r = sh("delete", "job", job, ns=a.namespace)
                print(
                    f"{stamp} {'DELETED' if not r.returncode else 'DELETE FAILED'} {job} {r.stderr.strip()}",
                    flush=True,
                )
                acted.add(job)
        for pod in list(hist):
            if pod not in req:
                del hist[pod]
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
