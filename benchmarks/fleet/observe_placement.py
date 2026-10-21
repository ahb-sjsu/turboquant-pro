"""Watch the namespace's jobs and record where each ran, without touching them.

The placement experiment would have re-run 1T scans on chosen nodes; the 1T
setup is back in real use, so this observes the real work instead. It only
reads the Kubernetes API (pods, nodes, the metrics API), creates nothing and
changes nothing, and writes two local files:

- ``events.jsonl``: one line per change seen in a pod (first seen, scheduled
  onto a node, phase change, container started or finished with its exit code),
  with the node's zone and region;
- ``metrics.jsonl``: the metrics API's CPU and memory for each running pod, with
  the API's own sample ``timestamp`` and ``window`` (which, measured on
  2026-09-27, is not the time the value was computed: read it as a trailing
  average refreshed about every 10 to 60 s).

``summarize()`` turns them into one row per pod: node, zone, container wall
time, exit code, mean metrics CPU. Joined with the job's own result files (the
1T drivers record per-server wall times), that gives the attribution of
wall-time spread to placement that the 1T record could not.

    python benchmarks/fleet/observe_placement.py --out DIR --hours 24 [--every 60]
    python benchmarks/fleet/observe_placement.py --summarize DIR

It ignores the fabric's own standing pods (the NATS leaf and the burst
controller) and, by default, pods older than when it started watching.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

NS = "ssu-atlas-ai"
STANDING = ("atlas-nats-leaf", "nats-bursting-controller")
ZONE, REGION = "topology.kubernetes.io/zone", "topology.kubernetes.io/region"


def _get(*args) -> dict | None:
    r = subprocess.run(
        ["kubectl", *args, "-o", "json"], capture_output=True, text=True, timeout=120
    )
    if r.returncode:
        return None
    try:
        return json.loads(r.stdout)
    except ValueError:
        return None


def _raw(path: str) -> dict | None:
    r = subprocess.run(
        ["kubectl", "get", "--raw", path], capture_output=True, text=True, timeout=120
    )
    try:
        return json.loads(r.stdout) if r.returncode == 0 else None
    except ValueError:
        return None


def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() if s else None


def _cores(q: str) -> float:
    for suf, f in (("n", 1e-9), ("u", 1e-6), ("m", 1e-3)):
        if q.endswith(suf):
            return int(q[:-1]) * f
    return float(q)


def _bytes(q: str) -> float:
    for suf, f in (
        ("Ki", 2**10),
        ("Mi", 2**20),
        ("Gi", 2**30),
        ("k", 1e3),
        ("M", 1e6),
        ("G", 1e9),
        ("m", 1e-3),
    ):
        if q.endswith(suf):
            return float(q[: -len(suf)]) * f
    return float(q)


def node_places() -> dict:
    d = _get("get", "nodes") or {"items": []}
    return {
        n["metadata"]["name"]: {
            "zone": (n["metadata"].get("labels") or {}).get(ZONE),
            "region": (n["metadata"].get("labels") or {}).get(REGION),
        }
        for n in d["items"]
    }


def pod_state(p: dict) -> dict:
    cs = (p["status"].get("containerStatuses") or [{}])[0]
    st = cs.get("state") or {}
    run, term = st.get("running") or {}, st.get("terminated") or {}
    return {
        "phase": p["status"].get("phase"),
        "node": p["spec"].get("nodeName"),
        "started": _ts(run.get("startedAt") or term.get("startedAt")),
        "finished": _ts(term.get("finishedAt")),
        "exit_code": term.get("exitCode"),
        "reason": term.get("reason"),
        "restarts": cs.get("restartCount"),
    }


def watch(
    out: Path, hours: float, every: float, include_old: bool, selector: str | None
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ev = open(out / "events.jsonl", "a", encoding="utf-8")
    mx = open(out / "metrics.jsonl", "a", encoding="utf-8")
    ev.write(
        json.dumps(
            {
                "t": time.time(),
                "watch_started": True,
                "selector": selector,
                "every_s": every,
                "include_old": include_old,
            }
        )
        + "\n"
    )
    ev.flush()
    places, places_t = node_places(), time.time()
    t_start, seen = time.time(), {}
    end = t_start + hours * 3600
    while time.time() < end:
        now = time.time()
        if now - places_t > 3600:  # nodes come and go; refresh hourly
            places, places_t = node_places(), now
        pods = _get("-n", NS, "get", "pods", *(["-l", selector] if selector else []))
        for p in (pods or {"items": []})["items"]:
            name = p["metadata"]["name"]
            if name.startswith(STANDING):
                continue
            created = _ts(p["metadata"].get("creationTimestamp"))
            if not include_old and created and created < t_start - 60:
                continue
            s = pod_state(p)
            if s["node"]:
                s.update(places.get(s["node"], {"zone": None, "region": None}))
            key = {k: s[k] for k in ("phase", "node", "started", "finished")}
            if seen.get(name) != key:
                seen[name] = key
                labels = p["metadata"].get("labels") or {}
                ev.write(
                    json.dumps(
                        {
                            "t": now,
                            "pod": name,
                            "job": labels.get("job-name"),
                            "batch": labels.get("atlas.io/batch"),
                            "created": created,
                            **s,
                        }
                    )
                    + "\n"
                )
        m = _raw(f"/apis/metrics.k8s.io/v1beta1/namespaces/{NS}/pods") or {}
        for it in m.get("items", []):
            name = it["metadata"]["name"]
            if name.startswith(STANDING) or name not in seen:
                continue
            for c in it.get("containers", []):
                mx.write(
                    json.dumps(
                        {
                            "t": now,
                            "pod": name,
                            "timestamp": it.get("timestamp"),
                            "window": it.get("window"),
                            "cpu_cores": _cores(c["usage"]["cpu"]),
                            "memory_bytes": _bytes(c["usage"]["memory"]),
                        }
                    )
                    + "\n"
                )
        ev.flush()
        mx.flush()
        time.sleep(max(1.0, every - (time.time() - now)))  # on Atlas, not in a pod
    ev.close()
    mx.close()


def summarize(out: Path) -> list:
    last, cpu = {}, {}
    for line in open(out / "events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        if "pod" not in e:  # a watch-start marker
            continue
        last.setdefault(e["pod"], {}).update(
            {k: v for k, v in e.items() if v is not None}
        )
    for line in open(out / "metrics.jsonl", encoding="utf-8"):
        m = json.loads(line)
        cpu.setdefault(m["pod"], []).append(m["cpu_cores"])
    rows = []
    for pod, e in sorted(last.items()):
        wall = (
            e["finished"] - e["started"]
            if e.get("finished") and e.get("started")
            else None
        )
        c = cpu.get(pod, [])
        rows.append(
            {
                "pod": pod,
                "job": e.get("job"),
                "batch": e.get("batch"),
                "node": e.get("node"),
                "zone": e.get("zone"),
                "region": e.get("region"),
                "phase": e.get("phase"),
                "exit_code": e.get("exit_code"),
                "container_wall_s": wall,
                "metrics_cpu_mean": sum(c) / len(c) if c else None,
                "metrics_samples": len(c),
            }
        )
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", help="directory to write events.jsonl and metrics.jsonl")
    p.add_argument("--hours", type=float, default=24.0)
    p.add_argument("--every", type=float, default=60.0, help="seconds between polls")
    p.add_argument(
        "--include-old",
        action="store_true",
        help="also record pods created before the watch began",
    )
    p.add_argument("--selector", help="label selector, e.g. atlas.io/batch=tqp-1t")
    p.add_argument("--summarize", help="print one row per pod from a watch directory")
    a = p.parse_args(argv)
    if a.summarize:
        print(json.dumps(summarize(Path(a.summarize)), indent=1))
        return 0
    if not a.out:
        p.error("--out is required to watch")
    watch(Path(a.out), a.hours, a.every, a.include_old, a.selector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
