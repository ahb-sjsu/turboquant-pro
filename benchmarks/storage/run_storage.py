"""Run the storage probe on NRP, one storage class at a time, and observe each run.

For every class it creates a small fresh PVC, submits ``probe.py`` as an
exempt-class Job through nats-bursting (the Go controller mounts the claim),
samples the pod's CPU and memory from the cluster metrics API while it runs
(the observer NRP's enforcement uses), collects the probe's JSON from the pod
log, then deletes the Job, waits for its pod to go (a terminating pod keeps an
RWO volume attached) and deletes the PVC. Classes run sequentially, so at most
one probe pod exists at a time.

Run on Atlas, with the environment that has nats-py:

    /home/claude/env/bin/python benchmarks/storage/run_storage.py --out DIR \\
        [--classes rook-cephfs,linstor-ha] [--submit]

Without ``--submit`` it prints each PVC and descriptor with the preflight result
and creates nothing.

Preflight (NRP rules, ``reference_nrp_job_policies.md``), enforced in code:
exempt class (1 CPU, memory from the measured footprint), ephemeral storage
declared, no GPU, a probe that never sleeps and bounds every phase, a
timestamped Job name, one probe at a time, PVCs of 5Gi (far under the 64Gi
admission cap) that are deleted after use.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import subprocess
import sys
import time
from pathlib import Path

NS = "ssu-atlas-ai"
IMAGE = "python:3.12-slim"  # the probe is standard library only: nothing to install
CLASSES = {
    # class: (access mode, fstype families the mount must be)
    "rook-cephfs": ("ReadWriteMany", "ceph"),
    "rook-cephfs-east": ("ReadWriteMany", "ceph"),
    "rook-ceph-block": ("ReadWriteOnce", "ext4,xfs"),
    "linstor-ha": ("ReadWriteOnce", "ext4,xfs"),
    "linstor-unl": ("ReadWriteOnce", "ext4,xfs"),
}
PVC_GI = 5
CPU, MEM_MI, EPHEMERAL = "1", 512, "256Mi"
MEASURED_PEAK_RSS_MIB = 26  # the Atlas rehearsal, /usr/bin/time -v
DIRTY_MIB = 64  # the probe fsyncs and drops its pages every 64 MiB
BATCH = "tqp-storage-probe"
ZONE_LABEL = "topology.kubernetes.io/zone"


def kubectl(*args, check=True, input=None) -> str:
    r = subprocess.run(
        ["kubectl", "-n", NS, *args],
        capture_output=True,
        text=True,
        input=input,
        timeout=120,
    )
    if check and r.returncode:
        raise RuntimeError(f"kubectl {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def pvc_manifest(name: str, cls: str) -> dict:
    mode = CLASSES[cls][0]
    return {
        "apiVersion": "v1",
        "kind": "PersistentVolumeClaim",
        "metadata": {
            "name": name,
            "namespace": NS,
            "labels": {"atlas.io/batch": BATCH},
        },
        "spec": {
            "accessModes": [mode],
            "storageClassName": cls,
            "resources": {"requests": {"storage": f"{PVC_GI}Gi"}},
        },
    }


PROBE_OPTS: dict = {"mode": "rw", "burn_s": 0, "settle_s": 120}  # set by main()


def descriptor(
    name: str, claim: str, cls: str, mib: int, seconds: int, zone: str | None = None
) -> dict:
    ro = PROBE_OPTS["mode"] == "ro"
    script = Path(__file__).with_name("probe.py").read_bytes()
    run = (
        "import base64,os;"
        "exec(compile(base64.b64decode(os.environ['SCRIPT_B64']),'probe.py','exec'))"
    )
    return {
        "name": name,
        "image": IMAGE,
        "command": ["python", "-c", run],
        "args": [],
        "env": {
            "SCRIPT_B64": base64.b64encode(script).decode(),
            "PROBE_PATH": "/mnt/probe",
            "PROBE_EXPECT": CLASSES[cls][1],
            "PROBE_LABEL": cls,
            "PROBE_MIB": str(mib),
            "PROBE_S": str(seconds),
            "PROBE_MODE": PROBE_OPTS["mode"],
            "PROBE_BURN_S": str(PROBE_OPTS["burn_s"]),
            "PROBE_SETTLE_S": str(PROBE_OPTS["settle_s"]),
            "PYTHONUNBUFFERED": "1",
        },
        "resources": {
            "cpu": CPU,
            "memory": f"{MEM_MI}Mi",
            "gpu": 0,
            "ephemeral_storage": EPHEMERAL,
        },
        "labels": {"atlas.io/batch": BATCH},
        "backoff_limit": 0,
        "volumes": [
            {
                "name": "probe",
                "mount_path": "/mnt/probe",
                "claim_name": claim,
                **({"read_only": True} if ro else {}),
            }
        ],
        **({"node_selector": {ZONE_LABEL: zone}} if zone else {}),
    }


def preflight(d: dict, pvc: dict) -> list[str]:
    bad = []
    r = d["resources"]
    if int(r["cpu"]) > 1:
        bad.append("cpu above the exempt class")
    mem = int(r["memory"].removesuffix("Mi"))
    if mem > 2048:
        bad.append("memory above the exempt class")
    if mem < 1.25 * (MEASURED_PEAK_RSS_MIB + DIRTY_MIB):
        bad.append(f"memory {mem} Mi under 1.25 x (measured peak + dirty window)")
    if int(r.get("gpu") or 0):
        bad.append("requests a GPU")
    if not r.get("ephemeral_storage"):
        bad.append("no ephemeral storage declared")
    src = base64.b64decode(d["env"]["SCRIPT_B64"]).decode()
    if "sleep" in src.replace("# ", ""):
        bad.append("the probe code sleeps")
    if not d["name"].rsplit("-", 1)[-1].isdigit():
        bad.append("Job name not timestamped")
    size = pvc["spec"]["resources"]["requests"]["storage"]
    if int(size.removesuffix("Gi")) > 64:
        bad.append("PVC above the 64Gi admission cap")
    mib, gib = int(d["env"]["PROBE_MIB"]), int(size.removesuffix("Gi"))
    if mib + 64 > gib * 1024 * 0.8:
        bad.append("the probe would fill more than 80% of its volume")
    return bad


async def submit(d: dict, wait_s: float = 60) -> list:
    """burst.status.> subscribed BEFORE publishing (rejections appear only there)."""
    import nats

    nc = await nats.connect("nats://localhost:4222", name="tqp-storage-submit")
    got: list = []

    async def on_status(msg):
        if msg.subject.endswith(d["name"]):
            got.append(json.loads(msg.data))

    await nc.subscribe("burst.status.>", cb=on_status)
    await nc.flush()
    await nc.publish(
        "burst.submit", json.dumps({"job_id": d["name"], "descriptor": d}).encode()
    )
    await nc.flush()
    end = time.monotonic() + wait_s
    while not got and time.monotonic() < end:
        await asyncio.sleep(0.5)
    await nc.drain()
    return got


def pod_metrics(job: str) -> list:
    """The cluster's view of the job's pod: CPU (cores) and memory (bytes)."""
    raw = subprocess.run(
        [
            "kubectl",
            "get",
            "--raw",
            f"/apis/metrics.k8s.io/v1beta1/namespaces/{NS}/pods",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout
    out = []
    try:
        items = json.loads(raw).get("items", [])
    except ValueError:
        return out
    for it in items:
        if not it["metadata"]["name"].startswith(job):
            continue
        for c in it.get("containers", []):
            out.append(
                {
                    "t": time.time(),
                    "timestamp": it.get("timestamp"),
                    "pod": it["metadata"]["name"],
                    "window": it.get("window"),
                    "cpu_cores": _cores(c["usage"]["cpu"]),
                    "memory_bytes": _bytes(c["usage"]["memory"]),
                }
            )
    return out


def _cores(q: str) -> float:
    if q.endswith("n"):
        return int(q[:-1]) / 1e9
    if q.endswith("u"):
        return int(q[:-1]) / 1e6
    if q.endswith("m"):
        return int(q[:-1]) / 1e3
    return float(q)


def _bytes(q: str) -> float:
    units = {
        "Ki": 2**10,
        "Mi": 2**20,
        "Gi": 2**30,
        "k": 1e3,
        "M": 1e6,
        "G": 1e9,
        "m": 1e-3,
    }
    for u, f in units.items():
        if q.endswith(u):
            return float(q[: -len(u)]) * f
    return float(q)


def create_pvc(claim: str, cls: str, ev) -> str:
    kubectl("apply", "-f", "-", input=json.dumps(pvc_manifest(claim, cls)))
    ev("pvc created")
    end = time.monotonic() + 300
    while kubectl("get", "pvc", claim, "-o", "jsonpath={.status.phase}") != "Bound":
        if time.monotonic() > end:
            raise RuntimeError("PVC did not bind within 300 s")
        time.sleep(3)
    pv = kubectl("get", "pvc", claim, "-o", "jsonpath={.spec.volumeName}")
    ev("pvc bound", volume=pv)
    return pv


def delete_pvc(claim: str, ev) -> None:
    kubectl("delete", "pvc", claim, "--ignore-not-found", "--wait=false", check=False)
    ev("pvc deleted")


def probe_job(job, claim, cls, mib, seconds, timeout_s, zone, ev) -> dict:
    """Submit one probe Job on ``claim``, observe it, collect it, delete it and
    wait until its pod is gone (a terminating pod keeps an RWO volume)."""
    rec: dict = {"class": cls, "pvc": claim, "job": job, "zone": zone}
    try:
        d = descriptor(job, claim, cls, mib, seconds, zone)
        rec["submit_status"] = asyncio.run(submit(d))
        ev("submitted", status=[s.get("state") for s in rec["submit_status"]])
        rec["job_created"] = kubectl(
            "get",
            "job",
            job,
            "-o",
            "jsonpath={.metadata.creationTimestamp}",
            check=False,
        )
        metrics, end, phase = [], time.monotonic() + timeout_s, None
        while time.monotonic() < end:
            st = kubectl(
                "get",
                "job",
                job,
                "-o",
                "jsonpath={.status.succeeded}/{.status.failed}",
                check=False,
            )
            metrics += pod_metrics(job)
            if st.startswith("1/"):
                phase = "succeeded"
                break
            if st.endswith("/1"):
                phase = "failed"
                break
            time.sleep(10)
        rec["phase"] = phase or "timeout"
        rec["cluster_metrics"] = metrics
        pods = kubectl(
            "get",
            "pods",
            "-l",
            f"job-name={job}",
            "-o",
            "jsonpath={range .items[*]}{.metadata.name} {.spec.nodeName}{end}",
            check=False,
        ).split()
        rec["pod"], rec["node"] = (pods + [None, None])[:2]
        log = kubectl("logs", f"job/{job}", check=False)
        lines = [x for x in log.splitlines() if x.startswith("{")]
        rec["probe"] = json.loads(lines[-1]) if lines else None
        rec["log_tail"] = log.splitlines()[-5:]
        ev(rec["phase"], node=rec["node"])
    finally:
        kubectl("delete", "job", job, "--ignore-not-found", "--wait=true", check=False)
        end = time.monotonic() + 300
        while kubectl(
            "get", "pods", "-l", f"job-name={job}", "-o", "name", check=False
        ):
            if time.monotonic() > end:
                ev("pod still terminating after 300 s")
                break
            time.sleep(3)
    return rec


def _events(tag, sink):
    def ev(what, **kw):
        sink.append({"t": time.time(), "what": what, **kw})
        print(f"[{tag}] {what} {kw if kw else ''}", flush=True)

    return ev


def run_class(cls, out, mib, seconds, timeout_s, zone) -> dict:
    """One class: a fresh PVC, one probe, the PVC deleted."""
    ts = int(time.time())
    name = f"tqp-storage-{cls}-{ts}"
    events: list = []
    ev = _events(cls, events)
    rec = {"class": cls, "phase": "not started"}
    try:
        create_pvc(name, cls, ev)
        rec = probe_job(name, name, cls, mib, seconds, timeout_s, zone, ev)
    finally:
        delete_pvc(name, ev)
    rec["events"] = events
    (out / f"{cls}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def run_locations(cls, zones, out, mib, seconds, timeout_s) -> list:
    """One class, one shared PVC, the same probe from each zone in turn: the
    object stays fixed and only the observer's position changes."""
    ts = int(time.time())
    claim = f"tqp-storage-{cls}-loc-{ts}"
    events: list = []
    recs = []
    create_pvc(claim, cls, _events(cls, events))
    try:
        for z in zones:
            job = f"tqp-storage-loc-{z}-{int(time.time())}"[:63]
            rec = probe_job(
                job,
                claim,
                cls,
                mib,
                seconds,
                timeout_s,
                z,
                _events(f"{cls}@{z}", events),
            )
            rec["events"] = events
            (out / f"{cls}@{z}.json").write_text(json.dumps(rec, indent=2), "utf-8")
            recs.append(rec)
    finally:
        delete_pvc(claim, _events(cls, events))
    return recs


REUSABLE = "tqp-fleet-1t-"  # the only existing claims the read-only probe may use
EXISTING_CLASS = "linstor-unl"


def claim_in_use(claim: str) -> list:
    """Pods in the namespace that mount ``claim`` now (any phase)."""
    raw = kubectl("get", "pods", "-o", "json", check=False)
    try:
        items = json.loads(raw).get("items", [])
    except ValueError:
        return ["(pod list unreadable)"]
    return [
        p["metadata"]["name"]
        for p in items
        for v in p["spec"].get("volumes") or []
        if (v.get("persistentVolumeClaim") or {}).get("claimName") == claim
    ]


def run_existing(claims, zones, out, mib, seconds, timeout_s) -> list:
    """Read-only probes of existing 1T index volumes, every claim from every zone
    (a crossed design: object x observer). One probe at a time; a claim in use by
    any pod is skipped, never shared."""
    recs = []
    for claim in claims:
        if not claim.startswith(REUSABLE):
            raise SystemExit(f"refusing {claim}: only {REUSABLE}* may be reused")
        for z in zones:
            busy = claim_in_use(claim)
            events: list = []
            ev = _events(f"{claim}@{z}", events)
            if busy:
                ev("skipped: claim in use", pods=busy)
                continue
            job = f"tqp-storage-ro-{claim.rsplit('-', 1)[-1]}-{z}-{int(time.time())}"
            rec = probe_job(
                job[:63], claim, EXISTING_CLASS, mib, seconds, timeout_s, z, ev
            )
            rec["events"] = events
            (out / f"{claim}@{z}.json").write_text(json.dumps(rec, indent=2), "utf-8")
            recs.append(rec)
    return recs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True)
    p.add_argument("--classes", default=",".join(CLASSES))
    p.add_argument("--write-mib", type=int, default=1024)
    p.add_argument("--seconds", type=int, default=90)
    p.add_argument("--timeout", type=int, default=900, help="per probe, seconds")
    p.add_argument("--zone", help="pin every probe to this zone (class sweep)")
    p.add_argument(
        "--zones",
        help="location sweep: one class (the first of --classes, RWX), one shared "
        "PVC, the same probe from each of these zones in turn",
    )
    p.add_argument("--mode", choices=["rw", "ro"], default="rw")
    p.add_argument("--burn", type=float, default=0, help="seconds of 1-core load")
    p.add_argument("--settle", type=float, default=120)
    p.add_argument(
        "--existing",
        help="read-only reuse: comma-separated tqp-fleet-1t-* claims, each probed "
        "from every --zones zone (needs --mode ro)",
    )
    p.add_argument("--submit", action="store_true")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    PROBE_OPTS.update(mode=args.mode, burn_s=args.burn, settle_s=args.settle)
    if args.existing:
        if args.mode != "ro" or not args.zones:
            print("--existing needs --mode ro and --zones", file=sys.stderr)
            return 2
        claims = [c for c in args.existing.split(",") if c]
        zones = [z for z in args.zones.split(",") if z]
        d = descriptor(
            "tqp-storage-ro-x-1",
            claims[0],
            EXISTING_CLASS,
            args.write_mib,
            args.seconds,
            zones[0],
        )
        bad = preflight(d, pvc_manifest("x", EXISTING_CLASS))
        print(
            json.dumps({"existing": claims, "zones": zones, "preflight": bad or "PASS"})
        )
        if bad or not args.submit:
            return 2 if bad else 0
        recs = run_existing(
            claims, zones, out, args.write_mib, args.seconds, args.timeout
        )
        print(json.dumps({f"{r['pvc']}@{r['zone']}": r["phase"] for r in recs}))
        return 0 if all(r["phase"] == "succeeded" for r in recs) else 1
    classes = [c for c in args.classes.split(",") if c]
    unknown = [c for c in classes if c not in CLASSES]
    if unknown:
        print(f"unknown classes {unknown}; known {list(CLASSES)}", file=sys.stderr)
        return 2
    ok = True
    for cls in classes:
        name = f"tqp-storage-{cls}-{int(time.time())}"
        d = descriptor(name, name, cls, args.write_mib, args.seconds)
        bad = preflight(d, pvc_manifest(name, cls))
        print(json.dumps({"class": cls, "preflight": bad or "PASS"}))
        ok &= not bad
    if not ok:
        print("refusing: preflight failed", file=sys.stderr)
        return 2
    if not args.submit:
        print("dry run: nothing created (pass --submit)", file=sys.stderr)
        return 0
    if args.zones:
        cls = classes[0]
        if CLASSES[cls][0] != "ReadWriteMany":
            print(f"a location sweep needs an RWX class; {cls} is not", file=sys.stderr)
            return 2
        zones = [z for z in args.zones.split(",") if z]
        results = run_locations(
            cls, zones, out, args.write_mib, args.seconds, args.timeout
        )
        print(json.dumps({r["zone"]: r["phase"] for r in results}))
    else:
        results = [
            run_class(c, out, args.write_mib, args.seconds, args.timeout, args.zone)
            for c in classes
        ]
        print(json.dumps({r["class"]: r["phase"] for r in results}))
    return 0 if all(r["phase"] == "succeeded" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
