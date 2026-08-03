#!/usr/bin/env python3
"""100M smoke submitted through the nats-bursting controller.

Replaces the hand-rolled kubectl driver for this workload. The controller owns
submission, the politeness backoff (30s doubling to 15m, max 15 attempts), the
concurrency and queue-depth limits, and the dashboard labels.

Nothing in this file retries or re-applies anything. Hand-rolled retry next to
the controller's own backoff is what produced two retry storms against the
cluster, so the only thing here is a read-only completion watch used to
sequence the phases. The controller publishes one status event on submission
and does not stream terminal states, which is why the watch exists at all.

Volumes reach the controller through the JobDescriptor field added for exactly
this workload: a per-server index PVC, the shared PVC, and the code ConfigMap.

Run on Atlas, where the controller and NATS both live.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

sys.path.insert(0, "/home/claude/src/nats-bursting/python")
from nats_bursting import Client, JobDescriptor, Resources, Volume  # noqa: E402

NS = "ssu-atlas-ai"
IMAGE = "python:3.12"
CODE_CM = "tqp-fleet-code"
SHARED = "tqp-fleet-shared"
BATCH = "tqp-smoke100m"

# 4 servers x 5 shards x 5M rows = 100M
N_SERVERS = 4
SHARDS_PER_SERVER = 5
QUERY_SHARDS = "0,5,10,15"
QUERIES_PER_SHARD = "25"

# Mirrors the command block the working manifests use. The ConfigMap supplies
# the fleet_*.py scripts at /work; the turboquant_pro package itself comes from
# the repo clone at /repo. Dropping the clone is what made the first burst
# submission fail with ModuleNotFoundError.
RUN = """set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1
pip install -q --no-cache-dir numpy
git clone -q --depth 1 https://github.com/ahb-sjsu/turboquant-pro.git /repo
cd /repo && git log --oneline -1
export PYTHONPATH=/repo:/work
python /work/{script}
"""


def vols(server, idx_ro):
    v = []
    if server is not None:
        v.append(Volume(name="idx", mount_path="/idx",
                        claim_name=f"tqp-smoke100m-{server}", read_only=idx_ro))
    v.append(Volume(name="shared", mount_path="/shared", claim_name=SHARED))
    v.append(Volume(name="code", mount_path="/work", config_map=CODE_CM))
    return v


def desc(name, script, env, server, idx_ro=False, cpu="6", mem="8Gi"):
    """Build one descriptor.

    Sizing has to satisfy two opposing constraints. NRP deletes the workload
    object of any pod sitting far below its CPU request, so an oversized
    request on a light job gets it swept. But qcache cannot be shrunk into the
    exempt class: its own docstring records that each query shard is a 5M-row
    generation of ~640 MB plus temporaries, and 2Gi OOMs it. Sizing it that
    small traded a sweep for an OOM.

    The floor compares usage against the request, so the fix is a small CPU
    request the job will actually saturate, paired with the memory it genuinely
    needs. cpu=1 with 8Gi keeps the utilisation ratio high while leaving room
    for the generation.
    """
    return JobDescriptor(
        name=name,
        image=IMAGE,
        command=["bash", "-lc"],
        args=[RUN.format(script=script)],
        env={"TQP_RUN_TAG": "100m", **env},
        resources=Resources(cpu=cpu, memory=mem, ephemeral_storage="5Gi"),
        labels={"atlas.io/batch": BATCH},
        volumes=vols(server, idx_ro),
    )


def job_state(name):
    """Read-only view of one Job. Never mutates and never re-submits."""
    out = subprocess.run(
        ["kubectl", "get", "job", name, "-n", NS, "-o",
         "jsonpath={.status.succeeded}/{.status.failed}"],
        capture_output=True, text=True)
    if out.returncode != 0:
        return "absent"
    succeeded, _, failed = out.stdout.partition("/")
    if succeeded.strip() == "1":
        return "succeeded"
    if failed.strip() not in ("", "0"):
        return "failed"
    return "running"


MAX_RESUBMITS = 4

# How long a job may be absent before the first resubmit. Short on purpose:
# resubmission goes back through the controller, which applies its own backoff
# (30s doubling to 15m), so an early retry is queued rather than hammered. The
# bound on total effort is MAX_RESUBMITS, not this delay.
ABSENT_GRACE_S = 120

# How long to wait for a deleted pod to actually go before treating it as
# stuck. Normal termination takes seconds. A pod on a NotReady node stays
# Terminating indefinitely, since the kubelet never confirms deletion, and its
# RWO volume stays attached with it. Observed 2026-08-03 when
# hcc-nrp-sec-c1109.unl.edu went NotReady mid-run and stalled the IVF phase for
# the full hour. Fail fast and name the cause instead.
DRAIN_LIMIT_S = 300


def pods_gone(name):
    """True once no pod carries this job's name label.

    A swept Job's pod keeps its read-write-once volume attached while it
    terminates, so resubmitting immediately gives the replacement a
    Multi-Attach error. Wait for the pod to actually go.
    """
    out = subprocess.run(
        ["kubectl", "get", "pods", "-n", NS, "-l", f"job-name={name}",
         "--no-headers"],
        capture_output=True, text=True)
    return out.returncode == 0 and not out.stdout.strip()


def phase(client, jobs, label, timeout_s=3600, poll_s=30):
    """Submit a phase through the controller, then watch until it settles.

    The split of responsibility, learned the hard way. The controller owns
    admission: politeness, concurrency limits, and the backoff deciding when a
    job may start. It is fire-and-forget afterwards, so it never learns that
    NRP's utilization sweep deleted a running job and it will not resubmit one.
    That recovery is this function's job.

    The resubmission goes back through burst.submit rather than kubectl, so the
    controller's backoff and limits apply to it too. Retrying with raw kubectl
    apply is what bypassed all of that and produced the retry storms.
    """
    by_name = {j.name: j for j in jobs}

    def submit(j):
        r = client.submit(j)
        # accepted=False only means no status event arrived in the client's
        # window. The controller emits one event on submission and nothing
        # after, so a missed event is common and is not a rejection. Whether
        # the Job exists is the authoritative answer, established by the watch.
        print(f"    {j.name}: job_id={r.job_id} status_event={r.accepted}",
              flush=True)

    print(f"\n=== {label}: submitting {len(jobs)} job(s) via burst.submit",
          flush=True)
    for j in jobs:
        submit(j)

    names = list(by_name)
    t_start = time.time()
    deadline = t_start + timeout_s
    seen = {}
    resubmits = {n: 0 for n in names}
    first_absent = {}
    drain_started = {}

    while time.time() < deadline:
        states = {n: job_state(n) for n in names}
        for n, st in states.items():
            if seen.get(n) != st:
                print(f"    {n}: {st}", flush=True)
                seen[n] = st

        if any(st == "failed" for st in states.values()):
            return False
        if all(st == "succeeded" for st in states.values()):
            return True

        for n, st in states.items():
            if st != "absent":
                first_absent.pop(n, None)
                continue
            # One rule for every absent job. An earlier version tried to tell
            # "swept after running" from "never created" by whether polling had
            # caught it in a running state, but a job created and swept between
            # two polls is indistinguishable from one never created. That is
            # exactly what stranded ref-1, whose events showed it had started
            # and been killed inside a single 30s poll interval.
            first_absent.setdefault(n, time.time())
            waited = time.time() - first_absent[n]

            if resubmits[n] >= MAX_RESUBMITS:
                print(f"    {n} absent after {resubmits[n]} resubmits, "
                      f"giving up", flush=True)
                return False
            # Give the controller its own backoff window before the first
            # resubmit, so a job it is deliberately holding is not hurried.
            if resubmits[n] == 0 and waited < ABSENT_GRACE_S:
                continue
            # Do not race a dying pod for its volume.
            if not pods_gone(n):
                drain_started.setdefault(n, time.time())
                draining = time.time() - drain_started[n]
                if draining > DRAIN_LIMIT_S:
                    # A pod whose node has gone NotReady stays Terminating
                    # until the node returns, because the kubelet never
                    # confirms deletion. Waiting forever turns one sick node
                    # into a stalled phase, so say what is wrong and stop.
                    print(f"    {n} pod has been terminating for "
                          f"{int(draining)}s and will not release its volume."
                          f" Its node is probably NotReady. Check with:",
                          flush=True)
                    print(f"      kubectl get pods -n {NS} -l job-name={n} "
                          f"-o wide", flush=True)
                    return False
                if seen.get(n) != "draining":
                    print(f"    {n} absent; waiting for its pod to release "
                          f"the volume (up to {DRAIN_LIMIT_S}s)", flush=True)
                    seen[n] = "draining"
                continue
            drain_started.pop(n, None)
            resubmits[n] += 1
            print(f"    {n} absent -> resubmitting "
                  f"{resubmits[n]}/{MAX_RESUBMITS}", flush=True)
            submit(by_name[n])
            seen[n] = "resubmitted"
            first_absent[n] = time.time()

        time.sleep(poll_s)

    print(f"    {label} timed out after {timeout_s}s", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print descriptors as JSON and submit nothing")
    a = ap.parse_args()

    common = {"TQP_QUERY_SHARDS": QUERY_SHARDS,
              "TQP_QUERIES_PER_SHARD": QUERIES_PER_SHARD}

    builds = [desc(f"smoke100m-build-{i}", "fleet_build.py",
                   {"TQP_SERVER_ID": str(i),
                    "TQP_SHARDS_PER_SERVER": str(SHARDS_PER_SERVER),
                    "TQP_WRITE_ORIGINALS": "0",
                    "TQP_EXPORT_PREFIX": "server100m_"}, server=i)
              for i in range(N_SERVERS)]
    qcache = [desc("smoke100m-qcache", "fleet_qcache.py",
                   {"TQP_QCACHE_NAME": "queries100m.npy", **common},
                   server=None, cpu="1", mem="8Gi")]
    refs = [desc(f"smoke100m-ref-{i}", "fleet_ref.py",
                 {"TQP_SERVER_ID": str(i), **common}, server=i, idx_ro=True)
            for i in range(N_SERVERS)]
    ivfs = [desc(f"smoke100m-ivf-{i}", "fleet_ivf.py",
                 {"TQP_SERVER_ID": str(i), **common}, server=i, idx_ro=True)
            for i in range(N_SERVERS)]
    score = [desc("smoke100m-score", "fleet_score10.py",
                  {"TQP_N_SERVERS": str(N_SERVERS),
                   "TQP_N_ROWS": "100000000", **common},
                  server=None, cpu="1", mem="8Gi")]

    if a.dry_run:
        for j in builds[:1] + qcache + refs[:1] + ivfs[:1] + score:
            print(json.dumps(j.to_dict(), indent=1))
        return 0

    t0 = time.time()
    with Client() as client:
        for jobs, label in ((builds, "build"), (qcache, "query cache"),
                            (refs, "exact reference"), (ivfs, "routed IVF"),
                            (score, "merge + score")):
            if not phase(client, jobs, label):
                print(f"\n{label} FAILED after {time.time() - t0:.0f}s", flush=True)
                return 1
    print(f"\n100M smoke complete in {time.time() - t0:.0f}s", flush=True)
    print("BURST100M_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
