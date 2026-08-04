# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Pool driver for the 1T build phase: servers as independent tasks, no waves.

Wave 1 measured the cost of driver1t.sh's wave barrier — one server whose
volume sat on an offline node blocked seven finished ones for 5.5 h. This
driver holds ``MAXPAR`` servers in flight from a single pool: a stuck server
idles one slot while the pool works ahead, which is the Argo-DAG semantics
(``wf_build_1t.yaml``) without the workflows RBAC the namespace does not have.

Submissions go through the EXISTING ``burst.submit`` NATS flow (the
nats-bursting controller applies the politeness layer and creates the Job);
job state is read-only ``kubectl get``, and recycling uses ``kubectl delete
job`` exactly as driver1t.sh's ``delete_job_wait`` did. Job names match the
bash driver's (``tqp-fleet-1t-build-N``) so dashboards and habits carry over.

Failure handling, ported from wait_wave plus wave-1 lessons:
- NotFound must be CONFIRMED (two consecutive polls with the API reachable)
  before a job is treated as vanished — one API blip once deleted a healthy
  wave.
- A job Active longer than ``WEDGE_S`` is recycled: build-4 sat wedged at the
  memory ceiling for 7 h; per-shard resume makes a recycle cost minutes.
- A job with no Running pod for ``PEND_S`` is recycled (transient attach
  hangs); after ``STUCKVOL_AFTER`` consecutive such recycles the driver emits
  ``STUCKVOL`` and leaves that server parked for the shepherd — the seeded
  PVC rebuild is deliberately not automated (it deletes a volume).
- Every event is one ``=== HH:MM`` line in the driver1t.sh format, so the
  resident shepherd monitor needs only the extra file path.

State (done/retry counts) checkpoints to ``STATE_PATH`` so the driver itself
is restartable. Completion prints ``POOL_BUILDS_DONE``; the post-build phases
(qcache/ref/ivf/score) remain driver1t.sh's, run once builds exist.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "/home/claude/src/nats-bursting/python")
from nats_bursting import Client, JobDescriptor, Resources, Volume  # noqa: E402

NS = "ssu-atlas-ai"
PREFIX = "tqp-fleet-1t-build-"
MAXPAR = int(os.environ.get("TQP_POOL_MAXPAR", "8"))
MAXTRIES = int(os.environ.get("TQP_MAXTRIES", "1000"))
WEDGE_S = int(os.environ.get("TQP_WEDGE_S", "18000"))  # 5 h; fresh build ~3.5 h
PEND_S = int(os.environ.get("TQP_PEND_S", "2700"))  # 45 min with no Running pod
STUCKVOL_AFTER = 3
POLL_S = 60
STATE_PATH = os.environ.get("TQP_POOL_STATE", "/home/claude/tqp_fleet/pool_state.json")

# Verbatim job_build_1t.yaml body — that file carries the reasoning for every
# line (staged tarball startup, pip fallback, cgroup MEM reporter). Keep the
# two in sync when either changes.
BUILD_SCRIPT = r"""
set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1
if [ -f /shared/stage/deps.tgz ] && tar tzf /shared/stage/deps.tgz >/dev/null 2>&1; then
  tar xzf /shared/stage/deps.tgz -C /tmp
  export PYTHONPATH=/tmp/pylibs:/tmp/repo:/work
else
  echo "staged tarball unusable, falling back to pip"
  pip install -q --no-cache-dir numpy
  if [ -d /shared/stage/repo ]; then
    export PYTHONPATH=/shared/stage/repo:/work
  else
    echo "no repo available at all" >&2; exit 1
  fi
fi
python -c "import numpy; print('numpy', numpy.__version__)"
( while true; do
    cur=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
    pk=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo 0)
    echo "MEM cur=$((cur/1048576))Mi peak=$((pk/1048576))Mi"
    sleep 5
  done ) &
python /work/fleet_build.py
"""


def log(msg: str) -> None:
    print(f"=== {time.strftime('%H:%M', time.gmtime())} {msg}", flush=True)


def kubectl_json(*args: str):
    """kubectl ... -o json, or None on API failure (never raises)."""
    r = subprocess.run(
        ["kubectl", "-n", NS, *args, "-o", "json"], capture_output=True, text=True
    )
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def descriptor(sid: int) -> JobDescriptor:
    return JobDescriptor(
        name=f"{PREFIX}{sid}",
        image="python:3.12-slim",
        command=["/bin/bash", "-lc", BUILD_SCRIPT],
        env={
            "TQP_SERVER_ID": str(sid),
            "TQP_SHARDS_PER_SERVER": "400",
            "TQP_WRITE_ORIGINALS": "0",
            "TQP_EXPORT_PREFIX": "server1t_",
        },
        resources=Resources(cpu="1", memory="2Gi", ephemeral_storage="5Gi"),
        labels={"app": "tqp-fleet", "atlas.io/batch": "tqp-1t"},
        backoff_limit=0,
        volumes=[
            Volume(name="idx", mount_path="/idx", claim_name=f"tqp-fleet-1t-{sid}"),
            Volume(name="shared", mount_path="/shared", claim_name="tqp-fleet-shared"),
            Volume(name="code", mount_path="/work", config_map="tqp-fleet-code"),
        ],
    )


class Pool:
    def __init__(self, servers: list[int]):
        self.pool = list(servers)
        self.done: set[int] = set()
        self.parked: set[int] = set()  # STUCKVOL — needs shepherd intervention
        self.active: dict[int, dict] = {}  # sid -> {t0, tries, notfound, pendfails}
        self._load()

    def _load(self) -> None:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, encoding="utf-8") as f:
                st = json.load(f)
            self.done = set(st.get("done", []))
            self.parked = set(st.get("parked", []))
            self.pool = [s for s in self.pool if s not in self.done]
            log(f"STATE loaded: {len(self.done)} done, {len(self.parked)} parked")

    def _save(self) -> None:
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"done": sorted(self.done), "parked": sorted(self.parked)}, f)
        os.replace(tmp, STATE_PATH)

    # -- job lifecycle ---------------------------------------------------- #
    def _submit(self, sid: int, tries: int) -> None:
        # One short-lived connection per submit: this driver runs for days,
        # and a held-open transport would die on the first network blip.
        with Client() as client:
            res = client.submit(descriptor(sid))
        self.active[sid] = {
            "t0": time.time(),
            "tries": tries,
            "notfound": 0,
            "pendfails": self.active.get(sid, {}).get("pendfails", 0),
        }
        log(f"SUBMIT {PREFIX}{sid} try {tries}/{MAXTRIES} job_id={res.job_id}")

    def _delete_job(self, sid: int) -> None:
        subprocess.run(
            [
                "kubectl",
                "-n",
                NS,
                "delete",
                "job",
                f"{PREFIX}{sid}",
                "--ignore-not-found",
                "--wait=false",
            ],
            capture_output=True,
        )

    def _recycle(self, sid: int, why: str) -> None:
        st = self.active[sid]
        if st["tries"] >= MAXTRIES:
            log(f"GAVE UP {PREFIX}{sid} after {MAXTRIES} tries")
            self.parked.add(sid)
            del self.active[sid]
            self._save()
            return
        log(f"RECYCLE {PREFIX}{sid}: {why}")
        self._delete_job(sid)
        # Pods must actually be gone before resubmitting or the RWO volume
        # churns; give the cascade a moment and force-kill a lingerer.
        for _ in range(20):
            time.sleep(15)
            pods = kubectl_json("get", "pods", "-l", f"job-name={PREFIX}{sid}")
            if pods is not None and not pods.get("items"):
                break
        else:
            for p in (
                kubectl_json("get", "pods", "-l", f"job-name={PREFIX}{sid}")
                or {"items": []}
            )["items"]:
                subprocess.run(
                    [
                        "kubectl",
                        "-n",
                        NS,
                        "delete",
                        "pod",
                        p["metadata"]["name"],
                        "--force",
                        "--grace-period=0",
                    ],
                    capture_output=True,
                )
            time.sleep(10)
        self._submit(sid, st["tries"] + 1)

    # -- one poll cycle --------------------------------------------------- #
    def poll(self) -> None:
        jobs = kubectl_json("get", "jobs", "-l", "app=tqp-fleet")
        if jobs is None:
            log("NOTE kubectl unclear, skipping cycle")  # API blip: change nothing
            return
        by_name = {j["metadata"]["name"]: j for j in jobs.get("items", [])}

        for sid in list(self.active):
            name = f"{PREFIX}{sid}"
            st = self.active[sid]
            j = by_name.get(name)
            if j is None:
                st["notfound"] += 1  # confirm across two cycles (wait_wave lesson)
                if st["notfound"] >= 2:
                    self._recycle(sid, "job vanished (confirmed)")
                continue
            st["notfound"] = 0
            status = j.get("status", {})
            if status.get("succeeded"):
                log(f"DONE {PREFIX}{sid}")
                self.done.add(sid)
                del self.active[sid]
                self._save()
                continue
            if status.get("failed"):
                self._recycle(sid, "job failed")
                continue
            age = time.time() - st["t0"]
            if age > WEDGE_S:
                self._recycle(sid, f"active {int(age / 3600)}h > wedge bound")
                continue
            if age > PEND_S:
                pods = kubectl_json("get", "pods", "-l", f"job-name={name}")
                running = any(
                    p["status"].get("phase") == "Running"
                    for p in (pods or {"items": []})["items"]
                )
                if not running:
                    st["pendfails"] = st.get("pendfails", 0) + 1
                    if st["pendfails"] >= STUCKVOL_AFTER:
                        log(
                            f"STUCKVOL {PREFIX}{sid}: {st['pendfails']} consecutive "
                            "no-Running recycles — parked; likely needs the seeded "
                            "PVC rebuild (see HANDOFF_2026-08-04.md)"
                        )
                        self.parked.add(sid)
                        self._delete_job(sid)
                        del self.active[sid]
                        self._save()
                    else:
                        self._recycle(sid, "no Running pod after 45m")

        while len(self.active) < MAXPAR and self.pool:
            sid = self.pool.pop(0)
            name = f"{PREFIX}{sid}"
            j = by_name.get(name)
            if j and j.get("status", {}).get("succeeded"):
                log(f"DONE {PREFIX}{sid} (pre-existing)")
                self.done.add(sid)
                self._save()
                continue
            if j:  # leftover from the bash driver: adopt rather than duplicate
                self.active[sid] = {"t0": time.time(), "tries": 1, "notfound": 0}
                log(f"ADOPT {PREFIX}{sid} (job already exists)")
                continue
            self._submit(sid, 1)


def main() -> None:
    lo = int(os.environ.get("TQP_POOL_FROM", "16"))
    hi = int(os.environ.get("TQP_POOL_TO", "499"))
    servers = list(range(lo, hi + 1))
    pool = Pool(servers)
    log(f"POOL start: servers {lo}..{hi}, {len(pool.pool)} to build, MAXPAR={MAXPAR}")
    while pool.pool or pool.active:
        pool.poll()
        time.sleep(POLL_S)
    log(f"POOL_BUILDS_DONE done={len(pool.done)} parked={sorted(pool.parked)}")
    if pool.parked:
        sys.exit(1)


if __name__ == "__main__":
    main()
