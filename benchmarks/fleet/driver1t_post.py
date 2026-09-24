# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License
"""Pool driver for the 1T post-build phases: query cache, exact reference, routed IVF, score.

``driver1t_pool.py`` finished the build phase as a pool of independent server
tasks; this runs the four phases that follow it the same way, in order:

  qcache  one job, the seeded query set cached on the shared volume
  ref     one job per server, the exact ADC full-scan partial top-K
  ivf     one job per server, the routed-IVF partials at nprobe 32 and 128
  score   one job, the exact merge and the recall

Each phase is a pool of ``MAXPAR`` jobs in flight with no wave barrier, every
submission through the existing ``burst.submit`` NATS flow (the controller's
politeness layer caps concurrency at 20 and paces submissions), job state read
by ``kubectl get``, and the same failure handling as the build pool: NotFound
confirmed across two polls, wedged jobs recycled, vanished or failed jobs
re-issued up to ``MAXTRIES``. Every phase script is idempotent on the volume,
so a re-issued job resumes at its own partial.

Query count. The 100B run used 500 queries (4 shards x 125). This run uses
100 (4 x 25), the 1B run's count, because the reference scan is compute-bound
in the query count and 500 would cost about 12,500 CPU-hours at 1T. The
query set is the first 25 rows of the same four seeded shards, so the
reference and the routed pass see the same queries by construction; the score
records ``nq`` from the reference partials. State checkpoints per phase to
``STATE_PATH``, so the driver is restartable. Completion prints
``DRIVER1T_POST_DONE`` and the score log is copied to ``score_1T.log``.
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
N_SERVERS = int(os.environ.get("TQP_N_SERVERS", "500"))
MAXPAR = int(os.environ.get("TQP_POOL_MAXPAR", "20"))
MAXTRIES = int(os.environ.get("TQP_MAXTRIES", "12"))
WEDGE_S = int(
    os.environ.get("TQP_WEDGE_S", str(16 * 3600))
)  # 100B ref max was 11.4 h at nq=500
PEND_S = int(os.environ.get("TQP_PEND_S", "2700"))
POLL_S = 60
STATE_PATH = os.environ.get("TQP_POST_STATE", "/home/claude/tqp_fleet/post_state.json")
SCORE_LOG = os.environ.get("TQP_SCORE_LOG", "/home/claude/tqp_fleet/score_1T.log")
QUERY_ENV = {"TQP_QUERY_SHARDS": "0,50000,100000,150000", "TQP_QUERIES_PER_SHARD": "25"}
LABELS = {"app": "tqp-fleet", "atlas.io/batch": "tqp-1t"}
IMAGE = "python:3.12"

SETUP = (
    "set -euo pipefail\n"
    "export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1\n"
    "pip install -q --no-cache-dir numpy\n"
)
CLONE = (
    "git clone -q --depth 1 https://github.com/ahb-sjsu/turboquant-pro.git /repo\n"
    "git -C /repo log -1 --format='repo %H'\n"
    "export PYTHONPATH=/repo:/work\n"
)

SHARED = [
    Volume(name="shared", mount_path="/shared", claim_name="tqp-fleet-shared"),
    Volume(name="code", mount_path="/work", config_map="tqp-fleet-code"),
]


def log(msg: str) -> None:
    print(f"=== {time.strftime('%H:%M', time.gmtime())} {msg}", flush=True)


def kubectl_json(*args: str):
    r = subprocess.run(
        ["kubectl", "-n", NS, *args, "-o", "json"], capture_output=True, text=True
    )
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def idx_volume(sid: int) -> Volume:
    return Volume(
        name="idx", mount_path="/idx", claim_name=f"tqp-fleet-1t-{sid}", read_only=True
    )


def desc_qcache(_sid: int) -> JobDescriptor:
    return JobDescriptor(
        name="aqx-qcache1t",
        image=IMAGE,
        command=[
            "/bin/bash",
            "-lc",
            SETUP + "export PYTHONPATH=/work\npython /work/fleet_qcache.py\n",
        ],
        env={"TQP_QCACHE_NAME": "queries1t.npy", **QUERY_ENV},
        resources=Resources(cpu="4", memory="8Gi", ephemeral_storage="2Gi"),
        labels=LABELS,
        backoff_limit=0,
        volumes=list(SHARED),
    )


def desc_ref(sid: int) -> JobDescriptor:
    return JobDescriptor(
        name=f"aqx-ref1t-{sid}",
        image=IMAGE,
        command=["/bin/bash", "-lc", SETUP + CLONE + "python /work/fleet_ref.py\n"],
        env={"TQP_SERVER_ID": str(sid), "TQP_RUN_TAG": "1t", **QUERY_ENV},
        resources=Resources(cpu="6", memory="8Gi", ephemeral_storage="5Gi"),
        labels=LABELS,
        backoff_limit=0,
        volumes=[idx_volume(sid), *SHARED],
    )


def desc_ivf(sid: int) -> JobDescriptor:
    return JobDescriptor(
        name=f"aqx-ivf1t-{sid}",
        image=IMAGE,
        command=["/bin/bash", "-lc", SETUP + CLONE + "python /work/fleet_ivf.py\n"],
        env={
            "TQP_SERVER_ID": str(sid),
            "TQP_RUN_TAG": "1t",
            "TQP_QCACHE_NAME": "queries1t.npy",
            **QUERY_ENV,
        },
        resources=Resources(cpu="6", memory="8Gi", ephemeral_storage="5Gi"),
        labels=LABELS,
        backoff_limit=0,
        volumes=[idx_volume(sid), *SHARED],
    )


def desc_score(_sid: int) -> JobDescriptor:
    return JobDescriptor(
        name="aqx-score1t",
        image=IMAGE,
        command=["/bin/bash", "-lc", SETUP + CLONE + "python /work/fleet_score10.py\n"],
        env={
            "TQP_RUN_TAG": "1t",
            "TQP_N_SERVERS": str(N_SERVERS),
            "TQP_N_ROWS": str(N_SERVERS * 2_000_000_000),
        },
        resources=Resources(cpu="2", memory="4Gi", ephemeral_storage="2Gi"),
        labels=LABELS,
        backoff_limit=0,
        volumes=list(SHARED),
    )


PHASES = [
    ("qcache", desc_qcache, [0]),
    ("ref", desc_ref, list(range(N_SERVERS))),
    ("ivf", desc_ivf, list(range(N_SERVERS))),
    ("score", desc_score, [0]),
]


class Pool:
    def __init__(self, phase: str, make, ids: list[int]):
        self.phase, self.make = phase, make
        self.pool = list(ids)
        self.done: set[int] = set()
        self.parked: set[int] = set()
        self.active: dict[int, dict] = {}
        self._load()

    def _state(self) -> dict:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _load(self) -> None:
        st = self._state().get(self.phase, {})
        self.done = set(st.get("done", []))
        self.parked = set(st.get("parked", []))
        self.pool = [s for s in self.pool if s not in self.done]
        log(
            f"{self.phase}: state loaded, {len(self.done)} done, {len(self.parked)} parked, {len(self.pool)} to go"
        )

    def _save(self) -> None:
        st = self._state()
        st[self.phase] = {"done": sorted(self.done), "parked": sorted(self.parked)}
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f)
        os.replace(tmp, STATE_PATH)

    def name(self, sid: int) -> str:
        return self.make(sid).name

    def _submit(self, sid: int, tries: int) -> None:
        with Client() as client:
            res = client.submit(self.make(sid))
        self.active[sid] = {
            "t0": time.time(),
            "tries": tries,
            "notfound": 0,
            "pendfails": self.active.get(sid, {}).get("pendfails", 0),
        }
        log(f"SUBMIT {self.name(sid)} try {tries}/{MAXTRIES} job_id={res.job_id}")

    def _delete_job(self, sid: int) -> None:
        subprocess.run(
            [
                "kubectl",
                "-n",
                NS,
                "delete",
                "job",
                self.name(sid),
                "--ignore-not-found",
                "--wait=false",
            ],
            capture_output=True,
        )

    def _recycle(self, sid: int, why: str) -> None:
        st = self.active[sid]
        if st["tries"] >= MAXTRIES:
            log(f"GAVE UP {self.name(sid)} after {MAXTRIES} tries")
            self.parked.add(sid)
            del self.active[sid]
            self._save()
            return
        log(f"RECYCLE {self.name(sid)}: {why}")
        if why == "job failed":
            # Keep the evidence before the job is deleted: the pod's exit reason and its last lines.
            pods = kubectl_json("get", "pods", "-l", f"job-name={self.name(sid)}") or {
                "items": []
            }
            for p in pods["items"]:
                for cs in p["status"].get("containerStatuses", []):
                    t = (
                        cs.get("state", {}).get("terminated")
                        or cs.get("lastState", {}).get("terminated")
                        or {}
                    )
                    log(
                        f"  {p['metadata']['name']} exit={t.get('exitCode')} reason={t.get('reason')}"
                    )
                r = subprocess.run(
                    ["kubectl", "-n", NS, "logs", p["metadata"]["name"], "--tail=3"],
                    capture_output=True,
                    text=True,
                )
                for line in r.stdout.strip().splitlines()[-3:]:
                    log(f"  | {line[:160]}")
        self._delete_job(sid)
        for _ in range(20):
            time.sleep(15)
            pods = kubectl_json("get", "pods", "-l", f"job-name={self.name(sid)}")
            if pods is not None and not pods.get("items"):
                break
        self._submit(sid, st["tries"] + 1)

    def poll(self) -> None:
        jobs = kubectl_json("get", "jobs", "-l", "app=tqp-fleet")
        if jobs is None:
            log("NOTE kubectl unclear, skipping cycle")
            return
        by_name = {j["metadata"]["name"]: j for j in jobs.get("items", [])}
        for sid in list(self.active):
            name, st = self.name(sid), self.active[sid]
            j = by_name.get(name)
            if j is None:
                st["notfound"] += 1
                if st["notfound"] >= 2:
                    self._recycle(sid, "job vanished (confirmed)")
                continue
            st["notfound"] = 0
            status = j.get("status", {})
            if status.get("succeeded"):
                log(f"DONE {name}")
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
                    if st["pendfails"] >= 3:
                        log(f"STUCK {name}: three no-Running recycles, parked")
                        self.parked.add(sid)
                        self._delete_job(sid)
                        del self.active[sid]
                        self._save()
                    else:
                        self._recycle(sid, "no Running pod after 45m")
        while len(self.active) < MAXPAR and self.pool:
            sid = self.pool.pop(0)
            j = by_name.get(self.name(sid))
            if j and j.get("status", {}).get("succeeded"):
                log(f"DONE {self.name(sid)} (pre-existing)")
                self.done.add(sid)
                self._save()
                continue
            if j:
                self.active[sid] = {"t0": time.time(), "tries": 1, "notfound": 0}
                log(f"ADOPT {self.name(sid)}")
                continue
            self._submit(sid, 1)

    def run(self) -> bool:
        while self.pool or self.active:
            self.poll()
            time.sleep(POLL_S)
        log(
            f"{self.phase.upper()}_DONE done={len(self.done)} parked={sorted(self.parked)}"
        )
        return not self.parked


def main() -> None:
    only = os.environ.get("TQP_PHASES", "qcache,ref,ivf,score").split(",")
    log(f"POST start: phases {only}, servers {N_SERVERS}, MAXPAR={MAXPAR}, nq=100")
    for phase, make, ids in PHASES:
        if phase not in only:
            continue
        if not Pool(phase, make, ids).run():
            log(f"POST stopped: {phase} left servers parked")
            sys.exit(1)
    r = subprocess.run(
        ["kubectl", "-n", NS, "logs", "job/aqx-score1t", "--tail=40"],
        capture_output=True,
        text=True,
    )
    with open(SCORE_LOG, "w", encoding="utf-8") as f:
        f.write(r.stdout)
    print(r.stdout)
    log("DRIVER1T_POST_DONE")


if __name__ == "__main__":
    main()
