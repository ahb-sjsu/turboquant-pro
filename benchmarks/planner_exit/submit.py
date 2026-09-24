"""Submit the planner exit test's 42 runs to NRP (run on Atlas). Operations only: nothing here
changes what a run computes (docs/PREREG_planner_exit.md section 5).

    python -m planner_exit.submit --commit <sha> --phase code
    python -m planner_exit.submit --commit <sha> --phase calibrate   # one run per arm, under the guard
    python -m planner_exit.submit --commit <sha> --phase run [--maxpar N]
    ... --dry-run                                                     # descriptors and preflight only

The same discipline as the observer-advantage submitter
(``observer_advantage/submit.py``), which encodes the NRP rules:

  code       one exempt pod stages the pinned commit's packages as ONE tar on the volume;
  calibrate  the first run of each arm (its sizing class), sent only while this campaign's
             utilization guard has a fresh heartbeat; the guard records measured usage;
  run        every other run, sized by ``nrp.sizing.request_for`` from its class's MEASURED
             usage, with the calibration request as the memory floor. A class with no
             measurement is refused, never modelled.

Pods read the campaign's corpora from the volume (``/data``) and write plan records to
``/data/pe/records``. They never read the campaign's results (``/data/results``), which stay
sealed until the records are committed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, "/home/claude/src/nats-bursting/python")
sys.path.insert(
    0, os.environ.get("OVB_PATH", "/archive/ahb-sjsu/tqp_rabitq_public/ovb")
)

from nrp import sizing as nrp_sizing  # noqa: E402
from rabitq_public.datasets import SPECS  # noqa: E402
from rabitq_public.grid import DIMS, ROWS  # noqa: E402

from planner_exit.run import N_PLAN, runs  # noqa: E402

NS = "ssu-atlas-ai"
PVC = "tqp-rbq-data"
APP = "tqp-pe"
BATCH = "tqp-planner-exit"
ZONE = {"topology.kubernetes.io/zone": "ucsd-nrp"}
REPO = "https://github.com/ahb-sjsu/turboquant-pro"
PACKAGES = (
    "turboquant_pro",
    "benchmarks/nrp",
    "benchmarks/rabitq_public",
    "benchmarks/planner_exit",
)
STATE_DIR = "/archive/ahb-sjsu/tqp_planner_exit/pool"
OBSERVATIONS = os.path.join(STATE_DIR, "observations.json")
GUARD_HEARTBEAT = os.path.join(STATE_DIR, "utilization_guard.heartbeat")
GUARD_MAX_AGE_S = 300
WANT_CPU = 2  # the planner is mostly one numpy thread; 2 keeps it above the 20% floor
OUT = "/data/pe/records"


def jobs() -> list:
    """The 42 registered runs, calibration order first: per arm, B1 then the rest."""
    return [
        dict(arm=arm, run=run, job_id=f"{arm}-{run}")
        for arm in DIMS
        for run in runs(DIMS[arm])
    ]


def code_tar(commit):
    return f"/data/pe/code/{commit}.tar"


def preamble(commit, cpu):
    return f"""set -euo pipefail
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS={cpu} OMP_NUM_THREADS={cpu}
export TQP_COMMIT={commit}
tar -xf /data/env/env.tar -C /tmp venv
mkdir -p /tmp/code && tar -xf {code_tar(commit)} -C /tmp/code
export PATH=/tmp/venv/bin:$PATH PYTHONPATH=/tmp/code:/tmp/code/benchmarks
"""


def job_name(job, role="run"):
    """Calibration and run Jobs get different names (a completed Job's name is not reusable)."""
    prefix = "pe-c-" if role == "calibrate" else "pe-"
    return prefix + re.sub(r"[^a-z0-9-]", "-", job["job_id"].lower())


def size_class(job):
    return job["arm"]


def model_gib(job):
    """First-run memory request for a calibration pod: an in-RAM corpus (the hdf5 arms load
    it whole), the 100k-row artifact and about five working copies of it (calibration rows,
    reconstructions, the fresh sample, a codec's training copy), and 1.5 GiB for the
    interpreter, faiss and score blocks. A model, used only for one calibration pod per arm.
    """
    d = DIMS[job["arm"]]
    corpus = ROWS[job["arm"]] * d * 4 if SPECS[job["arm"]].kind == "hdf5" else 0
    return (corpus + 6 * N_PLAN * d * 4) / 2**30 + 1.5


def calibrated():
    try:
        with open(os.path.join(STATE_DIR, "pe-calibrate.json"), encoding="utf-8") as f:
            return set(json.load(f).get("done", []))
    except (OSError, ValueError):
        return set()


def calibration_jobs_on_cluster():
    import subprocess

    r = subprocess.run(
        [
            "kubectl",
            "-n",
            NS,
            "get",
            "jobs",
            "-l",
            f"app={APP},atlas.io/role=calibrate",
            "-o",
            "jsonpath={.items[*].metadata.name}",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if r.returncode != 0:
        raise SystemExit(f"cannot list calibration Jobs: {r.stderr.strip()}")
    return set(r.stdout.split())


def calibration_jobs():
    done = calibrated()
    seen, out = set(), []
    for j in jobs():
        c = size_class(j)
        if c not in seen and job_name(j, "calibrate") not in done:
            seen.add(c)
            out.append(j)
    return out


def class_usage(cls):
    """Strictest measured usage in the class: highest peak against the lowest means."""
    try:
        with open(OBSERVATIONS, encoding="utf-8") as fh:
            obs = json.load(fh)
    except (OSError, ValueError):
        return None
    members = {
        job_name(j, role)
        for j in jobs()
        if size_class(j) == cls
        for role in ("calibrate", "run")
    }
    rows = [
        o
        for n, o in obs.items()
        if n in members and o.get("mean_cpu_cores") and o.get("mean_mem_gib")
    ]
    if not rows:
        return None
    return nrp_sizing.Usage(
        mean_cpu_cores=min(o["mean_cpu_cores"] for o in rows),
        mean_mem_gib=min(o["mean_mem_gib"] for o in rows),
        peak_mem_gib=max(o.get("peak_mem_gib") or o["mean_mem_gib"] for o in rows),
    )


def exempt_sized(cls):
    members = [j for j in jobs() if size_class(j) == cls]
    return bool(members) and model_gib(members[0]) <= nrp_sizing.EXEMPT_MEM_GIB


def exempt_class_proven(cls):
    members = [j for j in jobs() if size_class(j) == cls]
    if not members or model_gib(members[0]) > nrp_sizing.EXEMPT_MEM_GIB:
        return False
    return any(job_name(j, "calibrate") in calibrated() for j in members)


def guard_is_running():
    try:
        return time.time() - os.path.getmtime(GUARD_HEARTBEAT) < GUARD_MAX_AGE_S
    except OSError:
        return False


def descriptor(name, script, cpu, mem_gib, role):
    from nats_bursting import JobDescriptor, Resources, Volume

    return JobDescriptor(
        name=name,
        image="python:3.12",
        command=["/bin/bash", "-lc", script],
        resources=Resources(
            cpu=str(cpu), memory=f"{mem_gib}Gi", ephemeral_storage="6Gi"
        ),
        labels={"app": APP, "atlas.io/batch": BATCH, "atlas.io/role": role},
        node_selector=dict(ZONE),
        backoff_limit=0,
        volumes=[Volume(name="data", mount_path="/data", claim_name=PVC)],
    )


def code_descriptor(commit):
    tar = code_tar(commit)
    script = f"""set -euo pipefail
ls -la /data/env/env.tar
if [ -f {tar} ]; then echo "already staged: {tar}"; exit 0; fi
git clone -q --filter=blob:none --no-checkout {REPO} /tmp/src
git -C /tmp/src sparse-checkout set --no-cone {" ".join(PACKAGES)}
git -C /tmp/src checkout -q {commit}
test "$(git -C /tmp/src rev-parse HEAD)" = "{commit}"
mkdir -p /data/pe/code {OUT}
tar -cf {tar}.tmp.$$ -C /tmp/src {" ".join(PACKAGES)}
mv {tar}.tmp.$$ {tar}
echo "staged {tar}"
"""
    return descriptor(f"pe-code-{commit[:12]}", script, 1, 2, "code")


def run_descriptor(commit, job, cpu, mem_gib, role):
    script = preamble(commit, cpu) + (
        f"python -m planner_exit.run --arm {job['arm']} --run {job['run']} "
        f"--data-root /data --out {OUT}\n"
    )
    return descriptor(job_name(job, role), script, cpu, mem_gib, role)


def plan(commit, phase):
    """[(descriptor, why)] for a phase, or SystemExit naming every vetoed job."""
    if phase == "code":
        return [(code_descriptor(commit), "exempt class (1 CPU, 2 GiB)")]
    out, vetoes = [], []
    if phase == "calibrate":
        todo = calibration_jobs()
    else:
        taken = calibrated() | calibration_jobs_on_cluster()
        todo = [j for j in jobs() if job_name(j, "calibrate") not in taken]
    for j in todo:
        cls = size_class(j)
        usage = class_usage(cls)
        if phase == "calibrate":
            if usage is not None or exempt_class_proven(cls):
                continue
            if not guard_is_running():
                vetoes.append(
                    f"{job_name(j, phase)}: unmeasured, and no fresh guard heartbeat"
                )
                continue
            gib = model_gib(j)
            if gib <= nrp_sizing.EXEMPT_MEM_GIB:
                cpu, mem, why = 1, 2, f"calibrates {cls} (exempt, model {gib:.1f} GiB)"
            else:
                cpu, mem = WANT_CPU, int(gib + 0.999)
                why = f"calibrates {cls} (model {gib:.1f} GiB, under the guard)"
            out.append((run_descriptor(commit, j, cpu, mem, "calibrate"), why))
            continue
        if exempt_sized(cls) and (usage is not None or exempt_class_proven(cls)):
            why = f"exempt class: {cls} was calibrated at 1 CPU / 2 GiB"
            out.append((run_descriptor(commit, j, 1, 2, "run"), why))
            continue
        req = nrp_sizing.request_for(usage, WANT_CPU)
        if isinstance(req, nrp_sizing.Refusal):
            vetoes.append(f"{job_name(j)} ({cls}): {req}")
            continue
        cpu, mem = req.cpu, req.memory_gib
        if not req.exempt:
            # The calibration request is the memory evidence that held; sampled peaks miss spikes.
            mem = max(mem, int(model_gib(j) + 0.999))
            bad = nrp_sizing.check(cpu, mem, usage)
            if bad:
                vetoes.append(f"{job_name(j)} ({cls}): at {mem}Gi " + "; ".join(bad))
                continue
        if req.exempt:
            cpu, mem = 1, int(nrp_sizing.EXEMPT_MEM_GIB)
        out.append((run_descriptor(commit, j, cpu, mem, "run"), str(req)))
    if vetoes:
        raise SystemExit("PREFLIGHT VETO\n  " + "\n  ".join(vetoes))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--phase", required=True, choices=("code", "calibrate", "run"))
    ap.add_argument("--maxpar", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        raise SystemExit("--commit must be a full 40-hex sha, so the run is pinned")
    planned = plan(a.commit, a.phase)
    for d, why in planned:
        print(d.name, d.resources.cpu, d.resources.memory, "|", why, flush=True)
    print(f"{len(planned)} job(s), phase {a.phase}", flush=True)
    if a.dry_run:
        return
    from nats_bursting import Client
    from openvector_bench.nrp_pool import PoolRunner

    built = {d.name: d for d, _ in planned}

    def submit(item):
        with Client() as client:
            return client.submit(built[item["name"]])

    os.makedirs(STATE_DIR, exist_ok=True)
    runner = PoolRunner(
        [dict(name=n) for n in built],
        job_name=lambda it: it["name"],
        submit=submit,
        state_path=os.path.join(STATE_DIR, f"pe-{a.phase}.json"),
        ns=NS,
        maxpar=1 if a.phase == "code" else a.maxpar,
        maxtries=3,
        wedge_s=12 * 3600,
        pend_s=2700,
    )
    print("PHASE_OK" if runner.run() else "PHASE_PARKED", flush=True)


if __name__ == "__main__":
    main()
