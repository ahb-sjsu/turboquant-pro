"""Drive the RaBitQ public comparison on NRP: setup, staging, ground truth, then the cell pool.

Runs on Atlas (needs the nats-bursting checkout and a working kubeconfig):

    python submit_pool.py --phase setup|stage|gt|calibration|cells [--datasets ...] [--maxpar N] [--dry-run]

Every Job goes through the resident nats-bursting controller (``burst.submit``), and every
phase is an openvector-bench ``PoolRunner`` pool, so a stuck or preempted pod recycles one
slot without blocking the rest. Results are content-addressed: a cell whose JSON exists is
skipped inside the pod.

Platform rules enforced here as code, not memory (reference_nrp_job_policies.md):

- ``preflight()`` vetoes any descriptor whose measured usage would sit outside 20-200% of
  requested CPU or 20-150% of requested memory; requests <= 1 CPU and <= 2 GiB are exempt.
  Its CPU estimate is the class's *measured* mean cores (cell.py's meter, carried in
  factors.json). It used to be 0.8 x the request, a fabricated number that could not fail,
  and on 2026-09-15 eight cells ran at 1-4% of 4 CPUs with every preflight green.
- Memory request = 1.25 x the estimated peak anonymous memory (``footprints.py``). The
  ``calibration`` phase runs one registered cell per (arm, method) sized by the model; the
  ``cells`` phase vetoes any cell whose class has no measured calibration result.
- Jobs terminate by themselves, requests == limits (the renderer), ephemeral storage is
  declared, and every pod is pinned to zone ucsd-nrp, near the Ceph volume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, "/home/claude/src/nats-bursting/python")
sys.path.insert(
    0, os.environ.get("OVB_PATH", "/archive/ahb-sjsu/tqp_rabitq_public/ovb")
)  # openvector_bench.nrp_pool

from nrp import sizing as nrp_sizing  # noqa: E402

from rabitq_public import footprints  # noqa: E402
from rabitq_public.grid import cells, supplementary_cells  # noqa: E402

NS = "ssu-atlas-ai"
PVC = "tqp-rbq-data"
CODE_CM = "tqp-rbq-code"
IMAGE = "python:3.12"
BATCH = "tqp-rbq-public"
# ucsd-nrp: same campus as the SDSC Ceph; ucsd-suncave (15 GPU nodes, 180 CPUs) left 4-CPU
# cells pending for 40+ minutes on 2026-09-15
ZONE = {"topology.kubernetes.io/zone": "ucsd-nrp"}
TQP_COMMIT = (
    "856c4cbbde960d05b15c63a6e1f2de2a97e0c810"  # full sha: GitHub fetch needs it
)
# Amendment 3: the residual-coded IVF (turboquant_pro.ivf on the v3 scan kernel) at the
# commit that carries it, and the code map holding the harness with method tq_ivf. The
# registered cells keep TQP_COMMIT and CODE_CM; nothing they mount changes.
A3_COMMIT = "5d06ae857682a75d4784d8fd0902d7ebd5ebf115"
CODE_CM_A3 = "tqp-rbq-code-a3"
STATE_DIR = "/archive/ahb-sjsu/tqp_rabitq_public/pool"
FACTORS = os.path.join(STATE_DIR, "factors.json")  # written from the rbq-factors log
# benchmarks/nrp/utilization_guard.py touches this every cycle. A calibration cell is the one
# pod whose usage nobody has measured yet, so it may only go out while the guard is watching.
GUARD_HEARTBEAT = os.path.join(STATE_DIR, "utilization_guard.heartbeat")
OOM_FILE = os.path.join(STATE_DIR, "oom_kills.json")  # written by the guard
OBSERVATIONS = os.path.join(STATE_DIR, "observations.json")  # also the guard's
DONE_CELLS = os.path.join(
    STATE_DIR, "done_cells.txt"
)  # cell ids with a result on the volume
GUARD_MAX_AGE_S = 300

ENV_PREAMBLE = """set -euo pipefail
export PYTHONUNBUFFERED=1 PIP_ROOT_USER_ACTION=ignore HF_HOME=/tmp/hf
tar -xf /data/env/env.tar -C /tmp
export PATH=/tmp/venv/bin:$PATH PYTHONPATH=/code:/tmp/tqp
"""

SETUP_SCRIPT = f"""set -euo pipefail
export PIP_ROOT_USER_ACTION=ignore PYTHONUNBUFFERED=1
if [ -f /data/env/env.tar ]; then echo "env exists"; exit 0; fi
python -m venv /tmp/venv
/tmp/venv/bin/pip install -q --no-cache-dir faiss-cpu==1.15.0 rabitqlib==0.3.4 h5py pyarrow huggingface_hub pybind11
mkdir -p /tmp/src && cd /tmp/src && git init -q && git remote add origin https://github.com/ahb-sjsu/turboquant-pro
git fetch -q --depth 1 origin {TQP_COMMIT} && git checkout -q FETCH_HEAD
mkdir -p /tmp/tqp && cp -r /tmp/src/turboquant_pro /tmp/tqp/
/tmp/venv/bin/pip freeze > /tmp/freeze.txt
mkdir -p /data/env && tar -cf /data/env/env.tar.tmp -C /tmp venv tqp freeze.txt
mv /data/env/env.tar.tmp /data/env/env.tar && cp /tmp/freeze.txt /data/env/freeze.txt
echo SETUP_DONE
"""


def mem_gib(q: str) -> float:
    q = str(q)
    return float(q[:-2]) / 1024 if q.endswith("Mi") else float(q.rstrip("Gi"))


OOM_RECORD_MAX_AGE_S = 3 * 3600


def _oom_kills():
    """What the guard saw die recently, and at what request.

    Records expire: a kill says what the code of the time needed, and a fix that bounds a
    transient makes it wrong in the expensive direction. One cell compounded 1.5x per kill to
    a 48 GiB request and then ran at 3 GiB, where it was stopped for using 5% of it.
    """
    try:
        with open(OOM_FILE, encoding="utf-8") as fh:
            all_oom = json.load(fh)
    except (OSError, ValueError):
        return {}
    now = time.time()
    fresh = {}
    for job, rec in all_oom.items():
        stamp = rec.get("updated")
        try:
            age = now - time.mktime(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
        except (TypeError, ValueError):
            age = 0
        if age < OOM_RECORD_MAX_AGE_S:
            fresh[job] = rec
    return fresh


def _observed(name):
    """What the guard measured for this exact cell last time it ran, or None.

    The guard samples what the cluster judges, so this beats anything scaled from another
    cell of the class: classes metered before the meter recorded working sets carry only a
    memory.current mean, which reads high by however much reclaimable cache they held.
    """
    try:
        with open(OBSERVATIONS, encoding="utf-8") as fh:
            o = json.load(fh).get(name)
    except (OSError, ValueError):
        return None
    if not o or not o.get("mean_cpu_cores") or not o.get("mean_mem_gib"):
        return None
    return o


def guard_is_running() -> bool:
    """True when the utilization guard has checked in within the last GUARD_MAX_AGE_S."""
    try:
        return time.time() - os.path.getmtime(GUARD_HEARTBEAT) < GUARD_MAX_AGE_S
    except OSError:
        return False


def preflight(d, est_cpu: float, est_mem_gib: float) -> None:
    """Veto a descriptor the cluster would flag. The rules live in benchmarks/nrp/sizing.py.

    ``est_cpu`` and ``est_mem_gib`` are what this pod is expected to *average*, so they must
    come from a measurement; a cell whose class has none never reaches here.
    """
    cpu = float(d.resources.cpu)
    mem = mem_gib(d.resources.memory)
    if d.resources.gpu:
        raise SystemExit(f"PREFLIGHT VETO {d.name}: this campaign requests no GPUs")
    usage = nrp_sizing.Usage(
        mean_cpu_cores=est_cpu, mean_mem_gib=est_mem_gib, peak_mem_gib=est_mem_gib
    )
    problems = nrp_sizing.check(cpu, mem, usage)
    if not d.resources.ephemeral_storage:
        problems.append("ephemeral-storage not declared")
    if problems:
        raise SystemExit(f"PREFLIGHT VETO {d.name}: " + "; ".join(problems))


def _descriptor(
    name, script, cpu, memory, eph, role, extra_labels=None, code=True, code_map=CODE_CM
):
    from nats_bursting import JobDescriptor, Resources, Volume

    vols = [Volume(name="data", mount_path="/data", claim_name=PVC)]
    if code:
        vols.append(
            Volume(
                name="code",
                mount_path="/code/rabitq_public",
                config_map=code_map,
                read_only=True,
            )
        )
    labels = {"app": "tqp-rbq", "atlas.io/batch": BATCH, "atlas.io/role": role}
    labels.update(extra_labels or {})
    return JobDescriptor(
        name=name,
        image=IMAGE,
        command=["/bin/bash", "-lc", script],
        env={
            "TQP_COMMIT": TQP_COMMIT,
            "CELL_THREADS": str(cpu),
            # short enough that a line lands between the start of an allocation and the kill
            "TQP_USAGE_REPORT_S": "15",
        },
        resources=Resources(cpu=str(cpu), memory=memory, ephemeral_storage=eph),
        labels=labels,
        node_selector=dict(ZONE),
        backoff_limit=0,
        volumes=vols,
    )


def setup_items():
    return [dict(name="rbq-setup", kind="setup")]


def stage_items():
    return [
        dict(name=f"rbq-stage-{n}", kind="stage", only=n)
        for n in ("ann", "wiki1024", "dbpedia1536", "dbpedia3072")
    ]


def gt_items():
    items = [
        dict(name=f"rbq-gt-{d}", kind="gt", dataset=d)
        for d in ("wiki1024-10m", "dbpedia-ada002-1m", "dbpedia-3large-1536-1m")
    ]
    items += [
        dict(name=f"rbq-check-{d.split('-')[0]}", kind="check", dataset=d)
        for d in ("glove-100-angular", "nytimes-256-angular", "deep-image-96-angular")
    ]
    return items


def _sizeable(c):
    """True when this cell can be sized honestly: a measured class, or the exempt class."""
    if c["dataset"] in footprints.EXEMPT_ARMS:
        return True
    return (
        footprints.sizing(c, FACTORS) is not None
        and footprints.class_usage(c, FACTORS) is not None
    )


def _finished():
    """Cells with a result: ids listed from the volume, and job names every pool of this
    campaign recorded as done (any ``<tag>.json`` in the state dir). What has a result is
    recorded on the volume, not in one pool's state: results from earlier runs belong to
    no current pool."""
    finished_ids = set()
    try:
        with open(DONE_CELLS, encoding="utf-8") as fh:
            finished_ids = {ln.strip() for ln in fh if ln.strip()}
    except OSError:
        pass
    finished_jobs = set()
    try:
        tags = [f for f in os.listdir(STATE_DIR) if f.endswith(".json")]
    except OSError:
        tags = []
    for f in tags:
        try:
            with open(os.path.join(STATE_DIR, f), encoding="utf-8") as fh:
                state = json.load(fh)
            if isinstance(state, dict):
                finished_jobs |= set(state.get("done", []))
        except (OSError, ValueError):
            pass
    return finished_ids, finished_jobs


def _one_per_unmeasured_class(items):
    """One item per class lacking measured usage: the smallest model, which is cheapest to run.

    Cells that already have a result are skipped: such a cell returns in seconds without
    running, which is not a measurement, and choosing one leaves its class unmetered while the
    wave reports itself finished. Five classes ended that way.
    """
    # What has a result is recorded on the volume, not in a pool's state: results from earlier
    # runs of the campaign belong to no current pool, and picking one of those cells is how the
    # wave reported itself finished with classes still unmeasured. DONE_CELLS is refreshed from
    # the volume by the orchestrator.
    finished_ids, finished_jobs = _finished()
    pick = {}
    for it in items:
        c = it["cell"]
        if _sizeable(c) or it["name"] in finished_jobs or c["cell_id"] in finished_ids:
            continue
        key = (c["dataset"], footprints.sizing_class(c["method"]))
        size = footprints.model_bytes(c, 4)
        if key not in pick or size < pick[key][0]:
            pick[key] = (size, it)
    return [it for _size, it in sorted(pick.values(), key=lambda p: p[0])]


def _cell_item(c, calibrating):
    name = "rbq-c-" + hashlib.sha1(c["cell_id"].encode()).hexdigest()[:12]
    return dict(name=name, kind="cell", cell=c, calibrating=calibrating)


def cell_items(datasets=None, calibration=False):
    if calibration:
        return [
            _cell_item(c, True)
            for c in footprints.calibration_cells()
            if not datasets or c["dataset"] in datasets
        ]
    calib = {c["cell_id"] for c in footprints.calibration_cells()}
    return [_cell_item(c, False) for c in cells(datasets) if c["cell_id"] not in calib]


def descriptor(item):
    k = item["kind"]
    if k == "setup":
        d = _descriptor(
            item["name"], SETUP_SCRIPT, 1, "2Gi", "12Gi", "setup", code=False
        )
        return d, 1.0, 1.5
    if k == "stage":
        s = (
            ENV_PREAMBLE
            + f"python -m rabitq_public.stage --data-root /data --only {item['only']}\n"
        )
        return _descriptor(item["name"], s, 1, "2Gi", "30Gi", "stage"), 0.5, 1.5
    if k == "gt":
        s = (
            ENV_PREAMBLE
            + f"python -m rabitq_public.gt --dataset {item['dataset']} --data-root /data\n"
        )
        return _descriptor(item["name"], s, 1, "2Gi", "4Gi", "gt"), 1.0, 1.5
    if k == "factors":
        s = ENV_PREAMBLE + (
            "python -m rabitq_public.footprints --emit-factors --results /data/results\n"
        )
        return _descriptor(item["name"], s, 1, "1Gi", "4Gi", "factors"), 0.5, 0.5
    if k == "check":
        big = item["dataset"].startswith("deep")
        s = (
            ENV_PREAMBLE
            + f"python -m rabitq_public.gt --check {item['dataset']} --data-root /data\n"
        )
        mem = "6Gi" if big else "2Gi"
        return (
            _descriptor(item["name"], s, 1, mem, "4Gi", "check"),
            1.0,
            (4.6 if big else 1.2),
        )
    c = item["cell"]
    size = footprints.sizing(c, FACTORS, calibrating=item["calibrating"])
    if size is None:
        raise SystemExit(
            f"PREFLIGHT VETO {item['name']} ({c['cell_id']}): its class has no measured calibration cell yet"
        )
    cpu, est_gib, _source = size
    usage = footprints.class_usage(c, FACTORS)
    # The exempt arms run at 1 CPU / 2 GiB, where the cluster applies no floor, so they are
    # safe to submit without a measurement.
    exempt_arm = c["dataset"] in footprints.EXEMPT_ARMS
    metering = item.get("metering", False)
    if usage is None and not item["calibrating"] and not exempt_arm and not metering:
        raise SystemExit(
            f"PREFLIGHT VETO {item['name']} ({c['cell_id']}): its class has never been metered; "
            "run the calibration phase with the new cell.py first"
        )
    if usage is None and not exempt_arm and not guard_is_running():
        # includes the metering wave: measuring a class is exactly when the guard must watch
        raise SystemExit(
            f"PREFLIGHT VETO {item['name']} ({c['cell_id']}): a calibration cell measures a class "
            "nobody has measured, so it may only run while benchmarks/nrp/utilization_guard.py is "
            f"watching. No heartbeat newer than {GUARD_MAX_AGE_S}s at {GUARD_HEARTBEAT}"
        )
    # A quarter over the estimate, then, where the class has been metered, the window that
    # satisfies both ends at once: cover the peak, stay under mean / floor. Sizing from the
    # peak alone put a wiki cell at 10 GiB whose mean read 19% to the guard, and it was
    # stopped twenty-two minutes in, during its last phase.
    req = max(1, math.ceil(1.25 * est_gib))
    measured = footprints.scaled_usage(c, FACTORS, cpu)
    own = _observed(item["name"])
    if own:
        # This cell's own run, measured the way the cluster measures. The peak keeps whichever
        # figure is larger, since the guard sees working sets and the kernel kills on the
        # charged total, but never more than a few times what this cell was seen to peak at:
        # a class measured before the transients were bounded carries a peak that would ask
        # for 49 GiB on behalf of a cell that runs in 3.
        observed_peak = own.get("peak_mem_gib") or 0
        class_peak = measured[1] if measured else 0
        if observed_peak:
            class_peak = min(class_peak, observed_peak * 4)
        peak = max(observed_peak, class_peak)
        measured = (own["mean_mem_gib"], peak or own["mean_mem_gib"])
        usage = usage or {
            "mean_cpu_cores": own["mean_cpu_cores"],
            "mean_mem_gib": own["mean_mem_gib"],
            "peak_mem_gib": own.get("peak_mem_gib") or own["mean_mem_gib"],
        }
    if measured and usage:
        sized = nrp_sizing.request_for(
            nrp_sizing.Usage(
                own["mean_cpu_cores"] if own else usage["mean_cpu_cores"],
                measured[0],
                measured[1],
            ),
            cpu,
        )
        if isinstance(sized, nrp_sizing.Refusal):
            raise SystemExit(
                f"PREFLIGHT VETO {item['name']} ({c['cell_id']}): {sized.reason}"
            )
        req, cpu = sized.memory_gib, sized.cpu
    died_at = _oom_kills().get(item["name"], {}).get("killed_at_gib", 0)
    if own and own.get("peak_mem_gib"):
        # A kill cannot argue for more than a few times what the cell was last seen to peak at;
        # without this the bump compounds past anything the cell could fill.
        died_at = min(died_at, own["peak_mem_gib"] * 4)
    if died_at >= req:
        # A kill of this cell outranks any estimate, measured class or not: the class is
        # metered on its smallest configuration, and scaling that by the model missed what a
        # d'=1024 tq cell holds, so six of them died at 19 GiB and were resubmitted at 19 GiB.
        # This makes the record authoritative, which means clearing it after any change that
        # moves memory materially (pool/oom_kills.json).
        req = max(req, math.ceil(died_at * 1.5))
    if c["dataset"] in footprints.EXEMPT_ARMS:
        req, memory, est_gib = 2, "2Gi", min(est_gib, 1.9)  # exempt class: never swept
    else:
        memory = f"{req}Gi"
    kernel = ""
    if c["method"] == "tq":
        kernel = "python -m turboquant_pro._adc >/dev/null\n"
    elif c["method"] == "tqfix":  # Amendment 2: v2 kernel source ships in the code map
        kernel = (
            "cp /code/rabitq_public/adc_scan_v2.cpp"
            " /tmp/tqp/turboquant_pro/_adc/adc_scan.cpp\n"
            "python -m turboquant_pro._adc >/dev/null\n"
            'python -c "from turboquant_pro import _adc; assert _adc.is_available()"\n'
        )
    elif (
        c["method"] == "tq_ivf"
    ):  # Amendment 3: the whole package at the residual-IVF commit
        kernel = (
            "mkdir -p /tmp/src3 && cd /tmp/src3 && git init -q"
            " && git remote add origin https://github.com/ahb-sjsu/turboquant-pro"
            f" && git fetch -q --depth 1 origin {A3_COMMIT} && git checkout -q FETCH_HEAD"
            " && rm -rf /tmp/tqp/turboquant_pro && cp -r /tmp/src3/turboquant_pro /tmp/tqp/"
            " && cd /tmp\n"
            "python -m turboquant_pro._adc >/dev/null\n"
            'python -c "from turboquant_pro import _adc; assert _adc.load().VERSION == 3"\n'
        )
    # no background reporter loop: the cell JSON records its own peak anonymous memory
    s = (
        ENV_PREAMBLE
        + kernel
        + (
            f"python -m rabitq_public.cell --cell-id {c['cell_id']} --data-root /data --out /data/results --threads {cpu}\n"
        )
    )
    d = _descriptor(
        item["name"],
        s,
        cpu,
        memory,
        "6Gi",
        "cell",
        dict(
            {"atlas.io/cell": c["cell_id"][:63]},
            **({"atlas.io/meter": "true"} if metering else {}),
        ),
        code_map=CODE_CM_A3 if c["method"] == "tq_ivf" else CODE_CM,
    )
    ram = (
        c["method"] == "rabitqlib_ivf"
        and c["dataset"] in footprints.RAM_CORPUS_RABITQLIB
    )
    d.env["TQP_RBQ_SCRATCH"] = "ram" if ram else "/data/scratch"
    if usage is None:  # calibration: no measurement exists, the guard is the safety net
        return d, 0.8 * cpu, est_gib
    # What the cluster will average over this pod: the class's measured cores, and its
    # measured mean-to-peak memory ratio applied to this cell's estimated peak.
    ratio = min(1.0, usage["mean_mem_gib"] / max(usage["peak_mem_gib"], 1e-9))
    return d, float(usage["mean_cpu_cores"]), est_gib * ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        required=True,
        choices=(
            "setup",
            "stage",
            "gt",
            "calibration",
            "factors",
            "cells",
            "supplementary",
        ),
    )
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--maxpar", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--skip-unmeasured",
        action="store_true",
        help="cells phase: leave out (instead of vetoing) cells whose class is unmeasured",
    )
    ap.add_argument("--methods", nargs="*", help="cells phase: only these methods")
    ap.add_argument(
        "--meter-unmeasured",
        action="store_true",
        help="run one cell of every class that has no measured usage yet, cheapest first, "
        "sized from the model under the utilization guard: the wave that makes the rest of "
        "the grid sizeable from measurement",
    )
    ap.add_argument(
        "--bootstrap",
        action="store_true",
        help="measure a class by running real cells of it under the utilization guard: "
        "sizes from the model, as the calibration phase does, but from cells that have no "
        "result yet, since a finished cell returns without running and measures nothing",
    )
    ap.add_argument(
        "--tag", help="pool state name; disjoint pools may run side by side"
    )
    ap.add_argument(
        "--amendment",
        type=int,
        default=2,
        help="supplementary phase: which amendment's cells (2: tqfix, 3: tq_ivf)",
    )
    ap.add_argument(
        "--wedge-hours",
        type=float,
        default=20.0,
        help="recycle a job only after this long; a 10M-row flat RaBitQ cell takes ~7 h",
    )
    a = ap.parse_args()
    if a.phase == "factors":
        items = [dict(name="rbq-factors", kind="factors")]
    elif a.phase in ("setup", "stage", "gt"):
        items = dict(setup=setup_items, stage=stage_items, gt=gt_items)[a.phase]()
    else:
        items = (
            [_cell_item(c, False) for c in supplementary_cells(a.amendment)]
            if a.phase == "supplementary"
            else cell_items(a.datasets, calibration=a.phase == "calibration")
        )
        if a.methods:
            items = [it for it in items if it["cell"]["method"] in a.methods]
        if a.meter_unmeasured:
            items = _one_per_unmeasured_class(items)
            print(f"metering wave: {len(items)} classes have no measured usage")
            for it in items:
                # Sized by the model, deliberately. A class with no usage has only a memory
                # factor built from sampled anonymous RSS, and that missed the faiss training
                # spike badly enough to OOM four PQ cells; the model is the conservative
                # estimate, and a metering run is exactly when to be conservative.
                it["metering"] = True
                it["calibrating"] = True
        if a.bootstrap:
            for it in items:
                it["calibrating"] = True
        if a.skip_unmeasured and a.phase == "cells":
            keep = [it for it in items if _sizeable(it["cell"])]
            print(f"skipping {len(items) - len(keep)} cells whose class is unmeasured")
            items = keep
    if a.phase in ("cells", "supplementary"):
        # A rebuilt pool starts from an empty state; without this it resubmits every cell
        # that already has a result, and each such pod requests its measured size to exit
        # in seconds, which is the usage violation the sizing guard exists to prevent.
        finished_ids, finished_jobs = _finished()
        before = len(items)
        items = [
            it
            for it in items
            if it["cell"]["cell_id"] not in finished_ids
            and it["name"] not in finished_jobs
        ]
        print(
            f"skipping {before - len(items)} cells that already have a result",
            flush=True,
        )
    built, refused = {}, []
    for it in items:
        try:
            d, est_cpu, est_mem = descriptor(it)
            preflight(d, est_cpu, est_mem)
        except SystemExit as veto:
            # Park the cell, run the rest. A pool that dies on its first refusal leaves the
            # whole phase stopped over one cell, which is how the graded pool sat idle.
            refused.append(str(veto))
            continue
        built[it["name"]] = d
    for line in refused:
        print(f"SKIPPED {line}", flush=True)
    items = [it for it in items if it["name"] in built]
    print(f"{a.phase}: {len(built)} descriptors pass preflight")
    if a.dry_run:
        first = next(iter(built.values()))
        print(json.dumps(first.to_dict(), indent=1)[:3000])
        return
    from nats_bursting import Client
    from openvector_bench.nrp_pool import PoolRunner

    def submit(item):
        # Rebuilt per attempt, not taken from the startup pass: a kill recorded by the guard
        # changes this cell's request, and a pool that submits the descriptor it built at
        # start retries at the size that just killed the cell until its tries run out.
        try:
            d, est_cpu, est_mem = descriptor(item)
            preflight(d, est_cpu, est_mem)
        except SystemExit as veto:
            print(f"SKIPPED {veto}", flush=True)
            d = built[item["name"]]
        with Client() as client:
            return client.submit(d)

    os.makedirs(STATE_DIR, exist_ok=True)
    tag = a.tag or a.phase + ("-" + "-".join(a.datasets) if a.datasets else "")
    runner = PoolRunner(
        items,
        job_name=lambda it: it["name"],
        submit=submit,
        state_path=os.path.join(STATE_DIR, f"{tag}.json"),
        ns=NS,
        maxpar=a.maxpar,
        maxtries=4,
        wedge_s=int(a.wedge_hours * 3600),
        pend_s=2700,
    )
    ok = runner.run()
    if a.phase == "factors" and ok:
        import subprocess

        log = subprocess.run(
            ["kubectl", "-n", NS, "logs", "job/rbq-factors"],
            capture_output=True,
            text=True,
        ).stdout
        line = next(ln for ln in log.splitlines() if ln.startswith("FACTORS_JSON "))
        with open(FACTORS, "w", encoding="utf-8") as f:
            f.write(line[len("FACTORS_JSON ") :])
        subprocess.run(["kubectl", "-n", NS, "delete", "job", "rbq-factors"])
        print("factors written:", FACTORS, flush=True)
    print("PHASE_OK" if ok else "PHASE_PARKED", flush=True)


if __name__ == "__main__":
    main()
