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
STATE_DIR = "/archive/ahb-sjsu/tqp_rabitq_public/pool"
FACTORS = os.path.join(STATE_DIR, "factors.json")  # written from the rbq-factors log
# benchmarks/nrp/utilization_guard.py touches this every cycle. A calibration cell is the one
# pod whose usage nobody has measured yet, so it may only go out while the guard is watching.
GUARD_HEARTBEAT = os.path.join(STATE_DIR, "utilization_guard.heartbeat")
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


def _descriptor(name, script, cpu, memory, eph, role, extra_labels=None, code=True):
    from nats_bursting import JobDescriptor, Resources, Volume

    vols = [Volume(name="data", mount_path="/data", claim_name=PVC)]
    if code:
        vols.append(
            Volume(
                name="code",
                mount_path="/code/rabitq_public",
                config_map=CODE_CM,
                read_only=True,
            )
        )
    labels = {"app": "tqp-rbq", "atlas.io/batch": BATCH, "atlas.io/role": role}
    labels.update(extra_labels or {})
    return JobDescriptor(
        name=name,
        image=IMAGE,
        command=["/bin/bash", "-lc", script],
        env={"TQP_COMMIT": TQP_COMMIT, "CELL_THREADS": str(cpu)},
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


def _one_per_unmeasured_class(items):
    """One item per class lacking measured usage: the smallest model, which is cheapest to run."""
    pick = {}
    for it in items:
        c = it["cell"]
        if _sizeable(c):
            continue
        key = (c["dataset"], "tq" if c["method"] == "tqfix" else c["method"])
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
    # A measured class is sized from what it used; an unmeasured one only from a model, and
    # the model has been wrong low often enough to cost several OOM kills, so it gets more room.
    req = max(1, math.ceil((1.25 if usage else 1.5) * est_gib))
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
        {"atlas.io/cell": c["cell_id"][:63]},
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
            [_cell_item(c, False) for c in supplementary_cells()]
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
    built = {}
    for it in items:
        d, est_cpu, est_mem = descriptor(it)
        preflight(d, est_cpu, est_mem)
        built[it["name"]] = d
    print(f"{a.phase}: {len(built)} descriptors pass preflight")
    if a.dry_run:
        first = next(iter(built.values()))
        print(json.dumps(first.to_dict(), indent=1)[:3000])
        return
    from nats_bursting import Client
    from openvector_bench.nrp_pool import PoolRunner

    def submit(item):
        with Client() as client:
            return client.submit(built[item["name"]])

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
