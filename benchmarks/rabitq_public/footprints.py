"""Memory sizing for NRP cells: an explicit model, corrected by measurement.

NRP counts over-requesting as a violation (usage must sit at 20-150% of the memory request),
and a guessed request is how four CPU pods were OOM-killed on 2026-09-03. So requests come
from measurements. The model below only chooses the first request for one *calibration*
cell per (arm, method): the largest configuration, seed 0, a real registered cell. Its
measured peak anonymous memory sets a per-class correction factor, and every other cell in
the class is sized as model x factor. ``submit_pool.py`` refuses a non-calibration cell
whose class has no measurement.

    python -m rabitq_public.footprints --results /data/results    # print factors and requests

Model terms (bytes; N rows, d dim, T threads, L nlist, o PCA dim, cs code bytes):
corpus in RAM (hdf5 arms: 2 N d 4 while normalizing; npy arms: 3 blocks of 250k rows),
training samples and their float64 copies, the index arrays, per-block encode temporaries,
and search scratch (tq kernel: a re-blocked code copy plus N x 12 bytes per thread).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

from .grid import DIMS, ROWS, cells, supplementary_cells

GIB = 2**30
BASE = 0.7 * GIB
BLOCK = 250_000
HDF5 = ("glove-100-angular", "nytimes-256-angular", "deep-image-96-angular")
NQ = {
    "glove-100-angular": 2000,
    "nytimes-256-angular": 2000,
    "deep-image-96-angular": 2000,
}


def _corpus(ds):
    n, d = ROWS[ds], DIMS[ds]
    if ds in HDF5:
        return 2 * n * d * 4
    extra = (
        9 * n if ds == "wiki1024-10m" else 0
    )  # keep-map and mask for the held-out queries
    return 3 * BLOCK * d * 4 + extra


def _rabitq_code(d, bits):
    return math.ceil(d * bits / 8) + 12


def model_bytes(cell, threads, ram_corpus=None):
    ds, m = cell["dataset"], cell["method"]
    if ram_corpus is None:
        ram_corpus = ds in RAM_CORPUS_RABITQLIB
    n, d = ROWS[ds], DIMS[ds]
    total = BASE + _corpus(ds)
    pca_fit = 100_000 * d * 20 + d * d * 8
    if m in ("tq", "tqfix"):
        o = cell["out_dim"]
        total += pca_fit + n * (2 * o + 8) + threads * n * 12 + BLOCK * o * 16
    elif m == "rabitq_flat":
        total += 200_000 * d * 8 + n * _rabitq_code(d, cell["bits"]) + BLOCK * d * 8
    elif m == "rabitq_ivf":
        L = cell["nlist"]
        total += (
            40 * L * d * 8 + n * (_rabitq_code(d, cell["bits"]) + 8) + BLOCK * d * 8
        )
    elif m == "pca_rabitq_ivf":
        L, o = cell["nlist"], cell["out_dim"]
        total += (
            pca_fit
            + 40 * L * (d + o) * 4 * 2
            + n * (_rabitq_code(o, cell["bits"]) + 8)
            + BLOCK * (d + o) * 8
        )
    elif m == "rabitqlib_ivf":
        L = cell["nlist"]
        if ram_corpus:
            total += n * d * 4  # corpus held in RAM for the build (see cell.py)
        # corpus copy is memory-mapped (page cache); allow one internal float32 copy of the
        # largest cluster batch plus the codes
        total += (
            40 * L * d * 8 + n * 4 + n * _rabitq_code(d, cell["bits"]) + BLOCK * d * 8
        )
    elif m in ("pq", "opq"):
        total += 200_000 * d * 8 + n * cell["m"] + BLOCK * d * 8 + d * d * 8
    return total


# Arms whose every cell fits the exempt class (<= 1 CPU, <= 2 GiB). Measured 2026-09-15:
# a GloVe rabitq_flat cell peaked near 0.6 GiB, and four 4-CPU / 2.5-GiB GloVe and NYTimes
# calibration pods were deleted in one utilization sweep. Exempt pods are not swept.
EXEMPT_ARMS = ("glove-100-angular", "nytimes-256-angular")
# rabitqlib reads its build corpus in random order; memory-mapped over CephFS at 10M x 1024
# that stalled in disk sleep, so this arm keeps the corpus in RAM.
RAM_CORPUS_RABITQLIB = (
    "wiki1024-10m",
    "deep-image-96-angular",
    "dbpedia-ada002-1m",
    "dbpedia-3large-1536-1m",
)
# Classes whose calibration cell already held the corpus in RAM; the others were calibrated
# with the memory map, so the in-RAM corpus is added on top of their measured estimate.
# Deep-image joined on 2026-09-15 after four of its rabitqlib cells, writing 3.6 GiB maps to
# CephFS at once, sat below the CPU floor and were deleted together.
RAM_CORPUS_CALIBRATED = ("wiki1024-10m",)


def cpu_for(cell):
    """1 CPU (exempt class) for the small arms or when the model fits 2 GiB, else 4."""
    if cell["dataset"] in EXEMPT_ARMS:
        return 1
    return 1 if model_bytes(cell, 1) * 1.25 <= 2 * GIB else 4


def calibration_cells():
    """The largest-memory configuration of each (arm, method), seed 0."""
    best = {}
    for c in cells():
        if c["seed"] != 0:
            continue
        key = (c["dataset"], c["method"])
        if key not in best or model_bytes(c, 4) > model_bytes(best[key], 4):
            best[key] = c
    return list(best.values())


MIN_METERED_S = (
    120  # below this the cell did no real work (a finished cell returns at once)
)


def _real_usage(rec):
    """The cell's usage, or None if it never ran long enough to mean anything.

    A cell whose result already exists returns immediately, and its near-idle seconds would
    otherwise be recorded as the class's usage and shrink every later request to nothing.
    """
    u = rec.get("usage") or {}
    if not u.get("mean_cpu_cores") or not u.get("mean_mem_gib"):
        return None
    return u if (u.get("wall_s") or 0) >= MIN_METERED_S else None


def factors(results_dir):
    """Per-class measured/model ratio and measured usage, from the best finished cell.

    A cell that recorded usage is preferred, then the calibration cell, then the largest-model
    finished one: any real cell of a class meters it, and insisting on the calibration cell
    would mean re-running work already done, which measures nothing since a finished cell
    returns without running.

    ``usage`` is what the cluster averages over a pod's life (cell.py's meter). Without it a
    request is a guess: on 2026-09-15 the submitter's preflight passed every cell because it
    was handed a fabricated CPU estimate of 80% of the request, while the pods ran at 1-4%.
    """
    out = {}
    calib = {c["cell_id"]: c for c in calibration_cells()}
    known = {c["cell_id"]: c for c in list(cells()) + list(supplementary_cells())}
    best = {}
    for p in glob.glob(os.path.join(results_dir, "*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        cid = r["cell"]["cell_id"]
        c = calib.get(cid) or known.get(cid)
        if c is None or not r.get("peak_anon_gib"):
            continue
        key = f"{c['dataset']}/{c['method']}"
        # A cell that recorded usage outranks one that did not, whatever its role: the
        # calibration cells all finished before the meter existed, and preferring them by
        # role left every class unmeasured.
        rank = (
            bool(_real_usage(r)),
            cid in calib,
            model_bytes(c, r["threads"]),
        )
        if key not in best or rank > best[key][0]:
            best[key] = (rank, r, c)
    for key, (_rank, r, c) in best.items():
        u = _real_usage(r) or {}
        out[key] = dict(
            cell=c["cell_id"],
            measured_gib=r["peak_anon_gib"],
            model_gib=round(model_bytes(c, r["threads"]) / GIB, 3),
            factor=round(r["peak_anon_gib"] * GIB / model_bytes(c, r["threads"]), 3),
            usage=(
                dict(
                    mean_cpu_cores=u["mean_cpu_cores"],
                    mean_mem_gib=u["mean_mem_gib"],
                    peak_mem_gib=u["peak_mem_gib"],
                    mean_ws_gib=u.get("mean_ws_gib"),
                    peak_ws_gib=u.get("peak_ws_gib"),
                    threads=r["threads"],
                    wall_s=u.get("wall_s"),
                    phases=u.get("phases"),
                )
                if u.get("mean_cpu_cores") and u.get("mean_mem_gib")
                else None
            ),
        )
    return out


def class_usage(cell, factors_path):
    """The class's measured usage dict, or None when it has never been metered."""
    if not factors_path or not os.path.exists(factors_path):
        return None
    cls = "tq" if cell["method"] == "tqfix" else cell["method"]
    with open(factors_path, encoding="utf-8") as fh:
        f = json.load(fh).get(f"{cell['dataset']}/{cls}")
    return (f or {}).get("usage")


def scaled_usage(cell, factors_path, cpu):
    """The class's measured (mean, peak) GiB, scaled to this cell by the model's size ratio.

    The mean matters as much as the peak: a request must cover the peak or the pod dies, and
    stay under mean / floor or the cluster counts it as under-used. benchmarks/nrp/sizing.py
    turns the pair into a request, or refuses when no request satisfies both.
    """
    if not factors_path or not os.path.exists(factors_path):
        return None
    cls = "tq" if cell["method"] == "tqfix" else cell["method"]
    with open(factors_path, encoding="utf-8") as fh:
        f = json.load(fh).get(f"{cell['dataset']}/{cls}")
    u = (f or {}).get("usage")
    if not u or not u.get("peak_mem_gib") or not f.get("model_gib"):
        return None
    ratio = (model_bytes(cell, cpu) / GIB) / f["model_gib"]
    # The mean is the working set, because that is what the utilization sweep measures; the
    # peak stays the charged total, because that is what the kernel kills on.
    mean = u.get("mean_ws_gib") or u["mean_mem_gib"]
    return mean * ratio, u["peak_mem_gib"] * ratio


def sizing(cell, factors_path=None, calibrating=False):
    """(cpu, estimated peak GiB, source) or None when the class is unmeasured.

    ``factors_path`` is the JSON the driver writes from the ``--emit-factors`` job log.
    The CPU request is capped by the class's measured average, so an I/O-bound class is not
    handed cores it will never use; submit_pool.py applies the rest of the rules in
    benchmarks/nrp/sizing.py.
    """
    cpu = cpu_for(cell)
    est = model_bytes(cell, cpu) / GIB
    if calibrating:
        return cpu, est, "model"
    if not factors_path or not os.path.exists(factors_path):
        return None
    with open(factors_path, encoding="utf-8") as fh:
        cls = "tq" if cell["method"] == "tqfix" else cell["method"]
        f = json.load(fh).get(f"{cell['dataset']}/{cls}")
    if f is None:
        return None
    u = f.get("usage")
    if u and u.get("mean_cpu_cores"):
        cpu = max(1, min(cpu, int(u["mean_cpu_cores"] / 0.25)))
    if u and u.get("peak_mem_gib") and f.get("model_gib"):
        # Prefer the cgroup peak the meter recorded, scaled by the model's view of how this
        # cell compares with the metered one. The alternative below is a factor built from
        # peak anonymous RSS sampled twice a second, which misses a short allocation spike:
        # faiss PQ training spikes past twice its own average, and four PQ cells sized that
        # way were OOM-killed on 2026-09-16 at six of the ten GiB the model asked for.
        scaled = u["peak_mem_gib"] * (model_bytes(cell, cpu) / GIB) / f["model_gib"]
        return cpu, max(scaled, est * max(f["factor"], 0.25)), "measured-peak"

    factor = max(f["factor"], 0.25)
    if (
        cell["method"] == "rabitqlib_ivf"
        and cell["dataset"] in RAM_CORPUS_RABITQLIB
        and cell["dataset"] not in RAM_CORPUS_CALIBRATED
    ):
        n, d = ROWS[cell["dataset"]], DIMS[cell["dataset"]]
        base = model_bytes(cell, cpu, ram_corpus=False) / GIB
        return cpu, base * factor + n * d * 4 / GIB, "measured-factor+ram-corpus"
    return cpu, est * factor, "measured-factor"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results")
    ap.add_argument("--emit-factors", action="store_true", help="one log line")
    a = ap.parse_args()
    if a.emit_factors:
        print("FACTORS_JSON " + json.dumps(factors(a.results)), flush=True)
        return
    for c in sorted(calibration_cells(), key=lambda c: (c["dataset"], c["method"])):
        cpu = cpu_for(c)
        print(
            f"{c['cell_id']:48s} cpu={cpu} model={model_bytes(c, cpu) / GIB:6.2f} GiB request={math.ceil(1.25 * model_bytes(c, cpu) / GIB)} Gi"
        )
    if a.results:
        print(json.dumps(factors(a.results), indent=1))


if __name__ == "__main__":
    main()
