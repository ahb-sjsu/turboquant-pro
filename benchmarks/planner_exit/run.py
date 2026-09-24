"""One planner run of the P0 exit test (docs/PREREG_planner_exit.md), and its fresh re-evaluation.

    python -m planner_exit.run --arm glove-100-angular --run B2 \\
        --data-root /data --out /data/planner_exit

Reads the arm through the RaBitQ campaign's own loader (``rabitq_public.datasets``),
so the planner sees the same normalized corpus the grid saw. Never reads a grid
result. Writes ``<out>/<arm>/<run>.json``: the plan record (schema-validated),
the process CPU seconds of the planning call, and the same-scale re-evaluation of
the chosen configuration on a fresh disjoint sample.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np
from rabitq_public.datasets import Dataset, normalize
from rabitq_public.grid import DIMS
from rabitq_public.grid import configs as grid_configs

from turboquant_pro import consumers, plugins
from turboquant_pro.planner import (
    ABSTAIN,
    Artifact,
    Budget,
    CompressionPlanner,
    QualityFloor,
    WorkloadSpec,
    plan_schema,
)

N_PLAN = 100_000
Q_PLAN = 500
PLAN_SEED = 20260924
QUERY_SEED = 20260925
FRESH_SEED = 20260926
HOLDOUT = 0.3
FRESH_ROWS = int(round(N_PLAN * HOLDOUT))  # the held-out split's size: same scale
CONFIDENCE = 0.95
CONSUMER = {"k": 10, "rerank": 5, "n_queries": Q_PLAN}

# grid method -> (planner codec, config keys it carries)
METHOD_CODEC = {
    "tq": ("tq_embedding", ("bits", "out_dim")),
    "rabitq_flat": ("faiss_rabitq", ("bits",)),
    "pq": ("faiss_pq", ("m",)),
    "opq": ("faiss_opq", ("m",)),
}
CODEC_METHOD = {v[0]: k for k, v in METHOD_CODEC.items()}


def runs(dim: int) -> dict:
    """The seven registered runs of an arm (section 2)."""
    out = {}
    for b in (1, 2, 3, 4):
        out[f"B{b}"] = dict(objective="max_quality", max_bytes=-(-dim * b // 8) + 4)
    for f in (0.90, 0.95, 0.99):
        out[f"F{int(round(f * 100))}"] = dict(objective="min_cost", floor=f)
    return out


def pinned_configs(arm: str) -> dict:
    """The reachable grid, as the planner's candidate_configs."""
    out: dict = {}
    for c in grid_configs(arm):
        if c["method"] not in METHOD_CODEC:
            continue
        codec, keys = METHOD_CODEC[c["method"]]
        out.setdefault(codec, []).append({k: c[k] for k in keys})
    return out


def grid_key(arm: str, codec: str, config: dict) -> str:
    """The campaign's config key (cell id without the seed) of a planner choice."""
    from rabitq_public.grid import cell_id

    cfg = dict(method=CODEC_METHOD[codec], **config)
    return cell_id(arm, cfg, 0).rsplit("-s", 1)[0]


def draws(ds: Dataset, arm: str):
    """Artifact positions, planning queries, and fresh positions, all disjoint."""
    rng = np.random.default_rng(PLAN_SEED)
    if ds.spec.kind == "hdf5":
        import h5py

        with h5py.File(os.path.join(ds.root, ds.spec.path), "r") as f:
            nq = ds.spec.nq
            queries = normalize(np.asarray(f["test"][nq : nq + Q_PLAN], np.float32))
        taken = np.zeros(0, np.int64)
    else:
        qpos = np.sort(
            np.random.default_rng(QUERY_SEED).choice(ds.n, Q_PLAN, replace=False)
        )
        queries = ds.take(qpos)
        taken = qpos
    free = np.setdiff1d(np.arange(ds.n), taken, assume_unique=True)
    plan_pos = np.sort(rng.choice(free, min(N_PLAN, len(free)), replace=False))
    free = np.setdiff1d(free, plan_pos, assume_unique=True)
    fresh_pos = np.sort(
        np.random.default_rng(FRESH_SEED).choice(
            free, min(FRESH_ROWS, len(free)), replace=False
        )
    )
    return plan_pos, queries, fresh_pos


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def fresh_estimate(ds, codec: str, config: dict, fresh_pos, queries) -> dict:
    """The chosen configuration, once more, on a fresh sample of the held-out size."""
    x = ds.take(fresh_pos)
    q = plugins.create(codec, dim=ds.dim, **config)
    c = q.compress(x)
    recon = q.decompress(c)
    ctx = {"queries": queries}
    if callable(getattr(q, "search", None)):
        ctx["searcher"] = lambda qs, n: q.search(c, qs, n)
    cons = consumers.create_consumer("topk_inner_product", **CONSUMER)
    per = np.asarray(cons.per_item(x, recon, **ctx), float)
    return {"rows": int(len(x)), "mean": float(per.mean()), "per_query": per.tolist()}


def run(arm: str, run_id: str, data_root: str, out_dir: str) -> str:
    path = os.path.join(out_dir, arm, f"{run_id}.json")
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds = Dataset(arm, data_root)
    r = runs(DIMS[arm])[run_id]
    plan_pos, queries, fresh_pos = draws(ds, arm)
    art = Artifact(ds.take(plan_pos), name=f"{arm}:plan{N_PLAN}")
    spec = WorkloadSpec(
        target="embedding",
        consumer="topk_inner_product",
        consumer_config=CONSUMER,
        budget=Budget(max_bytes_per_vector=r.get("max_bytes")),
        floor=(
            QualityFloor(minimum=r["floor"], confidence=CONFIDENCE)
            if "floor" in r
            else None
        ),
        candidates=tuple(pinned_configs(arm)),
        candidate_configs=pinned_configs(arm),
        objective=r["objective"],
        seed=0,
        holdout_fraction=HOLDOUT,
        n_boot=512,
    )
    c0, t0 = cpu_seconds(), time.time()
    plan = CompressionPlanner(spec).plan(art, queries=queries)
    cpu, wall = cpu_seconds() - c0, time.time() - t0
    doc = plan.as_dict()
    import jsonschema

    jsonschema.validate(doc, plan_schema())
    rec = {
        "arm": arm,
        "run": run_id,
        "registered": r,
        "plan": doc,
        "cpu_seconds": round(cpu, 1),
        "wall_seconds": round(wall, 1),
        "choice": None,
    }
    if plan.selected_codec != ABSTAIN:
        cfg = dict(plan.selected_parameters)
        rec["choice"] = {
            "codec": plan.selected_codec,
            "config": cfg,
            "grid_key": grid_key(arm, plan.selected_codec, cfg),
            "holdout_bound": plan.expected_quality.bound,
            "holdout_mean": plan.expected_quality.mean,
        }
        rec["fresh"] = fresh_estimate(ds, plan.selected_codec, cfg, fresh_pos, queries)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, default=float)
    os.replace(tmp, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(DIMS))
    ap.add_argument("--run", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(run(a.arm, a.run, a.data_root, a.out), flush=True)


if __name__ == "__main__":
    main()
