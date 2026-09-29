"""Run one systems cell and write ``<out>/<cell_id>.json`` (+ ``.ids.npz``).

    python -m systems.cell --cell-id <id> --data-root /data --out /data/systems

The protocol is the same for every system (docs/PREREG_systems.md):

1. **Build**, timed.
2. For each variant the builder exposes:
   - **throughput**: one warm-up batch, then every query in one batch at
     ``threads`` threads, ``repeats`` times; QPS from the median;
   - **latency**: ``n_latency`` queries one at a time, each timed; p50, p95,
     p99 and mean in milliseconds;
   - **recall@10 single-stage** against the L1 ground truth (exact top-10
     under cosine on the fp32 rows);
   - **rerank** at each depth in ``RERANK_DEPTHS``: recall@10 after exact
     rescoring of the first ``depth`` candidates against the fp32 originals,
     with the gather and the rescoring timed apart and the bytes fetched.
3. **Stages**, each against its own reference, never against one number:
   - scoring (TurboQuant Pro): the fast scorer's top-10 against the exact
     scorer's, same index;
   - routing (IVF-style variants): top-10 against the same index searched
     over every partition;
   - representation: the exact compressed scan's recall against fp32 truth.
4. **Strata**: when ``<data-root>/strata/<dataset>.json`` exists (per-query
   difficulty labels, OVB FAMILY.md section on strata), every recall above is
   also reported per stratum.

Bytes per row are measured by the builder (``systems.py``). The record carries
the node, the CPU model, thread count, library versions and TurboQuant Pro's
scorer provenance, so a timing is never separated from the machine it ran on.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
from rabitq_public.cell import AnonPeak, environment, hits_at_10
from rabitq_public.datasets import Dataset

from .grid import cells
from .systems import BUILDERS

K = 50
RERANK_DEPTHS = (20, 50)


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            return next(
                ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")
            )
    except (OSError, StopIteration):
        return platform.processor()


def machine(threads: int) -> dict:
    return dict(
        node=os.environ.get("NODE_NAME", platform.node()),
        cpu_model=_cpu_model(),
        cpus_visible=os.cpu_count(),
        threads=threads,
    )


def _agree10(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per query: the fraction of b's top-10 that a's top-10 also returns."""
    return hits_at_10(b[:, :10], a[:, :10]) / 10.0


def throughput(search, q, k, repeats):
    search(q[: min(32, len(q))], k)  # warm-up: page in tables, spin up threads
    times, out = [], None
    for _ in range(repeats):
        t = time.perf_counter()
        out = search(q, k)
        times.append(time.perf_counter() - t)
    med = float(np.median(times))
    return out, dict(
        qps=round(len(q) / med, 2), batch_s=[round(x, 4) for x in times], nq=len(q)
    )


def latency(search, q, n, k=10):
    ms = []
    for i in range(min(n, len(q))):
        t = time.perf_counter()
        search(q[i : i + 1], k)
        ms.append((time.perf_counter() - t) * 1e3)
    ms = np.asarray(ms)
    return dict(
        n=len(ms),
        p50_ms=round(float(np.percentile(ms, 50)), 4),
        p95_ms=round(float(np.percentile(ms, 95)), 4),
        p99_ms=round(float(np.percentile(ms, 99)), 4),
        mean_ms=round(float(ms.mean()), 4),
    )


def rerank_timed(ds, ids, depth):
    """Exact cosine rescoring of the first ``depth`` candidates, the gather and
    the rescoring timed apart (the campaign's ``rerank``, instrumented)."""
    cand = ids[:, :depth]
    uniq = np.unique(cand[cand >= 0])
    t = time.perf_counter()
    rows = ds.take(uniq)
    gather = time.perf_counter() - t
    t = time.perf_counter()
    out = np.full((len(cand), 10), -1, np.int64)
    for i in range(len(cand)):
        c = cand[i][cand[i] >= 0]
        s = rows[np.searchsorted(uniq, c)] @ ds.queries[i]
        out[i, : min(10, len(c))] = c[np.argsort(-s, kind="stable")[:10]]
    rescore = time.perf_counter() - t
    cost = dict(
        depth=depth,
        gather_s=round(gather, 3),
        rescore_s=round(rescore, 3),
        rows_fetched=int(len(uniq)),
        bytes_fetched_per_query=int(depth * ds.dim * 4),
    )
    return out, cost


def by_stratum(values: np.ndarray, strata: dict | None) -> dict | None:
    if not strata:
        return None
    return {
        name: round(float(np.mean(values[np.asarray(idx)])), 5)
        for name, idx in strata.items()
        if len(idx)
    }


def run(cell: dict, data_root: str, out_dir: str, threads: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{cell['cell_id']}.json")
    if os.path.exists(path):
        return path
    t0 = time.time()
    sp = os.path.join(data_root, "strata", f"{cell['dataset']}.json")
    strata = None
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as f:
            strata = json.load(f)["strata"]
    with AnonPeak() as anon:
        with anon.phase("load"):
            ds = Dataset(cell["dataset"], data_root)
        if ds.gt is None:
            raise SystemExit(f"no ground truth for {cell['dataset']}; run gt.py first")
        with anon.phase("build"):
            built = BUILDERS[cell["method"]](ds, cell, threads)
        q, gt = ds.queries, ds.gt
        variants, ids_out = {}, {}
        for name, search in built.variants.items():
            with anon.phase(f"search:{name}"):
                (ids, _), tput = throughput(search, q, K, cell.get("repeats", 3))
                ids = np.asarray(ids, np.int64)
                lat = latency(search, q, cell.get("n_latency", 1000))
            single = hits_at_10(gt, ids[:, :10]) / 10.0
            v = dict(
                throughput=tput,
                latency=lat,
                recall10_single=round(float(single.mean()), 5),
                recall10_single_by_stratum=by_stratum(single, strata),
                rerank=[],
            )
            with anon.phase(f"rerank:{name}"):
                for depth in RERANK_DEPTHS:
                    rr, cost = rerank_timed(ds, ids, depth)
                    h = hits_at_10(gt, rr) / 10.0
                    v["rerank"].append(
                        dict(
                            **cost,
                            recall10=round(float(h.mean()), 5),
                            recall10_by_stratum=by_stratum(h, strata),
                        )
                    )
            variants[name] = v
            ids_out[name] = ids
        stages = {}
        if "exact" in ids_out and "fast" in ids_out:
            a = _agree10(ids_out["fast"], ids_out["exact"])
            stages["scoring_fast_vs_exact"] = dict(
                mean=round(float(a.mean()), 5),
                min=round(float(a.min()), 5),
                by_stratum=by_stratum(a, strata),
            )
            stages["representation_exact_vs_fp32"] = variants["exact"][
                "recall10_single"
            ]
        for routed, ref in built.routing_reference.items():
            a = _agree10(ids_out[routed], ids_out[ref])
            stages[f"routing_{routed}_vs_{ref}"] = dict(
                mean=round(float(a.mean()), 5),
                min=round(float(a.min()), 5),
                by_stratum=by_stratum(a, strata),
            )
    rec = dict(
        cell=cell,
        n=ds.n,
        dim=ds.dim,
        nq=len(q),
        bytes_per_row=round(built.bytes_per_row, 3),
        rerank_tier_bytes_per_row=ds.dim * 4,
        build_s=round(built.build_s, 2),
        variants=variants,
        stages=stages,
        provenance=built.provenance,
        machine=machine(threads),
        usage=anon.usage(),
        env=_env(),
        wall_s=round(time.time() - t0, 1),
        finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    np.savez_compressed(
        os.path.join(out_dir, f"{cell['cell_id']}.ids.npz"),
        **{k: v.astype(np.int32) for k, v in ids_out.items()},
    )
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    os.replace(tmp, path)
    return path


def _env() -> dict:
    env = environment()
    try:
        from importlib.metadata import version

        env["scann"] = version("scann")
    except Exception:  # noqa: BLE001 - optional dependency
        env["scann"] = None
    from turboquant_pro import scorer

    env["tqp_scorers"] = scorer.describe()
    return env


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cell-id")
    g.add_argument("--cell-json", help="an unregistered cell, for smoke tests only")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--threads", type=int, default=int(os.environ.get("CELL_THREADS", "4"))
    )
    a = ap.parse_args()
    if a.cell_json:
        cell = json.loads(a.cell_json)
    else:
        cell = next(c for c in cells() if c["cell_id"] == a.cell_id)
    print(run(cell, a.data_root, a.out, a.threads), flush=True)


if __name__ == "__main__":
    main()
