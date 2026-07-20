# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Placement experiment: does cell-aligned sharding make routing sparse?

The distributed layer claims a query should touch ``nprobe/nlist`` of the
fleet rather than all of it. Our 1B and 10B runs could not test that: shards
were round-robin row ranges, so every coarse cell had rows on every server and
``servers_touched`` was necessarily 100%. This builds the *same corpus twice*
— round-robin vs cell-aligned — from one shared basis and one global coarse
quantizer, and measures what changes.

Registered predictions (from `paper/systems/TRANSACTION_CONTROL.md`):

* **P1 sparsity** — cell-aligned touches ``min(nprobe, S)`` shards; round-robin
  touches all ``S``.
* **P2 exactness** — placement moves rows, it does not change the ranking:
  results at full-coverage ``nprobe`` must be *identical* between placements.
* **P3 transaction-bound latency** — wall time tracks shards touched, not bytes
  scanned (bytes are set by the probe and are equal across placements).
* **P4 imbalance is the cost** — real corpora have skewed cell occupancy, so
  cell-aligned shards are unequal in size. Reported, because it is the most
  likely way this optimization disappoints in practice.

Run (Atlas, GPU for the O(N·nlist) assignment)::

    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_placement.py \
        --rows 50000000 --nlist 4096 --shards 64
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np

from turboquant_pro import ShardedIndex
from turboquant_pro.distributed import Router, build_cell_placement
from turboquant_pro.ivf import _assign, _kmeans_unit
from turboquant_pro.adc_index import _normalize

BIGANN = "/archive/tqp_bigann/base.1B.u8bin"
HEADER, DIM = 8, 128


def read_bigann(n_rows: int, offset: int = 0) -> np.ndarray:
    with open(BIGANN, "rb") as f:
        f.seek(HEADER + offset * DIM)
        buf = f.read(n_rows * DIM)
    return np.frombuffer(buf, dtype=np.uint8).reshape(-1, DIM)


def build(out_dir, blocks_with_ids, basis_from, out_dim, bits, metric):
    """Write shards from (block, ids) pairs, reusing one basis."""
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    metas = []
    for i, (block, ids) in enumerate(blocks_with_ids):
        metas.append(
            ShardedIndex.write_shard(
                out_dir,
                block.astype(np.float32),
                i,
                ids=ids,
                basis_from=basis_from,
                output_dim=out_dim,
                bits=bits,
                metric=metric,
                keep_originals=False,
            )
        )
    return ShardedIndex.finalize_manifest(out_dir, metas, metric=metric)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000_000)
    ap.add_argument("--shards", type=int, default=64)
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--nprobes", default="8,32,128")
    ap.add_argument("--work", default="/archive/tqp_placement")
    ap.add_argument("--out", default="bench_placement_result.json")
    a = ap.parse_args()
    nprobes = [int(x) for x in a.nprobes.split(",")]
    res: dict = {"rows": a.rows, "shards": a.shards, "nlist": a.nlist,
                 "out_dim": a.out_dim, "bits": a.bits, "metric": "l2"}

    print(f"reading {a.rows} BIGANN rows", flush=True)
    t0 = time.time()
    x = read_bigann(a.rows)
    q = read_bigann(a.queries, offset=a.rows + 1000)  # disjoint from the corpus
    print(f"read {time.time() - t0:.0f}s {x.shape}", flush=True)

    # One basis + one global coarse quantizer, shared by BOTH placements, so the
    # only difference between them is where rows live.
    seed_dir = os.path.join(a.work, "seed")
    if os.path.exists(seed_dir):
        shutil.rmtree(seed_dir)
    os.makedirs(seed_dir, exist_ok=True)
    seed_meta = ShardedIndex.write_shard(
        seed_dir, x[:1_000_000].astype(np.float32), 0,
        output_dim=a.out_dim, bits=a.bits, metric="l2", keep_originals=False,
    )
    basis_from = os.path.join(seed_dir, seed_meta["path"])
    seed = ShardedIndex.finalize_manifest(seed_dir, [seed_meta], metric="l2")
    adc = seed._get_shard(0)._adc

    print("fitting global coarse quantizer", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(0)
    train = _normalize(
        adc._cent[np.asarray(adc._codes)][
            rng.choice(seed.n_rows, min(200_000, seed.n_rows), replace=False)
        ].astype(np.float32)
    )
    cent = _kmeans_unit(train, a.nlist, 12, rng, block=50000, device="gpu")

    # Assign every row to its cell (the O(N*nlist) step -> GPU).
    #
    # This MUST reproduce what build_ivf does internally, which assigns from the
    # *quantized* direction ``normalize(cent[codes])`` — not from the raw PCA
    # projection. Grouping rows by cells computed any other way would produce
    # shards that are not actually cell-aligned, and the experiment would
    # silently measure nothing.
    print("assigning cells (from quantized directions, as build_ivf does)", flush=True)
    cells = np.empty(len(x), dtype=np.int64)
    CH = 2_000_000
    for s in range(0, len(x), CH):
        blk = x[s : s + CH].astype(np.float32)
        xp = np.asarray(adc._pca.transform(blk), dtype=np.float32)
        cn = np.linalg.norm(xp, axis=1)
        rot = adc._tq._rotate(xp / np.maximum(cn[:, None], 1e-30))
        codes = np.searchsorted(adc._tq.boundaries, rot).astype(np.uint8)
        d = _normalize(adc._cent[codes].astype(np.float32))
        cells[s : s + CH] = _assign(d, cent, block=50000, device="gpu")
    print(f"assignment {time.time() - t0:.0f}s", flush=True)

    occ = np.bincount(cells, minlength=a.nlist)
    res["cell_occupancy"] = {
        "mean": float(occ.mean()), "max": int(occ.max()), "min": int(occ.min()),
        "p99_over_median": float(np.percentile(occ, 99) / max(np.median(occ), 1)),
        "empty_cells": int((occ == 0).sum()),
        # P4: Gini of occupancy — how unequal the cells are on REAL data.
        "gini": float(
            (2 * np.sum((np.arange(1, a.nlist + 1)) * np.sort(occ)))
            / (a.nlist * np.sum(occ)) - (a.nlist + 1) / a.nlist
        ),
    }
    print(json.dumps(res["cell_occupancy"]), flush=True)

    per = len(x) // a.shards
    ids_all = np.arange(len(x), dtype=np.int64)

    # --- placement A: round-robin row ranges (what 1B/10B used) ------------- #
    def rr_blocks():
        for i in range(a.shards):
            sl = slice(i * per, (i + 1) * per if i < a.shards - 1 else len(x))
            yield x[sl], ids_all[sl]

    # --- placement B: cell-aligned (contiguous cell ranges per shard) ------- #
    order = np.argsort(cells, kind="stable")
    bounds = np.searchsorted(cells[order], np.linspace(0, a.nlist, a.shards + 1))

    def ca_blocks():
        for i in range(a.shards):
            sel = order[bounds[i] : bounds[i + 1]]
            yield x[sel], ids_all[sel]

    for name, gen in (("round_robin", rr_blocks), ("cell_aligned", ca_blocks)):
        print(f"building {name}", flush=True)
        t0 = time.time()
        d = os.path.join(a.work, name)
        sh = build(d, gen(), basis_from, a.out_dim, a.bits, "l2")
        sh.build_ivf(centroids=cent)
        res.setdefault("build_s", {})[name] = round(time.time() - t0, 1)
        sizes = [s["n_rows"] for s in json.load(open(os.path.join(d, "manifest.json")))["shards"]]
        res.setdefault("shard_rows", {})[name] = {
            "min": int(min(sizes)), "max": int(max(sizes)),
            "max_over_mean": round(float(max(sizes) / np.mean(sizes)), 3),
        }

        # Routing sparsity, measured the way the coordinator would compute it.
        mans = [os.path.join(d, "manifest.json")]
        placement = build_cell_placement(mans, ["0"], out_dir=d)
        res.setdefault("nonempty_cells_in_index", {})[name] = len(placement)

        router = Router(d, {c: ["0"] for c in placement}, pipeline_manifest=mans[0])
        qf = q.astype(np.float32)
        stats = {}
        for npb in nprobes:
            probed = router.probed(qf, npb)
            # Which shards actually hold each probed cell?
            shard_of_cell: dict[int, set] = {}
            for si, s_ in enumerate(json.load(open(os.path.join(d, "manifest.json")))["shards"]):
                off = np.load(os.path.join(d, os.path.splitext(s_["path"])[0] + ".ivf.off.npy"))
                for c in np.nonzero(np.diff(off))[0]:
                    shard_of_cell.setdefault(int(c), set()).add(si)
            touched = [
                len(set().union(*[shard_of_cell.get(int(c), set()) for c in row]))
                for row in probed
            ]
            t1 = time.time()
            ids, _ = sh.search(qf, k=10, nprobe=npb, workers=8)
            wall = time.time() - t1
            stats[str(npb)] = {
                "shards_touched_mean": round(float(np.mean(touched)), 2),
                "shards_touched_frac": round(float(np.mean(touched)) / a.shards, 4),
                "wall_s": round(wall, 2),
                "qps": round(len(qf) / wall, 2),
            }
            res.setdefault("ids", {}).setdefault(name, {})[str(npb)] = ids[:20].tolist()
            print(json.dumps({name: {str(npb): stats[str(npb)]}}), flush=True)
        res.setdefault("routing", {})[name] = stats

    # P2: exactness — at full coverage the two placements must agree exactly.
    full = str(max(nprobes))
    same = np.array_equal(
        np.array(res["ids"]["round_robin"][full]),
        np.array(res["ids"]["cell_aligned"][full]),
    )
    res["p2_identical_at_full_coverage"] = bool(same)
    print(f"P2 identical results at nprobe={full}: {same}", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("RESULT_JSON " + json.dumps({k: v for k, v in res.items() if k != "ids"}),
          flush=True)
    print("PLACEMENT_DONE", flush=True)


if __name__ == "__main__":
    main()
