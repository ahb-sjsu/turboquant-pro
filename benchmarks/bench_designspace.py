# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Design-space exploration for distributed vector search.

Applies the *Geometric Methods* (Book 1) methodology to the configuration
space of the index itself, rather than to a model: sensitivity profiling to
find which parameters matter, subset enumeration over the survivors to find
interactions, Pareto extraction for the non-dominated frontier, and a
perturbation-response measure to find **cliffs** — configurations where a
small parameter change causes a large objective change.

Motivation is concrete: we found the 250-shards/server cliff *by accident*
(a 1-query request that would not complete in 900 s). A parameter space that
hides discontinuities like that from a point-sampling methodology is exactly
what the structural-fuzzing discipline exists to map.

**Cost structure drives the design.** Parameters split into

* *build-time* (expensive — require rebuilding the index):
  shards, bits, out_dim, placement;
* *query-time* (cheap — swept within one build): nprobe, workers.

So the sweep builds a modest number of indexes and sweeps the cheap axes
inside each, which is what makes an otherwise infeasible grid tractable
(Ch. 17's point: exhaustive enumeration breaks, so exploit structure).

Objectives (all measured, none modelled):
  recall@10 vs the exact ADC full-scan, wall/query, bytes/row, shards touched.
$/QPS is derived downstream by `cost_model.py` from these plus list prices.

    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_designspace.py \
        --rows 10000000 --mode screen
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import time

import numpy as np

from turboquant_pro import ShardedIndex
from turboquant_pro.adc_index import _normalize
from turboquant_pro.ivf import _assign, _kmeans_unit

BIGANN = "/archive/tqp_bigann/base.1B.u8bin"
HEADER, DIM = 8, 128

# The baseline configuration that one-at-a-time profiling perturbs around.
BASE = {
    "shards": 32,
    "bits": 4,
    "out_dim": 64,
    "placement": "cell_aligned",
    "nprobe": 32,
    "workers": 8,
}
# Levels per axis. Kept small deliberately: the point is to locate structure
# and cliffs, not to tile the space.
LEVELS = {
    "shards": [8, 16, 32, 64, 128, 256],
    "bits": [2, 3, 4],
    "out_dim": [32, 64, 128],
    "placement": ["round_robin", "cell_aligned"],
    "nprobe": [8, 32, 128, 512],
    "workers": [1, 4, 16, 64],
}
BUILD_KEYS = ("shards", "bits", "out_dim", "placement")


def read_bigann(n: int, offset: int = 0) -> np.ndarray:
    with open(BIGANN, "rb") as f:
        f.seek(HEADER + offset * DIM)
        return np.frombuffer(f.read(n * DIM), dtype=np.uint8).reshape(-1, DIM)


class Corpus:
    """Holds the corpus and the per-(bits,out_dim) cell assignment needed for
    cell-aligned placement, computed the way ``build_ivf`` computes it."""

    def __init__(self, rows: int, queries: int, work: str, nlist: int):
        self.x = read_bigann(rows)
        self.q = read_bigann(queries, offset=rows + 1000).astype(np.float32)
        self.work = work
        self.nlist = nlist
        self._cache: dict = {}

    def quantizer(self, bits: int, out_dim: int):
        key = (bits, out_dim)
        if key in self._cache:
            return self._cache[key]
        seed_dir = os.path.join(self.work, f"seed_b{bits}_d{out_dim}")
        if os.path.exists(seed_dir):
            shutil.rmtree(seed_dir)
        os.makedirs(seed_dir, exist_ok=True)
        meta = ShardedIndex.write_shard(
            seed_dir, self.x[:1_000_000].astype(np.float32), 0,
            output_dim=out_dim, bits=bits, metric="l2", keep_originals=False,
        )
        basis = os.path.join(seed_dir, meta["path"])
        sh = ShardedIndex.finalize_manifest(seed_dir, [meta], metric="l2")
        adc = sh._get_shard(0)._adc
        rng = np.random.default_rng(0)
        train = _normalize(
            adc._cent[np.asarray(adc._codes)][
                rng.choice(sh.n_rows, min(200_000, sh.n_rows), replace=False)
            ].astype(np.float32)
        )
        cent = _kmeans_unit(train, self.nlist, 12, rng, block=50000, device="gpu")
        cells = np.empty(len(self.x), dtype=np.int64)
        for s in range(0, len(self.x), 2_000_000):
            blk = self.x[s : s + 2_000_000].astype(np.float32)
            xp = np.asarray(adc._pca.transform(blk), dtype=np.float32)
            cn = np.linalg.norm(xp, axis=1)
            rot = adc._tq._rotate(xp / np.maximum(cn[:, None], 1e-30))
            codes = np.searchsorted(adc._tq.boundaries, rot).astype(np.uint8)
            d = _normalize(adc._cent[codes].astype(np.float32))
            cells[s : s + 2_000_000] = _assign(d, cent, block=50000, device="gpu")
        self._cache[key] = (basis, cent, cells)
        return self._cache[key]


def build_index(c: Corpus, cfg: dict, out_dir: str):
    basis, cent, cells = c.quantizer(cfg["bits"], cfg["out_dim"])
    S = cfg["shards"]
    ids_all = np.arange(len(c.x), dtype=np.int64)
    if cfg["placement"] == "cell_aligned":
        order = np.argsort(cells, kind="stable")
        bounds = np.searchsorted(cells[order], np.linspace(0, c.nlist, S + 1))
        blocks = [(c.x[order[bounds[i] : bounds[i + 1]]],
                   ids_all[order[bounds[i] : bounds[i + 1]]]) for i in range(S)]
    else:
        per = len(c.x) // S
        blocks = [
            (c.x[i * per : (i + 1) * per if i < S - 1 else len(c.x)],
             ids_all[i * per : (i + 1) * per if i < S - 1 else len(c.x)])
            for i in range(S)
        ]
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    metas = []
    for i, (blk, ids) in enumerate(blocks):
        if len(blk) == 0:
            continue
        metas.append(
            ShardedIndex.write_shard(
                out_dir, blk.astype(np.float32), i, ids=ids, basis_from=basis,
                output_dim=cfg["out_dim"], bits=cfg["bits"], metric="l2",
                keep_originals=False,
            )
        )
    sh = ShardedIndex.finalize_manifest(out_dir, metas, metric="l2")
    sh.build_ivf(centroids=cent)
    nbytes = sum(
        os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)
        if os.path.isfile(os.path.join(out_dir, f))
    )
    return sh, nbytes / len(c.x)


def measure(sh, q, cfg, ref_ids):
    t0 = time.time()
    ids, _ = sh.search(q, k=10, nprobe=cfg["nprobe"], workers=cfg["workers"])
    wall = time.time() - t0
    rec = (
        float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ids, ref_ids)]))
        if ref_ids is not None
        else None
    )
    return {"wall_s": round(wall, 3), "qps": round(len(q) / wall, 3),
            "recall_vs_adc_fullscan": rec}, ids


def configs_for(mode: str) -> list[dict]:
    """One-at-a-time profiling (Ch. 9 sensitivity) or a focused interaction
    grid (Ch. 11 subset enumeration, restricted to the axes screening keeps)."""
    if mode == "screen":
        out = [dict(BASE)]
        for axis, levels in LEVELS.items():
            for v in levels:
                if v == BASE[axis]:
                    continue
                cfg = dict(BASE)
                cfg[axis] = v
                out.append(cfg)
        return out
    if mode == "interact":
        # The pair the transaction model says should interact: S x nprobe.
        out = []
        for S, npb in itertools.product(LEVELS["shards"], LEVELS["nprobe"]):
            cfg = dict(BASE)
            cfg["shards"], cfg["nprobe"] = S, npb
            out.append(cfg)
        return out
    raise ValueError(mode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=10_000_000)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--nlist", type=int, default=2048)
    ap.add_argument("--mode", default="screen", choices=["screen", "interact"])
    ap.add_argument("--work", default="/archive/tqp_designspace")
    ap.add_argument("--out", default="designspace.json")
    a = ap.parse_args()

    os.makedirs(a.work, exist_ok=True)
    c = Corpus(a.rows, a.queries, a.work, a.nlist)
    cfgs = configs_for(a.mode)
    print(f"{len(cfgs)} configurations, {a.rows} rows, mode={a.mode}", flush=True)

    results, built = [], {}
    for i, cfg in enumerate(cfgs):
        bkey = tuple(cfg[k] for k in BUILD_KEYS)
        d = os.path.join(a.work, "idx_" + "_".join(map(str, bkey)))
        try:
            if bkey not in built:
                t0 = time.time()
                sh, bpr = build_index(c, cfg, d)
                # Reference for recall: this index's own exhaustive scan.
                ref, _ = sh.search(c.q, k=10, workers=8)
                built[bkey] = (sh, bpr, ref, round(time.time() - t0, 1))
            sh, bpr, ref, build_s = built[bkey]
            m, _ = measure(sh, c.q, cfg, ref)
            row = {**cfg, **m, "bytes_per_row": round(bpr, 2), "build_s": build_s}
        except Exception as e:  # a config that cannot run IS a result
            row = {**cfg, "error": f"{type(e).__name__}: {e}"}
        results.append(row)
        print(json.dumps(row), flush=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print("DESIGNSPACE_DONE", flush=True)


if __name__ == "__main__":
    main()
