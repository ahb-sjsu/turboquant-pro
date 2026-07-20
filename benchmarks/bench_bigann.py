# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""BIGANN/SIFT1B: recall against *published* ground truth.

The number the community can compare. Every previous scale result of ours was
measured against a reference we computed ourselves — self-consistent, but not
comparable to anything in the literature. BIGANN ships 10k queries and exact
top-100 ground truth for the full billion, so this is the first measurement
that can be placed beside published FAISS/DiskANN/SPANN numbers.

Design choices that matter, and why:

* **L2, not cosine.** BIGANN's ground truth is squared-L2 over un-normalized
  SIFT descriptors. Scoring cosine against an L2 oracle would understate recall
  for reasons unrelated to compression, so this run depends on the metric='l2'
  ADC path.
* **Cell-aligned placement, few large shards.** Round-robin placement makes
  every query touch every shard; the placement experiment measured 37x at
  nprobe=8 from fixing that. Shard count is set near the largest nprobe swept,
  per the S* ~ nprobe design rule.
* **Recall reported against BOTH references** — published fp32 truth *and*
  this index's own exact ADC ranking. The gap between them is the compressed
  domain's ceiling, and reporting only one of them is how that ceiling stays
  invisible.
* **Streaming build.** 128 GB of corpus is read in blocks; peak RSS stays
  bounded and no intermediate copy of the corpus is materialized.

    CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_bigann.py --rows 1000000000
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import time

import numpy as np

from turboquant_pro import ShardedIndex
from turboquant_pro.adc_index import _normalize
from turboquant_pro.ivf import _assign, _kmeans_unit

BASE = "/archive/tqp_bigann/base.1B.u8bin"
QUERY = "/archive/tqp_bigann/query.10K.u8bin"
GT = "/archive/tqp_bigann/gt.1B.ibin"
HEADER, DIM = 8, 128


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def read_rows(path: str, start: int, n: int, dim: int = DIM) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(HEADER + start * dim)
        return np.frombuffer(f.read(n * dim), dtype=np.uint8).reshape(-1, dim)


def load_queries_and_gt(n_queries: int):
    nq, dim = np.fromfile(QUERY, dtype=np.uint32, count=2)
    q = np.fromfile(QUERY, dtype=np.uint8, offset=HEADER).reshape(int(nq), int(dim))
    gn, gk = np.fromfile(GT, dtype=np.uint32, count=2)
    gt = np.fromfile(GT, dtype=np.uint32, offset=HEADER, count=int(gn) * int(gk))
    gt = gt.reshape(int(gn), int(gk))
    return q[:n_queries].astype(np.float32), gt[:n_queries].astype(np.int64)


def recall_at(got: np.ndarray, ref: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(got, ref)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000_000)
    ap.add_argument("--shard-rows", type=int, default=8_000_000)
    ap.add_argument("--nlist", type=int, default=16384)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--nprobes", default="16,64,256")
    ap.add_argument("--train-rows", type=int, default=2_000_000)
    ap.add_argument("--work", default="/archive/tqp_bigann_idx")
    ap.add_argument("--out", default="/archive/tqp_bigann_idx/result.json")
    a = ap.parse_args()
    nprobes = [int(x) for x in a.nprobes.split(",")]
    n_shards = -(-a.rows // a.shard_rows)

    q, gt = load_queries_and_gt(a.queries)
    res: dict = {
        "dataset": "BIGANN/SIFT1B",
        "rows": a.rows,
        "metric": "l2",
        "shards": n_shards,
        "shard_rows": a.shard_rows,
        "nlist": a.nlist,
        "out_dim": a.out_dim,
        "bits": a.bits,
        "queries": len(q),
        "ground_truth": "published (GT.public.1B.ibin)",
    }
    print(json.dumps(res), flush=True)

    # --- basis + global coarse quantizer, fitted once on a prefix ----------- #
    seed_dir = os.path.join(a.work, "seed")
    if not os.path.exists(os.path.join(seed_dir, "manifest.json")):
        if os.path.exists(seed_dir):
            shutil.rmtree(seed_dir)
        os.makedirs(seed_dir, exist_ok=True)
        t0 = time.time()
        meta = ShardedIndex.write_shard(
            seed_dir,
            read_rows(BASE, 0, a.train_rows).astype(np.float32),
            0,
            output_dim=a.out_dim,
            bits=a.bits,
            metric="l2",
            keep_originals=False,
        )
        ShardedIndex.finalize_manifest(seed_dir, [meta], metric="l2")
        print(f"seed built {time.time() - t0:.0f}s", flush=True)
    seed = ShardedIndex.open(os.path.join(seed_dir, "manifest.json"))
    basis_from = os.path.join(seed_dir, "shard_00000.tqe")
    adc = seed._get_shard(0)._adc

    cent_path = os.path.join(a.work, "coarse.npy")
    if os.path.exists(cent_path):
        cent = np.load(cent_path)
    else:
        t0 = time.time()
        rng = np.random.default_rng(0)
        train = _normalize(
            adc._cent[np.asarray(adc._codes)][
                rng.choice(seed.n_rows, min(400_000, seed.n_rows), replace=False)
            ].astype(np.float32)
        )
        cent = _kmeans_unit(train, a.nlist, 12, rng, block=20000, device="gpu")
        np.save(cent_path, cent)
        print(
            f"coarse quantizer {a.nlist} cells in {time.time() - t0:.0f}s", flush=True
        )

    def cells_for(block: np.ndarray) -> np.ndarray:
        """Cell ids the way build_ivf computes them: from quantized directions."""
        xp = np.asarray(adc._pca.transform(block.astype(np.float32)), dtype=np.float32)
        cn = np.linalg.norm(xp, axis=1)
        rot = adc._tq._rotate(xp / np.maximum(cn[:, None], 1e-30))
        codes = np.searchsorted(adc._tq.boundaries, rot).astype(np.uint8)
        return _assign(
            _normalize(adc._cent[codes].astype(np.float32)),
            cent,
            block=20000,
            device="gpu",
        )

    # --- build: cell-aligned, streaming ------------------------------------ #
    idx_dir = os.path.join(a.work, "index")
    manifest = os.path.join(idx_dir, "manifest.json")
    if not os.path.exists(manifest):
        os.makedirs(idx_dir, exist_ok=True)
        # Two passes over the corpus: assign cells, then emit shards by cell
        # range. Each pass streams; neither holds the corpus.
        t0 = time.time()
        cell_of = np.empty(a.rows, dtype=np.int32)
        CH = 4_000_000
        for s in range(0, a.rows, CH):
            n = min(CH, a.rows - s)
            cell_of[s : s + n] = cells_for(read_rows(BASE, s, n))
            if (s // CH) % 25 == 0:
                print(
                    f"  assign {s + n}/{a.rows} ({time.time() - t0:.0f}s)", flush=True
                )
        assign_s = time.time() - t0
        print(f"assignment {assign_s:.0f}s", flush=True)

        # Cell -> shard, then partition by ONE sequential pass over the corpus.
        #
        # The obvious implementation — for each shard, seek to each of its rows
        # — costs one random transaction per row (10^9 of them at full scale).
        # That is precisely the anti-pattern this paper is about, so the
        # partition streams the file once and appends each row to its shard's
        # spill file; each shard is then built from a sequential read.
        edges = np.linspace(0, a.nlist, n_shards + 1).astype(np.int64)
        shard_of_cell = np.searchsorted(edges[1:-1], np.arange(a.nlist), "right")
        spill_dir = os.path.join(a.work, "spill")
        os.makedirs(spill_dir, exist_ok=True)
        t0 = time.time()
        handles = [
            open(os.path.join(spill_dir, f"s{i:05d}.bin"), "wb")
            for i in range(n_shards)
        ]
        id_bufs: list[list] = [[] for _ in range(n_shards)]
        for s in range(0, a.rows, CH):
            n = min(CH, a.rows - s)
            blk = read_rows(BASE, s, n)
            tgt = shard_of_cell[cell_of[s : s + n]]
            for i in np.unique(tgt):
                m = tgt == i
                handles[i].write(np.ascontiguousarray(blk[m]).tobytes())
                id_bufs[i].append(np.flatnonzero(m).astype(np.int64) + s)
            if (s // CH) % 25 == 0:
                print(
                    f"  partition {s + n}/{a.rows} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
        for h in handles:
            h.close()
        print(f"partition {time.time() - t0:.0f}s", flush=True)

        t0 = time.time()
        metas = []
        for i in range(n_shards):
            p = os.path.join(spill_dir, f"s{i:05d}.bin")
            ids = (
                np.concatenate(id_bufs[i])
                if id_bufs[i]
                else np.empty(0, dtype=np.int64)
            )
            if len(ids) == 0:
                continue
            blk = np.fromfile(p, dtype=np.uint8).reshape(-1, DIM)
            assert len(blk) == len(ids), f"shard {i}: {len(blk)} rows vs {len(ids)} ids"
            metas.append(
                ShardedIndex.write_shard(
                    idx_dir,
                    blk.astype(np.float32),
                    i,
                    ids=ids,
                    basis_from=basis_from,
                    output_dim=a.out_dim,
                    bits=a.bits,
                    metric="l2",
                    keep_originals=False,
                )
            )
            os.unlink(p)  # reclaim as we go: the spill is a full corpus copy
            print(
                f"  shard {i + 1}/{n_shards} rows={len(ids)} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
        shutil.rmtree(spill_dir, ignore_errors=True)
        sh = ShardedIndex.finalize_manifest(idx_dir, metas, metric="l2")
        res["build_s"] = round(time.time() - t0 + assign_s, 1)
        t0 = time.time()
        sh.build_ivf(centroids=cent)
        res["build_ivf_s"] = round(time.time() - t0, 1)
    else:
        sh = ShardedIndex.open(manifest)
        print("index exists, reusing", flush=True)

    nbytes = sum(
        os.path.getsize(os.path.join(idx_dir, f))
        for f in os.listdir(idx_dir)
        if os.path.isfile(os.path.join(idx_dir, f))
    )
    res["index_bytes"] = nbytes
    res["bytes_per_row"] = round(nbytes / a.rows, 2)
    res["peak_rss_gib"] = round(peak_rss_gib(), 2)
    print(
        json.dumps({k: res[k] for k in ("bytes_per_row", "peak_rss_gib")}), flush=True
    )

    # --- measure ------------------------------------------------------------ #
    print("exact ADC full-scan reference", flush=True)
    t0 = time.time()
    ref, _ = sh.search(q, k=10, workers=16)
    res["fullscan"] = {
        "wall_s": round(time.time() - t0, 1),
        "recall10_vs_published_gt": recall_at(ref, gt, 10),
    }
    print(json.dumps(res["fullscan"]), flush=True)

    res["ivf"] = {}
    for npb in nprobes:
        t0 = time.time()
        got, _ = sh.search(q, k=10, nprobe=npb, workers=16)
        wall = time.time() - t0
        res["ivf"][str(npb)] = {
            "wall_s": round(wall, 1),
            "qps": round(len(q) / wall, 2),
            # Both references: the compressed domain's own answer, and truth.
            "recall10_vs_adc_fullscan": recall_at(got, ref, 10),
            "recall10_vs_published_gt": recall_at(got, gt, 10),
        }
        print(json.dumps({str(npb): res["ivf"][str(npb)]}), flush=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("RESULT_JSON " + json.dumps(res), flush=True)
    print("BIGANN_DONE", flush=True)


if __name__ == "__main__":
    main()
