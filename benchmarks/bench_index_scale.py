# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Scale benchmark for the memory-mapped / sharded index.

Two phases, run as separate processes so the search process's peak RSS reflects
*only* memmap search (not the build, which necessarily holds the corpus):

    # build a sharded index on disk + a held-out query set with exact ground truth
    python benchmarks/bench_index_scale.py build --n 5000000 --dim 64 --out-dir /path/idx
    # memory-mapped search: reports peak RSS, recall@k vs exact, and QPS
    python benchmarks/bench_index_scale.py search --out-dir /path/idx

The headline the benchmark demonstrates: peak search RSS stays a small fraction of
the on-disk index size (bounded by ``n_queries * block``), while reranked recall
tracks the exact ranking — memmap makes indexes larger than RAM searchable.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np

from turboquant_pro import ShardedIndex


def _peak_rss_gb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # KiB on Linux
    return kb / (1024 * 1024)


def _gen_chunk(n, dim, basis, scale, rng):
    coeffs = rng.standard_normal((n, len(basis))) * scale
    return (coeffs @ basis + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)


def _exact_topk_blocked(queries, base_shards_iter, k):
    """Exact cosine top-k of queries against a corpus streamed in blocks."""
    qn = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-30)
    nq = len(qn)
    best_sc = np.full((nq, 0), -np.inf, np.float32)
    best_ix = np.full((nq, 0), -1, np.int64)
    off = 0
    for blk in base_shards_iter:
        bn = blk / np.maximum(np.linalg.norm(blk, axis=1, keepdims=True), 1e-30)
        sims = (qn @ bn.T).astype(np.float32)
        ix = np.broadcast_to(np.arange(off, off + len(blk), dtype=np.int64), sims.shape)
        csc = np.concatenate([best_sc, sims], axis=1)
        cix = np.concatenate([best_ix, ix], axis=1)
        part = np.argpartition(-csc, min(k, csc.shape[1] - 1), axis=1)[:, :k]
        best_sc = np.take_along_axis(csc, part, axis=1)
        best_ix = np.take_along_axis(cix, part, axis=1)
        off += len(blk)
    return best_ix


def _build(args):
    rng = np.random.default_rng(0)
    rank = args.dim // 2
    basis = rng.standard_normal((rank, args.dim))
    scale = np.linspace(1.0, 0.3, rank)
    os.makedirs(args.out_dir, exist_ok=True)

    # Stream the corpus into the sharded index one shard at a time (bounded build RAM).
    corpus_path = os.path.join(args.out_dir, "corpus.npy")
    corpus = np.lib.format.open_memmap(
        corpus_path, mode="w+", dtype=np.float32, shape=(args.n, args.dim)
    )
    for s in range(0, args.n, args.shard_size):
        e = min(s + args.shard_size, args.n)
        corpus[s:e] = _gen_chunk(e - s, args.dim, basis, scale, rng)
    corpus.flush()

    t0 = time.perf_counter()
    ShardedIndex.create(
        corpus,
        args.out_dir,
        shard_size=args.shard_size,
        output_dim=args.out_dim,
        bits=args.bits,
        metric="cosine",
    )
    build_s = time.perf_counter() - t0

    # Held-out queries + exact ground truth (streamed over the corpus memmap).
    q_idx = rng.choice(args.n, size=args.queries, replace=False)
    queries = np.asarray(corpus[q_idx])
    gt = _exact_topk_blocked(
        queries,
        (corpus[s : s + args.shard_size] for s in range(0, args.n, args.shard_size)),
        args.k,
    )
    np.save(os.path.join(args.out_dir, "queries.npy"), queries)
    np.save(os.path.join(args.out_dir, "gt.npy"), gt)
    del corpus
    os.remove(corpus_path)  # only the shards + manifest are needed to search

    total = sum(
        os.path.getsize(os.path.join(args.out_dir, f))
        for f in os.listdir(args.out_dir)
        if f.endswith(".tqe")
    )
    print(
        json.dumps(
            {
                "phase": "build",
                "n": args.n,
                "dim": args.dim,
                "shard_size": args.shard_size,
                "index_bytes": total,
                "index_gib": round(total / 1024**3, 3),
                "build_s": round(build_s, 1),
                "build_peak_rss_gib": round(_peak_rss_gb(), 3),
            },
            indent=2,
        )
    )


def _search(args):
    sh = ShardedIndex.open(
        os.path.join(args.out_dir, "manifest.json"), mmap=not args.no_mmap
    )
    queries = np.load(os.path.join(args.out_dir, "queries.npy"))
    gt = np.load(os.path.join(args.out_dir, "gt.npy"))
    index_bytes = sum(
        os.path.getsize(os.path.join(args.out_dir, f))
        for f in os.listdir(args.out_dir)
        if f.endswith(".tqe")
    )

    # warm + timed
    sh.search(queries[:8], k=args.k, rerank=args.rerank, block=args.block)
    t0 = time.perf_counter()
    ids, _ = sh.search(queries, k=args.k, rerank=args.rerank, block=args.block)
    dt = time.perf_counter() - t0

    k = gt.shape[1]
    recall = float(np.mean([len(set(a) & set(g)) / k for a, g in zip(ids[:, :k], gt)]))
    peak = _peak_rss_gb()
    print(
        json.dumps(
            {
                "phase": "search",
                "mmap": not args.no_mmap,
                "n_rows": sh.n_rows,
                "n_shards": sh.n_shards,
                "index_gib": round(index_bytes / 1024**3, 3),
                "search_peak_rss_gib": round(peak, 3),
                "rss_over_index": round(peak / (index_bytes / 1024**3), 3),
                "queries": len(queries),
                f"recall_at_{k}": round(recall, 4),
                "qps": round(len(queries) / dt, 1),
                "rerank": args.rerank,
                "block": args.block,
            },
            indent=2,
        )
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Memmap/sharded index scale benchmark.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--n", type=int, default=5_000_000)
    b.add_argument("--dim", type=int, default=64)
    b.add_argument("--out-dim", type=int, default=32)
    b.add_argument("--bits", type=int, default=3)
    b.add_argument("--shard-size", type=int, default=1_000_000)
    b.add_argument("--queries", type=int, default=200)
    b.add_argument("--k", type=int, default=10)
    b.add_argument("--out-dir", required=True)
    b.set_defaults(func=_build)
    s = sub.add_parser("search")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--rerank", type=int, default=10)
    s.add_argument("--block", type=int, default=262_144)
    s.add_argument(
        "--no-mmap",
        action="store_true",
        help="load the shards fully into RAM instead of memory-mapping "
        "(the baseline the memmap path is compared against)",
    )
    s.set_defaults(func=_search)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
