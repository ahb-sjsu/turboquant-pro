# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Sharded IVF search benchmark on local storage (NVMe).

Builds a sharded index, adds the IVF coarse layer (`build_ivf`), and compares the
locality-optimized `search(nprobe=...)` against the brute-force full-scan fan-out:
recall@k (vs the full scan it approximates), fraction of rows scanned, QPS, and peak
RSS. This is the payoff of the coarse layer — high recall while touching a few percent
of the corpus — on storage where random reads are cheap (unlike a network FS).

    python benchmarks/bench_ivf_sharded.py --n 50000000 --nlist 2048 --out-dir /nvme/idx
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import time

import numpy as np

from turboquant_pro import ShardedIndex
from turboquant_pro.ivf import _normalize


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _gen(n, dim, rng):
    rank = dim // 2
    basis = rng.standard_normal((rank, dim))
    coeffs = rng.standard_normal((n, rank)) * np.linspace(1.0, 0.3, rank)
    return (coeffs @ basis + 0.05 * rng.standard_normal((n, dim))).astype(np.float32)


def _recall(got, ref, k):
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(got[:, :k], ref)]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sharded IVF search benchmark (NVMe).")
    ap.add_argument("--n", type=int, default=50_000_000)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--out-dim", type=int, default=24)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=5_000_000)
    ap.add_argument("--nlist", type=int, default=2048)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--nprobe",
        default="8,16,32,64",
        help="comma-separated nprobe values to sweep (scale up with nlist)",
    )
    ap.add_argument(
        "--workers",
        default="1",
        help="comma-separated parallel per-shard fan-out thread counts to sweep",
    )
    ap.add_argument(
        "--no-originals",
        action="store_true",
        help="skip fp32 originals (ADC-only; ~5x smaller index, no rerank)",
    )
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv)
    nprobes = [int(x) for x in args.nprobe.split(",")]
    workers_list = [int(x) for x in str(args.workers).split(",")]
    rng = np.random.default_rng(0)

    # Build the sharded index (corpus streamed one shard at a time to bound RAM).
    def blocks():
        for s in range(0, args.n, args.shard_size):
            yield _gen(min(args.shard_size, args.n - s), args.dim, rng)

    t0 = time.perf_counter()
    sh = ShardedIndex.create_streaming(
        blocks(),
        args.out_dir,
        output_dim=args.out_dim,
        bits=args.bits,
        shard_size=args.shard_size,
        keep_originals=not args.no_originals,
    )
    build_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    sh.build_ivf(nlist=args.nlist)
    ivf_s = time.perf_counter() - t0
    index_bytes = sum(
        os.path.getsize(f) for f in glob.glob(os.path.join(args.out_dir, "shard_*.tqe"))
    )
    print(
        json.dumps(
            {
                "phase": "build",
                "n": args.n,
                "nlist": args.nlist,
                "index_gib": round(index_bytes / 1024**3, 2),
                "build_s": round(build_s, 1),
                "build_ivf_s": round(ivf_s, 1),
                "peak_rss_gib": round(_peak_rss_gb(), 2),
            }
        ),
        flush=True,
    )

    # Queries = a random sample of real rows; the reference is the full-scan ADC top-k.
    q = _gen(args.queries, args.dim, np.random.default_rng(1))
    sh.search(q[:4], k=args.k)  # warm
    t0 = time.perf_counter()
    ref, _ = sh.search(q, k=args.k)  # brute full-scan fan-out
    brute_dt = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "mode": "brute full-scan",
                "scan_fraction": 1.0,
                "qps": round(len(q) / brute_dt, 2),
                "search_s": round(brute_dt, 1),
            }
        ),
        flush=True,
    )

    # Global cell sizes for an exact scan fraction.
    csize = np.zeros(args.nlist, dtype=np.int64)
    for f in sorted(glob.glob(os.path.join(args.out_dir, "shard_*.ivf.off.npy"))):
        csize += np.diff(np.load(f))
    cent = np.load(os.path.join(args.out_dir, "coarse_centroids.npy"))
    rad = np.load(os.path.join(args.out_dir, "coarse_radius.npy"))
    qr, _ = sh._get_shard(0)._adc._query_terms(q)
    theta = np.arccos(np.clip(_normalize(qr) @ cent.T, -1.0, 1.0))
    ub = np.cos(np.maximum(0.0, theta - 0.5 * rad[None, :]))
    order = np.argsort(-ub, axis=1)

    for nprobe in nprobes:
        if nprobe > args.nlist:
            continue
        probed = order[:, :nprobe]
        scan = float(np.mean([csize[probed[i]].sum() for i in range(len(q))])) / args.n
        recall = None  # identical across workers (exact); compute once
        for w in workers_list:
            sh.search(q[:4], k=args.k, nprobe=nprobe, workers=w)  # warm
            t0 = time.perf_counter()
            ids, _ = sh.search(q, k=args.k, nprobe=nprobe, workers=w)
            dt = time.perf_counter() - t0
            if recall is None:
                recall = round(_recall(ids, ref, args.k), 4)
            print(
                json.dumps(
                    {
                        "mode": f"ivf nprobe={nprobe}",
                        "workers": w,
                        "scan_fraction": round(scan, 5),
                        f"recall_at_{args.k}": recall,
                        "qps": round(len(q) / dt, 2),
                        "speedup_vs_brute": round(brute_dt / dt, 1),
                        "peak_rss_gib": round(_peak_rss_gb(), 2),
                    }
                ),
                flush=True,
            )
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
