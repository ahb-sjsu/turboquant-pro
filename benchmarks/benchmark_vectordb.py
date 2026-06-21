#!/usr/bin/env python3
"""Vector-database retrieval-cost benchmark on REAL embeddings (PVLDB paper).

Loads an embedding matrix (.npy), holds out queries, computes exact cosine
ground truth, and compares storage / build / throughput / recall for:

  * fp32-flat        -- exact baseline (faiss IndexFlatIP)
  * faiss-PQ         -- product quantization at a matched byte budget
  * faiss-HNSW       -- graph index, fp32 (fast, no compression)
  * turboquant-pro   -- PCA-Matryoshka(out_dim) + TQ{bits}, two ways:
       single-stage  : search the compressed (reconstructed) vectors
       +rerank       : retrieve k*oversample candidates, rerank by exact fp32
                       on the retained originals (the standard two-stage ANN
                       protocol; rerank needs the originals, so we report the
                       compressed-only storage separately).

Reports bytes/vector, compression x, build time, queries/s, recall@10 and @100.

Run on Atlas (CPU; thread-capped):
  OMP_NUM_THREADS=20 /home/claude/env/bin/python benchmark_vectordb.py \\
    --npy /archive/results_aesthetics/bip_sample_200k_labse.npy \\
    --queries 1000 --out-dim 256 --bits 2 3 4 --json /tmp/vectordb_bench.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def exact_topk(Q: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """Exact cosine top-k (inputs assumed L2-normalized). Chunked over queries."""
    out = np.zeros((len(Q), k), dtype=np.int64)
    chunk = 256
    for s in range(0, len(Q), chunk):
        sims = Q[s : s + chunk] @ X.T
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(idx.shape[0]):
            idx[r] = idx[r][np.argsort(-sims[r, idx[r]])]
        out[s : s + chunk] = idx
    return out


def recall(gt: np.ndarray, ap: np.ndarray, k: int) -> float:
    return float(
        np.mean([len(set(gt[i, :k]) & set(ap[i, :k])) / k for i in range(len(gt))])
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--bits", type=int, nargs="+", default=[3])
    ap.add_argument("--oversample", type=int, default=5)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--json", default="/tmp/vectordb_bench.json")
    a = ap.parse_args()

    import faiss

    faiss.omp_set_num_threads(a.threads)

    X = np.load(a.npy, mmap_mode="r").astype(np.float32)
    X = normalize(X)
    nq = a.queries
    Q, C = X[:nq].copy(), X[nq:].copy()
    N, dim = C.shape
    print(f"corpus={N} dim={dim} queries={nq}", flush=True)

    print("exact ground truth...", flush=True)
    gt = exact_topk(Q, C, 100)

    res = []

    def rec(method, bpv, build_s, qps, r10, r100, note=""):
        row = dict(
            method=method, n=N, dim=dim, bytes_per_vec=round(bpv, 1),
            compression_x=round(dim * 4 / bpv, 1), build_s=round(build_s, 2),
            qps=round(qps, 1), recall_at_10=round(r10, 4),
            recall_at_100=round(r100, 4), note=note,
        )
        res.append(row)
        print(" ", row, flush=True)

    # --- fp32 flat (exact) ---
    t = time.perf_counter(); flat = faiss.IndexFlatIP(dim); flat.add(C)
    bt = time.perf_counter() - t
    t = time.perf_counter(); _, I = flat.search(Q, 100); qs = nq / (time.perf_counter() - t)
    rec("fp32-flat", dim * 4, bt, qs, recall(gt, I, 10), recall(gt, I, 100))

    # --- faiss PQ at a byte budget matched to PCA*bits ---
    for bits in a.bits:
        m = max(1, (a.out_dim * bits) // 8)  # bytes per code ~ matched budget
        m = min(m, dim)
        while dim % m != 0:
            m -= 1
        try:
            t = time.perf_counter()
            pq = faiss.IndexPQ(dim, m, 8)
            pq.train(C[: min(N, 50000)]); pq.add(C)
            bt = time.perf_counter() - t
            t = time.perf_counter(); _, I = pq.search(Q, 100)
            qs = nq / (time.perf_counter() - t)
            rec(f"faiss-PQ(m={m})", m, bt, qs, recall(gt, I, 10), recall(gt, I, 100),
                f"matched to ~{bits}-bit budget")
        except Exception as e:
            print(f"  PQ m={m} failed: {e}", flush=True)

    # --- faiss HNSW fp32 (fast, no compression) ---
    t = time.perf_counter()
    hnsw = faiss.IndexHNSWFlat(dim, 32); hnsw.hnsw.efConstruction = 200; hnsw.add(C)
    bt = time.perf_counter() - t
    hnsw.hnsw.efSearch = 128
    t = time.perf_counter(); _, I = hnsw.search(Q, 100); qs = nq / (time.perf_counter() - t)
    rec("faiss-HNSW-fp32", dim * 4, bt, qs, recall(gt, I, 10), recall(gt, I, 100))

    # --- turboquant-pro: PCA-Matryoshka + TQ ---
    from turboquant_pro import PCAMatryoshka

    for bits in a.bits:
        t = time.perf_counter()
        pca = PCAMatryoshka(input_dim=dim, output_dim=a.out_dim)
        pca.fit(C[: min(N, 50000)])
        pipe = pca.with_quantizer(bits=bits)
        codes = pipe.compress_batch(C)
        recon = normalize(np.asarray(pipe.decompress_batch(codes), dtype=np.float32))
        rdim = recon.shape[1]  # decompress reconstructs to original dim (or PCA space)
        Qp = Q if rdim == dim else normalize(np.asarray(pca.transform(Q), dtype=np.float32))
        idx = faiss.IndexFlatIP(rdim); idx.add(recon)
        bt = time.perf_counter() - t
        try:
            bpv = float(pipe.estimate_storage(N)) / N
        except Exception:
            bpv = a.out_dim * bits / 8.0 + 4  # packed codes + L2 norm
        # single-stage
        t = time.perf_counter(); _, I1 = idx.search(Qp, 100)
        qs1 = nq / (time.perf_counter() - t)
        rec(f"tq-pro PCA{a.out_dim}+TQ{bits} (1-stage)", bpv, bt, qs1,
            recall(gt, I1, 10), recall(gt, I1, 100))
        # two-stage: oversample then rerank by exact fp32 on originals
        kk = 10 * a.oversample
        t = time.perf_counter()
        _, cand = idx.search(Qp, kk)
        rr = np.zeros((nq, 10), dtype=np.int64)
        for i in range(nq):
            c = cand[i]
            s = C[c] @ Q[i]
            rr[i] = c[np.argsort(-s)[:10]]
        qs2 = nq / (time.perf_counter() - t)
        rec(f"tq-pro PCA{a.out_dim}+TQ{bits} (+rerank x{a.oversample})", bpv, bt, qs2,
            recall(gt, rr, 10), float("nan"),
            "rerank uses retained fp32 originals")

    with open(a.json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {a.json}", flush=True)

    print("\n| method | bytes/vec | comp x | build s | qps | r@10 | r@100 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in res:
        print(f"| {r['method']} | {r['bytes_per_vec']} | {r['compression_x']} | "
              f"{r['build_s']} | {r['qps']} | {r['recall_at_10']} | {r['recall_at_100']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
