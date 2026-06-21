#!/usr/bin/env python3
"""Vector-database retrieval-cost benchmark on REAL embeddings (PVLDB paper).

Loads an embedding matrix (.npy), holds out queries, computes exact cosine
ground truth, and compares storage / build / throughput / recall for:

  * fp32-flat        -- exact baseline (faiss IndexFlatIP)
  * faiss-PQ         -- product quantization at a matched byte budget
  * faiss-OPQ        -- PQ with an optimized rotation (stronger baseline)
  * faiss-IVFPQ      -- inverted-file + PQ (the production ANN index)
  * faiss-HNSW       -- graph index, fp32 (fast, no compression)
  * turboquant-pro   -- PCA-Matryoshka(out_dim) + TQ{bits}, single-stage and
                        +rerank (oversample then rerank by exact fp32 on the
                        retained originals -- the standard two-stage ANN
                        protocol; compressed-only storage is the bytes/vec col).

Reports bytes/vector, compression x, build time, queries/s, recall@10 and @100.
Use --methods to run a subset (e.g. skip the slow HNSW build between iterations).

Run on Atlas (CPU; thread-capped):
  OMP_NUM_THREADS=20 /home/claude/env/bin/python benchmark_vectordb.py \\
    --npy /archive/results_aesthetics/bip_sample_200k_labse.npy \\
    --queries 1000 --out-dim 256 --bits 2 3 4 \\
    --methods flat pq opq ivfpq tq --json /tmp/vectordb_bench.json
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np


def normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def exact_topk(Q: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((len(Q), k), dtype=np.int64)
    for s in range(0, len(Q), 256):
        sims = Q[s : s + 256] @ X.T
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(idx.shape[0]):
            idx[r] = idx[r][np.argsort(-sims[r, idx[r]])]
        out[s : s + 256] = idx
    return out


def recall(gt: np.ndarray, ap: np.ndarray, k: int) -> float:
    return float(
        np.mean([len(set(gt[i, :k]) & set(ap[i, :k])) / k for i in range(len(gt))])
    )


def rerank_top10(cand: np.ndarray, Q: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Two-stage: re-rank candidate ids by exact fp32 cosine on the originals."""
    rr = np.zeros((len(Q), 10), dtype=np.int64)
    for i in range(len(Q)):
        c = cand[i]
        s = C[c] @ Q[i]
        rr[i] = c[np.argsort(-s)[:10]]
    return rr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--out-dim", type=int, default=256)
    ap.add_argument("--bits", type=int, nargs="+", default=[3])
    ap.add_argument("--oversample", type=int, default=5)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument(
        "--methods",
        nargs="+",
        default=["flat", "pq", "opq", "ivfpq", "rabitq", "hnsw", "tq"],
    )
    ap.add_argument("--json", default="/tmp/vectordb_bench.json")
    a = ap.parse_args()
    M = set(a.methods)

    import faiss

    faiss.omp_set_num_threads(a.threads)

    X = normalize(np.load(a.npy, mmap_mode="r").astype(np.float32))
    nq = a.queries
    Q, C = X[:nq].copy(), X[nq:].copy()
    N, dim = C.shape
    print(f"corpus={N} dim={dim} queries={nq} methods={sorted(M)}", flush=True)

    print("exact ground truth...", flush=True)
    gt = exact_topk(Q, C, 100)
    train = C[: min(N, 50000)]
    res = []

    def rec(method, bpv, build_s, qps, r10, r100, note=""):
        row = dict(
            method=method,
            n=N,
            dim=dim,
            bytes_per_vec=round(bpv, 1),
            compression_x=round(dim * 4 / bpv, 1),
            build_s=round(build_s, 2),
            qps=round(qps, 1),
            recall_at_10=round(r10, 4),
            recall_at_100=round(r100, 4),
            note=note,
        )
        res.append(row)
        print(" ", row, flush=True)

    def budget_m(bits):
        m = min(max(1, (a.out_dim * bits) // 8), dim)
        while dim % m != 0:
            m -= 1
        return m

    if "flat" in M:
        t = time.perf_counter()
        flat = faiss.IndexFlatIP(dim)
        flat.add(C)
        bt = time.perf_counter() - t
        t = time.perf_counter()
        _, nn = flat.search(Q, 100)
        qs = nq / (time.perf_counter() - t)
        rec("fp32-flat", dim * 4, bt, qs, recall(gt, nn, 10), recall(gt, nn, 100))

    for fac, tag in (("PQ", "pq"), ("OPQ", "opq")):
        if tag not in M:
            continue
        for bits in a.bits:
            m = budget_m(bits)
            try:
                spec = f"PQ{m}x8" if fac == "PQ" else f"OPQ{m},PQ{m}"
                t = time.perf_counter()
                index = faiss.index_factory(dim, spec, faiss.METRIC_INNER_PRODUCT)
                index.train(train)
                index.add(C)
                bt = time.perf_counter() - t
                t = time.perf_counter()
                _, nn = index.search(Q, 100)
                qs = nq / (time.perf_counter() - t)
                rec(
                    f"faiss-{fac}(m={m})",
                    m,
                    bt,
                    qs,
                    recall(gt, nn, 10),
                    recall(gt, nn, 100),
                    f"~{bits}-bit budget",
                )
                _, cand = index.search(Q, 10 * a.oversample)
                rr = rerank_top10(cand, Q, C)
                rec(
                    f"faiss-{fac}(m={m}) +rerank x{a.oversample}",
                    m,
                    bt,
                    qs,
                    recall(gt, rr, 10),
                    float("nan"),
                    "rerank with fp32 originals",
                )
            except Exception as e:
                print(f"  {fac} m={m} failed: {e}", flush=True)

    if "ivfpq" in M:
        nlist = min(4096, max(64, int(8 * math.sqrt(N))))
        for bits in a.bits:
            m = budget_m(bits)
            try:
                t = time.perf_counter()
                index = faiss.index_factory(
                    dim, f"IVF{nlist},PQ{m}", faiss.METRIC_INNER_PRODUCT
                )
                index.train(C[: min(N, 100000)])
                index.add(C)
                index.nprobe = min(64, nlist)
                bt = time.perf_counter() - t
                t = time.perf_counter()
                _, nn = index.search(Q, 100)
                qs = nq / (time.perf_counter() - t)
                rec(
                    f"faiss-IVFPQ(m={m},nlist={nlist})",
                    m,
                    bt,
                    qs,
                    recall(gt, nn, 10),
                    recall(gt, nn, 100),
                    f"nprobe={index.nprobe}",
                )
                _, cand = index.search(Q, 10 * a.oversample)
                rr = rerank_top10(cand, Q, C)
                rec(
                    f"faiss-IVFPQ(m={m}) +rerank x{a.oversample}",
                    m,
                    bt,
                    qs,
                    recall(gt, rr, 10),
                    float("nan"),
                    "rerank with fp32 originals",
                )
            except Exception as e:
                print(f"  IVFPQ m={m} failed: {e}", flush=True)

    if "rabitq" in M:  # 2024 SOTA: ~1 bit/dim (-> dim/8 bytes, ~32x at 768-d)
        try:
            t = time.perf_counter()
            index = faiss.index_factory(dim, "RaBitQ", faiss.METRIC_INNER_PRODUCT)
            index.train(train)
            index.add(C)
            bt = time.perf_counter() - t
            t = time.perf_counter()
            _, nn = index.search(Q, 100)
            qs = nq / (time.perf_counter() - t)
            bpv = dim / 8.0
            rec(
                "faiss-RaBitQ",
                bpv,
                bt,
                qs,
                recall(gt, nn, 10),
                recall(gt, nn, 100),
                "1-bit",
            )
            _, cand = index.search(Q, 10 * a.oversample)
            rr = rerank_top10(cand, Q, C)
            rec(
                f"faiss-RaBitQ +rerank x{a.oversample}",
                bpv,
                bt,
                qs,
                recall(gt, rr, 10),
                float("nan"),
                "rerank with fp32 originals",
            )
        except Exception as e:
            print(f"  RaBitQ failed: {e}", flush=True)

    if "hnsw" in M:
        t = time.perf_counter()
        hnsw = faiss.IndexHNSWFlat(dim, 32)
        hnsw.hnsw.efConstruction = 200
        hnsw.add(C)
        bt = time.perf_counter() - t
        hnsw.hnsw.efSearch = 128
        t = time.perf_counter()
        _, nn = hnsw.search(Q, 100)
        qs = nq / (time.perf_counter() - t)
        rec("faiss-HNSW-fp32", dim * 4, bt, qs, recall(gt, nn, 10), recall(gt, nn, 100))

    if "tq" in M:
        from turboquant_pro import PCAMatryoshka

        for bits in a.bits:
            t = time.perf_counter()
            pca = PCAMatryoshka(input_dim=dim, output_dim=a.out_dim)
            pca.fit(train)
            pipe = pca.with_quantizer(bits=bits)
            codes = pipe.compress_batch(C)
            recon = normalize(
                np.asarray(pipe.decompress_batch(codes), dtype=np.float32)
            )
            rdim = recon.shape[1]
            Qp = (
                Q
                if rdim == dim
                else normalize(np.asarray(pca.transform(Q), dtype=np.float32))
            )
            idx = faiss.IndexFlatIP(rdim)
            idx.add(recon)
            bt = time.perf_counter() - t
            try:
                bpv = float(pipe.estimate_storage(N)) / N
            except Exception:
                bpv = a.out_dim * bits / 8.0 + 4
            t = time.perf_counter()
            _, I1 = idx.search(Qp, 100)
            qs1 = nq / (time.perf_counter() - t)
            rec(
                f"tq-pro PCA{a.out_dim}+TQ{bits} (1-stage)",
                bpv,
                bt,
                qs1,
                recall(gt, I1, 10),
                recall(gt, I1, 100),
            )
            kk = 10 * a.oversample
            t = time.perf_counter()
            _, cand = idx.search(Qp, kk)
            rr = np.zeros((nq, 10), dtype=np.int64)
            for i in range(nq):
                c = cand[i]
                s = C[c] @ Q[i]
                rr[i] = c[np.argsort(-s)[:10]]
            qs2 = nq / (time.perf_counter() - t)
            rec(
                f"tq-pro PCA{a.out_dim}+TQ{bits} (+rerank x{a.oversample})",
                bpv,
                bt,
                qs2,
                recall(gt, rr, 10),
                float("nan"),
                "rerank uses retained fp32 originals",
            )

    with open(a.json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nwrote {a.json}", flush=True)

    print("\n| method | bytes/vec | comp x | build s | qps | r@10 | r@100 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in res:
        print(
            f"| {r['method']} | {r['bytes_per_vec']} | {r['compression_x']} | "
            f"{r['build_s']} | {r['qps']} | {r['recall_at_10']} | {r['recall_at_100']} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
