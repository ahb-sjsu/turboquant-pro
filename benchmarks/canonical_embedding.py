#!/usr/bin/env python3
"""Canonical embedding-compression benchmark harness (Review-1 deliverable).

ONE table, ALL methods, IDENTICAL rerank protocol — so the headline claim
("beats RaBitQ on recall, ties OPQ at scale, builds faster") is reproducible
in a single call. Used by notebooks/claims/00_canonical_sota_embedding.ipynb
(auto-downloads public GloVe) and callable directly on any corpus.

Methods (all at a matched ~out_dim*bits byte budget where applicable):
  fp32-flat  PQ  OPQ  IVFPQ  RaBitQ  PCA-only  TQ-only  PCA+TQ  ADCIndex

Every ANN method is measured twice: single-stage, and +rerank with the SAME
oversample factor (candidates = 10 * oversample), reranked by exact fp32 cosine
on the retained originals. bytes/vector is computed ANALYTICALLY (not via
estimate_storage(), which currently reports fixed 1024->384 dims regardless of
the actual pipeline — see docs/claims.md note).

    from canonical_embedding import run_canonical, to_markdown
    rows = run_canonical(C, Q, gt, out_dim=64, bits=3, oversample=5)
    print(to_markdown(rows))
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------- primitives
def normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-30)


def exact_topk(Q: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
    """Exact cosine top-k ground truth (for corpora without provided neighbors)."""
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


def _rerank(cand: np.ndarray, Q: np.ndarray, C: np.ndarray, k: int = 10) -> np.ndarray:
    """Two-stage: re-rank candidate ids by exact fp32 cosine on the originals."""
    rr = np.full((len(Q), k), -1, dtype=np.int64)
    for i in range(len(Q)):
        c = cand[i][cand[i] >= 0]
        if len(c) == 0:
            continue
        s = C[c] @ Q[i]
        top = c[np.argsort(-s)[:k]]
        rr[i, : len(top)] = top
    return rr


def _divisor_m(target_m: int, dim: int) -> int:
    m = min(max(1, target_m), dim)
    while dim % m != 0:
        m -= 1
    return m


# ---------------------------------------------------------------- main driver
def run_canonical(
    C: np.ndarray,
    Q: np.ndarray,
    gt: np.ndarray,
    out_dim: int = 64,
    bits: int = 3,
    oversample: int = 5,
    methods: Sequence[str] | None = None,
    threads: int = 8,
    reps: int = 2,
    train_cap: int = 200_000,
) -> list[dict]:
    """Run the canonical method ladder. C/Q assumed L2-normalized. gt = top-k ids.

    Returns a list of row dicts with identical schema across methods so the
    caller can render one table. `oversample` is shared by EVERY +rerank row.
    """
    import faiss

    faiss.omp_set_num_threads(threads)
    from turboquant_pro import ADCIndex, PCAMatryoshka

    N, dim = C.shape
    nq = len(Q)
    kcand = 10 * oversample
    train = C[: min(N, train_cap)]
    all_methods = ["flat", "pq", "opq", "ivfpq", "rabitq", "pca", "tq", "pca_tq", "adc"]
    M = list(methods) if methods else all_methods
    rows: list[dict] = []

    def bench(fn) -> float:
        best = 1e30
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t)
        return nq / best

    def add(method, bpv, build_s, qps1, qps_rr, r10_1, r10_rr, r100, note=""):
        rows.append(
            dict(
                method=method,
                n=N,
                dim=dim,
                bytes_per_vec=round(float(bpv), 1),
                compression_x=round(dim * 4 / float(bpv), 1),
                build_s=round(float(build_s), 3),
                qps_1stage=None if qps1 is None else round(float(qps1), 1),
                qps_rerank=None if qps_rr is None else round(float(qps_rr), 1),
                recall_at_10=None if r10_1 is None else round(float(r10_1), 4),
                recall_at_10_rerank=None if r10_rr is None else round(float(r10_rr), 4),
                recall_at_100=None if r100 is None else round(float(r100), 4),
                ram_mb=round(float(bpv) * N / 1e6, 1),
                note=note,
            )
        )
        print("  ", rows[-1], flush=True)

    # -- fp32-flat exact baseline -------------------------------------------
    if "flat" in M:
        t = time.perf_counter()
        flat = faiss.IndexFlatIP(dim)
        flat.add(C)
        bt = time.perf_counter() - t
        _, nn = flat.search(Q, 100)
        qps = bench(lambda: flat.search(Q, 10))
        add("fp32-flat", dim * 4, bt, qps, None,
            recall(gt, nn, 10), None, recall(gt, nn, 100), "exact baseline")

    # -- PQ / OPQ at matched byte budget ------------------------------------
    for fac, tag in (("PQ", "pq"), ("OPQ", "opq")):
        if tag not in M:
            continue
        m = _divisor_m((out_dim * bits) // 8, dim)
        try:
            spec = f"PQ{m}x8" if fac == "PQ" else f"OPQ{m},PQ{m}"
            t = time.perf_counter()
            index = faiss.index_factory(dim, spec, faiss.METRIC_INNER_PRODUCT)
            index.train(train)
            index.add(C)
            bt = time.perf_counter() - t
            _, nn = index.search(Q, 100)
            _, cand = index.search(Q, kcand)
            rr = _rerank(cand, Q, C)
            q1 = bench(lambda index=index: index.search(Q, 10))
            qr = bench(lambda index=index: index.search(Q, kcand))
            add(f"faiss-{fac}(m={m})", m, bt, q1, qr,
                recall(gt, nn, 10), recall(gt, rr, 10), recall(gt, nn, 100),
                f"~{bits}-bit budget; +rerank x{oversample}")
        except Exception as e:  # noqa: BLE001
            print(f"  {fac} failed: {e}", flush=True)

    # -- IVFPQ (production ANN index) ---------------------------------------
    if "ivfpq" in M:
        import math

        nlist = min(4096, max(64, int(8 * math.sqrt(N))))
        m = _divisor_m((out_dim * bits) // 8, dim)
        try:
            t = time.perf_counter()
            index = faiss.index_factory(dim, f"IVF{nlist},PQ{m}", faiss.METRIC_INNER_PRODUCT)
            index.train(train)
            index.add(C)
            index.nprobe = min(64, nlist)
            bt = time.perf_counter() - t
            _, nn = index.search(Q, 100)
            _, cand = index.search(Q, kcand)
            rr = _rerank(cand, Q, C)
            q1 = bench(lambda index=index: index.search(Q, 10))
            qr = bench(lambda index=index: index.search(Q, kcand))
            add(f"faiss-IVFPQ(m={m},nlist={nlist})", m, bt, q1, qr,
                recall(gt, nn, 10), recall(gt, rr, 10), recall(gt, nn, 100),
                f"nprobe={index.nprobe}; +rerank x{oversample}")
        except Exception as e:  # noqa: BLE001
            print(f"  IVFPQ failed: {e}", flush=True)

    # -- RaBitQ (2024 SOTA, ~1 bit/dim) -------------------------------------
    if "rabitq" in M:
        try:
            t = time.perf_counter()
            index = faiss.index_factory(dim, "RaBitQ", faiss.METRIC_INNER_PRODUCT)
            index.train(train)
            index.add(C)
            bt = time.perf_counter() - t
            _, nn = index.search(Q, 100)
            _, cand = index.search(Q, kcand)
            rr = _rerank(cand, Q, C)
            q1 = bench(lambda index=index: index.search(Q, 10))
            qr = bench(lambda index=index: index.search(Q, kcand))
            add("faiss-RaBitQ", dim / 8.0, bt, q1, qr,
                recall(gt, nn, 10), recall(gt, rr, 10), recall(gt, nn, 100),
                f"1-bit/dim; +rerank x{oversample}")
        except Exception as e:  # noqa: BLE001
            print(f"  RaBitQ unavailable in this faiss build: {e}", flush=True)

    # -- PCA-only (truncation, fp32 kept dims) — isolates dimension reduction
    if "pca" in M:
        pca = PCAMatryoshka(input_dim=dim, output_dim=out_dim)
        pca.fit(train)
        t = time.perf_counter()
        Cp = normalize(np.asarray(pca.transform(C), dtype=np.float32))
        Qp = normalize(np.asarray(pca.transform(Q), dtype=np.float32))
        idx = faiss.IndexFlatIP(out_dim)
        idx.add(Cp)
        bt = time.perf_counter() - t
        _, nn = idx.search(Qp, 100)
        _, cand = idx.search(Qp, kcand)
        rr = _rerank(cand, Q, C)
        q1 = bench(lambda: idx.search(Qp, 10))
        qr = bench(lambda: idx.search(Qp, kcand))
        ev = float(np.sum(pca._eigenvalues) / np.sum(pca._all_eigenvalues))
        add(f"PCA-only({out_dim}d, fp32)", out_dim * 4, bt, q1, qr,
            recall(gt, nn, 10), recall(gt, rr, 10), recall(gt, nn, 100),
            f"truncation only; var={ev:.2f}; +rerank x{oversample}")

    # -- TQ-only (scalar quant on full dim, no truncation) — isolates SQ ----
    if "tq" in M:
        pcaf = PCAMatryoshka(input_dim=dim, output_dim=dim)
        pcaf.fit(train)
        pipe = pcaf.with_quantizer(bits=bits)
        t = time.perf_counter()
        codes = pipe.compress_batch(C)
        recon = normalize(np.asarray(pipe.decompress_batch(codes), dtype=np.float32))
        idx = faiss.IndexFlatIP(dim)
        idx.add(recon)
        bt = time.perf_counter() - t
        _, nn = idx.search(Q, 100)
        _, cand = idx.search(Q, kcand)
        rr = _rerank(cand, Q, C)
        q1 = bench(lambda: idx.search(Q, 10))
        qr = bench(lambda: idx.search(Q, kcand))
        add(f"TQ-only({dim}d, {bits}b)", dim * bits / 8.0, bt, q1, qr,
            recall(gt, nn, 10), recall(gt, rr, 10), recall(gt, nn, 100),
            f"scalar-quant only; +rerank x{oversample}")

    # -- PCA + TQ (the combined pipeline) -----------------------------------
    if "pca_tq" in M:
        pca = PCAMatryoshka(input_dim=dim, output_dim=out_dim)
        pca.fit(train)
        pipe = pca.with_quantizer(bits=bits)
        t = time.perf_counter()
        codes = pipe.compress_batch(C)
        recon = normalize(np.asarray(pipe.decompress_batch(codes), dtype=np.float32))
        rdim = recon.shape[1]
        Qp = normalize(np.asarray(pca.transform(Q), dtype=np.float32)) if rdim != dim else Q
        idx = faiss.IndexFlatIP(rdim)
        idx.add(recon)
        bt = time.perf_counter() - t
        _, nn = idx.search(Qp, 100)
        _, cand = idx.search(Qp, kcand)
        rr = _rerank(cand, Q, C)
        q1 = bench(lambda: idx.search(Qp, 10))
        qr = bench(lambda: idx.search(Qp, kcand))
        add(f"PCA+TQ({out_dim}d, {bits}b)", out_dim * bits / 8.0, bt, q1, qr,
            recall(gt, nn, 10), recall(gt, rr, 10), recall(gt, nn, 100),
            f"combined pipeline; +rerank x{oversample}")

    # -- ADCIndex (compressed-domain search, no reconstruction) -------------
    if "adc" in M:
        pca = PCAMatryoshka(input_dim=dim, output_dim=out_dim)
        pca.fit(train)
        t = time.perf_counter()
        index = ADCIndex(pca.with_quantizer(bits=bits)).add(C)
        bt = time.perf_counter() - t
        i1, _ = index.search(Q, k=100)
        ir = index.search(Q, k=10, rerank=oversample, originals=C)
        q1 = bench(lambda: index.search(Q, k=10))
        qr = bench(lambda: index.search(Q, k=10, rerank=oversample, originals=C))
        add(f"ADCIndex({out_dim}d, {bits}b)", out_dim * bits / 8.0, bt, q1, qr,
            recall(gt, np.asarray(i1), 10), recall(gt, np.asarray(ir), 10),
            recall(gt, np.asarray(i1), 100),
            f"compressed-domain ADC; +rerank x{oversample}")

    return rows


# ---------------------------------------------------------------- rendering
COLS = [
    ("method", "Method", "l"),
    ("n", "N", "r"),
    ("dim", "Dim", "r"),
    ("compression_x", "Comp x", "r"),
    ("bytes_per_vec", "B/vec", "r"),
    ("ram_mb", "RAM MB", "r"),
    ("recall_at_10", "R@10", "r"),
    ("recall_at_10_rerank", "R@10 +rr", "r"),
    ("recall_at_100", "R@100", "r"),
    ("qps_1stage", "QPS", "r"),
    ("build_s", "Build s", "r"),
    ("note", "Notes", "l"),
]


def to_markdown(rows: list[dict]) -> str:
    head = "| " + " | ".join(h for _, h, _ in COLS) + " |"
    sep = "|" + "|".join("---:" if a == "r" else "---" for _, _, a in COLS) + "|"
    lines = [head, sep]
    for r in rows:
        cells = []
        for key, _, _ in COLS:
            v = r.get(key)
            cells.append("-" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":  # smoke test on synthetic data with exact ground truth
    rng = np.random.default_rng(0)
    dim = 96
    C = normalize(rng.standard_normal((20_000, dim)).astype(np.float32))
    Q = normalize(rng.standard_normal((200, dim)).astype(np.float32))
    gt = exact_topk(Q, C, 100)
    rows = run_canonical(C, Q, gt, out_dim=48, bits=3, oversample=5, threads=4)
    print("\n" + to_markdown(rows))
