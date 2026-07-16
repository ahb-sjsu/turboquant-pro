# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Self-contained, CPU-only replay smoke for ``tqp replay``.

Reproduces the Track-1 mechanism with **no downloads and no GPU**: a fixed,
declared PCA-Matryoshka + low-bit recipe compresses a deterministic synthetic
corpus, and we measure the metric retrieval actually consumes — **recall@10**
of compressed-domain nearest neighbours against the exact ranking — at a real
compression ratio.

Recall, not reconstruction cosine, is the acceptance metric: cosine is the
promise this project shows can read ~0.97 while the ranking collapses (see
``docs/KV_KEYS_FINDING.md`` and ``a2_probe``). ``mean_cosine`` is reported only
as a labelled secondary diagnostic, for contrast.

    python benchmarks/replay_smoke.py --out results.json [--full]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from turboquant_pro.pca import PCAMatryoshka


def _synthetic_embeddings(n: int, dim: int, seed: int) -> np.ndarray:
    """Moderate-rank embeddings with spread directions — a stand-in corpus."""
    rng = np.random.default_rng(seed)
    rank = dim // 2
    basis = rng.standard_normal((rank, dim))
    scale = np.linspace(1.0, 0.3, rank)  # decaying spectrum
    coeffs = rng.standard_normal((n, rank)) * scale
    x = coeffs @ basis + 0.05 * rng.standard_normal((n, dim))
    return x.astype(np.float32)


def _cosine_topk(base: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    bn = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-30)
    qn = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-30)
    sims = qn @ bn.T
    return np.argsort(-sims, axis=1)[:, :k]


def _recall_at_k(exact: np.ndarray, approx: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(a) & set(e)) / k for a, e in zip(approx, exact)]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Track-1 recall replay smoke.")
    ap.add_argument("--out", default="results.json", help="results.json path")
    ap.add_argument(
        "--full", action="store_true", help="larger set (still CPU/seconds)"
    )
    ap.add_argument("--bits", type=int, default=4, help="quantizer bits (default 4)")
    ap.add_argument("--k", type=int, default=10, help="recall@k (default 10)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    n, dim, n_queries = (20000, 256, 500) if args.full else (4000, 128, 200)
    pca_dim = dim // 2
    emb = _synthetic_embeddings(n, dim, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    q_idx = rng.choice(n, size=n_queries, replace=False)
    queries = emb[q_idx]

    # Fixed, declared recipe — no cosine-driven search.
    pca = PCAMatryoshka(input_dim=dim, output_dim=pca_dim)
    pca.fit(emb[:1000])
    pipe = pca.with_quantizer(bits=args.bits, seed=args.seed)
    recon = np.stack([pipe.decompress(pipe.compress(v)) for v in emb])

    exact = _cosine_topk(emb, queries, args.k)  # ground truth: full-precision DB
    approx = _cosine_topk(recon, queries, args.k)  # compressed-domain DB
    recall = _recall_at_k(exact, approx, args.k)

    cos = np.sum(emb * recon, axis=1) / (
        np.linalg.norm(emb, axis=1) * np.linalg.norm(recon, axis=1) + 1e-30
    )

    results = {
        "claim": "track1_recall_smoke",
        "n": n,
        "dim": dim,
        "recipe": f"PCA-{pca_dim} + TQ{args.bits}",
        "ratio": round(float(pipe.compression_ratio), 3),
        "recall_at_10": round(recall, 6),
        "n_queries": n_queries,
        "secondary_diagnostic_mean_cosine": round(float(cos.mean()), 6),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
