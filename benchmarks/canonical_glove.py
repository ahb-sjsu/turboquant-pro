# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond
# MIT License

"""Executable GloVe recall claim for ``tqp replay`` — hermetic or full.

This is the runnable core of the Track-1 embedding claim on real public data:
a fixed, declared PCA-Matryoshka (full-dim rotation) + low-bit recipe feeding
compressed-domain ADC search preserves **recall@10** — the metric retrieval
actually consumes — against the exact ranking, at a real compression ratio,
with exact rerank as the second stage. It complements the fuller headline table
in ``notebooks/claims/00_canonical_sota_embedding.ipynb`` (claim
``embedding_27x_high_recall``) with a gated, one-command reproduction.

Two modes, one recipe:

* ``--small`` (CI, hermetic): a tiny **real** glove-100-angular subset bundled
  at ``benchmarks/fixtures/glove_tiny.npz`` (public data, same distribution and
  dimensionality, just few vectors). No network, no GPU, no faiss — numpy plus
  ``turboquant_pro`` only. This is what ``tqp replay`` runs in CI.
* ``--full`` (Atlas / local): the real 1.18M-vector glove-100-angular from an
  ann-benchmarks HDF5. Resolved from ``$TQP_GLOVE_HDF5``, else a cached
  ``glove-100-angular.hdf5`` in the cwd, else downloaded. Uses the provided
  exact ground-truth neighbours.

Acceptance is **reranked recall@10** (the consumer metric) at the achieved
compression, never reconstruction cosine — cosine is the reading this project
shows can sit near 0.97 while the ranking collapses (see
``docs/KV_KEYS_FINDING.md`` and ``a2_probe``). ``mean_cosine`` is emitted only
as a labelled secondary diagnostic, for contrast.

    python benchmarks/canonical_glove.py --small --out results.json
    python benchmarks/canonical_glove.py --full  --out results.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from turboquant_pro import ADCIndex, PCAMatryoshka

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "glove_tiny.npz")
GLOVE_URL = "http://ann-benchmarks.com/glove-100-angular.hdf5"


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def _exact_topk(queries: np.ndarray, base: np.ndarray, k: int) -> np.ndarray:
    """Exact cosine (inner-product on normalized) top-k ground truth."""
    out = np.zeros((len(queries), k), dtype=np.int64)
    for s in range(0, len(queries), 256):
        sims = queries[s : s + 256] @ base.T
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(idx.shape[0]):
            idx[r] = idx[r][np.argsort(-sims[r, idx[r]])]
        out[s : s + 256] = idx
    return out


def _recall_at_k(gt: np.ndarray, approx: np.ndarray, k: int) -> float:
    return float(
        np.mean([len(set(a[:k]) & set(g[:k])) / k for g, a in zip(gt, approx)])
    )


def _load_small() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A tiny real GloVe subset bundled in-tree; exact GT recomputed."""
    if not os.path.exists(FIXTURE):
        raise FileNotFoundError(
            f"missing hermetic fixture {FIXTURE}; regenerate with "
            "benchmarks/make_glove_fixture.py"
        )
    with np.load(FIXTURE) as npz:
        base = _normalize(npz["train"].astype(np.float32))
        queries = _normalize(npz["test"].astype(np.float32))
    gt = _exact_topk(queries, base, 100)
    return base, queries, gt


def _load_full(
    queries: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real glove-100-angular via ann-benchmarks HDF5 (cached / downloaded)."""
    import h5py

    path = os.environ.get("TQP_GLOVE_HDF5") or "glove-100-angular.hdf5"
    if not os.path.exists(path):
        import urllib.request

        print(f"downloading {GLOVE_URL} -> {path} ...", flush=True)
        urllib.request.urlretrieve(GLOVE_URL, path)
    with h5py.File(path, "r") as f:
        base = _normalize(np.asarray(f["train"], dtype=np.float32))
        q = _normalize(np.asarray(f["test"], dtype=np.float32))
        gt = np.asarray(f["neighbors"], dtype=np.int64)  # provided exact top-100
    return base, q[:queries], gt[:queries, :100]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GloVe recall@10 replay (ADC search).")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--small",
        action="store_true",
        help="hermetic tiny bundled GloVe subset (default; CI-safe)",
    )
    mode.add_argument(
        "--full", action="store_true", help="real 1.18M glove-100-angular"
    )
    ap.add_argument("--out", default="results.json", help="results.json path")
    ap.add_argument(
        "--out-dim",
        type=int,
        default=0,
        help="PCA output dim (0 = full dim, rotation only; the GloVe-optimal "
        "config — truncation discards real variance on this concentrated set)",
    )
    ap.add_argument("--bits", type=int, default=3, help="quantizer bits")
    ap.add_argument("--oversample", type=int, default=12, help="rerank candidate mult")
    ap.add_argument("--k", type=int, default=10, help="recall@k")
    ap.add_argument("--queries", type=int, default=1000, help="query count (--full)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    full = args.full  # --small is the default when neither flag is given
    if full:
        base, queries, gt = _load_full(args.queries)
        dataset = "glove-100-angular"
    else:
        base, queries, gt = _load_small()
        dataset = "glove-100-angular (tiny bundled subset)"

    n, dim = base.shape
    out_dim = dim if args.out_dim <= 0 else min(args.out_dim, dim)

    # Fixed, declared recipe — no cosine-driven search. PCA-Matryoshka to
    # `out_dim`, `bits`-bit quantizer, compressed-domain ADC search, then exact
    # rerank of the top `k * oversample` candidates.
    pca = PCAMatryoshka(input_dim=dim, output_dim=out_dim)
    pca.fit(base[: min(n, 200_000)])
    pipe = pca.with_quantizer(bits=args.bits, seed=args.seed)
    index = ADCIndex(pipe).add(base)

    single, _ = index.search(queries, k=args.k)
    reranked = index.search(queries, k=args.k, rerank=args.oversample, originals=base)
    recall = _recall_at_k(gt, np.asarray(single), args.k)
    recall_rr = _recall_at_k(gt, np.asarray(reranked), args.k)

    # Labelled secondary diagnostic ONLY — never the acceptance metric.
    # decompress inverts PCA back to the original space, so compare there; if a
    # config keeps the reconstruction in PCA space, compare against the
    # projection instead.
    recon = _normalize(np.asarray(pipe.decompress_batch(pipe.compress_batch(base))))
    if recon.shape[1] == dim:
        ref = _normalize(base)
    else:
        ref = _normalize(np.asarray(pca.transform(base), dtype=np.float32))
    cos = float(np.mean(np.sum(ref * recon, axis=1)))

    ratio = float(pipe.compression_ratio)
    results = {
        "claim": "embedding_glove_recall",
        "dataset": dataset,
        "n": int(n),
        "dim": int(dim),
        "out_dim": int(out_dim),
        "bits": int(args.bits),
        "recipe": f"PCA-{out_dim} + TQ{args.bits} + ADC, rerank x{args.oversample}",
        "compression_ratio": round(ratio, 3),
        "recall_at_10": round(recall, 6),
        "recall_at_10_rerank": round(recall_rr, 6),
        "n_queries": int(len(queries)),
        "oversample": int(args.oversample),
        "secondary_diagnostic_mean_cosine": round(cos, 6),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
