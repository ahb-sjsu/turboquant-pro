"""
Autotune — Find the optimal compression configuration for your data.

Connects to a PostgreSQL database, samples embeddings, and evaluates
all PCA dimension x bit-width combinations to find the Pareto-optimal
configurations for your specific corpus.

Usage:
    turboquant-pro autotune --source "dbname=mydb user=me" \
        --table chunks --column embedding --min-recall 0.95

    turboquant-pro autotune --source "dbname=atlas user=claude" \
        --sample-size 5000 --min-recall 0.80
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("turboquant-autotune")


# ---------------------------------------------------------------------------
# Configuration space
# ---------------------------------------------------------------------------

DEFAULT_PCA_DIMS = [128, 256, 384, 512]
DEFAULT_BIT_WIDTHS = [2, 3, 4]
DEFAULT_SAMPLE_SIZE = 5000
DEFAULT_QUERY_COUNT = 50
DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TuneResult:
    """Result of evaluating one compression configuration."""

    pca_dim: int
    bits: int
    compression_ratio: float
    mean_cosine: float
    min_cosine: float
    recall_at_k: float
    top_k: int
    original_mb: float
    compressed_mb: float
    saved_mb: float
    fit_time_s: float
    compress_time_s: float
    eval_time_s: float
    variance_explained: float
    pareto_optimal: bool = False
    recommended: bool = False


@dataclass
class AutotuneReport:
    """Complete autotune report."""

    source: str
    table: str
    column: str
    input_dim: int
    sample_size: int
    corpus_size: int
    query_count: int
    top_k: int
    min_recall: float
    results: list[TuneResult] = field(default_factory=list)
    recommended: TuneResult | None = None
    timestamp: str = ""
    total_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Core autotune logic
# ---------------------------------------------------------------------------


def _parse_embedding(raw) -> np.ndarray:
    """Parse an embedding from PostgreSQL (pgvector string or list)."""
    if isinstance(raw, str):
        # pgvector format: '[0.1,0.2,...]'
        cleaned = raw.strip("[] ")
        return np.fromstring(cleaned, sep=",", dtype=np.float32)
    if isinstance(raw, (list, tuple)):
        return np.array(raw, dtype=np.float32)
    return np.array(raw, dtype=np.float32)


def sample_embeddings(
    dsn: str,
    table: str,
    column: str,
    sample_size: int,
) -> tuple:
    """Sample embeddings from PostgreSQL.

    Returns (embeddings_array, total_count, input_dim).
    """
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 required. Install with:")
        print("  pip install turboquant-pro[pgvector]")
        sys.exit(1)

    conn = psycopg2.connect(dsn)
    cur = conn.cursor()

    # Get total count
    cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
    total = cur.fetchone()[0]

    # Get dimension from first embedding
    cur.execute(
        f"SELECT {column} FROM {table} "  # noqa: S608
        f"WHERE {column} IS NOT NULL LIMIT 1"
    )
    row = cur.fetchone()
    if row is None:
        print(f"ERROR: No embeddings found in {table}.{column}")
        sys.exit(1)

    raw = row[0]
    sample_emb = _parse_embedding(raw)
    input_dim = len(sample_emb)

    # Sample randomly
    n = min(sample_size, total)
    cur.execute(
        f"SELECT {column} FROM {table} "  # noqa: S608
        f"WHERE {column} IS NOT NULL "
        f"ORDER BY RANDOM() LIMIT {n}"
    )
    rows = cur.fetchall()
    embeddings = np.array([_parse_embedding(r[0]) for r in rows])

    conn.close()
    logger.info(
        "Sampled %d/%d embeddings (%d-dim) from %s.%s",
        len(embeddings),
        total,
        input_dim,
        table,
        column,
    )
    return embeddings, total, input_dim


def compute_recall_at_k(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_queries: int,
    k: int,
) -> float:
    """Compute recall@k: fraction of true top-k in approximate top-k."""
    if len(original) < k + n_queries:
        return 0.0

    # Use first n_queries as queries, rest as corpus
    queries = original[:n_queries]
    corpus = original[n_queries:]
    recon_corpus = reconstructed[n_queries:]

    if len(corpus) < k:
        return 0.0

    # Normalize
    q_norm = queries / np.maximum(np.linalg.norm(queries, axis=1, keepdims=True), 1e-30)
    c_norm = corpus / np.maximum(np.linalg.norm(corpus, axis=1, keepdims=True), 1e-30)
    r_norm = recon_corpus / np.maximum(
        np.linalg.norm(recon_corpus, axis=1, keepdims=True), 1e-30
    )

    # Exact rankings
    exact_sims = q_norm @ c_norm.T
    exact_top_k = np.argsort(-exact_sims, axis=1)[:, :k]

    # Approximate rankings (search over reconstructed)
    approx_sims = q_norm @ r_norm.T
    approx_top_k = np.argsort(-approx_sims, axis=1)[:, :k]

    # Recall
    total_recall = 0.0
    for i in range(n_queries):
        exact_set = set(exact_top_k[i])
        approx_set = set(approx_top_k[i])
        total_recall += len(exact_set & approx_set) / k

    return total_recall / n_queries


def evaluate_config(
    embeddings: np.ndarray,
    pca_dim: int,
    bits: int,
    n_queries: int,
    top_k: int,
    corpus_size: int,
) -> TuneResult:
    """Evaluate one PCA dim x bit-width configuration."""
    from .pca import PCAMatryoshka, PCAMatryoshkaPipeline
    from .pgvector import TurboQuantPGVector

    input_dim = embeddings.shape[1]

    # Fit PCA
    t0 = time.monotonic()
    pca = PCAMatryoshka(input_dim=input_dim, output_dim=pca_dim)
    fit_result = pca.fit(embeddings)
    fit_time = time.monotonic() - t0

    # Create pipeline
    tq = TurboQuantPGVector(dim=pca_dim, bits=bits, seed=42)
    pipeline = PCAMatryoshkaPipeline(pca=pca, quantizer=tq)

    # Compress batch
    t0 = time.monotonic()
    compressed = pipeline.compress_batch(embeddings)
    compress_time = time.monotonic() - t0

    # Decompress for quality measurement
    t0 = time.monotonic()
    reconstructed = pipeline.decompress_batch(compressed)

    # Cosine similarity
    mean_cos, min_cos, _ = pipeline.batch_cosine_similarity(embeddings)

    # Recall@k
    recall = compute_recall_at_k(embeddings, reconstructed, n_queries, top_k)
    eval_time = time.monotonic() - t0

    # Storage estimates for full corpus
    storage = PCAMatryoshkaPipeline.estimate_storage(
        corpus_size, input_dim, pca_dim, bits
    )

    return TuneResult(
        pca_dim=pca_dim,
        bits=bits,
        compression_ratio=storage["ratio"],
        mean_cosine=round(mean_cos, 4),
        min_cosine=round(min_cos, 4),
        recall_at_k=round(recall, 3),
        top_k=top_k,
        original_mb=storage["original_mb"],
        compressed_mb=storage["compressed_mb"],
        saved_mb=storage["saved_mb"],
        fit_time_s=round(fit_time, 2),
        compress_time_s=round(compress_time, 2),
        eval_time_s=round(eval_time, 2),
        variance_explained=round(fit_result.total_variance_explained, 4),
    )


def find_pareto_optimal(results: list[TuneResult]) -> list[TuneResult]:
    """Mark Pareto-optimal configs (best recall for compression level)."""
    # Sort by compression ratio ascending
    sorted_results = sorted(results, key=lambda r: r.compression_ratio)

    pareto = []
    best_recall = -1.0
    for r in sorted_results:
        if r.recall_at_k > best_recall:
            r.pareto_optimal = True
            pareto.append(r)
            best_recall = r.recall_at_k

    return pareto


def recommend(
    results: list[TuneResult],
    min_recall: float,
) -> TuneResult | None:
    """Recommend the highest-compression config meeting min_recall."""
    candidates = [r for r in results if r.recall_at_k >= min_recall]
    if not candidates:
        return None

    # Highest compression ratio among qualifying configs
    best = max(candidates, key=lambda r: r.compression_ratio)
    best.recommended = True
    return best


def run_autotune(
    dsn: str,
    table: str = "chunks",
    column: str = "embedding",
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    pca_dims: list[int] | None = None,
    bit_widths: list[int] | None = None,
    min_recall: float = 0.80,
    n_queries: int = DEFAULT_QUERY_COUNT,
    top_k: int = DEFAULT_TOP_K,
    output: str | None = None,
) -> AutotuneReport:
    """Run the full autotune sweep."""
    if pca_dims is None:
        pca_dims = DEFAULT_PCA_DIMS
    if bit_widths is None:
        bit_widths = DEFAULT_BIT_WIDTHS

    t_start = time.monotonic()

    # Sample
    print(f"Sampling {sample_size} embeddings from {table}.{column}...")
    embeddings, corpus_size, input_dim = sample_embeddings(
        dsn, table, column, sample_size
    )

    # Evaluate all configurations
    results: list[TuneResult] = []
    configs = [(d, b) for d in pca_dims for b in bit_widths if d <= input_dim]

    print(f"\nEvaluating {len(configs)} configurations...")
    print(
        f"{'Config':>20s}  {'Ratio':>6s}  {'Cosine':>7s}  "
        f"{'Recall':>7s}  {'Var%':>5s}  {'Time':>5s}"
    )
    print("-" * 62)

    for pca_dim, bits in configs:
        label = f"PCA-{pca_dim} + TQ{bits}"
        try:
            result = evaluate_config(
                embeddings,
                pca_dim,
                bits,
                n_queries,
                top_k,
                corpus_size,
            )
            results.append(result)
            total_t = result.fit_time_s + result.compress_time_s + result.eval_time_s
            print(
                f"{label:>20s}  {result.compression_ratio:>5.1f}x  "
                f"{result.mean_cosine:>7.4f}  "
                f"{result.recall_at_k:>6.1%}  "
                f"{result.variance_explained:>4.1%}  "
                f"{total_t:>4.1f}s"
            )
        except Exception as e:
            print(f"{label:>20s}  ERROR: {e}")

    # Find Pareto-optimal and recommend
    pareto = find_pareto_optimal(results)
    rec = recommend(results, min_recall)

    elapsed = time.monotonic() - t_start

    report = AutotuneReport(
        source=dsn,
        table=table,
        column=column,
        input_dim=input_dim,
        sample_size=len(embeddings),
        corpus_size=corpus_size,
        query_count=n_queries,
        top_k=top_k,
        min_recall=min_recall,
        results=results,
        recommended=rec,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        total_time_s=round(elapsed, 1),
    )

    # Print summary
    print(f"\n{'=' * 62}")
    print("AUTOTUNE RESULTS")
    print(f"{'=' * 62}")
    print(f"Corpus: {corpus_size:,} embeddings ({input_dim}-dim)")
    print(f"Sample: {len(embeddings):,} embeddings")
    print(f"Time:   {elapsed:.1f}s")
    print()

    print("Pareto-optimal configurations:")
    for r in pareto:
        star = " << RECOMMENDED" if r.recommended else ""
        print(
            f"  PCA-{r.pca_dim} + TQ{r.bits}: "
            f"{r.compression_ratio:.1f}x compression, "
            f"{r.mean_cosine:.4f} cosine, "
            f"{r.recall_at_k:.1%} recall@{r.top_k}"
            f"{star}"
        )

    if rec:
        print(f"\nRecommendation (min recall >= {min_recall:.0%}):")
        print(
            f"  PCA-{rec.pca_dim} + TQ{rec.bits}: "
            f"{rec.compression_ratio:.1f}x compression"
        )
        print(f"  Cosine similarity: {rec.mean_cosine:.4f}")
        print(f"  Recall@{rec.top_k}: {rec.recall_at_k:.1%}")
        print(
            f"  Storage: {rec.original_mb:.0f} MB -> "
            f"{rec.compressed_mb:.0f} MB "
            f"(saves {rec.saved_mb:.0f} MB)"
        )
        print("\n  Usage:")
        print(
            f"    pca = PCAMatryoshka("
            f"input_dim={input_dim}, "
            f"output_dim={rec.pca_dim})"
        )
        print(f"    tq = TurboQuantPGVector(" f"dim={rec.pca_dim}, bits={rec.bits})")
        print("    pipeline = PCAMatryoshkaPipeline(pca, tq)")
    else:
        print(f"\nNo configuration meets min recall " f">= {min_recall:.0%}.")
        print("Try lowering --min-recall or using more PCA dims.")

    # Save JSON if requested
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "source": report.source,
            "table": report.table,
            "column": report.column,
            "input_dim": report.input_dim,
            "sample_size": report.sample_size,
            "corpus_size": report.corpus_size,
            "min_recall": report.min_recall,
            "timestamp": report.timestamp,
            "total_time_s": report.total_time_s,
            "results": [asdict(r) for r in report.results],
            "recommended": asdict(rec) if rec else None,
        }
        out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"\nResults saved to {output}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for turboquant-pro autotune."""
    parser = argparse.ArgumentParser(
        prog="turboquant-pro autotune",
        description=(
            "Find the optimal compression configuration " "for your embeddings."
        ),
    )
    parser.add_argument(
        "--source",
        required=True,
        help='PostgreSQL DSN (e.g., "dbname=atlas user=claude")',
    )
    parser.add_argument(
        "--table",
        default="chunks",
        help="Table containing embeddings (default: chunks)",
    )
    parser.add_argument(
        "--column",
        default="embedding",
        help="Embedding column name (default: embedding)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of embeddings to sample (default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.80,
        help="Minimum recall@k threshold (default: 0.80)",
    )
    parser.add_argument(
        "--pca-dims",
        type=str,
        default=None,
        help="Comma-separated PCA dims (default: 128,256,384,512)",
    )
    parser.add_argument(
        "--bits",
        type=str,
        default=None,
        help="Comma-separated bit widths (default: 2,3,4)",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=DEFAULT_QUERY_COUNT,
        help=f"Number of queries for recall@k (default: {DEFAULT_QUERY_COUNT})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"K for recall@k (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    pca_dims = None
    if args.pca_dims:
        pca_dims = [int(d) for d in args.pca_dims.split(",")]

    bit_widths = None
    if args.bits:
        bit_widths = [int(b) for b in args.bits.split(",")]

    run_autotune(
        dsn=args.source,
        table=args.table,
        column=args.column,
        sample_size=args.sample_size,
        pca_dims=pca_dims,
        bit_widths=bit_widths,
        min_recall=args.min_recall,
        n_queries=args.queries,
        top_k=args.top_k,
        output=args.output,
    )


if __name__ == "__main__":
    main()
