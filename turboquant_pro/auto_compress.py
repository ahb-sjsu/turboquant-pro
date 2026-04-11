# TurboQuant Pro: Open-source PolarQuant+QJL for LLM KV cache compression
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
General-purpose auto-compression with Pareto sweep.

Sweeps PCA dimensions, bit widths, and packing strategies to find the
Pareto-optimal compression configuration that meets a quality target.

Usage::

    from turboquant_pro.auto_compress import auto_compress

    result = auto_compress(
        embeddings,               # (n, dim) float32
        target="cosine > 0.97",   # quality constraint
    )

    print(result.config)          # Best config found
    print(result.compressed)      # Compressed embeddings
    print(result.ratio)           # Achieved compression ratio
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Result dataclass                                                     #
# ------------------------------------------------------------------ #


@dataclass
class AutoCompressResult:
    """Result of an auto_compress sweep.

    Attributes:
        config: Dictionary describing the chosen configuration.
        compressed: List of compressed embedding objects.
        ratio: Achieved compression ratio (original / compressed).
        mean_cosine: Mean cosine similarity of round-trip reconstruction.
        min_cosine: Minimum cosine similarity.
        candidates: All Pareto-optimal configurations found.
    """

    config: dict
    compressed: list
    ratio: float
    mean_cosine: float
    min_cosine: float
    candidates: list[dict]


# ------------------------------------------------------------------ #
# Target parsing                                                       #
# ------------------------------------------------------------------ #

_TARGET_PATTERN = re.compile(r"(cosine|recall@\d+|ratio)\s*(>|>=|<|<=)\s*([\d.]+)")


def _parse_target(target: str) -> tuple[str, str, float]:
    """Parse a target constraint string.

    Returns (metric, operator, threshold).

    Supported formats:
        - ``"cosine > 0.97"``
        - ``"cosine >= 0.95"``
        - ``"ratio > 20"``
        - ``"recall@10 > 0.90"``
    """
    m = _TARGET_PATTERN.match(target.strip())
    if not m:
        raise ValueError(
            f"Cannot parse target '{target}'. "
            f"Expected format like 'cosine > 0.97' or 'ratio > 20'."
        )
    return m.group(1), m.group(2), float(m.group(3))


def _meets_target(metric: str, op: str, threshold: float, result: dict) -> bool:
    """Check whether a candidate result meets the target constraint."""
    if metric == "ratio":
        value = result.get("ratio", 0.0)
    elif metric.startswith("recall@"):
        value = result.get("mean_cosine", 0.0)  # approximate
    else:
        value = result.get("mean_cosine", 0.0)

    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    return False


# ------------------------------------------------------------------ #
# Sweep engine                                                         #
# ------------------------------------------------------------------ #


def _evaluate_config(
    embeddings: np.ndarray,
    sample: np.ndarray,
    pca_dim: int | None,
    bits: int,
    weighted: bool,
    avg_bits: float,
    seed: int,
) -> dict:
    """Evaluate one compression configuration."""
    from .pca import PCAMatryoshka
    from .pgvector import TurboQuantPGVector

    dim = embeddings.shape[1]
    label_parts = []

    t0 = time.perf_counter()

    if pca_dim is not None and pca_dim < dim:
        pca = PCAMatryoshka(input_dim=dim, output_dim=pca_dim)
        pca.fit(embeddings[:1000])

        if weighted:
            pipeline = pca.with_weighted_quantizer(avg_bits=avg_bits, seed=seed)
            label_parts.append(f"PCA-{pca_dim}")
            label_parts.append("+".join(f"{n}d@{b}b" for n, b in pipeline.bit_schedule))

            sims = []
            compressed_list = []
            for emb in sample:
                c = pipeline.compress(emb)
                r = pipeline.decompress(c)
                sim = float(
                    np.dot(emb, r) / (np.linalg.norm(emb) * np.linalg.norm(r) + 1e-30)
                )
                sims.append(sim)
                compressed_list.append(c)

            sims_arr = np.array(sims)
            ratio = pipeline.compression_ratio
        else:
            pipeline = pca.with_quantizer(bits=bits, seed=seed)
            label_parts.append(f"PCA-{pca_dim}")
            label_parts.append(f"TQ{bits}")

            sims = []
            compressed_list = []
            for emb in sample:
                c = pipeline.compress(emb)
                r = pipeline.decompress(c)
                sim = float(
                    np.dot(emb, r) / (np.linalg.norm(emb) * np.linalg.norm(r) + 1e-30)
                )
                sims.append(sim)
                compressed_list.append(c)

            sims_arr = np.array(sims)
            ratio = pipeline.compression_ratio
    else:
        # No PCA, just TurboQuant
        tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)
        label_parts.append(f"TQ{bits}")

        sims = []
        compressed_list = []
        for emb in sample:
            c = tq.compress_embedding(emb)
            r = tq.decompress_embedding(c)
            sim = float(
                np.dot(emb, r) / (np.linalg.norm(emb) * np.linalg.norm(r) + 1e-30)
            )
            sims.append(sim)
            compressed_list.append(c)

        sims_arr = np.array(sims)
        bytes_per_emb = compressed_list[0].size_bytes
        ratio = (dim * 4) / bytes_per_emb

    elapsed = time.perf_counter() - t0

    return {
        "label": " + ".join(label_parts),
        "pca_dim": pca_dim,
        "bits": bits,
        "weighted": weighted,
        "avg_bits": round(avg_bits if weighted else bits, 2),
        "mean_cosine": round(float(sims_arr.mean()), 6),
        "min_cosine": round(float(sims_arr.min()), 6),
        "std_cosine": round(float(sims_arr.std()), 6),
        "ratio": round(ratio, 1),
        "time_s": round(elapsed, 3),
        "compressed": compressed_list,
    }


def _pareto_filter(candidates: list[dict]) -> list[dict]:
    """Keep only Pareto-optimal candidates (quality vs compression)."""
    # Sort by ratio descending (most compression first)
    candidates.sort(key=lambda c: c["ratio"], reverse=True)

    pareto = []
    best_quality = -1.0
    for c in candidates:
        if c["mean_cosine"] > best_quality:
            best_quality = c["mean_cosine"]
            pareto.append(c)

    return pareto


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #


def auto_compress(
    embeddings: np.ndarray,
    target: str = "cosine > 0.95",
    sample_size: int = 100,
    pca_dims: list[int] | None = None,
    bit_widths: list[int] | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> AutoCompressResult:
    """Automatically find the best compression for embeddings.

    Sweeps PCA dimensions, bit widths (2/3/4), and uniform vs
    eigenvalue-weighted quantization.  Returns the configuration with
    the highest compression ratio that meets the quality target.

    Args:
        embeddings: 2D float32 array of shape ``(n, dim)``.
        target: Quality constraint string.  Supported formats:

            - ``"cosine > 0.97"`` — mean cosine similarity
            - ``"cosine >= 0.95"``
            - ``"ratio > 20"`` — minimum compression ratio

        sample_size: Number of embeddings to evaluate quality on
            (random subsample for speed).
        pca_dims: PCA output dimensions to sweep.  ``None`` auto-selects
            based on input dim (e.g., ``[dim, dim//2, dim//4]``).
        bit_widths: Bit widths to sweep.  Default ``[2, 3, 4]``.
        seed: Random seed for reproducibility.
        verbose: Print progress and results.

    Returns:
        AutoCompressResult with the best config, compressed data,
        and all Pareto-optimal candidates.

    Example::

        result = auto_compress(embeddings, target="cosine > 0.97")
        print(f"Best: {result.config['label']} at {result.ratio}x")
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    n, dim = embeddings.shape
    metric, op, threshold = _parse_target(target)

    if bit_widths is None:
        bit_widths = [2, 3, 4]

    if pca_dims is None:
        # Auto-select PCA dims based on input dimension
        candidates_dims = [dim]  # No PCA
        for frac in [0.75, 0.5, 0.375, 0.25, 0.125]:
            d = max(16, int(dim * frac))
            if d < dim:
                candidates_dims.append(d)
        pca_dims = sorted(set(candidates_dims), reverse=True)

    # Subsample for evaluation speed
    rng = np.random.default_rng(seed)
    if n > sample_size:
        sample_idx = rng.choice(n, sample_size, replace=False)
        sample = embeddings[sample_idx]
    else:
        sample = embeddings

    if verbose:
        print(f"Auto-compress: {n:,} x {dim}d, target: {target}")
        print(f"Sweeping: PCA dims {pca_dims}, bits {bit_widths}")
        print()

    candidates = []

    for pca_dim in pca_dims:
        effective_dim = pca_dim if pca_dim < dim else None

        for bits in bit_widths:
            # Uniform quantization
            result = _evaluate_config(
                embeddings,
                sample,
                effective_dim,
                bits,
                weighted=False,
                avg_bits=float(bits),
                seed=seed,
            )
            candidates.append(result)

            if verbose:
                print(
                    f"  {result['label']:>30s}  "
                    f"cosine={result['mean_cosine']:.4f}  "
                    f"ratio={result['ratio']:>5.1f}x  "
                    f"({result['time_s']:.2f}s)"
                )

        # Eigenweighted (only if PCA is applied)
        if effective_dim is not None:
            for avg in [2.5, 3.0, 3.5]:
                result = _evaluate_config(
                    embeddings,
                    sample,
                    effective_dim,
                    3,
                    weighted=True,
                    avg_bits=avg,
                    seed=seed,
                )
                candidates.append(result)

                if verbose:
                    print(
                        f"  {result['label']:>30s}  "
                        f"cosine={result['mean_cosine']:.4f}  "
                        f"ratio={result['ratio']:>5.1f}x  "
                        f"({result['time_s']:.2f}s)"
                    )

    # Filter to Pareto-optimal
    pareto = _pareto_filter(candidates)

    # Find best that meets the target
    meeting = [c for c in pareto if _meets_target(metric, op, threshold, c)]

    if meeting:
        # Among those meeting the target, pick the one with highest compression
        best = max(meeting, key=lambda c: c["ratio"])
    else:
        # Nothing meets the target — pick the highest quality
        best = max(candidates, key=lambda c: c["mean_cosine"])
        if verbose:
            print(
                f"\nWARNING: No config meets target '{target}'. "
                f"Returning highest quality instead."
            )

    if verbose:
        print(
            f"\nBest: {best['label']} — "
            f"cosine={best['mean_cosine']:.4f}, "
            f"ratio={best['ratio']}x"
        )

    # Strip compressed lists from candidates for memory
    pareto_clean = [{k: v for k, v in c.items() if k != "compressed"} for c in pareto]

    return AutoCompressResult(
        config={k: v for k, v in best.items() if k != "compressed"},
        compressed=best["compressed"],
        ratio=best["ratio"],
        mean_cosine=best["mean_cosine"],
        min_cosine=best["min_cosine"],
        candidates=pareto_clean,
    )
