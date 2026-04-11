#!/usr/bin/env python3
# TurboQuant Pro: Multi-modal compression benchmark
# Copyright (c) 2025 Andrew H. Bond
# MIT License

"""
Benchmark: TurboQuant compression across embedding modalities.

Generates synthetic embeddings that mimic the spectral properties of
real models (text, vision, audio, code) and measures compression
quality and throughput using per-modality recommended presets.

Modality spectral profiles:
  - Text:   steep log-space eigenvalue decay (most energy in first dims)
  - Vision: nearly uniform variance (CLIP-like flat spectrum)
  - Audio:  moderate decay (between text and vision)
  - Code:   text-like decay (transformer-based encoders)

Usage:
    python benchmarks/benchmark_multimodal.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, ".")

from turboquant_pro.modality import (  # noqa: E402
    ModalityPreset,
    get_modality_preset,
    list_modality_presets,
)
from turboquant_pro.pca import PCAMatryoshka  # noqa: E402
from turboquant_pro.pgvector import TurboQuantPGVector  # noqa: E402

# ------------------------------------------------------------------ #
# Synthetic embedding generation                                       #
# ------------------------------------------------------------------ #

# Eigenvalue decay rates by modality category.
# Higher alpha = steeper decay = more energy in leading dims.
_SPECTRAL_PROFILES: dict[str, float] = {
    "text": 60.0,  # steep log-space decay
    "vision": 500.0,  # nearly uniform (slow decay)
    "audio": 120.0,  # moderate decay
    "code": 70.0,  # text-like decay
}


def generate_synthetic_embeddings(
    n: int,
    dim: int,
    modality: str,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic embeddings with modality-appropriate covariance.

    Creates embeddings with an eigenvalue spectrum that mimics the given
    modality: text models have steep spectral decay (most variance in
    leading dimensions), while vision models (e.g. CLIP) have a flatter
    spectrum.

    Args:
        n: Number of embeddings to generate.
        dim: Embedding dimension.
        modality: One of "text", "vision", "audio", "code".
        seed: Random seed for reproducibility.

    Returns:
        Array of shape (n, dim) with float32 embeddings.
    """
    rng = np.random.default_rng(seed)
    alpha = _SPECTRAL_PROFILES.get(modality, 100.0)

    # Eigenvalues: exponential decay controlled by alpha
    # alpha=500 → nearly flat; alpha=60 → steep
    eigenvalues = np.exp(-np.arange(dim, dtype=np.float64) / alpha)
    scale = np.sqrt(eigenvalues).astype(np.float32)

    # Generate i.i.d. Gaussian, then scale dims by sqrt(eigenvalue)
    X = rng.standard_normal((n, dim)).astype(np.float32) * scale
    return X


# ------------------------------------------------------------------ #
# Cosine similarity helper                                             #
# ------------------------------------------------------------------ #


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Mean cosine similarity between corresponding vectors."""
    dot = np.sum(a * b, axis=-1)
    norm_a = np.linalg.norm(a, axis=-1)
    norm_b = np.linalg.norm(b, axis=-1)
    return float(np.mean(dot / np.maximum(norm_a * norm_b, 1e-30)))


# ------------------------------------------------------------------ #
# Per-preset benchmark                                                 #
# ------------------------------------------------------------------ #


def benchmark_preset(
    preset: ModalityPreset,
    n_samples: int = 5000,
    seed: int = 42,
) -> dict:
    """Benchmark a single modality preset.

    Args:
        preset: The modality preset to evaluate.
        n_samples: Number of synthetic embeddings.
        seed: Random seed.

    Returns:
        Dictionary with cosine_sim, compression_ratio, compress_throughput,
        decompress_throughput metrics.
    """
    # 1. Generate synthetic embeddings
    X = generate_synthetic_embeddings(
        n=n_samples,
        dim=preset.dim,
        modality=preset.modality,
        seed=seed,
    )

    # 2. Optionally apply PCA reduction
    if preset.recommended_pca_dim is not None:
        pca = PCAMatryoshka(
            input_dim=preset.dim,
            output_dim=preset.recommended_pca_dim,
        )
        pca.fit(X)
        X_reduced = pca.transform(X)
        quant_dim = preset.recommended_pca_dim
    else:
        X_reduced = X
        quant_dim = preset.dim

    # 3. Compress with TurboQuant
    tq = TurboQuantPGVector(dim=quant_dim, bits=preset.recommended_bits, seed=42)

    # Warmup
    _ = tq.compress_batch(X_reduced[:10])

    # Timed compress
    t0 = time.perf_counter()
    compressed = tq.compress_batch(X_reduced)
    compress_time = time.perf_counter() - t0

    # 4. Decompress (reconstruct)
    t0 = time.perf_counter()
    X_recon = tq.decompress_batch(compressed)
    decompress_time = time.perf_counter() - t0

    # 5. Measure quality
    cos_sim = _cosine_similarity(X_reduced, X_recon)

    # 6. Compression ratio: original float32 size / compressed size
    original_bytes = n_samples * preset.dim * 4  # float32
    compressed_bytes = sum(c.size_bytes for c in compressed)
    ratio = original_bytes / max(compressed_bytes, 1)

    return {
        "cosine_sim": cos_sim,
        "compression_ratio": ratio,
        "compress_throughput": n_samples / max(compress_time, 1e-9),
        "decompress_throughput": n_samples / max(decompress_time, 1e-9),
        "compress_time_ms": compress_time * 1000,
        "decompress_time_ms": decompress_time * 1000,
    }


# ------------------------------------------------------------------ #
# Main benchmark                                                       #
# ------------------------------------------------------------------ #


def run_benchmark(n_samples: int = 5000) -> None:
    """Run multi-modal compression benchmark across all presets."""
    print("=" * 90)
    print("TurboQuant Pro — Multi-Modal Compression Benchmark")
    print("=" * 90)
    print(f"Samples per modality: {n_samples}")
    print()

    header = (
        f"{'Model':<20} {'Modality':<8} {'Dim':>5} "
        f"{'PCA':>5} {'Bits':>4} {'Cos Sim':>8} "
        f"{'Ratio':>7} {'Comp/s':>10} {'Decomp/s':>10}"
    )
    print(header)
    print("-" * len(header))

    results: list[dict] = []
    preset_names = list_modality_presets()

    for name in preset_names:
        preset = get_modality_preset(name)
        metrics = benchmark_preset(preset, n_samples=n_samples)

        pca_str = str(preset.recommended_pca_dim) if preset.recommended_pca_dim else "—"

        print(
            f"{preset.name:<20} {preset.modality:<8} {preset.dim:>5} "
            f"{pca_str:>5} {preset.recommended_bits:>4} "
            f"{metrics['cosine_sim']:>8.4f} "
            f"{metrics['compression_ratio']:>7.1f}x "
            f"{metrics['compress_throughput']:>10,.0f} "
            f"{metrics['decompress_throughput']:>10,.0f}"
        )

        results.append({"preset": name, **metrics})

    # Summary by modality
    print()
    print("=" * 90)
    print("Summary by Modality")
    print("=" * 90)

    modalities_seen: dict[str, list[dict]] = {}
    for name, metrics in zip(preset_names, results):
        preset = get_modality_preset(name)
        modalities_seen.setdefault(preset.modality, []).append(metrics)

    summary_header = (
        f"{'Modality':<10} {'Models':>6} {'Avg Cos':>8} "
        f"{'Min Cos':>8} {'Avg Ratio':>10}"
    )
    print(summary_header)
    print("-" * len(summary_header))

    for modality in ("text", "vision", "audio", "code"):
        entries = modalities_seen.get(modality, [])
        if not entries:
            continue
        avg_cos = np.mean([e["cosine_sim"] for e in entries])
        min_cos = min(e["cosine_sim"] for e in entries)
        avg_ratio = np.mean([e["compression_ratio"] for e in entries])
        print(
            f"{modality:<10} {len(entries):>6} {avg_cos:>8.4f} "
            f"{min_cos:>8.4f} {avg_ratio:>9.1f}x"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    run_benchmark()
