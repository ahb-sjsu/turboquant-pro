"""
Comprehensive benchmark for TurboQuant-KV.

Measures compression ratio, cosine similarity, MSE, and throughput
across bit widths, head dimensions, and sequence lengths.

Usage:
    python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

from turboquant_kv import TurboQuantKV, TurboQuantKVCache


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    dot = np.sum(a_flat * b_flat, axis=-1)
    norm_a = np.linalg.norm(a_flat, axis=-1)
    norm_b = np.linalg.norm(b_flat, axis=-1)
    return float(np.mean(dot / np.maximum(norm_a * norm_b, 1e-30)))


def benchmark_compression():
    """Benchmark compression across configurations."""
    print("=" * 78)
    print("TurboQuant-KV Compression Benchmark")
    print("=" * 78)
    print()

    configs = [
        # (head_dim, n_heads, seq_len, batch)
        (64, 8, 512, 1),
        (128, 16, 1024, 1),
        (256, 16, 2048, 1),
        (256, 16, 4096, 1),
    ]

    header = (
        f"{'head_dim':>8} {'n_heads':>7} {'seq_len':>7} "
        f"{'bits':>4} {'ratio':>7} {'cos_sim':>8} "
        f"{'MSE':>10} {'comp_ms':>8} {'decomp_ms':>9}"
    )
    print(header)
    print("-" * len(header))

    rng = np.random.default_rng(42)

    for head_dim, n_heads, seq_len, batch in configs:
        tensor = rng.standard_normal((batch, n_heads, seq_len, head_dim)).astype(
            np.float32
        )

        for bits in [2, 3, 4]:
            tq = TurboQuantKV(
                head_dim=head_dim,
                n_heads=n_heads,
                bits=bits,
                use_gpu=False,
                seed=0,
            )

            # Warmup
            c = tq.compress(tensor, packed=True)
            tq.decompress(c)

            # Timed compress
            t0 = time.perf_counter()
            n_iters = 5
            for _ in range(n_iters):
                c = tq.compress(tensor, packed=True)
            comp_ms = (time.perf_counter() - t0) / n_iters * 1000

            # Timed decompress
            t0 = time.perf_counter()
            for _ in range(n_iters):
                recon = tq.decompress(c)
            decomp_ms = (time.perf_counter() - t0) / n_iters * 1000

            ratio = c.compression_ratio(head_dim)
            cos_sim = _cosine_similarity(tensor, recon)
            mse = float(np.mean((tensor - recon) ** 2))

            print(
                f"{head_dim:>8} {n_heads:>7} {seq_len:>7} "
                f"{bits:>4} {ratio:>7.2f} {cos_sim:>8.4f} "
                f"{mse:>10.6f} {comp_ms:>8.1f} {decomp_ms:>9.1f}"
            )


def benchmark_streaming_cache():
    """Benchmark the streaming tiered cache."""
    print()
    print("=" * 78)
    print("TurboQuant-KV Streaming Cache Benchmark")
    print("=" * 78)
    print()

    configs = [
        # (head_dim, n_heads, n_tokens, hot_window)
        (128, 8, 2048, 256),
        (128, 8, 4096, 512),
        (256, 16, 2048, 256),
        (256, 16, 4096, 512),
    ]

    header = (
        f"{'head_dim':>8} {'n_heads':>7} {'tokens':>7} "
        f"{'hot_win':>7} {'mem_MB':>7} {'uncomp_MB':>9} "
        f"{'ratio':>6} {'tok/s':>8}"
    )
    print(header)
    print("-" * len(header))

    rng = np.random.default_rng(42)

    for head_dim, n_heads, n_tokens, hot_window in configs:
        cache = TurboQuantKVCache(
            head_dim=head_dim,
            n_heads=n_heads,
            bits=3,
            hot_window=hot_window,
            use_gpu=False,
            seed=0,
        )

        t0 = time.perf_counter()
        for _ in range(n_tokens):
            k = rng.standard_normal((1, n_heads, 1, head_dim)).astype(np.float32)
            v = rng.standard_normal((1, n_heads, 1, head_dim)).astype(np.float32)
            cache.append(k, v)
        elapsed = time.perf_counter() - t0

        stats = cache.memory_stats()
        tok_per_sec = n_tokens / elapsed

        print(
            f"{head_dim:>8} {n_heads:>7} {n_tokens:>7} "
            f"{hot_window:>7} {stats['total_bytes']/1e6:>7.1f} "
            f"{stats['uncompressed_equivalent_bytes']/1e6:>9.1f} "
            f"{stats['effective_ratio']:>6.1f} {tok_per_sec:>8.0f}"
        )


def benchmark_memory_estimates():
    """Print memory estimates for popular models."""
    print()
    print("=" * 78)
    print("Memory Estimates for Popular Models (3-bit, bit-packed)")
    print("=" * 78)
    print()

    models = [
        ("Llama 3.1 8B", 32, 8, 128, 8192),
        ("Llama 3.1 70B", 80, 8, 128, 8192),
        ("Gemma 4 27B", 36, 16, 256, 8192),
        ("Mistral 7B", 32, 8, 128, 32768),
        ("Qwen2 72B", 80, 8, 128, 32768),
    ]

    header = (
        f"{'Model':>20} {'Original':>10} {'Compressed':>12} "
        f"{'Saved':>8} {'Ratio':>6}"
    )
    print(header)
    print("-" * len(header))

    for name, n_layers, n_kv_heads, head_dim, seq_len in models:
        est = TurboQuantKV.estimate_memory(
            n_layers=n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            bits=3,
            bit_packed=True,
        )
        print(
            f"{name:>20} {est['original_gb']:>9.3f}G "
            f"{est['compressed_gb']:>11.3f}G "
            f"{est['saved_gb']:>7.3f}G {est['ratio']:>5.1f}x"
        )


if __name__ == "__main__":
    benchmark_compression()
    benchmark_streaming_cache()
    benchmark_memory_estimates()
