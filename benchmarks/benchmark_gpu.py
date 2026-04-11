"""
CPU vs GPU compression throughput benchmark for TurboQuant Pro.

Measures embedding compression speed at various batch sizes and
validates the 1M embeddings/sec GPU target on Volta+ hardware.

Optionally uses batch-probe (pip install batch-probe) to auto-discover
the maximum batch size that fits in GPU memory.

Usage:
    python benchmarks/benchmark_gpu.py
    python benchmarks/benchmark_gpu.py --dim 384 --bits 3
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np

from turboquant_pro.pgvector import TurboQuantPGVector

try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

try:
    from batch_probe import probe as batch_probe  # type: ignore[import-untyped]

    _HAS_BATCH_PROBE = True
except ImportError:
    _HAS_BATCH_PROBE = False


def _time_compress(
    tq: TurboQuantPGVector,
    embeddings: np.ndarray,
    use_gpu: bool,
    warmup: int = 1,
    repeats: int = 3,
) -> float:
    """Time compress_batch and return best throughput (emb/sec)."""
    n = len(embeddings)

    # Warmup
    for _ in range(warmup):
        tq.compress_batch(embeddings, use_gpu=use_gpu)

    if use_gpu:
        cp.cuda.Stream.null.synchronize()

    best = float("inf")
    for _ in range(repeats):
        gc.disable()
        t0 = time.perf_counter()
        tq.compress_batch(embeddings, use_gpu=use_gpu)
        if use_gpu:
            cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        gc.enable()
        best = min(best, elapsed)

    return n / best


def benchmark_gpu_compression(
    scales: list[int],
    dim: int = 1024,
    bits: int = 3,
    seed: int = 42,
) -> list[dict]:
    """Benchmark CPU vs GPU compression at various batch sizes.

    Returns list of result dicts for each scale.
    """
    tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)
    rng = np.random.default_rng(seed)

    results = []
    print(f"{'N':>10s}  {'CPU emb/s':>12s}  {'GPU emb/s':>12s}  {'Speedup':>8s}")
    print("-" * 50)

    for n in scales:
        embs = rng.standard_normal((n, dim)).astype(np.float32)

        cpu_rate = _time_compress(tq, embs, use_gpu=False)

        if _HAS_CUPY:
            gpu_rate = _time_compress(tq, embs, use_gpu=True)
            speedup = gpu_rate / cpu_rate
        else:
            gpu_rate = float("nan")
            speedup = float("nan")

        row = {
            "n": n,
            "dim": dim,
            "bits": bits,
            "cpu_emb_per_sec": cpu_rate,
            "gpu_emb_per_sec": gpu_rate,
            "speedup": speedup,
        }
        results.append(row)

        gpu_str = f"{gpu_rate:12,.0f}" if _HAS_CUPY else "      N/A"
        spd_str = f"{speedup:7.1f}x" if _HAS_CUPY else "     N/A"
        print(f"{n:10,d}  {cpu_rate:12,.0f}  {gpu_str}  {spd_str}")

    return results


def probe_max_batch(dim: int, bits: int, seed: int) -> int | None:
    """Use batch-probe to find max GPU batch size (optional)."""
    if not _HAS_BATCH_PROBE or not _HAS_CUPY:
        return None

    tq = TurboQuantPGVector(dim=dim, bits=bits, seed=seed)

    def work(batch_size: int) -> None:
        embs = np.random.default_rng(0).standard_normal(
            (batch_size, dim)
        ).astype(np.float32)
        tq.compress_batch(embs, use_gpu=True)
        cp.cuda.Stream.null.synchronize()

    return batch_probe(work, high=2_000_000, headroom=0.1, backend="cupy")


def main() -> None:
    parser = argparse.ArgumentParser(description="TurboQuant GPU benchmark")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("TurboQuant Pro: CPU vs GPU Compression Benchmark")
    print("=" * 60)
    print(f"  dim={args.dim}  bits={args.bits}  seed={args.seed}")
    print(f"  CuPy available: {_HAS_CUPY}")
    print(f"  batch-probe available: {_HAS_BATCH_PROBE}")

    if _HAS_CUPY:
        dev = cp.cuda.Device(0)
        print(f"  GPU: device {dev.id}, compute capability {dev.compute_capability}")
        mem = dev.mem_info
        print(f"  GPU memory: {mem[1] / 1e9:.1f} GB total, {mem[0] / 1e9:.1f} GB free")

    # Auto-probe max batch size if batch-probe is available
    scales = [1_000, 10_000, 100_000, 500_000]
    if _HAS_BATCH_PROBE and _HAS_CUPY:
        print("\nProbing maximum GPU batch size...")
        max_batch = probe_max_batch(args.dim, args.bits, args.seed)
        if max_batch:
            print(f"  Max safe batch size: {max_batch:,}")
            if max_batch >= 1_000_000:
                scales.append(1_000_000)
            scales = [s for s in scales if s <= max_batch]
    elif _HAS_CUPY:
        scales.append(1_000_000)

    print()
    results = benchmark_gpu_compression(
        scales, dim=args.dim, bits=args.bits, seed=args.seed
    )

    # Report target validation
    print()
    if _HAS_CUPY and results:
        best_gpu = max(r["gpu_emb_per_sec"] for r in results)
        target = 1_000_000
        if best_gpu >= target:
            print(f"TARGET MET: {best_gpu:,.0f} emb/sec >= {target:,} emb/sec")
        else:
            print(f"TARGET NOT MET: {best_gpu:,.0f} emb/sec < {target:,} emb/sec")
    else:
        print("CuPy not available — GPU benchmark skipped.")


if __name__ == "__main__":
    main()
