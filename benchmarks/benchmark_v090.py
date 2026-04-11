"""
v0.9.0 feature benchmarks: measure actual improvement from each new feature.

Usage:
    python benchmarks/benchmark_v090.py
"""

from __future__ import annotations

import time

import numpy as np

from turboquant_pro import TurboQuantKV
from turboquant_pro.hnsw import CompressedHNSW
from turboquant_pro.pca import PCAMatryoshka
from turboquant_pro.pgvector import TurboQuantPGVector
from turboquant_pro.rope import RoPEAwareQuantizer


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    dot = np.sum(a_flat * b_flat, axis=-1)
    norm_a = np.linalg.norm(a_flat, axis=-1)
    norm_b = np.linalg.norm(b_flat, axis=-1)
    return float(np.mean(dot / np.maximum(norm_a * norm_b, 1e-30)))


# ================================================================== #
# Benchmark 1: Asymmetric K/V bit allocation                          #
# ================================================================== #


def benchmark_asymmetric_kv():
    print("=" * 70)
    print("BENCHMARK 1: Asymmetric K/V Bit Allocation")
    print("=" * 70)

    rng = np.random.default_rng(42)
    configs = [
        (64, 8, 128),
        (128, 16, 256),
        (256, 16, 128),
    ]

    print(
        f"\n{'head_dim':>8s}  {'Config':>12s}  {'Key CosSim':>11s}  "
        f"{'Val CosSim':>11s}  {'Avg bits':>9s}  {'Packed KB':>10s}"
    )
    print("-" * 70)

    for head_dim, n_heads, seq_len in configs:
        tensor = rng.standard_normal((1, n_heads, seq_len, head_dim)).astype(np.float32)

        for label, kb, vb in [
            ("K3/V3", 3, 3),
            ("K4/V3", 4, 3),
            ("K4/V2", 4, 2),
            ("K3/V2", 3, 2),
        ]:
            tq = TurboQuantKV(
                head_dim=head_dim,
                n_heads=n_heads,
                key_bits=kb,
                value_bits=vb,
                use_gpu=False,
                seed=0,
            )
            ck = tq.compress(tensor, packed=True, kind="key")
            cv = tq.compress(tensor, packed=True, kind="value")
            rk = tq.decompress(ck)
            rv = tq.decompress(cv)

            key_sim = _cosine_similarity(tensor, rk)
            val_sim = _cosine_similarity(tensor, rv)
            total_kb = (ck.nbytes() + cv.nbytes()) / 1024

            avg_bits = (kb + vb) / 2.0
            print(
                f"{head_dim:>8d}  {label:>12s}  {key_sim:>11.6f}  "
                f"{val_sim:>11.6f}  {avg_bits:>9.1f}  {total_kb:>10.1f}"
            )
        print()

    # Memory estimation comparison
    print("Memory estimation (32 layers, 8 heads, head_dim=128, 8K context):")
    for label, kb, vb in [("K3/V3", 3, 3), ("K4/V3", 4, 3), ("K4/V2", 4, 2)]:
        est = TurboQuantKV.estimate_memory(
            n_layers=32,
            n_kv_heads=8,
            head_dim=128,
            seq_len=8192,
            key_bits=kb,
            value_bits=vb,
            bit_packed=True,
        )
        print(
            f"  {label}: {est['compressed_gb']:.3f} GB "
            f"({est['ratio']:.1f}x vs fp16)"
        )


# ================================================================== #
# Benchmark 2: Eigenvalue-weighted mixed-precision                     #
# ================================================================== #


def benchmark_eigenweighted():
    print("\n" + "=" * 70)
    print("BENCHMARK 2: Eigenvalue-Weighted Mixed-Precision Quantization")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # Create realistic embedding data with decaying variance
    input_dim = 1024
    output_dim = 384
    n_train = 2000
    n_test = 200

    # Simulate BGE-M3-like structure: some dims much more important
    eigenvalues = np.logspace(2, -2, input_dim).astype(np.float32)
    cov = np.diag(eigenvalues)
    X_train = rng.multivariate_normal(np.zeros(input_dim), cov, size=n_train).astype(
        np.float32
    )
    X_test = rng.multivariate_normal(np.zeros(input_dim), cov, size=n_test).astype(
        np.float32
    )

    pca = PCAMatryoshka(input_dim=input_dim, output_dim=output_dim)
    pca.fit(X_train)

    print(f"\nPCA: {input_dim} -> {output_dim} dims")
    print(f"Training: {n_train} vectors, Testing: {n_test} vectors\n")

    print(
        f"{'Method':>30s}  {'Avg bits':>9s}  {'Mean CosSim':>12s}  "
        f"{'Min CosSim':>11s}  {'Std':>8s}  {'Ratio':>7s}"
    )
    print("-" * 82)

    # Uniform quantization at different bit widths
    for bits in [2, 3, 4]:
        pipeline = pca.with_quantizer(bits=bits)
        sims = []
        for emb in X_test:
            c = pipeline.compress(emb)
            r = pipeline.decompress(c)
            dot = np.dot(emb, r)
            sim = dot / (np.linalg.norm(emb) * np.linalg.norm(r) + 1e-30)
            sims.append(float(sim))
        sims = np.array(sims)
        ratio = pipeline.compression_ratio
        print(
            f"{'Uniform ' + str(bits) + '-bit':>30s}  {bits:>9.1f}  "
            f"{sims.mean():>12.6f}  {sims.min():>11.6f}  "
            f"{sims.std():>8.6f}  {ratio:>6.1f}x"
        )

    # Eigenweighted at different avg bits
    for avg in [2.0, 3.0, 4.0]:
        pipeline = pca.with_weighted_quantizer(avg_bits=avg)
        mean, mn, std = pipeline.batch_cosine_similarity(X_test)
        ratio = pipeline.compression_ratio
        schedule = pipeline.bit_schedule
        sched_str = "+".join(f"{n}d@{b}b" for n, b in schedule)
        print(
            f"{'Eigen ' + sched_str:>30s}  {pipeline.avg_bits:>9.1f}  "
            f"{mean:>12.6f}  {mn:>11.6f}  "
            f"{std:>8.6f}  {ratio:>6.1f}x"
        )


# ================================================================== #
# Benchmark 3: RoPE-aware KV cache quantization                       #
# ================================================================== #


def benchmark_rope_aware():
    print("\n" + "=" * 70)
    print("BENCHMARK 3: RoPE-Aware KV Cache Quantization")
    print("=" * 70)

    rng = np.random.default_rng(42)

    print(
        f"\n{'head_dim':>8s}  {'max_seq':>8s}  {'Config':>20s}  "
        f"{'CosSim':>10s}  {'Avg bits':>9s}  {'Boosted dims':>13s}"
    )
    print("-" * 75)

    for head_dim in [128, 256]:
        tensor = rng.standard_normal((1, 8, 64, head_dim)).astype(np.float32)

        for max_seq in [4096, 8192, 32768, 131072]:
            # Uniform baseline
            tq_uniform = TurboQuantKV(
                head_dim=head_dim,
                n_heads=8,
                bits=3,
                use_gpu=False,
                seed=0,
            )
            c_u = tq_uniform.compress(tensor, packed=True)
            r_u = tq_uniform.decompress(c_u)
            sim_uniform = _cosine_similarity(tensor, r_u)

            # RoPE-aware
            rq = RoPEAwareQuantizer(
                head_dim=head_dim,
                n_heads=8,
                default_bits=3,
                boost_bits=4,
                max_seq_len=max_seq,
                use_gpu=False,
                seed=0,
            )
            c_r = rq.compress(tensor, packed=True)
            r_r = rq.decompress(c_r)
            sim_rope = _cosine_similarity(tensor, r_r)

            stats = rq.stats()
            n_boost = stats["n_boost_dims"]

            print(
                f"{head_dim:>8d}  {max_seq:>8d}  "
                f"{'Uniform 3-bit':>20s}  {sim_uniform:>10.6f}  "
                f"{'3.0':>9s}  {'0':>13s}"
            )
            print(
                f"{'':>8s}  {'':>8s}  "
                f"{'RoPE-aware 4/3':>20s}  {sim_rope:>10.6f}  "
                f"{rq.avg_bits:>9.2f}  {n_boost:>13d}"
            )
            improvement = sim_rope - sim_uniform
            print(
                f"{'':>8s}  {'':>8s}  " f"{'Improvement':>20s}  {improvement:>+10.6f}"
            )
            print()


# ================================================================== #
# Benchmark 4: Lossless graph compression (ANS codec)                  #
# ================================================================== #


def benchmark_lossless_graph():
    print("=" * 70)
    print("BENCHMARK 4: Lossless Graph Compression (ANS Codec)")
    print("=" * 70)

    from turboquant_pro.ans_codec import ANSCodec

    codec = ANSCodec()

    # Test on realistic neighbor ID patterns
    rng = np.random.default_rng(42)
    print(
        f"\n{'Pattern':>20s}  {'N IDs':>7s}  {'Raw bytes':>10s}  "
        f"{'Compressed':>11s}  {'Ratio':>7s}"
    )
    print("-" * 60)

    for label, ids in [
        ("Sequential", list(range(1000))),
        ("Sparse uniform", sorted(rng.choice(100000, 1000, replace=False).tolist())),
        ("Dense cluster", sorted(rng.integers(5000, 6000, size=1000).tolist())),
        ("HNSW neighbors", sorted(rng.choice(50000, 32, replace=False).tolist())),
    ]:
        raw = len(ids) * 4  # 4 bytes per int32
        encoded = codec.encode_integers(ids)
        comp = len(encoded)
        ratio = raw / comp if comp > 0 else float("inf")
        print(
            f"{label:>20s}  {len(ids):>7d}  {raw:>10d}  " f"{comp:>11d}  {ratio:>6.1f}x"
        )

    # Measure on actual CompressedHNSW graph
    print("\nCompressedHNSW memory breakdown:")
    for n_vectors, dim in [(1000, 64), (5000, 128), (10000, 256)]:
        tq = TurboQuantPGVector(dim=dim, bits=3, seed=0)
        index = CompressedHNSW(tq, M=16, ef_construction=100, seed=42)

        corpus = rng.standard_normal((n_vectors, dim)).astype(np.float32)
        t0 = time.perf_counter()
        index.batch_insert(list(range(n_vectors)), corpus)
        build_time = time.perf_counter() - t0

        mem = index.memory_usage_bytes()
        float32_bytes = n_vectors * dim * 4

        if isinstance(mem, dict):
            graph_uncomp = mem.get("graph_uncompressed", 0)
            graph_comp = mem.get("graph_compressed", 0)
            total = mem.get("total", sum(mem.values()))
            graph_ratio = graph_uncomp / max(graph_comp, 1)
            vs_float32 = total / float32_bytes
            print(
                f"  {n_vectors:>6,}vec x {dim:>3d}d: "
                f"graph {graph_uncomp:>10,}B -> {graph_comp:>10,}B "
                f"({graph_ratio:.1f}x), "
                f"total {total:>12,}B "
                f"({vs_float32:.2f}x vs float32), "
                f"build {build_time:.1f}s"
            )
        else:
            vs_float32 = mem / float32_bytes
            print(
                f"  {n_vectors:>6,}vec x {dim:>3d}d: "
                f"total {mem:>12,}B ({vs_float32:.2f}x vs float32), "
                f"build {build_time:.1f}s"
            )


# ================================================================== #
# Main                                                                 #
# ================================================================== #

if __name__ == "__main__":
    benchmark_asymmetric_kv()
    benchmark_eigenweighted()
    benchmark_rope_aware()
    benchmark_lossless_graph()
