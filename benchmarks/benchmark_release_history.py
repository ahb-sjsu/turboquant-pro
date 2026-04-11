"""
Differential benchmark: feature progression from v0.5.0 to v1.0.0.

Shows the incremental improvement each release brought, measured on
synthetic embeddings that mimic BGE-M3 1024-dim properties.

Usage:
    python benchmarks/benchmark_release_history.py
"""

from __future__ import annotations

import numpy as np

from turboquant_pro import TurboQuantKV
from turboquant_pro.autoconfig import AutoConfig
from turboquant_pro.learned_codebook import fit_codebook
from turboquant_pro.pca import PCAMatryoshka
from turboquant_pro.pgvector import TurboQuantPGVector
from turboquant_pro.rope import RoPEAwareQuantizer


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel()
    b_f = b.ravel()
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-30))


def _batch_cosine(originals: np.ndarray, reconstructed: np.ndarray) -> tuple:
    dot = np.sum(originals * reconstructed, axis=1)
    na = np.linalg.norm(originals, axis=1)
    nb = np.linalg.norm(reconstructed, axis=1)
    sims = dot / (na * nb + 1e-30)
    return float(sims.mean()), float(sims.min()), float(sims.std())


def main():
    rng = np.random.default_rng(42)
    dim = 1024
    n_train = 2000
    n_test = 200

    # Simulate BGE-M3-like data with decaying variance
    eigenvalues = np.logspace(1.5, -1.5, dim).astype(np.float32)
    cov = np.diag(eigenvalues)
    X_train = rng.multivariate_normal(np.zeros(dim), cov, size=n_train).astype(
        np.float32
    )
    X_test = rng.multivariate_normal(np.zeros(dim), cov, size=n_test).astype(np.float32)

    # KV cache test data
    kv_tensor = rng.standard_normal((1, 8, 64, 128)).astype(np.float32)

    print("=" * 95)
    print("TURBOQUANT PRO: DIFFERENTIAL RELEASE BENCHMARKS (v0.5.0 -> v1.0.0)")
    print("=" * 95)
    print(f"Test data: {dim}-dim, {n_test} vectors (BGE-M3-like variance structure)")
    print()

    # ================================================================ #
    # EMBEDDING COMPRESSION: progression across versions                #
    # ================================================================ #

    print("-" * 95)
    print("EMBEDDING COMPRESSION (1024-dim, 3-bit baseline)")
    print("-" * 95)
    print(
        f"{'Version':>8s}  {'Feature':>35s}  {'Mean CosSim':>12s}  "
        f"{'Min CosSim':>11s}  {'Ratio':>7s}  {'Improvement':>12s}"
    )
    print("-" * 95)

    results = []

    # v0.5.0: Basic TurboQuant 3-bit (uniform, full dim)
    tq_basic = TurboQuantPGVector(dim=dim, bits=3, seed=42)
    sims = []
    for emb in X_test:
        c = tq_basic.compress_embedding(emb)
        r = tq_basic.decompress_embedding(c)
        sims.append(_cosine_similarity(emb, r))
    sims_arr = np.array(sims)
    baseline_mean = sims_arr.mean()
    results.append(
        ("v0.5.0", "TQ3 uniform (Lloyd-Max)", sims_arr.mean(), sims_arr.min(), 10.5)
    )
    print(
        f"{'v0.5.0':>8s}  {'TQ3 uniform (Lloyd-Max)':>35s}  "
        f"{sims_arr.mean():>12.6f}  {sims_arr.min():>11.6f}  "
        f"{'10.5x':>7s}  {'baseline':>12s}"
    )

    # v0.5.0: PCA-384 + TQ3
    pca = PCAMatryoshka(input_dim=dim, output_dim=384)
    pca.fit(X_train)
    pipeline = pca.with_quantizer(bits=3, seed=42)
    sims = []
    for emb in X_test:
        c = pipeline.compress(emb)
        r = pipeline.decompress(c)
        sims.append(_cosine_similarity(emb, r))
    sims_arr = np.array(sims)
    results.append(("v0.5.0", "PCA-384 + TQ3", sims_arr.mean(), sims_arr.min(), 27.7))
    print(
        f"{'v0.5.0':>8s}  {'PCA-384 + TQ3':>35s}  "
        f"{sims_arr.mean():>12.6f}  {sims_arr.min():>11.6f}  {'27.7x':>7s}  {'—':>12s}"
    )

    # v0.9.0: Eigenweighted mixed-precision
    ew_pipeline = pca.with_weighted_quantizer(avg_bits=3.0, seed=42)
    mean_ew, min_ew, _ = ew_pipeline.batch_cosine_similarity(X_test)
    delta = mean_ew - sims_arr.mean()
    results.append(("v0.9.0", "PCA-384 + eigenweighted 3.0b", mean_ew, min_ew, 29.7))
    print(
        f"{'v0.9.0':>8s}  {'PCA-384 + eigenweighted 3.0b':>35s}  "
        f"{mean_ew:>12.6f}  {min_ew:>11.6f}  {'29.7x':>7s}  {delta:>+12.6f}"
    )

    # v1.0.0: Learned codebook
    lq = fit_codebook(X_train, dim=dim, bits=3, seed=42)
    sims_lq = []
    for emb in X_test:
        c = lq.compress_embedding(emb)
        r = lq.decompress_embedding(c)
        sims_lq.append(_cosine_similarity(emb, r))
    sims_lq_arr = np.array(sims_lq)
    delta_lq = sims_lq_arr.mean() - baseline_mean
    results.append(
        ("v1.0.0", "TQ3 learned codebook", sims_lq_arr.mean(), sims_lq_arr.min(), 10.5)
    )
    print(
        f"{'v1.0.0':>8s}  {'TQ3 learned codebook':>35s}  "
        f"{sims_lq_arr.mean():>12.6f}  {sims_lq_arr.min():>11.6f}  "
        f"{'10.5x':>7s}  {delta_lq:>+12.6f}"
    )

    # ================================================================ #
    # KV CACHE COMPRESSION: progression across versions                 #
    # ================================================================ #

    print()
    print("-" * 95)
    print("KV CACHE COMPRESSION (head_dim=128, 8 heads, 64 tokens)")
    print("-" * 95)
    print(
        f"{'Version':>8s}  {'Feature':>35s}  {'Key CosSim':>11s}  "
        f"{'Val CosSim':>11s}  {'Avg bits':>9s}"
    )
    print("-" * 95)

    # v0.5.0: Uniform 3-bit
    tq_u = TurboQuantKV(head_dim=128, n_heads=8, bits=3, use_gpu=False, seed=0)
    ck = tq_u.compress(kv_tensor, packed=True)
    sim_k_u = _batch_cosine(
        kv_tensor.reshape(-1, 128), tq_u.decompress(ck).reshape(-1, 128)
    )[0]
    sim_v_u = sim_k_u
    print(
        f"{'v0.5.0':>8s}  {'Uniform K3/V3':>35s}  "
        f"{sim_k_u:>11.6f}  {sim_v_u:>11.6f}  {'3.0':>9s}"
    )

    # v0.9.0: Asymmetric K4/V3
    tq_a = TurboQuantKV(
        head_dim=128, n_heads=8, key_bits=4, value_bits=3, use_gpu=False, seed=0
    )
    ck_a = tq_a.compress(kv_tensor, packed=True, kind="key")
    cv_a = tq_a.compress(kv_tensor, packed=True, kind="value")
    sim_k_a = _batch_cosine(
        kv_tensor.reshape(-1, 128), tq_a.decompress(ck_a).reshape(-1, 128)
    )[0]
    sim_v_a = _batch_cosine(
        kv_tensor.reshape(-1, 128), tq_a.decompress(cv_a).reshape(-1, 128)
    )[0]
    print(
        f"{'v0.9.0':>8s}  {'Asymmetric K4/V3':>35s}  "
        f"{sim_k_a:>11.6f}  {sim_v_a:>11.6f}  {'3.5':>9s}"
    )

    # v0.9.0: RoPE-aware (simulated for LLaMA-3)
    rq = RoPEAwareQuantizer(
        head_dim=128,
        n_heads=8,
        default_bits=3,
        boost_bits=4,
        rope_base=500000.0,
        max_seq_len=8192,
        use_gpu=False,
        seed=0,
    )
    c_r = rq.compress(kv_tensor, packed=True)
    r_r = rq.decompress(c_r)
    sim_rope = _batch_cosine(kv_tensor.reshape(-1, 128), r_r.reshape(-1, 128))[0]
    print(
        f"{'v0.9.0':>8s}  {'RoPE-aware 4/3 (LLaMA-3)':>35s}  "
        f"{sim_rope:>11.6f}  {'—':>11s}  {rq.avg_bits:>9.2f}"
    )

    # v0.9.1: Auto-config (balanced)
    cfg = AutoConfig.from_pretrained("llama-3-8b", target="balanced")
    tq_auto = cfg.build_quantizer(seed=0)
    ck_auto = tq_auto.compress(kv_tensor, packed=True, kind="key")
    cv_auto = tq_auto.compress(kv_tensor, packed=True, kind="value")
    sim_k_auto = _batch_cosine(
        kv_tensor.reshape(-1, 128), tq_auto.decompress(ck_auto).reshape(-1, 128)
    )[0]
    sim_v_auto = _batch_cosine(
        kv_tensor.reshape(-1, 128), tq_auto.decompress(cv_auto).reshape(-1, 128)
    )[0]
    print(
        f"{'v0.9.1':>8s}  {'AutoConfig balanced K4/V3':>35s}  "
        f"{sim_k_auto:>11.6f}  {sim_v_auto:>11.6f}  {'3.5':>9s}"
    )

    # ================================================================ #
    # TEST COUNT PROGRESSION                                            #
    # ================================================================ #

    print()
    print("-" * 95)
    print("RELEASE HISTORY: FEATURE & TEST PROGRESSION")
    print("-" * 95)
    print(f"{'Version':>8s}  {'Key Features':>55s}  {'Tests':>6s}  {'Modules':>8s}")
    print("-" * 95)
    releases = [
        ("v0.5.0", "Autotune CLI, FAISS, vLLM plugin, Rust pgext", 175, 8),
        ("v0.6.0", "Model weight compression, SVD", 175, 9),
        ("v0.7.0", "Activation-space PCA, head-wise granularity", 175, 10),
        ("v0.8.0", "CUDA kernels, CompressedHNSW, L2 cache, GPU batch", 244, 14),
        ("v0.9.0", "Asymmetric K/V, eigenweighted, RoPE-aware, ANS codec", 272, 18),
        ("v0.9.1", "AutoConfig, from_model(), target presets", 303, 19),
        ("v0.10.0", "auto_compress, hardware profiles, incr. HNSW, export", 351, 23),
        ("v1.0.0", "Learned codebooks, multi-modal, observability", 397, 27),
    ]
    for ver, features, tests, modules in releases:
        print(f"{ver:>8s}  {features:>55s}  {tests:>6d}  {modules:>8d}")

    # ================================================================ #
    # SUMMARY TABLE                                                     #
    # ================================================================ #

    print()
    print("=" * 95)
    print("SUMMARY: QUALITY PROGRESSION (3-bit, 1024-dim embeddings)")
    print("=" * 95)
    print(
        f"{'Version':>8s}  {'Method':>35s}  {'CosSim':>8s}  "
        f"{'vs baseline':>12s}  {'Error reduction':>16s}"
    )
    print("-" * 85)
    baseline_err = 1.0 - baseline_mean
    for ver, method, mean, _, _ in results:
        err = 1.0 - mean
        delta = mean - baseline_mean
        err_reduction = (baseline_err - err) / baseline_err * 100
        print(
            f"{ver:>8s}  {method:>35s}  {mean:>8.6f}  "
            f"{delta:>+12.6f}  {err_reduction:>15.1f}%"
        )


if __name__ == "__main__":
    main()
