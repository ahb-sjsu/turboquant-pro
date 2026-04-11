"""
Auto-config benchmark: demonstrate the benefit of automatic tuning
across models and target presets vs. naive uniform 3-bit compression.

Usage:
    python benchmarks/benchmark_autoconfig.py
"""

from __future__ import annotations

import numpy as np

from turboquant_pro import TurboQuantKV
from turboquant_pro.autoconfig import AutoConfig, list_models
from turboquant_pro.rope import RoPEAwareQuantizer


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    dot = np.sum(a_flat * b_flat, axis=-1)
    norm_a = np.linalg.norm(a_flat, axis=-1)
    norm_b = np.linalg.norm(b_flat, axis=-1)
    return float(np.mean(dot / np.maximum(norm_a * norm_b, 1e-30)))


def benchmark_autoconfig_vs_naive():
    """Compare auto-configured compression vs naive uniform 3-bit."""
    print("=" * 90)
    print("AUTO-CONFIG vs NAIVE: Quality & Memory Comparison")
    print("=" * 90)
    print()

    rng = np.random.default_rng(42)

    print(
        f"{'Model':>15s}  {'Method':>22s}  "
        f"{'Key CosSim':>10s}  {'Val CosSim':>10s}  "
        f"{'KV GB':>7s}  {'Ratio':>6s}  {'Saved GB':>9s}"
    )
    print("-" * 90)

    for model in ["llama-3-8b", "gemma-4-27b", "qwen2.5-7b", "mistral-7b"]:
        cfg = AutoConfig.from_pretrained(model)
        head_dim = cfg.head_dim
        n_heads = cfg.n_kv_heads
        seq_len = min(cfg.max_seq_len, 8192)

        tensor = rng.standard_normal((1, n_heads, 64, head_dim)).astype(np.float32)

        # --- Naive: uniform 3-bit ---
        tq_naive = TurboQuantKV(
            head_dim=head_dim,
            n_heads=n_heads,
            bits=3,
            use_gpu=False,
            seed=0,
        )
        ck_n = tq_naive.compress(tensor, packed=True)
        cv_n = tq_naive.compress(tensor, packed=True)
        rk_n = tq_naive.decompress(ck_n)
        rv_n = tq_naive.decompress(cv_n)
        sim_k_naive = _cosine_similarity(tensor, rk_n)
        sim_v_naive = _cosine_similarity(tensor, rv_n)
        mem_naive = TurboQuantKV.estimate_memory(
            n_layers=cfg.n_layers,
            n_kv_heads=n_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            bits=3,
            bit_packed=True,
        )

        print(
            f"{model:>15s}  {'Naive K3/V3':>22s}  "
            f"{sim_k_naive:>10.6f}  {sim_v_naive:>10.6f}  "
            f"{mem_naive['compressed_gb']:>7.3f}  "
            f"{mem_naive['ratio']:>5.1f}x  "
            f"{mem_naive['saved_gb']:>9.3f}"
        )

        # --- Auto-config targets ---
        for target in ["balanced", "quality", "compression"]:
            cfg_t = AutoConfig.from_pretrained(model, target=target)
            tq_auto = cfg_t.build_quantizer(seed=0)

            ck_a = tq_auto.compress(tensor, packed=True, kind="key")
            cv_a = tq_auto.compress(tensor, packed=True, kind="value")
            rk_a = tq_auto.decompress(ck_a)
            rv_a = tq_auto.decompress(cv_a)
            sim_k_auto = _cosine_similarity(tensor, rk_a)
            sim_v_auto = _cosine_similarity(tensor, rv_a)

            mem_auto = cfg_t.estimate_memory(seq_len=seq_len)

            label = f"Auto {target} K{cfg_t.key_bits}/V{cfg_t.value_bits}"

            print(
                f"{'':>15s}  {label:>22s}  "
                f"{sim_k_auto:>10.6f}  {sim_v_auto:>10.6f}  "
                f"{mem_auto['compressed_gb']:>7.3f}  "
                f"{mem_auto['ratio']:>5.1f}x  "
                f"{mem_auto['saved_gb']:>9.3f}"
            )
        print()


def benchmark_rope_aware_by_model():
    """Show RoPE-aware improvement for each model at its native context."""
    print("=" * 90)
    print("ROPE-AWARE IMPROVEMENT BY MODEL")
    print("=" * 90)
    print()

    rng = np.random.default_rng(42)

    print(
        f"{'Model':>15s}  {'rope_theta':>12s}  {'max_seq':>8s}  "
        f"{'Boosted':>8s}  {'Uniform':>10s}  {'RoPE-aware':>10s}  "
        f"{'Delta':>8s}  {'Avg bits':>9s}"
    )
    print("-" * 90)

    for model in ["llama-3-8b", "gemma-4-27b", "qwen2.5-7b", "mistral-7b"]:
        cfg = AutoConfig.from_pretrained(model, target="balanced")
        head_dim = cfg.head_dim
        n_heads = cfg.n_kv_heads

        tensor = rng.standard_normal((1, n_heads, 64, head_dim)).astype(np.float32)

        # Uniform 3-bit baseline
        tq_u = TurboQuantKV(
            head_dim=head_dim,
            n_heads=n_heads,
            bits=3,
            use_gpu=False,
            seed=0,
        )
        c_u = tq_u.compress(tensor, packed=True)
        sim_u = _cosine_similarity(tensor, tq_u.decompress(c_u))

        # RoPE-aware
        rq = cfg.build_rope_quantizer(seed=0)
        if isinstance(rq, RoPEAwareQuantizer):
            c_r = rq.compress(tensor, packed=True)
            r_r = rq.decompress(c_r)
            sim_r = _cosine_similarity(tensor, r_r)
            stats = rq.stats()
            n_boost = stats["n_boost_dims"]
            avg_bits = rq.avg_bits
        else:
            sim_r = sim_u
            n_boost = 0
            avg_bits = 3.0

        delta = sim_r - sim_u
        print(
            f"{model:>15s}  {cfg.rope_theta:>12.0f}  "
            f"{cfg.max_seq_len:>8d}  "
            f"{n_boost:>7d}/{head_dim}  "
            f"{sim_u:>10.6f}  {sim_r:>10.6f}  "
            f"{delta:>+8.6f}  {avg_bits:>9.2f}"
        )


def benchmark_memory_savings():
    """Show memory savings across models and targets at 8K context."""
    print("\n" + "=" * 90)
    print("MEMORY SAVINGS: Auto-Config at 8K Context (fp16 baseline)")
    print("=" * 90)
    print()

    print(
        f"{'Model':>15s}  {'Layers':>7s}  {'fp16 GB':>8s}  "
        f"{'Target':>12s}  {'Config':>7s}  "
        f"{'Comp GB':>8s}  {'Ratio':>6s}  {'Saved':>8s}"
    )
    print("-" * 82)

    for model in list_models():
        cfg_b = AutoConfig.from_pretrained(model, target="balanced")
        mem_fp16 = cfg_b.estimate_memory(seq_len=8192)
        first = True

        for target in ["balanced", "compression", "extreme"]:
            cfg = AutoConfig.from_pretrained(model, target=target)
            mem = cfg.estimate_memory(seq_len=8192)
            label = f"K{cfg.key_bits}/V{cfg.value_bits}"

            if first:
                print(
                    f"{model:>15s}  {cfg.n_layers:>7d}  "
                    f"{mem_fp16['original_gb']:>8.3f}  "
                    f"{target:>12s}  {label:>7s}  "
                    f"{mem['compressed_gb']:>8.3f}  "
                    f"{mem['ratio']:>5.1f}x  "
                    f"{mem['saved_gb']:>7.3f}"
                )
                first = False
            else:
                print(
                    f"{'':>15s}  {'':>7s}  {'':>8s}  "
                    f"{target:>12s}  {label:>7s}  "
                    f"{mem['compressed_gb']:>8.3f}  "
                    f"{mem['ratio']:>5.1f}x  "
                    f"{mem['saved_gb']:>7.3f}"
                )
        print()


def main():
    benchmark_autoconfig_vs_naive()
    benchmark_rope_aware_by_model()
    benchmark_memory_savings()

    print("\n" + "=" * 90)
    print("SUMMARY: One-liner API")
    print("=" * 90)
    print()
    print("  # Before (manual, error-prone):")
    print("  tq = TurboQuantKV(head_dim=128, n_heads=8, bits=3)")
    print()
    print("  # After (auto-tuned, model-aware):")
    print('  tq = TurboQuantKV.from_model("llama-3-8b")')
    print('  tq = TurboQuantKV.from_model("gemma-4-27b", target="compression")')
    print()

    cfg = AutoConfig.from_pretrained("llama-3-8b")
    print(f"  {cfg!r}")
    print(f"  Summary: {cfg.summary()}")


if __name__ == "__main__":
    main()
