"""
llama.cpp integration benchmark for TurboQuant Pro.

Measures the overhead of TurboQuant Pro compression in a simulated
llama.cpp inference pipeline. This benchmark does not require an
actual llama.cpp installation -- it simulates the KV cache access
patterns.

For real llama.cpp benchmarks, integrate TurboQuantKVCache into the
llama-cpp-python bindings and measure end-to-end generation speed.

Usage:
    python benchmarks/benchmark_llama.py
"""

from __future__ import annotations

import time

import numpy as np

from turboquant_pro import TurboQuantKVCache


def simulate_llama_generation(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    prompt_len: int,
    gen_len: int,
    hot_window: int,
    bits: int,
) -> dict:
    """Simulate autoregressive generation with compressed KV cache.

    Creates one TurboQuantKVCache per layer (as a real model would)
    and measures throughput.

    Returns:
        Dict with timing and memory statistics.
    """
    rng = np.random.default_rng(42)

    # One cache per layer
    caches = [
        TurboQuantKVCache(
            head_dim=head_dim,
            n_heads=n_kv_heads,
            bits=bits,
            hot_window=hot_window,
            use_gpu=False,
            seed=layer_idx,
        )
        for layer_idx in range(n_layers)
    ]

    total_tokens = prompt_len + gen_len

    # Prefill: process prompt tokens in batch
    t_prefill_start = time.perf_counter()
    for layer_cache in caches:
        k_prompt = rng.standard_normal((1, n_kv_heads, prompt_len, head_dim)).astype(
            np.float32
        )
        v_prompt = rng.standard_normal((1, n_kv_heads, prompt_len, head_dim)).astype(
            np.float32
        )
        # Append one token at a time (simulates cache-aware prefill)
        for t in range(prompt_len):
            layer_cache.append(
                k_prompt[:, :, t : t + 1, :],
                v_prompt[:, :, t : t + 1, :],
            )
    t_prefill = time.perf_counter() - t_prefill_start

    # Decode: generate tokens one at a time
    t_decode_start = time.perf_counter()
    for _ in range(gen_len):
        for layer_cache in caches:
            k = rng.standard_normal((1, n_kv_heads, 1, head_dim)).astype(np.float32)
            v = rng.standard_normal((1, n_kv_heads, 1, head_dim)).astype(np.float32)
            layer_cache.append(k, v)

            # Simulate attention: read full KV cache
            _ = layer_cache.get_keys(0, layer_cache.length)
            _ = layer_cache.get_values(0, layer_cache.length)
    t_decode = time.perf_counter() - t_decode_start

    # Memory stats (sum across layers)
    total_mem = 0.0
    total_uncomp = 0.0
    for c in caches:
        stats = c.memory_stats()
        total_mem += stats["total_bytes"]
        total_uncomp += stats["uncompressed_equivalent_bytes"]

    return {
        "prefill_ms": t_prefill * 1000,
        "decode_ms": t_decode * 1000,
        "decode_tok_per_s": gen_len / t_decode,
        "total_mem_mb": total_mem / 1e6,
        "uncomp_mem_mb": total_uncomp / 1e6,
        "ratio": total_uncomp / max(total_mem, 1),
        "total_tokens": total_tokens,
    }


def main():
    print("=" * 78)
    print("TurboQuant Pro: Simulated llama.cpp Generation Benchmark")
    print("=" * 78)
    print()

    configs = [
        # (model_name, n_layers, n_kv_heads, head_dim, prompt, gen, hot)
        ("Llama-3.1-8B (sim)", 32, 8, 128, 256, 256, 256),
        ("Gemma-4-27B (sim)", 36, 16, 256, 256, 256, 256),
    ]

    for name, n_layers, n_kv_heads, head_dim, prompt, gen, hot in configs:
        print(f"Model: {name}")
        print(
            f"  layers={n_layers}, kv_heads={n_kv_heads}, "
            f"head_dim={head_dim}, prompt={prompt}, gen={gen}"
        )

        for bits in [2, 3, 4]:
            result = simulate_llama_generation(
                n_layers=n_layers,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                prompt_len=prompt,
                gen_len=gen,
                hot_window=hot,
                bits=bits,
            )
            print(
                f"  {bits}-bit: "
                f"prefill={result['prefill_ms']:.0f}ms, "
                f"decode={result['decode_tok_per_s']:.1f} tok/s, "
                f"mem={result['total_mem_mb']:.1f}MB "
                f"(vs {result['uncomp_mem_mb']:.1f}MB), "
                f"ratio={result['ratio']:.1f}x"
            )
        print()


if __name__ == "__main__":
    main()
