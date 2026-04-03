"""
Streaming KV cache with TurboQuant Pro tiered storage.

Demonstrates the L1/L2 tiered cache that automatically compresses
older tokens as the context window grows.
"""

import numpy as np

from turboquant_pro import TurboQuantKVCache

# Create a streaming cache
# - hot_window=512: keep 512 most recent tokens uncompressed
# - bits=3: compress older tokens at 3-bit precision (~5x savings)
cache = TurboQuantKVCache(
    head_dim=128,
    n_heads=8,
    bits=3,
    hot_window=512,
    use_gpu=False,
    seed=42,
)

# Simulate autoregressive generation: append one token at a time
rng = np.random.default_rng(0)
n_tokens = 2048

for i in range(n_tokens):
    # In practice, these come from your model's attention layer
    key = rng.standard_normal((1, 8, 1, 128)).astype(np.float32)
    value = rng.standard_normal((1, 8, 1, 128)).astype(np.float32)
    cache.append(key, value)

    if (i + 1) % 512 == 0:
        stats = cache.memory_stats()
        print(
            f"Token {i+1:4d}: "
            f"hot={cache.hot_length:3d}, "
            f"cold={cache.cold_length:4d}, "
            f"ratio={stats['effective_ratio']:.1f}x, "
            f"mem={stats['total_bytes']/1e6:.1f} MB"
        )

# Query the full context for attention
keys = cache.get_keys(0, cache.length)
values = cache.get_values(0, cache.length)
print(f"\nFull context: keys={keys.shape}, values={values.shape}")

# Query just the last 256 tokens (hot window only -- no decompression)
start = cache.length - 256
keys_recent = cache.get_keys(start, cache.length)
print(f"Recent 256:  keys={keys_recent.shape}")

# Final stats
stats = cache.memory_stats()
print(f"\nFinal memory: {stats['total_bytes']/1e6:.1f} MB")
print(f"Uncompressed equivalent: {stats['uncompressed_equivalent_bytes']/1e6:.1f} MB")
print(f"Effective ratio: {stats['effective_ratio']:.1f}x")
