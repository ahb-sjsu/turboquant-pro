"""
Basic TurboQuant-KV usage: compress and decompress KV cache tensors.
"""

import numpy as np

from turboquant_kv import TurboQuantKV

# Simulate a KV cache tensor: (batch=1, n_heads=16, seq_len=1024, head_dim=256)
rng = np.random.default_rng(42)
key_tensor = rng.standard_normal((1, 16, 1024, 256)).astype(np.float16)

# Create compressor (3-bit quantization, CPU mode)
tq = TurboQuantKV(head_dim=256, n_heads=16, bits=3, use_gpu=False, seed=0)

# Compress with bit-packing for maximum savings
compressed = tq.compress(key_tensor, packed=True)
print(f"Original:   {key_tensor.nbytes / 1e6:.1f} MB")
print(f"Compressed: {compressed.nbytes() / 1e6:.1f} MB")
print(f"Ratio:      {compressed.compression_ratio(256):.1f}x")

# Decompress
reconstructed = tq.decompress(compressed)
print(f"Shape:      {reconstructed.shape}")

# Verify quality
flat_orig = key_tensor.astype(np.float32).reshape(-1, 256)
flat_recon = reconstructed.reshape(-1, 256)
dot = np.sum(flat_orig * flat_recon, axis=-1)
norm_o = np.linalg.norm(flat_orig, axis=-1)
norm_r = np.linalg.norm(flat_recon, axis=-1)
cos_sim = np.mean(dot / np.maximum(norm_o * norm_r, 1e-30))
print(f"Cosine sim: {cos_sim:.4f}")

# Estimate memory for a real model
est = TurboQuantKV.estimate_memory(
    n_layers=36,
    n_kv_heads=16,
    head_dim=256,
    seq_len=8192,
    bits=3,
    bit_packed=True,
)
print("\nGemma 4 27B @ 8K context:")
print(f"  Original:   {est['original_gb']:.3f} GB")
print(f"  Compressed: {est['compressed_gb']:.3f} GB")
print(f"  Saved:      {est['saved_gb']:.3f} GB ({est['ratio']:.1f}x)")
