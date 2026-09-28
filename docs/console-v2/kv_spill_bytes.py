"""Measure compressed key bytes per token for the HF drop-in's spill sizes.

TurboQuantLayer spills `overflow` tokens per update. During decode that is 1 token
per step; a prefill longer than hot_window spills a block. The key quantizer is
PerChannelKV(head_dim, n_heads, bits=4, nf4_asym=True, outlier_frac=0.02), as in
hf_cache.py. numpy only, CPU, no model.
"""
import json
import sys

import numpy as np

sys.path.insert(0, ".")
from turboquant_pro.per_channel_kv import PerChannelKV  # noqa: E402

H, D = 8, 128
rng = np.random.default_rng(0)
kq = PerChannelKV(head_dim=D, n_heads=H, bits=4, nf4_asym=True, outlier_frac=0.02)
out = {}
for S in (1, 2, 16, 64, 256, 1024):
    x = rng.standard_normal((1, H, S, D)).astype(np.float32)
    c = kq.compress(x, packed=True)
    nb = c.nbytes() if hasattr(c, "nbytes") else None
    fp16 = 2 * H * S * D
    out[S] = {"compressed_bytes": nb, "fp16_bytes": fp16,
              "ratio_vs_fp16": (fp16 / nb) if nb else None,
              "bytes_per_token_per_head": (nb / (H * S)) if nb else None}
    print(S, out[S], flush=True)
json.dump(out, open("kv_spill_bytes.json", "w"), indent=1)
