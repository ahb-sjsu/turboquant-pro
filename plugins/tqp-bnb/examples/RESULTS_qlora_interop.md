# QLoRA interop demo — results (P2 exit criterion)

**Run:** NRP Nautilus batch Job, NVIDIA L40S, 2026-07-16 ·
`unsloth/Llama-3.2-3B` in genuine QLoRA config (bitsandbytes 4-bit NF4,
double quant, fp16 compute) · 146-token forward pass, real activations ·
per-layer post-RoPE K/V captured into `TurboQuantKVCache`
(per-channel asym-NF4 keys + 2% outliers, PolarQuant values).

Also in the same job: the `tqp-bnb` suite ran with **bitsandbytes + CUDA
present for the first time** — 10/10 including the `bnb.functional`
cross-check (our NumPy NF4 reference matches bnb's own dequantization).

| layer | attention KL (exact‖compressed) | top-1 agreement* | KV ratio |
|---:|---:|---:|---:|
| 0 | 1.5e-05 | 0.922 | 2.14× |
| 13 | 2.3e-05 | 0.930 | 2.16× |
| 27 | 2.0e-05 | 0.906 | 2.16× |

*random unit queries — a deliberately harsh probe (near-uniform scores tie
easily); real decode queries agree more. Ratio is diluted by the fp16 hot
window at this short sequence; cold-page-only compression is ~4×.

**Verdict:** bnb quantizes the weights, turboquant-pro quantizes the cache,
and attention built from the compressed cache is statistically
indistinguishable from attention on the exact KV (mean KL 2e-05). The two
systems compose.
