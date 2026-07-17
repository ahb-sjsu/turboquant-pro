# Long-context fused decode — 64k / 128k, H100

Harness: [`h200_longcontext_bench.py`](h200_longcontext_bench.py). Steady-state
`TurboQuantKVCache.fused_decode` (prepared per-page PCK blocks + CuPy kernels)
vs the decompress-then-attend fallback, per decode step, at contexts no smaller
GPU in the fleet can hold. 8 heads × 128 dim, 4-bit per-channel asym-NF4 keys
(2 % fp16 outliers), PolarQuant values, 512-token fp16 hot window.

**Run:** lightning H100 80GB HBM3, torch 2.8.0+cu128, cupy-cuda12x.

| ctx | cold pages | fill | first call | **steady fused** | reconstruct | **speedup** | KV ratio | max err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 65,536 | 254 | 10.3 s | 3771.8 ms | **128.28 ms** | 2588.7 ms | **20.2×** | 6.49× | 2.5e-08 |
| 131,072 | 510 | 19.9 s | 7356.7 ms | **245.91 ms** | 5121.1 ms | **20.8×** | 6.63× | 2.8e-08 |

This extends the measured 2k–32k curve (2.0× / 5.6× / 12.5× on GV100) into the
64k–128k regime: the fused-decode speedup **plateaus at ~20×** and KV compression
holds at **~6.5×**, exact to ~3e-8 even at 128k tokens. The first call builds the
prepared per-page blocks (a one-time 3.8 s / 7.4 s at these sizes); every
subsequent decode step is the steady number. Reconstruct cost scales linearly
with context (2.6 s → 5.1 s) while fused scales with it too but ~20× lower, so
the absolute saving per decode step grows with the cache — 2.5 s saved at 64k,
4.9 s at 128k.

(Env note: the lightning image shipped numpy 2.5.1, so cupy/scipy logged
"compiled against NumPy 1.x" warnings during import — cosmetic; the bench and the
CuPy kernels ran, hence the real numbers. The bench is named `h200_` for the
141 GB-class target it was written for; H100 80 GB holds 128k of this config
comfortably.)
