# Matched KVQuant's 4-bit KV-cache quality on LongBench — without the calibration step

*(r/LocalLLaMA. Also fits r/MachineLearning as `[P]`.)*

**TL;DR:** KVQuant gets near-lossless 4-bit KV cache but needs an offline Fisher-gradient + K-means calibration pass per model. I tried to match it with *zero* calibration — just per-channel keys + a fixed NF4 codebook + keeping the top ~2% of key magnitudes in fp16. On LongBench (Llama-2-7B-chat, full 200-sample splits) it's a dead heat: beats KVQuant on triviaqa, trails by 0.24 on qasper. Shipped in `turboquant-pro` v1.3.0. Honest writeup below, including the bugs that almost gave me fake results.

---

## The setup

KV cache is the long-context memory bottleneck, so 4-bit KV quant is standard. The strong methods need calibration, though — **KVQuant** runs a Fisher-information pass (backprop over calibration data) + per-channel K-means to learn non-uniform code points. Great quality, but it's an offline pipeline.

Question: how close can you get **calibration-free**?

## The recipe (all data-independent)

1. **Per-channel keys.** Per-vector normalization ("quantize the direction") is fine for values but *destroys* keys — it discards the per-channel scale that `softmax(Q·Kᵀ)` actually reads. Keys need per-channel asymmetric scales.
2. **NF4** — fixed NormalFloat-4 codebook (16 levels placed by the Gaussian, scaled per channel by abs-max). Non-uniform quantization with no calibration.
3. **1–2% dense-sparse outliers** — keep the top-magnitude entries *per channel* in fp16.

## Results (LongBench, Llama-2-7B-chat, full 200-sample splits, single harness)

| KV scheme | trec | triviaqa | qasper |
|---|---:|---:|---:|
| fp16 | 64.0 | 83.26 | 22.06 |
| KVQuant nuq4-1% (Fisher + K-means) | 64.0 | 83.16 | **21.06** |
| per-channel **uniform** 4-bit | 62.5 | 81.84 | **14.38** |
| **NF4 + 2% outliers + sink** (no calib) | 63.5 | **83.32** | **20.82** |

The outlier sweep is the punchline. qasper at **1% / 2% / 3% = 20.23 / 20.82 / 20.67** — peaks at 2%.

## Why it works: it's a handful of outlier *key* channels

Uniform 4-bit drops qasper from 22.06 to **14.38** — a collapse, not a slope. The reason: a few key channels carry huge values that dominate attention, and uniform quantization burns its whole range covering them, wrecking precision everywhere else. Keep the top 2% in fp16 → back to 20.82. Concentrated loss, cheap fix.

## The bugs that almost fooled me (the actually-useful part)

- **My "quantized" cache was secretly running fp16.** In transformers 4.38, `model.generate(past_key_values=my_cache)` is **silently ignored** — generate instantiates its own `DynamicCache`. I only caught it because my 2-bit sanity run scored *identical* to fp16 (64.0/83.26/22.06, exact to the decimal — impossible if anything were actually quantizing). Fix: monkeypatch `DynamicCache.update` globally. If you're benchmarking a custom KV cache through `.generate()`, **always run an aggressive-bit sanity check** — if 2-bit ≈ fp16, your cache isn't wired in.
- **An NF4 dtype landmine.** The NF4 codebook was float32; `nf4[idx] * amax` promoted the dequantized keys to float32 → SDPA threw a dtype mismatch against fp16 queries. Never surfaced earlier *because the first bug meant the NF4 path never ran.* Bugs hiding bugs.
- **A harness mismatch.** My first comparison accidentally put KVQuant and the baselines on two different LongBench harnesses (different truncation) — worth ~6 points on qasper. Absolute LongBench scores are *not* portable across setups; only same-harness rows are comparable. Re-ran everything in one harness.
- **Consumer-GPU roulette.** Ran this on spot RTX-3090 nodes: one node died mid-run, one GPU hard-faulted ("Unable to determine device handle"), kubelets dropped repeatedly. Checkpoint every variant off-box.

## Honest caveats

- **It's a tie, not a win.** KVQuant keeps a 0.24-pt qasper edge. Within noise, but it's ahead.
- **Simulation numbers** — faithful-but-slow reference cache that re-quantizes the settled window each step. A production cache quantizes incrementally as tokens leave the hot window.
- Single internal harness; rows are comparable to each other, not to published LongBench numbers.

## Use it

```python
from turboquant_pro import TurboQuantKVCache
cache = TurboQuantKVCache(key_nf4=True, key_outlier_frac=0.02)
```

`pip install turboquant-pro` (v1.3.0). Quantizer, cache, tests, and the LongBench runner are all in the repo. There's also a **reproduction notebook** (`benchmarks/reproduce_calibration_free_kv.ipynb`) — the CPU mechanism demo runs in seconds (no GPU/model), plus copy-paste commands for the full GPU LongBench run. Happy to answer questions / take shots at the methodology.
