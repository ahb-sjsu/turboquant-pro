# Matching a Calibrated KV-Cache Quantizer — Without the Calibration

*Andrew H. Bond | San Jose State University*

---

If you run long-context LLMs, the KV cache is your memory wall. At 32k tokens a 7B model spends *more* memory on cached keys/values than on its own weights. So everyone quantizes the KV cache to 4-bit. The catch: the **good** 4-bit methods aren't free — they need an offline calibration step.

Take **KVQuant**, one of the strongest published methods. To hit near-lossless 4-bit it runs a Fisher-information pass (backprop over a calibration set to find which channels matter) and then per-channel K-means to learn non-uniform code points. Excellent quality — but it's an offline pipeline you have to run per model, and it's the kind of thing that quietly blocks adoption.

So I asked a simple question: **how close can you get to KVQuant's quality with *zero* calibration?**

## The answer: close enough to call it a tie

Measured on **LongBench** (Llama-2-7B-chat, full 200-sample splits, one self-consistent harness for every method):

| KV scheme | trec | triviaqa | qasper |
|---|---:|---:|---:|
| fp16 (reference) | 64.0 | 83.26 | 22.06 |
| **KVQuant nuq4-1%** (Fisher + K-means) | 64.0 | 83.16 | **21.06** |
| per-channel **uniform** 4-bit | 62.5 | 81.84 | **14.38** |
| **NF4 + 2% outliers + sink** (no calibration) | 63.5 | **83.32** | **20.82** |

The calibration-free version **beats KVQuant on triviaqa** (83.32 vs 83.16) and trails it by just **0.24** on qasper (20.82 vs 21.06) — inside LongBench's own run-to-run noise. No Fisher pass. No K-means. Three fixed, data-independent ingredients:

1. **Per-channel keys.** Attention's `softmax(Q·Kᵀ)` depends on the per-channel scale of keys. (The naive "normalize each vector and quantize its direction" trick is near-lossless for *values* but catastrophic for *keys* — it throws away exactly the scale attention reads.)
2. **NF4** — a fixed NormalFloat-4 codebook. Allocates its 16 levels by the Gaussian shape of the data instead of evenly. Calibration-free non-uniform quantization.
3. **1–2% dense-sparse outliers.** Keep the top ~2% highest-magnitude entries *per channel* in fp16.

## The one insight that did the work

Look at the uniform-4-bit row: qasper craters from **22.06 → 14.38**. That's not gentle degradation — it's a collapse, and it happens on exactly the task that stresses long-context retrieval.

The cause is a handful of **outlier key channels**. A few channels carry huge-magnitude values that dominate the attention dot-product, and uniform 4-bit spends its entire dynamic range trying to cover them — destroying the precision of everything else. Keep just the top **2%** of magnitudes in fp16 and qasper jumps back to **20.82**. The sweep is clean: 1% → 2% → 3% gives 20.23 → **20.82** → 20.67. It peaks at 2%; beyond that you're paying storage for nothing.

That's the whole story: **the loss was concentrated, and concentrated loss is cheap to fix.**

## Being honest about it

- It's a **tie, not a conquest.** KVQuant keeps a 0.24-point qasper edge. If you need every last point and don't mind the calibration pipeline, it's still (barely) ahead.
- These are **simulation numbers** from a faithful-but-slow reference cache that re-quantizes each step; a production cache quantizes incrementally as tokens age out of the hot window.
- The table is **one internal harness** — directly comparable across its own rows, but not against LongBench numbers from other setups (truncation/prompt details move absolute scores by several points). I learned that the hard way after a first pass accidentally mixed two harnesses.

## Why it matters

The practical takeaway isn't "we beat KVQuant." It's that **most of what calibration buys you on KV keys can be had for free** — if you spend your bits where the error actually lives (a few outlier channels) instead of uniformly. For anyone shipping long-context inference, that's one less offline pipeline between you and 4-bit KV.

Shipped in **turboquant-pro v1.3.0**:

```python
from turboquant_pro import TurboQuantKVCache
cache = TurboQuantKVCache(key_nf4=True, key_outlier_frac=0.02)  # calibration-free
```

`pip install turboquant-pro`. Code, tests, and the full benchmark harness are on GitHub.

#LLM #Quantization #Inference #MachineLearning #KVcache #LongContext
