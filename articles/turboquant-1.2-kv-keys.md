# When 0.99 cosine similarity isn't enough: fixing KV-cache *key* quantization

**TurboQuant Pro 1.2.0** ships a correction we did not expect to find in our own
library: the quantizer we used for KV-cache **keys** was silently destroying model
quality — and every reconstruction-fidelity benchmark we had said it was fine.

## The setup

TurboQuant's KV quantizer is *PolarQuant*: rotate each key/value vector by a fixed
random rotation, normalize it by its L2 norm, and quantize the **direction** against a
Lloyd-Max codebook. It's elegant, it's fast, and on our fidelity benchmarks it looked
excellent — **0.995 cosine similarity** for 4-bit keys.

So we did something we should have done earlier: we measured **perplexity** — actual
generation quality — instead of reconstruction error.

## The result

On Qwen2.5 (1.5B and 7B), quantizing the post-RoPE keys with PolarQuant at 4 bits:

| key quantizer | bits | perplexity | reconstruction error |
|---|---:|---:|---:|
| none (fp16) | 16 | 12.2 | 0.000 |
| **PolarQuant (what we shipped)** | 4 | **≈ 10,000** | 0.095 |
| **per-channel (the fix)** | 4 | **≈ 15** | 0.062 |

Read that twice. The PolarQuant keys reconstruct *more faithfully* (0.095 error) than
our previous benchmarks demanded, yet perplexity blows up by **three orders of
magnitude**. Values, by contrast, are near-lossless under PolarQuant. The failure is
specific to keys, and it holds across model sizes and from 512 to 4k context.

## Why keys are different

Attention computes `softmax(Q·Kᵀ / √d)`. The score for each key is a **dot product**,
and softmax amplifies small, correlated perturbations. PolarQuant preserves each key's
*norm* but quantizes its *direction* — discarding the **per-channel scale** that the dot
product depends on. Values don't care: attention *averages* them, which is robust to
direction error. Keys care enormously. This is exactly why KIVI and KVQuant treat keys
**per-channel**; we had been treating them per-vector.

## The uncomfortable part: our metric was the problem

The reason this survived multiple releases is that our KV benchmarks measured
**reconstruction fidelity** (cosine similarity, per-layer attention-output error). For
keys, that metric is **anti-correlated** with the thing that matters: the per-channel
scheme that *wins* on perplexity has *higher* reconstruction error. A benchmark that
can't distinguish a working model from a broken one isn't a benchmark — it's a comfort
blanket. 1.2.0 adds a generation/perplexity gate (`benchmarks/kv_quant_shootout.py`,
`benchmark_kvcache_postrope.py`) so this can never hide again.

## The fix

`PerChannelKV` quantizes each head-dim channel with its own asymmetric (optionally
non-uniform) scale, with 2/3/4-bit packing. `TurboQuantKVCache` now uses **per-channel
keys by default**, with PolarQuant retained for values — asymmetric *by quantizer*, not
just by bit-width. Keys return to near-fp16 perplexity at the same bit budget.

```python
from turboquant_pro import TurboQuantKVCache
cache = TurboQuantKVCache(head_dim=128, n_heads=8, bits=4)  # per-channel keys by default
```

## What we're taking from this

1. **Measure the end metric.** Reconstruction fidelity is a proxy; for KV-cache keys it's
   a *misleading* proxy. Perplexity is cheap — there's no excuse not to gate on it.
2. **Asymmetry is structural, not just numeric.** Keys and values fail differently, so
   they deserve different *algorithms*, not merely different bit-widths.
3. **Publish the negative result.** We shipped a quantizer that was wrong for keys.
   Saying so plainly, with the numbers, is the whole point.

Full write-up and reproduction: [`docs/KV_KEYS_FINDING.md`](../docs/KV_KEYS_FINDING.md).
Reproduce in five minutes on CPU: `python benchmarks/kv_quant_shootout.py`.

*TurboQuant Pro is MIT-licensed. `pip install turboquant-pro` (>= 1.2.0).*
