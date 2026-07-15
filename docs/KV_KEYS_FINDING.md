# KV keys need per-channel quantization, not PolarQuant

**TL;DR.** TurboQuant's PolarQuant quantizer (random rotation + per-vector L2
normalization + Lloyd-Max codebook) is **near-lossless for values** but
**catastrophic for keys** — it destroys generation while *passing* reconstruction
tests. A **per-channel** key quantizer (`PerChannelKV`) fixes it at equal or fewer
bits. The existing reconstruction-fidelity benchmarks cannot detect the failure
because, for keys, reconstruction error is **anti-correlated** with perplexity.

## Evidence

Qwen2.5-1.5B-Instruct, wikitext-2 (test), perplexity via fake-quantization at the
**post-RoPE** key tensors (`apply_rotary_pos_emb` monkeypatch); values via PolarQuant.
Two-tier fp16 sink+hot applied to keys. fp16 baseline ppl = **12.24**.

| key quantizer | bits | ppl | dppl | key recon (mean / max) |
|---|---:|---:|---:|---:|
| *(none — values-only PolarQuant)* | — | 13.12 | +0.88 | — |
| **PolarQuant (TurboQuantKV, shipped)** | K4 | **10643** | **+10631** | 0.095 / 0.119 |
| per-channel uniform | K4 | **14.91** | +2.67 | 0.062 / 0.080 |
| per-channel NUQ | K3 | **15.77** | +3.54 | 0.148 / 0.190 |

Same harness, same injection point, same values — **the only variable is the key
quantizer.** Pre-RoPE keys behave the same (PolarQuant ppl ~22k), so it is not a
domain artifact; K3 vs K4 does not rescue it.

### The reconstruction metric is anti-correlated with quality (for keys)
PolarQuant K4 has the **best** key reconstruction (0.095) and the **worst**
perplexity (10643). Per-channel NUQ K3 has **2.4× higher** reconstruction error
(0.148) and **670× better** perplexity (15.8). **Cosine-similarity / per-layer
attention-output error cannot detect this failure** — only a generation
(perplexity) metric can.

## Why PolarQuant fails on keys

PolarQuant normalizes each key vector by its L2 norm and quantizes the **direction**
(it stores the norm exactly and the direction in a fixed Gaussian codebook). That is
ideal for **values**, which attention *averages* (robust to direction error). But
**keys feed the dot product** `softmax(Q·K^T / √d)`, where:

- per-channel scale matters: a few key channels dominate `Q·K`, and per-vector
  normalization discards exactly that per-channel scale structure;
- softmax amplifies small, correlated key perturbations nonlinearly;
- GQA compounds it (here 2 KV heads serve 12 query heads — corrupted keys poison
  every query head).

This is precisely why KIVI and KVQuant treat keys **per-channel**, not per-vector.

## The fix: `PerChannelKV`

A per-channel asymmetric (optionally non-uniform / NUQ) quantizer for keys:
each head-dim **channel** gets its own scale computed over the token axis, preserving
the per-channel structure attention relies on. Validated to recover near-fp16
perplexity (above), with unit tests covering round-trip fidelity, **outlier-channel
isolation**, and **dot-product top-k preservation** (the property attention needs).

**Recommended configuration:** asymmetric by *quantizer*, not just by bits —
**keys → `PerChannelKV`**, **values → PolarQuant (`TurboQuantKV`)**. TurboQuant's
two-tier hot/cold cache and value path are unchanged and excellent.

## The general principle: condition (A2)

This finding is an instance of a theorem-level boundary, made precise in the
companion theory paper ([the-angular-observer](https://github.com/ahb-sjsu/the-angular-observer),
"Keep the Angle" v0.8, deterministic angular-transfer theorem). A quotient
that separates scale from direction and discards scale is safe **exactly
when the consumer's metric is carried by the tangential part of the
displacement** — condition (A2): `|Δz|² − (Δ|z|)²` must lower-bound the
metric being preserved. Cosine ranking over isotropic embeddings satisfies
(A2); `softmax(Q·Kᵀ)` over post-RoPE keys does not — the logits read
per-channel scale structure that per-vector normalization deletes, and the
keys' directions concentrate in a cone so tightly that the informative
angular displacement sits *below the quantizer's cell size* (which is also
why reconstruction cosine stays deceptively high). The paper cites this
finding back as the scope boundary of the polar move: *keep the geometry —
sometimes that is the angle, sometimes angle plus norm, sometimes
per-channel scale.*

The lesson is now an installed instrument, not a memory:
`turboquant_pro.a2_probe.recommend_key_quantizer` runs the family choice as
a calibration-time probe against the declared consumer metric (it reproduces
this incident on synthetic key statistics in `tests/test_a2_probe.py`), and
`QualityMonitor` streams the (A2) tangential fraction so norm-dominated
drift becomes a dashboard alert rather than a release.

## Benchmark gap to close

Add a **generation/perplexity gate** to the KV benchmark suite (e.g. wikitext-2 /
C4 perplexity, then LongBench at ≥4k context). The current reconstruction-fidelity
benchmarks gave false confidence on keys; perplexity catches it immediately.

## Reproduce

```
# scheme-level, portable (CPU/GPU/Colab):
python benchmarks/kv_quant_shootout.py
# post-RoPE real-library + per-channel fix:
NB_MODEL=Qwen/Qwen2.5-1.5B-Instruct NB_SEQ=512 python benchmarks/benchmark_kvcache_postrope.py
# unit tests:
pytest tests/test_per_channel_kv.py -q
```

*Caveats (honest): fake-quantization, not the streaming-cache kernels; validated on
Qwen2.5 (1.5B/7B). The mechanism is established by controls (values + per-channel
keys both work in the same harness); a non-GQA model and LongBench@4k are the
recommended next confirmations.*
