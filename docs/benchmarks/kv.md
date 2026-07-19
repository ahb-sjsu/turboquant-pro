# KV-cache benchmarks

Generation-quality and memory-savings numbers for TurboQuant Pro's KV-cache compression, evaluated on the metric that matters (perplexity / memory) rather than reconstruction cosine.

Part of [TurboQuant Pro](../../README.md). See also [KV keys finding](../KV_KEYS_FINDING.md).

### KV-cache: generation quality

**Perplexity is the metric that matters.** Fake-quantized KV during a real forward pass on wikitext-2 (keys quantized post-RoPE, values via PolarQuant); `ppl / key-recon-error` shown. Reproduce on CPU in ~5 min: `python benchmarks/kv_quant_shootout.py`.

| model | ctx | fp16 ppl | PolarQuant K4 keys | **per-channel K4 keys (v1.2.0)** |
|-------|----:|---------:|-------------------:|---------------------------------:|
| Qwen2.5-1.5B | 512  | 12.2 | 10,643 / 0.095 | **14.9 / 0.062** |
| Qwen2.5-1.5B | 4096 |  9.96 | 49,043 / 0.101 | **11.8 / 0.080** |
| Qwen2.5-7B   | 512  |  8.98 |  4,231 / 0.096 | **9.6 / 0.058**  |

PolarQuant keys blow perplexity up by **2–4 orders of magnitude** while values stay near-lossless. Reconstruction error is **not** a valid proxy: the per-channel **NUQ-3bit** variant reconstructs *worse* than PolarQuant-K4 (0.148 vs 0.095) yet scores **~700× better** perplexity (ppl ≈ 16).

> ⚠️ A high "Key CosSim" (0.995 for PolarQuant K4) hides this blow-up — which is exactly why the prior reconstruction-only KV benchmarks (below) couldn't detect it. Keys now use `PerChannelKV`. See [`docs/KV_KEYS_FINDING.md`](../KV_KEYS_FINDING.md).

#### vs KIVI / KVQuant

Per-channel key quantization is the same insight behind [KIVI](https://arxiv.org/abs/2402.02750) and [KVQuant](https://arxiv.org/abs/2401.18079) — v1.2.0 adopts it; it does not claim to beat them. `benchmarks/kv_quant_shootout.py` compares the *quantization schemes* by perplexity at matched bit-width, fake-quantized on Qwen2.5-1.5B (wikitext-2):

| scheme | eff bits | ppl | ΔPPL vs fp16 |
|--------|---------:|----:|-------------:|
| fp16 | 16.0 | 12.2 | — |
| **KIVI** (2-bit; per-channel K / per-token V) | 2.88 | 26.9 | +14.6 |
| **KVQuant** (3-bit; per-channel NUQ + 1% outliers) | 3.29 | 15.4 | +3.1 |
| per-channel keys, uniform + two-tier | 4.73 | 17.5 | +5.3 |
| per-channel keys + NUQ + outliers + two-tier | 4.98 | 14.4 | +2.1 |

KVQuant's non-uniform codebook + dense-sparse outliers is the strongest quality-per-bit lever, and v1.2.0 exposes the same via `PerChannelKV(nuq=True)`; KIVI sits at the 2-bit max-compression corner. **Honest scope:** this small-model (1.5B) shootout uses *portable fake-quant reimplementations of the published schemes — not the authors' CUDA kernels*. For the broader, same-harness picture across four models (Llama-2-7B/13B, Mistral-7B, Qwen2.5-7B) on full LongBench + WikiText-2 — including the **codebook-dependent collapse** that this 1.5B shootout is too small to surface — see [`benchmarks/kvquant_matrix/`](../../benchmarks/kvquant_matrix/) (and the v1.4.0 highlight above). Method + the [`KVQuant`](https://arxiv.org/abs/2401.18079) / [`KIVI`](https://arxiv.org/abs/2402.02750) reproductions: [`docs/KV_KEYS_FINDING.md`](../KV_KEYS_FINDING.md).

**Reconstruction fidelity (head_dim=128) — historical, per-vector metric:**

| Version | Method | Key CosSim | Val CosSim | Avg Bits |
|---------|--------|-----------|-----------|----------|
| v0.5.0 | Uniform K3/V3 | 0.979 | 0.979 | 3.0 |
| v0.9.0 | Asymmetric K4/V3 | 0.995 | 0.979 | 3.5 |
| v0.9.0 | RoPE-aware 4/3 (LLaMA-3) | 0.986 | — | 3.45 |
| v1.0.0 | Learned codebook 3-bit | 0.983 | 0.983 | 3.0 |
| **v1.2.0** | **Per-channel keys + PolarQuant values** | 0.998 | 0.983 | 3.0–4.0 |

**Compression quality on random Gaussian KV (head_dim=256, n_heads=16, fp16 baseline):**

| Bits | Compression Ratio | Cosine Similarity | MSE |
|------|------------------:|------------------:|---------:|
| 2 | 7.5× | 0.926 | 0.001178 |
| 3 | 5.1× | 0.978 | 0.000349 |
| 4 | 3.9× | 0.995 | 0.000082 |

### KV-cache: memory savings

**Auto-config savings (8K context, fp16 baseline):**

| Model | fp16 | Balanced (K4/V3) | Compression (K4/V2) | Extreme (K2/V2) |
|-------|------|-----------------|--------------------|--------------  |
| LLaMA 3 8B | 1.0 GB | 0.23 GB (4.3×) | 0.17 GB (5.8×) | 0.14 GB (7.1×) |
| Gemma 2 27B | 6.0 GB | 1.36 GB (4.4×) | 0.98 GB (6.1×) | 0.80 GB (7.5×) |
| Qwen 2.5 72B | 2.5 GB | 0.59 GB (4.3×) | 0.43 GB (5.8×) | 0.35 GB (7.1×) |

**Gemma 4 27B-A4B at long context** (MoE / multi-query attention keeps the KV cache small; users report ~22 GB model+KV at 240K with IQ4_NL + q8_0 KV):

| Context | fp16 KV | q8_0 KV | TurboQuant K4/V3 | Saved vs q8_0 |
|---------|---------|---------|-----------------|--------------|
| 8K | 0.38 GB | 0.19 GB | 0.09 GB (4.4×) | 0.10 GB |
| 131K | 6.0 GB | 3.0 GB | 1.4 GB (4.4×) | 1.6 GB |
| 240K | 11.0 GB | 5.5 GB | 2.5 GB (4.4×) | 3.0 GB |
| 262K | 12.0 GB | 6.0 GB | 2.7 GB (4.4×) | 3.3 GB |

At 262K, K4/V3 saves **3.3 GB** over q8_0 — headroom for longer context or larger batches on the same GPU.
