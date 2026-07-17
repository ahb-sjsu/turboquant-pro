# Model card — attention keys (`SOFTMAX_SCORE`)

**Regime:** attention keys feed the score `softmax(Q·Kᵀ/√d)`. Selection-like
sensitivity: per-channel scale and small correlated perturbations matter, and
softmax + GQA amplify them.

**Discipline (what to do):** quantize keys **per-channel with a zero-point
(asymmetric NF4)** — `TurboQuantKVCache.robust()`. Never per-vector direction
quantization (PolarQuant) for keys, and never symmetric NF4 on high-ratio-GQA
models.

**Consumer metric:** perplexity + LongBench task score. Reconstruction cosine /
per-layer error are **blind or anti-correlated** here — the whole point.

---

## Models validated (real weights)

| Model | Attention (Q:KV) | Source |
|---|---|---|
| Llama-2-7B-chat | MHA 1:1 | `benchmarks/kvquant_matrix/results_matrix.json` |
| Llama-2-13B-chat | MHA 1:1 | same |
| Mistral-7B | GQA 4:1 | same |
| Qwen2.5-7B | GQA 7:1 | same |
| Qwen2.5-1.5B-Instruct | GQA 6:1 | `docs/KV_KEYS_FINDING.md` |

Harness: `benchmarks/kvquant_matrix/tq_paper_lb_shard.py` (LongBench, full 200
samples/task, greedy) and the post-RoPE fake-quant perplexity harness in
`docs/KV_KEYS_FINDING.md`.

## Headline result — codebook choice is model-dependent; asym-NF4 wins everywhere

4-bit keys, LongBench **qasper**, identical outliers/sink. First three columns are
*naive* codebooks (they motivate the problem); asym-NF4 is the resolution:

| model | Q:KV | fp16 | NF4 | uniform | **asym-NF4** |
|---|---|---:|---:|---:|---:|
| Llama-2-7B | 1:1 | 22.06 | 20.82 | 15.58 | **20.81** |
| Llama-2-13B | 1:1 | 17.06 | 16.86 | 10.52 | **16.41** |
| Mistral-7B | 4:1 | 29.43 | 29.96 | 21.06 | **28.74** |
| Qwen2.5-7B | 7:1 | 43.77 | **4.69 💥** | 33.81 | **41.91** |

WikiText-2 perplexity corroborates (`results_matrix.json`): Qwen2.5-7B fp16
**7.457** → symmetric NF4 **74.66** (10× collapse) → asym-NF4 **7.499**
(near-lossless); Llama-2-7B 6.942 / 7.177 / 6.970; Mistral-7B 5.942 / 6.000 / 5.955.

## Negative cases (preserved)

1. **PolarQuant destroys keys while passing reconstruction.** Qwen2.5-1.5B, post-RoPE
   keys, fp16 ppl **12.24**:

   | key quantizer | bits | ppl | Δppl | key recon (mean/max) |
   |---|---:|---:|---:|---:|
   | PolarQuant (shipped values quantizer) | K4 | **10643** | +10631 | 0.095 / 0.119 |
   | per-channel uniform | K4 | 14.91 | +2.67 | 0.062 / 0.080 |
   | per-channel NUQ | K3 | 15.77 | +3.54 | 0.148 / 0.190 |

   PolarQuant has the **best** key reconstruction (0.095) and the **worst**
   perplexity — reconstruction error is *anti-correlated* with quality for keys.
   (`docs/KV_KEYS_FINDING.md`.)

2. **Symmetric NF4 collapses on high-ratio GQA.** Qwen2.5-7B qasper 43.77 → **4.69**
   at 4-bit; **3-bit uniform (34.92) beats 4-bit NF4 (4.69) by 7×** — the codebook
   *shape* matters more than the bit budget. Outliers, shorter context, and 8-bit
   values each fail to rescue it; only changing the codebook does
   (`benchmarks/kvquant_matrix/KV_QUANT_GUIDE.md`).

## Why

Keys carry a per-channel DC offset (per-group range/absmax ≈ 0.9). A zero-centered
grid (symmetric NF4) wastes resolution on the empty side of zero, and
error-sensitive high-GQA models — where a handful of KV heads serve many query
heads — cannot absorb it. A per-channel zero-point (asym-NF4) spends every level
on the occupied range. Per-vector normalization (PolarQuant) discards exactly the
per-channel scale the dot product needs.

## Reproduce / instrument

- Family choice is checkable at **calibration time**, no generation run:
  `tqp probe --consumer attention_logits` (`turboquant_pro.a2_probe`,
  `recommend_key_quantizer`) reproduces the PolarQuant-collapse ranking as a unit
  test.
- Claim ledger: `kv_keys_per_channel` in `claims.yaml`.
- Raw data: `benchmarks/kvquant_matrix/results_matrix.json`,
  `results_rescue.json`; `benchmarks/RESULTS_longbench.md`.

**Scope/caveat:** the KVQuant/KIVI comparisons use portable fake-quant
reimplementations of the published schemes, not the authors' CUDA kernels; the
asym-NF4 recommendation is the robust default across the families tested, not a
claim of SOTA over every vendor kernel.
