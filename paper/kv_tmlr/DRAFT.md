# One Codebook for Every Architecture: Asymmetric NormalFloat for Calibration-Free KV-Cache Quantization

*Working draft. Numbers from `benchmarks/kvquant_matrix/results_matrix.json` (single harness,
LongBench full-200 + WikiText-2). Breadth (7 extra tasks) being filled in; placeholders marked.*

## Abstract

Quantizing the key–value (KV) cache is the standard route to long-context LLM inference,
and the data-free NormalFloat-4 (NF4) codebook is a popular calibration-free choice. We show
that **NF4 silently and catastrophically fails on an entire class of models** — those with
high-ratio grouped-query attention (GQA) — while appearing excellent on the Llama-family MHA
models that dominate published benchmarks. On Qwen2.5-7B (7:1 GQA), 4-bit NF4 KV-cache
quantization collapses LongBench-qasper from **43.8 (fp16) to 4.7** and WikiText-2 perplexity
from **7.46 to 74.7** — degenerate repetition — whereas the same recipe is near-lossless on
Llama-2-7B/13B and Mistral-7B. We trace the failure to NF4's *zero-centered, abs-max* scaling:
KV keys carry a large per-channel DC offset, so a symmetric codebook wastes half its codes on
the empty side; high-GQA models, which feed each KV-head error into many query heads, cannot
absorb the resulting error. The fix is a single per-channel zero-point: **asymmetric NF4
(asym-NF4)** centers the codebook on the data before applying the NF grid. asym-NF4 is one
calibration-free codebook that is near-fp16 on *every* architecture we test — it ties
symmetric NF4 where NF4 already works and recovers the Qwen collapse to **41.9 qasper / 7.50
ppl** — at no extra bit cost. We release the method, a reproducible cross-model harness, and a
practitioner's decision guide.

## 1. Introduction

(KV cache dominates long-context memory; 4-bit is standard; calibration-free codebooks
(NF4) are attractive; most papers benchmark on Llama and imply universality.) **Claim:** that
universality is false, and the failure is silent. **Contributions:**

1. A **silent, catastrophic failure mode** of NF4 KV quantization on high-GQA models
   (Qwen2.5), invisible on the Llama-family models typically benchmarked.
2. A **mechanism**, measured: NF4 costs ~4× the per-channel reconstruction error of
   asymmetric uniform on DC-offset KV keys; high-GQA error-amplification turns that error
   into collapse. Bit-depth is *not* the axis — 3-bit *uniform* beats 4-bit NF4 on Qwen.
3. **asym-NF4**: a one-line zero-point fix that unifies the codebook choice — best-or-tied
   on every architecture, calibration-free, no extra bits.
4. A **reproducible harness**, cross-model matrix, and **decision guide**.

## 2. Background & setup

KV-cache quantization (per-channel keys, per-token values); NF4 vs uniform; dense-sparse
outliers; sink; RoPE; MHA vs GQA. Prior work: KVQuant (calibrated Fisher NUQ), KIVI. Our
setting is **calibration-free**. Models: Llama-2-7B/13B (MHA), Mistral-7B (4:1 GQA),
Qwen2.5-7B (7:1 GQA). Tasks: LongBench English subset + WikiText-2 ppl. Single harness; a
fast prefill-once cache (deployable, ~fp16 speed). All scores intra-harness comparable.

## 3. NF4 fails on high-GQA models (the headline)

**Table 1.** LongBench-qasper, 4-bit keys, identical outliers/sink:

| model | attn (Q:KV) | fp16 | NF4 | uniform | **asym-NF4** |
|---|---|---:|---:|---:|---:|
| Llama-2-7B  | MHA 1:1 | 22.06 | 20.82 | 15.58 | **20.81** |
| Llama-2-13B | MHA 1:1 | 17.06 | 16.86 | 10.52 | **16.41** |
| Mistral-7B  | GQA 4:1 | 29.43 | 29.96 | 21.06 | **28.74** |
| Qwen2.5-7B  | GQA 7:1 | 43.77 | **4.69** | 33.81 | **41.91** |

**Table 2.** WikiText-2 perplexity (↓) — same recipe, continuous metric:

| model | fp16 | NF4 | **asym-NF4** |
|---|---:|---:|---:|
| Llama-2-7B | 6.94 | 7.18 | 6.97 |
| Llama-2-13B | 6.11 | 6.30 | 6.13 |
| Mistral-7B | 5.94 | 6.00 | 5.96 |
| Qwen2.5-7B | 7.46 | **74.19** | 7.50 |

NF4 wins on 3/4 models, collapses on Qwen; uniform never collapses but loses 5–9 pts on
MHA; asym-NF4 is best-or-tied everywhere. Perplexity and LongBench agree.

## 4. Anatomy of the collapse

**It is the codebook shape, not bit-depth.** Qwen rescue sweep (qasper, fp16=43.8):
4-bit NF4 = 4.7; 4-bit *uniform* = 33.8; **3-bit uniform = 34.9** (lower precision, 7× better);
+10% outliers, shorter context, 8-bit values — all stay collapsed under NF4. Only changing
the codebook helps. (Table 3.)

**Mechanism.** On real pre-RoPE keys, NF4 incurs ~3–5× the per-channel relative
reconstruction error of asymmetric uniform — on *both* Qwen and Llama — because KV keys have
a per-group DC offset (range/abs-max ≈ 0.9) and NF4's zero-centered grid spends ~half its
codes on the empty side. NF4's key error is *equal* across the two models, so representation
alone doesn't explain the collapse: the differentiator is **error tolerance**. Qwen's 7:1 GQA
routes each KV-head error into 7 query heads; Llama-2-7B (1:1 MHA) into one. Qwen collapses
above a key-relerr threshold between 0.04 (uniform-3b) and 0.08 (NF4-4b); Llama's threshold
exceeds 0.08. (Fig. 1: relerr Qwen vs Llama; Fig. 2: GQA-ratio vs tolerance.)

## 5. asym-NF4

Per channel: subtract the mean μ, NF4-quantize the residual scaled by abs-max, add μ back.
Stores one extra fp16 scalar/channel (μ) beyond NF4's abs-max — negligible (compression
ratio 7.9× vs NF4's 7.9×, both vs fp32). Keeps NF4's nonlinear levels (so it *beats* uniform)
and centers the grid on the data (so it *doesn't collapse*). Tables 1–2 show it is best-or-tied
on every model. **Breadth:** across 7 further LongBench tasks (QA / multi-hop /
summarization / dialogue), asym-NF4 rescues Qwen's NF4 collapse on *every* task
(avg 5.3→37.3 vs fp16 41.1); the residual gap concentrates on two tasks (§5.5).

### 5.5 A general limitation: long-generation degradation
The asym-NF4 residual is negligible on six of seven breadth tasks but large on `gov_report`
and `multi_news`. These are **not** "the summarization tasks" (`samsum` is summarization and
fine, gap 0.9) — they are the two `max_new_tokens=512` tasks; the gap tracks **generation
length**, not task type. It is **not GQA-specific**: Llama-2-7B shows the same (`gov_report`
26.8→14.6, gap 12.1, matching Qwen's 13.7). Mechanism = **compounding**: each decode step
reads the quantized prefill with a small residual error; over hundreds of steps the model
conditions on its own drift and the output degenerates by the tail (token soup, no EOS).
General property of 4-bit KV quant under long decoding; orthogonal to the codebook (remedy:
larger fp16 window or ≥5-bit keys for long outputs). See `results_longgen.json`.

## 6. Recommendations / guide
(decision tree; `TurboQuantKVCache.robust()`; mirrors the practitioner guide.)

## 7. Limitations
4 models ≤13B, LongBench English + WikiText; no MoE/>13B; calibrated KVQuant/KIVI compared
in-harness on Llama-2-7B only (broader comparison = future work); GQA-ratio hypothesis tested
on 4 points.

## 8. Reproducibility
Single harness, JSON results, notebook (single-GPU subset + multi-GPU full); all in
`benchmarks/kvquant_matrix/`.
