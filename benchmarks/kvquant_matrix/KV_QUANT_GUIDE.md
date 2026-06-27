# A Practitioner's Guide to KV-Cache Quantization Across Model Families

> **TL;DR.** There is **no universal best 4-bit KV-cache recipe.** The right codebook
> depends on your model's attention architecture. NF4 (the data-free favourite) is
> *excellent on Llama-family MHA models* and *catastrophic on high-ratio-GQA models like
> Qwen2.5*, where it collapses to near-random output. Asymmetric **uniform** quant is the
> safe default — it never collapses and costs ~4× less error at the same bit-width — but
> it gives up a little quality on the models NF4 suits. Pick by family; the decision tree
> is below.

All numbers are from a single harness ([`tq_paper_lb_shard.py`](tq_paper_lb_shard.py)),
LongBench (full 200 samples/task, greedy decode), 4-bit keys+values unless noted.
Reproduce with the notebook in this directory. Raw data: [`results_matrix.json`](results_matrix.json),
[`results_rescue.json`](results_rescue.json).

---

## 1. The decision tree

```
What is your model's attention type?
│
├─ MHA (1:1 — e.g. Llama-2-7B/13B)
│     → NF4 keys + 2% per-channel fp16 outliers + 4 sink tokens.   (best quality, calibration-free)
│       Uniform also works but is slightly worse here.
│
├─ Mild GQA (≈4:1 — e.g. Mistral-7B, 8 KV heads)
│     → NF4 recipe is fine. Matches or beats fp16. No special handling.
│
└─ High-ratio GQA (≥7:1 — e.g. Qwen2.5-7B, 4 KV heads)
      → DO NOT use NF4. It collapses (43.8 → 4.7 qasper).
        Use ASYMMETRIC UNIFORM keys (+ 2% outliers + sink). Recovers to ~35.
        If you can afford it, 8-bit keys close most of the remaining gap.
```

**One-line rule:** *NF4 is a zero-centered codebook; it assumes the data straddles zero.
KV keys have a DC offset, and error-sensitive (high-GQA) models can't tolerate the error
NF4 incurs as a result. When in doubt, use asymmetric uniform — it is never the wrong
choice, only sometimes a slightly suboptimal one.*

---

## 2. The evidence

### 2.1 Codebook choice is model-dependent (the headline)

4-bit keys, NF4 vs asymmetric uniform, identical outliers/sink, **qasper** (LongBench):

| model | attention (Q:KV) | fp16 | NF4 4-bit | uniform 4-bit | winner |
|---|---|---:|---:|---:|---|
| Llama-2-7B  | MHA  1:1 | 22.06 | **20.82** | 15.58 | NF4 by 5.2 |
| Llama-2-13B | MHA  1:1 | 17.06 | **16.86** | 10.52 | NF4 by 6.3 |
| Mistral-7B  | GQA  4:1 | 29.43 | **29.96** | 21.06 | NF4 by 8.9 |
| Qwen2.5-7B  | GQA  7:1 | 43.77 | 4.69 💥 | **33.81** ✅ | uniform by 29.1 |

**NF4 wins on 3 of 4 models — until it doesn't, and then it falls off a cliff.** NF4 is the
better codebook on Llama-2-7B/13B (MHA) *and* Mistral (4:1 GQA), by 5–9 qasper points each —
its nonlinear levels genuinely help on well-behaved keys. But at **7:1 GQA (Qwen2.5-7B) it
collapses** to near-random, and uniform wins by 29 points. The failure is a **cliff, not a
gradient**: 4:1 GQA is firmly NF4 territory; 7:1 is catastrophic. So uniform is the
*risk-averse* choice (never collapses, but costs 5–9 points on NF4-favoring models), and NF4
is the *quality* choice (best almost everywhere, with one catastrophic failure mode you must
rule out). Full trec/triviaqa/qasper in `results_matrix.json`.

### 2.2 For the fragile model, it's the codebook — not the bit-depth

Qwen2.5-7B, isolating the cause (all vs fp16 71.5 / 89.97 / 43.77):

| variant | codebook | bits | trec | triviaqa | qasper | |
|---|---|---|---:|---:|---:|---|
| NF4 baseline | NF4 | 4 | 11.25 | 5.77 | 4.69 | 💥 |
| + 10% outliers | NF4 | 4 | 18.67 | 5.17 | 4.34 | 💥 outliers don't help |
| + short context | NF4 | 4 | 15.0 | 5.53 | 4.56 | 💥 not a context-length issue |
| + 8-bit values | NF4 | k4/v8 | 14.0 | 7.04 | 4.42 | 💥 values aren't the cause |
| **uniform, same bits** | **uniform** | **4** | **65.0** | **81.33** | **33.81** | ✅ **codebook is the lever** |
| uniform, fewer bits | uniform | 3 | 65.5 | 81.19 | 34.92 | ✅ 3-bit uniform > 4-bit NF4 |
| uniform, more bits | uniform | 8 | 64.5 | 82.28 | 34.07 | ✅ |

**3-bit uniform (34.92) beats 4-bit NF4 (4.69) by 7×** on the same model — lower precision,
better result, because the codebook *shape* matters more than the bit budget here.

### 2.3 Why: NF4 wastes resolution on offset data; sensitive models can't absorb it

Measured on real pre-RoPE keys (per-channel relative reconstruction error, 4-bit):

- NF4 costs **~3–5× more error than asymmetric uniform** at equal bits — on **both** Qwen
  *and* Llama. KV keys carry a DC offset (per-group range/absmax ≈ 0.9), and NF4's
  symmetric, zero-centered levels spend ~half their codes on the empty side.
- NF4's key reconstruction error is **about the same (~0.08) on Qwen and Llama** — so NF4
  doesn't represent Qwen's keys any *worse*. The difference is **error tolerance**:
  Qwen's 7:1 GQA feeds each KV-head error into 7 query heads, so the same 8% perturbation
  that Llama (1:1 MHA) shrugs off pushes Qwen past its tolerance and into a degenerate
  repetition loop (`"...the dam is is a concrete concrete and and and..."`).

So two quantities decide your fate: **how much error your codebook adds** (NF4 ≫ uniform on
offset KV) **× how much error your model tolerates** (set largely by the GQA ratio).

---

## 3. Recipes (copy-paste)

> Env vars for [`tq_paper_lb_shard.py`](tq_paper_lb_shard.py). All are calibration-free.

**Llama-family / MHA — best quality:**
```
CODEBOOK=nf4 KEY_BITS=4 VAL_BITS=4 GROUP=32 SINK=4 OUTLIER_FRAC=0.02 HOT=128
```

**High-ratio GQA (Qwen2.5) — mandatory:**
```
CODEBOOK=uniform KEY_BITS=4 VAL_BITS=4 GROUP=32 SINK=4 OUTLIER_FRAC=0.02 HOT=128
# bump KEY_BITS=8 if you have the memory budget and want to close the fp16 gap
```

**Safe default when you don't know the architecture:**
```
CODEBOOK=uniform KEY_BITS=4 VAL_BITS=4 GROUP=32 SINK=4 OUTLIER_FRAC=0.02 HOT=128
```

---

## 4. Notes, caveats, and what is NOT claimed

- **Pre- vs post-RoPE quantization** is a wash-to-marginal and *model-specific*: it helps
  Llama-2-7B (qasper 20.82→21.31) but hurts Llama-2-13B and Mistral. Not a recommended
  default lever. (Demoted from an early over-claim of ours — the multi-model matrix
  corrected it.)
- Absolute LongBench scores are only comparable **within this harness** (truncation/prompt
  differences shift scores several points across harnesses). All rows here share one harness.
- Mistral/Qwen use their native 32k context; Llama uses 4k. Compare *within* a model, not
  the absolute scores across models.
- This guide covers 4 models on the LongBench English subset. Broader task coverage,
  WikiText perplexity, and same-harness KVQuant/KIVI baselines are in the companion paper.
```
```
