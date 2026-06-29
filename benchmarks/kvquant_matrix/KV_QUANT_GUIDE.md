# A Practitioner's Guide to KV-Cache Quantization Across Model Families

> **TL;DR.** Use **asymmetric (zero-point) NF4** — `TurboQuantKVCache.robust()`. It is a
> single calibration-free codebook that is near-fp16 on *every* architecture we tested. The
> naive choices each fail somewhere: plain (symmetric) **NF4** is great on Llama/Mistral but
> **catastrophically collapses on high-ratio-GQA models like Qwen2.5** (qasper 43.8→4.7);
> asymmetric **uniform** never collapses but loses 5–9 points to NF4 elsewhere. asym-NF4
> keeps NF4's nonlinear levels *and* adds a per-channel zero-point, so it **ties NF4 where
> NF4 wins and recovers the Qwen collapse to near-fp16 (→41.9)**. The rest of this guide is
> the *evidence* for why no single naive codebook can do this — but the recommendation is
> just: **asym-NF4 + 2% outliers + sink.**

All numbers are from a single harness ([`tq_paper_lb_shard.py`](tq_paper_lb_shard.py)),
LongBench (full 200 samples/task, greedy decode), 4-bit keys+values unless noted.
Reproduce with the notebook in this directory. Raw data: [`results_matrix.json`](results_matrix.json),
[`results_rescue.json`](results_rescue.json).

---

## 1. The decision tree

```
Just use asymmetric (zero-point) NF4:

    cache = TurboQuantKVCache.robust(head_dim=..., n_heads=...)
    # == key_nf4_asym=True, key_outlier_frac=0.02, 4-bit K/V

It is near-fp16 on every architecture we tested. You only need the branches below if you
are choosing a NAIVE codebook (no zero-point) and want to know where each one breaks:

├─ MHA (1:1 — Llama-2-7B/13B) or mild GQA (≈4:1 — Mistral-7B)
│     symmetric NF4 is fine (best of the naive options). asym-NF4 ties it.
│
└─ High-ratio GQA (≥7:1 — Qwen2.5-7B, 4 KV heads)
      symmetric NF4 COLLAPSES (43.8 → 4.7 qasper). asymmetric uniform recovers to ~34;
      asym-NF4 recovers to 41.9 (best). Never ship symmetric NF4 here.
```

**One-line rule:** *Symmetric NF4 is zero-centered; it assumes the data straddles zero. KV
keys have a per-channel DC offset, and error-sensitive (high-GQA) models can't tolerate the
error a zero-centered grid wastes on the empty side. asym-NF4 adds a per-channel zero-point
that fixes this with no downside — so it is always the right default.*

---

## 2. The evidence

### 2.1 Codebook choice is model-dependent (the headline)

4-bit keys, identical outliers/sink, **qasper** (LongBench). The first three columns are the
*naive* codebooks (they motivate the problem); **asym-NF4 is the resolution**:

| model | attention (Q:KV) | fp16 | NF4 | uniform | **asym-NF4** |
|---|---|---:|---:|---:|---:|
| Llama-2-7B  | MHA  1:1 | 22.06 | 20.82 | 15.58 | **20.81** |
| Llama-2-13B | MHA  1:1 | 17.06 | 16.86 | 10.52 | **16.41** |
| Mistral-7B  | GQA  4:1 | 29.43 | 29.96 | 21.06 | **28.74** |
| Qwen2.5-7B  | GQA  7:1 | 43.77 | 4.69 💥 | 33.81 | **41.91** |

**The naive codebooks each fail somewhere; asym-NF4 wins everywhere.** Plain NF4 is best on
Llama-2-7B/13B (MHA) *and* Mistral (4:1 GQA) by 5–9 qasper points — its nonlinear levels help
on well-behaved keys — but at **7:1 GQA (Qwen2.5-7B) it collapses** (4.69), a cliff not a
gradient. Uniform never collapses but loses 5–9 points on the MHA models. **asym-NF4 ties NF4
on all three NF4-favoring models** (within ~1 pt) **and recovers Qwen to 41.91** — beating
uniform there by +8 and landing within 1.9 of fp16. One codebook, no failure mode. Full
trec/triviaqa/qasper in `results_matrix.json`.

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

**In the package — the recommended default (works everywhere):**
```python
from turboquant_pro import TurboQuantKVCache
cache = TurboQuantKVCache.robust(head_dim=128, n_heads=32)
# asymmetric NF4 keys + 2% dense-sparse outliers + per-token uniform values, 4-bit, calibration-free
```

**In the research harness** ([`tq_paper_lb_shard.py`](tq_paper_lb_shard.py)) — all calibration-free:
```
# Recommended default (robust across architectures):
CODEBOOK=nf4a KEY_BITS=4 VAL_BITS=4 GROUP=32 SINK=4 OUTLIER_FRAC=0.02 HOT=128

# Naive alternatives (for comparison only):
#   CODEBOOK=nf4      -> best naive on Llama/Mistral, COLLAPSES on Qwen2.5 (do not ship)
#   CODEBOOK=uniform  -> never collapses but loses 5-9 qasper to nf4/nf4a on MHA models
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
