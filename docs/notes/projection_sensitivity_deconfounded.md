# Projection sensitivity under quantization is metric- and operator-dependent

*A de-confounded look at "The Illusion of Equivalency" (Rababah, Akcora & Leung,
arXiv:2607.08734), with two new instruments and a result that reverses the
paper's headline ranking once behavior — not weight statistics — is measured.*

Andrew H. Bond · TurboQuant Pro technical note · 2026-07-15

---

## TL;DR

Rababah et al. make two claims: (1) accuracy and perplexity hide behavioral
change under quantization — they introduce **Correctness Agreement** (CA) to
expose it; and (2) the **query/key projections are more sensitive** than
value/output, so future methods should "allocate higher precision to K and Q
while compressing V and O more aggressively."

We agree with (1) and sharpen it. We show (2) does not survive scrutiny:

- **Their Q/K sensitivity is confounded.** It is measured under llama.cpp
  K-quantization, whose schemes *give V and O more bits than Q/K* (their own
  appendix: `Q2_K` quantizes Q,K at 2-bit but V at 4-bit, O at 3-bit). "Q/K
  drift more" is entangled with "Q/K got fewer bits."
- **At matched bits, weight-space drift is equal across projections.** Quantizing
  all four projections with the *same* uniform per-output-channel *b*-bit
  quantizer, weight-cosine and relative-Frobenius drift are within ~5% across
  Q/K/V/O on both Qwen2.5-1.5B and Gemma-3-4B. The confound *was* the effect.
- **Behaviorally, V/O are the *more* sensitive projections, not less.** Perturbing
  only V or O moves the model's output distribution **2.4–6× more** than
  perturbing Q or K at the same bit-width, across both models and every bit level
  8→2. The paper's ranking — and its precision-allocation advice — **reverse**
  once you measure behavior instead of weight statistics.

The reconciliation is the interesting part: **projection sensitivity is
operator-dependent.** TurboQuant Pro's own finding — that quantizing the *KV-cache
keys* is catastrophic while values are cheap — is about **activation**
quantization on the score path, where `softmax(Q·Kᵀ)` reads per-channel key scale
([`KV_KEYS_FINDING.md`](../KV_KEYS_FINDING.md), condition (A2)). The paper is about
**weight** quantization, where the direct residual-writing projections (V/O)
dominate behavior. Different operator, different sensitive side. The one invariant
across both: **weight-distribution and reconstruction statistics — skewness,
kurtosis, cosine, exactly the paper's tools — do not predict behavioral impact.**
That is TurboQuant's central, repeated warning, and this note is another instance
of it.

Two artifacts ship with this note:

- `turboquant_pro.behavioral_agreement` — a corrected behavioral metric (symmetric
  flip rate + prediction-level agreement + a **noise floor**), fixing two defects
  in CA.
- `experiments/matched_bit_projection_sensitivity.py` — the de-confounding
  experiment above.

---

## 1. Where we agree, and how to sharpen it

The paper's real contribution is the reframing: *aggregate metrics can be
preserved while individual decisions churn.* This is correct and important, and it
echoes TurboQuant's own posture ("evaluate on the metric that matters; cosine and
reconstruction mislead"). But Correctness Agreement, as defined, has two defects.

**Defect 1 — CA only counts joint-correct.** `CA = mean(base✓ ∧ quant✓)`, bounded
above by `min(acc_base, acc_quant)`. It is blind to base-wrong → quant-right
recoveries and it conflates *accuracy loss* with *answer churn*: a model that
uniformly drops accuracy and a model that swaps *which* items it gets right can
report the same CA.

**Defect 2 — no noise floor.** Near-threshold multiple-choice items flip under
*any* perturbation. In the paper's own Table (macro-avg accuracy/CA over
HellaSwag/Winogrande/ARC), CA for Llama-3.2-3B is **41.4 at near-lossless Q8_0**
against a base accuracy of 55.5 — a **14-point gap already at 8-bit** — and it is
essentially flat from Q8 (41.4) through Q4_K (40.9), only dropping at Q2_K (38.5).
Most of the "moderate quantization" CA gap is item brittleness, not quantization.
Without a control that measures the *free* flip rate between two behaviorally
equivalent runs, a raw flip count cannot be attributed to quantization.

### The corrected metric (`behavioral_agreement.py`)

- `flip_rate` — the symmetric McNemar split: **regressions** (base✓→quant✗) *and*
  **recoveries** (base✗→quant✓); their sum (`churn`) is the movement accuracy
  nets to zero.
- `behavioral_agreement` — prediction-level: do the two models return the *same
  answer*, right or wrong? The label-independent drift signal.
- `noise_floor` — disagreement between two near-lossless (e.g. two 8-bit)
  requantizations; quantization drift is then reported as **excess over floor**
  with a z-score.

**Demonstration** (Qwen2.5-1.5B, next-token prediction on WikiText-2, 6,120
positions; all weights quantized per-output-channel; noise floor = two 8-bit
variants):

| bits | accuracy | CA (paper) | churn | behavioral agree | disagreement | excess/floor | z |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 0.462 | — | — | 1.000 | 0.000 | — | — |
| 8 | 0.462 | 0.459 | 0.007 | 0.978 | 0.022 | +0.008 | 5.1 |
| 4 | 0.367 | 0.331 | 0.166 | **0.591** | 0.409 | +0.395 | 262 |
| 3 | 0.002 | 0.002 | 0.461 | 0.004 | 0.996 | +0.982 | 653 |

Noise floor (two 8-bit variants) = **0.014 disagreement**. Read the 4-bit row: a
~10-point accuracy drop hides that **41% of token predictions changed**
(behavioral agreement 0.591), churn (16.6%) ≫ accuracy delta (9.5%), and the
excess over the noise floor is ~28σ. This is the illusion of equivalency made
quantitative — and separated from free churn, which CA cannot do.

---

## 2. The "Q/K most sensitive" claim does not survive de-confounding

### 2.1 The confound

The paper measures projection sensitivity as the weight-distribution drift
(skewness, kurtosis, cosine, KL of weight entries) of `W^Q, W^K, W^V, W^O` under
llama.cpp legacy and K-quantization. But K-quant is **not** equal-bit across
projections. From the paper's own appendix (and llama.cpp):

| scheme | K, Q | V | O |
|---|---|---|---|
| `Q4_K`, `Q5_K` | assigned bits | **Q6_K** | assigned |
| `Q3_K` | assigned | **Q5_K** | **Q4_K** |
| `Q2_K` | 2-bit | **Q4_K** | **Q3_K** |

Under `Q2_K`, Q and K get 2 bits while V gets 4 and O gets 3. "Q/K drift more" is
inseparable from "Q/K were quantized harder." (Under *legacy*, equal-bit quant,
the paper itself notes the projections stay near-identical.)

### 2.2 The de-confounded experiment

`experiments/matched_bit_projection_sensitivity.py` quantizes all four projections
with the **same** uniform per-output-channel symmetric absmax quantizer at each
bit-width, and measures drift two ways: the paper's **weight-space** lens
(cosine, KL, |Δkurtosis|, relative Frobenius) and a **functional** lens — perturb
*only* projection L in every layer and measure the movement of the model's output
distribution, `mean KL(p_base‖p_quant)` and the top-1 token-flip rate.

**Result 1 — at matched bits, weight-space drift is equal across projections.**
Relative-Frobenius drift, ratio (Q,K averaged)/(V,O averaged):

| model | b8 | b4 | b3 | b2 |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0.94 | 0.94 | 0.95 | 1.00 |
| Gemma-3-4B | 1.03 | 1.04 | 1.03 | — |

≈1.0 everywhere. The paper's weight-space Q/K "sensitivity" is the bit-allocation
confound; remove it and all four projections drift the same.

**Result 2 — behaviorally, V/O move the output *more* than Q/K.** Functional
impact `mean KL(p_base‖p_quant)`, ratio (Q,K)/(V,O):

| model | b8 | b4 | b3 | b2 |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0.38 | 0.26 | 0.17 | 0.27 |
| Gemma-3-4B | 0.83 | 0.43 | 0.43 | — |

Every entry < 1: perturbing Q/K moves behavior **less** than perturbing V/O — by
2.4–6× on Qwen and ~2.3× on Gemma. Per-projection top-1 flip rate at 4-bit
(Qwen2.5-1.5B): Q 0.050, K 0.067, V 0.100, **O 0.127** — the ordering is
**O > V > K > Q**, the reverse of the paper's, and consistent with the broader PTQ
literature that output/down projections are among the most sensitive.

**The paper's advice — "compress V and O more aggressively" — is therefore
backwards for weight PTQ.** At matched bits V/O are exactly the projections whose
error the model tolerates *least*.

Full tables: [`experiments/results_matched_bit/`](../../experiments/results_matched_bit/).

---

## 3. Reconciling with TurboQuant's KV-keys finding

This looks, at first glance, to contradict TurboQuant's own headline that KV-cache
**keys** are the fragile side. It does not — the two results are about different
operators:

- **KV-cache (activation) quantization** corrupts the *cached K activations* that
  enter `softmax(Q·Kᵀ)` directly, per token, carrying a large per-channel DC
  offset the logits read. Per-vector polar quantization deletes exactly that scale
  → perplexity 10⁴ at 0.995 key cosine. Keys are fragile; values, which attention
  averages, are cheap. This is the score path failing condition (A2)
  ([`KV_KEYS_FINDING.md`](../KV_KEYS_FINDING.md)).
- **Weight quantization** (this note, and the paper) perturbs `W^K`, changing
  `K = xW_Kᵀ` *consistently for all tokens*; the model recomputes softmax with the
  perturbed keys, and attention — a relative comparison — absorbs much of a shared
  key-space perturbation. Meanwhile V/O sit on the direct residual-writing path, so
  their weight error passes to the output almost linearly.

Same architectural fact (softmax mediates Q/K; V/O flow linearly) produces
*opposite* sensitivity orderings depending on whether you quantize **activations
on the score path** (keys fragile) or **weights** (V/O fragile). The lesson is not
"Q/K are sensitive" or "V/O are sensitive" in the abstract — it is:

> **The sensitive component depends on the operator, and weight-distribution /
> reconstruction statistics do not reveal it. Only a behavioral (or declared-
> consumer-metric) measurement does.**

That is precisely the (A2) discipline TurboQuant already ships as
`a2_probe.recommend_key_quantizer` and the `QualityMonitor` tangential-fraction
stream — pick the family against the *declared consumer metric*, never against
cosine.

---

## 4. What each side can take from this

**For the paper's authors:**
1. Adopt a decision-level metric *with a noise-floor control*; report excess over
   floor, and prefer a symmetric flip / prediction-agreement metric to joint-
   correctness (CA is bounded by `min(acc)` and blind to recoveries).
2. Do not infer projection sensitivity from weight-distribution statistics; at
   matched bits those are equal across projections, and the functional ranking is
   the opposite. Re-run the sensitivity claim at equal bit-width with a behavioral
   readout before recommending precision allocation.
3. The CA table's flatness from Q8→Q4 and the 14-point gap at lossless Q8 are worth
   confronting directly — most of the "moderate-quantization" signal there is item
   brittleness.

**For TurboQuant Pro:**
1. `behavioral_agreement` is now the top rung of the KV evidence ladder (cosine →
   perplexity → LongBench → **answer agreement + noise floor**). Wire it into the
   KV harness to test whether the 4-bit "balanced" default gives the *same
   answers*, not just similar perplexity — quantifying the honest long-decode
   caveat already noted in the README.
2. Extend `ModelCompressor`'s per-head sensitivity with a per-projection
   behavioral probe; the matched-bit result argues for protecting **V/O** weights
   (not Q/K) under weight PTQ — inverse of the KV-cache key rule, exactly because
   it is the inverse operator.

---

## 5. Honest scope

Fake-quantization (quantize→dequantize weights, fp inference); a clean
per-output-channel symmetric absmax control (not any vendor kernel); two models
(Qwen2.5-1.5B, Gemma-3-4B), next-token prediction on a WikiText-2 sample, single
GPU. The claims are **relative and directional** (Q/K vs V/O at matched bits), not
absolute degradation numbers, and are consistent across the two architectures and
all bit-widths tested. Natural next confirmations: one of the paper's exact models
(Llama-3.2-3B, Mistral-7B), the downstream MCQ tasks the paper uses, and the
authors' own llama.cpp pipeline forced to equal bit-width.

### Reproduce

```bash
# de-confounded projection sensitivity
NB_MODEL=Qwen/Qwen2.5-1.5B-Instruct NB_BITS=8,4,3,2 NB_SEQ=256 NB_SAMPLES=24 \
    python experiments/matched_bit_projection_sensitivity.py
NB_MODEL=unsloth/gemma-3-4b-it NB_DTYPE=bfloat16 \
    python experiments/matched_bit_projection_sensitivity.py

# behavioral metric demo (CA vs flip vs agreement vs noise floor)
NB_MODEL=Qwen/Qwen2.5-1.5B-Instruct NB_BITS=8,4,3 \
    python experiments/behavioral_metric_demo.py

pytest tests/test_behavioral_agreement.py -q
```

## References

- B. Rababah, C. G. Akcora, C. K. Leung. *The Illusion of Equivalency:
  Statistical Characterization of Quantization Effects in LLMs.* arXiv:2607.08734.
- A. H. Bond. *Keep the Angle: A Universal Geometry-Preserving Basis in Spectral
  Embeddings* (the-angular-observer) — condition (A2), the angular-transfer scope
  boundary invoked by `a2_probe`.
- TurboQuant Pro, [`docs/KV_KEYS_FINDING.md`](../KV_KEYS_FINDING.md) — per-channel
  keys, and why reconstruction error is anti-correlated with generation quality.
