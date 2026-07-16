# Projection sensitivity under quantization is metric- and operator-dependent

*A de-confounded look at "The Illusion of Equivalency" (Rababah, Akcora & Leung,
arXiv:2607.08734), with two new instruments and a result that reverses the
paper's headline ranking once behavior — not weight statistics — is measured.*

Andrew H. Bond · TurboQuant Pro technical note · 2026-07-15 · rev. 2026-07-16
(adds Llama-3.2-3B — both of the paper's model families now covered)

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
  quantizer, weight-cosine and relative-Frobenius drift are within ~13% across
  Q/K/V/O on all four models tested — Qwen2.5-1.5B, Gemma-3-4B, and **both of the
  paper's own models: Mistral-7B-v0.1 and Llama-3.2-3B**. The confound *was* the
  effect.
- **Behaviorally, V/O are the *more* sensitive projections, not less.** At 4-bit —
  the regime where precision-allocation advice operates — perturbing only V or O
  moves the model's output distribution **2.3–3.8× more** than perturbing Q or K,
  on all four models. On the paper's own Mistral-7B the reversal is 3.4×; on the
  paper's own Llama-3.2-3B, 2.8×. The paper's ranking — and its
  precision-allocation advice — **reverse** once you measure behavior instead of
  weight statistics.
- **Below 4 bits, single-projection fragilities appear that neither ranking
  describes** (new with the Llama run): at 3-bit, Llama-3.2-3B's **K alone**
  becomes the most damaging projection (~6× V) while its weight drift stays
  ordinary. The mechanism is measured, not conjectured: the damage lives in the
  **long-wavelength (DC) RoPE rows of `W^K`** — sparing 12.5% of K-rows recovers
  87% of it — and the same coupling exists at 25× smaller amplitude in models
  without extreme rope-scaling (§2.3).

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

The same instrument on the paper's own **Mistral-7B-v0.1** (48 seqs, 12,240
positions; noise floor two 8-bit variants = **0.007 disagreement**) tells the
identical story: at 4-bit, accuracy 0.537→0.495 (−4.3 pts) hides that **23% of
predictions changed** (behavioral agreement 0.768), churn (10.1%) ≫ accuracy delta
(4.3%), excess ~33× the floor (z≈299).

And on the paper's own **Llama-3.2-3B** — the model its Q8 CA-gap table is built
on — the decomposition settles the 8-bit question directly (32 seqs, 8,160
positions; noise floor = **0.012 disagreement**):

| bits | accuracy | CA (paper) | churn (regr / recov) | behavioral agree | excess/floor | z |
|---|---:|---:|---:|---:|---:|---:|
| base | 0.496 | — | — | 1.000 | — | — |
| 8 | **0.496** | 0.493 | 0.007 (**0.0033 / 0.0033**) | 0.978 | +0.010 | 8.6 |
| 4 | 0.429 | 0.396 | 0.134 (0.100 / 0.033) | 0.681 | +0.307 | 254 |
| 3 | 0.054 | 0.045 | 0.460 (0.451 / 0.009) | 0.081 | +0.907 | 752 |

At 8-bit, accuracy is bit-identical, regressions and recoveries are *exactly*
symmetric (0.33% each — zero net damage), and over half the total prediction
disagreement (2.2%) is the measured noise floor (1.2%). CA still reports a "drop"
(0.493 vs 0.496) — that drop is Llama's baseline item brittleness (base accuracy
≈ 0.5 puts half the items near the decision boundary) being billed to
quantization. The paper's 14-point Q8 CA gap on this same model is the same
artifact at task scale (Defect 2). The instrument convicts in the other direction
too: at 4-bit the excess over floor is +30.7% at z=254 with regressions 3×
recoveries — real, directional damage, cleanly separated from free churn. CA can
make neither statement.

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
| Mistral-7B-v0.1 (paper) | 1.13 | 1.12 | 1.09 | 1.01 |
| Llama-3.2-3B (paper) | 1.05 | 1.05 | 1.04 | 1.00 |

Within ~13% of 1.0 everywhere. The paper's weight-space Q/K "sensitivity" is the
bit-allocation confound; remove it and all four projections drift the same amount
in weight space.

**Result 2 — behaviorally, V/O move the output *more* than Q/K.** Functional
impact `mean KL(p_base‖p_quant)`, ratio (Q,K)/(V,O):

| model | b8 | b4 | b3 | b2 |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0.38 | 0.26 | 0.17 | 0.27 |
| Gemma-3-4B | 0.83 | 0.43 | 0.43 | — |
| Mistral-7B-v0.1 (paper) | 0.29 | 0.29 | 0.44 | 1.05 |
| Llama-3.2-3B (paper) | 0.40 | 0.36 | **2.77** | 1.13 |

At 8- and 4-bit every entry is < 1 on all four models: perturbing Q/K moves
behavior **less** than perturbing V/O — at 4-bit by 3.8× on Qwen, 2.3× on Gemma,
3.4× on the paper's Mistral-7B, and 2.8× on the paper's Llama-3.2-3B.
Per-projection top-1 flip rate at 4-bit: Qwen2.5-1.5B Q 0.050, K 0.067, V 0.100,
**O 0.127** (ordering **O > V > K > Q**); Mistral-7B Q 0.034, K 0.036, **V 0.075**,
O 0.048 (**V > O > K > Q**); Llama-3.2-3B Q 0.048, K 0.050, **V 0.090, O 0.091**
(**O ≈ V > K > Q**). All put V and O above Q and K — the reverse of the paper's
ranking, and consistent with the broader PTQ literature that value/output
projections are among the most sensitive. At the destructive b2 (74–92% of tokens
flip; the models are broken) the ratios drift toward parity — i.e. the paper's
ordering approaches holding only where the model is already gone. The one
above-1 entry in a usable regime — Llama at b3 — is not the paper's ordering
returning either; it is a single-projection anomaly worth its own section.

**The paper's advice — "compress V and O more aggressively" — is therefore
backwards for weight PTQ at the bit-widths where the advice would be applied.**
At matched 4-bit, V/O are exactly the projections whose error the model tolerates
*least*, on every architecture tested including both of the paper's own.

Full tables: [`experiments/results_matched_bit/`](../../experiments/results_matched_bit/).

### 2.3 Below 4 bits: single-projection fragilities, not a ranking

The Llama-3.2-3B run adds a boundary the three-model version of this note could
not see. At 3-bit, Llama's **K projection alone** explodes: out_kl **0.741**
against Q 0.089, V 0.125, O 0.174 — K is ~6× more damaging than V — while K's
weight drift stays ordinary (rel-Fro 0.433 vs ~0.40 for the others; QK/VO weight
ratio 1.04). The effect is functional, not weight-statistical: K's
amplification (output movement per unit weight drift) hits **1.71** where every
other projection sits at 0.2–0.4. No other model does this at b3 (Qwen 0.17,
Gemma 0.43, Mistral 0.44) — though Qwen shows a milder cousin in its **O**
projection at b3 (out_kl 0.62 vs V 0.18).

Two readings, both consistent with this note's thesis. First, the practical one:
**below 4 bits there is no stable projection ranking at all** — neither the
paper's "protect Q/K" nor this note's 4-bit "protect V/O" survives; fragility
becomes architecture- and projection-specific, and only a behavioral probe finds
it. Second, a mechanistic hypothesis, which we pre-registered and then tested:
Llama-3.2 carries the most extreme low-frequency RoPE structure of the four
models (128k context, rope-scaling factor 32), and `W^K` rows are where the
RoPE-frequency-structured outlier channels live
([`RESULTS_rope_offsets.md`](../../benchmarks/RESULTS_rope_offsets.md) measured
exactly this structure in the *activation* domain). **Prediction as registered:**
per-row damage in Llama's `W^K` at 3-bit rises with the row's rotary wavelength,
and is comparatively flat on a no-anomaly control (Mistral-7B).

**Measured** (`experiments/k_wavelength_probe.py`: rows of every `k_proj`
grouped into 8 wavelength octiles from the model's own `inv_freq`; per octile,
quantize *only* those rows / quantize *all but* those rows at 3-bit):

| | Llama-3.2-3B | Mistral-7B-v0.1 (control) |
|---|---:|---:|
| full-K out_kl @3-bit | 0.741 | 0.030 |
| Spearman(octile-only damage, wavelength) | **+0.905** | +0.571 |
| damage share, 2 longest-λ octiles | 72% | 79% |
| sparing the longest octile alone (12.5% of rows) | 0.741 → **0.099** (−87%) | 0.030 → 0.015 (−49%) |
| longest wavelengths | 1.9–8.2 ×10⁷ tok | 2.0–5.4 ×10⁴ tok |

The prediction was **half right, and the miss is the finding**. The
wavelength–damage coupling is not Llama-specific — Mistral's residual K damage
concentrates in its longest-wavelength rows too. The coupling is *universal*;
what is Llama-specific is the **amplitude**: rope-scaling stretches 45 of 64
frequencies beyond any practical window (pure DC — these rows carry the
per-channel offsets that softmax reads as its absolute reference), and at 3-bit
the row-absmax quantizer's error in exactly those rows crosses from nuisance
(Mistral, 0.03) to model-breaking (Llama, 0.74). The b3 anomaly is the keys
finding surfacing in weight space.

Three sharp corollaries. (1) **The fix is surgical, not global**: keeping the
single longest-wavelength octile in higher precision — 12.5% of `W^K` rows,
~0.4% of attention weights — recovers 87% of the damage; "give K more bits
everywhere" is the wrong shape. (2) **Weight statistics are blind to it, twice
over**: per-row relative error correlates with wavelength at only +0.12 on
Llama, and the control correlates *higher* (+0.22) while suffering 25× less —
the fragile rows are not quantized worse, they feed an operator that tolerates
less. (3) The effect is **superadditive** (octile-only damages sum to 0.17
against 0.74 jointly, and sparing a mid-band octile can *worsen* the total via
cross-band error cancellation) — so the spared-probe numbers, not the
quantize-only numbers, are the deployment-relevant ones.

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
3. *(shipped and validated)* The §2.3 mechanism is wired into the weight path:
   `ModelCompressor.quantize_weights(bits, rope_aware_k=True)` quantizes FFN +
   attention weights per output channel while keeping the long-wavelength RoPE
   rows of `W^K` in full precision (or at `k_protect_bits`), with the protected
   set read from the model's own `inv_freq` buffer. Default `k_protect_frac`
   is 0.125 — the measured 87%-recovery octile. End-to-end validation through
   the shipped path (`experiments/validate_rope_aware_k.py`, NRP L40S —
   different silicon than the GV100 probe): K-only 3-bit out_kl **0.0994
   protected vs 0.7352 unprotected**, an 86.5% recovery against the
   pre-registered ~87%, reproducing the probe to the third decimal.
   Two additional findings: `k_protect_bits=8` is indistinguishable from
   full-precision protection (the fix costs ~0.16 bits/weight averaged over
   the four projections); and at whole-model 4-bit the option buys only ~2.5%
   — consistent with §2.2 (V/O dominate at 4-bit), so `rope_aware_k` earns
   its keep at ≤3-bit K, and that is its claim.

---

## 5. Honest scope

Fake-quantization (quantize→dequantize weights, fp inference); a clean
per-output-channel symmetric absmax control (not any vendor kernel); four models
(Qwen2.5-1.5B, Gemma-3-4B, and **both of the paper's own models:
Mistral-7B-v0.1 and Llama-3.2-3B**), next-token prediction on a WikiText-2
sample, single GPU. The claims are **relative and directional** (Q/K vs V/O at
matched bits), not absolute degradation numbers; the 4-bit reversal is
consistent across all four architectures, and the sub-4-bit regime is claimed
only as *unstable* (§2.3), not as any fixed ordering. Covering both of the
paper's model families closes the most important gap — the reversal is not an
artifact of a different model family. Llama-3.2-3B provenance: run from the
`unsloth/Llama-3.2-3B` mirror of the gated `meta-llama` repo (byte-identical
weights; same route as the Gemma runs) in fp16 on Volta (bf16-native model —
base logprobs verified finite; the clean 8-bit row rules out overflow
artifacts). The §2.3 K-wavelength mechanism is measured on one subject and one
control model at one bit-width (3) and one window (256); its pre-registered
prediction was half wrong (the control's profile is not flat — the coupling is
universal, the amplitude is not), which is reported as found. Remaining
confirmations: the downstream MCQ tasks the paper uses, and the authors' own
llama.cpp pipeline forced to equal bit-width.

### Reproduce

```bash
# de-confounded projection sensitivity
NB_MODEL=Qwen/Qwen2.5-1.5B-Instruct NB_BITS=8,4,3,2 NB_SEQ=256 NB_SAMPLES=24 \
    python experiments/matched_bit_projection_sensitivity.py
NB_MODEL=unsloth/gemma-3-4b-it NB_DTYPE=bfloat16 \
    python experiments/matched_bit_projection_sensitivity.py
NB_MODEL=mistralai/Mistral-7B-v0.1 NB_BITS=8,4,3,2 NB_SEQ=256 NB_SAMPLES=32 \
    python experiments/matched_bit_projection_sensitivity.py   # paper's model 1
NB_MODEL=unsloth/Llama-3.2-3B NB_BITS=8,4,3,2 NB_SEQ=256 NB_SAMPLES=32 \
    python experiments/matched_bit_projection_sensitivity.py   # paper's model 2
                       # (ungated mirror of meta-llama/Llama-3.2-3B)

# K-row damage vs rotary wavelength (the section-2.3 mechanism)
NB_MODEL=unsloth/Llama-3.2-3B NB_BITS=3 NB_SEQ=256 NB_SAMPLES=32 \
    python experiments/k_wavelength_probe.py
NB_MODEL=mistralai/Mistral-7B-v0.1 NB_BITS=3 NB_SEQ=256 NB_SAMPLES=32 \
    python experiments/k_wavelength_probe.py   # control

# behavioral metric demo (CA vs flip vs agreement vs noise floor)
NB_MODEL=Qwen/Qwen2.5-1.5B-Instruct NB_BITS=8,4,3 \
    python experiments/behavioral_metric_demo.py
NB_MODEL=unsloth/Llama-3.2-3B NB_BITS=8,4,3 NB_SEQ=256 NB_SAMPLES=32 \
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
