# Model card — MoE routing (`GATE_SELECTION`)

**Regime:** a top-k mixture-of-experts router reads only the *order* of the gate
logits `l = W_gate·x`. Selection is invariant to a common-mode shift (add a
constant to every expert → same top-k), so common-mode quantization error is free;
only the **differential** error across experts flips the selection, and only where
it exceeds the per-token **routing margin** (the gap between the k-th and
(k+1)-th logit). Prediction: **low-margin tokens flip first.**

**Discipline (what to do):** keep the router in **high precision** (it is a tiny
tensor) or protect the differential logit component. Do **not** quantize the gate
coarsely with the rest of the weights.

**Consumer metric:** top-k expert-set **flip rate** vs the fp16 router — split by
margin. Reconstruction error on the gate weights does not see selection.

---

## Model validated (real weights)

**`allenai/OLMoE-1B-7B-0924`** — 64 experts, **top-8** routing, 16 MoE layers.
262,144 real routing decisions captured over WikiText-2 (test).
Harness: [`benchmarks/validate_olmoe_routing.py`](../../benchmarks/validate_olmoe_routing.py);
raw data: [`benchmarks/results_olmoe_routing.json`](../../benchmarks/results_olmoe_routing.json).

Real routing margins are **tiny**: argmax (k=1) median **0.0146** (p10 0.0021);
top-8 set boundary (k=8) median **0.00152**. The "which 8 of 64" boundary is soft.

## Result 1 — the margin mechanism (controlled perturbation)

Inject a differential logit perturbation at the margin scale and measure the top-k
set flip rate, split at the median margin (low vs high). This isolates the
margin's role — and low-margin tokens flip **first, by orders of magnitude**:

**Top-8 set boundary (k=8):**

| σ / median margin | overall flip | low-margin | high-margin | **low/high ratio** |
|---:|---:|---:|---:|---:|
| 0.25 | 10.6% | 21.2% | 0.012% | **1740×** |
| 0.5 | 20.2% | 38.6% | 1.7% | **22.6×** |
| 1.0 | 35.7% | 58.7% | 12.7% | 4.6× |
| 2.0 | 57.7% | 77.8% | 37.5% | 2.1× |

**Argmax (k=1):** 0.25 → 1256×, 0.5 → 19.3×, 1.0 → 3.9×, 2.0 → 1.8×.

At a perturbation near the margin scale the low-margin tokens flip ~**10³× more**
than high-margin ones; the ratio compresses toward 1 as the noise grows past the
margins (everything flips). This is the differential/(A2) split restated for
selection, confirmed on a real router.

## Result 2 — naive gate quantization is catastrophic (the warning)

Quantize the router **gate weights** per-tensor and recompute the routing:

| gate weights | top-8 set flip rate | low/high ratio |
|---|---:|---:|
| 4-bit | **91.8%** | 1.09× |
| 3-bit | **98.7%** | 1.01× |

Naive 4-bit quantization of the router reshuffles the top-8 expert set for **92%**
of tokens. The low/high ratio is ~1 here **not** because the margin doesn't matter
but because 4-bit gate error is *far larger than the margins* (median 0.0015) — it
saturates every bucket. The margin structure (Result 1) is why: with near-zero
boundary margins, any coarse perturbation is above threshold for almost all tokens.

## Honest scope (model-specific refinement)

The dramatic low-vs-high flip ratio is real but appears at **controlled fine
perturbation**, not at practical bit-depths, because OLMoE's **top-8** set
membership is intrinsically soft (tiny boundary margins). A **top-2** router
(e.g. Mixtral) has larger margins and the effect survives to coarser
quantization; validating that specific regime is future work. The robust,
model-agnostic conclusion stands: **selection fragility is carried by the routing
margin, so the router must be kept precise — reconstruction error on the gate is
the wrong acceptance metric.**

## Instrument

```python
from turboquant_pro.operator_sensitivity import routing_sensitivity, routing_margins
s = routing_sensitivity(gate_logits, k=8)   # margin p10/p50/mean
m = routing_margins(gate_logits, k=1)        # per-token argmax margin
```

`operator_trace.recommend_quantization` classifies router gates as
`GATE_SELECTION`.
