# Opt-in online key calibration — an honest negative on the consumer metric

**Headline: lightweight online calibration does *not* beat the calibration-free
default on the metric attention consumes.** It lowers raw reconstruction error,
but softmax-KL — the thing the model actually uses — gets *worse*. This is the
project's own thesis (reconstruction error is not the target) re-confirmed
against a feature we hoped would help. The API ships **experimental / opt-in**;
the recommended key path remains calibration-free asym-NF4 + 2% outliers.

## What was tried

`calibrate_key_quantizer` / `PerChannelKV.calibrate` fit a per-(head, channel)
**Lloyd-Max (MSE-optimal)** codebook from a representative sample of real keys and
reuse it — the data-fit idea behind KVQuant's offline calibration, made
lightweight (Lloyd over a sample, optional importance weights).

## Result — measured, not hoped

Acceptance metric is **softmax-KL** of `q·Kᵀ` vs fp16 (the attention consumer),
matched at 4-bit + 2% outliers, on a realistic post-RoPE-like regime
(per-channel DC offset + heavy Student-t(3) tails), on a held-out block.

**Single favorable seed (what looked like a win):**

| key codebook | key \|err\| | softmax-KL |
|---|---:|---:|
| asym-NF4 (calibration-free default) | 0.238 | 0.365 |
| calibrated — equal-mass quantiles | 0.146 | 1.698 |
| calibrated — Lloyd-Max | **0.116** | 0.288 |

**Across seeds (the truth):** repeating over independent distribution / query /
test draws, the Lloyd-calibrated codebook's softmax-KL ratio to the
calibration-free default is **mean 1.29, median 1.26 — it lost in 6/6 trials**
(≈29% *worse* attention fidelity). The single-seed 0.288 above was
query-seed luck.

## Two honest findings

1. **Reconstruction ≠ attention, again.** Lloyd calibration reliably *lowers key
   reconstruction error* (it minimizes MSE by construction), yet reliably
   *raises* softmax-KL. Optimizing the marginal per-channel codebook discards the
   explicit DC-offset / zero-point modeling that makes asym-NF4 well-matched to
   post-RoPE keys — so the reconstruction gets closer while the attention
   distribution gets further. This is exactly the failure mode the project was
   built to expose.
2. **Equal-mass quantile calibration is worse still** (KL 1.70). Whatever the
   right objective is, it is neither "equalize mass" nor "minimize marginal MSE".

## Scope and what would settle it

- This is a **synthetic proxy** (softmax-KL on a controlled distribution), not a
  task score. It does **not** prove the option is useless on *real* data: KVQuant
  reports a ~0.24 qasper gain from a **sensitivity-weighted (Fisher) K-means**
  codebook — a stronger objective than plain Lloyd, on real key/query
  correlations. Whether that closes the gap here is a real-model question:
  `benchmarks/kv_quant_shootout.py` / `benchmarks/kvquant_matrix/` with a
  calibrated key quantizer, on a GPU box.
- So the API is provided **experimental**, for users who want to try a data-fit
  codebook and measure it on *their* task — with the honest expectation set here
  that a lightweight Lloyd codebook did not beat the calibration-free default on
  our attention proxy. Closing the last fraction likely needs Fisher-weighting
  against the real query distribution, not a lighter method.
