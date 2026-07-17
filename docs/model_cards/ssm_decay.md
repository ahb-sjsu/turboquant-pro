# Model card — SSM state decay (`STATE_DECAY`)

**Regime:** a state-space model's per-channel linear recurrence
`h_t = a·h_{t-1} + b_t`. The steady-state gain is `1/(1-a)` and the memory length
`~1/(1-a)`, so a fixed error in a slow channel (`a → 1`) is amplified far more and
**compounds over the sequence** — the recurrent analog of the RoPE-slow-channel
key finding.

**Discipline (what to do):** quantize the decay in the **native log-time-constant
basis** — a uniform grid on `A_log` (equivalently `log(-log a)`), what
`operator_sensitivity.quantize_decay(basis="log_tau")` implements — which puts
fine resolution where `a → 1`. **Never** a linear grid on the continuous decay
`A = -exp(A_log)`.

**Consumer metric:** WikiText-2 perplexity (generation). Reconstruction error on
`A` is blind to the amplification.

---

## Model validated (real weights)

**`state-spaces/mamba-790m-hf`** — 48 SSM layers, `A_log` shape (3072, 16) =
2,359,296 decay channels. WikiText-2 (test), fp32 eval (fp32 so a collapsed
variant reports a finite perplexity rather than overflowing to NaN).
Harness: [`benchmarks/validate_mamba_decay.py`](../../benchmarks/validate_mamba_decay.py);
raw data: [`benchmarks/results_mamba_decay.json`](../../benchmarks/results_mamba_decay.json).

## Result — the basis is nine orders of magnitude

Per-layer 3-bit fake-quantization of the model's own `A_log`, put back into the
model, WikiText-2 perplexity:

| decay quantization (3-bit) | perplexity |
|---|---:|
| fp32 baseline | **11.65** |
| **linear grid on the continuous decay `A`** | **1.01 × 10¹⁰ 💥** |
| **native `A_log` / log-time-constant basis** | **14.44** |

Same bit budget, same layers — **only the basis differs**. The linear grid
collapses the model (perplexity ~10¹⁰); the native `A_log` basis stays within
~2.8 perplexity of the fp32 baseline. Ratio ≈ **7 × 10⁸**.

## Why

The fragile channels are the slow, long-memory ones (`a → 1`), where the gain
`1/(1-a)` blows up. On this model the shipped `state_decay_sensitivity` reports a
**max steady-state gain of 42,690** across the real decays — a linear grid spends
almost all of its levels on fast channels and leaves the slow ones catastrophically
under-resolved, so their decay error is amplified and compounds over the 2048-token
window. The log-time-constant basis concentrates levels exactly where `a → 1` (the
SSM analog of NF4's non-uniform key levels).

## Instrument

```python
from turboquant_pro.operator_sensitivity import (
    state_decay_sensitivity, quantize_decay,
)
sens = state_decay_sensitivity(decays, seq_len=2048)   # max_gain, slow-channel fraction
a_q  = quantize_decay(decays, bits=3, basis="log_tau")  # the recommended basis
```

`operator_trace.recommend_quantization` classifies these tensors as `STATE_DECAY`.

**Scope/caveat:** this is a per-layer, per-tensor fake-quant of `A_log` on
WikiText-2 — it isolates the *basis* effect (the paper's claim), not an
end-to-end deployed SSM quantizer. The magnitude of the linear-basis collapse is
implementation-sensitive (per-tensor 3-bit here overflows fp16, hence the fp32
eval); the **direction and ~9-order-of-magnitude gap are the robust finding.**
