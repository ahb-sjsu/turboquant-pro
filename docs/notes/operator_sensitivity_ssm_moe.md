# Operator sensitivity beyond attention: gates and state-space decay

*The (A2) boundary for the two regimes hybrid architectures add — measured on
synthetic operators, acted on by the tooling. Formal derivations belong to the
companion theory paper; this note reports what the experiments show, and only
that.*

Andrew H. Bond · TurboQuant Pro technical note · 2026-07-15

---

## Why attention's answer doesn't transfer

Condition (A2) localized the sensitive coordinate for attention: per-channel
scale for keys (score path), symmetric residual for V/O. Hybrid and
state-space models (Mamba, RetNet, deep MoE) add operators whose sensitive
coordinate is *neither* — a top-k router reads only the order of its logits, and
a linear recurrence integrates a per-channel decay over time. `operator_trace`
already classifies these as `GATE_SELECTION` and `STATE_DECAY`; this note
supplies the sensitivity analysis that turns those labels into bit allocations.

Two boundaries, each measured before it was believed.

## 1. Gates: selection is carried by the margin, not the magnitude

A top-k router selects experts by the **order** of the logits `ℓ = W_gate·x`.
Two consequences:

- **Common-mode is free.** Adding the same constant to every expert's logit
  leaves the selection unchanged, so a common-mode quantization error costs
  nothing. Only the component of the perturbation *orthogonal to the all-ones
  direction* — the **differential** part — can flip routing. That is the
  tangential/(A2) split restated for selection, and it is exactly what
  `differential_fraction` measures (the routing analog of `tangential_fraction`).
- **The fragile tokens are the low-margin ones.** A token's routing survives a
  perturbation iff it stays below the **margin** — the gap between the k-th and
  (k+1)-th logit. So the sensitive statistic is the *margin distribution*, not
  the logit magnitude.

**Measured** (synthetic gate, 16 experts, per-output-channel symmetric
weight quant): top-1 flips concentrate on low-margin tokens, and the
concentration sharpens as precision rises —

| gate bits | overall top-1 flip | low-margin half | high-margin half | ratio |
|---|---:|---:|---:|---:|
| 2-bit | 0.55 | 0.66 | 0.44 | 1.5× |
| 3-bit | 0.23 | 0.40 | 0.06 | 6.3× |
| 4-bit | 0.10 | 0.20 | 0.002 | **88×** |

At usable precision almost every surviving flip is a low-margin token — the
boundary is the margin. **Discipline:** size the gate's bit budget against the
low percentile of the margin distribution (`routing_sensitivity` →
`predict_routing_flips`), and protect the differential component; the
common-mode is disposable. Unlike keys, no per-channel DC term is needed —
what must survive is *relative* logit order.

## 2. Recurrences: the slow channels are fragile, and error compounds

For a per-channel linear recurrence `hₜ = a·hₜ₋₁ + bₜ` the steady-state gain is
`1/(1-a)` and the memory length is `~1/(1-a)`. A fixed error in the decay `a`
is therefore amplified far more in **slow (long-memory) channels** (`a → 1`),
and it **accumulates over the sequence** — the recurrent analog of the
RoPE-slow-channel key finding (where the *slow rotary* channels carried the
fragile DC offset).

**Measured** (per-channel recurrence, fixed decay error ε = 0.01):

| seq len T | mean \|drift\| | corr(a, drift) | slow (a>0.8) / fast (a<0.5) |
|---|---:|---:|---:|
| 16 | 0.072 | +0.41 | 28× |
| 64 | 0.132 | +0.44 | 53× |
| 256 | 0.119 | +0.43 | 52× |

`decay_sensitivity` returns the per-channel coefficient
`dS/da = Σₜ t·a^{t-1}` (steady state `1/(1-a)²`), which grows toward `a → 1`
and with sequence length, quantifying the table.

**The fix, also measured.** Because the fragility lives at `a → 1`, quantizing
the decay on a **log-time-constant basis** — `τ = -log a`, then a uniform grid
on `log τ` — concentrates levels where they matter, the SSM analog of NF4's
non-uniform key levels:

| decay quant | 3-bit state drift | 4-bit state drift |
|---|---:|---:|
| linear grid on `a` | 1.67 | 1.09 |
| **`log_tau` basis** | **0.31** | **0.18** |
| improvement | **5.4×** | **6.1×** |

`quantize_decay(a, bits, basis="log_tau")` ships this. (Mamba's native `A_log`
parameterization is a special case of the same warp, and shows the same
2.5–3.5× effect.)

## What this is, and is not

- **Measured, not proven.** Both boundaries and the `log_tau` win are empirical,
  on *synthetic* operators (a random gate; a per-channel linear recurrence).
  They are consistent with, and motivated by, the (A2) theory and the keys/RoPE
  findings, but the formal sensitivity theorems for magnitude-gated selection
  and selective recurrences are the theory work of `the-angular-observer`, not
  claimed here.
- **Not yet validated on real models.** Real Mamba / RetNet decays and real MoE
  routers (Mixtral, Qwen-MoE) are the natural next step — feed their gate logits
  to `routing_sensitivity` and their decays to `state_decay_sensitivity`, and
  confirm the boundary end-to-end. Until then these are operator-level results.
- **A negative result kept next to them:** consumer-weighted Lloyd (reweighting
  a scalar codebook's distortion) does *not* beat MSE for cosine / dot / top-k
  consumers — for scalar codebooks the (A2) dissociation is structural (the
  quotient choice), not a distortion-metric one. We did not ship a win there
  because there wasn't one.

## Reproduce

```bash
pytest tests/test_operator_sensitivity.py -q
```
