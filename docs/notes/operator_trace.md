# Operator-regime tracing: inferring the (A2) consumer

*From a declared consumer to an inferred one — the first step toward
human-out-of-the-loop, operator-dependent quantization.*

Andrew H. Bond · TurboQuant Pro technical note · 2026-07-15

---

## The gap this closes

Condition (A2) (`a2_probe`, and the companion theory paper) says a
scale-discarding quotient is safe exactly when the **consumer's** metric is
carried by the tangential part of the displacement. Until now the consumer had
to be **declared**: a human knew that KV-cache keys feed `softmax(Q·Kᵀ)` (so
per-channel scale is vital) and that V/O weights write the residual stream
linearly (so symmetric quantization is safe), and passed
`consumer="attention_logits"` by hand.

`operator_trace` **infers** the consumer from the model. It maps every
parameter tensor to the operator its output flows into, and from that regime to
its quantization discipline — no declaration.

## The taxonomy

| Regime | Consumer operator | Example tensors |
|---|---|---|
| `SOFTMAX_SCORE` | `softmax(Q·Kᵀ)` — the attention score path | `q_proj`, `k_proj` |
| `LINEAR_RESIDUAL` | writes the residual stream ~linearly | `v_proj`, `o_proj`, MLP |
| `GATE_SELECTION` | top-k routing on gate-logit margins/order | MoE `router` |
| `STATE_DECAY` | SSM per-channel recurrence / decay | Mamba `A_log`, `x_proj` |
| `NORM` | layernorm / rmsnorm scale | `*.norm.weight` |
| `UNKNOWN` | no evidence | (falls back to the safe default) |

## The discipline table is the (A2) content

The prescription has **two independent axes**: the *family / DC-protection* (a
correctness choice — which quotient is safe) and the *sensitivity / bit
allocation* (a budget choice). Both are resolved per `(regime, target)`, and
the target matters because **the sensitive coordinate flips between quantizing
weights and quantizing cached activations**:

| regime | `WEIGHT` (weight PTQ) | `KV_ACTIVATION` (cache quant) |
|---|---|---|
| `SOFTMAX_SCORE` | symmetric · **low** sensitivity — softmax absorbs a shared perturbation to `W_K` (matched-bit finding) | **per-channel + zero-point** · high — cached keys carry a DC offset softmax reads (keys finding) |
| `LINEAR_RESIDUAL` | symmetric · **high** sensitivity — V/O write the residual linearly and are 2.3–6× more behaviorally sensitive than Q/K | **polar** · low — values are averaged by attention (values finding) |
| `GATE_SELECTION` | per-channel + zero-point · high — clipping flips the top-k expert choice | per-channel · high |
| `STATE_DECAY` | per-channel + zero-point · high — the recurrence's time constant is per-channel | per-channel · high |
| `NORM` | keep fp | keep fp |
| `UNKNOWN` | per-channel + zero-point · medium — never discard scale on an unclassified tensor | same |

The `SOFTMAX_SCORE` row is the crux: the same projection is the **robust** side
when you quantize its weights and the **fragile** side when you quantize its
cached activations. That is the keys finding and the projection-sensitivity
reversal reconciled ([`KV_KEYS_FINDING.md`](../KV_KEYS_FINDING.md),
[`projection_sensitivity_deconfounded.md`](projection_sensitivity_deconfounded.md))
— one architecture, opposite fragile coordinate, selected by the operator.

## Two front-ends, combined (`prefer="auto"`)

- **Structural** (always available): classify each parameter from its owning
  module's *type* and *name*. `nn.LayerNorm` → `NORM` regardless of name;
  `…k_proj` → `SOFTMAX_SCORE`; a router `gate` → `GATE_SELECTION` (with a
  negative-lookahead so SwiGLU's `gate_proj` is **not** misread as a router);
  a bare `nn.Linear` with no signal defaults to `LINEAR_RESIDUAL`. Robust and
  dependency-light, but blind to renamed or novel operators.
- **Graph (`torch.fx`)** (best-effort): symbolically trace the model, find the
  *sink* operators — `softmax`/`exp`, `topk`/`argmax` (routing),
  `cumsum`/scan (recurrence) — and backtrace each to the `Linear` layers that
  feed it. This is what tags the score/gate/state tensors when the names give
  nothing away, so the pass works on an **unseen** architecture. Where fx has
  positive evidence it overrides the structural guess (`method="fx"`);
  everything else keeps the structural baseline. If the model cannot be traced
  (dynamic control flow), the plan degrades gracefully to structural with
  `traced=False`.

The unit tests exercise the two hard cases directly: an `ObfuscatedAttention`
whose `left`/`right` linears are only identifiable as the score path *by the
graph*, and a `SwiGLU` whose `gate_proj` must **not** be classified as a
router.

## Usage

```python
from turboquant_pro import recommend_quantization, QuantTarget

# human out of the loop: model in, per-tensor discipline out
for name, d in recommend_quantization(model, target="weight").items():
    print(name, d.family, d.protect_dc, d.sensitivity)

# same model, KV-cache target — keys become the fragile per-channel+DC side
recommend_quantization(model, target=QuantTarget.KV_ACTIVATION)
```

## Honest scope

The structural classifier is name/type heuristics — it covers the common
families (Llama/Mistral/Qwen attention + MLP, MoE routers, Mamba-style SSM,
standard norms) and defaults unknown tensors to the conservative discipline; it
is not a guarantee. The fx pass depends on `torch.fx.symbolic_trace`, which many
production HF models defeat with data-dependent control flow (the plan then
falls back to structural). Regimes are assigned from operator *identity*, not
yet confirmed against a calibration batch — the natural next step is to gate a
`SOFTMAX_SCORE`/`GATE_SELECTION` decision through the existing `a2_probe` on
real activations. And the `GATE_SELECTION`/`STATE_DECAY` disciplines are, for
now, the conservative "protect per-channel scale" prior; their precise (A2)
boundaries for magnitude-gated routing and selective recurrences are the theory
work of item 3, to be derived in `the-angular-observer` and folded back here.

## Reproduce

```bash
pytest tests/test_operator_trace.py -q
```
