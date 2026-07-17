# Operator-aware quantization guide

**The thesis.** A tensor is not just numbers; it is numbers *about to be consumed by
an operator*. The operator decides which part of the tensor carries the signal — and
therefore which quotient you may quantize away for free and which you must protect.
Reconstruction error averages over the whole tensor and so is **blind** to this; it
is the wrong acceptance metric, and for some operators (attention keys) it is
actively *anti-correlated* with quality.

Every regime below is measured, has a shipped instrument, and is validated on real
models (see [`model_cards/`](../model_cards/)).

## The map

| Tensor → operator | What carries the signal | Discipline | Free to discard | Instrument | Evidence |
|---|---|---|---|---|---|
| **Attention keys** → `softmax(Q·Kᵀ)` | per-channel scale (a few channels dominate the dot product) | **per-channel, asymmetric NF4** | — (per-vector norm is *not* free here) | `a2_probe` | [attention_keys](../model_cards/attention_keys.md) |
| **Attention values** → weighted average | direction, robustly averaged | **PolarQuant** (rotation + norm + direction codes) | small direction error | `TurboQuantKV` | [KV guide](../../benchmarks/kvquant_matrix/KV_QUANT_GUIDE.md) |
| **MoE router gate** → top-k `argmax` | the *order* of logits (the routing margin) | **keep high precision**; protect the differential component | common-mode shift | `operator_sensitivity.routing_sensitivity` | [moe_routing](../model_cards/moe_routing.md) |
| **SSM decay** → linear recurrence | slow channels (`a → 1`), amplified by `1/(1-a)` | **log-time-constant (A_log) basis** | resolution far from `a=1` | `operator_sensitivity.state_decay_sensitivity` | [ssm_decay](../model_cards/ssm_decay.md) |
| **Embeddings** → cosine retrieval | the *ranking* (angle) | **PCA + low-bit + exact rerank** | per-vector magnitude (metric-dependent) | `rank_certificate`, recall@k | [claims](../claims.md) |
| **Model weights** → matmul | behavior on real inputs | **behavioral agreement**, projection-sensitivity-aware | error below the noise floor | `behavioral_agreement` | `experiments/results_matched_bit/` |

## The unifying quantity — (A2)

Each row is the same split applied to a different operator:

- For **keys / embeddings**, the (A2) *tangential fraction* measures how much of the
  quantization displacement survives row-normalization. When displacements are
  norm-dominated (tangential fraction low), throwing away magnitude is *safe*; when
  they are directional, it destroys the ranking.
- For **gates**, the exact analog is the *differential fraction*: selection is
  invariant to a common-mode logit shift, so only the differential error across
  experts — beyond the routing margin — flips the choice.
- For **decays**, it is the slow-channel amplification `1/(1-a)`: the fragile
  coordinate is the long-memory channel, and it must get the fine resolution.

## The cautionary tale (why reconstruction cosine is banned as acceptance)

On real Qwen2.5-1.5B, the shipped PolarQuant quantizer applied to *keys* produces the
**best** key reconstruction (0.095 error) and the **worst** perplexity (12.24 →
**10643**) — reconstruction is anti-correlated with quality because keys feed a dot
product, not an average. A per-channel quantizer with *higher* reconstruction error
gives 670× better perplexity. This is the founding finding
([KV_KEYS_FINDING.md](../KV_KEYS_FINDING.md)); it is why every acceptance check in
this project is a consumer-metric, and why the [certification guide](certification.md)
matters.

## Using it

Let the tracer classify a model's tensors and hand back the discipline per tensor:

```python
from turboquant_pro import recommend_quantization
disciplines = recommend_quantization(model, target="kv_activation")
# {tensor_name: Discipline(family=..., protect_dc=..., sensitivity=...)}
```

Or check a single regime at calibration time — no generation run needed:

```bash
tqp probe --npy keys.npy --consumer attention_logits   # keys: per-channel vs polar
```

For the runtime side — backing off automatically where an operator is fragile — see
the [production lifecycle guide](production_lifecycle.md) and `TQPRuntimePolicy`.
