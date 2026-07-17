# Model cards — real-model operator-sensitivity validation

Phase 7 turns turboquant-pro's operator-sensitivity *thesis* — that the right
quantization discipline is dictated by what the tensor **operator** does, not by
reconstruction error — into **real-model evidence**. Each card documents one
operator regime validated on real weights: the model, the harness, the numbers,
the discipline that follows, and the **negative cases** (where the naive choice
fails), preserved rather than hidden.

The unifying rule across every card: **acceptance is the metric the operator's
consumer actually uses** — perplexity / task score for generation, expert-set
flip rate for routing, state drift / perplexity for recurrences — **never
reconstruction cosine**, which is repeatedly shown here to be blind or even
anti-correlated with quality.

| Card | Regime | Model(s) | Consumer metric | Verdict |
|---|---|---|---|---|
| [attention_keys.md](attention_keys.md) | `SOFTMAX_SCORE` (attention keys) | Llama-2-7B/13B, Mistral-7B, Qwen2.5-7B/1.5B | perplexity, LongBench | per-channel / asym-NF4; per-vector PolarQuant **collapses** keys |
| [moe_routing.md](moe_routing.md) | `GATE_SELECTION` (MoE router) | OLMoE-1B-7B | top-k expert-set flip rate | selection is carried by the **margin**; low-margin tokens are fragile |
| [ssm_decay.md](ssm_decay.md) | `STATE_DECAY` (SSM recurrence) | Mamba-790m | WikiText-2 perplexity | quantize decay in the **native A_log** basis, not linearly |

Each regime maps to a discipline family in `turboquant_pro.operator_trace`
(`recommend_quantization`) and to a measurement primitive:
`turboquant_pro.a2_probe` (keys), `operator_sensitivity.routing_sensitivity`
(gates), `operator_sensitivity.state_decay_sensitivity` (decays).
