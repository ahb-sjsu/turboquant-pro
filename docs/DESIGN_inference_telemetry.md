# Design: consumer-relative telemetry for a live model, and the control it could later drive

**Layer 1 implemented (PR #319); layer 2 not.** Section 6 says what was built, where it departs
from this design, and what it measured on Gemma 3 4B. This builds out the `--model` source that
console v2 lists and never built
(`docs/DESIGN_console_v2.md`, section 3, and its open decision 3, a GPU for the model source).
It has two layers. Layer 1, telemetry, only reads and can be built now. Layer 2, control, changes
what the model computes and waits on Part V's verdicts (`docs/PREREG_full_consumer_kv.md`).

## 0. The idea

A serving stack reports throughput, latency, cache occupancy and memory. None of it says what
the model's consumer is reading, or whether the KV compression in use still suits the traffic
it now serves. The library already computes both offline: read operators
(`read_operators.py`), read-weighted bit allocation (`read_allocation.py`), consumer-relative
distortion, and certificates that expire when their observer changes
(`docs/DESIGN_certificate_expiry.md`). This note moves those measurements into a running model
and shows them the way the console shows an index: as instruments, not as a report.

## 1. What makes it cheap: a head's key reads are its queries

For softmax attention, the derivative of a head's attention weights with respect to a key is
`∂p_i/∂k_s = p_i (e_s − p_is) q_i / √d`, so the subspace in which the head reads its keys is
spanned by its queries. readscope verified this as an exact identity (calibration C-3b,
resolution 1.000 on 36 heads of Llama, Qwen and Mistral). Part II's per-channel read weight
`w_j = Σ_i q_ij²` is therefore a running second moment of the queries the model already
computes. It needs no backward pass and no probe.

- **Cost.** One multiply-add per query channel per token per layer, plus a `d × d` outer
  product when the full operator (not only its diagonal) is wanted, sampled. That is small
  against the attention it observes. The sampling rate and its measured overhead are a
  smoke-test number, not claimed here.
- **Coordinates.** Keys are cached after the rotary embedding, so queries must be read after
  it too (readscope hooks `q_proj` and the model's own `rotary_emb`). Under grouped-query
  attention, a key head's operator sums over the query heads that share it.
- **What this does not give.** The deeper consumers of Part V (`head`: through softmax, values
  and `W_O`; `fisher`: the model's output) need the values, the output projection or a
  backward pass. Layer 1 starts with the logit consumer and adds the head consumer (values
  and `W_O` are also already computed). The Fisher consumer stays offline.

## 2. Layer 1: telemetry (read-only)

Readings, per layer and key head, over a window of tokens:

| reading | what it is | consumer-relative? |
|---|---|---|
| read spectrum | eigenvalues (or the diagonal) of the windowed query second moment | yes, it is the consumer |
| effective read rank | participation ratio of the read spectrum | yes |
| attention entropy | mean entropy of the head's attention rows | descriptive |
| KV bytes | stored bytes per head, hot and cold, from the cache in use | no, accounting |
| predicted distortion | `tr(P_C Σ_δ)` for the coding in use, `Σ_δ` from the codec's calibrated error | yes |
| plan drift | `κ_P` and subspace resolution between the operator the KV plan was calibrated on and the live window | yes |
| plan status | VALID, EXPIRED (with the reason) or INCONCLUSIVE, by certificate-expiry rules | yes |

**Plan drift is the distinctive reading.** Part IV's theory
(`docs/THEORY_consumer_mismatch.md`, draft PR #316) bounds what a structure fitted to one
consumer loses under another by `κ_P κ_Σ`, and splits a budgeted loss into a part from moved
directions and a part from a moved allocation. Live, that becomes a monitor: a KV plan is
calibrated on some traffic; when the live read operator departs from the calibration
operator by more than the plan's certified tolerance, the plan's certificate expires, and the
console says which heads moved and whether the move is mostly scale (correctable in closed
form, Proposition 7) or mostly direction. Reconstruction cosine is never shown as a quality
gauge (the library's standing rule); it can appear only as a labelled diagnostic.

**Where the hooks attach.**

| runtime | what is reachable | layer 1 coverage |
|---|---|---|
| Hugging Face + `hf_cache.TurboQuantCache` | queries, keys, values, `W_O` in-process | full |
| vLLM + `vllm_plugin` | today only cache statistics (`LayerStats`: lengths, bytes, ratio) | accounting rows only, until an attention-side hook exists |
| llama-server, other servers | `/metrics` and `/slots` counters | accounting rows only |

So layer 1 is built first on the Hugging Face path. A vLLM attention hook is a separate piece
of work, and its cost inside vLLM's kernels is unknown until measured.

**Transport and display.** Readings go through the existing telemetry contract
(`turboquant_pro/telemetry`, context-scoped tracers, never a global hook). The console gets a
Models page: the scope shows a chosen head's read spectrum over time, the spectrum analyzer
shows the live operator against the calibrated one, and the plan-status badge sits in the
header next to the observer badge. Prometheus export follows `tqp monitor`'s format.

## 3. A store DBAs can query: telemetry in PostgreSQL with `tqvector`

The repository already ships a PostgreSQL extension in Rust, `pgext/` (crate `tqvector`, pgrx
0.12, PostgreSQL 14 to 17, optional CUDA through cudarc). Telemetry can land there as rows a
DBA queries with SQL:

- one row per (model, layer, key head, window): timestamp, the scalar readings above, and the
  window's read spectrum as a vector column;
- views for drift per head, plan status over time, and bytes against distortion.

The vector column is not there to save space (a 128-channel spectrum is small). It is there
for search: "find the past windows, or the calibrated plans, whose read spectrum is nearest to
now". That turns plan selection into a lookup over stored consumers. Retrieval may use any
similarity, but **acceptance never does**: a retrieved plan is accepted only by the consumer
metric (`κ_P`, predicted distortion), as everywhere else in the library. Whether `tqvector`'s
compressed type or plain pgvector suits these rows is a measurement, and the sequential-scan
cost recorded in `docs/tqvector-benchmark-findings.md` applies.

### Reusing `pgext`'s Rust codec on the inference side

Part of `pgext` can be reused, but not its GPU path as it stands:

- **Reusable.** The codec core in `compress.rs` and `codebook.rs` is plain Rust: the seeded
  rotation (cached per dimension and seed), the 2-, 3- and 4-bit codebooks, and the bit packing,
  with the encoding frozen by golden bytes (#165). Only the `TqVector` type is tied to
  PostgreSQL (it derives `PostgresType`), and the crate depends on pgrx unconditionally. Splitting
  a `tqcore` crate with no pgrx, which `pgext` then depends on, would let a Rust runtime, a PyO3
  binding or a C ABI (for llama.cpp) use the same codec. KV pages coded that way would also be
  byte-compatible with `tqvector` rows.
- **Not a fit for the KV hot path.**
  - The codec spends one bit width on a whole vector. Read-weighted allocation (Parts II and V)
    needs per-channel widths.
  - The CUDA kernel (`gpu.rs`) only compresses: one thread per vector, a dense rotation per
    vector, and results copied back to the host. Decompression runs on the CPU.
  - Inference needs coding and decode-attention fused on the device. The library already has
    that in Triton and CUDA (`kv_triton.py`, `kv_fused.py`, `cuda_kernels.py`).

## 4. Layer 2: control (waits on Part V)

Consumer-aware inference re-allocates KV bits per head and channel from the live operator,
under a declared quality contract, and re-plans when the plan's certificate expires.
`runtime_policy.TQPRuntimePolicy` and `read_allocation.allocate_bits` are the pieces it would
compose. A query surface in `tqp query`'s style could read (proposed syntax, nothing exists):

```
ANALYZE MODEL 'qwen2.5-7b' USING TRAFFIC 'window'
SHOW READS FROM MODEL 'qwen2.5-7b' WHERE layer = 12
EXPLAIN KV PLAN FOR MODEL 'qwen2.5-7b' WITH (PPL_DELTA <= 0.05, BYTES <= 0.25, CERTIFY)
```

**Why it waits.** Part II found that allocating key bits by logit reads lowered perplexity on
all three 7B models but improved qasper only on one (K4 inconclusive), and did not beat key
variance alone (K4b). Part V asks whether deeper consumers fix that (V1 to V3) and whether a
consumer fitted on one text serves another (V4). Control is worth building only for the
consumer Part V finds pays, and online re-planning is worth building only if V4 says fitted
consumers do not transfer on their own. Layer 1 is useful whichever way Part V comes out: it
shows when a plan has stopped fitting its traffic.

## 5. Build order and decisions

1. Hugging Face hook for queries after the rotary embedding; windowed second moment;
   readings through the telemetry contract. Overhead measured in a smoke run on Atlas's free
   GPU, or on NRP (owner decides, as in console v2's decision 3).
2. Plan drift and plan status, reusing certificate-expiry rules.
3. Console Models page.
4. PostgreSQL sink with `tqvector`, and the views.
5. After Part V is scored: control, for the consumer it supports.

Decisions for the owner: the GPU for step 1; whether step 4 targets `tqvector` first or plain
pgvector; and whether a vLLM attention hook is in scope.

## 6. Implemented, and measured (2026-10-15)

Steps 1 to 4 are built: `turboquant_pro/model_telemetry.py` (monitor, plan, status),
`model_watch.py` and `tqp model calibrate|watch`, `model_telemetry_pg.py`, and the console's
model page (`console/model_source.py`, `console/model_view.py`, the Go client's layout).
Usage is in `docs/CLI.md`. Step 5 is not built.

**What changed from the design above.**

- **Where the hook sits.** The design said "queries after the rotary embedding". The monitor
  goes one step further down: it wraps the attention function in transformers'
  `AttentionInterface`, so it reads the exact tensors, scale and mask the attention is given
  (after QK-norm, with Gemma's `query_pre_attn_scalar`, with sliding-window and padding masks).
  Nothing is rebuilt per architecture.
- **The operator is exact, not the query second moment alone.** For softmax attention the
  Jacobian Gram over all keys collapses to `a s^2 q q^T` per query row, with
  `a = sum_t g_t^2 p_t^2 (1 - 2 p_t + S2)`, so the `softmax` consumer costs O(S) per row. The
  query second moment is kept too, as the `logit` consumer. The `head` consumer (through the
  values and `W_O`) is not built.
- **Distortion is measured, not predicted.** A watched `TurboQuantCache` reports each spilled
  key block with its reconstruction (`spill_hooks`), so `tr(P Sigma_delta)` uses the codec's
  actual error.
- **Status follows the certificate-expiry vocabulary:** STALE (not EXPIRED), INCONCLUSIVE and
  UNCHECKED. Two rules were added after the first runs on Gemma, both because a number read
  against 1 rather than against its own noise misled:
  - each plan records its **calibration floor**, what its two calibration halves score
    against each other for overlap, kappa and regret. A head whose own halves already fall
    below the bar is INCONCLUSIVE ("calibrate on more traffic"), and every plan reading
    carries its floor as `baseline`;
  - a number the window cannot support is withheld with its reason. Kappa needs as many
    sampled rows as the plan's subspace has directions, and regret needs as many key rows as
    channels; fewer leave the estimate singular or rank-deficient.
- **The plan's read subspace is sized by trace** (the smallest eigenspace carrying 0.9 of it),
  not by `validity`'s fixed 8-direction sketch. On Gemma's 256-channel heads that sketch
  carried less than 0.8 of the trace in 131 of 136 heads, so every one abstained. The overlap
  is relative to the plan's own (1 when nothing changed). `validity.check_validity` compares
  the raw share, which a sketch carrying 0.80 to 0.85 of its trace fails against its own
  operator. That is not changed here and is noted for its owner.
- **Search is by cosine.** The `tqvector` installed on Atlas (built 2026-04-09) has `<=>` and
  none of the repository's L2 functions, and cosine suits a non-negative, trace-normalized
  profile.
- **Not built:** the KV-bytes reading, Prometheus export, the scope and spectrum-analyzer
  views of a head (the model page uses the PanelView shape), and a vLLM attention hook.

**Measured on Gemma 3 4B** (`unsloth/gemma-3-4b-it`; Atlas CPU, fp32, 16 threads; results in
`/archive/ahb-sjsu/tqp-arch-smoke/`). Calibration: eight English prose prompts, each prefilled
and continued for 128 greedy tokens, 1,236 tokens. Then four unseen prose prompts (674 tokens)
and four code prompts (710 tokens), each also continued for 128 tokens. Window 1,024 tokens,
4 bits per key channel, 34 layers by 4 key heads.

| per key head, median | unseen prose | code | the plan's own halves |
|---|---|---|---|
| status | 117 VALID, 0 STALE, 5 INCONCLUSIVE, 14 UNCHECKED | 69 VALID, 34 STALE, 19 INCONCLUSIVE, 14 UNCHECKED | |
| overlap | 0.939 | 0.860 | 0.916 |
| kappa | 22 | 121 | 51 |
| regret | 1.64 | 2.33 | 1.83 |
| regret over its floor | 0.89 | 1.23 | |

- **Observing changes no output.** Logits with and without the monitor are bit-identical on
  Gemma 3 4B. All 34 layers were observed and the hook raised no error.
- **The 14 UNCHECKED heads** have operators that need more than half the 256 channels to
  carry 0.9 of the trace. There is no low-dimensional read subspace to test against.
- **Two earlier runs used smaller plans, before the floors were added.**
  - A plan from 331 prefill tokens put 15 in-domain prose heads at STALE, with overlap 0.82
    against a bar of 0.85.
  - The same plan scored generated prose STALE in 95 heads.

  With the 1,236-token plan that includes decoding, none of those prose heads is STALE. These
  runs do not separate the two differences, decoding against prefill and 1,236 against 331
  tokens. The floor rule exists so that a plan too small to tell abstains instead of going
  STALE. It is tested on synthetic halves and was not re-run on the 331-token plan.
- **Key codec** (`TurboQuantCache`, hot window 16, the shipped 4-bit asymmetric-NF4 keys):
  `tr(P Sigma_delta)` from 0.0004 to 0.015 per head, median 0.0026, on unseen prose.
- **Overhead** (median of 3; prefill of 324 tokens, then a 32-token generation):

  | monitor | prefill | generation |
  |---|---|---|
  | none | 10.1 s | 29.9 s |
  | every call | 11.1 s (+10%) | 33.7 s (+13%) |
  | every 8th call | 9.8 s (−2%, within noise) | 31.5 s (+5%) |

  GPU overhead is not measured.
- **PostgreSQL** (Atlas, PostgreSQL 16, a scratch database). One prose run with the cache wrote
  1,088 readings, 136 with `tqvector` read profiles, 448 kB, and the status view answered.
- **Console.** `tqp console --model-readings` drew the page from those readings in a real pty
  (snapshot taken with P).

**What this does and does not show.** The readings separate English prose from code on one
model, with a plan from one text. They do not show that a STALE head predicts a task loss,
that a re-fitted allocation recovers one, or that any of this transfers to another model. Those
are Part V's questions (V1 to V4) and the control layer's.
