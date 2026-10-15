# Design: consumer-relative telemetry for a live model, and the control it could later drive

**Not implemented.** This builds out the `--model` source that console v2 lists and never built
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
