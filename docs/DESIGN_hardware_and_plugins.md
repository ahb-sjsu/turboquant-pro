# Design Doc — Widening the Base: Hardware Compatibility & Out-of-Tree Quantization Plugins

**Status:** proposed · **Owner:** TBD · **Goal:** make turboquant-pro the
*certification and fused-decode substrate* that production quantization recipes
plug into — rather than a competing recipe zoo — and run it on the hardware
people actually have.

## 1. Position, honestly stated

TensorRT-LLM ships FP4, FP8, FP8 KV cache, NVFP4 KV cache, GPTQ, and AWQ as
production recipes with NVIDIA-tuned kernels. bitsandbytes owns LLM.int8,
QLoRA/4-bit (NF4/FP4 blockwise), 8-bit optimizers, and is actively broadening
accelerator support. We will not out-kernel TensorRT-LLM on NVIDIA silicon and
we should not try. What neither offers:

1. **Task-geometry instruments** — rank certificates (distribution-free τ/ρ
   floors), the (A2) probe with its two failure classes, `operator_trace`
   regime→discipline inference, `behavioral_agreement`. Today these only speak
   to our in-tree formats.
2. **A format-agnostic fused decode** — M4 proved that any key format whose
   dequant is *affine per channel* (`dequant = μ + a · grid[code]`, optional
   sparse overlay) gets compute-on-codes attention from **one kernel**: only
   the grid table and scales differ.
3. **A recipe-neutral evidence ladder** — pre-registered, public-data,
   reproduction-status-tracked claims (`CLAIMS.md`), applied to *someone
   else's* recipe as readily as ours.

The strategic move is therefore: **define a plugin contract narrow enough that
one kernel and one instrument suite serve every affine recipe, and let
TensorRT-LLM-style and bitsandbytes-style formats live out of tree.** The
pitch to a recipe owner: register your format, get certification + fused
decode + the comparison harness for free.

## 2. The plugin contract

### 2.1 Core protocol (`turboquant_pro.plugins`)

```python
class QuantizerPlugin(Protocol):
    name: str                      # "bnb_nf4", "fp8_kv", "nvfp4_kv", "awq", ...
    tier: str                      # "experimental" | "beta" (api-stability.md)
    targets: set[QuantTarget]      # KV_ACTIVATION, WEIGHT, EMBEDDING

    def compress(self, x, **kw) -> Container: ...
    def decompress(self, c: Container) -> Array: ...

    # Optional capability: the affine contract (unlocks fused decode + ADC)
    def grid_params(self, c) -> tuple[mu_HD, weight_HD, grid_L] | None: ...
    def outlier_csr(self, c) -> tuple[row_ptr, cols, deltas] | None: ...

    # Optional capability: hardware-native execution (unlocks dtype passthrough)
    def native_dtype(self) -> torch.dtype | None: ...   # e.g. float8_e4m3fn
```

Design rules:

- **`grid_params` is the whole fused-decode story.** M4's
  `PreparedPCKBlock`/`pck_block_partials_cuda` already consume exactly
  `(μ, a, grid)` + optional CSR. A plugin that returns them inherits the fused
  path, the per-flush prepared-page cache, and the exactness gate *unchanged*.
  A plugin that returns `None` gets decompress-then-attend (still correct,
  still certified).
- **Instruments require only `compress`/`decompress`.** The rank certificate,
  (A2) probe, and `behavioral_agreement` are consumer-metric tools — they need
  round-trips, not internals. Every registered plugin is certifiable on day
  one.
- **Containers extend TQE1.** A plugin declares a format tag + version;
  serialization stays in the TQE1 envelope so mixed-format caches and indexes
  remain one wire format.
- **Discovery via entry points.** `[project.entry-points."turboquant_pro.plugins"]`
  — out-of-tree packages (`tqp-bnb`, `tqp-trtllm`) register without touching
  this repo. `AutoConfig` grows `key_format="auto" | <plugin name>` and
  resolves through the registry; `operator_trace`'s regime→discipline table
  maps to plugin names, so recipe selection becomes model-driven.

### 2.2 Conformance kit

`turboquant_pro.plugins.conformance` — a pytest suite a plugin package runs in
its own CI:

1. round-trip shape/dtype/error envelope; packed and unpacked
2. TQE1 serialize → deserialize → bitwise-equal container
3. if `grid_params`: **fused-score exactness vs decompress-then-attend**
   (the M4 gate, parameterized over the plugin)
4. if `outlier_csr`: CSR-vs-scatter equivalence (the `TestCSR` gate)
5. instrument smoke: certificate + (A2) probe run and return finite numbers

This is how "out-of-tree" stays trustworthy: the contract is executable.

### 2.3 Dogfood first (P0 exit criterion)

In-tree formats — `PerChannelKV` (all zero-point modes), PolarQuant values,
the embedding scalar quantizers — re-register through the same protocol. If
the interface can't express our own formats cleanly, it's wrong; fix it before
anyone external builds on it.

## 3. Mapping the named recipes onto the contract

| Recipe | Structure | Contract fit | Notes |
|---|---|---|---|
| **bitsandbytes NF4/FP4 (QLoRA)** | blockwise (64) absmax scale over a fixed 16-entry table | **affine, exact**: `μ=0`, `a=absmax/block`, `grid=NF4/FP4 table` | Closest relative of our asym-NF4; double-quant is a scale-of-scales detail inside `grid_params`. Natural **first external plugin**. |
| **LLM.int8** | vector-wise int8 + fp16 outlier *columns* | affine + our dense-sparse overlay, column-granular | Its mixed-precision outlier decomposition is the same idea as our top-2% overlay — the CSR just gets dense columns (`row_ptr` degenerates). Weight-side target. |
| **GPTQ** | per-group scale/zero int4 (Hessian-aware rounding) | **affine, exact**: `μ=zero·a`, `a=scale`, `grid=arange(16)` | Rounding cleverness lives in `compress`; decode side is plain uniform-asymmetric — already our uniform path. |
| **AWQ** | per-group scale int4 + activation-aware channel scaling | affine after folding the AWQ channel scale into `a` | Same decode contract as GPTQ. |
| **FP8 KV cache** | per-tensor (or per-head) scale, hardware e4m3 | affine with `grid = the 256 e4m3 values`, or **native-dtype passthrough** | Two modes: (a) code-space, works everywhere incl. the fused kernel (grid is just 256 entries — kernel already takes arbitrary tables); (b) `native_dtype` passthrough on H100+/Ada where the tensor cores eat fp8 directly and fusing is the hardware's job. |
| **NVFP4 KV cache** | block-16 fp4 with fp8 block scales + tensor fp32 scale | affine per block-16 (finer than per-channel: `a` becomes (H, S/16-blocked, D) — needs a small contract extension for block-granular weight) | Blackwell-native passthrough later; code-space emulation testable **today** via `ml_dtypes`. |

**The pre-registerable science claim** (house style — write it down before
running): the keys finding predicts *per-tensor-scaled FP8 keys* sit on the
fragile side for DC-offset-key families (Qwen-class), while *block-scaled
NVFP4* and *per-channel anything* sit on the robust side — because the failure
mode is per-channel scale/offset destruction, not bit width. The (A2) probe
and LongBench harness we already have can adjudicate this across recipes the
week the plugins exist. That comparison table is the marketing for the whole
plugin program.

## 4. Hardware compatibility

Current reality: NumPy CPU + CuPy CUDA (Volta+), AVX2 ADC kernel, RawKernel
fused decode. Widening, in leverage order:

### 4.1 Torch backend (highest leverage single step)

The `xp=` seam (numpy/cupy) generalizes to a small backend layer:
`Backend = {numpy, cupy, torch}`. Torch buys, in one dependency (already
optional in the tree): CUDA, **ROCm**, **Apple MPS**, **Intel XPU**, CPU — and
it is the interop plane both bitsandbytes and TensorRT-LLM live in (zero-copy
via `__cuda_array_interface__`/DLPack against CuPy). Rules:

- torch stays an *optional* extra; core remains numpy-only.
- reference paths (`kv_fused*.py` einsums) must run on every torch device —
  that alone makes decompress-then-attend + all instruments portable to
  ROCm/MPS/XPU with no kernel work.
- hardware dtypes (`float8_e4m3fn`, fp4 when torch exposes it) come from the
  torch backend, not CuPy RawKernels.

### 4.2 Portable fused kernels: Triton

Port the two kernels that matter — M2 (PolarQuant split-K) and M4
(per-channel + CSR) — to Triton, keyed off the same prepared-page structs.
Triton gives NVIDIA + ROCm from one source; the CuPy RawKernel stays as the
CUDA reference implementation and the exactness oracle. The M4 §8.5 "next"
items (batched per-page launch, packed-code dense loop) should land **in the
Triton port**, not the RawKernel, so the effort compounds.

### 4.3 Emulated dtypes for CI-without-exotic-hardware

`ml_dtypes` (numpy fp8/fp4/bfloat16) lets the FP8/NVFP4 plugins run their
conformance suites and instrument sweeps on CPU CI. Policy: **every plugin
must be conformance-testable without its target hardware**; hardware CI is for
performance numbers, not correctness gates. (This is how we keep the CI
matrix from exploding: correctness = CPU-emulated everywhere; perf = one
runner per architecture as available.)

### 4.4 CPU breadth

AVX2 ADC exists; AVX-512 and NEON (Apple/Graviton) variants are mechanical
ports behind the existing dispatch. Low priority until a consumer asks, but
the dispatch seam should be named in the backend layer now.

### 4.5 Explicitly out of scope

Competing with TensorRT-LLM's engine-level graph optimization, paged-attention
serving, or NVIDIA-tuned GEMMs. Where TRT-LLM or vLLM already executes a
format natively, our role is **upstream recipe certification and selection**
(operator_trace → recipe; certificate → confidence), and the existing vLLM
manager stays the integration point.

## 5. Sequencing

| Phase | Deliverable | Exit criterion |
|---|---|---|
| **P0** | Plugin protocol + entry-point registry + conformance kit; in-tree formats re-registered through it | in-tree formats pass their own conformance suite; zero behavior change (full suite green) |
| **P1** | Torch backend for reference paths + instruments | (A2) probe + certificate + decompress-then-attend run on CUDA/ROCm/MPS via torch; numbers match numpy to tolerance |
| **P2** | `tqp-bnb` (out-of-tree): NF4/FP4 blockwise + LLM.int8 adapters | conformance green; fused decode exactness for bnb-NF4 keys via `grid_params`; QLoRA interop demo (bnb-quantized model, our KV cache) |
| **P3** | FP8 KV + NVFP4 KV plugins (code-space first, `ml_dtypes`-emulated CI; block-granular `a` extension) | the pre-registered FP8-vs-NVFP4-vs-per-channel keys comparison published to `benchmarks/` with LongBench numbers |
| **P4** | GPTQ/AWQ weight-side adapters wired to `operator_trace` WEIGHT discipline | operator_trace recommends {AWQ\|GPTQ\|bnb} for weights and {per-channel family} for KV on an unseen architecture, end-to-end |
| **P5** | Triton port of M2/M4 (+ batched pages, packed codes) | exactness vs RawKernel oracle; runs on one ROCm target |

Dependencies: P2–P4 need P0; P1 unblocks the non-NVIDIA story everywhere; P5
is independent of P2–P4. Phases are individually shippable and each lands
behind the api-stability tiers (plugins enter **Experimental**, promote to
**Beta** on conformance + one public-data validation).

## 6. Risks

- **Contract too narrow:** something production-relevant isn't affine
  (nuq-style learned tables, vector quantizers). Mitigation: the contract
  already has the graceful degrade (no `grid_params` → decompress path), same
  as nuq today.
- **Contract too wide, too early:** freezing `grid_params` before NVFP4's
  block-granular weights are designed forces a v2. Mitigation: ship P0 with
  the block-granularity extension *specified* (weight may be (H, D) or
  (H, ⌈S/16⌉, D)) even if only (H, D) is implemented.
- **Maintenance surface:** out-of-tree means version skew. Mitigation: the
  conformance kit is versioned with the protocol; plugins pin
  `turboquant-pro>=X,<Y` and CI runs the kit, not vibes.
- **Perf-claim confusion:** users will compare our emulated FP8 to TRT-LLM's
  hardware FP8. Mitigation: `CLAIMS.md` discipline — every number tagged
  code-space vs native, with hardware.

## 7. Relation to existing work

- M4 (`DESIGN_fused_kv_decode.md` §8) supplies the affine decode contract and
  the prepared-page cache the plugins inherit.
- `operator_trace` (1.6.0) supplies the *selection* layer plugins slot into.
- `api-stability.md` tiers govern promotion; `CLAIMS.md` + the evidence ladder
  govern every cross-recipe number.
- The keys finding / (A2) probe supply the scientific frame: recipes differ in
  *which geometry they preserve*, and we are the ones who can measure that.
