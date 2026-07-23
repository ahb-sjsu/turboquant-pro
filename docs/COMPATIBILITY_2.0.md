# 2.0 compatibility matrix

**The rule: "supported" means "included in CI."** Anything not listed here is
*expected to work* at best, and the connector's identity profile will refuse
to reuse KV across any configuration it cannot verify (uncertain ⇒ miss).

This file is the published matrix required by the 2.0 roadmap's compatibility
policy; it is updated in the same commit as any CI-lane change, so the matrix
can never claim more than CI proves.

## Current state (2.0.0-alpha, honest)

| Component | Supported (in CI) | Notes |
|---|---|---|
| Python | 3.10, 3.11, 3.12 | full test matrix, every push |
| numpy | ≥1.24 (CI installs latest) | core dependency |
| vLLM | **none yet** | connector ships as a protocol-verified scaffold; the pinned engine lane is the alpha→beta gate. Target pins: exact minors, starting `0.9.x`, because the V1 connector API is experimental upstream |
| SGLang | **none** (release 2.1) | will target the public HiCache configurable storage-backend surface, own pinned lane |
| PyTorch | optional extra; not in the engine lane yet | `[torch]` powers `tqp trace` paths tested in CI |
| CUDA / ROCm | **none** (CPU CI only) | GPU paths (CuPy/Triton) are Experimental-tier |
| GPU architectures | **none in CI** | hardware profiles exist (Volta→Blackwell) but are not CI-verified |
| Model families | **none engine-verified** | the P1-M6 matrix (MHA/GQA/MQA/MLA/RoPE-scaling/sliding-window/TP/LoRA/spec-decode/multimodal/chunked-prefill) populates with the engine lane; unsupported combinations are detected and rejected, not assumed |
| Attention layouts | (B,H,S,D) per-layer KV assumed | the store normalizes 3-D/5-D to it; layout version is an identity-profile field |
| PostgreSQL | **none in CI** | Track A (compressed-storage bridge) is tested against psycopg2 mocks only; live-PG lane is a Track A 2.0 upgrade |
| DuckDB | 1.5.x (local test suite; CI skip-if-absent) | promote to CI-installed before beta |
| pyarrow | 23.x with DuckDB suite | same lane as DuckDB |
| TQE reader | TQE1 v1 + v2 records | golden corpus enforced in CI on every push (format break ⇒ red) |

## Lane-addition checklist

Adding a row = adding CI, in this order:
1. Pin exact versions in a dedicated workflow job (no ranges for experimental APIs).
2. Wire the smoke/conformance test the row claims.
3. Update this file in the same commit.
4. For engine lanes: record the pin in the identity profile's
   `kv_layout_version`/`attention_backend` vocabulary so reuse is refused
   across unverified combinations by construction.

## Deprecation policy

A row leaves the matrix one minor release after its CI lane is removed, with a
CHANGELOG entry. The TQE1 reader row never narrows within a major (see the
format compatibility promise in `ROADMAP_2.0.md` P2-M4).
