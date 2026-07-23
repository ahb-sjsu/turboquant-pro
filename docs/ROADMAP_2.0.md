# TurboQuant Pro 2.0 — the platform release

Three pillars, one thesis: **1.x proved the compression is certifiable; 2.0 makes it
infrastructure** — something a cluster operator enables with a flag, a file format other
tools read, and a SQL surface enterprises already know how to operate.

Status legend: 🟢 shipping on `master` · 🟡 scaffolded/designed · ⚪ designed only.

---

## Pillar 1 — Native vLLM & SGLang KV connectors 🟡

**Goal:** `vllm serve meta-llama/... --kv-transfer-config
'{"kv_connector":"TurboQuantConnector", ...}'` — swap standard FP8/INT4 KV caching
for turboquant-pro **without forking vLLM**, via the first-class V1 connector
interface (`KVConnectorBase_V1`), scheduler + worker roles.

**2.0 MVP semantics — the offload/persistence tier.** The connector quantizes KV
blocks on save (per-channel keys / polar values — the (A2)-correct disciplines) into
a TurboQuant block store (CPU RAM and/or TQE1 spill files), and restores them on
prefix-cache miss. What that buys a production cluster:

- **Bigger effective prefix cache**: ~4–5× more cached prefixes per GPU-adjacent
  byte, at the disciplines that provably preserve attention behaviour
  (`docs/KV_KEYS_FINDING.md` — acceptance on perplexity/agreement, never cosine).
- **Cross-restart / cross-node prefix reuse**: quantized blocks are TQE1 records —
  persistable, shippable, hash-verifiable.
- Behavior gated the tqp way: the connector exposes its (A2) probe so a deployment
  can *certify* the discipline choice per model before enabling it.

**Shipped in this commit:** `turboquant_pro/connectors/vllm_v1.py` —
`TurboQuantKVConnector` implementing the V1 protocol surface (late-bound against the
installed vLLM; degrades to an importable, testable shim without vLLM), plus
`register()` for `KVConnectorFactory`, block store, and protocol-shape tests that run
without vLLM installed.

**Milestones to 2.0-final:**
1. ⚪ CI lane with real vLLM pins (`vllm>=0.9`) running an end-to-end smoke
   (tiny model, save→evict→reload, agreement check vs. uncompressed).
2. ⚪ SGLang connector: same block store behind SGLang's hierarchical-cache
   (HiCache) host-offload hooks; the store is engine-agnostic by design.
3. ⚪ Fused dequant-into-attention on load (reuse the Triton fused-decode kernels)
   so restore cost is a kernel, not a round-trip.
4. ⚪ `tqp plan kv --connector` emits a ready `--kv-transfer-config` JSON from the
   (A2) probe verdict.

## Pillar 2 — TQE1 as the interchange format for compressed tensors 🟡
(items 2–3a shipped: golden corpus + single-file Python reader, conformance-tested)

**Goal:** position TQE (record layout in [`FORMAT_SPEC.md`](FORMAT_SPEC.md), v3
bit-packing) as the **GGUF/safetensors equivalent for *compressed* KV blocks and
Matryoshka embeddings** — a format other runtimes read without this library.

What "standardize" concretely means here (in order):
1. **Freeze + version the spec as an RFC-style document** (`TQE1-SPEC v1.0`):
   magic, endianness, record framing, quantizer-parameter blocks, integrity hashes —
   normative MUST/SHOULD language, with the existing conformance kit promoted to a
   **format conformance suite** any third-party reader can run against golden files.
2. **Golden corpus**: small committed `.tqe` files + their exact decoded tensors,
   the cross-implementation test target (the safetensors playbook).
3. **Reference readers**: (a) a dependency-free single-file Python reader
   (~200 lines, stdlib+numpy) usable without turboquant-pro; (b) a Rust reader
   crate seeded from the pgext `src/` code — one decoder shared by pgext and DuckDB.
4. **KV-block profile**: TQE1 today frames embedding records; the KV connector
   (Pillar 1) writes a `kv_block` record profile (layer, head, position range,
   discipline id) — spec'd in the same RFC so a saved KV tier is portable across
   engines.
5. Registration niceties: reserved magic + suffix (`.tqe`), a `tqp format
   validate` command, and spec badges for third-party readers ("reads TQE1 v1.0").

## Pillar 3 — Enterprise vector stores: pgvector-class SQL over compressed indexes ⚪

**Goal:** query turboquant-pro compressed indexes **directly via standard SQL**,
without decompressing the corpus into host RAM first.

- **PostgreSQL (`pgext/`, exists as `tqvector` skeleton):** expand to (1) a table
  access path over memmapped TQE shards (compressed-domain ADC scan in the
  extension, top-k pushdown), (2) `tqe_search(index, query, k)` set-returning
  function, (3) operator-class integration so `ORDER BY embedding <-> :q LIMIT k`
  plans through the compressed index. Rust core shared with the Pillar-2 reader.
- **DuckDB:** a Python/Arrow extension module (`turboquant_pro.duckdb_ext`)
  registering `tqe_search('x.tqe', :q, k)` and `tqe_scan('x.tqe')` table functions
  that stream **compressed blocks → Arrow batches** (block-at-a-time ADC scoring —
  the 1.9.0 block-streamed search behind a SQL face). Zero-copy into DuckDB's
  pipeline; RAM bounded by block size, not corpus size.
- Acceptance stays coherent: both surfaces support `WITH (RECALL >= r)`-style
  planning from a stored calibration catalog (the `tqp query` machinery — 1.9.1's
  ANALYZE catalogs are the planner input here too).

## Versioning & compatibility promises for 2.0

- 2.0 is **additive at the API level**: everything Stable in 1.9.x remains; the
  connector/format/SQL work introduces new modules, not breaking changes. The major
  bump marks the *scope* change (library → platform) and the TQE1 spec freeze.
- The format-stability rule hardens: post-2.0, TQE readers MUST open every 1.9+
  file forever; writers may add record profiles only via the RFC's extension
  mechanism.

## Sequencing

`2.0.0-alpha`: Pillar 1 MVP (this commit) + Pillar 2 items 1–2.
`2.0.0-beta`: vLLM CI lane green; single-file reader; DuckDB table functions.
`2.0.0`: SGLang connector, pgext ADC pushdown, spec v1.0 frozen + golden corpus,
model-card-grade benchmark evidence for the connector (agreement + throughput).
