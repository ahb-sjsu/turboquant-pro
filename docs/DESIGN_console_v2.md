# Design — console v2: every feature of TurboQuant Pro, visible

**Status: DESIGN, for the owner's review (2026-09-26). No code has changed yet.** Branch
`feat/console-v2`, off `origin/master` at 24d49ac. Requirements:
`docs/notes/TurboQuantPro_ReadScope_Visual_Interface_Requirements.docx` (concept v0.1) and its
mockup. This document builds on `docs/DESIGN_console.md` (Phase 0 and a Phase 1 slice, shipped)
and does not replace its decisions.

The appendices in `docs/console-v2/` are the full surveys this design rests on, one row per
feature of the library, read from master at 24d49ac:

- `survey-retrieval.md`: search engines, codes and formats, bit allocation, rerank, geometry and
  hubness, planning and benchmarks, caches and connectors (about 45 rows).
- `survey-models.md`: KV cache, kernels, weights, tracing and (A2), runtime policy, serving,
  plugins, model benchmarks (about 50 rows).
- `survey-evidence.md`: observers, certificates and validity, compose, capabilities,
  feasibility, planning records, claims, schemas, the existing console, and an artifact inventory
  of about 30 kinds.

## 0. What the owner asked for

1. The console should look like the requirements mockup, not like the current sparse web page.
2. It should visualize **every** feature of the library, including PostgreSQL / pgvector, the CLI,
   PyTorch and Hugging Face, llama.cpp, and the rest.
3. Both surfaces, **the terminal UI first**. The web view stays opt-in behind `--web`.
4. A panel with no real data **measures or says why**. Nothing in the mockup is copied as data.

## 1. Constraints kept from the shipped console

- The TUI is the default and never opens a browser (the owner: "is btop a web app?").
- `server.py`'s security design stays whole: bind to 127.0.0.1, per-session token in the URL
  fragment, Host-header check, read-only endpoints, the CSP, `textContent` only.
- The telemetry contract stays the only source for panels. Every number carries unit, window,
  source and kind (measured, sampled, derived, estimated). Unavailable is shown with its reason,
  never as 0 (`dumps` maps NaN to null).
- No new core dependency. `psutil` and NVML are optional and reported as unavailable when absent.
- Everything already built is kept: the oscilloscope (triggers, peak detect, phosphor, segments,
  masks, periodogram), the spectrum analyzer (water level, reference and delta, waterfall), `.tqs`
  setups, the query inspector with replay, and JSON export.

## 2. What the survey found before any panel is drawn

Each of these would make a prettier console show a false picture, so each is fixed or shown.

**In the shipped console (fixed on this branch):**

1. **`tqp console --index` is silent on real indexes.** It opens the index memory-mapped, and
   `TQEIndex.search` then takes its blocked scan, which never reaches `ADCIndex.search`, the only
   instrumented path. Only `--demo` produces traces. Sharded indexes likewise.
2. **`--rerank` with `--originals` throws on every query.** `server.py:104` passes `originals=` to
   `TQEIndex.search` and `ShardedIndex.search`, which take no such argument.
3. **Certificate validity is never computed.** The console shows the issue-time `validity`
   section, which has no `status`, so every certificate reads UNCHECKED, although the console holds
   the originals and queries `validity.check_validity` needs. The UIs also lack the fourth state,
   INCONCLUSIVE.

**In the library (reported to the owner; outside the console's files):**

4. **The Hugging Face KV drop-in makes decode-time keys about 7x larger than fp16.** Measured on
   Atlas CPU with the cache's settings (`PerChannelKV(128, 8, bits=4, nf4_asym=True,
   outlier_frac=0.02)`): a 1-token spill, which is every decode step once the hot window is full,
   costs 1,856 B per token per head against 256 B for fp16, because the outlier rule keeps every
   entry when S = 1. A 256-token spill costs 83 B. Script: `docs/console-v2/kv_spill_bytes.py`.
   **Fixed on master in #248 (8e252ec)**: the cache now spills in blocks (`spill_block`, default
   max(hot_window // 2, 64)), reproduced with the real cache at 7.25x before and 0.36x fp16 after
   for keys. The codec's outlier rule is deliberately unchanged, since the keys campaign pins it,
   so the Models panels should still show bytes per cold token per chunk, which would reveal a
   regression to small spills.
5. From reading only, not yet measured: the HF drop-in compresses on the CPU, rebuilds dense K/V on
   the device every step (so the saving is host RAM, not device memory), compresses nothing under
   512 tokens, and grows step time with the count of cold chunks. `memory_stats()` ratios use an
   fp32 baseline where other paths use fp16.
6. Only `ADCIndex.search` is instrumented. TQEIndex, IVF, sharded, HNSW, FAISS, adaptive rerank,
   tiered rerank, distributed search and the whole model path emit no telemetry.
7. Only 11 of about 30 artifact kinds have a JSON Schema, and almost none record the command that
   produced them. All 24 CLI writers pass through `cli._emit_doc`.
8. No real llama.cpp or GGUF integration and no NATS KV transport exist. The llama example feeds
   random K/V; the GGUF runner is a labelled baseline.

## 3. The shape: sources feed one contract, workspaces read it

```
 sources (read-only adapters)          telemetry contract            surfaces
 ─────────────────────────────         ──────────────────           ─────────────
 demo index                   ─┐       metric readings    ─┐       TUI (default)
 index file: TQE, IVF, sharded │       query traces        │       web (--web)
   HNSW, FAISS                 ├──────▶ model/kv events     ├──────▶ export / setups
 pgvector table (DSN from env) │       entities            │
 DuckDB / Arrow                │       artifacts by schema ─┘
 HF / PyTorch model + cache    │
 vLLM connector counters       │
 llama-server /metrics,/slots  │
 tqp artifact files           ─┘
```

A **source** is an adapter with one job: turn what the library already computes into readings,
traces, events and entities. The console attaches one or more (`--index`, `--demo`, `--model`,
`--pgvector`, `--open FILE`). A workspace reads only the contract, so the same panel works for a
demo, a real index or a model, and a panel whose source is not attached says so.

## 4. Workspaces and their panels

Status: 🟢 data exists today, 🟡 small change (hours), 🟠 medium (days), ⚪ not built in the
library (the panel says so). Requirement ids from the requirements document.

### 4.1 Overview (the mockup's screen)

The top-level screen, in the mockup's grid, filled from whatever sources are attached.

| Panel (mockup name) | What it shows | Data | Status |
|---|---|---|---|
| Header | product, version, tagline, attached sources, observer badge, freshness, PAUSED | capabilities, session | 🟢 |
| System Overview | mode, dataset rows, dim, compression, index size, QPS, quality metric, latency, then block meters for CPU, memory, disk I/O, GPU | readings; GPU via NVML when present, else "unavailable: no NVML" | 🟢 / 🟡 GPU |
| Query Throughput | per-second bar histogram, gradient-coloured, current / peak / avg | `search.qps` history | 🟢 |
| Quality strip (mockup "Recall@10") | the source's own quality signal against its reference: rerank agreement vs exact rescoring for an index, attention KL on probe layers for a model | readings, labelled with reference | 🟢 index / 🟠 model |
| Quantization | method, code size, bits, rerank on/off, exact mode available, score delta, kernel, training needed | index entity, effective cache config | 🟢 / 🟡 |
| Vector Space (mockup "t-SNE") | PCA-2 projection of a row sample in braille dots, coloured by IVF cell or cluster, the latest query and its neighbours marked, labelled "projection, not the geometry" (VS-003) | new `/api/projection` | 🟡 |
| Index Shards | one row per real shard or IVF list: rows, bytes, QPS, quality, status; a single in-process index shows one row, not eight | sharded stats; IVF cell counts | 🟡 |
| Pipeline | stage boxes with mean ms: encode, scan (kernel path), cold fetch, rerank, results; for a model: prefill, decode, spill, materialize | traces; model clock | 🟢 index / 🟠 model |
| Compression vs Baselines | measured in-process on this workload at start: exact float, the source's compressed path, with rerank; FAISS rows only when faiss is installed; a pgvector table's exact search when attached | new `/api/baselines` | 🟡 |
| Observability (mockup checklist) | observer contract, certificate validity (four states), deterministic replay (last result), exact mode, hubness report, health, each ✓ / ✗ / — with a word, never all green by default | readscope, validity, replays | 🟡 (validity) |

### 4.2 Queries (QI-001 to QI-006)

Stream table, inspector, replay (all kept). Adds: a per-query stage waterfall for **every** index
type (needs tracing TQEIndex, IVF, sharded, HNSW, FAISS, adaptive and tiered rerank: 🟠), IVF
probe stats, adaptive-rerank stage strip against the fixed k·r line, escalation badges.

### 4.3 Models (new; the first live source)

| Panel | What it shows | Status |
|---|---|---|
| Effective config and mode | codec per tensor role, host codec vs fused kernel, plan vs running diff | 🟡 |
| KV tier and bytes per layer | hot vs cold tokens, three baselines (fp16-equivalent, host compressed split by component, device transient); this is the panel that shows finding 4 | 🟡 |
| Decode timeline | tokens/s and ms/step split into model compute, cold decompress, spill compress, with cold-chunk count | 🟡 |
| Spill waterfall | step x layer, chunk size, bytes, ms | 🟡 |
| Attention KL heatmap | layer x head KL against an fp16 shadow of 1 to 3 probe layers, sampled; the metric the project accepts for keys | 🟠 |
| DC-offset heatmap | head x channel zero point, RoPE long-wavelength channels marked | 🟡 |
| Operator anatomy | `tqp trace` regimes and disciplines per tensor, fx confidence | 🟢 |
| (A2) probe | Spearman polar vs per-channel per layer, tangential fraction | 🟠 |
| Runtime policy | decision lanes, measured value against floor, reason; labelled advisory, since nothing in generation calls it | 🟡 |
| Weight plan | layer x matrix bits, budget, dual-bound gap | 🟢 artifact |
| Serving | vLLM connector hits, misses by cause, latency percentiles, logical vs physical bytes | 🟡 |
| llama.cpp | `llama-server` `/metrics` and `/slots` read-only, labelled "llama.cpp's own cache, not TurboQuant" | 🟡 |

### 4.4 ReadScope (RS-001 to RS-008)

Observer card and badge, certificate card with the four-state validity checklist and mu(kappa)
curve, provenance graph linking arrays, contracts, certificates and plans by sha256, composition
chain, capability board, feasibility verdict with the rate-distortion curve, refinement and
cross-observer compatibility heatmap, the spectrum analyzer (kept), and replay (kept, extended to
another observer when the library supports it: ⚪ today).

### 4.5 Vector Space (VS-001 to VS-005)

Neighbour inspector (approximate vs exact), k-occurrence histogram with hub and anti-hub lists and
the mechanism quadrant, recall by count decile against the aggregate line, STRATA small multiples
with ABSTAIN hatched (never green), drift view, projection (4.1).

### 4.6 Index & Shards (IS-001 to IS-005)

TQE header (live vs tombstoned, bytes per vector, format version, section treemap with CRC state),
IVF occupancy histogram with Gini and p99/median, shard grid with skew, HNSW layer pyramid and
memory bar, FAISS size bar (labelled estimated), distributed fan-out when workers answer, pgvector
table size and calibration operating point. Actions stay read-only (IS-005).

### 4.7 Benchmarks (BM-001 to BM-005)

One Pareto component (recall against bytes per vector, frontier, budget, certified points) shared by
plan embeddings, autotune, the query catalog and pgvector calibration; the plan funnel and frontier
with intervals (ABSTAIN shown as an answer); the claims ledger with status chips (2 of 17
executable, retracted kept visible); the kvquant matrix heatmap; kernel latency curves with the
running session's point overlaid. Every panel shows "reproduce with", recorded or derived.

### 4.8 Artifacts (new)

Open any `tqp` output by its `schema` id (or field shape for the four schema-less outputs), validate
against its JSON Schema when one exists ("no schema shipped" otherwise, never "valid"), render with
the panel above, link by sha256, compare two of a kind, export a bundle. Never execute an
artifact's command.

### 4.9 System and Settings

Runtime card (device via CuPy and torch, arch, memory), NVML utilization, power and temperature,
kernel inventory (compiled, Triton available), plugin registry and conformance grid, cache hit
rates, connection profiles (pgvector DSN from the environment, never shown), sampling rate, themes.

## 5. TUI layout

The mockup maps onto a btop-style grid at 160 x 48 and degrades to 80 x 24 by collapsing panels
(UX-001). Rendering stays curses with 256 colours, using:

- block meters `▏▎▍▌▋▊▉█` for resources, eighth-block resolution;
- `▁▂▃▄▅▆▇█` bar histograms with a purple to cyan to green to yellow gradient for throughput;
- braille (2 x 4 dots per cell) for the projection scatter and quality strips;
- box-drawing panel frames with a coloured title, a number key, a freshness dot and a kind glyph.

Colour roles follow the requirements' Appendix B: cyan live, green pass, amber stale or warning,
red fail, purple observer and comparison, grey unavailable. Status is never colour alone; each
carries a word or symbol (NFR-008).

A pure `Canvas` of cells and attributes sits under curses, so a frame can be rendered to text, ANSI
or HTML for tests and documentation screenshots without a terminal.

## 6. Phases

Each phase ends with tests passing on Atlas (never the laptop) and a screenshot per changed screen.

1. **Fix and instrument.** The three console bugs; trace TQEIndex (blocked and kernel paths) and
   ShardedIndex; run `check_validity` for a loaded certificate; the `invocation` block in
   `cli._emit_doc`; a schema registry. Touches `index.py`, `sharded_index.py` and `cli.py`, which
   are outside the console, so each change is agreed with the session that owns them first.
2. **TUI Overview v2** in the mockup's grid, driven by `--demo` and a real TQE index, with the
   projection and baselines endpoints.
3. **Models source and workspace** on a tiny HF model on Atlas CPU (about 30M parameters, 20
   threads, the thermal cap), then on a GPU only when the owner decides the keys campaign can
   yield GPU 1 or NRP is used.
4. **Artifacts workspace** and the ReadScope cards (certificate, provenance graph, plan funnel).
5. **Vector Space, Index & Shards, Benchmarks** panels, pgvector and DuckDB sources.
6. **Web view** brought to the same layout, still opt-in.

## 7. Decisions needed from the owner

1. Phase 1 edits library files outside the console (`index.py`, `sharded_index.py`, `cli.py`).
   Agree that this branch may carry them, or route them through the owning session.
2. The HF drop-in's 7x decode inflation (finding 4) is fixed on master (#248). Still open and
   the owner's call (listed in #248): the dense rebuild every step, so the saving is host RAM and
   not device memory; compression on the CPU; nothing compressed under `hot_window`; and
   `memory_stats()` reading against an fp32 baseline. The Models panels show each as it is.
3. A GPU for the model source: keep to CPU until the keys campaign ends, or pause it, or use NRP.
4. A pgvector instance for Phase 5: an existing one on Atlas, or a container stood up there.
