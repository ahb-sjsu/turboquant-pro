# Design — the console: an instrument panel for TurboQuant Pro, with ReadScope as its microscope

**Status: Phase 0 and a Phase 1 slice LANDED on `feat/console-phase0` (2026-09-25).** Requirements:
`docs/notes/TurboQuantPro_ReadScope_Visual_Interface_Requirements.docx` (concept v0.1, September
2026; requirement ids such as UX-001 and RS-005 below refer to it). Status marks follow
`POSITIONING_2.0.md`: 🟢 shipped, 🟡 partial, ⚪ designed.

## 0. Thesis

One local console should answer both "how fast and healthy is the system?" and "why did this
query produce this result under this observer?" It shows only what a defined source measured,
with its unit, window and freshness, and it says when a number is estimated, sampled or derived.
It is read-mostly, local-first, and adds no dependency to the core package.

## 1. What already exists (the baseline the console reads)

Surveyed 2026-09-25. The console adds collection and presentation. It does not re-derive anything
the library already records.

| console entity | existing source | gap |
|---|---|---|
| Observer | `observer.ObserverContract` (`.tqo`, sha256 `digest()`, `reference()` block), schema `observer_contract.schema.json` | none for display |
| Certificate | `rank_certificate` + `certify_report` (schema `rank_certificate.schema.json`), `validity.check_validity` (VALID / STALE / UNCHECKED with reasons), `composition` chain | none for display |
| Replay | `planner.replay_plan` (`compression-plan-replay`: identity match, codec agreement, delta), `tqp replay` (`replay-report`) | no query-level replay |
| Index | `TQEIndex.stats()`, `IVFIndex.stats()`, `ShardedIndex.stats()` + manifest, `index_info` (per-section CRC32), `drift()` | no content hash or codebook identity; no per-shard bytes |
| Query | `IVFIndex` `ProbeStats`, `ADCIndex.last_survivors`, `ShardedIndex._last_shards_scanned`, `tqp query` batch latency | **no stage timing anywhere; scan path not recorded; rerank drops approximate scores** |
| Hubness | `anatomy.hub_anatomy`, `anatomy.hub_differential` (exact vs approximate neighbour lists) | none for display |
| Metrics | `monitor.QualityMonitor.metrics_dict`, `connectors.metrics.ConnectorMetrics` (the one reusable latency reservoir) | no QPS, no resource sampling, no units |
| Runtime | `hardware.detect_gpu`, `certify_report._certify_environment`, kernel flags (`_adc.is_available`, `_HAS_CUPY`, `has_triton`) | kernel choice per search not recorded |
| ExperimentRun | `claims.yaml` ledger, `benchmarks/artifacts/*` bundle | no run registry |

## 2. Decisions (the requirements' section 14 questions, answered for Phases 0 and 1)

1. **ReadScope is a workspace inside the console, on a shared shell.** Both read the same entity
   store. A separately deployable ReadScope view is a later packaging choice, not an architecture.
2. **The contract comes first (Phase 0).** `turboquant_pro/telemetry/` defines metric specs, the
   query-trace format, the entities and version negotiation, each with a JSON Schema and fixtures.
   Every panel reads only these documents; no panel parses logs.
3. **Tracing is off by default and free when off.** A disabled tracer is a module-level no-op, so
   the search paths pay one attribute lookup. Sampling is a rate in [0, 1] (NFR-003).
4. **Lossless vs sampled.** Query traces and observations used for replay are lossless when
   captured. Operational metrics (latency, QPS, resources) are aggregated in rolling windows and
   say so in their spec.
5. **Payload privacy.** Traces carry the query's sha256 and shape by default. Raw vectors are
   captured only with an explicit flag (NFR-006).
6. **No new core dependency.** The server is Python's standard library (`http.server`, threads,
   Server-Sent Events for the live stream). The UI is one static HTML/JS/CSS bundle with no build
   step, shipped as package data. React/TypeScript (the requirement's non-binding suggestion) can
   replace the bundle later against the same API. Resource sampling uses `psutil` when installed
   and otherwise reports those metrics as unavailable, never as zero.
7. **Security by design.** Bind to 127.0.0.1 by default; every request needs a per-session random
   token (printed with the URL, as Jupyter does); read-only in Phases 0–1; nothing from the
   browser is executed. Remote use goes through an SSH tunnel until auth scopes exist (API-007).
8. **Where live data comes from.** turboquant-pro is a library, so the console hosts the workload:
   it opens an index and replays a query file at a set rate, or answers searches from its own
   process. A library user can also attach the tracer to their own process and export traces.

## 3. Phase 0: the telemetry contract 🟡

- **Metric spec** (`telemetry.metrics`): name, unit, aggregation, window, source, update interval,
  kind (`measured` | `estimated` | `sampled` | `derived`), and for quality metrics the evaluation
  reference. A registry lists every metric the console can show; a panel cannot show a metric
  that is not registered.
- **Query trace** (`telemetry.trace`): query id, input sha256 and shape, parameters, index
  identity, observer reference, scan path actually taken (`kernel` | `kernel_pruned` | `numpy` |
  `exact`), and an ordered list of stage spans (`encode`, `scan`, `candidates`, `rerank`,
  `merge`), each with elapsed time, candidate count and score range. Results carry the
  approximate score, the exact score when reranked, and the rank movement.
- **Entities**: Session, Runtime, Index, Shard, Query, Observation, Observer, Certificate,
  ExperimentRun, Event (requirements section 7). Observer and Certificate embed the existing
  contract and certificate documents by reference (sha256), not by copy.
- **Version negotiation**: `GET /api/version` returns the API and schema versions and the
  capability list (API-008).

**Shipped:** metric specs and readings (`telemetry.metrics`, schema `metric_reading`), query
traces (`telemetry.trace`, schema `query_trace`), capabilities (`telemetry.api`), and
`ADCIndex.search` instrumented (every index kind routes through it). **Not yet:** the entity
schemas beyond traces and readings (Session, Runtime, Shard, Event, ExperimentRun), stage
spans inside `IVFIndex` and `ShardedIndex` (a sharded search shows one trace per shard).

## 4. Phase 1: local MVP 🟡

`tqp console --index PATH [--queries Q.npy --qps N] [--observer X.tqo] [--certificate C.json]`
opens a local page with the Overview (KPIs, latency percentiles, stage timing, mode: exact or
approximate, kernel), the live query stream and Query Inspector (stage trace, top-k with approx
vs exact, rank movement, replay of a captured query), the ReadScope workspace (observer
definition, provenance chain, certificate and validity state), and the Index panel. Keyboard
first (section 5.2 of the requirements), stale-state indicators, JSON export of the selected
context.

**Shipped:** all of the above (`turboquant_pro/console`, `tqp console`, `docs/CLI.md`).
Verified in headless Chromium against `--demo`: QPS held its 40 target, and every panel
filled from live data. **Not yet:** replay under a *different* configuration or observer
(QI-004's second half, RS-003), the compare view (QI-003 between arbitrary traces), panel
rearranging, density presets, the command palette, light and high-contrast themes.

## 5. The instrument model: an oscilloscope, and a spectrum analyzer for ReadScope 🟡

The console borrows the front panel of two mature instruments, so an operator's existing
habits carry over: what a key does, what the status line means, how a one-shot is caught.

### 5.1 Oscilloscope (time domain: the query stream)

| instrument | console |
|---|---|
| channels CH1–CH4, Tek colours (yellow, cyan, magenta, green) | per-query signals from traces: `latency` (total ms), `encode`, `scan`, `rerank` (stage ms), `candidates`, `agree` (rerank agreement), `move` (largest rank movement), `err` (approximate minus exact score of the top result) |
| vertical scale (units/div), position, on/off | per channel; 8 vertical divisions |
| time base (s/div), horizontal position | over query arrival time; 10 horizontal divisions; roll mode when the sweep is slower than 1 s/div |
| trigger: edge (level, slope), pulse width (above level for N queries), logic (conditions ANDed, including non-numeric ones such as `scan_path == numpy`) | `console.scope.Trigger` |
| trigger modes Auto, Normal, **Single**; holdoff; trigger position (pre-trigger %) | Auto sweeps without a trigger; Normal updates only on one; Single arms, captures one record around the trigger event, and stops; that is how one-shots are caught |
| Run/Stop, Single, Force trigger | Space, `s`, `f` |
| acquisition: Sample, **Peak detect**, Average | Peak detect keeps each screen column's min and max, so a one-query spike survives any time base: a fast event is slowed to a visible one |
| **digital phosphor persistence** | a time × value hit histogram with exponential decay (or infinite): how often a value occurs is shown by intensity, so rare events are dim, not invisible |
| automatic measurements + statistics | per channel: mean, min, max, pk-pk, σ, p50/p95/p99, with count over acquisitions |
| cursors | two time cursors and two value cursors with Δ readouts |
| segmented memory / FastFrame / history | every triggered record is kept as a segment; step through them, and open any query in the inspector |
| mask / pass-fail test | limit lines from an SLA or a certificate floor; violations counted; optional stop-on-fail |
| Autoset, Save/Recall setup | scales from the data; a setup file (channels, trigger, time base) saved alongside the observer contract |

### 5.2 Spectrum analyzer (the ReadScope instrument: what the observer reads)

x is the eigen-direction index of the observed space (the "frequency"), y is energy in dB.
Traces: corpus variance σᵢ², the observer's sensitivity λᵢ (`read_operators`), the
quantization noise per direction, and the water level θ of `read_allocation` as the
reference line. Max-hold and min-hold traces, markers with peak search and Δ markers, limit
lines (a certificate's floor as a mask), and a waterfall of the spectrum over time, which is
drift made visible. A time-domain FFT of any scope channel finds periodic interference in the
query stream (collection pauses, thermal cycles).

**Shipped (2026-09-25):** `console.spectrum` (engine) and `console.spectrum_view`
(screen), `v` from the scope. Pinned by tests: the noise trace sums exactly to
`realised_distortion`, the predicted trace equals `min(w, theta)` in every direction (so
the water level is the right limit line: a direction over it is one where the codec does
worse than the optimal allocation of the same bits would there), and water-filling spends
exactly the budget. The session's read operator follows the last 256 workload queries;
the budget is the index's stored bits per vector; the readout gives the overall gap
between realised and predicted distortion in dB.

### 5.3 Build order

1. `console.scope`: the acquisition engine, pure and tested (channels, trigger, peak detect,
   persistence, measurements, segments, masks). 2. The scope screen in the terminal UI. 3.
   The spectrum analyzer view. The Overview panels remain as one view of several.

## 6. Out of scope here

Everything the requirements put out of scope for the MVP, plus operator actions (Phase 3):
the console does not change an index, a quantizer or a production observer.
