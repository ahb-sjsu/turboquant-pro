# Code Quality & the SQLite-Grade Roadmap

turboquant-pro's north star is to be the **"SQLite of embedding/model
compression"**: ubiquitous, zero-config, rock-solid, permissively licensed. That
bar is about **reliability, format stability, and trust** — not just clever
compression. This document tracks where we are and what's left, and is
re-measured each release.

## How we measure

We run the [Prometheus](https://github.com/ahb-sjsu/prometheus) complexity-fitness
analyzer and compare against **SQLite's own source** as the gold-standard
baseline.

```bash
prometheus turboquant_pro --library          # the shipped library
git clone --depth 1 https://github.com/sqlite/sqlite /tmp/sqlite
prometheus /tmp/sqlite/src --library         # the baseline
```

## Current state (measured 2026-06-20, Prometheus)

| Metric | `turboquant_pro/` | SQLite `src/` (baseline) |
|---|---:|---:|
| Source files | 22 | 149 |
| Total LOC | 8,672 | 205,602 |
| **Avg cyclomatic complexity** | **3.29** | 4.28 |
| Observability score | 100/100 | — |
| Fitness quadrant | FORTRESS | FORTRESS |
| "LOC per feature" flag | 1971 🔴 | 1490 🔴 |

**Read this correctly.** By per-function complexity — the best proxy for
"tightness" — the library is already *tighter than SQLite* (3.29 vs 4.28; with a
Python-vs-C caveat, since C scores higher CC for equivalent logic). The scary
**"over-engineered / LOC-per-feature"** verdict is a **metric artifact**: SQLite,
the literal gold standard, earns the *same* flag, because the heuristic expects
app-style "features/endpoints" and mis-reads any focused library. We do **not**
act on it. The code is tight; the real work is elsewhere.

## SQLite-grade roadmap (the real bar)

Prometheus measures complexity + resilience; SQLite's actual superpowers are
things it can't see. Prioritized:

### P0 — Repo hygiene
- [ ] Move `experiments/` (research WIP) out of the working tree into a separate
  repo. It contributes *every* complexity hotspot and *every* network-timeout
  vulnerability Prometheus found, dragging full-repo avg CC 3.29 → 4.63. A
  trustworthy standard ships a clean tree.

### P1 — Format stability (the #1 standardization lever)
- [x] Freeze and **document a versioned compressed-format spec** (magic +
  version byte, bit-packed codes, codebook, preserved L2 norm) —
  [`docs/FORMAT_SPEC.md`](docs/FORMAT_SPEC.md), implemented as TQE1 v1/v2 in
  `turboquant_pro/format.py`.
- [x] Compatibility guarantee: data compressed by vX stays readable on vY —
  the version prefix is permanent, readers reject unknown versions, and a new
  rotation field was added as v2 with v1 kept byte-identical (see the spec's
  "Versioning & compatibility policy").
- [x] A **conformance test suite** so a third party could implement a reader
  from the spec alone (`tests/test_format.py`: round-trip, self-describing
  header, bad-magic / unknown-version / truncation, mixed v1+v2 batches). Golden
  cross-language fixture files remain a nice-to-have. This is the SQLite
  file-format discipline, and it is what makes stored compressed data
  trustworthy forever.

### P1 — Test rigor
- [ ] Coverage target (e.g. ≥95% line / ≥90% branch on `turboquant_pro/`),
  enforced in CI. (For the exact current suite size run `pytest -q --co | tail -1`;
  the README's version-history table snapshots the count per release. Avoid
  pinning a fixed number in prose — it drifts.)
- [ ] Property-based tests (Hypothesis) for compress→decompress round-trip
  invariants and norm preservation.
- [ ] Fuzz the decoder against malformed/adversarial input — it must never crash
  or read out of bounds.
- [ ] Cross-platform × cross-Python stability matrix in CI.

### P2 — I/O resilience (the one *real* Prometheus finding)
- [ ] Timeouts on all network calls in the NATS / pgvector / remote-FAISS
  integrations (Prometheus: Retries 8/100, Circuit Breakers 0/100 — SQLite
  sidesteps this by doing no I/O at all).
- [ ] Retry-with-backoff where a transient failure is genuinely recoverable.

### P2 — API & versioning guarantees
- [ ] Documented stable public API surface + semantic-versioning policy.
- [ ] Deprecation policy.

## Tracking

Re-run the commands above each release and update the table. Targets: hold
`turboquant_pro/` avg cyclomatic complexity **< 5**, and drive the roadmap boxes
to done — format stability and test rigor first, since those, not code size, are
what turned SQLite into infrastructure.
