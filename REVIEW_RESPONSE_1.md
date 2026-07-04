# Response to Review 1 — action plan

The review is positive ("serious tool with several publishable ideas") and the asks are about
**evidence organization, claim calibration, metadata, and production hygiene** — not correctness.
Below: each concern, our action, and owner (✎ = safe edit done/doable now; ⧗ = larger work).

## Major concerns

**1. Clearer evidence ladder. ✎/⧗**
Add explicit reproducibility labels to every benchmark: *Colab-reproducible / public-data+local-HW /
private-or-large-scale / paper-not-yet-reproduced / engineering-unit-tests-only*. Done: added
`docs/claims.md` (an evidence ladder mapping each headline claim to its level + where to reproduce).
Remaining ⧗: annotate each in-line benchmark in the README with a one-word label.

**2. Separate embedding vs. KV-cache. ✎/⧗**
Agree there are two tools/papers in one package. Done now: `docs/api-stability.md` and `docs/claims.md`
split the two tracks explicitly. Remaining ⧗ (structural, recommended): two top-level README benchmark
sections, two "what this is not" boxes, and eventually two paper abstracts — the raw material already
exists (`docs/KV_KEYS_FINDING.md`, `benchmarks/`).

**3. Tighter headline wording. ✎ (done)**
The 27× headline already carried "(with 5× oversampling + reranking — all methods benchmarked
identically)". Fixed the other one: the **114×** figure is now scoped as *storage-only, dataset-dependent,
recall preserved via oversampling+reranking* rather than an unqualified maximum.

**4. Inconsistent test-count / version metadata. ✎ (done)**
Worse than noted: README said 489, the release table 497, GitHub "About" 397 — and the actual
`def test_` count is **445**. Per the recommendation, removed the static "489 tests" from the README
headline; the **CI Tests badge is the single source of truth**. (GitHub "About" text and the PyPI
long-description headline must be updated by hand — flagged below.)

**5. API surface too broad → stability tiers. ✎ (done)**
Added `docs/api-stability.md` with the recommended tiers: **Stable** (PCAMatryoshka, embedding
compression, basic TurboQuantKV, TQE1 format), **Beta** (ADCIndex, TurboQuantKVCache, FAISS/pgvector
wrappers), **Experimental** (CUDA fused decode, vLLM manager, weight compressor, PostgreSQL extension,
NATS transport).

## Benchmarking / documentation recommendations ⧗
- Canonical embedding table (flat/PQ/OPQ/RaBitQ/PCA-only/TQ-only/PCA+TQ/ADCIndex/FAISS/pgvector) with
  identical rerank candidate counts; canonical KV-cache table (fp16/uniform-K4/NF4/asym-NF4/+outliers/
  KVQuant) with model versions, prompts, context/gen lengths, batch, hardware, exact commands.
- README hierarchy split: `README.md` (install + 60-sec examples + one summary), `docs/embeddings.md`,
  `docs/kv-cache.md`, `docs/benchmarks.md`, `docs/claims.md` (added), `docs/api-stability.md` (added).

## Production / security checklist ⧗
- Wheels or explicit AVX2/CUDA/Rust build instructions; CI matrix (Py 3.9–3.12, CPU-only, optional
  FAISS, optional GPU); property tests for compress/decompress round-trips + bit packing;
  endian/version-compat tests for the TQE1 format; deterministic-seed reproducibility tests; benchmarks
  that fail loudly when fast kernels are missing; a security note for loading serialized artifacts; a
  perf-regression dashboard. (These are real work; sequenced for the next minor release.)

## Positioning ✎ (done)
Adopt the reviewer's framing — a *research-to-production compression toolkit* with **two** contributions
(PCA-reordered dims + scalar quant for high-recall compressed retrieval; architecture-aware
per-channel/asymmetric KV-key quantization) — rather than "one method for everything." Added a
two-contribution positioning line to the README.

## Needs the author (outside the repo)
- Update the **GitHub "About"** blurb (still says 397 tests → remove the number, or match CI).
- Update the **PyPI long-description** headline if it repeats a static test count.
