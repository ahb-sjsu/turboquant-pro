# Geometry-aware retrieval fuzzing

`tqp fuzz` searches for query mutations whose exact nearest-neighbor ranking differs from the ranking produced by a persisted TurboQuant-Pro index.
The MVP is deliberately query-only: the corpus and index remain immutable, and every mutation receives newly recomputed exact top-k truth.
This makes a retained finding reproducible without claiming that an ordinary retrieval discrepancy is a formal certificate failure.

## Profile the immutable corpus

Create the profile once, using the same original vectors that were used to build an index which retains originals.

```powershell
tqp geometry profile --embeddings corpus.npy --k 10 --sample 10000 --seed 42 --out geometry.json --format json
```

The profile estimates sample covariance in float64 and records its unregularized rank.
It uses an eigendecomposition to floor every eigenvalue at `max(1e-12, largest_eigenvalue * 1e-8)` before building the whitening transform.
`covariance.singular` therefore describes the original covariance, while `covariance.regularized`, `regularized_dimensions`, and `eigenvalue_floor` describe the finite transform actually used.
This means rank-deficient corpora and corpora with fewer rows than dimensions can be profiled safely, but the profile never conceals singularity.

The profile also stores a fixed-seed sampled exact reverse-kNN estimate, exact-neighbor margins, spectral and effective-rank summaries, and central-hub, peripheral-hub, and central-anti-hub strata.
Its coverage quantile edges are frozen in `geometry.json` and are not re-fit while fuzzing.

## Run a deterministic campaign

```powershell
tqp fuzz retrieval --index corpus.tqe --queries queries.npy --geometry geometry.json --mutators radial,shell --budget 200 --seed 42 --k 10 --out fuzz-run --format json
```

The index must be a single TQE index with stored original vectors and must have the same corpus hash as the profile.
An optional `--truth` artifact is recorded only as provenance because precomputed truth is stale for a mutated query.

`radial` applies `z' = alpha * z` in the profile's regularized whitened space.
`shell` changes angular placement while preserving each whitened norm.
Both mutations are seeded deterministically, and each candidate is evaluated against fresh exact top-k results and the actual TQE search path.

Coverage uses four frozen signals: Mahalanobis centrality, reverse-kNN hubness, exact-neighbor margin, and one minus recall at k.
The campaign deterministically retains a candidate that reaches a new class-and-cell combination or has greater oracle severity within that combination.
Classifications are closed to `certified_violation`, `consumer_regression`, and `stress_discovery`.

## Replay evidence

Each retained directory under `fuzz-run/cases/` is a self-contained replay bundle.
It includes canonical JSON case and geometry records, exact and observed top-k evidence, mutated queries, the immutable corpus, and a byte snapshot of the index.
The bundle manifest and `checksums.txt` carry SHA-256 checksums, and the case records the seed, tool version, hashes, codec parameters, replay tolerances, and mutation details.

```powershell
tqp fuzz replay fuzz-run/cases/case-000000 --format json
```

Replay verifies the schema and version, payload checksums, canonical JSON, corpus and geometry hashes, index hash, codec metadata, recorded evidence, and tolerances before it opens the bundled index.
It then recomputes both retrieval paths without refitting the geometry profile.
Missing, corrupt, tampered, incomplete, or incompatible bundles exit with an error rather than being evaluated.

## Limits and interpretation

The MVP is CPU-compatible and uses only NumPy for the profile, mutators, coverage, and exact reference path.
It supports radial and shell mutations against one regular TQE index rather than sharded or third-party ANN indexes.
Sampled reverse-kNN counts are estimates over the profile's stored sample, so they are useful coverage signals rather than a population-wide hubness guarantee.
An exact-versus-compressed mismatch is normally a consumer regression or stress discovery, not a certificate violation, unless a recorded applicable certificate is explicitly contradicted.
