# Geometry-aware retrieval fuzzing (`tqp geometry`, `tqp fuzz`)

Issue [#162](https://github.com/ahb-sjsu/turboquant-pro/issues/162), MVP.
Modules: `turboquant_pro/geometry.py`, `turboquant_pro/fuzz.py`.

Aggregate recall is a mean, and a mean can stay green while particular queries
lose their neighbours. The fuzzer goes looking for those queries. It mutates
queries in the corpus's own geometry, compares exact retrieval with
TurboQuant-Pro's compressed retrieval on every candidate, keeps the cases that
reach new regions or do more damage, and writes each one as a bundle that
replays deterministically.

The research question the MVP is built to ask:

> At fixed Mahalanobis centrality, can angular mutations create hubs whose
> nearest-neighbour rankings are disproportionately unstable under compression?

The `shell` mutator holds Mahalanobis centrality fixed exactly, so a campaign
using only `--mutators shell` asks it directly.

## Commands

```bash
tqp geometry profile --embeddings corpus.npy --k 10 [--sample N] \
    [--shrinkage ledoit-wolf|A] [--eig-floor 1e-6] --out geometry.json

tqp fuzz retrieval --embeddings corpus.npy --queries queries.npy \
    --out fuzz-run/ [--k 10] [--budget 1000] [--seed 42] \
    [--mutators radial,shell] [--recall-floor 0.9] \
    [--bits 4] [--out-dim D] [--codec-seed 0] [--metric cosine] [--scorer exact]

tqp fuzz replay fuzz-run/cases/case-000017 [--embeddings corpus.npy]
```

Every command takes `--format json`. `tqp fuzz retrieval` exits 1 if any case is
a certified violation, 2 on bad input, and 0 otherwise. `tqp fuzz replay` exits
0 when the case reproduces, 1 on a mismatch, and 2 when the bundle cannot be
trusted.

## The geometry profile

For a corpus `X` (n rows, d columns) the profile reports:

* **Mahalanobis centrality**: the radius `r = ||Sigma^(-1/2) (x - mu)||` of every
  row, under the regularized covariance described below. A small radius is
  central.
* **Reverse-kNN occurrence counts `N_k`**: how often each row is among the k
  nearest neighbours of the others. These are exact by default. With
  `--sample m`, m seeded rows query the corpus and the counts are scaled by
  n/m. That is an unbiased estimate, because each row's count is a sum over
  queries.
* **Regions**:
  * central and peripheral rows are the lowest and highest `central_quantile`
    of the radius;
  * hubs and anti-hubs are at or above `hub_quantile`, and at or below
    `anti_quantile`, of `N_k`.

  The profile lists the central hubs, peripheral hubs and central anti-hubs,
  and counts all four region crossings.
* `corr_n_k_centrality`: the Spearman correlation of `N_k` with `-r`.
* `hub_stability_k_2k`: the Jaccard overlap of the hub sets at k and 2k.
* The spectrum: effective rank (`exp` of the normalized-spectrum entropy),
  participation ratio, and the top eigenvalue shares.

Centrality in `tqp anatomy` is Euclidean distance to the mean. Here it is
Mahalanobis, which measures each direction against its own spread.

## Covariance regularization and singular covariance

The sample covariance `S` is singular whenever `n <= d`, and for most embedding
corpora it is ill-conditioned even when it is not singular. Whitening with
`S^(-1/2)` then divides by near-zero variances. Those are exactly the
low-variance directions a compressor discards first, so an unregularized radius
is dominated by noise in them. Two guards apply, and both are recorded in the
profile and in every case under `regularization` or `whitener`:

1. **Shrinkage toward the scaled identity.**
   `Sigma = (1 - a) S + a (tr S / d) I`.
   * The default, `ledoit-wolf`, estimates `a` analytically (Ledoit and Wolf,
     *A well-conditioned estimator for large-dimensional covariance matrices*,
     2004). It shrinks harder when n is small relative to d.
   * A number in `[0, 1]` fixes `a` instead.
   * Any `a > 0` makes `Sigma` full rank. A direction outside the data's span
     gets variance `a tr(S)/d`, so a row's radius along it is measured against
     the average variance, not against zero.
2. **An eigenvalue floor.** Eigenvalues below `eig_floor * max eigenvalue` are
   raised to it.
   * With shrinkage on, the floor almost never binds.
   * With `--shrinkage 0` on a singular `S`, the floor is what keeps the radius
     finite. In that case `n_floored` reports how many directions it decided,
     and radii along those directions carry no information about the data. Read
     them as "outside the observed span", not as measurements.

A corpus with zero total variance is an error. There is no geometry to profile.

`condition_number` (after both guards) and `whitener_sha256` (a hash of the
fitted map) are recorded. Replay checks the corpus hash, so a bundle is never
evaluated against a different whitening.

## The campaign

The fuzzed object is a **query**. The corpus and its compressed index stay
fixed for the whole campaign.

* **Compressed path:** `TQEmbeddingCodec` (PCA followed by TurboQuant) and an
  `ADCIndex` over its codes. This is the real compressed search, not a
  reconstruction proxy. The default `--scorer exact` uses the numpy reference
  scorer, so a case replays bit for bit on any machine. `--scorer fast` uses
  the AVX2 kernel where present, and the bundle records which scorer ran.
* **Exact path:** `metrics.exact_scores` over the fp32 corpus, with ties
  broken by row index.
* **Mutators**, applied in the whitened space `z`:
  * `radial`, `z' = alpha z` with `alpha` log-uniform in [0.25, 4], moves the
    query toward or away from the centre and keeps its direction;
  * `shell` turns `z` by a random angle up to pi/4 at the same `||z||`, so
    centrality is held exactly.

  Each mutation has its own seed, `[campaign seed, candidate number]`.

**Oracle (per candidate):**
* recall@k;
* the first incorrect rank;
* the rerank depth: how deep in the compressed ranking the whole exact top-k
  sits;
* Kendall tau over the candidate pool, the union of the exact and compressed
  top `max(5k, 50)`;
* the exact top-k margin;
* the certificate check described below.

**Coverage cell:**

| Component | Bands |
|---|---|
| The query's Mahalanobis-radius quantile in the corpus | 5, plus a 6th for "beyond every row" |
| The `N_k` quantile of its exact nearest neighbour | 5 |
| The exact margin quantile, against the seed queries | 5 |
| The recall band: 1.0, at least 0.9, at least 0.7, at least 0.5, below | 5 |

**Retention.** A candidate is kept when any of these holds:
* it enters a new cell;
* it is more severe, meaning lower recall@k (with tau as the tie-break), than
  the cell's worst so far;
* it brings a class the cell has not had.

Kept candidates join the parent pool.

## Classification

A retrieval discrepancy is not a correctness failure, so the three classes are
kept apart:

* **`certified_violation`**: on this case's own candidate distances, the
  *strict* rank certificate (`rank_certificate.certify(lo=0, hi=100)`, the
  unconditional worst-case regime) certifies a Kendall-tau floor above zero,
  and the observed tau is below it. For those distances the floor is a theorem,
  so this class means a bug in the certificate, the codec or the metric
  plumbing. It is never just a hard query.
* **`consumer_regression`**: recall@k below `--recall-floor`.
* **`stress_discovery`**: kept for coverage or severity, but neither of the
  above.

Inner product has no distance, so its cases report the certificate as not
applicable.

## Replay bundles

Each kept case is a directory, `cases/case-NNNNNN/`:

| File | Contents |
|---|---|
| `case.json` | Schema `turboquant-pro/fuzz-case` v1. Records: the campaign and mutation seeds; the mutator chain with parameters; the parent case; the coverage signature and its inputs; the oracle record and class; the codec configuration (bits, PCA dimension, codec seed, training rows, metric, scorer); the corpus sha256 and shape; the whitening metadata and the geometry profile's hash; the tool version; and the replay tolerances |
| `query.npy` | The case's query |
| `expected_exact.json` | The exact top-k ids and scores |
| `observed_tqp.json` | The compressed top-k ids and scores |
| `environment.json` | Python, numpy, platform, and whether the ADC kernel was available |
| `corpus.npy` | The corpus, when it is at most 4 MiB, so the bundle replays on its own. Otherwise pass `--embeddings` to replay |
| `replay.sh` | Runs `tqp fuzz replay` on the directory |
| `checksums.txt` | The sha256 of every other file |

**Replay fails closed.** It refuses (exit 2) any bundle with:
* no `checksums.txt`, or one that does not cover a required file;
* a listed file that is missing or does not match its checksum;
* an unknown schema or version;
* no corpus available;
* a corpus whose hash differs from the recorded one.

Otherwise it rebuilds the codec from the recorded parameters and re-evaluates
the query. To count as reproduced, the case must match within the declared
tolerances: exact top-k ids, compressed top-k ids and the class identical, and
the recorded floats within `1e-6`.

## Not in the MVP

The issue's follow-ups are left for later: `tqp fuzz minimize`, corpus-level
mutators (hub-boundary, cluster mass, radius spectrum), hub-identity and
certificate-strength oracles beyond the per-case check, OpenVector Bench
generators, FAISS, DiskANN and cuVS oracles, and distributed campaigns.
