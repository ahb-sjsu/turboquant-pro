# Pre-registration — procedural reference corpus (RC-1), v2

**Supersedes v1 (2026-07-20).** v1 was written, a smoke run was executed, and
external review identified that v1 over-licensed its own result and contained
a self-contradictory null policy. This v2 is registered **before any gating
run**. The v1 text and the reason for each change are preserved in §9 so the
revision history is auditable rather than silently rewritten.

Discipline follows *Geometric Methods* (Book 1): posited/measured separation,
admission filters fixed in advance, frozen nulls, misses reported as misses.

---

## 1. The claim, scoped

**RC-1 (registered claim).** A deterministic generator, fitted to a real
embedding corpus, is *equivalent to that corpus on the prespecified geometric
diagnostics of §4, over the prespecified ranges of sample size and
neighbourhood scale in §3*, under the frozen metric of §2.

**RC-2 (the claim that would matter).** A generator passing RC-1 *predicts
ANN system behaviour it was not fitted on* (§6).

Deliberately **not** claimed by either: "geometrically indistinguishable at
any scale"; "scientifically legitimate as a 1T reference corpus"; anything
semantic or task-level. A 1T reference corpus requires RC-2 plus an explicit
extrapolation argument, and the extrapolation from 2×10⁵ to 10¹² is stated as
an assumption, never as a result.

**Falsification rule (unchanged from v1, and binding).** If the generator
fails, it remains a random-number benchmark and is published as such. A
failed realism test will not be renamed a successful stress test.

---

## 2. Frozen operational metric (registered before measurement)

The deployed index scores **cosine** similarity, and the target corpus is
unit-normalized (Cohere Embed-V3). The operational geometry is therefore
**angular: all corpora are L2-normalized before any diagnostic**, and
neighbours are exact Euclidean neighbours of the normalized vectors, which
induce exactly the cosine ranking.

Consequences, registered rather than discovered:
- Vector norms lie **outside** the measured geometry by construction. Norm CV
  and the angular/full agreement are **reported descriptors, never gates**
  (this replaces v1's G8; see §9 D1).
- Any generator must adopt the target's normalization convention. That is a
  structural requirement, not evidence of fidelity, and is labelled as such.
- Raw-Euclidean diagnostics are reported in an appendix for the record; they
  do not gate.

---

## 3. Measurement grid (replaces v1's single operating point)

Diagnostics are computed over a grid, and **curves are compared, not
endpoints**:

- sample size `n ∈ {25k, 50k, 100k, 200k}`
- neighbourhood scale `k ∈ {10, 30, 100}`
- 5 subsamples per cell

One exact k-NN structure per `(corpus, n, subsample)` at `k=100`; the `k=10`
and `k=30` diagnostics are read from prefixes of the same neighbour lists, so
the three scales describe one neighbourhood graph.

**Reported per diagnostic:** the value at each `(n, k)` cell, and the fitted
scaling exponent in `n` (slope of `log T` vs `log n`) at each `k`. Both the
level and the exponent enter the admission rule (§5).

**Honest description of the 5 subsamples:** they are overlapping draws from
one corpus and therefore measure **subsampling stability, not independent
replication**. They are never described as independent corpora. Uncertainty
intervals derived from them are labelled subsampling intervals.

**Extrapolation check (required before any 1T claim, not part of RC-1):**
repeat the grid at `n = 10⁶` and `n = 10⁷` on the real corpus and confirm the
generator's fitted exponents continue to agree there.

---

## 4. Two batteries (v1 measured only the first, while claiming the second)

**Battery A — corpus-to-corpus.** Queries are held-out corpus points, never
members of the searched base. Characterizes the corpus's internal geometry.

**Battery B — query-to-corpus.** Queries are the **real held-out query
embeddings** shipped with the target (`queries.npy`), searched against the
corpus. Real systems query a document distribution with a query distribution
that is generally *different*; v1 conflated the two.

For null and generated corpora, Battery B uses queries produced by that
corpus's own generating process (the analogous query set). A procedural
benchmark ultimately requires a **query generator** as well as a corpus
generator; RC-1 covers the corpus, and the query generator is registered
separately before it is claimed.

Gates in §5 apply to **both batteries**; a generator must pass both.

---

## 5. Diagnostics and the admission rule

Measured on normalized vectors, per §2–§4.

| # | diagnostic | statistic |
|---|---|---|
| G1 | intrinsic dimension | two-NN MLE (Facco 2017), trimmed |
| G2 | intrinsic dimension | ball-growth slope |
| G3 | spectral profile | effective rank (participation ratio) |
| G4 | spectral profile | dims for 90% energy |
| G5 | distance concentration | relative contrast at k |
| G6 | hubness | skewness of k-occurrence N_k |
| G7 | local ID | IQR of per-point Levina–Bickel LID |
| G8 | neighbourhood retention | Jaccard of k-NN under PCA-256 vs full |

(v1's G8 angle/radius is now a descriptor per §2; v1's G9 is renumbered G8.)

**Equivalence testing, not point comparison.** For each diagnostic `T` and
each `(n, k)` cell, form the ratio `R = T_gen / T_real`. The gate passes only
if the **95% subsampling interval for R lies entirely within [0.85, 1.15]**
(two one-sided tests). A point estimate inside the band with an interval
crossing it is a **fail**, recorded as such. Equivalence bands per diagnostic:

- G1, G3, G5, G6, G8: `[0.85, 1.15]`
- G2, G4, G7: `[0.80, 1.20]` (estimators with known higher variance)
- scaling exponents: absolute difference `|Δslope| ≤ 0.05` per decade

**Admission rule.** A generator is admitted iff, in **both batteries**:
1. G1, G5, G6 pass in **every** `(n, k)` cell (these govern ANN difficulty);
2. at least 6 of the 8 gates pass in **at least 10 of 12** cells; and
3. the scaling-exponent criterion passes for G1, G5, G6.

If a gate is ever removed, the count in (2) is reduced by one and the
requirement stays "all but two" — defined now, so removal cannot loosen
admission (v1 left this undefined).

---

## 6. Sealed predictive validation (RC-2) — registered now, run after RC-1

Geometric diagnostics (§5) are the **fitting and selection criteria**. The
following are **never used during fitting** and are opened only once, against
a single frozen generator:

1. IVF recall vs `nprobe` (and vs scan fraction) at matched build parameters;
2. coarse-cell occupancy distribution and imbalance;
3. nearest-neighbour margin distribution (`d_k+1 / d_1`);
4. per-query difficulty distribution (per-query recall at fixed `nprobe`);
5. rerank shortlist depth required for a fixed true recall;
6. a graph index (HNSW) recall/latency curve, if it runs in this environment.

**RC-2 passes** iff the generated corpus reproduces the real corpus's curves
for (1)–(5) within the §5 equivalence logic. This is the question that
matters: *does matching the geometry predict the behaviour of ANN systems
that were not used to fit the generator?*

---

## 7. Data partitioning and the seal

The real corpus is split by hash of row index into **train (50%) /
validation (25%) / sealed test (25%)**, disjoint, fixed at the start.

- **train** — fit distributional parameters;
- **validation** — choose generator family and hyperparameters, iterate
  freely, report the iteration count;
- **sealed test** — evaluated **once**, against one frozen candidate.

Before the seal is opened, the generator's source and parameters are hashed
(SHA-256) and the hash recorded in `SEAL.md` with a timestamp. Any change
after that point requires a new seal and is reported as a second evaluation.

---

## 8. Frozen nulls and the corrected null policy

1. **iid Gaussian**, matched dimension (normalized per §2).
2. **Per-feature shuffle** of the real corpus — marginals kept, joint destroyed.
3. **Low-rank + noise** at matched effective rank — the recipe behind the
   existing 1B/10B synthetic corpus; the honest baseline to beat.

**Corrected policy (v1 was self-contradictory).** A targeted null is
*expected* to pass the property it was built to preserve — null 3 should pass
G3 — and that is evidence the battery measures distinct properties, not that
G3 is useless. Therefore:

- The criterion is at the level of the admission rule: **no frozen null may
  satisfy §5's admission rule.** If one does, the battery is insufficient and
  must be strengthened *by addition*, never by relaxing thresholds.
- **Gates are not deleted after seeing null results.** A gate may be
  demoted only for a reason registered independently of the data (as in §2's
  metric freeze).
- Per-gate discrimination is *reported* (which nulls each gate separates),
  because a gate that separates nothing is informative about the battery —
  but it is not silently removed.

---

## 9. Revision history (v1 → v2)

- **D1 (G8 angle/radius).** v1 gated on norm CV / angular agreement. The
  smoke run surfaced that the target is unit-normalized (norm CV 1.8e-05),
  making the gate vacuous on the target and trivially satisfiable by any
  generator that normalizes. **v2 resolves this by freezing the metric (§2)**,
  under which norms are outside the measured geometry by construction; the
  statistic becomes a descriptor. Recorded honestly: the *need* was surfaced
  by data, the *resolution* follows from a registered decision, and the
  admission arithmetic is unchanged (v1's "≥7 of 9" with a free-pass gate and
  v2's "all but two of 8" permit the same number of real failures).
- **Scope.** v1 claimed legitimacy as a 1T reference corpus; v2 scopes RC-1
  to the measured grid and makes 1T contingent on RC-2 plus a stated
  extrapolation assumption (§1, §3).
- **Grid.** v1 measured one `(n, k)` point; v2 requires curves and exponents (§3).
- **Queries.** v1 claimed held-out queries but measured corpus-to-corpus only;
  v2 registers two batteries and requires both (§4).
- **Metric.** v1 left the metric implicit (raw Euclidean in code); v2 freezes
  it as angular (§2).
- **Inference.** v1 compared medians to a ±tolerance; v2 requires equivalence
  intervals (TOST) and relabels the 5 seeds as subsampling, not replication (§5).
- **Leakage.** v1's fit/gate split becomes a development set under iteration;
  v2 adds a validation portion and a hashed seal (§7).
- **Null policy.** v1's "strike any gate the nulls pass" contradicted having a
  targeted null; v2 moves the criterion to the admission rule (§8).
- **Implementation.** v1's k-NN would allocate ~1.7 TB in the NumPy path and
  a 1.64 GB matrix in the Torch path; v2 requires doubly-blocked exact search
  with device-aware block sizes (`corpus_geometry.py`).
