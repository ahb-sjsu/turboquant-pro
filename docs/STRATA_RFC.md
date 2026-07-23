# STRATA — Stratified anatomy & area-scoped guarantees (RFC draft)

**Status: DRAFT ⚪ (proposed; external review 2026-07-23).** Companion to
[`HUBNESS_PRIMER.md`](HUBNESS_PRIMER.md) (motivation), [`TQE1_RFC.md`](TQE1_RFC.md)
(format hooks), and the recall contract in [`ROADMAP_2.0.md`](ROADMAP_2.0.md).
MUST/SHOULD/MAY per RFC 2119. Status legend: 🟢 shipping · 🟡 scaffolded · ⚪ designed.

> **Thesis.** Global scalars hide regional pathology; global staleness makes
> guarantees unaffordable. STRATA partitions measurement, remedies, operating
> points, and certificates by *area*, so verdicts are taken as **min-over-strata**
> and mutations stale **bounded regions**, not the world.
>
> **Governing rules (inherited, restated):** uncertain ⇒ no verdict. An
> incomplete area-map profile matches nothing, including itself. Unregistered
> verdict/warning causes raise `KeyError`. Trust the tail, not the mean —
> now spatially.

The OSPF analogy is a design aid, not an isomorphism (§7): hierarchy, border
summarization, and bounded blast radius transfer; administrative boundaries do
not — embedding space requires overlap (§1.2).

---

## 1. Definitions (normative)

Let `X` be the corpus, `Q` the query multiset, `d` the metric, `k` the depth.
All quantities below are properties of the experiment `(X, Q, d, k, M)` —
never of the corpus alone (primer, "who is asking").

### 1.1 Area map

An **area map** `M = (A, a, a_Q, O)`:

| component | meaning | constraints |
|---|---|---|
| `A = {A_1..A_m}` | area identifiers | `m ≥ 1` |
| `a : X → 2^A \ ∅` | corpus assignment | multi-assignment permitted (§1.2) |
| `a_Q : Q → 2^A \ ∅` | query assignment | same rule as `a`, or a declared metadata key (e.g. `--by language`) |
| `O ⊆ A × A` | adjacency/overlap relation | reflexive; drives staleness (§5) |

**Identity.** `tqp-area-map/1` is the canonical, content-addressed profile:
canonical JSON (identity-module conventions: sorted keys, `,`/`:` separators,
`ensure_ascii`, `allow_nan=False`) over
`{profile, algorithm_id, params, seed, corpus_fingerprint, assignment_rule,
query_assignment, overlap_policy, software_version}` → SHA-256 digest.
An incomplete profile matches nothing, including itself. Two artifacts
computed under different digests MUST NOT be compared, merged, or gated
together; tools MUST refuse, not warn.

### 1.2 Boundary rule

Nearest neighbors ignore borders. Conforming configurations MUST either
(a) multi-assign boundary rows (`|a(x)| > 1`) or (b) probe `home ∪ O(home)`
at query time. A single-area probe under a partition (non-cover) map is a
**non-conforming configuration** for any §5 guarantee: certificates MUST NOT
be issued over it.

### 1.3 Counts and per-area statistics

For `x ∈ X`, `Q' ⊆ Q`: `N_k(x | Q') = |{q ∈ Q' : x ∈ topk(q)}|`.

- intra count `N_k^intra(x) = N_k(x | {q : a_Q(q) ∩ a(x) ≠ ∅})`
- transit count `N_k^trans(x) = N_k(x | Q) − N_k^intra(x)`
- transit fraction `τ(x) = N_k^trans(x) / max(N_k(x), 1)`

Per area `A_i` (over `x` with `A_i ∈ a(x)`): count skew `S_k(A_i)`,
`max N_k`, mean transit `τ̄(A_i)`, and the anatomy correlation fields
(`corr(count, −d_k)`, `corr(count, centrality)`) restricted to `A_i` —
mechanism attribution is per-area, since §7's non-identifiability holds
per-area too.

### 1.4 Area classification (reported, never hidden)

All thresholds `θ_*` are **fields of the report**, not constants.

| class | predicate (defaults reported) | OSPF analogue |
|---|---|---|
| backbone member | `τ(x) ≥ θ_τ ∧ N_k(x) ≥ θ_N` | Area 0 transit |
| hub area | high `S_k(A_i)`; mechanism per corr fields | high-degree area |
| **not-so-hubby area (NSHA)** | high `S_k^intra(A_i)`, low `τ̄(A_i)` — local hubs, no global transit | NSSA¹ |
| stub area | low `N_k` in **and** out | stub area |

¹ The name is a pun with a license: OSPF's NSSA is the *not-so-stubby area*.
It is retained because it is (a) accurate and (b) irreversible now.

### 1.5 Minimum-evidence rule (ABSTAIN)

A per-stratum verdict REQUIRES `n_i ≥ n_min` corpus rows and `|Q_i| ≥ q_min`
queries. Below either bound the stratum verdict is **ABSTAIN** — reported,
counted (registered cause `stratum_insufficient_n`), and excluded from
pass/fail aggregation. ABSTAIN is not a pass. Skewness on thin strata is
noise; uncertain ⇒ no verdict. The estimator (`exact@sample` vs
`adc@full`) MUST be declared per stratum; approximate graphs over-return
hubs and bias `S_k` upward (primer caveat, applied per-area).

---

## 2. Phase 1 🟢 shipped — stratified instruments (measurement only)

*Status 2026-07-23: ALL Phase-1 gates met. Instruments implemented
(`turboquant_pro/strata.py` + CLI
`--strata`/`--by`/`--min-stratum-n`/`--abstain-fails`), schema frozen with
golden report fixtures in CI, causes registered, min-over-strata and
ABSTAIN paths tested, first slice of the relational surface
(`duckdb_ext.attach_strata`/`hub_census`/`transit_by_area`/`strata_gate`,
digest-guarded), and the per-language measured run is committed
([`RESULTS_multilingual_strata.md`](RESULTS_multilingual_strata.md) +
artifacts). The on-record predictions were WRONG in the informative
direction, twice: per-language S_k varies far less than predicted (P1 not
confirmed — BGE-M3 homogenizes), and P2 inverted — the trained interlingua
(LaBSE) does not concentrate transit on a central core; it diffuses it,
turning 7 of 13 eligible languages into the first measured
**backbone-class areas**, while emergent BGE-M3 keeps language borders
(all-NSHA) and routes its smaller transit through a central region. The
§1.4 taxonomy's backbone class, speculative at design time, exists in
production encoders and is objective-dependent.*

### 2.1 CLI surface

```bash
tqp anatomy --npy corpus.npy --k 10 --strata kmeans:64 --seed 7
tqp anatomy --npy corpus.npy --k 10 --by language --queries q.npy
tqp hubdiff --exact exact_ids.npy --approx ann_ids.npy \
    --strata map.json --min-anti-recall 0.90 --min-stratum-n 2000
```

`--strata` accepts a clustering spec (computed, then fingerprinted) or a
saved map artifact; `--by KEY` derives the map from row metadata. Gates
become **min over eligible (non-ABSTAIN) strata**. Exit codes: `0` pass ·
`1` a stratum failed a gate · `3` only-ABSTAIN (opt-in `--abstain-fails`
maps 3→1 for CI).

### 2.2 Report artifact (JSON, provenance-first)

```json
{ "schema": "tqp-strata-report/1",
  "provenance": { "area_map_digest": "…", "corpus_fingerprint": "…",
    "n": 250000, "k": 10, "seed": 7, "estimator": "exact@50000",
    "software_version": "…", "timestamp": "…" },
  "thresholds": { "tau": 0.5, "N": 50, "n_min": 2000, "q_min": 500 },
  "areas": [ { "id": "A_17", "n": 4812, "verdict": "pass|fail|ABSTAIN",
    "count_skew": 2.31, "max_Nk": 112, "tau_mean": 0.08,
    "corr_neg_dk": 0.61, "corr_centrality": 0.12,
    "anti_hub_recall": 0.94, "p05_recall": 0.88, "class": "NSHA" } ] }
```

The report is the certification record: no field of it may be claimed in
prose that is not present in the artifact. New verdict/warning causes are
registered before use: `stratum_insufficient_n`, `stratum_anti_hub_gap`,
`transit_concentration`, `area_map_mismatch`. Schema fields, cause IDs, and
class names are **API** — closed registry, deprecation policy, matrix row.

**Phase-1 gates:** JSON schema frozen with golden report fixtures in CI ·
causes registered · min-over-strata semantics tested · ABSTAIN paths tested
· per-language run on the 37-language corpus committed as the first
measured artifact (prediction on record: per-language `S_k` varies widely;
cross-lingual hubs concentrate at borders — to be *measured*, not assumed).

---

## 3. Phase 2 ⚪ — per-area remedies (mechanism selects remedy)

The anatomy vector is a prescription pad:

| per-area diagnosis | remedy | mechanism note |
|---|---|---|
| centrality-driven (`corr_centrality` high; hubs far from population) | **per-area (localized) centering** — one stored centroid per area | the v1.4.0 theorem at region scale: a local DC offset a global codebook wastes codes on; subtract the local mean |
| density-driven (`corr(−d_k)` high) | **CSLS / mutual-proximity scalar** `r_k(x)` — one float32 per record | corrects asymmetric neighborliness without touching geometry |

**Format hook (decide before TQE1 1.0 freeze).** `r_k(x)` is carried as a
TQE1 **optional length-delimited trailer** — registered trailer ID from the
`0x10–0x7F` range, payload `{r_k: f32, k: u8, estimator_id: u8}`, carrying
its own hash per TQE1 §3 coverage. Unknown-optional ⇒ skip: old readers
remain conformant (TQE1 §5), corrected-ADC readers get hubness-aware
scoring for four bytes. Per-area centroids are **encoder parameters**:
they enter the identity profile; changing them is an epoch event, not a
tweak.

**Threat-model clause (normative; added 2026-07-23).** Per-area centering
has an **adversarial dual**: localized centroids LOWER the bar for
centrality-injection attacks, because a local centroid only needs to
dominate a small neighborhood — validated on real BGE embeddings by the
Black-Hole Attack work (arXiv:2604.05480, Apr 2026: cluster-wise centroids
amplify hub capture; injected vectors reach the vast majority of top-10
results). Consequences: (a) the tqp threat model carries two
hubness-adjacent entries (this one + the timing side channel); (b) Phase-2
remedies MUST NOT be deployed as write-path trust — a stored per-area
centroid is attacker-relevant state and enters the identity profile
precisely so it cannot drift silently; (c) the anatomy tool gains a second
job: a sudden per-area mechanism shift from density to centrality is a
**poisoning signature** (mechanism attribution as intrusion detection —
cf. the Adversarial Hubness Detector, arXiv:2602.22427, whose
cross-cluster retrieval-spread statistic is a convergent cousin of τ).
Monitoring hook: diff `mechanism` and `hub_frac_central` per area across
index mutations; an area flipping central without a pipeline change is an
investigation, not a re-quantization.

**Phase-2 gates:** trailer ID ratified in the TQE1 registry · per-area
centering behind a flag · efficacy notebook measuring **Δ anti-hub recall**
per stratum (approved claim shape: "Δ measured in notebook N on corpus C";
banned: any unconditional improvement claim) · threat-model review of the
centering deployment against the clause above.

---

## 4. Phase 3 ⚪ — per-area operating points (identity-gated)

Bits, PCA dim, and codebooks per area — LOPQ ancestry acknowledged
(Kalantidis & Avrithis 2014) — with one new allocator: capacity is spent by
**measured fragility** (per-stratum `hubdiff`: anti-hub recall, p05), not by
variance alone. This completes a symmetry: the eigenvalue-weighted
quantizer stratifies the *spectral* axis; STRATA stratifies the *spatial*
axis. Same doctrine, orthogonal dimension.

**Identity (normative).** Record metadata gains
`{area_map_digest, area_id, area_codec_params}`. A reader encountering an
unknown `area_map_digest` MUST refuse to decode — enumerate, don't decode
(TQE1 §5/§8 rule, inherited verbatim). Cross-map reads are cache misses,
never best-effort. **Phase-3 gates:** identity fields ratified ·
refuse-on-mismatch tests · a measured capacity/recall trade per area on
one public corpus.

---

## 5. Phase 4 ⚪ (2.2-class) — area-scoped guarantees & blast radius

The recall-contract catalog key extends from
*(index fingerprint, query population, metric, operating-point family,
software version)* to include *(area_map_digest, area_id)*.

**Staleness (the OSPF payoff, normative):**

| event | stale set |
|---|---|
| insert/delete/update in `A_i` (fixed digest) | `cert(A_i) ∪ {cert(A_j) : (A_i,A_j) ∈ O}` |
| area-map recompute (digest change) | all area-scoped certificates |
| operating-point change in `A_i` | `cert(A_i)` + matrix row update |

Every index type claiming §5 conformance MUST document its worst-case
stale set per mutation class. A guarantee that can go stale silently is
not a guarantee — now with a bounded, *named* blast radius. Recalibration
becomes incremental: the flapping area re-runs SPF; the backbone does not.

Cross-track note (informative): KV prefix popularity is hub-shaped — the
shared system prompt is Area 0; unique user prefixes are stub areas.
P1-M4's "expected future reuse" term SHOULD reuse this taxonomy rather
than invent a parallel one.

---

## 6. Prior art & the narrow novelty claim

Partitioned indexes: IVF; per-cell codebooks: LOPQ; boundary duplication:
SPANN; emergent hub backbones: HNSW upper layers. Hubness measurement &
reduction: Radovanović et al. 2010; mutual proximity (Schnitzer et al.
2012); localized centering (Hara et al. 2015); CSLS (Conneau et al. 2018);
`scikit-hubness` (Feldbauer et al.) implements the measurement/reduction
canon and MUST be cited in the primer's further reading.

The claimable composition is narrow and stated exactly: **areas that are
hubness-diagnosed, fragility-allocated, identity-versioned, and carry
per-region certified guarantees with bounded staleness.** No "first"
claim is made before a literature-check notebook exists (positioning rule;
cf. the "standard" embargo, TQE1 §10).

*Lit-check, partial (2026-07-23).* Decomposed honestly: SQL-over-vectors is
commodity (pgvector, DuckDB, MyScale's Vector SQL); hierarchical
partitioning is commodity (IVF, HNSW layers); hubness measurement is
scikit-hubness + sixteen years of literature; and "hubness via SQL" as a
primitive is a five-line GROUP BY once the kNN graph is an edge table. The
security community is additionally converging from the adversarial side
(arXiv:2604.05480 validates centrality-mechanism hubness as an attack
surface; arXiv:2602.22427 ships an open-source hubness-poisoning scanner
across FAISS/Pinecone/Qdrant/Weaviate). The remaining gap this RFC claims:
**the relational surface over *certified* stratified hubness** — columns
whose values know their estimator, their n_min, and their area-map digest,
with cross-map joins refused by the relation itself. The detectors ship
scanners; the databases ship DISTANCE(); the composition is the claim.

---

## 7. Analogy limits (normative for prose)

| OSPF | transfers? | note |
|---|---|---|
| hierarchy / Area 0 | ✅ | measured emergent backbones exist |
| border summarization | ✅ | centroids/certs summarize at ABRs |
| bounded flooding / blast radius | ✅ | §5 stale sets |
| administrative boundaries | ❌ | geometry ignores them ⇒ §1.2 overlap is mandatory |
| link-state truth | ❌ | strata are workload-dependent (`a_Q` matters; corpus→corpus is not a substitute) |

## 8. Phrasing

**Approved:** "area-scoped recall, measured per stratum; verdicts abstain
below n_min; certificates expire per §5." **Banned:** "eliminates
hubness" · "uniform recall guaranteed" · "self-healing areas" ·
unconditional cross-corpus claims.

## 9. Open decisions (recorded, blocking)

1. `n_min`/`q_min` defaults and whether `--abstain-fails` is default-on in
   CI (proposal: off in dev, on in release lanes).
2. **Ratify now, ahead of the TQE1 1.0 freeze:** the `r_k` trailer ID and
   the `tqp-area-map/1` metadata fields — else the freeze walls them into
   a 1.1 spec revision. Rides with the standing `kv_block` freeze decision.
   *Resolved 2026-07-23 — TQE1 RFC §9a: registry freezes before profiles;
   trailer `0x10` (`hubness-scalar/1`) and the area-map metadata fields are
   allocated with syntax frozen and semantics provisional; `kv_block` ships
   labeled-experimental under the 1.1-draft revision.*
3. Overlap policy default: multi-assignment vs adjacent-probe (§1.2a vs b)
   — affects both recall and §5 stale-set size; measure before choosing.

*Trust the tail, not the mean — in every area.*
