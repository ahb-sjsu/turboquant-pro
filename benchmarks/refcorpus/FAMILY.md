# The benchmark family: nested tiers, three truth layers

**The object is not a file.** It is a versioned, content-addressed *family*
from which identical corpora at 10⁶ … 10¹² rows can be reconstructed, each a
strict subset of the next, with per-tier ground truth and three independent
notions of "correct answer".

Working name: **OpenVector Atlas** (name-collision check pending — "Atlas" is
already a Meta retrieval model and this project's own workstation).

---

## 1. Why a nested family rather than a big corpus

Today, results at 10⁶ and 10⁹ are reported on *different corpora*, so scaling
behaviour is confounded with distribution change: when recall drops from one
paper's 1M benchmark to another's 1B benchmark, nobody can separate "search
got harder with N" from "the data changed". A nested family holds the
distribution fixed and varies only N, which makes the scaling question
answerable at all.

**Tiers.** `T6, T7, T8, T9, T10, T11, T12` (rows = 10^k), with
`T_k ⊂ T_{k+1}` exactly.

**The nesting requires a seeded shuffle, not a prefix.** Source corpora arrive
ordered by language, dump order, crawl date, or article ID. A naive prefix
would make T6 monolingual or single-topic and silently poison every
cross-scale comparison. Tier membership is therefore defined by a
content-addressed permutation: `rank(row) = H(salt ‖ row_id)`, sorted; T_k is
the first 10^k rows of that order. The salt is published in the manifest, the
permutation is regenerable, and stratification (language, source, domain) is
*reported per tier* so representativeness is measured, not assumed.

**Ground truth is not nested.** A query's true neighbours change as the corpus
grows, so GT is computed and published **per tier**. This is the dominant cost
of the whole programme (one exact pass per tier) and the reason the artifact
is valuable: the cost is paid once, and the result is a few MB.

---

## 2. Three truth layers

The layers are independent, and the gap between them is the scientific point.

**L1 — geometric truth.** Exact k-NN under the frozen metric. What ANN papers
measure. Cost: one exhaustive pass per tier.

**L2 — structural relevance.** Relevance labels derived from document
structure, available at corpus scale and free of annotation cost:

| signal | query side | relevant target |
|---|---|---|
| hyperlink | page (or anchor context) | linked page |
| citation | citation context | cited work (OpenAlex) |
| community Q&A | question | accepted/high-score answer |
| multilingual link | page in language A | same entity in language B |

**These labels are biased and must be characterized, not assumed.** Link
graphs are popularity-skewed; citations carry field and recency bias; Q&A
pairs are noisy; and a linked page is *related*, not necessarily an answer.
Each label family ships with measured bias statistics (degree distribution,
temporal skew, language coverage) and is usable as a *relative* signal across
systems, never as an absolute relevance oracle.

**L3 — human gold.** 1,000–5,000 queries with independent judgments.
**Preference: reuse existing judged sets** (TREC/BEIR/MS MARCO-style) where
licensing permits, rather than commissioning new annotation — cheaper, and
comparable to published work. Commissioned annotation only for gaps.

**The claim the layers exist to test.** Our measurements already show
self-recall ≫ truth-recall (0.999 vs 0.592 at 10⁹). L2/L3 extend this one
level further: a system can preserve L1 neighbours while degrading L2/L3
relevance, or vice versa. A benchmark reporting only L1 cannot see this.
Formally, the family is designed so that **vector recall, neighbour truth,
and semantic relevance are measured separately and can be shown to
dissociate.**

---

## 3. Difficulty strata (average recall hides the failures)

Every query carries a precomputed stratum label so results are reported by
stratum, not only in aggregate:

- **local intrinsic dimension** of the query neighbourhood (low/med/high);
- **margin** `d_{k+1}/d_1` — tight margins are where quantization flips ranks;
- **hubness exposure** — whether true neighbours are hubs or anti-hubs;
- **neighbour dispersion** — are the k true neighbours one cluster or many;
- **L1/L2 agreement** — queries where geometric truth and structural
  relevance disagree are the most diagnostic of all.

Strata are computed from L1 + the geometry battery, published with the GT, and
fixed per tier.

---

## 4. Sources and the real/procedural seam

| tier | source | nature |
|---|---|---|
| T6–T8 | Wikipedia Embed-V3 (real, 1024-d), BIGANN slices | **real** |
| T9 | BIGANN (real, published GT), Wikipedia multilingual (~2.5×10⁸) | **real** |
| T10–T12 | procedural extension, RC-1/RC-2 validated | **generated** |

**The seam is the honesty requirement of the whole design.** Above it, no
public corpus of real neural embeddings exists — Common Crawl could supply
10¹⁰–10¹¹ *passages*, but embedding them is GPU-years, and no one has
published such a set. Tiers above the seam are legitimate **only** to the
extent RC-1 (geometric equivalence) and RC-2 (predicting unfitted ANN
behaviour) hold, and every tier is labelled `real` or `generated` in the
manifest so no result can quietly straddle it.

---

## 5. What each tier costs (order of magnitude)

| tier | corpus bytes (128-d u8) | GT pass | notes |
|---|--:|--:|---|
| T9 | 128 GB | ~10³ CPU-h | BIGANN GT already published |
| T10 | 1.3 TB | ~10⁴ CPU-h | |
| T11 | 13 TB | ~10⁵ CPU-h | grant-scale |
| T12 | 128 TB | ~10⁵–10⁶ CPU-h | never materialized; per-shard fetch only |
| T13 | 1.3 PB | ~10⁶ CPU-h | **defined but GT not realized** |

Nobody stores T11+; workers fetch or regenerate shards, verify against the
Merkle manifest (`DISTRIBUTION.md`), and discard. The published artifacts are
manifests, GT, strata, and labels — all small.

---

## 6. Status and order of work

1. RC-1 geometry battery (running) — is procedural extension legitimate?
2. Reconstruction experiment at T7 (`DISTRIBUTION.md` §6) — is the object
   verifiable end-to-end?
3. L1 + strata at T6–T8 on real data; L2 labels from Wikipedia links.
4. RC-2 sealed predictive validation.
5. Only then: tiers above the seam.

Nothing above the seam is published as a benchmark until 1, 2, and 4 pass.
If RC-1/RC-2 fail, the family stops at the seam and says so — a real,
nested, three-truth-layer benchmark to 10⁹ is still worth having.
