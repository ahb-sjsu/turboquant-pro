# Hubness, anti-hubs, and why your recall number is hiding something

*A primer for people meeting these ideas for the first time. No prior exposure
assumed. The tools it explains: [`tqp anatomy`](CLI.md) and
[`tqp hubdiff`](CLI.md).*

## The phenomenon in one experiment

Take any embedding corpus — a million sentence vectors, say. For every vector,
find its 10 nearest neighbours. Now ask the reverse question: **how many times
does each vector appear in someone else's top-10 list?**

Call that number N₁₀. If neighbour relationships were roughly symmetric, N₁₀
would be about 10 for everyone. In low dimensions it nearly is. In the
dimensions embeddings actually live in (hundreds to thousands), it is not even
close: a few vectors appear in *hundreds* of lists — **hubs** — while a large
fraction appear in almost none — **anti-hubs**. The distribution of N₁₀ grows a
long right tail, and the standard one-number summary of that tail is its
skewness (our gate G6 elsewhere; `count_skew` in `tqp anatomy`).

This is not a bug in your index. It is a measured property of high-dimensional
geometry, documented across datasets since Radovanović et al. (JMLR 2010):
hubness *emerges* with dimension. Two intuitions for why:

1. **Distances concentrate.** In high dimensions, the distances from a query to
   most points crowd into a narrow band. When everyone is *almost* the same
   distance away, tiny systematic advantages decide who makes the top-10 — and
   a point with a tiny advantage over many queries wins in many lists at once.
2. **Centrality is such an advantage.** A point slightly closer to the data
   mean is slightly closer to *everything* on average. Concentration amplifies
   that whisper into hundreds of list memberships.

Density is another such advantage: a point sitting where the corpus is locally
denser has more close approaches. Which brings us to the important part.

## Two corpora, same hubness number, opposite behaviour

Here is the trap this primer exists to flag: **the skewness scalar does not
tell you what your hubs *are*.** We have measured two corpora with essentially
the same G6 reading whose hubs were completely different objects:

| | corpus A (a real text corpus) | corpus B (a synthetic look-alike) |
|---|---|---|
| count skew | ≈ target | ≈ target (same number) |
| max N₁₀ | 78 | **369** |
| what predicts being a hub | **local density** (corr with −d₁₀ ≈ +0.67) | centrality (hubs 34% closer to the mean than average) |
| hubs' local scale | ~8% below population | ~34% below population |

Corpus A's hubness is a smooth tail of moderately popular points in denser
regions. Corpus B's is a handful of centrality "super-hubs" vacuuming up
lists. Same scalar; different mechanism; and an ANN index, a quantizer, or a
graph builder will treat them very differently. That is why `tqp anatomy`
reports a **vector**, not a number:

```bash
tqp anatomy --npy corpus.npy --k 10
# corpus->corpus k=10 n=250000: count skew 2.31 max 112;
# corr(count, -d_k) +0.61 corr(count, centrality) +0.12; ...
```

Reading it: high `corr(count, -d_k)` with hub medians near the population =
density-driven (the healthy, real-corpus pattern). High
`corr(count, centrality)` with hubs far from the population = super-hubs (often
a pipeline artifact — an aggressive mean shift, a collapsed subspace, an
over-regularised compression).

One more subtlety worth knowing: hubness depends on **who is asking**. The
counts are collected over a query set, so hubness is a property of the
*(corpus, query distribution, metric, k)* experiment, not the corpus alone. A
corpus that looks mildly hubby under its own rows as queries can look extremely
hubby under a real query workload that keeps visiting the same popular regions.
If you have real queries, measure with them (`--queries q.npy`); the
corpus→corpus reading is not a substitute.

## Anti-hubs: where compressed indexes go to fail

Flip to the other tail. Anti-hubs are the rows almost nobody's list contains —
isolated points, fine distinctions, rare content. Queries whose true nearest
neighbours are anti-hubs are the *hardest retrieval cases you have*: the
signal separating the right answer from the crowd is the smallest.

Now compress the index. Quantization rounds fine distinctions away first — that
is what quantization *is* — so the damage lands disproportionately on exactly
those queries. And here is the operational trap: **aggregate recall barely
moves.** Anti-hub queries are, by definition, a minority; a mean over all
queries can stay green while the tail collapses.

This is the same failure shape this project documents for reconstruction
cosine (a key quantizer can hold cosine ≈ 0.995 while the attention ranking it
feeds falls apart — [`KV_KEYS_FINDING.md`](KV_KEYS_FINDING.md)). A blind
aggregate hides the consumer's tail. `tqp hubdiff` is the retrieval-space
instrument for it:

```bash
tqp hubdiff --original corpus.npy --reconstructed corpus_recon.npy --k 10
# recall@10 0.9812 (p05 0.8000) | hub-rank corr +0.942 hub-set Jaccard 0.780 |
# anti-hub recall 0.6120 (9.8% of queries) — WARNING: anti-hub gap 0.369; ...
```

Four things it checks that recall@k alone cannot:

- **anti-hub recall** — recall restricted to queries whose true nearest
  neighbour is a bottom-decile row. The first casualty of over-compression.
- **p05 per-query recall** — the distributional version of the same warning.
- **hub-rank correlation** — do the exact and compressed searches even agree
  about *which rows are hubs*? Compression that reshapes the hub structure
  changes system behaviour (cache hit patterns, graph connectivity, load
  balance) in ways recall never shows.
- **hub-set Jaccard** — overlap of the top-1% hub sets under each search.

It is system-agnostic: pass `--exact/--approx` neighbour-id arrays and compare
*any* two systems — HNSW vs. exact, two graph build orders, two shardings, two
quantizer operating points. Gate it in CI like the rank certificate:

```bash
tqp hubdiff --exact exact_ids.npy --approx hnsw_ids.npy --n-base 1000000 \
    --min-anti-recall 0.9    # exit 1 when the tail collapses
```

## The one-paragraph theory summary

For the theoretically inclined: a corpus + metric partition query space into
"capture basins" — the region where a given row makes the top-k. A row's count
is the query mass falling in its basin, so hubness factors into *corpus-side*
structure (basin sizes: density, centrality) and *query-side* structure (where
the query mass goes). The same skew can come from huge basins or from
concentrated query mass — a second, independent reason the scalar is
non-identifying, and the reason the anatomy report and the query-battery
distinction exist. The full treatment, with the measurements behind the table
above, lives in the OpenVector Bench companion project
([`spec/BOND_METRIC.md`](https://github.com/ahb-sjsu/openvector-bench/blob/main/spec/BOND_METRIC.md)).

## Cheat sheet

| symptom | instrument | healthy reading |
|---|---|---|
| "is my corpus hubby?" | `tqp anatomy --npy x.npy` | skew ~1–10 at k=10; grows with n |
| "are my hubs real or an artifact?" | corr fields of `tqp anatomy` | density-driven: corr(−d_k) high, hub medians near population |
| "did compression hurt?" | `tqp hubdiff --original/--reconstructed` | anti-hub gap < 0.05, hub-rank corr ≈ 1 |
| "do these two systems agree?" | `tqp hubdiff --exact/--approx` | hub-set Jaccard near 1, p05 near mean |
| "keep it that way" | `--min-anti-recall` in CI | exit 0 |

*Further reading: Radovanović, Nanopoulos & Ivanović, "Hubs in Space" (JMLR
2010) — the emergence result; this repo's
[`CERTIFICATE_SPEC.md`](CERTIFICATE_SPEC.md) for the rank-certificate side of
the same acceptance philosophy.*
