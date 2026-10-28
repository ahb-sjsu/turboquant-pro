# Hubness as a query/corpus density ratio: pilot result

**Registered before measurement:** [`docs/PREREG_hubness_density_ratio.md`](../docs/PREREG_hubness_density_ratio.md)
(commit c46a8ad, pushed before the run).
**Scripts:** [`hubness_density_ratio.py`](hubness_density_ratio.py) (pilot),
[`hubness_density_ratio_analyze.py`](hubness_density_ratio_analyze.py) (scores the registered predictions),
[`hubness_density_ratio_posthoc.py`](hubness_density_ratio_posthoc.py) (exploratory, after P1 failed).
**Raw output:** `results/hdr_results.json`, `results/hdr_scored.txt`, `results/hdr_posthoc.json`, `results/hdr.log`.
**Setup:** real Cohere Embed-V3 wiki1024 on Atlas, uniform random rows, cosine, exact top-k, 1,000 real queries.
Same-distribution queries are held-out corpus rows. Run time 17 minutes on 10 CPU cores, 2026-10-28.

## Verdicts

| prediction | verdict | numbers |
|---|---|---|
| **P1.** With real queries, the query/corpus density ratio predicts hubs better than query mass (by ≥ 0.05 Spearman) | **FAILS** | ratio 0.260 vs query mass 0.404 (k=20); 0.385 vs 0.491 (k=40) |
| **P2.** Deconvolved hub skew depends on n and k only through n/k | **holds as registered, but uninformative** | 87% of same-n/k pairs agree, but the registered control (same n, different k) agrees 80%, not clearly lower |
| **P3.** With same-distribution queries the ratio carries no signal, hubness is geometric, and real queries are hubbier | **HOLDS** | ratio 0.05 and 0.03; local density 0.38 and 0.37; skew real 6.36 vs same-distribution 2.51 at matched n=12.5k, k=40, ρ=1.6 |

## What the pilot shows

**The first-order density-ratio law is not supported.** Dividing query mass by corpus density makes the
predictor worse. That is true with a global bandwidth (registered) and, post hoc, with each point's own k-NN
ball (0.39 vs 0.40 and 0.46 vs 0.49). Corpus density by itself barely predicts hubs under real queries
(Spearman 0.06–0.10). The finding from round 13 of the generator campaign stands as it was: query mass
predicts retrieval. Normalizing it for corpus density adds noise, not signal.

**A point's capture basin is not its own k-NN ball.** With same-distribution queries, the query mass inside
each point's own k-NN ball is flat (Spearman 0.03–0.04 with counts), as the first-order picture predicts. Yet
the counts still follow local density (0.37). So same-distribution hubness comes from a density effect beyond
the first-order ball approximation: the geometric mechanism of Radovanović et al., not a query-side one.

**Same-distribution hub structure is flat in corpus size.** Deconvolved skew stays at 2.0–2.2 from n = 12,500 to
200,000 and for k from 5 to 80, at constant budget ρ = 2.5. The only departure is at the smallest n/k (k large
relative to n). This is why the collapse test in P2 cannot discriminate: almost nothing varies.

**An open discrepancy to resolve before using that last point.** The generator campaign recorded real hub
structure rising with n by about +0.5 per decade on the same corpus, with the same deconvolved statistic. This
pilot finds no rise for same-distribution queries on uniformly sampled rows. Two explanations fit, and the
pilot cannot separate them:
1. The earlier measurement used real queries, so the rise is a query-side property.
2. The earlier measurement used a row-ordered pool. Wikipedia row order is topically clustered, which the
   campaign already flagged for battery A.

Which one holds should be checked against the campaign's harness before either number is cited.

## Limits

- One corpus, one embedding model, one metric.
- Three seeds.
- The real-query cells sit at low budget (ρ = 0.8 and 1.6), because only 1,000 real queries exist, split in
  half to keep predictors and counts independent.
- Query mass from 500 real queries is sparse: 48–67% of corpus points have none inside the bandwidth. That
  limits every real-query correlation, the ratio's most of all. A larger real query set would test P1 more
  strongly. As run, P1 fails.
