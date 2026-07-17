# IVF coarse-partition: recall vs fraction of corpus scanned

Reproduce: [`benchmarks/bench_ivf.py`](bench_ivf.py). Ground truth is the **brute-force
ADC top-k** over the whole corpus — the ranking IVF is trying to reproduce — so
`recall@10 = 1.0` means "identical to scanning everything." Synthetic corpus,
n = 100,000, dim 64 → PCA 32 → 4-bit, `nlist = 316` (√n), mean cell radius **65°**.

| mode | recall@10 | scan fraction | speedup vs brute | qps |
|---|--:|--:|--:|--:|
| brute-force ADC (baseline) | 1.000 | 100% | 1× | 709 |
| fixed nprobe=4 | 0.583 | 1.3% | 79× | 3743 |
| fixed nprobe=8 | 0.749 | 2.6% | 39× | 2197 |
| fixed nprobe=16 | 0.881 | 5.1% | 20× | 1133 |
| fixed nprobe=32 | 0.963 | 10.2% | 10× | 615 |
| **weighted A\* β=0.25** | 0.804 | **3.2%** | **31×** | 1145 |
| **weighted A\* β=0.50** | **0.999** | 22% | 4.5× | 56 |
| weighted A\* β=0.75 | 1.000 | 60% | 1.7× | 26 |
| admissible A\* (β=1) | 1.000 | 89% | 1.1× | 19 |

## What it shows

- **Coarse partitioning makes search sublinear.** Best-first `nprobe` recovers **96%
  recall while touching ~10% of the corpus** (10× fewer rows), and 88% at ~5%. The
  partition + best-first order is the workhorse — a clean recall/scan dial.
- **Weighted A\* is the adaptive dial.** Instead of picking `nprobe`, set how much
  recall you'll trade: `β` shrinks the cell radius to prune more. **β=0.5 lands at
  recall 0.999 while scanning 22%**, per-query, with no manual `nprobe`. β=0.25 is
  aggressive (0.80 @ 3%); β→1 is exact.
- **The admissible (worst-case) A\* stop is exact but barely prunes.** At 65° cell
  radii the bound `cos(θ − r)` is ≈1 for almost every cell, so the provably-safe stop
  scans 89%. This is the honest limitation of literal A\* here: an admissible
  heuristic gives no free selectivity in high dimension — you *must* relax it
  (weighted A\*, or a fixed `nprobe`) to get speed. Reported, not hidden.

## Notes

- Everything derives from the codes the ADC index already stores (the coarse space is
  `normalize(cent[codes])`), so the partition adds no per-row storage.
- QPS here is the pure-NumPy prototype path; the win is *rows scanned*, which is what
  carries to memmap (a probe faults in only its cells' pages) and to sharding (the same
  best-first order across per-shard IVF). This is the layer that takes search from
  O(N) toward the few-percent scans a trillion-vector index needs.
- Acceptance is recall of the shortlist against the exact ADC ranking — never
  reconstruction cosine.
