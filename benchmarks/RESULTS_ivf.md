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

## Sharded IVF at scale, on local NVMe

The single-node numbers above use `IVFIndex`. The same coarse layer folds into
`ShardedIndex` (`build_ivf` + `search(nprobe=…)`) for indexes that span many shards.
[`benchmarks/bench_ivf_sharded.py`](bench_ivf_sharded.py), **50M vectors** (dim 32 →
PCA 24 → 4-bit, 10 shards, `nlist=2048`, 7.87 GiB index) on a **local NVMe** volume
(Atlas), 200 queries; reference = the full-scan ADC top-k, so recall 1.0 = identical.

Build streamed shard-by-shard in **169 s**; `build_ivf` (`nlist=2048`, ≈ 24k
rows/cell) in **258 s**; peak RSS **6.94 GiB** throughout (< the index).

| search | scan fraction | recall@10 | QPS | speedup vs brute |
|---|--:|--:|--:|--:|
| brute full-scan | 100% | 1.000 | 0.93 | 1× |
| ivf `nprobe=8` | 0.58% | 0.412 | 13.6 | **14.7×** |
| ivf `nprobe=16` | 1.11% | 0.555 | 9.6 | 10.4× |
| ivf `nprobe=32` | 2.03% | 0.702 | 5.4 | 5.8× |
| ivf `nprobe=64` | 3.84% | **0.839** | 3.8 | 4.1× |

**Finer cells → higher recall per row scanned.** Rebuilding the same corpus with
`nlist=8192` (≈ 6.1k rows/cell, near the √N sweet spot) — `build_ivf` **1295 s** (~4×,
since it is `O(N·nlist)`), same 6.9 GiB RSS — and sweeping `nprobe` higher:

| search | scan fraction | recall@10 | QPS | speedup vs brute |
|---|--:|--:|--:|--:|
| brute full-scan | 100% | 1.000 | 0.95 | 1× |
| ivf `nprobe=16` | 0.37% | 0.470 | 16.1 | **17.0×** |
| ivf `nprobe=32` | 0.68% | 0.579 | 8.9 | 9.4× |
| ivf `nprobe=64` | 1.23% | 0.706 | 7.0 | 7.3× |
| ivf `nprobe=128` | 2.28% | 0.832 | 3.6 | 3.8× |
| ivf `nprobe=256` | 4.18% | **0.923** | 2.4 | 2.6× |

At a **matched scan fraction the finer partition wins consistently** — recall **0.92 vs
0.84 at ~4%**, 0.83 vs 0.70 at ~2.3% — because each cell more tightly bounds where the
neighbours are. The price is `build_ivf` time (∝ `nlist`); search touches fewer rows
per probe, so you sweep `nprobe` higher to reach a given scan. Rule of thumb: set
`nlist ≈ √N`, then dial recall/latency with `nprobe`.

- **This is the storage-medium lesson made concrete.** The same search on a 1B index
  over **CephFS** (a network FS) was I/O-bound to impracticality — scoring
  fancy-indexes hundreds of thousands of *random* rows into the codes memmap per
  query × shard, i.e. random page faults over the network (see
  `RESULTS_index_scale.md`). On **local NVMe** those reads are cheap, so the search
  runs at interactive speed and the sublinear scan turns into real wall-clock: brute
  full-scan is **0.93 QPS** (216 s for 200 queries), IVF is **4–15×** faster while
  touching **0.6–4%** of the corpus.
- **Locality optimizations that make NVMe pay:** the fan-out is grouped *by cell* (each
  probed cell's rows + codes read once per shard and scored against all its queries in
  one batched matmul), posting lists are memory-mapped `.npy` (a probe touches only the
  cell slices it needs), and inverted-list members are ascending so code reads within a
  cell are sequential.
- **QPS is the pure-NumPy path** — the win is *rows scanned* (sublinear); a fused
  kernel and a parallel per-shard fan-out lift absolute throughput on top.

## NRP block storage (Linstor) + parallel fan-out

The CephFS failure at 1B was the *network filesystem*, not the index. NRP's **Linstor**
(`linstor-ha`) is local-NVMe-backed block storage (DRBD/XFS) — the same 50M / nlist=8192
index, on a Linstor PVC in an 8-CPU NRP pod, sweeping `--workers` 1 vs 8:

| search | scan | recall@10 | QPS (w=1) | QPS (w=8) | speedup vs brute |
|---|--:|--:|--:|--:|--:|
| brute full-scan | 100% | 1.000 | 0.45 | — | 1× |
| ivf `nprobe=16` | 0.37% | 0.470 | 35.2 | **47.4** | 78–105× |
| ivf `nprobe=64` | 1.23% | 0.706 | 15.2 | 17.4 | 34–38× |
| ivf `nprobe=256` | 4.18% | **0.923** | 7.0 | 8.5 | 16–19× |

- **Block storage removes the wall.** On Linstor the random code reads are cheap, so the
  IVF search runs at **interactive speed (35–47 QPS at nprobe=16, 78–105× over brute)**
  — the same search that was impractical over CephFS. RSS 6.2 GiB (bounded).
- **Parallel per-shard fan-out** (`workers=8`) adds a **modest ~1.2–1.35×** here — only
  10 shards and a small per-query working set, so thread/shard-open overhead eats much
  of the gain (at low `nprobe` it can even lose). It scales with shard count and
  per-query work; the storage medium is by far the dominant factor.

## 1B vectors on Linstor block storage

The billion-scale run: **1,000,000,000 vectors** (dim 32 → PCA 24 → 4-bit,
`nlist=2048`, `--no-originals`), built on a Linstor (`linstor-ha`) block PVC in a
right-sized 6-CPU / 8 GiB NRP pod. Index **38.2 GiB** on disk; **peak RSS 7.45 GiB**
throughout — RAM stays bounded at a billion rows (streaming build + memmap search).
Build streamed in **46 min** + `build_ivf` **31 min** (CPU). Reference = the full-scan
ADC top-k; **brute-force baseline 0.08 QPS** (2542 s/batch).

| nprobe | scan | recall@10 | QPS w=1 | QPS w=8 | speedup w=1 | speedup w=8 |
|---|--:|--:|--:|--:|--:|--:|
| 32 | 1.9% | 0.839 | 0.14 | 0.37 | 1.8× | **4.7×** |
| 64 | 3.6% | 0.928 | 0.13 | 0.32 | 1.7× | 4.1× |
| 128 | 6.9% | **0.976** | 0.12 | 0.26 | 1.5× | 3.3× |
| 256 | 13.3% | **0.9925** | 0.11 | 0.20 | 1.4× | 2.5× |

- **Recall scales cleanly** with `nprobe`: 0.84 → 0.93 → 0.98 → 0.99. `nprobe=128`
  reaches recall **0.976 while scanning just 6.9%** of a billion rows.
- **Fan-out (`workers=8`) is the lever**, a consistent **~2.4–2.6× on top of IVF**.
  Single-worker IVF barely beats brute (1.4–1.8×) — at 1B on block storage,
  random-access latency (not compute) is the wall, so parallel per-shard reads matter.
- **Single-node speedup *decays* as `nprobe` grows** (4.7× → 2.5× at w=8): more probed
  cells = more *scattered* random page-faults, so scan cost outruns the recall gain.
  This is exactly what motivates **hierarchical IVF** (below) — cluster probes into a
  few top cells so reads stay contiguous — and the distributed router (scatter across
  nodes, not just threads). Single-node IVF on block storage tops out ~3–5× brute; the
  1T throughput levers are locality + fan-out across servers.

## Hierarchical IVF (IVF-of-IVF)

`build_ivf(hierarchical=True, top_nlist=…, sub_nlist=…)` builds a **two-level** coarse
quantizer: `top_nlist` top cells, each split into `sub_nlist` leaf cells (leaves stored
top-major, leaf id `top*sub_nlist + sub`). `search(nprobe=…, top_probe=…)` scores the
top cells, keeps the best `top_probe`, then ranks only the leaves under them. Three
wins, all aimed at the 1B findings above:

- **Cheaper assignment** — `O(N·(top_nlist + sub_nlist))` vs the flat `O(N·nlist)`
  (top-assign, then sub-assign within the winning top).
- **Finer quantizer at the same build cost** — the 1T coarse-quality fix.
- **Locality + routing** — the probed leaves provably cluster into `≤ top_probe` top
  cells, so a query reads *contiguous* leaves and, placed top-cell → server, touches
  only a few servers. This is what lets the IVF-as-router coordinator skip servers
  *without* manually cell-aligning shards. Validated in
  `tests/test_sharded_index.py::test_sharded_hierarchical_ivf_matches_and_is_local`
  (recall matches full-scan; probes stay within `top_probe` tops).

## Format v3: packed codes (storage economy)

Index format v3 bit-packs the stored codes (2 codes/byte at 3–4 bit), elides
arange-reconstructible ids + empty tombstones, and writes `uint32` IVF member
sidecars. Packing is a lossless re-encoding of the quantizer levels, so every
ranking is **bit-identical** to v2 (asserted by exact-equality tests — recall
does not change by construction). Measured with this benchmark (2M rows,
dim 32 → PCA 24 → 4-bit, `nlist=1024`, `--no-originals`, Atlas local disk):

| | v2 layout (1B run above) | **v3 measured** |
|---|--:|--:|
| codes | 24 B/row (uint8/dim) | **12 B/row** (packed) |
| ids + tombstones | 9 B/row | **0** (elided, arange) |
| IVF members | 8 B/row (int64) | **4 B/row** (uint32) |
| norms (cnorm+vrnorm) | 8 B/row | 8 B/row |
| **all-in** | **41.0 B/row** (38.2 GiB @ 1B) | **24.1 B/row** (48.2 MB @ 2M) |

The memmap IVF scan gathers *packed* rows and unpacks in-memory (two shift/mask
passes), so the storage-bound random reads shrink by the same 2× the codes do.

## GPU `build_ivf`

The coarse-quantizer assignment is `O(N·nlist)` and is the build wall at scale (~weeks
of NumPy at a trillion rows even for a coarse `nlist`). `build_ivf(device="gpu")` runs
the k-means + assignment on the GPU (CuPy). Atlas GPU 1 (GV100), 10M vectors,
`nlist=4096`:

| build_ivf | time | recall@10 |
|---|--:|--:|
| CPU (NumPy) | 324.1 s | 0.9490 |
| **GPU (CuPy)** | **18.9 s** | 0.9490 |

**17× overall, recall identical.** The assignment matmul itself is ~1000× on the GPU;
the overall figure is Amdahl-capped by the O(N) host steps (normalize, argsort, sidecar
writes) and widens as `nlist` grows. This is what makes a fine `nlist` — hence high
recall at low scan — affordable at a billion-plus rows.

## Notes

- Everything derives from the codes the ADC index already stores (the coarse space is
  `normalize(cent[codes])`), so the partition adds no per-row storage.
- QPS here is the pure-NumPy prototype path; the win is *rows scanned*, which is what
  carries to memmap (a probe faults in only its cells' pages) and to sharding (the same
  best-first order across per-shard IVF). This is the layer that takes search from
  O(N) toward the few-percent scans a trillion-vector index needs.
- Acceptance is recall of the shortlist against the exact ADC ranking — never
  reconstruction cosine.
