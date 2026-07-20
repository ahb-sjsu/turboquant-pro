# Systems paper — outline v0.1 (for review)

**Working title:** *Where the Time Actually Goes: An Anatomy of Distributed
Compressed Vector Search at Ten Billion Rows*

**Target:** VLDB / SIGMOD (core or Experiments & Analyses), MLSys.
**Status:** outline for author review. Marked ⬛ = measured and committed,
⬜ = required before submission.

---

## 1. The claim

> **The index was sublinear. The service path was not. The ranking was nearly
> exact. The answer was not.**

Compressed-domain ANN research reports recall-vs-scan-fraction curves and
QPS. At billion-plus scale on commodity shared infrastructure, neither
number means what it appears to mean: throughput is dominated by *per-request
setup* and *storage medium*, not by the scan the curves describe, and recall
against the index's own exact ranking is nearly independent of recall against
the truth the application wants. We build a 10¹⁰-row compressed index on a
shared research cluster, measure the full anatomy, and show that two of our
own previously-published numbers were measuring the wrong thing.

**The contribution is not "our index is faster."** It is that at this scale,
standard search metrics can describe the *algorithm* while failing to
describe the *system being run* — and that the gap is causal and traceable:

> per-PVC admission cap → forced shard proliferation → per-request setup-cost
> explosion → measured latency and even the chosen architecture change.

Infrastructure policy is therefore not an anecdote to be endured but a
variable that alters measured system behaviour, and it can be traced end to
end. This is what §5.4 and §5.8 exist to establish; a list of cluster
annoyances would not be a result.

**Contributions**
1. **Format economics.** A lossless on-disk re-encoding (bit-packed codes,
   elided reconstructible id/tombstone columns, narrowed posting lists) takes
   an ADC index from 41 → **24.01 B/row** with bit-identical rankings, and
   halves code I/O on the memmap scan path. ⬛
2. **Zero-movement distributed build.** A seed-defined corpus lets every
   worker regenerate exactly its own row range, so a distributed build moves
   no corpus bytes; 10B rows built as 8 independent ranges. ⬛
3. **Scale validation.** Routed IVF at 10¹⁰ rows reproduces the exact ADC
   ranking at **recall 0.9988** (nprobe=128; 0.982 at 32), 5.1× faster than
   full scan. ⬛
4. **An anatomy of the wall.** Request latency ≈ *per-request shard opens ×
   shards-per-server* (~3.5 s/shard on network block storage); storage medium
   dominates everything (78–105× CephFS→block); cold-tier random reads cost
   ~0.5 s each regardless of size and serialize per inode. ⬛ diagnosis /
   ⬜ fix + before-after
5. **The acceptance-metric dissociation, and its cost.** Self-fidelity
   (0.999) vs truth-fidelity (0.592) at the same instant, with a
   shortlist-bounded exact rescore restoring 0.991 — and the storage/latency
   price of that rescue. ⬛ (shared with the Blindfold paper — see §8)
6. **Reproducible artifact** on shared infrastructure, including the platform
   constraints (utilization floors, per-PVC caps) that shape what is even
   measurable. ⬛

---

## 2. Background & related work

IVF/PQ/ADC lineage (IVFADC, OPQ, ScaNN anisotropic quantization); graph
indexes (HNSW, DiskANN/SPANN) and their disk-based variants; billion-scale
benchmarking (Big-ANN NeurIPS'21); reranking as standard practice — we claim
*not* novelty for rerank, but for the **measurement discipline** around it.
⬜ Position explicitly against DiskANN/SPANN's I/O accounting: they optimize
SSD random reads per query; we show a distinct per-request *setup* term that
appears when an index is sharded across many files per server.

---

## 3. System under test

Per-node engine (streaming build, memmap search, IVF coarse layer,
hierarchical IVF, GPU coarse quantizer); the v3 storage format; the
distributed layer (scatter-gather, exact merge, IVF-as-router, NATS
transport); the tiered rerank. Short — this is context, not the contribution.

---

## 4. Experimental setup

- **Hardware/platform:** shared national research cluster (Kubernetes),
  per-server network block volumes (Linstor/DRBD) + a shared parallel FS
  (CephFS); 6-CPU batch pods for build/measure. ⬜ Report node heterogeneity
  explicitly: we measured >10× client-bandwidth variance and two wedged nodes.
- **Datasets:** ⬜ **BIGANN/SIFT1B** (1B × 128 uint8) as the primary — in
  acquisition; ⬜ Deep1B or Text2Image-1B as a second distribution;
  ⬛ 15M × 1024-d real Wikipedia Embed-V3 (real queries) for the
  distribution-transfer result; ⬛ a seeded synthetic corpus for the 10B
  scale point and for reproducibility-without-download.
- **Metrics & acceptance:** recall@k against (a) the exact ADC ranking and
  (b) true fp32 neighbours; wall-clock decomposed into open / read / score;
  bytes-per-row on disk; build throughput. **Never reconstruction error.**
- ⬜ **Protocol rigor:** ≥3 repeated trials per cell, medians + IQR, node
  pinning or explicit node-variance reporting, nq ≥ 500 with CIs.

---

## 5. Results

**5.1 Storage economics.** v3 vs v2 at matched rankings; the byte budget
table (codes 12 / norms 8 / members 4 / ids 0); 2-bit variant at 18 B/row. ⬛

**5.2 Build.** 1B in ~31 min distributed (vs 77 min single-node); 10B as two
waves of 4; build rate ≈1.35M rows/min/CPU; GPU coarse quantizer 17×. ⬛
⬜ Add BIGANN build (real data changes PCA/quantizer cost).

**5.3 Search quality at scale.** Recall vs nprobe at 1B and 10B against the
exact ADC ranking (0.982 → 0.9988); the same curve on real embeddings. ⬛
⬜ Same on BIGANN with published ground truth — this is the number the
community can compare.

**5.4 The latency anatomy — the paper's core systems result.**
- Storage medium: 78–105× block vs parallel FS at matched work. ⬛
- Per-request setup: latency ≈ 3.5 s × shards/server; a 1-query request to a
  250-shard server does not complete in 900 s at ~2% CPU. ⬛
- **Our own numbers, corrected:** the 1B run's 0.44–0.61 QPS was
  shard-open latency (50 × 3.5 s ≈ 175 s ≈ observed 163–230 s), not scan
  throughput. ⬛ ← *the honest-reinterpretation hook*
- ⬜ **The fix:** persistent bounded open-shard cache; before/after at 50 and
  250 shards/server; re-measured QPS that finally reflects scanning.
- ⬜ Decomposition figure: open / posting-list read / code gather / score.

**5.5 The cold tier.** Random-read cost ~0.5 s/location regardless of size,
per-inode serialization, and what that does to rerank latency (1186 s → row-
parallel fix); the design implication that a rerank tier needs an object
store or local NVMe, not a shared parallel FS. ⬛ ⬜ quantified before/after.

**5.6 Acceptance metrics.** The dissociation triplet at 1B (synthetic) and
15M (real): 0.999/0.592/0.991 and 0.990/0.689/0.981. ⬛
⬜ On BIGANN, where "truth" is the published ground truth — the cleanest
possible version of this experiment.

**5.7 Baselines.** ⬜ **Required.** FAISS IVF-PQ and DiskANN at matched
recall on the same pods and volumes; ScaNN if it will run in this
environment. Report recall/latency/bytes-per-row and be explicit where we
lose (we expect to lose on QPS; the contribution is the anatomy and the
economics, not beating tuned kernels).

**5.8 Platform effects as first-class results.** Utilization floors delete
under-busy serving pods; per-PVC admission caps reshape a 10B layout into 8
volumes; 6-month volume purges. These determine what a shared-infrastructure
system can be. ⬛ — unusual for a paper, and (we argue) exactly what makes
shared-cluster results reproducible by others.

---

## 6. Discussion — design rules extracted

Fewer, larger shards per server; keep hot codes on block storage and cold
originals somewhere random-read-capable; make acceptance consumer-relative;
budget the reference oracle (an exact scan is hours at 10B and must be
scattered); expect the platform, not the algorithm, to set the shape.

---

## 7. Threats to validity (write this section honestly and early)

Shared cluster timing variance; single index family (our own — hence §5.7);
no online-serving workload (batch measurement); QPS not competitive by design
in the pure-NumPy path.

**The synthetic corpus is geometrically unlike real embeddings — measured,
not suspected.** Under the RC-1 battery (ahb-sjsu/openvector-bench, PREREG RC-1 v2),
the low-rank+noise recipe behind our 1B/10B runs separates clearly from real
Wikipedia Embed-V3 on the properties that govern ANN difficulty. The sharpest
separator is PCA-256 neighbour retention — ≈0.99 for the synthetic corpus vs
≈0.62 for real embeddings — i.e. the synthetic corpus is trivially
compressible in a way real data is not; intrinsic dimension and hubness also
separate. *(Numbers pending the full registered grid; an earlier smoke run's
figures are withdrawn because they were measured under raw-Euclidean with
queries drawn from the base, which the registered metric freeze and the
held-out-query battery both replace — one comparison reverses sign under the
corrected method, which is itself a caution worth reporting.)*

**Consequence, stated plainly:** the 10B results are valid as *systems*
measurements — I/O, storage economics, build scaling, self-fidelity — and are
**not** evidence about retrieval difficulty on real data. Retrieval-difficulty
claims rest on the real-embedding (10⁷) and BIGANN (10⁹) points, and closing
this gap at scale is the motivation for the procedural reference corpus.
**Note the dependency:** any future use of a procedural corpus at 10¹²
inherits whatever RC-1/RC-2 establish, and no more — the extrapolation from
2×10⁵ to 10¹² is an assumption, and is labelled as one.

---

## 8. Relationship to the companion papers (avoid salami-slicing)

- *Keep the Angle* (theory) — spectral basis; cited only.
- *The Quantizer's Blindfold* — owns the **acceptance-metric thesis** and its
  cross-consumer convergence (retrieval + attention softmax on a real LLM).
- **This paper** — owns **scale, storage, build, and the latency anatomy**.
  §5.6 appears here only as a systems cost argument (what the rescue costs in
  bytes and milliseconds) and cites Blindfold for the claim itself.
  ⬜ Decide before submission whether §5.6 stays at all.

---

## 9. Artifact

Public repo, MIT: harness (`benchmarks/fleet/`), seeded corpus (no download
needed to reproduce the scale points), manifests, result JSONs, and the
platform notes. ⬜ Add a one-command reproduction path + a scaled-down
variant that runs on a laptop.

---

## Critical path to submission

1. ⬜ BIGANN 1B: build, recall vs published GT, latency decomposition.
2. ⬜ Shard-open fix + before/after (turns our biggest embarrassment into
   §5.4's contribution).
3. ⬜ Baselines: FAISS IVF-PQ, DiskANN.
4. ⬜ Repeated trials + node-variance protocol.
5. ⬜ Figures: byte budget; recall-vs-nprobe at 1B/10B/BIGANN; latency
   decomposition; before/after the fix.
6. ⬜ Decide §5.6's fate; finalize the split with Blindfold.
