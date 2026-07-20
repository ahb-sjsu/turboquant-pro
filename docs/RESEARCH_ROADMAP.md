# Research roadmap

*A program, not a backlog: open questions, what each experiment would settle,
and what would make us stop. Spans `turboquant-pro` (the engine and the systems
paper) and `openvector-bench` (the benchmark). Last updated 2026-07-20.*

Status vocabulary — **measured** (data in hand, committed) · **assumed**
(believed, not tested) · **contingent** (predicted, depends on unfinished work)
· **withdrawn** (claimed earlier, retracted on evidence).

---

## 1. The questions

**Q1 — What actually limits distributed vector search at 10⁹⁺?**
*Measured, on network storage:* per-transaction costs dominate —
*Withdrawn:* that the placement experiment evidences this. Re-measurement
showed both placements open every shard; its **42–47×** (corrected up from 37×,
and now with repeated trials) is scan **fragmentation/locality** on a single
box — ~59k fragments of ~153 rows vs ~928 of ~9,752 — at identical rows and
FLOPs. Real result, different mechanism. Per-transaction costs: ~3.5 s/shard-open on network
block, ~0.5 s/location on the shared FS regardless of size, 78–105× CephFS→block.
*Withdrawn:* that this framing is novel — the ANN literature (DiskANN, SPANN,
FlashANNS) already reasons per-transaction on SSD, and the HPC metadata-wall
literature already owns file-open cost. **What survives is the join**: nobody
appears to have run a billion-scale index on a distributed parallel filesystem
and shown the metadata wall sets query latency.

**Q2 — Does the compressed index answer the question the application asked?**
*Measured:* no, not by itself. At 1B: 0.999 recall against its own exact
ranking, **0.592** against fp32 truth, **0.991** after a bounded cold-tier
rescore. Transfers to real embeddings (15M Wikipedia: 0.990 / 0.689 / 0.981)
and to a second consumer family (attention softmax on Qwen2.5-1.5B, where
configs with identical quantizer ratings differ 17 points in behavioural
agreement).

**Q3 — Can a benchmark exist at scales nobody can download?**
*Assumed, being tested.* Corpus defined by a signed Merkle manifest;
regeneration verified by hash, falling back to byte fetch. Ranged fetch is
**measured** byte-identical to canonical. Untested: the full reconstruction
experiment.

**Q4 — Is a procedurally generated corpus legitimate for systems benchmarking?**
*Measured (round 1):* the battery discriminates — all three frozen nulls
rejected. **Our own synthetic corpus fails it**: ~2× the intrinsic dimension
of real embeddings, one-seventh the hubness, PCA-256 retaining 0.99 of
neighbours vs 0.62. So the 10B results are valid as *systems* measurements and
say nothing about retrieval difficulty.
*Open:* whether any generator can pass (RC-1), and whether passing predicts
unfitted ANN behaviour (RC-2).

**Q5 — What does a given recall cost, in dollars?**
*Assumed.* Model specified (§6 of the systems outline); constants measured;
prices are public list inputs, never presented as our bill.

---

## 2. Dependency chain

```
   BIGANN acquired ──┐
   L2 metric ────────┤
                     ├─→ E1 BIGANN 1B build + recall vs published GT ──┐
   v3 format ────────┘                                                 │
                                                                       ├─→ systems paper
   E2 placement (37×, done) ──→ E3 shard-size sweep S* ≈ nprobe ───────┤
                                                                       │
   E4 shard-open cache (the fix) ──→ E5 before/after + true QPS ───────┤
                                                                       │
   E6 baselines: FAISS IVF-PQ, DiskANN ────────────────────────────────┘

   RC-1 battery (done) ──→ E7 fit generator ──→ E8 seal ──→ E9 RC-2 ──┐
                                                                      ├─→ benchmark release
   E10 reconstruction experiment (T7) ────────────────────────────────┤
   E11 L1 ground truth + strata, E12 L2 structural labels ────────────┘

   E13 design-space map (running) ──→ E14 Pareto frontier ──→ E15 cost model
```

**Critical path to the systems paper:** E1 → E6 (baselines are the reviewer's
first question) with E4/E5 close behind, because our published QPS figures
currently measure shard-open latency rather than throughput and must be
re-stated.

**Critical path to the benchmark:** E7 → E8 → E9. If RC-2 fails, the family
stops at the real/procedural seam and says so.

---

## 3. Deliverables

| deliverable | gate | blocking work |
|---|---|---|
| **Systems paper** (VLDB/SIGMOD E&A, MLSys) | baselines + real public data + the fix measured | E1, E4–E6, repeated trials |
| **Blindfold paper** (acceptance metrics) | third consumer family or a sharper second | mostly written; needs the BIGANN version of Q2 |
| **OpenVector Bench** (public release) | RC-2 pass **or** an honest stop at the seam | E7–E12 |
| **1T run** | NRP approval (requested 2026-07-20) + E3/E4 | not started; do not start before the fix |

---

## 4. Budget

Measured constants: **24.01 B/row** (4-bit), **~1.35M rows/min/CPU** build.

| item | cost |
|---|---|
| BIGANN 1B build | ~44 GB, ~2–3 h on Atlas |
| design-space screen (10M) | ~20 builds, ~4 h |
| RC-1 generator fit + seal | GPU-hours, small |
| 1T (if approved) | ~26 TB block, ~12,500 CPU-h, ~1 week |
| ground truth per tier | one exhaustive pass — the dominant cost of the benchmark |

Atlas does the iteration (data local, GPU free, no policy friction); NRP does
scale-out. Nothing runs on the laptop.

---

## 5. Kill criteria — stated in advance

- **RC-1/RC-2 fail** → the procedural corpus is published as a random-number
  benchmark, the family stops at 10⁹, and the tiers above the seam are not
  released. (Registered in `spec/PREREG_RC1.md`; binding.)
- **Baselines show we lose on the anatomy, not just QPS** → the systems paper
  becomes an experience/measurement report, not a design claim.
- **The shard-open fix does not move latency** → the transaction model is
  wrong in a way the retrodiction hid, and §1.0 must be rewritten around
  whatever does move it.
- **1T not approved** → the paper stands on 10⁷–10¹⁰ with an honest
  extrapolation; no synthetic mega-number is published as a headline.

---

## 6. Open anomalies (unexplained; do not build on these)

- ~~Placement timing at 50M~~ **RESOLVED**: single-shot timing on a contended
  box, with 5–11× run-to-run variance; two draws coinciding at 14.87 s is
  unremarkable at that spread. Search results were bit-identical across all
  trials, so the *search* is deterministic; only the clock varied. Harness now
  runs repeated trials (warm-up discarded, min/median/max reported).
  Corrected warm minima: **0.835 / 2.196 / 4.155 s** at nprobe 8/32/128.
- ~~Superlinear speedup~~ **RESOLVED, and it changes the mechanism**: not
  superlinear in anything causal. Both placements open all 64 shards and scan
  identical rows; the difference is **fragmentation** (63.7× ratio, bracketing
  the 42–47× observed). Wall time tracks rows scanned, not shards touched.
- **NEW — routing sparsity is computed and discarded**: the router derives
  cell→server placement, then `_ivf_scan_shards` opens every shard anyway.
  Making the scan skip shards with no probed cells would convert the routing
  metric into an actual transaction saving. Untested; the honest test of the
  transaction thesis. ⬜
- **NEW — cold/warm is a 22× effect** at nprobe=8 (22.30 s cold vs 0.835 s
  warm). Every previously reported placement number was fully warm and must be
  labelled as such.
- **G5 (relative contrast) discriminates nothing** in the RC-1 battery, despite
  being a standard ANN-hardness predictor and a *mandatory* gate. Reported, not
  removed.
