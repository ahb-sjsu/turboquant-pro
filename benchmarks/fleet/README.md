# NRP multi-node fleet experiments (1B, 10B, 100B done · 1T needs approval)

The scaling-roadmap "production multi-node fleet run" and its follow-ups, on
NRP Nautilus (`ssu-atlas-ai`). Everything here is reproducible from this
directory: the corpus is **seed-defined** (`fleet_common.gen_block`), so any
worker regenerates any row-range bit-identically — the distributed build moves
zero corpus bytes.

## Protocol (common)

- **Corpus**: per-global-shard seeded low-rank blocks, 5M rows × dim 32
  (`rng = default_rng(777000 + gshard)`), identical recipe to
  `bench_ivf_sharded`. PCA 24, 4-bit, **index format v3** (24.01 B/row all-in,
  measured).
- **Shared artifacts** (CephFS `tqp-fleet-shared`): bootstrap = global shard 0
  → the one PCA basis every `write_shard(basis_from=…)` reuses + the global
  `nlist=2048` coarse quantizer; every server adopts the bootstrap *radius*
  file so the whole fleet probes identical cells.
- **Hot tier**: one Linstor (block) RWO PVC per server — the storage law
  (`RESULTS_index_scale.md`): random-access codes need block storage, not a
  network FS.
- **Acceptance**: recall of shortlists vs reference rankings — the exact
  distributed ADC full-scan merge, and (where a cold store exists) true fp32
  ground truth. Never reconstruction cosine.
- **Platform compliance**: batch jobs are CPU-pegged start-to-finish
  (`cpu=6/8Gi`, requests==limits, ≤4 at a time, `backoffLimit: 0`, TTL
  cleanup, no sleep). Serving pods are **traffic-shaped** and therefore sized
  into the platform's exempt class (`cpu≤1, mem≤2Gi`) and exist only inside
  short measurement windows (`fleet_driver.sh` tears them down on any exit).
  Utilization thresholds (20–200% CPU of request) are enforced by workload
  deletion — measured first-hand.

## Run 1: 1B — COMPLETE (2026-07-19)

4 × 250M rows; build ~31 min wall (vs 77 min single-node v2); 6.0 GB/server.
Cold store: 200 per-shard fp32 `.npy` on CephFS (128 GB, single-writer files).
Ground truth: scattered over 4 jobs + exact top-K merge. Serve: 4 warm
`nats_worker` pods on the `atlas-nats` leaf fabric; coordinator in-cluster.

Results (`results/fleet_run_1B.json`, also `../RESULTS_ivf.md`): routed IVF
`nprobe=128` recall **0.999** vs the exact ADC merge; ADC-only true recall
**0.592**; **tiered rerank 0.991** — the compressed-domain ceiling broken from
the cold tier at 1B.

## Run 2: 10B — the largest literal step inside standing limits

Same recipe ×10, reshaped by a measured admission limit: namespace Gatekeeper
caps `linstor-ha` PVCs at **64Gi apiece**, so 10B spans **8 volumes × 40Gi**
(`tqp-fleet-10b-{0..7}`, 1.25B rows ≈ 30 GB each,
`TQP_SHARDS_PER_SERVER=250`), built in **two waves of 4** CPU-pegged jobs
(heavy-pod cap) — the same wave scheduling a 1T build would use. **No cold
store** (1.28 TB of fp32 originals exceeds quota; rerank/true-recall are
established at 1B). Build: `pvc_10b.yaml` + `job_build_10b.yaml`, ~2.5 h per
job, ~5 h total. Measurement plan (post-build, short exempt-class serve
window, 8 endpoints): fleet full-scan ADC reference at reduced `nq=25`,
routed `nprobe` sweep at `nq=100` vs that reference, per-node scan-rate
constants for the 1T projection.

## Run 3: 100B — COMPLETE (2026-07-28)

50 shard-ranges × 2B rows on 64Gi Linstor RWO volumes, 400 shards per range,
built in waves of 8. Results in `results/fleet_run_100B.json`.

| nprobe | recall vs exact ADC full-scan | speedup | wall/server median |
|---|---|---|---|
| 32 | **0.9834** | 14.7× | 1,245 s |
| 128 | **0.9986** | 8.5× | 1,956 s |

Reference full-scan is 19,041 s per server at the median (range 5,935–41,100).
Recall at 100B is within 0.0002 of the 10B figure at the same `nprobe`, so
routing quality is flat across two more orders of magnitude. No cold store at
this scale, so true-fp32 recall and tiered rerank remain established at 1B
only.

**Read the wall-clock with care.** Server 11 took 6,892 s at `nprobe=32` and
16,467 s at `nprobe=128`, which is 5.5× and 8.4× the median. Fleet wall-clock
is set by that one node. It is node variance on a shared cluster rather than a
property of the method, and the median column above is the honest per-server
cost.

**What it took to finish.** The run aborted four times before this one. NRP's
utilization-floor sweep deletes job objects that sit below 20% CPU during
image pull, and pods are preempted roughly every two hours while a 400-shard
build needs four to five. The driver only converged after two fixes: per-shard
resume sidecars in `fleet_build.py` (891ed60) so a restarted pod rebuilds only
what is missing, and `build_ivf(resume=True)` (9c29b0e) so the ~84 min
non-resumable finalize could survive preemption. A big run here is not a long
job, it is a job that has to be restartable at every level.

## Run 4: 1T — requires written approval (see `docs/notes/NRP_SCALE_REQUEST.md`)

Measured constants → 1T ≈ **24 TB** hot tier at 4-bit (18 TB at 2-bit +
rerank; Linstor caps volumes at 10 TB, so ~40–80 per-server volumes of
0.3–0.6 TB) and ≈ **12,300 CPU-hours** of build. Both are beyond standing
namespace limits; platform policy requires written approval with workflow and
duration for large experiments. Remaining engineering before 1T:
cell-aligned (hierarchical top-cell → server) placement for cross-server
routing sparsity; batched per-shard cold-tier fetch; the 64/256-endpoint
topology validation in exempt-class pods.

## Operational lessons (hard-won, all encoded in the scripts)

1. **CephFS**: never mmap-for-write; one writer per file (a shared 128 GB
   `.npy` corrupted twice — unaligned slice boundaries share pages across
   clients); sequential `np.load` ≈ 10× faster than mmap-fault reads; quota
   counts sparse file *sizes* and drains asynchronously after deletes.
2. **Node variance is extreme** (~170 KB/s to >100 MB/s CephFS): scatter
   long reads across jobs (exact merge), pin re-rolls to known-fast nodes.
3. **Fleets need a readiness barrier** (probe every endpoint) and a
   **bounded coordinator fan-out** (concurrent full-scans OOM shared pods).
4. **Background pollers die; detached drivers survive** (`setsid` +
   log-file, poll one-shot).
