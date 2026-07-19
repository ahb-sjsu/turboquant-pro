# DRAFT — NRP large-experiment request: 1T-row compressed-index validation

*Status: draft for Andrew Bond to review and submit via Nautilus Support
(portal form / Matrix), per the platform policy that large experiments
require written approval with workflow details and expected duration. Not
sent.*

## Who / what

**Namespace:** `ssu-atlas-ai` (Andrew H. Bond, San José State University,
andrew.bond@sjsu.edu). **Project:** [turboquant-pro]
(https://github.com/ahb-sjsu/turboquant-pro) — an open-source (MIT)
compressed vector index (bit-packed ADC codes + IVF + distributed
scatter-gather). Goal of this experiment: validate the index's published
scaling claims at **one trillion rows**, producing an open, reproducible
benchmark (all harness code and results are committed to the public repo;
prior runs at 50M/1B on Nautilus are already published in
`benchmarks/RESULTS_ivf.md`).

## What we have already validated on Nautilus (within standing limits)

- **1B-row fleet run (2026-07-19, complete):** distributed build (4 jobs ×
  250M rows, ~31 min wall) onto per-server Linstor volumes at a measured
  **24.01 B/row**; served by 4 NATS shard-servers; routed IVF recall 0.999
  vs the exact distributed full-scan reference; a cold-tier exact rerank
  restoring true fp32 recall from 0.592 to 0.991.
- **10B-row build (in flight):** 8 × 40Gi Linstor volumes (inside the 64Gi
  per-PVC admission limit), two waves of 4 CPU-pegged batch jobs.
- Compliance posture: batch jobs are CPU-saturated start-to-finish with
  requests==limits; serving pods run only in short measurement windows and
  are sized in the exempt class (cpu≤1/mem≤2Gi); no idle pods; TTL cleanup
  on all jobs. (Early serving pods were correctly flagged by utilization
  enforcement; the harness was redesigned around it and the lessons are
  documented in the repo.)

## The 1T request

Using the measured constants (24.01 B/row at 4-bit; build throughput
~1.35M rows/min per CPU):

| resource | ask | basis |
|---|---|---|
| **Block storage (Linstor or RBD)** | **~26 TB** total, as ~64 volumes × ~400 GB (well under the 10 TB/volume platform ceiling; current namespace admission cap is 64Gi/PVC — this is the core exception requested) | 1T × 24 B/row + 10% headroom |
| **CPU** | **~12,500 CPU-hours**, shaped as up to **64 concurrent 6-CPU batch jobs** (384 CPUs) for ~2 days of build, else fewer jobs over proportionally longer | measured build rate |
| **CephFS** | 1 TB (bootstrap artifacts, manifests, results; no fp32 cold store at 1T — acceptance is recall vs the exact distributed ADC reference) | |
| **Duration** | ~1 week end-to-end: ~2 days build, short serving windows for measurement (≤8 exempt-class pods, minutes each), teardown | |
| **GPUs** | none | |

Deliverables back to the community: the complete harness + results published
in the open repo; a writeup of Nautilus as the platform for
trillion-row open-source vector-index benchmarking; storage released
immediately after the measurement windows (nothing archival — consistent
with the 6-month purge policy).

We are happy to adjust shape (fewer concurrent jobs over longer, different
storage class, scheduling windows) to whatever is least disruptive.

## Open questions for NRP

1. Preferred mechanism for the per-PVC admission-limit exception (64Gi →
   ~400 GB × 64 volumes) — namespace label, storage-class grant, or a
   different class (RBD)?
2. Is 64 × 6-CPU concurrent batch acceptable under the utilization policy
   (all pods CPU-pegged, requests==limits), or should we cap lower?
3. Any preferred nodes/regions for sustained CephFS + Linstor throughput
   (we measured large node-to-node variance in CephFS client bandwidth)?
