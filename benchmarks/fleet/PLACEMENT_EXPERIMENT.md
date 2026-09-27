# Why identical 1T scans took 1,062 to 16,308 s: a placement experiment

Registered before any experiment job ran. The code is `placement_experiment.py` (the
driver, run on Atlas), `placement_scan.py` (the pod) and `fingerprint.py` (the placement
record every fleet job now prints).

## The question

The 1T measurement (`docs/notes/1T_RECALL_RESULT_2026-09-27.md`) ran the same reference
full scan 500 times: one server's 2e9 rows, 100 queries, one CPU per job. Wall time ran from
1,062 to 16,308 s, with a median of 2,664 s. The IVF passes had a p90/p10 of about 1.9 and a
worst case 11 to 13 times the median. The record does not say which node ran each job, so
the spread cannot be attributed after the fact.

Two node properties that could explain it were measured on 2026-09-27
(`RESULTS_storage_probe.md`, turboquant-pro feat/console-v2):

- **Single-core speed.** A fixed SHA-256 benchmark ran at 5,839 to 26,909 hashes/s across
  six nodes, a 4.6x spread. The two slow nodes were in the fullerton and humboldt zones.
- **Distance to the data.** Every 1T volume is one replica in LINSTOR pool `unl`. The cold
  block round trip `r_blk` to such a volume ran from 16 ms (from unl) to 184 ms (from korea).
  Sequential read ran from 257 to 37 MB/s, in the same order.

## The design

- **Servers:** 16 of them, `SERVERS = 11 + 31 i`, i = 0..15.
- **Zones:** each server's scan is re-run once, read-only, in a zone assigned by a seeded
  shuffle (`SEED = 20260927`), so which server goes where is not chosen:
  - unl: 3
  - mizzou: 2
  - sdstate: 1
  - usd: 1
  - mghpcc: 2
  - ucsd-nrp: 2
  - fullerton: 2
  - humboldt: 2
  - korea: 1
- **The computation.** Each pod repeats `fleet_ref.py` exactly: the same index, the same
  query cache, `max_open_shards=2`, `block=65536`, k=10.
- **What each pod records:**
  - its placement record: node, CPU model, hashes/s, `r_blk` over 50 cold first-4 KiB reads of
    the volume's files, and phase marks with CPU seconds and bytes read;
  - its top-10 ids, compared with the partial the 1T run recorded for the same server.
- **Constraints:**
  - Every clone is pinned to one commit.
  - At most 8 jobs run at a time, in the exempt class the 1T pilot measured (1 CPU, 2 GiB).
  - The volume and the shared results are mounted read-only. Nothing is written.
- **Before the 16:** one pilot job on a server outside the 16 checks the pipeline end to end.
  Its numbers are not part of the scored set.

## Predictions

**M1, identity.** Every completed replica returns top-10 ids identical to the 1T partial of
its server. The wall times below are for the same computation.

**M2, the placement model.** Over the completed replicas, a least-squares fit of scan wall
time, `wall = b0 + b1 * (1 / hashes_per_s) + b2 * r_blk_ms`:
- has R^2 >= 0.8;
- has b1 > 0 and b2 > 0;
- has a leave-one-out median absolute error of at most 20 % of the observed wall.

**M3, where the time goes.** Take the scan phase's CPU seconds divided by its wall seconds:
- on nodes with `r_blk < 25 ms` it is at least 0.8;
- it falls as `r_blk` grows: Spearman < -0.5.

So near the data the scan is CPU-bound, and far from it, it waits on I/O.

**M4, fixed cost.** Container start to scan start (pip, clone, fingerprint, index open) is at
most 10 % of the job's wall time, at the median.

**M5, the recommendation.** Replicas in zone `unl` on a fast core (hashes/s >= 20,000) finish
the scan below the 1T run's median of 2,664 s.

**Reported descriptively, not scored:** each server's replica wall time over its recorded 1T
wall time. The replica ran on a known node; the recorded run ran on an unknown one.

**Caveats fixed in advance:**
- One replica per server.
- n of at most 16. Zones that do not schedule a job within 45 min are recorded as unscheduled
  and reduce n.
- One time of day.
- Node variance within a zone is large, and this design does not separate it from the zone.
