# Storage on NRP, observed with known I/O

This applies the method of `RESULTS_fabric_leaf.md` to storage. A pod issues I/O whose
every byte and operation it counts, and each observer reports on that same I/O:

- **the application**: bytes and operations issued, and wall time;
- **the kernel's per-process accounting** (`/proc/self/io`): what reached the storage layer;
- **the filesystem** (`statvfs`): used space, sampled after each phase;
- **the cluster metrics API**: the pod's CPU and memory, sampled from Atlas. This is the
  observer NRP's utilisation enforcement uses.

Code: `benchmarks/storage/probe.py` (the pod) and `benchmarks/storage/run_storage.py`
(the driver, run on Atlas).

Every probe runs in the exempt class, on a fresh 5 Gi PVC that is deleted afterwards. It is
submitted through nats-bursting (the Go controller mounts the claim), one probe at a time.
Each phase is bounded by bytes and by seconds, and no phase waits on a timer.

## Pilot, 2026-09-27: rook-cephfs, read from Korea

The scheduler placed the pilot on `yge-nrp-01.kreonet.net` (zone `korea`), across the
Pacific from the Ceph cluster. It completed, was cleaned up, and gave the first
observations:

| phase | observed |
|---|---|
| sequential write, 1 GiB | 27 MB/s. `statvfs` used space rose by exactly 1024.0 MiB |
| fdatasync of 4 KiB | p50 248 ms, p99 1.03 s, max 5.8 s |
| sequential read, cold | 8.2 MB/s. The 90 s bound stopped it at 716 MiB |
| random 4 KiB reads, cold | 0.68 IOPS, p50 1.87 s. The bound stopped it at 62 reads |
| metadata (500 files) | create 6.7/s, stat 116k/s (cached), unlink 6.7/s |
| cleanup | used space returned to 0.0 MiB at once |

The observers disagree, and each disagreement shows what that observer can see:

- **The kernel's block counters are blind to CephFS reads.** `read_bytes` stayed at 0
  through 716 MiB of reads; only `rchar` (bytes returned by read calls) saw them. The
  CephFS kernel client fetches over the network, so nothing reaches the block layer.
  Writes are counted, because `write_bytes` is charged at dirtying time.
- **The cluster metrics observer averages over a 3-minute window.**
  - It reported 0.001 to 0.006 cores and 15 to 24 MiB.
  - The pod accounted 2.67 CPU-seconds over 442 s, a mean of 0.006 cores, and 29 MiB peak RSS.
  - The first sample arrived 107 s after the pod started.
  - A pod shorter than that window is seen smeared, or not seen at all.

## Class sweep, pinned to zone ucsd-nrp, 2026-09-27

The pilot's latencies are dominated by where the pod ran, so the classes are compared from
one location. All five probes succeeded on the same node, `node-2-10.sdsc.optiputer.net`.
Each wrote 1 GiB, and every read was cold. Records:
`storage/results/classes_ucsd-nrp_20260927/`. Analysis:
`storage/results/analysis_20260927.json`.

| class | filesystem in the pod | write MB/s | fdatasync p50 / p99 ms | cold sequential read MB/s | random 4 KiB read p50 ms (reads done) | create / unlink ms |
|---|---|---|---|---|---|---|
| rook-cephfs | ceph | 147 | 38 / 277 | 88 | 54 (1491) | 9.9 / 9.8 |
| rook-cephfs-east | ceph | 48 | 122 / 362 | 9.5 (820 MiB in 90 s) | 642 (104) | 61 / 61 |
| rook-ceph-block | xfs | 111 | 78 / 441 | 60 | 8.0 (2000) | 0.05 / 0.009 |
| linstor-ha | xfs | 69 | 87 / 199 | 137 | 43 (2000) | 0.06 / 0.009 |
| linstor-unl | xfs | 119 | 87 / 90 | 147 | 43 (2000) | 0.12 / 0.009 |

**What each observer saw**

1. **Block-layer accounting depends on the filesystem.**
   - On the three xfs block volumes (Ceph RBD and LINSTOR), the kernel's `read_bytes` and
     `write_bytes` equal the bytes issued (ratio 1.000).
   - On both CephFS classes, `read_bytes` is 0, and only `rchar` sees the reads.
2. **Space accounting is exact in amount but late in time.**
   - On xfs, used space rose by exactly the bytes written. It still read +1024.8 MiB right
     after the delete: xfs frees space in the background after a delete.
   - On rook-cephfs, used space read +704 MiB right after the 1 GiB write, not yet the whole.
3. **The enforcement observer misses short pods.** The cluster metrics API averages over 3
   minutes.
   - It never sampled the 68 s rook-ceph-block probe.
   - It read 0.0000 cores for linstor-unl, whose own accounting was 0.018.
4. **Latency is set by metadata round trips.**
   - A file create costs 0.05 to 0.12 ms on block volumes against 10 to 61 ms on CephFS.
   - A random 4 KiB read costs 8 ms on Ceph RBD and 43 ms on LINSTOR.
   - Seen from the west, rook-cephfs-east is 6 to 12x slower than rook-cephfs on every latency.

S3 is not included. Creating an ObjectBucketClaim (`rook-ceph-bucket-central`) is forbidden
to this namespace, and no S3 keys exist on Atlas. S3 needs keys issued through the NRP portal.

## Location sweep: the same object seen from different places

**The Observation Theory framing.** Keep the object fixed: one `rook-cephfs` volume, shared
(RWX) by every probe. Run the identical probe from observers placed at different points of
the network: `ucsd-nrp`, `fullerton`, `humboldt`, `unl`, `mghpcc` and `korea`, in turn. What
changes between observations is only the observer's position. So each observable is either:

- **invariant**: a property of the object, the same from everywhere;
- **covariant**: it changes with the observer, ideally through a small number of
  coordinates;
- **unobservable** to a given instrument: that instrument's quotient.

**Predictions, fixed before the sweep runs:**

- **P1, invariance.** At every location, `statvfs` used space rises by exactly the bytes
  written (to the MiB) and returns to its baseline after the delete. Storage accounting is a
  property of the object, not of the observer.
- **P2, a location-independent blind spot.** At every location, the kernel's `read_bytes`
  stays at 0 while `rchar` equals the bytes read. The block-layer observer's quotient
  removes network filesystem reads wherever it stands.
- **P3, one coordinate.** Take a location coordinate `r`: the mean latency of one metadata
  create, a single synchronous round trip to the metadata server, measured inside each probe.
  Across locations, fdatasync p50, random-read p50 and unlink latency should each be
  approximately affine in `r`, with Pearson correlation above 0.9 over the locations that
  complete. So position enters every latency-bound observable through one parameter.
  Sequential throughput is bandwidth-bound and is not predicted to follow.
- **P4, observer resolution.** The cluster metrics observer's CPU for each pod stays within
  a factor of 3 of the pod's own lifetime mean CPU. Any phase shorter than the metrics window
  is not separately visible to it.

**Honesty about the design.** One probe per location, taken sequentially, at one time of
day. Node-to-node variance within a zone is known to be large, up to 1000x on CephFS reads
(`benchmarks/fleet/README.md`). Correlations over six points are reported with their n and
are evidence, not proof.

## Location sweep, 2026-09-27: results

One rook-cephfs volume was shared by every probe. Probes ran in turn from ucsd-nrp,
fullerton, humboldt, unl, mghpcc and korea. Records:
`storage/results/locations_rook-cephfs_20260927/`.

**unl did not complete.**
- The pod ran on `hcc-nrp-sec-c1109.unl.edu`. It was still running at the driver's 900 s
  limit, using 0.003 cores and 23.5 MiB, and was deleted without output. The probe prints only
  at the end.
- Its termination took longer than 300 s.
- The metadata phase was not bounded by time: 500 creates, where every other phase stopped at
  90 s. It is the likely place it stalled, but that is not confirmed. That phase is now bounded
  like the others.
- n = 5 below.

| zone | node | r = create ms | fdatasync p50 ms | random 4 KiB p50 ms | unlink ms | write MB/s | cold read MB/s |
|---|---|---|---|---|---|---|---|
| ucsd-nrp | node-2-10.sdsc.optiputer.net | 9.8 | 21.6 | 56 | 9.9 | 138 | 89 |
| fullerton | nautilus-it-cpu14.fullerton.edu | 11.1 | 24.9 | 78 | 10.9 | 71 | 77 |
| humboldt | cph-blade15.humboldt.edu | 22.6 | 37.9 | 129 | 22.5 | 95 | 65 |
| mghpcc | service-01.nrp.mghpcc.org | 69.6 | 86.6 | 470 | 68.6 | 48 | 24 |
| korea | yge-nrp-01.kreonet.net | 152.3 | 173.6 | 1766 | 152.5 | 26 | 8.7 |

**Scores of the predictions fixed before the sweep**

**P1, invariance: FAILED as stated.**
- The amount is invariant: every probe's written GiB was eventually counted.
- The timing is not:
  - Right after the write, used space read +1024 MiB at ucsd-nrp, fullerton and mghpcc, but
    +704 MiB at humboldt and korea.
  - Right after the delete, it still read +1024 MiB everywhere.
- CephFS counts space asynchronously. So used space is a property of the object only in the
  limit. Read at a given instant, it depends on how soon the observer looks. This matches the
  fleet run's quota errors after deletes.

**P2, a location-independent blind spot: HELD.**
- At every location, the kernel's `read_bytes` was 0 while `rchar` covered the bytes read.
- The block-layer observer removes network-filesystem reads from wherever it stands.

**P3, one coordinate: HELD.**
- Pearson correlation with r = create latency, over n = 5:
  - fdatasync p50: 0.9999
  - random-read p50: 0.983
  - unlink: 0.99997
- Spearman is 1.0 for all three.
- Leaving any one location out keeps every Pearson value at or above 0.98. Korea's leverage
  does not produce the result.
- Least-squares fits:
  - fdatasync p50 is about 12.8 ms + 1.06 r: one metadata-server round trip plus a fixed cost.
  - random 4 KiB p50 is about 11.8 r - 127 ms: each cold random read costs about twelve round
    trips. The likely reason is that the CephFS client fetches a large extent per miss, so the
    transfer is window-limited. That explanation is not measured here.
- Caveat: unlink and create are both one metadata round trip, so their correlation is close to
  true by construction. The informative results are fdatasync and random reads.
- Not predicted, but observed: throughput falls with r. Spearman is -0.9 for write and -1.0
  for read, as a bandwidth-delay bound would give.
- Position enters every latency-bound observable here through one number.

**P4, the metrics observer within 3x: FAILED.**
- The ratio of the cluster metrics' CPU to the pod's own lifetime mean was 0.29 to 1.10 across
  the five probes.
- It under-reported by up to 3.4x (humboldt), and it saw 2 to 28 samples depending only on how
  long the pod lived.
- In the class sweep, it never sampled a 68 s pod at all.
- The observer that NRP's utilisation enforcement uses cannot resolve short workloads. The
  shorter the pod, the less it sees and the lower it reads. This is the same property that
  swept the fleet's image-pull windows.

**In Observation Theory terms.** Of the five observers, placed at five points of the network:
- **Invariant:** the eventual space accounting (P1's amount) and the kernel's blindness to
  network reads (P2).
- **Covariant through one coordinate:** every latency (P3), with r a single metadata round trip
  measured by the observer itself.
- **Resolution-limited:** the cluster metrics observer (P4). Its quotient is everything shorter
  than its window.
- The failures are as informative as the passes: P1 separates the amount from the timing, and
  P4 bounds what enforcement can know.

**Limits.**
- One probe per location, sequential, at one time of day.
- One node per zone. Node variance within a zone is known to be large.
- n = 5 after unl.
- The results are consistent with the one-coordinate account at these five points. They do not
  establish it in general.

## Rehabilitation of P1 and P4, and reuse of the 1T volumes: predictions before the runs

P1 and P4 failed as stated. Rehabilitation here does not re-score the old data against a
looser bar. Each failure is restated as the sharper claim the failure pointed to, and that
claim is tested on fresh runs. The probe gains three things for this: a settle loop after the
write and after the delete, a known CPU load, and the pod's own CPU timeline. The predictions
below are committed before any of those runs.

### R1, replacing P1: the amount is invariant, and its timing is not an observer coordinate

**How it is measured.** After the write, and again after the delete, the probe polls
`statvfs` until used space matches within 1 MiB, bounded at 120 s. Each poll is a round trip
to the filesystem's server, so the loop never waits on a timer.

- **R1a.** At every location, used space converges to exactly the bytes written, within
  1 MiB, inside 120 s, and returns to the baseline inside 120 s after the delete.
- **R1b.** How long that takes is set by the filesystem's accounting cadence, not by where the
  observer stands: |Spearman(settle time, r)| < 0.6 over the locations that complete. This is
  exploratory, with n of about 6.

### R4, replacing P4: the metrics observer is a trailing-window average, and nothing else

**How it is measured.** The pod records its own CPU seconds at every phase boundary, and
about twice a second through a 360 s phase that keeps one core busy. That gives the true
CPU history c(t). The driver records each metrics sample's own timestamp `ts` and `window` W.

- **R4a.** Every sample whose window lies inside the pod's life equals the pod's own mean
  over the window, (c(ts) - c(ts - W)) / W, within 0.1 cores.
- **R4b.** Samples whose window lies wholly inside the busy phase read 1.0 ± 0.1 cores.

If R4 holds, P4's failure is explained by the observer's window and not by an error in what it
measures. Enforcement can then be predicted from the workload's own timeline.

### L, reusing the 1T volumes before they are released: the same object seen from many places

**The volumes.** 500 PVCs `tqp-fleet-1t-0..499` of `linstor-unl`, each 56 GiB. That is
28,000 GiB, or 27.34 TiB, provisioned. Each is one replica (`autoPlace: 1`) in LINSTOR storage
pool `unl`, and together they hold the 24.0 TB (decimal) 1T index. Every volume holds the same
structure: 400 shards built by one generator. So the volumes are copies of one object, all
stored at UNL, and a pod anywhere reads them over the network unless it runs where the replica
is.

**The design.** Two volumes, `tqp-fleet-1t-17` and `tqp-fleet-1t-311`, are each read
**read-only** (mount `ro`; the probe refuses a writable mount) from zones `unl`, `ucsd-nrp`,
`fullerton`, `humboldt`, `mghpcc` and `korea`. That is a crossed design, object by observer.
Nothing is written. A claim that any pod is using is skipped.

**Measurements.**
- r_blk, the location coordinate: the median cold latency of one 4 KiB read at offset 0 of 50
  distinct shard files.
- Up to 1 GiB of cold sequential read.
- 2,000 cold random 4 KiB reads across all the files.

**Predictions.**
- **L1, one coordinate for block storage.** For each volume, random-read p50 has Pearson > 0.9
  with r_blk across zones, and sequential MB/s has Spearman < -0.8 with r_blk.
- **L2, the observer dominates the object.** At a fixed zone, the two volumes differ by less
  than 1.3x on r_blk, random p50 and sequential MB/s. Across zones, each of those spans more
  than 3x.
- **L3, invariance.** Both observers of a volume see the same `statvfs` used bytes, exactly.
  The volumes are read-only and nothing changes them.
- **L4, the block counter sees block reads.** On these xfs volumes, the kernel's `read_bytes`
  for the sequential read equals the application's bytes within 5 %, at every location. This is
  P2's counterpart on block storage.

**What the 1T record cannot give.** The run's per-server wall times (median 2,664 s for the
reference scan, range 1,062 to 16,308 s) are the same workload measured 500 times. But the
record does not keep which node each job ran on, so that variance cannot be attributed to
observer position after the fact. The driver should log `spec.nodeName` for each job. The
crossed probe above measures the attribution directly, on two of the same volumes.

### R4', registered after the first rehabilitation probe and scored only on the other zones

**Why a new prediction exists.** The first rehabilitation probe (ucsd-nrp) arrived before the
other five. Its metrics samples show that R4 as registered measures against the wrong clock.
- The `timestamp` the metrics API returns changes on every query.
- The value itself changes only about every 56 to 60 s: 0.534, 0.978, 1.129, 0.972 and 0.887
  were each repeated for 5 to 6 queries.
- So `timestamp` is not the time the value was computed. On ucsd-nrp, R4 fails as registered:
  max |error| is 0.196 cores, and the busy phase reads 0.887 to 1.129.
- That failure is kept on the record.

**R4', a mechanism fitted on ucsd-nrp only.** The observer computes v = Δc / W, with W = 180 s
nominal. Δc is the pod's CPU seconds over an actual interval whose length can differ from W by
up to 30 s. The interval ends at a scrape time within the 60 s before the value first appears.
The API then repeats v until the next scrape. So:

- **R4'a.** For every distinct reading v in a held-out zone, some interval [s - W', s] exists
  with s in [t_first - 60, t_first] and W' in [150, 210], such that
  |v - (c(s) - c(s - W')) / W| ≤ 0.02, using the pod's own CPU history c.
- **R4'b.** Every distinct reading whose possible windows all lie inside the busy phase is in
  [0.83, 1.17]. That is the ±30 s window jitter on a truly 1.0-core load.
- **How it is scored.** Only on fullerton, humboldt, unl, mghpcc and korea. None of those
  records had been looked at when this was committed.

**R4'', corrected on the fit zone before any held-out record was opened.** Scoring R4' on
ucsd-nrp itself, still fit data, showed two misses. They are the first two readings (0.018 and
0.024 cores, first seen 30 to 41 s after the busy phase began), which need a scrape lag of more
than 60 s. The other 16 match within 0.005. The refresh cadence also varies: every query
(about 11 s) from 178 to 314 s, then about every 60 s. **R4''** is R4' with the lag bound at
90 s. Both R4' and R4'' are scored on the five held-out zones, and both results are reported.

## Rehabilitation results, 2026-09-27

Fresh run: `rook-cephfs` with one shared volume, probed from six zones. Every probe succeeded,
including unl, which stalled in the first sweep before the metadata phase was bounded. Records:
`storage/results/rehab_rook-cephfs_20260927/`. Scores: `rehab_analysis_20260927.json`.

| zone | node | r = create ms | fdatasync p50 ms | random 4 KiB p50 ms | write MB/s | read MB/s |
|---|---|---|---|---|---|---|
| ucsd-nrp | node-2-10.sdsc | 9.7 | 35.8 | 42 | 153 | 94 |
| fullerton | nautilus-it-cpu11 | 10.6 | 26.1 | 73 | 77 | 75 |
| humboldt | cph-blade15 | 22.4 | 40.5 | 131 | 60 | 63 |
| unl | hcc-nrp-shor-c5930 | 48.1 | 66.7 | 317 | 63 | 31 |
| mghpcc | service-01.nrp.mghpcc | 68.8 | 85.6 | 453 | 46 | 24 |
| korea | yge-nrp-01.kreonet | 151.0 | 169.7 | 1870 | 26 | 8.5 |

**R1a, the amount converges exactly: PASSED, 6 of 6.**
- After the write, the settle loop matched on its first poll at every zone.
- After the delete it matched within the bound everywhere: at once at three zones, 1.5 s at
  unl, 4.1 s at humboldt and 6.6 s at korea.
- **P1's amount claim is rehabilitated:** used space is exact at every location once the
  accounting has caught up.

**R1b, settle time is not an observer coordinate: FAILED.**
- Spearman(write settle time, r) was 0.94.
- The operationalisation was flawed by construction: the shortest possible settle time is one
  `statvfs` round trip, and that is r.
- The multi-second delete settles at the three far or slow zones may also follow r, but these
  data cannot separate that from the polling round trip. This is not claimed either way.

**R4, the metrics are an exact trailing-window mean: FAILED again.** The maximum error was 0.354
cores. The API's `timestamp` is the query time, not the scrape time.

**R4' and R4'', the scrape-time mechanism.** Fitted on ucsd-nrp only, and scored blind on the
five held-out zones:
- **The exact version, (a) within 0.02 cores: FAILED.**
  - 88 of 93 readings matched.
  - 5 missed by up to 0.071: four early low readings at fullerton (0.022 to 0.028) and one at
    unl (0.785).
  - Widening the lag from 60 to 90 s (R4'') changed nothing.
- **The magnitude version, (b), readings under a steady one-core load in [0.83, 1.17]:
  PASSED.** Readings were 0.90 to 1.08.

**Net for P4.** The observer that NRP enforcement uses reads a sustained load correctly to
about ±10 %, once its window lies inside the load. Its behaviour at transitions is not captured
exactly by a scrape-lag and window-jitter model. That part stays unexplained.

## 1T reuse results, 2026-09-27: L1 to L4 all PASSED

Read-only probes of `tqp-fleet-1t-17` and `tqp-fleet-1t-311`, each from six zones. Both volumes
are LINSTOR pool `unl`, one replica each, so the stored object sits at UNL. Records:
`storage/results/onet_readonly_20260927/`.

**Two probes did not run: 17 from ucsd-nrp and 311 from fullerton.**
- Each timed out at 1,200 s with no log output and no metrics samples, so the container most
  likely never started.
- A volume-attach problem is suspected but not confirmed; the pod events were not captured.
- 311 from ucsd-nrp, and 17 from fullerton, both ran normally.
- n = 5 per volume below.

| volume | zone | node | r_blk = cold first 4 KiB ms | random 4 KiB p50 ms (reads) | sequential MB/s |
|---|---|---|---|---|---|
| 17 | unl | hcc-nrp-shor-c5830 | 16.8 | 13.3 (2000) | 257 |
| 17 | mghpcc | service-01.nrp.mghpcc | 33.7 | 32.7 (2000) | 183 |
| 17 | fullerton | nautilus-it-cpu11 | 47.5 | 44.6 (2000) | 124 |
| 17 | humboldt | cph-blade15 | 53.9 | 51.4 (1746) | 119 |
| 17 | korea | yge-nrp-01.kreonet | 183.9 | 167.8 (536) | 37.7 |
| 311 | unl | hcc-nrp-shor-c6017 | 16.1 | 10.9 (2000) | 230 |
| 311 | mghpcc | service-01.nrp.mghpcc | 36.1 | 32.7 (2000) | 185 |
| 311 | ucsd-nrp | node-2-1.sdsc | 45.2 | 43.1 (2000) | 141 |
| 311 | humboldt | cph-blade11 | 54.6 | 51.5 (1747) | 119 |
| 311 | korea | yge-nrp-01.kreonet | 187.0 | 170.6 (527) | 37.1 |

**L1, one coordinate for block storage: PASSED for both volumes.**
- Pearson(random p50, r_blk) is 0.9996 for volume 17 and 0.9994 for 311.
- Spearman(sequential MB/s, r_blk) is -1.0 for both.

**L2, the observer dominates the object: PASSED.**
- At a fixed zone, the two volumes differ by at most 1.21x on any of the three measures.
- Across zones, r_blk spans 11.6x, random p50 15.6x and sequential throughput 6.9x.

**L3, invariance: PASSED.**
- Every observer of each volume read the same used bytes: 48,015,597,568.
- The two volumes read exactly the same figure, since both hold the same shard structure.

**L4, the block counter sees block reads: PASSED.** The kernel's `read_bytes` over the bytes
issued was 1.0033 at every location. That is the xfs log and metadata on top of 1 GiB of data.
It is P2's counterpart: on block storage, the observer CephFS hides from sees everything.

**What the coordinate is.** For the 1T volumes, the ordering is unl, then mghpcc, then the
California zones, then korea. That is the network distance to the object's home at UNL, not to
the observer's region. MGHPCC in Boston reads UNL's volumes faster than any California node
does. For the San Diego CephFS it was ucsd-nrp that came first. In Observation Theory terms:
- the coordinate belongs to the pair (observer, object), not to the observer alone;
- one number per pair, a single cold block round trip, predicts every latency and throughput
  here;
- the object's own state (L3) and the block observer's accounting (L4) do not move at all.

**The 1T volumes.** They were only read, through read-only mounts. All 500 remain bound and
unchanged. They stay bound until the owner releases them.
