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
