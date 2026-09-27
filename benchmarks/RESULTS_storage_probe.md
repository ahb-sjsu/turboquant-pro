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

## Class sweep, pinned to one zone

The pilot's latencies are dominated by where the pod ran, so classes are compared from one
location, zone `ucsd-nrp`. The classes: `rook-cephfs`, `rook-cephfs-east`,
`rook-ceph-block`, `linstor-ha` and `linstor-unl`.

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
