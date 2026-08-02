# Announcement to NRP: 1T-vector index build, namespace `ssu-atlas-ai`

NRP does not run an allocation system. Support's position is that a user may
use a resource provided the use adheres to cluster policy and the workflow
does not impact other users. So this is an announcement, not a request. It
tells the community what we are about to run, why it complies, and how to
reach us if it is in anyone's way.

Post to the Nautilus Matrix channel before the first 1T job is submitted.

---

## Short form (Matrix)

Hi all. San Jose State (`ssu-atlas-ai`) is starting a one-trillion-vector
index build over the next few weeks, and we want the community to know what
the load looks like before it appears.

**What it is.** We measure how far a vector index can be compressed before
retrieval quality degrades. We have run 1B, 10B, and 100B on Nautilus already.
Recall stayed flat across all three at 0.999, 0.9988, and 0.9986. 1T is the
next point on that curve. Code and results are public.

**The load.** Around 24 TB of block storage held about six weeks, and roughly
12,000 CPU-hours of batch build spread over that window. No GPU. Batch jobs
only, no long-lived services.

**Why it should not affect you.** Every job saturates its cores for its whole
life, so nothing sits idle holding a reservation. Builds checkpoint per shard,
so a preempted pod resumes rather than restarting and we do not thrash the
scheduler retrying work. We submit in waves within the heavy-pod limit instead
of flooding the queue. We release volumes as soon as a measurement is
recorded. The 100B run finished 28 July and its 3.2 TB went back on 2 August.

**If it is in your way, tell us.** We will shrink it, move it, or pause it. I
will watch this channel for the duration. If a slow window for the cluster
would suit better, say so and we will schedule around it.

Andrew Bond, SJSU, `andrew.bond@sjsu.edu`

---

## Long form (for the storage class or a ticket, if anyone wants detail)

### What the workload is

San Jose State, namespace `ssu-atlas-ai`. An open-source vector compression
library and its benchmark suite. The question is narrow and measurable. How
far can a vector index be compressed before retrieval quality degrades, and
does the answer change with corpus size.

### Why this size

Three measured points on Nautilus so far. Recall against an exact
same-hardware reference was 0.999 at one billion vectors, 0.9988 at ten
billion, and 0.9986 at one hundred billion, all at 24 bytes per vector. A flat
line across two orders of magnitude is a much stronger result than any single
point. One trillion is the next point that tells us something new.

### The footprint

| | |
|---|---|
| Block storage | ~24 TB at 4-bit encoding, held ~6 weeks |
| Compute | ~12,000 CPU-hours, batch, spread across the window |
| GPU | none |
| Long-running services | none |
| Namespace | `ssu-atlas-ai` |

The storage number follows from a measured 24.01 bytes per row. It is the peak
we would hold, not padded. Our namespace caps a single volume at 64 GiB, so
this lands as a large number of volumes. If a different layout or storage
class is easier on the cluster, we will use it.

### How it behaves, measured rather than promised

Every property below comes from the completed 100B run.

- Build and scan jobs are CPU-bound for their entire lifetime. We run no idle
  placeholder pods.
- Builds write a per-shard completion record, so a preempted pod resumes
  instead of rebuilding the range. The index finalize step is separately
  resumable. Without both, a build at this size never converges and instead
  retries forever, which is load on the cluster with nothing to show for it.
- Concurrency is capped to the standing heavy-pod limit and submitted in
  waves.
- Volumes are released as soon as the measurement is captured, as the 100B run
  showed.

### The risk we are watching

At fifty servers our slowest node ran 6.9 times the median, and fleet
wall-clock is set by the slowest node rather than the average. At 1T the fleet
is several hundred servers and that effect grows. This matters to the cluster
and not only to us, because a run stretched by one slow node holds storage
longer than announced. We are adding straggler re-issue to the driver before
starting, and if that does not work we will run a smaller size instead of
holding 24 TB open-ended.

### What we will publish

The measurement is a few hundred kilobytes of JSON and goes in a public
repository with the driver scripts and raw logs. If it is useful we are glad
to write up how to run a build of this shape on Nautilus, especially the
checkpointing, which took several failed attempts to get right.

---

## Contact

Andrew H. Bond, Department of Computer Engineering, San Jose State University.
`andrew.bond@sjsu.edu`
