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

**The load.** About 27 TiB of block storage held roughly six weeks, and around
12,000 CPU-hours of batch build spread over that window. No GPU. Batch jobs
only, no long-lived services. The index itself is about 24 TB. The claim is
larger because it is 500 volumes of 56 GiB each and every one carries build
headroom.

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
| Block storage | 27.3 TiB provisioned (500 x 56 GiB), ~24 TB of index in it, held ~6 weeks |
| Compute | ~12,000 CPU-hours, batch, spread across the window |
| GPU | none |
| Long-running services | none |
| Namespace | `ssu-atlas-ai` |

The index size follows from a measured 24.01 bytes per row. The provisioned
figure is larger than that because each volume is sized with build headroom.
27.3 TiB is what we actually claim and it is the number to hold us to.

It lands as 500 volumes rather than a few large ones because each build job
writes its own shard and the volumes are read-write-once, so the volume count
is set by how many builders run, not by any size limit. We use `linstor-unl`,
which places one replica, so the physical cost equals the claim. On a
three-replica class the same run would cost three times as much, which is why
we do not use one. If a different layout or storage class is easier on the
cluster, we will use it.

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
longer than announced. We have written straggler re-issue into the driver but it
is switched off for the first 1T attempt, because that code has never run at
this scale and a re-issue bug would double the load rather than shorten it. We
watch the tail by hand instead, and if the tail stretches the run we will turn
re-issue on for a later attempt or drop to a smaller size rather than hold 27
TiB open-ended.

### What we will publish

The measurement is a few hundred kilobytes of JSON and goes in a public
repository with the driver scripts and raw logs. If it is useful we are glad
to write up how to run a build of this shape on Nautilus, especially the
checkpointing, which took several failed attempts to get right.

---

## Correction to post as a follow-up

The short form above was posted to Matrix on 2 August with a storage figure of
24 TB and a claim that our namespace caps a single volume at 64 GiB. Both were
wrong. There is no per-volume cap, and 24 TB is the size of the index rather
than the size of the claim. Post this as a reply to the original so the number
the community sees is the one we actually hold.

> Correction to my 1T post. I said 24 TB of block storage. That is the size of
> the index, not the size of the claim. The claim is 500 volumes of 56 GiB,
> which is 27.3 TiB, on `linstor-unl` at one replica so the physical cost is
> the same number. I also said our namespace caps a volume at 64 GiB, which is
> not true and was my error. Sorry for the sloppy figure. Everything else in
> the post stands, and the offer to shrink or move the run if it is in anyone's
> way stands with it.

---

## Contact

Andrew H. Bond, Department of Computer Engineering, San Jose State University.
`andrew.bond@sjsu.edu`
