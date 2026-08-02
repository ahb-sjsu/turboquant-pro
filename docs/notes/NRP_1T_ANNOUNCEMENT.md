# Announcement to NRP: 1T-vector index build, namespace `ssu-atlas-ai`

Draft for the Nautilus Matrix channel and the storage request. Sent before any
1T resource is claimed. Written to be read by an admin in one minute.

---

## Short form (Matrix)

Hi all. San Jose State (`ssu-atlas-ai`) is planning a one-trillion-vector
index build and we want to announce it before we take any space.

**What it is.** We build compressed nearest-neighbour indexes and measure
whether retrieval quality survives the compression. We have completed runs at
1B, 10B, and 100B vectors on Nautilus. Recall held flat across all three
(0.999, 0.9988, 0.9986 at the same setting), which is the result we publish.
1T is the next point on that curve.

**What we would need.** About 24 TB of block storage held for roughly six
weeks, and on the order of 12,000 CPU-hours of batch build spread across that
window. No GPU. All batch jobs, no long-lived services.

**How we intend to run it.** Every job is CPU-saturating for its whole life,
in waves that respect the standing heavy-pod limit. Builds checkpoint per
shard, so a preempted pod resumes instead of restarting, and we hold no idle
pods. We release volumes as soon as a measurement is captured. We did that
last week, returning 3.2 TB within days of finishing the 100B run.

**What we are asking.** Whether 24 TB for six weeks is acceptable, whether you
would prefer it split across storage classes or sites, and whether there is a
window you would rather we avoid. We will adjust the shape to fit, including
running smaller if that is the better answer.

Happy to move this to email or a ticket if that suits you better. We have
appreciated the standing exception you granted us for the `atlas-nats-leaf`
service and want to keep operating the same way, by asking first.

---

## Long form (storage request or email)

### Who and what

San Jose State University, namespace `ssu-atlas-ai`. The project is an
open-source vector compression library and its benchmark suite. The scientific
question is narrow and measurable: how far can a vector index be compressed
before retrieval quality degrades, and does the answer change with corpus
size.

### Why 1T

We have three measured points on Nautilus. Recall against an exact
same-hardware reference was 0.999 at one billion vectors, 0.9988 at ten
billion, and 0.9986 at one hundred billion, all at 24 bytes per vector. A flat
line across two orders of magnitude is a much stronger claim than any single
point, and one trillion is the next point that would tell us something new.
The results and the code are public.

### Resource request

| | |
|---|---|
| Block storage | ~24 TB at 4-bit encoding, held ~6 weeks |
| Compute | ~12,000 CPU-hours, batch, spread over the same window |
| GPU | none |
| Services | none, batch jobs only |
| Namespace | `ssu-atlas-ai` |

The storage figure follows from the measured 24.01 bytes per row and is the
number we would hold at peak, not a request for headroom. Volume count depends
on how you would prefer it laid out; the per-volume cap in our namespace is
64 GiB today, which would mean many volumes, so we would welcome direction
here.

### How the workload behaves

These properties are measured from the 100B run, not aspirational.

- **Jobs saturate their cores.** Build and scan jobs are CPU-bound for their
  entire lifetime. We do not run idle placeholder pods.
- **Everything checkpoints.** Builds write a per-shard completion record, so a
  pod that is preempted resumes from where it stopped rather than starting the
  shard-range again. The index finalize step is separately resumable. Without
  both of these a 100B build does not converge, and we would not attempt 1T
  without them.
- **We run in waves.** Concurrency is capped to the standing heavy-pod limit
  rather than submitted all at once.
- **We return storage.** The 100B run finished on 28 July and its 3.2 TB was
  released on 2 August, once the measurement was safely recorded.

### What we would report back

The measurement itself is small, a few hundred kilobytes of JSON, and it goes
in a public repository along with the driver scripts and raw logs. If it is
useful to you we are glad to write up what we learn about running a build of
this shape on Nautilus, particularly the checkpointing pattern, which took us
several failed attempts to get right.

### The honest risk

Our 100B run showed that at fifty servers the slowest node ran 6.9 times the
median, and fleet wall-clock is set by the slowest node rather than the
average. At 1T the fleet is several hundred servers and that effect grows. We
are building straggler re-issue into the driver before we would start. We
raise it because it affects you as well as us: a run that stretches because of
one slow node holds storage longer than planned. If we cannot demonstrate that
the driver handles stragglers, we will not ask you to hold 24 TB for us.

### Timing

Nothing is scheduled. We will not claim storage until we have an answer, and
we are happy to fit whatever window suits the platform.

---

## Contact

Andrew H. Bond, Department of Computer Engineering, San Jose State University.
`andrew.bond@sjsu.edu`
