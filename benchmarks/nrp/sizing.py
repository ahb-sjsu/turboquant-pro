"""Size a pod's requests from measured time averages, or refuse to submit it.

NRP counts both over- and under-use as violations: CPU usage must sit at 20-200% of the
request and memory at 20-150%, and pods at or below 1 CPU / 2 GiB are exempt. Because
requests equal limits here, a pod is OOM-killed at its memory request, so the request has to
cover the peak while the *average* stays above the floor.

That makes the compliant memory window explicit:

    request >= peak            (or the pod dies)
    request <= mean / 0.20     (or the cluster flags it)

so a class is sizeable only when its peak is at most 5x its mean.

When a class fails that test, raising the request is the wrong move and the guard will stop the
pod anyway: a wiki PQ cell bumped to 18 GiB to survive a 15 GiB transient then averaged 3 and
was stopped at 17% of its request. The transient has to be bounded instead. That one came from
encoding a 250k-row batch at once, where faiss builds a batch x m x 256 distance table; batching
the encode took the peak to 4 GiB and the class became sizeable. CPU has one side, since a
pod cannot exceed a request that equals its limit:

    request <= mean_cores / 0.20

On 2026-09-15 eight campaign pods were flagged at 1-4% of 4 requested CPUs: sizing had been
done from peak memory alone and CPU was never measured at all. Every rule here therefore takes
a measurement, never a model, and ``request_for`` returns a refusal rather than a guess when
the measurement is missing or no compliant request exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CPU_FLOOR = MEM_FLOOR = 0.20
EXEMPT_CPU, EXEMPT_MEM_GIB = 1, 2.0
MARGIN = 1.25  # aim this far above the floor, so one slow phase does not trip it
WINDOW_INSET = 0.90  # how far inside the floor the largest compliant request sits
PEAK_HEADROOM = 1.10  # memory request covers the measured peak by this much


@dataclass(frozen=True)
class Usage:
    """What a pod actually did, averaged over its whole life (not a snapshot)."""

    mean_cpu_cores: float
    mean_mem_gib: float
    peak_mem_gib: float

    @property
    def burstiness(self) -> float:
        return self.peak_mem_gib / max(self.mean_mem_gib, 1e-9)


@dataclass(frozen=True)
class Request:
    cpu: int
    memory_gib: int
    exempt: bool
    note: str

    def __str__(self):
        tag = " (exempt class)" if self.exempt else ""
        return f"cpu={self.cpu} memory={self.memory_gib}Gi{tag}: {self.note}"


@dataclass(frozen=True)
class Refusal:
    reason: str

    def __str__(self):
        return f"REFUSED: {self.reason}"


def cpu_request(mean_cpu_cores: float, want: int) -> int:
    """The largest request up to ``want`` whose floor the measured average clears."""
    return max(1, min(want, int(mean_cpu_cores / (CPU_FLOOR * MARGIN))))


def memory_window(usage: Usage) -> tuple[float, float]:
    """(smallest safe, largest compliant) memory request in GiB.

    The top of the window is the cluster's own floor kept a little inside, not the floor
    times the sizing margin: multiplying both ends by their margins closed the window on
    classes the cluster would accept. A dbpedia OPQ class with a 9.8 GiB peak and a 2.5 GiB
    mean is sizeable at 11 GiB, where the mean sits at 23%.
    """
    return (
        usage.peak_mem_gib * PEAK_HEADROOM,
        usage.mean_mem_gib / MEM_FLOOR * WINDOW_INSET,
    )


def request_for(usage: Usage | None, want_cpu: int) -> Request | Refusal:
    """A compliant request for a class with this measured usage, or a refusal explaining why not."""
    if usage is None:
        return Refusal(
            "no measured usage for this class; run its calibration cell first"
        )
    if usage.mean_cpu_cores <= 0 or usage.mean_mem_gib <= 0:
        return Refusal(f"measurement is empty ({usage})")
    lo, hi = memory_window(usage)
    if usage.peak_mem_gib <= EXEMPT_MEM_GIB and (
        want_cpu <= EXEMPT_CPU or usage.mean_cpu_cores < 1
    ):
        return Request(
            EXEMPT_CPU,
            math.ceil(max(lo, 1.0)),
            True,
            "fits the exempt class, no floors apply",
        )
    if lo > hi:
        return Refusal(
            f"no compliant memory request exists: peak {usage.peak_mem_gib:.1f} GiB is "
            f"{usage.burstiness:.1f}x the mean {usage.mean_mem_gib:.1f} GiB, and the floor allows "
            f"at most {1 / (MEM_FLOOR * MARGIN):.0f}x. Restructure the workload so the peak is "
            "shorter or the average higher."
        )
    mem = math.ceil(lo)
    if mem > hi:  # rounding up left the window
        mem = math.floor(hi)
    if mem < lo:  # no whole number both covers the peak and clears the floor
        return Refusal(
            f"the window is too narrow to land in: covering a {usage.peak_mem_gib:.1f} GiB peak "
            f"needs {lo:.1f} GiB and the floor allows at most {hi:.1f} GiB for a mean of "
            f"{usage.mean_mem_gib:.1f}. Bound the transient rather than resize."
        )
    cpu = cpu_request(usage.mean_cpu_cores, want_cpu)
    left = check(cpu, mem, usage)
    if left:
        # Below the exempt class there is no request small enough: one CPU is the floor of the
        # scale, so a pod averaging under 0.2 cores can only comply by fitting in 1 CPU / 2 GiB
        # or by spending less of its life waiting on I/O.
        return Refusal(
            f"the smallest sensible request (cpu={cpu}, memory={mem}Gi) would still be flagged for "
            + "; ".join(left)
            + ". Either the class fits the exempt 1 CPU / 2 GiB, or its idle phases have to go."
        )
    note = f"mean {usage.mean_cpu_cores:.2f} cores, mean {usage.mean_mem_gib:.1f} / peak {usage.peak_mem_gib:.1f} GiB"
    if cpu < want_cpu:
        note += f"; {want_cpu} CPUs asked but the average only supports {cpu}"
    return Request(cpu, mem, False, note)


def check(request_cpu: int, request_mem_gib: float, usage: Usage) -> list[str]:
    """Violations a pod with this request and this usage would be flagged for."""
    bad = []
    if request_cpu <= EXEMPT_CPU and request_mem_gib <= EXEMPT_MEM_GIB:
        return bad
    if usage.mean_cpu_cores < CPU_FLOOR * request_cpu:
        bad.append(
            f"cpu {100 * usage.mean_cpu_cores / request_cpu:.0f}% of {request_cpu}"
        )
    if usage.mean_mem_gib < MEM_FLOOR * request_mem_gib:
        bad.append(
            f"memory {100 * usage.mean_mem_gib / request_mem_gib:.0f}% of {request_mem_gib:g}Gi"
        )
    if usage.peak_mem_gib > request_mem_gib:
        bad.append(
            f"peak memory {usage.peak_mem_gib:.1f} GiB exceeds the {request_mem_gib:g}Gi request"
        )
    return bad
