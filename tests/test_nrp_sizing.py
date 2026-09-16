"""The NRP request-sizing rules (benchmarks/nrp/sizing.py) on the 2026-09-15 pods."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from nrp.sizing import (  # noqa: E402
    Refusal,
    Request,
    Usage,
    check,
    cpu_request,
    memory_window,
    request_for,
)

# What the cluster dashboard reported for the flagged pods, as usage: a deep-image-96
# rabitqlib cell (4% of 4 CPUs, 62% of 6 GiB) and a wiki1024-10m tqfix cell
# (1% of 4 CPUs, 3% of 19 GiB).
DEEP = Usage(mean_cpu_cores=0.04 * 4, mean_mem_gib=0.62 * 6, peak_mem_gib=5.0)
TQFIX = Usage(mean_cpu_cores=0.01 * 4, mean_mem_gib=0.03 * 19, peak_mem_gib=17.9)
HEALTHY = Usage(mean_cpu_cores=3.8, mean_mem_gib=5.7, peak_mem_gib=6.4)


def test_the_flagged_pods_would_have_been_refused():
    for usage in (DEEP, TQFIX):
        out = request_for(usage, want_cpu=4)
        assert isinstance(out, Refusal), out
        assert check(
            4, 6, usage
        )  # and the request they actually ran with was a violation


def test_a_healthy_cell_is_sized_and_passes_its_own_check():
    out = request_for(HEALTHY, want_cpu=4)
    assert isinstance(out, Request) and not out.exempt
    assert out.cpu == 4  # 3.8 cores supports the full request
    assert out.memory_gib >= HEALTHY.peak_mem_gib  # never OOM at the request
    assert check(out.cpu, out.memory_gib, HEALTHY) == []


def test_cpu_request_follows_the_measured_average():
    assert cpu_request(3.8, want=4) == 4
    assert (
        cpu_request(0.9, want=4) == 3
    )  # 0.9 cores hold 3 CPUs at 30%, above the floor
    assert cpu_request(0.9, want=2) == 2  # never above what was asked
    assert cpu_request(0.16, want=4) == 1  # never below one CPU
    assert cpu_request(12.0, want=4) == 4


def test_a_bursty_class_has_no_compliant_request():
    bursty = Usage(mean_cpu_cores=3.0, mean_mem_gib=1.0, peak_mem_gib=30.0)
    out = request_for(bursty, want_cpu=4)
    assert isinstance(out, Refusal) and "Restructure" in out.reason
    lo, hi = memory_window(bursty)
    assert lo > hi


def test_a_small_class_goes_to_the_exempt_pool_where_no_floor_applies():
    small = Usage(mean_cpu_cores=0.1, mean_mem_gib=0.5, peak_mem_gib=0.9)
    out = request_for(small, want_cpu=1)
    assert isinstance(out, Request) and out.exempt and out.cpu == 1
    assert check(out.cpu, out.memory_gib, small) == []


def test_an_unmeasured_class_is_refused_rather_than_guessed():
    assert isinstance(request_for(None, want_cpu=4), Refusal)
    assert isinstance(request_for(Usage(0.0, 0.0, 0.0), want_cpu=4), Refusal)


@pytest.mark.parametrize(
    "cpu,mem,ok", [(1, 2.0, True), (1, 4.0, False), (4, 2.0, False)]
)
def test_the_exempt_class_is_both_limits_at_once(cpu, mem, ok):
    idle = Usage(mean_cpu_cores=0.01, mean_mem_gib=0.1, peak_mem_gib=0.2)
    assert (check(cpu, mem, idle) == []) is ok


def test_a_request_is_never_returned_below_the_peak():
    """A tight window must refuse, not emit a request the pod would be OOM-killed at."""
    tight = Usage(mean_cpu_cores=3.9, mean_mem_gib=2.0, peak_mem_gib=9.8)
    out = request_for(tight, want_cpu=4)
    if isinstance(out, Request):
        assert out.memory_gib >= tight.peak_mem_gib, out
        assert check(out.cpu, out.memory_gib, tight) == []
    else:
        assert "peak" in out.reason  # says which end could not be satisfied
