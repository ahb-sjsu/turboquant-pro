# TurboQuant Pro: Open-source TurboQuant for LLM KV cache compression
# Copyright (c) 2026 Andrew H. Bond. MIT License.
"""The utilization guard reads usage from the metrics API.

`kubectl top pods` failed the whole namespace when one pod had no metrics yet
(a pod that just completed), and the guard then sampled nothing. The metrics
API returns the pods it has; these tests pin the parsing of its units,
including memory in milli-bytes, which the old parser crashed on.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from nrp import utilization_guard as g  # noqa: E402


def test_units():
    assert g.parse_cpu("1350m") == 1.35
    assert g.parse_cpu("250000000n") == 0.25
    assert g.parse_cpu("0") == 0.0
    assert g.parse_mem("3329188Ki") == 3329188 * 1024
    assert g.parse_mem("10933589333m") == 10933589333e-3
    assert g.parse_mem("2Gi") == 2 * 2**30


def test_metrics_list_sums_containers_and_skips_nothing_it_has():
    doc = {
        "items": [
            {
                "metadata": {"name": "a"},
                "containers": [
                    {"usage": {"cpu": "1350m", "memory": "3329188Ki"}},
                    {"usage": {"cpu": "50m", "memory": "1Mi"}},
                ],
            },
            {
                "metadata": {"name": "leaf"},
                "containers": [{"usage": {"cpu": "0", "memory": "10933589333m"}}],
            },
        ]
    }
    got = g.parse_metrics(doc)
    assert set(got) == {"a", "leaf"}
    assert abs(got["a"][0] - 1.4) < 1e-12
    assert got["a"][1] == 3329188 * 1024 + 2**20
    assert got["leaf"] == (0.0, 10933589333e-3)
    assert g.parse_metrics({}) == {}


GIB = 2**30


def test_a_short_job_is_measured_before_the_grace_period():
    """A 4-minute Job never reached the old guard's 900 s grace, so it left no
    measurement and its class could never be sized."""
    h = [(2.8, 4 * GIB), (2.9, 5 * GIB)]
    measured, low = g.assess(h, 240, 4, 12 * GIB, grace=900, window=10, floor=0.2)
    assert measured == pytest.approx((2.85, 4.5 * GIB, 5 * GIB))
    assert low == []  # too young to judge, even though it is measured


def test_judgement_waits_for_grace_and_a_full_window():
    h = [(0.1, 1 * GIB)] * 10
    _, low = g.assess(h, 899, 4, 12 * GIB, grace=900, window=10, floor=0.2)
    assert low == []
    _, low = g.assess(h[:9], 1000, 4, 12 * GIB, grace=900, window=10, floor=0.2)
    assert low == []
    _, low = g.assess(h, 1000, 4, 12 * GIB, grace=900, window=10, floor=0.2)
    assert len(low) == 2  # cpu 0.1/4 and memory 1/12 GiB both under 20%


def test_exempt_requests_are_never_judged():
    h = [(0.01, 0.1 * GIB)] * 10
    _, low = g.assess(h, 5000, 1, 2 * GIB, grace=900, window=10, floor=0.2)
    assert low == []


def test_one_sample_is_not_a_measurement():
    assert g.assess([(1.0, GIB)], 60, 4, 8 * GIB, grace=900, window=10, floor=0.2) == (
        None,
        [],
    )


def test_an_api_stall_is_a_missed_sample(monkeypatch):
    """A hung kubectl (2026-10-15, 18:03Z) killed the guard; it now reads as a failed
    call, which every caller already treats as no data for this cycle."""
    import subprocess

    def hang(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))

    monkeypatch.setattr(g.subprocess, "run", hang)
    r = g.sh("get", "pods", ns="x")
    assert r.returncode == 124 and "timed out" in r.stderr
    assert g.requests("x", "app=y") == {}
