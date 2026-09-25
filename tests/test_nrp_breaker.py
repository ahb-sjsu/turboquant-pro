"""The NRP circuit breaker's state machine (benchmarks/nrp_breaker.py), no cluster."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

import nrp_breaker as B  # noqa: E402


def tick(br, now, jobs):
    gone = br.observe(jobs, now)
    by_name = {v[0]: v[1:] for v in jobs.values()}
    return gone, br.step(now, gone, by_name)


def J(name, active=True, ok=False, bad=False):
    return (name, active, ok, bad)


def test_starts_open_and_half_opens_only_after_a_quiet_period():
    br = B.Breaker(0, quiet=100)
    assert tick(br, 0, {})[1] == "OPEN" and not br.may_submit()
    assert tick(br, 99, {})[1] == "OPEN"
    assert tick(br, 100, {})[1] == "HALF_OPEN" and br.may_submit()


def test_completed_or_failed_jobs_leaving_are_not_deletions():
    br = B.Breaker(0, quiet=100)
    tick(br, 0, {"a": J("x", active=False, ok=True), "b": J("y", False, bad=True)})
    gone, _ = tick(br, 10, {})
    assert gone == [] and br.deletions == []


def test_active_job_vanishing_is_a_deletion_and_resets_the_quiet_clock():
    br = B.Breaker(0, quiet=100)
    tick(br, 0, {"a": J("x")})
    gone, st = tick(br, 90, {})
    assert gone == ["x"] and st == "OPEN"
    assert tick(br, 150, {})[1] == "OPEN"  # 60 s after the deletion, not 100
    assert tick(br, 190, {})[1] == "HALF_OPEN"


def test_probe_survival_closes_and_a_burst_reopens():
    br = B.Breaker(0, quiet=100)
    tick(br, 100, {})
    br.probe = ("p", 100)
    assert tick(br, 200, {"u1": J("p")})[1] == "HALF_OPEN"
    assert tick(br, 100 + B.PROBE_OK, {"u1": J("p")})[1] == "CLOSED"
    assert br.may_submit()
    tick(br, 2000, {"u2": J("q"), "u3": J("r")})
    assert tick(br, 2010, {})[1] == "OPEN"  # two deletions inside the window


def test_probe_deleted_reopens_with_doubled_quiet_capped():
    br = B.Breaker(0, quiet=100)
    tick(br, 100, {})
    br.probe = ("p", 100)
    tick(br, 110, {"u1": J("p")})
    gone, st = tick(br, 120, {})
    assert gone == ["p"] and st == "OPEN" and br.quiet == 200 and br.probe is None
    br.quiet = B.MAX_QUIET
    tick(br, 120 + B.MAX_QUIET, {})
    br.probe = ("p", 120 + B.MAX_QUIET)
    tick(br, 120 + B.MAX_QUIET + 1, {"u9": J("p")})
    tick(br, 120 + B.MAX_QUIET + 2, {})
    assert br.quiet == B.MAX_QUIET


def test_probe_that_completes_closes_and_one_that_fails_frees_the_slot():
    br = B.Breaker(0, quiet=100)
    tick(br, 100, {})
    br.probe = ("p", 100)
    assert tick(br, 130, {"u1": J("p", active=False, bad=True)})[1] == "HALF_OPEN"
    assert br.probe is None and br.may_submit()
    br.probe = ("p2", 140)
    assert tick(br, 150, {"u2": J("p2", active=False, ok=True)})[1] == "CLOSED"


def test_a_hung_or_unauthenticated_kubectl_is_a_failed_listing(monkeypatch):
    """The breaker died on a kubectl timeout (2026-09-25). A timeout, an auth failure,
    and unparseable output all read as a failed listing, never an empty namespace.
    """
    import subprocess

    def hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="kubectl", timeout=120)

    monkeypatch.setattr(B.subprocess, "run", hang)
    jobs, why = B.list_jobs()
    assert jobs is None and "TimeoutExpired" in why

    def denied(*a, **k):
        return subprocess.CompletedProcess(a, 1, "", "getting credentials: exec failed")

    monkeypatch.setattr(B.subprocess, "run", denied)
    jobs, why = B.list_jobs()
    assert jobs is None and "credentials" in why

    monkeypatch.setattr(
        B.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, "<html>", ""),
    )
    assert B.list_jobs()[0] is None
