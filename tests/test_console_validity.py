"""The console decides whether a loaded certificate still applies.

Before this it showed the certificate's issue-time ``validity`` section, which
has no ``status``, so every certificate read UNCHECKED although the console
holds the data ``check_validity`` needs. The panels now show all four states
(VALID, STALE, INCONCLUSIVE, UNCHECKED) as words, with the reason and the
sample the check read.
"""

from __future__ import annotations

import numpy as np

from turboquant_pro.console import tui
from turboquant_pro.console.server import ConsoleServer, demo_index
from turboquant_pro.validity import validity_section


def _session(cert, originals="corpus"):
    index, Q, X, source, _ = demo_index(n=1500, dim=64, out_dim=32)
    orig = X if isinstance(originals, str) else originals
    return X, ConsoleServer(
        index, Q, qps=10, k=5, originals=orig, certificate=cert, http=False
    )


def _cert(sample):
    return {
        "passed": True,
        "certificate": {},
        "validity": validity_section(sample=sample),
    }


def _frame_text(s):
    st = {
        "snap": s.snapshot(),
        "traces": [],
        "readscope": s.readscope(),
        "qps_hist": [],
        "p95_hist": [],
        "sel": 0,
        "focus": 4,
        "paused": False,
        "overlay": None,
        "inspected": None,
        "replay": None,
        "message": "",
    }
    return "\n".join(tui.frame(st, 160, 50).text())


def test_same_data_is_valid_and_says_what_it_read():
    X, _ = _session(None)
    _, s = _session(_cert(X[:1000]))
    v = s.readscope()["validity"]
    assert v["status"] == "VALID"
    assert v["checks"]["data_coverage"]["status"] == "ok"
    assert v["data"] == {
        "source": "--originals",
        "rows": 1500,
        "of": 1500,
        "seed": 0,
        "kind": "measured",
    }
    assert s.validity() is v  # computed once: the TUI asks every frame
    assert "VALID" in _frame_text(s)


def test_shifted_data_is_stale_with_its_reason_and_action():
    X, _ = _session(None)
    _, s = _session(_cert(X[:1000]), originals=X + 3.0)
    v = s.validity()
    assert v["status"] == "STALE" and v["action"] == "RECERTIFY"
    text = _frame_text(s)
    assert "STALE" in text and "RECERTIFY" in text


def test_without_originals_it_is_unchecked_and_says_why():
    X, _ = _session(None)
    _, s = _session(_cert(X[:1000]), originals=None)
    v = s.validity()
    assert v["status"] == "UNCHECKED"
    assert "no --originals" in v["data"]["reason"]


def test_a_failing_check_is_shown_not_raised_and_is_never_a_pass():
    _, s = _session(_cert(np.zeros((50, 7), dtype=np.float32)))  # wrong dimension
    v = s.validity()
    assert v["status"] == "UNCHECKED"
    assert v["reason"].startswith("the check failed")


def test_no_certificate_no_validity():
    _, s = _session(None)
    assert s.readscope()["validity"] is None
