"""On-screen notes: every graph says what it shows (`i` hides them)."""

from __future__ import annotations

from turboquant_pro.console import tui
from turboquant_pro.console.scope import Scope
from turboquant_pro.console.spectrum import Analyzer


def _state(view, **kw):
    return {
        "view": view,
        "scope": Scope(),
        "analyzer": Analyzer(),
        "now": 0.0,
        "snap": {},
        "traces": [],
        "readscope": {},
        "qps_hist": [],
        "p95_hist": [],
        **kw,
    }


def test_the_scope_names_its_traces_axes_and_markers():
    text = "\n".join(tui.frame(_state("scope"), 160, 48).text())
    assert "x: time, newest at the right edge" in text
    assert "trigger point" in text and "i hides these notes" in text
    for ch in Scope().channels:
        if ch.on:
            assert f": {ch.signal}, " in text  # each enabled channel is named


def test_notes_can_be_hidden():
    text = "\n".join(tui.frame(_state("scope", annotate=False), 160, 48).text())
    assert "x: time" not in text and "i hides" not in text


def test_the_overview_says_what_its_graphs_span():
    text = "\n".join(tui.frame(_state("overview"), 160, 48).text())
    assert "1 sample/s, bar height 0..max" in text
